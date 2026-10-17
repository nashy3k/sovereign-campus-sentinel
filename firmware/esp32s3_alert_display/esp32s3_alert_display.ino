/**
 * Sovereign Campus Sentinel - Fight Alert Display
 * Target: Waveshare ESP32-S3-Touch-LCD-5 (800x480) or -5B (1024x600), RGB LCD with GT911 touch.
 * The board profile is chosen in esp_panel_board_supported_conf.h; the layout scales to either.
 *
 * Joins the local Wi-Fi and exposes a tiny HTTP API. When the Sentinel dashboard confirms a
 * fight in a CCTV clip it POSTs /alert and this screen flashes red with the warning message.
 *
 *   POST /alert   form fields: clip, severity, time (all optional)  -> show the alert
 *   POST /clear                                                     -> back to monitoring screen
 *   GET  /status                                                    -> {"alert":bool,"count":n,...}
 *
 * POST requests must carry the `X-Sentinel-Token` header (see config.h). Touching ACKNOWLEDGE
 * on the screen also clears the alert. Everything stays on the local network.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <esp_display_panel.hpp>
#include <lvgl.h>
#include "lvgl_v8_port.h"

#if __has_include("config.h")
#include "config.h"
#else
#error "Missing config.h: copy config.h.example to config.h and fill in your Wi-Fi details."
#endif

using namespace esp_panel::drivers;
using namespace esp_panel::board;

#define ALERT_MESSAGE        "Fighting is detected. Please take immediate actions now!"
#define WIFI_CONNECT_TIMEOUT_MS  20000
#define WIFI_RETRY_INTERVAL_MS   10000
#define FIELD_MAX_LEN        60
#define RGB_BOUNCE_LINES     30  // must divide the panel height into an even number of chunks (600/30 = 20)

#define COLOR_IDLE_BG    lv_color_hex(0x0B0F19)
#define COLOR_ALERT_A    lv_color_hex(0xDC2626)
#define COLOR_ALERT_B    lv_color_hex(0x7F1D1D)
#define COLOR_OK         lv_color_hex(0x34D399)

static WebServer server(80);

static lv_obj_t *ui_icon;
static lv_obj_t *ui_headline;
static lv_obj_t *ui_detail;
static lv_obj_t *ui_ack_btn;
static lv_timer_t *flash_timer;

static volatile bool alert_active = false;
static volatile uint32_t alert_count = 0;  // alerts received since the last acknowledge
static bool flash_phase = false;
static bool wifi_was_connected = false;
static uint32_t last_wifi_retry_ms = 0;

// ---------------------------------------------------------------------------------------------
// UI. The show_* / refresh_* helpers touch LVGL objects: call them with lvgl_port_lock() held
// (the LVGL task already holds it inside event callbacks and timers).
// ---------------------------------------------------------------------------------------------
static void idle_detail_text(char *out, size_t len)
{
    if (WiFi.status() == WL_CONNECTED) {
        snprintf(out, len, "Monitoring - no active alert\nhttp://%s.local  (%s)",
                 SENTINEL_HOSTNAME, WiFi.localIP().toString().c_str());
    } else {
        snprintf(out, len, "Monitoring - no active alert\nWi-Fi disconnected, retrying...");
    }
}

static void show_idle()
{
    char detail[160];
    alert_active = false;
    alert_count = 0;
    lv_timer_pause(flash_timer);

    lv_obj_set_style_bg_color(lv_scr_act(), COLOR_IDLE_BG, 0);
    lv_label_set_text(ui_icon, LV_SYMBOL_EYE_OPEN);
    lv_obj_set_style_text_color(ui_icon, COLOR_OK, 0);
    lv_label_set_text(ui_headline, "Campus Sentinel");
    idle_detail_text(detail, sizeof(detail));
    lv_label_set_text(ui_detail, detail);
    lv_obj_add_flag(ui_ack_btn, LV_OBJ_FLAG_HIDDEN);
}

static void show_alert(const String &clip, const String &severity, const String &when)
{
    char detail[256];
    int n = snprintf(detail, sizeof(detail), "Clip: %s   Severity: %s",
                     clip.length() ? clip.c_str() : "unknown",
                     severity.length() ? severity.c_str() : "n/a");
    if (when.length() && n > 0 && (size_t)n < sizeof(detail)) {
        n += snprintf(detail + n, sizeof(detail) - n, "\n%s", when.c_str());
    }
    if (alert_count > 0 && n > 0 && (size_t)n < sizeof(detail)) {  // a previous alert was never acknowledged
        snprintf(detail + n, sizeof(detail) - n, "   (+%lu earlier unacknowledged)", (unsigned long)alert_count);
    }

    alert_active = true;
    alert_count = alert_count + 1;
    flash_phase = false;
    lv_obj_set_style_bg_color(lv_scr_act(), COLOR_ALERT_A, 0);
    lv_label_set_text(ui_icon, LV_SYMBOL_WARNING);
    lv_obj_set_style_text_color(ui_icon, lv_color_white(), 0);
    lv_label_set_text(ui_headline, ALERT_MESSAGE);
    lv_label_set_text(ui_detail, detail);
    lv_obj_clear_flag(ui_ack_btn, LV_OBJ_FLAG_HIDDEN);
    lv_timer_resume(flash_timer);
}

static void flash_timer_cb(lv_timer_t *)
{
    flash_phase = !flash_phase;
    lv_obj_set_style_bg_color(lv_scr_act(), flash_phase ? COLOR_ALERT_B : COLOR_ALERT_A, 0);
}

static void ack_btn_cb(lv_event_t *)
{
    Serial.println("Alert acknowledged on screen");
    show_idle();
}

static void build_ui()
{
    lv_obj_t *scr = lv_scr_act();
    // Layout offsets are designed for a 480 px tall panel and scaled for taller ones (5B is 600 px)
    const lv_coord_t v_scale_pct = lv_disp_get_ver_res(NULL) * 100 / 480;
    auto scaled = [v_scale_pct](lv_coord_t px) { return (lv_coord_t)(px * v_scale_pct / 100); };
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);
    lv_obj_set_style_text_color(scr, lv_color_white(), 0);

    ui_icon = lv_label_create(scr);
    lv_obj_set_style_text_font(ui_icon, &lv_font_montserrat_48, 0);
    lv_obj_align(ui_icon, LV_ALIGN_TOP_MID, 0, scaled(20));

    ui_headline = lv_label_create(scr);
    lv_label_set_long_mode(ui_headline, LV_LABEL_LONG_WRAP);
    lv_obj_set_width(ui_headline, LV_PCT(92));
    lv_obj_set_style_text_align(ui_headline, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_font(ui_headline, &lv_font_montserrat_48, 0);
    lv_obj_align(ui_headline, LV_ALIGN_CENTER, 0, -scaled(25));

    ui_detail = lv_label_create(scr);
    lv_label_set_long_mode(ui_detail, LV_LABEL_LONG_WRAP);
    lv_obj_set_width(ui_detail, LV_PCT(92));
    lv_obj_set_style_text_align(ui_detail, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_font(ui_detail, &lv_font_montserrat_24, 0);
    lv_obj_align(ui_detail, LV_ALIGN_CENTER, 0, scaled(110));

    ui_ack_btn = lv_btn_create(scr);
    lv_obj_set_size(ui_ack_btn, 360, scaled(72));
    lv_obj_align(ui_ack_btn, LV_ALIGN_BOTTOM_MID, 0, -scaled(16));
    lv_obj_set_style_bg_color(ui_ack_btn, lv_color_white(), 0);
    lv_obj_add_event_cb(ui_ack_btn, ack_btn_cb, LV_EVENT_CLICKED, NULL);
    lv_obj_t *ack_label = lv_label_create(ui_ack_btn);
    lv_label_set_text(ack_label, "ACKNOWLEDGE");
    lv_obj_set_style_text_font(ack_label, &lv_font_montserrat_28, 0);
    lv_obj_set_style_text_color(ack_label, COLOR_ALERT_B, 0);
    lv_obj_center(ack_label);

    flash_timer = lv_timer_create(flash_timer_cb, 600, NULL);
    show_idle();
}

// Replace the headline/detail with a transient status line (used while Wi-Fi comes up).
static void show_boot_status(const char *text)
{
    if (lvgl_port_lock(1000)) {
        lv_label_set_text(ui_detail, text);
        lvgl_port_unlock();
    }
}

// ---------------------------------------------------------------------------------------------
// HTTP API
// ---------------------------------------------------------------------------------------------
static void send_json(int code, const char *body)
{
    server.sendHeader("Cache-Control", "no-store");
    server.send(code, "application/json", body);
}

static bool authorized()
{
    if (strlen(SENTINEL_TOKEN) == 0 || server.header("X-Sentinel-Token") == SENTINEL_TOKEN) {
        return true;
    }
    send_json(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return false;
}

static String field(const char *name)
{
    String value = server.arg(name);
    value.trim();
    if (value.length() > FIELD_MAX_LEN) {
        value = value.substring(0, FIELD_MAX_LEN);
    }
    return value;
}

static void handle_alert()
{
    if (!authorized()) {
        return;
    }
    String clip = field("clip"), severity = field("severity"), when = field("time");
    if (!lvgl_port_lock(1000)) {
        send_json(503, "{\"ok\":false,\"error\":\"display busy\"}");
        return;
    }
    show_alert(clip, severity, when);
    lvgl_port_unlock();
    Serial.printf("ALERT clip='%s' severity='%s' (pending: %lu)\n", clip.c_str(), severity.c_str(),
                  (unsigned long)alert_count);

    char body[64];
    snprintf(body, sizeof(body), "{\"ok\":true,\"alert\":true,\"count\":%lu}", (unsigned long)alert_count);
    send_json(200, body);
}

static void handle_clear()
{
    if (!authorized()) {
        return;
    }
    if (!lvgl_port_lock(1000)) {
        send_json(503, "{\"ok\":false,\"error\":\"display busy\"}");
        return;
    }
    show_idle();
    lvgl_port_unlock();
    Serial.println("Alert cleared via API");
    send_json(200, "{\"ok\":true,\"alert\":false,\"count\":0}");
}

static void handle_status()
{
    char body[96];
    snprintf(body, sizeof(body), "{\"ok\":true,\"alert\":%s,\"count\":%lu,\"uptime_s\":%lu}",
             alert_active ? "true" : "false", (unsigned long)alert_count, (unsigned long)(millis() / 1000));
    send_json(200, body);
}

// ---------------------------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------------------------
static void on_wifi_connected()
{
    Serial.printf("Wi-Fi connected, IP %s\n", WiFi.localIP().toString().c_str());
    MDNS.end();
    if (MDNS.begin(SENTINEL_HOSTNAME)) {
        MDNS.addService("http", "tcp", 80);
        Serial.printf("mDNS: http://%s.local\n", SENTINEL_HOSTNAME);
    } else {
        Serial.println("mDNS start failed; use the IP address instead");
    }
}

static void refresh_idle_screen()
{
    if (alert_active) {
        return;  // never overwrite an active alert
    }
    char detail[160];
    idle_detail_text(detail, sizeof(detail));
    if (lvgl_port_lock(1000)) {
        if (!alert_active) {
            lv_label_set_text(ui_detail, detail);
        }
        lvgl_port_unlock();
    }
}

static void watch_wifi()
{
    bool connected = WiFi.status() == WL_CONNECTED;
    if (connected != wifi_was_connected) {
        wifi_was_connected = connected;
        if (connected) {
            on_wifi_connected();
        } else {
            Serial.println("Wi-Fi lost");
        }
        refresh_idle_screen();
    }
    if (!connected && millis() - last_wifi_retry_ms > WIFI_RETRY_INTERVAL_MS) {
        last_wifi_retry_ms = millis();
        WiFi.disconnect();
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
}

// ---------------------------------------------------------------------------------------------
void setup()
{
    Serial.begin(115200);

    Serial.println("Initializing board");
    Board *board = new Board();
    board->init();
#if LVGL_PORT_AVOID_TEARING_MODE
    auto lcd = board->getLCD();
    // When avoid tearing function is enabled, the frame buffer number should be set in the board driver
    lcd->configFrameBufferNumber(LVGL_PORT_DISP_BUFFER_NUM);
#if ESP_PANEL_DRIVERS_BUS_ENABLE_RGB && CONFIG_IDF_TARGET_ESP32S3
    auto lcd_bus = lcd->getBus();
    // Anti-tearing uses more PSRAM bandwidth on ESP32-S3: enable the RGB bounce buffer. Wi-Fi traffic competes
    // for the same bandwidth and shows up as flicker on the 1024x600 panel, hence the larger buffer.
    if (lcd_bus->getBasicAttributes().type == ESP_PANEL_BUS_TYPE_RGB) {
        static_cast<BusRGB *>(lcd_bus)->configRGB_BounceBufferSize(lcd->getFrameWidth() * RGB_BOUNCE_LINES);
    }
#endif
#endif
    assert(board->begin());

    Serial.println("Initializing LVGL");
    lvgl_port_init(board->getLCD(), board->getTouch());

    lvgl_port_lock(-1);
    build_ui();
    lvgl_port_unlock();

    show_boot_status("Connecting to Wi-Fi...");
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(SENTINEL_HOSTNAME);
    WiFi.setSleep(true);   // modem sleep: far less radio activity competing with the RGB panel for memory bandwidth
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
        delay(250);
    }
    last_wifi_retry_ms = millis();

    static const char *collected_headers[] = {"X-Sentinel-Token"};
    server.collectHeaders(collected_headers, 1);
    server.on("/alert", HTTP_POST, handle_alert);
    server.on("/clear", HTTP_POST, handle_clear);
    server.on("/status", HTTP_GET, handle_status);
    server.onNotFound([]() { send_json(404, "{\"ok\":false,\"error\":\"not found\"}"); });
    server.begin();
    Serial.println("HTTP server started on port 80");

    watch_wifi();  // announces mDNS and refreshes the idle screen (or retries in the background)
    if (!wifi_was_connected) {
        refresh_idle_screen();
    }
}

void loop()
{
    server.handleClient();
    watch_wifi();
    delay(10);  // 10 ms is plenty for alerts; polling faster only adds memory-bus contention
}
