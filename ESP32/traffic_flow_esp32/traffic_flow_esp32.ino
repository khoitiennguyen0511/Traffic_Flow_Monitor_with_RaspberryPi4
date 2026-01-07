#include <TM1637Display.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <FirebaseESP32.h> 
#include <ArduinoJson.h>
// ================= CẤU HÌNH WIFI & MQTT =================
const char* ssid = "Iphone 12";
const char* password = "05112004";
const char* mqtt_server = "172.20.10.5"; 
const char* topic_sub_cmd = "he_thong_giam_sat_luu_luong/green_time_cmd"; 
const char* topic_sub_manual = "he_thong_giam_sat_luu_luong/control"; 
const char* topic_sub_vehicle_count = "he_thong_giam_sat_luu_luong/vehicle_count"; 
WiFiClient espClient;
PubSubClient client(espClient);
// ================= CẤU HÌNH FIREBASE =================
#define FIREBASE_HOST "traffic-flow-monitor-e7ede-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH "AIzaSyDcQwwzi0I1bS_-U2Uf0l0HvOSLZVtuoxI" 
FirebaseData fbdo;
// Thêm hai đối tượng cấu hình
FirebaseConfig firebaseConfig;
FirebaseAuth firebaseAuth;
// ================= CẤU HÌNH PHẦN CỨNG =================
#define R_1 23
#define Y_1 22
#define G_1 21
#define R_2 19
#define Y_2 18
#define G_2 5
#define TM1637_CLK 17
#define TM1637_DIO 16
#define SW0 32
#define SW1 33
#define SW2 26 
#define SW3 27
TM1637Display display(TM1637_CLK, TM1637_DIO);
// ================= BIẾN TOÀN CỤC & ENUM =================
enum TrafficMode { MODE_AUTO, MODE_MANUAL };
enum TrafficState { STATE_1_GREEN, STATE_1_YELLOW, STATE_2_GREEN, STATE_2_YELLOW };
TrafficMode currentMode = MODE_AUTO;
TrafficState currentState = STATE_1_GREEN;
bool sleepMode = false;
int greenTime1 = 10; 
int greenTime2 = 10; 
int yellowTime = 3;
unsigned long lastChangeTime = 0;
int currentTimeLeft1 = 0;
int currentTimeLeft2 = 0;
unsigned long debounceDelay = 50;
unsigned long lastMqttPub = 0;
int regionCounts[4][4] = {{0}}; // 4 regions, 4 classes: 0-motorbike,1-car,2-bus,3-truck
String densityLevel = "LOW"; // Mật độ mặc định
const int MIN_GREEN = 10;
const int MAX_GREEN = 30;
// ================= KHAI BÁO HÀM =================
void updateLEDs();
void updateTimeLeft();
void displayTime();
void publishTrafficStatusToFirebase(); 
void setup_wifi();
void reconnect();
void enterSleepMode();
void checkSleepMode();
void checkButtons();
void autoMode();
void calculateGreenTimes();
// ================= HÀM HỖ TRỢ WIFI & MQTT =================
void setup_wifi() {
    delay(10);
    Serial.println();
    Serial.print("Dang ket noi WiFi: ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi da ket noi");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}
// Hàm nhận tin nhắn từ MQTT
void callback(char* topic, byte* payload, unsigned int length) {
    String message;
    for (int i = 0; i < length; i++) {
        message += (char)payload[i];
    }
    Serial.print("Nhan lenh MQTT tu topic: ");
    Serial.print(topic);
    Serial.print(", Message: ");
    Serial.println(message);
    // 1. XỬ LÝ LỆNH GREEN TIME 
    if (String(topic) == topic_sub_cmd) {
        if (currentMode == MODE_AUTO) { 
            int new_green_time = message.toInt();
            if (new_green_time > 0) {
                greenTime1 = new_green_time;
                greenTime2 = new_green_time; // Nếu lệnh thủ công, set đồng đều
                Serial.print("Cap nhat Green Time moi (thủ công): ");
                Serial.println(new_green_time);
                
                lastChangeTime = millis();
                updateTimeLeft();
                updateLEDs();
            }
        }
    }
    
    // 2. XỬ LÝ LỆNH ĐIỀU KHIỂN TAY
    else if (String(topic) == topic_sub_manual) {
        if (message == "AUTO") {
            currentMode = MODE_AUTO;
            Serial.println("Remote: Chuyen sang AUTO");
        } else if (message == "MANUAL") {
            currentMode = MODE_MANUAL;
            Serial.println("Remote: Chuyen sang MANUAL");
        } else if (message == "SLEEP") {
            sleepMode = !sleepMode;
            if(!sleepMode) { 
                lastChangeTime = millis();
                updateTimeLeft();
                updateLEDs();
            }
        }
    }
    
    // 3. NHẬN DỮ LIỆU VEHICLE COUNT TỪ RPi
    else if (String(topic) == topic_sub_vehicle_count) {
        Serial.println("Nhận dữ liệu vehicle count từ RPi");
        DynamicJsonDocument doc(1024);
        DeserializationError error = deserializeJson(doc, message);
        if (error) {
            Serial.print("deserializeJson() failed: ");
            Serial.println(error.c_str());
            return;
        }
        for (int i = 0; i < 4; i++) {
            String reg = "region_" + String(i+1);
            JsonObject obj = doc[reg];
            regionCounts[i][0] = obj["motorbike"] | 0;
            regionCounts[i][1] = obj["car"] | 0;
            regionCounts[i][2] = obj["bus"] | 0;
            regionCounts[i][3] = obj["truck"] | 0;
        }
        // calculate total for density
        int totalVehicles = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                totalVehicles += regionCounts[i][j];
            }
        }
        if (totalVehicles > 50) densityLevel = "HIGH";
        else if (totalVehicles > 20) densityLevel = "MEDIUM";
        else densityLevel = "LOW";
        
        // Nếu AUTO, tính green times adaptive
        if (currentMode == MODE_AUTO) {
            calculateGreenTimes();
            lastChangeTime = millis();
            updateTimeLeft();
            updateLEDs();
        }
        // Update Firebase immediately for realtime
        publishTrafficStatusToFirebase();
    }
}
void reconnect() {
    while (!client.connected()) {
        Serial.print("Dang ket noi MQTT...");
        if (client.connect("ESP32_TrafficLight_Client")) {
            Serial.println("Da ket noi!");
            client.subscribe(topic_sub_cmd); 
            client.subscribe(topic_sub_manual); 
            client.subscribe(topic_sub_vehicle_count); // Thêm subscribe vehicle count
        } else {
            Serial.print("Loi, rc=");
            Serial.print(client.state());
            Serial.println(" thu lai sau 5 giay");
            delay(5000);
        }
    }
}
// Hàm tính green times dựa trên counts (adaptive)
void calculateGreenTimes() {
    int countLane1 = 0;
    for (int j = 0; j < 4; j++) {
        countLane1 += regionCounts[0][j] + regionCounts[2][j];
    }
    int countLane2 = 0;
    for (int j = 0; j < 4; j++) {
        countLane2 += regionCounts[1][j] + regionCounts[3][j];
    }
    int totalCounts = countLane1 + countLane2;
    
    if (totalCounts == 0) {
        greenTime1 = MIN_GREEN;
        greenTime2 = MIN_GREEN;
    } else {
        float prop1 = (float)countLane1 / totalCounts;
        greenTime1 = MIN_GREEN + (int)((MAX_GREEN - MIN_GREEN) * prop1);
        greenTime2 = MIN_GREEN + (int)((MAX_GREEN - MIN_GREEN) * (1 - prop1));
    }
    
    // Giới hạn
    if (greenTime1 < MIN_GREEN) greenTime1 = MIN_GREEN;
    if (greenTime1 > MAX_GREEN) greenTime1 = MAX_GREEN;
    if (greenTime2 < MIN_GREEN) greenTime2 = MIN_GREEN;
    if (greenTime2 > MAX_GREEN) greenTime2 = MAX_GREEN;
    
    Serial.print("Green Time 1: ");
    Serial.println(greenTime1);
    Serial.print("Green Time 2: ");
    Serial.println(greenTime2);
}
// Hàm Gửi Trạng thái lên Firebase
void publishTrafficStatusToFirebase() {
    String currentStateStr;
    switch(currentState) {
        case STATE_1_GREEN: currentStateStr = "1_GREEN"; break;
        case STATE_1_YELLOW: currentStateStr = "1_YELLOW"; break;
        case STATE_2_GREEN: currentStateStr = "2_GREEN"; break;
        case STATE_2_YELLOW: currentStateStr = "2_YELLOW"; break;
    }
    FirebaseJson json;
    
    if (sleepMode) {
      json.set("esp32_mode", "SLEEP");
      json.set("esp32_state", "ALL_OFF");
      json.set("time_left_1", 0);
      json.set("time_left_2", 0);
    } else {
      json.set("esp32_mode", currentMode == MODE_AUTO ? "AUTO" : "MANUAL");
      json.set("esp32_state", currentStateStr.c_str());
      json.set("time_left_1", currentTimeLeft1);
      json.set("time_left_2", currentTimeLeft2);
      json.set("green_time_1", greenTime1); 
      json.set("green_time_2", greenTime2); 
      json.set("latest_status", densityLevel.c_str());
    }
    
    // Thêm region counts chi tiết
    FirebaseJson regionObject;
    int totalVehiclesAll = 0;
    for (int i = 0; i < 4; i++) {
        int regionTotal = 0;
        for (int j = 0; j < 4; j++) {
            regionTotal += regionCounts[i][j];
        }
        
        String regionKey = "Region_" + String(i + 1); 
        
        FirebaseJson regionJson;
        regionJson.set("motorbike", regionCounts[i][0]);
        regionJson.set("car", regionCounts[i][1]);
        regionJson.set("bus", regionCounts[i][2]);
        regionJson.set("truck", regionCounts[i][3]);
        regionJson.set("total_in_region", regionTotal);
        
        // Gán đối tượng con vào khóa tùy chỉnh trong regionObject
        regionObject.set(regionKey.c_str(), regionJson); 
        
        totalVehiclesAll += regionTotal;
    }
    
    json.set("region_counts", regionObject); 
    json.set("total_vehicles_all_time", totalVehiclesAll);
    json.set("timestamp", String(time(nullptr)));
    
    // Cập nhật cùng document với RPi
    if (Firebase.updateNode(fbdo, "/traffic_system/latest_status", json)) {
      // Thành công, không cần in ra
    } else {
      Serial.print("Firebase update failed: ");
      Serial.println(fbdo.errorReason());
    }
}
// ================= CÁC HÀM LOGIC ĐÈN GIAO THÔNG =================
void enterSleepMode() {
    digitalWrite(R_1, LOW); digitalWrite(Y_1, LOW); digitalWrite(G_1, LOW);
    digitalWrite(R_2, LOW); digitalWrite(Y_2, LOW); digitalWrite(G_2, LOW);
    uint8_t sleepData[] = {
        SEG_A | SEG_F | SEG_G | SEG_C | SEG_D, 
        SEG_D | SEG_E | SEG_F, 
        0x00, 0x00
    };
    display.setSegments(sleepData);
    delay(100);
}
void checkSleepMode() {
    static unsigned long lastPress = 0;
    if (digitalRead(SW3) == LOW && millis() - lastPress > 300) {
        delay(debounceDelay);
        if (digitalRead(SW3) == LOW) {
            sleepMode = !sleepMode;
            Serial.println(sleepMode ? "Vào chế độ ngủ" : "Thoát chế độ ngủ");
            lastPress = millis();
            if (!sleepMode) {
                lastChangeTime = millis();
                updateTimeLeft();
                updateLEDs();
            }
        }
    }
}
void checkButtons() {
    static unsigned long lastPressSW0 = 0;
    static unsigned long lastPressSW1 = 0;
    
    // SW0: Auto/Manual
    if (digitalRead(SW0) == LOW && millis() - lastPressSW0 > 300) {
        delay(debounceDelay);
        if (digitalRead(SW0) == LOW) {
            currentMode = (currentMode == MODE_AUTO) ? MODE_MANUAL : MODE_AUTO;
            Serial.println(currentMode == MODE_AUTO ? "Chế độ TỰ ĐỘNG" : "Chế độ THỦ CÔNG");
            lastChangeTime = millis();
            updateTimeLeft();
            lastPressSW0 = millis();
        }
    }
    
    // SW1: Next State (Manual)
    if (currentMode == MODE_MANUAL && digitalRead(SW1) == LOW && millis() - lastPressSW1 > 300) {
        delay(debounceDelay);
        if (digitalRead(SW1) == LOW) {
            switch(currentState) {
                case STATE_1_GREEN: currentState = STATE_1_YELLOW; break;
                case STATE_1_YELLOW: currentState = STATE_2_GREEN; break;
                case STATE_2_GREEN: currentState = STATE_2_YELLOW; break;
                case STATE_2_YELLOW: currentState = STATE_1_GREEN; break;
            }
            updateLEDs();
            lastChangeTime = millis();
            updateTimeLeft();
            Serial.println("Chuyển trạng thái thủ công");
            lastPressSW1 = millis();
        }
    }
}
void autoMode() {
    unsigned long currentTime = millis();
    unsigned long elapsedTime = (currentTime - lastChangeTime) / 1000;
    
    switch(currentState) {
        case STATE_1_GREEN:
            currentTimeLeft1 = greenTime1 - elapsedTime;
            currentTimeLeft2 = greenTime1 + yellowTime - elapsedTime;
            if (elapsedTime >= greenTime1) {
                currentState = STATE_1_YELLOW;
                updateLEDs();
                lastChangeTime = currentTime;
            }
            break;
        case STATE_1_YELLOW:
            currentTimeLeft1 = yellowTime - elapsedTime;
            currentTimeLeft2 = yellowTime - elapsedTime;
            if (elapsedTime >= yellowTime) {
                currentState = STATE_2_GREEN;
                updateLEDs();
                lastChangeTime = currentTime;
            }
            break;
        case STATE_2_GREEN:
            currentTimeLeft1 = greenTime2 + yellowTime - elapsedTime;
            currentTimeLeft2 = greenTime2 - elapsedTime;
            if (elapsedTime >= greenTime2) {
                currentState = STATE_2_YELLOW;
                updateLEDs();
                lastChangeTime = currentTime;
            }
            break;
        case STATE_2_YELLOW:
            currentTimeLeft1 = yellowTime - elapsedTime;
            currentTimeLeft2 = yellowTime - elapsedTime;
            if (elapsedTime >= yellowTime) {
                currentState = STATE_1_GREEN;
                updateLEDs();
                lastChangeTime = currentTime;
            }
            break;
    }
    if (currentTimeLeft1 < 0) currentTimeLeft1 = 0;
    if (currentTimeLeft2 < 0) currentTimeLeft2 = 0;
}
void updateLEDs() {
    digitalWrite(R_1, LOW); digitalWrite(Y_1, LOW); digitalWrite(G_1, LOW);
    digitalWrite(R_2, LOW); digitalWrite(Y_2, LOW); digitalWrite(G_2, LOW);
    switch(currentState) {
        case STATE_1_GREEN: digitalWrite(G_1, HIGH); digitalWrite(R_2, HIGH); break;
        case STATE_1_YELLOW: digitalWrite(Y_1, HIGH); digitalWrite(R_2, HIGH); break;
        case STATE_2_GREEN: digitalWrite(R_1, HIGH); digitalWrite(G_2, HIGH); break;
        case STATE_2_YELLOW: digitalWrite(R_1, HIGH); digitalWrite(Y_2, HIGH); break;
    }
}
void updateTimeLeft() {
    switch(currentState) {
        case STATE_1_GREEN: currentTimeLeft1 = greenTime1; currentTimeLeft2 = greenTime1 + yellowTime; break;
        case STATE_1_YELLOW: currentTimeLeft1 = yellowTime; currentTimeLeft2 = yellowTime; break;
        case STATE_2_GREEN: currentTimeLeft1 = greenTime2 + yellowTime; currentTimeLeft2 = greenTime2; break;
        case STATE_2_YELLOW: currentTimeLeft1 = yellowTime; currentTimeLeft2 = yellowTime; break;
    }
}
void displayTime() {
    if (sleepMode) {
    } else if (currentMode == MODE_MANUAL) {
        uint8_t manualData[] = { SEG_G, SEG_G, SEG_G, SEG_G };
        display.setSegments(manualData);
    } else {
        int displayNumber = (currentTimeLeft1 * 100) + currentTimeLeft2;
        display.showNumberDecEx(displayNumber, 0b1110000, true);
    }
}
// ================= SETUP & LOOP =================
void setup() {
    Serial.begin(115200);
    
    // Khởi tạo chân
    pinMode(R_1, OUTPUT); pinMode(Y_1, OUTPUT); pinMode(G_1, OUTPUT);
    pinMode(R_2, OUTPUT); pinMode(Y_2, OUTPUT); pinMode(G_2, OUTPUT);
    pinMode(SW0, INPUT_PULLUP); pinMode(SW1, INPUT_PULLUP);
    pinMode(SW2, INPUT_PULLUP); pinMode(SW3, INPUT_PULLUP);
    
    display.setBrightness(0x0f);
    display.clear();
    
    // Kết nối WiFi & MQTT
    setup_wifi();
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
    
    // KHỞI TẠO FIREBASE
    Serial.println("Khoi tao Firebase...");
    configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov");
    // Chờ cho thời gian được cập nhật (tối đa 5s)
    unsigned long start = millis();
    while (time(nullptr) < 100000 && millis() - start < 5000) {
        delay(100);
    }
    // 1. Gán Host và API Key vào đối tượng cấu hình (FirebaseConfig)
    firebaseConfig.host = FIREBASE_HOST;
    firebaseConfig.api_key = FIREBASE_AUTH; 
    // 2. CẤU HÌNH firebaseAuth
    firebaseAuth.user.email = "test@gmail.com";     
    firebaseAuth.user.password = "123456";           
    // 3. Khởi tạo Firebase
    Firebase.begin(&firebaseConfig, &firebaseAuth);
    Firebase.reconnectWiFi(true);
    
    // Khởi tạo trạng thái
    lastChangeTime = millis();
    updateTimeLeft();
    updateLEDs();
    displayTime();
    Serial.println("He thong da khoi dong");
}
void loop() {
    // 1. Duy trì kết nối MQTT
    if (!client.connected()) {
        reconnect();
    }
    client.loop();
    // 2. Logic Đèn Giao Thông
    checkSleepMode();
    
    if (sleepMode) {
        enterSleepMode();
        // Gửi trạng thái SLEEP lên Firebase
        if (millis() - lastMqttPub > 2000) { 
            publishTrafficStatusToFirebase();
            lastMqttPub = millis();
        }
        return;
    }
    
    checkButtons();
    
    if (currentMode == MODE_AUTO) {
        autoMode();
    }
    
    displayTime();
    
    // 3. Gửi dữ liệu TRẠNG THÁI ĐÈN + VEHICLE COUNTS lên FIREBASE mỗi 1 giây
    if (millis() - lastMqttPub > 1000) {
        publishTrafficStatusToFirebase();
        lastMqttPub = millis();
    }
    
    delay(10); 
}