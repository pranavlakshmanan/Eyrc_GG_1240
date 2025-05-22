#include <WiFi.h>
#define in1 32
#define in2 33
#define in3 25
#define in4 26

// WiFi credentials
const char* ssid = "Tesla_5G";       //Enter your wifi hotspot ssid
const char* password = "Pi3141$Na60221023";  //Enter your wifi hotspot password
const uint16_t port = 8002;
const char* host = "192.168.29.174";  //Enter the ip address of your laptop after connecting it to wifi hotspot

// External peripherals
int buzzerPin = 13;
int redLed = 27;

char incomingPacket[80];
WiFiClient client;

String msg = "";
int counter = 0;

int start_time = millis();

void setup() {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  Serial.begin(115200);  //Serial to print data on Serial Monitor

  // Output Pins
  pinMode(buzzerPin, OUTPUT);
  pinMode(redLed, OUTPUT);
  // Initially off
  digitalWrite(buzzerPin, HIGH);  // Negative logic Buzzer
  digitalWrite(redLed, LOW);

  //Connecting to wifi
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }

  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());
}


void loop() {
  client.connect(host, port);
  if (!client.connected()) {
    Serial.println("Connection to host failed");
    digitalWrite(buzzerPin, HIGH);
    digitalWrite(redLed, LOW);
    delay(200);
    return;
  }
  msg = client.readStringUntil('\n');  //Read the message through the socket until new line char(\n)
  client.print(msg);                   //Send an acknowledgement to host(laptop)
  if (msg == "M") {
    Serial.println("Bot is moving");
    while (millis() - start_time <= 10000) {

      digitalWrite(in1, HIGH);
      digitalWrite(in2, 1);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, 1);
    }
  }
}