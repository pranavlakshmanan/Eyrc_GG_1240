#include <WiFi.h>
#define buzzerPin 13
#define redLed 2
#define greenLed 4

// WiFi credentials
// const char* ssid = "Airtel_pran_7942";
// const char* password = "air49948";
const char* ssid = "moto g52_3180";
const char* password = "tesla8yvgrvq22";
const uint16_t port = 1024;
const char* host = "192.168.0.121";  //Enter the ip address of your laptop after connecting it to wifi hotspot




char incomingPacket[80];
WiFiClient client;

String msg = "0";
int counter = 0;



void setup() {

  Serial.begin(115200);  //Serial to print data on Serial Monitor

  // Output Pins
  pinMode(buzzerPin, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(redLed, OUTPUT);
  // Initially off
  digitalWrite(buzzerPin, HIGH);  // Negative logic Buzzer
  digitalWrite(greenLed, LOW);
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

  if (!client.connect(host, port)) {
    Serial.println("Connection to host failed");
    digitalWrite(buzzerPin, HIGH);
    digitalWrite(greenLed, LOW);
    delay(200);
    return;
  }

  while (1) {
    msg = client.readStringUntil('\n');  //Read the message through the socket until new line char(\n)
    client.print("Hello from ESP32!");   //Send an acknowledgement to host(laptop)
    counter = msg.toInt();
    Serial.println(counter);  //Print data on Serial monitor
    if (counter % 2 == 0) {
      digitalWrite(buzzerPin, LOW);  //If counter value is even turn on Buzzer & LEDs
      digitalWrite(greenLed, HIGH);
      digitalWrite(redLed, HIGH);

    } else {
      digitalWrite(buzzerPin, HIGH);  //Else keep them off
      digitalWrite(greenLed, LOW);
      digitalWrite(redLed, LOW);
    }
  }
}
