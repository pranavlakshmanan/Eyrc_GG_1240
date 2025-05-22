#include <WiFi.h>

#define buzzerPin 13
#define redLed 12
#define greenLed 4


const char* ssid = "Tesla";
const char* password = "Pi3141@Na60221023";
const char* host = "192.168.29.174";  //Enter the ip address of your laptop after connecting it to wifi hotspot
// const char* ssid = "moto g52_3180";
// const char* password = "tesla8yvgrvq22";
// const char* host = "192.168.71.121"; 
const uint16_t port = 1240;


char incomingPacket[80];
WiFiClient client;

String msg = "0";
int counter = 0;

void setup() {

  Serial.begin(115200);
  pinMode(buzzerPin, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(redLed, OUTPUT);

  digitalWrite(buzzerPin, HIGH);
  digitalWrite(greenLed, LOW);
  digitalWrite(redLed, LOW);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(10);
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
