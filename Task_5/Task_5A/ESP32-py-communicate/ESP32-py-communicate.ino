#include <WiFi.h>

// const char* ssid = "moto g52_3180";
// const char* password = "tesla8yvgrvq22";
const char* ssid = "Airtel_pran_7942";
const char* password = "air49948";

WiFiServer server(80);

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (client) {
    Serial.println("New client connected");

    // Read the incoming data
    String request = client.readStringUntil('\r');
    Serial.println("Received: " + request);

    // Send a response to the client
    client.println("Hello from ESP32!");

    // Close the connection
    client.stop();
    Serial.println("Client disconnected");
  }
}
