// #include "string.h"
#include <WiFi.h>

#define MAX_SPEED 255

// const int swerve_Utime = 1000;
// const int swerve_Ltime = 400;
const int right_speed_offset = 10;
// const char* ssid = "Airtel_pran_7942";
// const char* password = "air49948";

#define buzzerPin 13
#define redLed 2
#define greenLed 4

#define In1 33
#define In2 25
#define In3 26
#define In4 27
#define Right_Speed 12
#define Left_Speed 32

#define IR1 35
#define IR2 34
#define IR3 21
#define IR4 22
#define IR5 23

#define PWR_IR1 18
#define PWR_IR2 19
#define PWR_IR3 5
#define PWR_IR4 15
#define PWR_IR5 14

int current_time;
bool sp_ack = 0;
String inp = "";

const bool wifi_connect = 1;
const int buzzer_loudness = 0;
// Pins Array
const byte irSensorPins[5] = { IR1, IR2, IR3, IR4, IR5 };
const byte irPowerPins[5] = { PWR_IR1, PWR_IR2, PWR_IR3, PWR_IR4, PWR_IR5 };
//  const byte motorPins[4] = { In3, In4, In1, In2 };
const byte motorPins[4] = { In1, In2, In3, In4 };

// Setting default values
// char sensorValues[5] = { (digitalRead(IR5)+'0'),
//                          (digitalRead(IR4)+'0'),
//                          (digitalRead(IR3)+'0'),
//                          (digitalRead(IR2)+'0'),
//                          (digitalRead(IR1)+'0') };
bool sensorValues[5] = { 0, 0, 0, 0, 0 };
// int time_swerve = 0;
void motor_control(bool LM_direction, bool RM_direction, short speed) {
  // Left Motor = In3,In4  --> 0
  // Right Motor = In1,In2 --> 1
  // dir = backward --> 0
  // dir = forward  --> 1
  // Speed ranges from 0 to 10
  if (speed == 0) {
    for (byte i = 0; i < 4; i++)
      digitalWrite(motorPins[i], 0);
    analogWrite(Left_Speed, 0);
    analogWrite(Right_Speed, 0);
    // digitalWrite(redLed, 1);
  } else {
    digitalWrite(motorPins[0], !RM_direction);
    digitalWrite(motorPins[1], RM_direction);
    digitalWrite(motorPins[2], !LM_direction);
    digitalWrite(motorPins[3], LM_direction);
    // if (LM_direction == 0 && RM_direction == 1)
    //   analogWrite(Left_Speed, speed - right_speed_offset);
    // else
    analogWrite(Left_Speed, speed);
    analogWrite(Right_Speed, speed);
    // digitalWrite(redLed, 0);
  }
}
void bot_move(byte command, byte speed) {
  /*
      Commands -
      0 --> Move Backward
      1 --> Take Left
      2 --> Take Right
      3 --> Move Forward
      4 --> Stop
    */
  switch (command) {

    case 0:
      {
        motor_control(0, 0, speed);
        break;
      }
    case 1:
      {
        motor_control(0, 1, speed);
        break;
      }
    case 2:
      {
        motor_control(1, 0, speed);
        break;
      }
    case 3:
      {
        motor_control(1, 1, speed);
        break;
      }
    default:
      {
        motor_control(0, 0, 0);
      }
  }
}


const int turn_delay = 600;
void turn_Clk(int in) {
  bot_move(4, 0);
  bot_move(2, MAX_SPEED);
  if (in == 1) {
    delay(400);
  } else
    delay(turn_delay);
  while (!(!sensorValues[0] && sensorValues[1] && !sensorValues[2])) {
    IR_Sensors_Update();
    bot_move(2, MAX_SPEED - 20);
  }
  //delay(1100);
  bot_move(4, 0);
}

void turn_AClk(int in) {
  bot_move(4, 0);
  bot_move(1, MAX_SPEED);
  if (in == 1)
    delay(100);
  else
    delay(turn_delay);
  while (!(!sensorValues[0] && sensorValues[1] && !sensorValues[2])) {
    IR_Sensors_Update();
    bot_move(1, MAX_SPEED - 20);
  }
  bot_move(4, 0);
}


void IR_Sensors_Update() {
  for (short i = 0; i <= 4; i++) {
    digitalWrite(irPowerPins[i], 1);
    delay(1);
    sensorValues[4 - i] = !!digitalRead(irSensorPins[i]);
    delay(1);
    digitalWrite(irPowerPins[i], 0);
  }
}
void jumpJunction() {
  const int moveforward_delay = 500;  // To move forward after encountering a Junction.
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);
}
void connect_to_wifi() {
  const char* ssid = "moto g52_3180";
  const char* password = "tesla8yvgrvq22";
  // const char* ssid = "Pranav’s iPhone";
  // const char* password = "Pranav12032003";
  WiFiServer server(80);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());
  server.begin();
  while (1) {
    WiFiClient client = server.available();
    if (client) {
      Serial.println("Bot connected");
      digitalWrite(redLed, 1);
      inp = client.readStringUntil('\r');
      Serial.println("Received: " + inp);
      client.println("Instructions Recieved");
      digitalWrite(redLed, 0);
      client.stop();
      Serial.println("Bot disconnected");
      break;
    }
  }
  delay(11000);
}

void setup() {
  Serial.begin(115200);
  pinMode(buzzerPin, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(redLed, OUTPUT);
  analogWrite(buzzerPin, 1024);

  for (short i = 0; i < 5; i++)
    pinMode(irSensorPins[i], INPUT_PULLUP);
  for (short i = 0; i < 5; i++)
    pinMode(irPowerPins[i], OUTPUT);
  for (short i = 0; i < 4; i++) {
    pinMode(motorPins[i], OUTPUT);
  }
  digitalWrite(motorPins[0], 0);
  digitalWrite(motorPins[1], 0);
  digitalWrite(motorPins[2], 0);
  digitalWrite(motorPins[3], 0);
  IR_Sensors_Update();

  if (wifi_connect == 0)
    connect_to_wifi();
  else
    inp = "l0a1l0f0l0f0l0f0El0a0l0f0l0r0l0f0l0r0bzf0l0a0l0r0azf0l0r0l0f0l0r0Dzf0l0f0Czu0l0a0l0f0l0r0l0a0ss";
  // inp = "Czu0";
  int ack = 0;
  while (!ack) {
    ack = backwards_line_follow();
  }
  bot_move(4, 0);
}



int line_follow() {
  IR_Sensors_Update();
  if ((!sensorValues[0] && sensorValues[1] && !sensorValues[2]) ||  //
      (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
    Serial.println("Moving Forward");
    bot_move(3, MAX_SPEED);
    return 0;
  } else if ((!sensorValues[0] && !sensorValues[1] && sensorValues[2]) ||  //
             (!sensorValues[0] && sensorValues[1] && sensorValues[2])) {
    Serial.println("Moving Right");
    bot_move(2, MAX_SPEED - 30);
    return 0;
  } else if ((sensorValues[0] && !sensorValues[1] && !sensorValues[2]) ||  //
             (sensorValues[0] && sensorValues[1] && !sensorValues[2]) ||   //
             (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && !sensorValues[4])) {
    Serial.println("Moving Left");
    bot_move(1, MAX_SPEED - 30);
    return 0;
  } else if ((!sensorValues[0] && sensorValues[1] && sensorValues[2] && sensorValues[3] && sensorValues[4]) ||  //
             (sensorValues[0] && sensorValues[1] && sensorValues[2])) {
    bot_move(4, 0);
    Serial.println("Stop");
    Serial.println("Junction Detected");
    return (1);
  } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && !sensorValues[4]) {
    bot_move(2, MAX_SPEED - 30);
    delay(10);
    return 0;
  } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && sensorValues[4]) {
    bot_move(1, MAX_SPEED - 20);
    delay(10);
    return 0;
  } else {
    bot_move(4, 0);
    return 0;
  }
}
int backwards_line_follow() {
  IR_Sensors_Update();
  if ((!sensorValues[0] && sensorValues[1] && !sensorValues[2]) ||  //
      (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
    Serial.println("Moving Forward");
    bot_move(0, MAX_SPEED);
    return 0;
  } else if ((!sensorValues[0] && !sensorValues[1] && sensorValues[2]) ||  //
             (!sensorValues[0] && sensorValues[1] && sensorValues[2])) {
    Serial.println("Moving Right");
    bot_move(2, MAX_SPEED - 30);
    return 0;
  } else if ((sensorValues[0] && !sensorValues[1] && !sensorValues[2]) ||  //
             (sensorValues[0] && sensorValues[1] && !sensorValues[2]) ||   //
             (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && !sensorValues[4])) {
    Serial.println("Moving Left");
    bot_move(1, MAX_SPEED - 30);
    return 0;
  } else if ((!sensorValues[0] && sensorValues[1] && sensorValues[2] && sensorValues[3] && sensorValues[4]) ||  //
             (sensorValues[0] && sensorValues[1] && sensorValues[2])) {
    bot_move(4, 0);
    Serial.println("Stop");
    Serial.println("Junction Detected");
    return (1);
  } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && !sensorValues[4]) {
    bot_move(2, MAX_SPEED - 30);
    delay(10);
    return 0;
  } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && sensorValues[4]) {
    bot_move(1, MAX_SPEED - 20);
    delay(10);
    return 0;
  } else {
    bot_move(4, 0);
    return 0;
  }
}

void loop() {
}