#include "string.h"
#define MAX_SPEED 255
// const int swerve_Utime = 1000;
// const int swerve_Ltime = 400;
const int swerve_delay = 100;
const int right_speed_offset = 10;


#define buzzerPin 13
#define redLed 12
#define greenLed 4

#define In1 32
#define In2 33
#define In3 25
#define In4 26
#define Right_Speed 27
#define Left_Speed 2

#define IR1 18
#define IR2 19
#define IR3 21
#define IR4 22
#define IR5 23

int current_time;
int swerve_state = 0;

// Pins Array
const byte irSensorPins[5] = { IR1, IR2, IR3, IR4, IR5 };
//  const byte motorPins[4] = { In3, In4, In1, In2 };
const byte motorPins[4] = { In1, In2, In3, In4 };

// Setting default values
// char sensorValues[5] = { (digitalRead(IR5)+'0'),
//                          (digitalRead(IR4)+'0'),
//                          (digitalRead(IR3)+'0'),
//                          (digitalRead(IR2)+'0'),
//                          (digitalRead(IR1)+'0') };
char sensorValues[5] = { '0', '0', '0', '0', '0' };
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
    digitalWrite(redLed, 1);
  } else {
    digitalWrite(motorPins[0], !RM_direction);
    digitalWrite(motorPins[1], RM_direction);
    digitalWrite(motorPins[2], !LM_direction);
    digitalWrite(motorPins[3], LM_direction);
    if (LM_direction == 0 && RM_direction == 1)
      analogWrite(Right_Speed, speed - right_speed_offset);
    else
      analogWrite(Right_Speed, speed);
    analogWrite(Left_Speed, speed);
    digitalWrite(redLed, 0);
  }
}
void bot_move(byte command, byte speed) {
  /*
      Commands -
      0 --> Move Backward
      1 --> Take Right
      2 --> Take Left
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

void swerve() {
  if (swerve_state == 0 || swerve_state == 1) {
    bot_move(2, MAX_SPEED);
    Serial.println("Swerving Left");
  } else if (swerve_state == 2 || swerve_state == 3) {
    bot_move(1, MAX_SPEED);
    Serial.println("Swerving Right");
  }
  swerve_state = (swerve_state < 3)
                   ? ++swerve_state
                   : 0;
  delay(swerve_delay);
  bot_move(4, 0);
}
void setup() {
  Serial.begin(115200);
  pinMode(buzzerPin, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(redLed, OUTPUT);
  digitalWrite(buzzerPin, 1);

  for (byte i = 0; i < 5; i++) {
    pinMode(irSensorPins[i], INPUT_PULLUP);
  }
  for (byte i = 0; i < 4; i++) {
    pinMode(motorPins[i], OUTPUT);
  }
  digitalWrite(motorPins[0], 0);
  digitalWrite(motorPins[1], 0);
  digitalWrite(motorPins[2], 0);
  digitalWrite(motorPins[3], 0);
  for (short i = 4; i >= 0; i--) {
    sensorValues[i] = digitalRead(irSensorPins[i]);
  }
  current_time = millis();
}

void loop() {
  for (short i = 4; i >= 0; i--) {
    sensorValues[i] = (digitalRead(irSensorPins[i]) + '0');
  }
  for (short i = 4; i >= 0; i--) {
    Serial.print(sensorValues[i]);
  }
  Serial.print("\n");

  if (strcmp(sensorValues, "00010") == 0) {
    Serial.println("Moving Forward");
    bot_move(3, MAX_SPEED);
  } else if (strcmp(sensorValues, "00001") == 0 || strcmp(sensorValues, "00011") == 0) {
    Serial.println("Moving Right");
    bot_move(1, MAX_SPEED);
  } else if (strcmp(sensorValues, "00100") == 0 || strcmp(sensorValues, "00110") == 0) {
    Serial.println("Moving Left");
    bot_move(2, MAX_SPEED);
  } else if (strcmp(sensorValues, "00111") == 0) {
    Serial.println("Junction Detected");
    digitalWrite(greenLed, 1);
    delay(500);
    digitalWrite(greenLed, 0);
    // Serial.println("Moving Forward");
    // bot_move(3, MAX_SPEED - TURNING_OFFSET);
    Serial.println("Stop");
    bot_move(4, 0);
  } else if (strcmp(sensorValues, "11111") == 0) {
    Serial.println("Stop");
    bot_move(4, 0); 
  } else {
    // if (millis() - time_swerve > swerve_Utime || time_swerve == 0)
    //   time_swerve = millis();
    // swerve();
    // time_swerve == 0;
    // Serial.println("Swerving");
    swerve();
    Serial.println("Swerving");

    //bot_move(4, 0);
  }

  // bot_move(0, MAX_SPEED);
  // delay(2000);
  // bot_move(1, MAX_SPEED);
  // delay(2000);
  // bot_move(2, MAX_SPEED);
  // delay(2000);
  // bot_move(3, MAX_SPEED);
  // delay(2000);
  //bot_move(4, 0);

  delay(50);
}