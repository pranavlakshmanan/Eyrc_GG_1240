#include "string.h"
#define MAX_SPEED 255

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
// Pins Array
const byte irSensorPins[5] = { IR1, IR2, IR3, IR4, IR5 };
//  const byte motorPins[4] = { In3, In4, In1, In2 };
const byte motorPins[4] = { In1, In2, In3, In4 };
int t;
// Setting default values
// char sensorValues[5] = { (digitalRead(IR5)+'0'),
//                          (digitalRead(IR4)+'0'),
//                          (digitalRead(IR3)+'0'),
//                          (digitalRead(IR2)+'0'),
//                          (digitalRead(IR1)+'0') };
char sensorValues[5] = { '0', '0', '0', '0', '0' };

void motor_control(bool LM_direction, bool RM_direction, int speed) {
  // Left Motor = In4,In3  --> 0
  // Right Motor = In2,In1 --> 1
  // dir = backward --> 0
  // dir = forward  --> 1
  // Speed ranges from 0 to 10
  int freq = 500;
  if (speed == 0) {
    for (byte i = 0; i < 4; i++)
      digitalWrite(motorPins[i], 0);
    digitalWrite(redLed, 1);
    delay(100);
    digitalWrite(redLed, 0);
  } else {
    //int ontime = (1 / freq) / speed;
    digitalWrite(motorPins[0], !RM_direction);
    digitalWrite(motorPins[1], RM_direction);
    digitalWrite(motorPins[2], !LM_direction);
    digitalWrite(motorPins[3], LM_direction);
    analogWrite(Right_Speed, speed);
    analogWrite(Left_Speed, speed);
    // delay(ontime);
    // digitalWrite(Right_Speed, 0);
    // digitalWrite(Left_Speed, 0);
    // delay((1 / freq) - ontime);
  }
}
void bot_move(byte command, int speed) {
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
        motor_control(1, 0, speed);
        break;
      }
    case 2:
      {
        motor_control(0, 1, speed);
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
void setup() {
  Serial.begin(115200);
  pinMode(buzzerPin, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(redLed, OUTPUT);
  digitalWrite(buzzerPin, 1);

  // for (byte i = 0; i < 5; i++) {
  //   pinMode(irSensorPins[i], INPUT);
  // }
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
  t = millis();
}

void loop() {
  // for (short i = 4; i >= 0; i--) {
  //   sensorValues[i] = (digitalRead(irSensorPins[i]) + '0');
  // }
  // for (short i = 4; i >= 0; i--) {
  //   Serial.print(sensorValues[i]);
  // }
  // Serial.print("\n");
  // if (sensorValues[0] == 1 && sensorValues[1] == 0 && sensorValues[2] == 1 && sensorValues[3] == 0 && sensorValues[4] == 1)
  //   bot_move(3, 10);
  // else if (sensorValues[0] == 0 && sensorValues[1] == 0 && sensorValues[2] == 0 && sensorValues[3] == 1 && sensorValues[4] == 0)
  //   bot_move(1, 10);
  // else if (sensorValues[0] == 0 && sensorValues[1] == 1 && sensorValues[2] == 0 && sensorValues[3] == 0 && sensorValues[4] == 0)
  //   bot_move(2, 10);
  // else
  //   bot_move(4, 0);
  // for (int i = 50; i < 61; i++) {
  //   bot_move(3, i);
  //   delay(100);
  // }
  int test_speed = 220;
  // while (millis() - t < 3000) {
  //   bot_move(3, test_speed);
  //   delay(100);
  // }
  // while (millis() - t < 6000) {
  //   bot_move(0, test_speed);
  //   delay(100);
  // }
  while (millis() - t < 9000) {
    bot_move(3, MAX_SPEED);
    delay(100);
  }
  // while (millis() - t < 12000) {
  //   bot_move(2, test_speed);i
  //   delay(100);
  // }
  bot_move(4, 0);
}