// #include "string.h"
#define MAX_SPEED 255

// const int swerve_Utime = 1000;
// const int swerve_Ltime = 400;
const int swerve_delay = 50;
const int right_speed_offset = 10;


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
int swerve_state = 1;
bool sp_ack = 0;
bool a_ack = 0;
bool b_ack = 0;
bool c_ack = 0;
bool d_ack = 0;
bool e_ack = 0;
bool f_ack = 0;
bool g_ack = 0;
bool h_ack = 0;
bool i_ack = 0;

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

const int swerve_step = 6;
// void swerve() {
//   if (swerve_state <= swerve_step / 2) {
//     bot_move(1, MAX_SPEED - 10);
//     Serial.println("Swerving Left");
//     delay(swerve_delay);
//   } else {
//     bot_move(2, MAX_SPEED);
//     Serial.println("Swerving Right");
//     delay(swerve_delay);
//   }
//   swerve_state = (swerve_state < swerve_step)
//                    ? ++swerve_state
//                    : 1;

//   bot_move(4, 0);
// }


void turn_Clk() {
  bot_move(4, 0);
  bot_move(2, MAX_SPEED);
  delay(550);
  while (!(sensorValues[0] && !sensorValues[1] && !sensorValues[2])) {
    IR_Sensors_Update();
    bot_move(2, MAX_SPEED);
  }
  //delay(1100);
  bot_move(4, 0);
}
void turn_AClk() {
  bot_move(4, 0);
  bot_move(1, MAX_SPEED);
  delay(550);
  while (!(!sensorValues[0] && !sensorValues[1] && sensorValues[2])) {
    IR_Sensors_Update();
    bot_move(1, MAX_SPEED);
  }
  bot_move(4, 0);
}
void IR_Sensors_Update() {
  for (short i = 0; i <= 4; i++) {
    digitalWrite(irPowerPins[i], 1);
    delay(2);
    sensorValues[4 - i] = !!digitalRead(irSensorPins[i]);
    delay(2);
    digitalWrite(irPowerPins[i], 0);
    if (digitalRead(irPowerPins[1]) == 0 && i == 1)
      digitalWrite(greenLed, 1);
    else
      digitalWrite(greenLed, 0);
  }
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

  // turn_Clk();
  // delay(1000);
  // turn_AClk();
  const int moveforward_delay = 500;  // To move forward after encountering a Junction.
  const int buzzer_loudness = 125;
  digitalWrite(redLed, 1);
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  digitalWrite(redLed, 0);
  analogWrite(buzzerPin, 1024);

  while (!a_ack)  // Start Point to A
    a_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(1, MAX_SPEED);
  delay(100);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);

  while (!b_ack)  // A to B
    b_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);

  while (!c_ack)  // B to C
    c_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay + 50);
  bot_move(4, 0);
  turn_Clk();

  while (!d_ack)  // C to D
    d_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay - 20);
  bot_move(4, 0);
  turn_AClk();

  while (!e_ack)  // D to E
    e_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);
  // bot_move(2, MAX_SPEED);
  // delay(100);
  bot_move(4, 0);
  turn_Clk();

  while (!f_ack)  // E to F
    f_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay - 20);
  bot_move(4, 0);
  turn_Clk();

  while (!g_ack)  // F to G
    g_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);

  while (!h_ack)  // G to H
    h_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);
  turn_Clk();

  while (!i_ack)  // H to I
    i_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  bot_move(4, 0);

  b_ack = 0;
  a_ack = 0;

  while (!b_ack)  // I to B
    b_ack = line_follow();
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);
  turn_AClk();

  while (!a_ack)  // B to A
    a_ack = line_follow();
  analogWrite(buzzerPin, buzzer_loudness);
  delay(1000);
  analogWrite(buzzerPin, 1024);
  bot_move(3, MAX_SPEED);
  delay(moveforward_delay);

  int cnt = 0;
  while (cnt < 20) {  // A to SP
    IR_Sensors_Update();
    sp_ack = line_follow();
    if (sensorValues[0] == 0 && sensorValues[1] == 0 && sensorValues[2] == 0 && sensorValues[3] == 0 && sensorValues[4] == 0)
      cnt++;
  }
  bot_move(4, 0);
  digitalWrite(redLed, 1);
  analogWrite(buzzerPin, buzzer_loudness);
  delay(5000);
  analogWrite(buzzerPin, 1024);
  digitalWrite(redLed, 0);
  bot_move(4, 0);
}


int line_follow() {
  IR_Sensors_Update();
  for (short i = 0; i <= 4; i++) {
    Serial.print(sensorValues[i]);
  }
  Serial.print("\n");
  if ((!sensorValues[0] && sensorValues[1] && !sensorValues[2]) || (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
    Serial.println("Moving Forward");
    bot_move(3, MAX_SPEED);
    return 0;
  } else if ((!sensorValues[0] && !sensorValues[1] && sensorValues[2]) || (!sensorValues[0] && sensorValues[1] && sensorValues[2])) {
    Serial.println("Moving Right");
    bot_move(2, MAX_SPEED - 20);
    return 0;
  } else if ((sensorValues[0] && !sensorValues[1] && !sensorValues[2]) || (sensorValues[0] && sensorValues[1] && !sensorValues[2]) || (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && !sensorValues[4])) {
    Serial.println("Moving Left");
    bot_move(1, MAX_SPEED - 20);
    return 0;
  } else if ((!sensorValues[0] && sensorValues[1] && sensorValues[2] && sensorValues[3] && sensorValues[4]) || (sensorValues[0] && sensorValues[1] && sensorValues[2] && sensorValues[3] && sensorValues[4]) || (sensorValues[0] && sensorValues[1] && sensorValues[2] && !sensorValues[3] && !sensorValues[4]) || (sensorValues[0] && sensorValues[1] && sensorValues[2] && sensorValues[3] && !sensorValues[4])) {
    bot_move(4, 0);
    Serial.println("Stop");
    Serial.println("Junction Detected");
    return (1);
  } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && !sensorValues[4]) {
    // swerve();
    // Serial.println("Swerving");
    bot_move(2, MAX_SPEED - 20);
    delay(20);

    return 0;
  } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && sensorValues[4]) {
    // swerve();
    // Serial.println("Swerving");
    bot_move(1, MAX_SPEED - 20);
    delay(20);
    return 0;
  } else {
    bot_move(4, 0);
    return 0;
  }
  // delay(5);
  //
  // delay(5);
}

void loop() {
  // int a = line_follow();
  //delay(10);
}