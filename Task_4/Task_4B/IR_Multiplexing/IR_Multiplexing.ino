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
// #define green 4

byte IR_PWR[5] = { PWR_IR1, PWR_IR2, PWR_IR3, PWR_IR4, PWR_IR5 };
byte IR[5] = { IR1, IR2, IR3, IR4, IR5 };
void setup() {
  Serial.begin(115200);
  // pinMode(IR1, INPUT);
  // pinMode(IR2, INPUT);
  for (int i = 0; i < 5; i++)
    pinMode(IR[i], INPUT_PULLUP);
  for (int i = 0; i < 5; i++)
    pinMode(IR_PWR[i], OUTPUT);
  // pinMode(green, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i < 5; i++) {
    digitalWrite(IR_PWR[i], 1);
    delay(2);
    Serial.print(digitalRead(IR[i]));
    delay(2);
    digitalWrite(IR_PWR[i], 0);
    // if (digitalRead(IR[1]) == 0 && i == 1)
    //   digitalWrite(green, 1);
    // else
    //   digitalWrite(green, 0);
  }
  Serial.println("");
}
