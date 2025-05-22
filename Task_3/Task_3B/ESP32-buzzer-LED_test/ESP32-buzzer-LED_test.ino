#define RED_LED 1
#define GREEN_LED 2
#define BUZZER 3

void setup() {
  // put your setup code here, to run once:
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(RED_LED, 1);
  digitalWrite(GREEN_LED, 1);
  digitalWrite(BUZZER, 1);
  delay(100);
  digitalWrite(RED_LED, 0);
  digitalWrite(GREEN_LED, 0);
  digitalWrite(BUZZER, 0);
  delay(100);
}
