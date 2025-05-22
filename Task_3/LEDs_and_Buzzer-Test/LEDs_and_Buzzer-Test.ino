#define green 4
#define red 2
#define buzz 13

void setup() {
  // put your setup code here, to run once:
  pinMode(green, OUTPUT);
  pinMode(red, OUTPUT);
  pinMode(buzz, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(green, 1);
  digitalWrite(red, 1);
  digitalWrite(buzz, 0);
  delay(1000);
  digitalWrite(green, 0);
  digitalWrite(red, 0);
  digitalWrite(buzz, 1);
  delay(1000);
}
