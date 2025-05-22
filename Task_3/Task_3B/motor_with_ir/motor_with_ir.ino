//Motor pin connections
#define in1 13
#define in2 12
#define in3 14
#define in4 27
//ir pin connection
#define irPin 22

void setup() {
  Serial.begin(115200);
  //Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  //sensor pin as input
  pinMode(irPin, INPUT);
  //Keeping all motors off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

}

void loop() {
  if(digitalRead(irPin)){
    //Move
    Serial.println("Moving Forward");
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);    

  }
  else{
    //Stop
    Serial.println("Stop");
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
  }
}