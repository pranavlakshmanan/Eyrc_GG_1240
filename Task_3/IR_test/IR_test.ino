//Ir_Pins
const byte IR1 = 18,
           IR2 = 19,
           IR3 = 21,
           IR4 = 22,
           IR5 = 23;  //left most in forward


// Pins Array
const byte irSensorPins[5] = { IR1, IR2, IR3, IR4, IR5 };

// Setting default values
char sensorValues[5] = { (digitalRead(IR5)+'0'),
                         (digitalRead(IR4)+'0'),
                         (digitalRead(IR3)+'0'),
                         (digitalRead(IR2)+'0'),
                         (digitalRead(IR1)+'0') };


void setup() {
  Serial.begin(115200);

  for (byte i = 0; i < 5; i++) {
    pinMode(irSensorPins[i], INPUT_PULLUP);
  }
  for (short i = 4; i >= 0; i--) {
    sensorValues[i] = digitalRead(irSensorPins[i]);
  }
}

void loop() {
  for (short i = 4; i >= 0; i--) {
    sensorValues[i] = (digitalRead(irSensorPins[i])+'0');
  }
  for (short i = 4; i >= 0; i--) {
    Serial.print(sensorValues[i]);
  }
  Serial.print("\n");
  delay(100);  // Adjust the delay based on your requirements
}