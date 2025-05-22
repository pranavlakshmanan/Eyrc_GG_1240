/*
 * Team Id: GG_1240
 * Author List: Srikar Bharadwaj, Pranav Lakshmanan
 * Filename: Follow_Path.ino
 * Theme: GeoGuide
 * Functions: motor_control(bool,bool,short), bot_move(byte,byte) , turn_Clk(int) , turn_AClk(int) , IR_Sensors_Update(),
 * connect_to_wifi(), line_follow(), setup()
 * Global Variables: right_speed_offset, current_time, sp_ack, inp, wifi_connect, buzzer_loudness, irSensorPins, 
 * irPowerPins, motorPins, sensorValues, turn_delay, speed_decay 
 */

#include <WiFi.h>

#define MAX_SPEED 240

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
const int buzzer_loudness = 400;
// Pins Array
const byte irSensorPins[5] = { IR1, IR2, IR3, IR4, IR5 };
const byte irPowerPins[5] = { PWR_IR1, PWR_IR2, PWR_IR3, PWR_IR4, PWR_IR5 };
const byte motorPins[4] = { In1, In2, In3, In4 };

// Setting default values
bool sensorValues[5] = { 0, 0, 0, 0, 0 };

/*
* Function Name: motor_control
* Input:
    - LM_direction (bool): Direction of the Left Motor. 0 for backward, 1 for forward.
    - RM_direction (bool): Direction of the Right Motor. 0 for backward, 1 for forward.
    - speed (short): Speed of the motors. Ranges from 0 to 10.
* Output: None. This function does not return a value.
* Logic:
    - If the speed is 0, it stops both motors by setting all motor pins to 0 and setting the speed of both motors to 0.
    - If the speed is not 0, it sets the direction of the motors according to the LM_direction and RM_direction parameters and sets the speed of both motors to the given speed.
* Example Call: motor_control(1, 0, 150); // Sets the Left Motor to move forward and the Right Motor to move backward with a speed of 150 out of 255.
*/

void motor_control(bool LM_direction, bool RM_direction, short speed) {
  // Left Motor = In3,In4  --> 0
  // Right Motor = In1,In2 --> 1
  // dir = backward --> 0
  // dir = forward  --> 1
  // Check if the speed is 0
  if (speed == 0) {
    // Loop through all motor pins and set them to 0
    for (byte i = 0; i < 4; i++)
      digitalWrite(motorPins[i], 0);  // motorPins is an array that contains the pin numbers of the motors
    // Set the speed of both motors to 0
    analogWrite(Left_Speed, 0);   // Left_Speed is the pin number of the left motor speed control
    analogWrite(Right_Speed, 0);  // Right_Speed is the pin number of the right motor speed control
  } else {
    // Set the direction of the motors
    digitalWrite(motorPins[0], !RM_direction);  // motorPins[0] is the pin number of the right motor direction control
    digitalWrite(motorPins[1], RM_direction);   // motorPins[1] is the pin number of the right motor direction control
    digitalWrite(motorPins[2], !LM_direction);  // motorPins[2] is the pin number of the left motor direction control
    digitalWrite(motorPins[3], LM_direction);   // motorPins[3] is the pin number of the left motor direction control
    // Set the speed of both motors to the given speed
    analogWrite(Left_Speed, speed);   // Left_Speed is the pin number of the left motor speed control
    analogWrite(Right_Speed, speed);  // Right_Speed is the pin number of the right motor speed control
  }
}
/*
* Function Name: bot_move
* Input:
    - command (byte): The command to control the movement of the bot. The commands are:
        0 --> Move Backward
        1 --> Take Left
        2 --> Take Right
        3 --> Move Forward
        4 --> Stop
    - speed (byte): The speed of the bot.
* Output: None. This function does not return a value.
* Logic:
    - The function takes a command and speed as input.
    - It uses a switch case to execute a different motor control function based on the command.
    - For each case, it calls the motor_control function with different parameters based on the command.
    - If the command is not recognized, it stops the bot.
* Example Call: bot_move(1, 150); // Makes the bot take a left turn with a speed of 150.
*/
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
    case 0:  // Move Backward
      {
        motor_control(0, 0, speed);  // Calls the motor_control function to move the bot backward
        break;
      }
    case 1:  // Take Left
      {
        motor_control(0, 1, speed);  // Calls the motor_control function to make the bot take a left turn
        break;
      }
    case 2:  // Take Right
      {
        motor_control(1, 0, speed);  // Calls the motor_control function to make the bot take a right turn
        break;
      }
    case 3:  // Move Forward
      {
        motor_control(1, 1, speed);  // Calls the motor_control function to move the bot forward
        break;
      }
    default:  // Stop
      {
        motor_control(0, 0, 0);  // Calls the motor_control function to stop the bot
      }
  }
}

/* 
* Function Name: turn_Clk
* Input: 
    - in (int): An integer input to control the delay in the turn.
* Output: None. This function does not return a value.
* Logic: 
    - The function first stops the bot and then makes it turn clockwise at MAX_SPEED.
    - It then introduces a delay based on the input parameter 'in'.
    - After the delay, it enters a loop where it keeps updating the IR sensors and making the bot turn clockwise until a sensor detects a line.
    - Once the condition is met, it stops the bot.
* Example Call: turn_Clk(1); // Makes the bot turn clockwise.
*/

const int turn_delay = 330;  // Variable to hold the value of the delay when 'in' is not 1

void turn_Clk(int in) {
  bot_move(4, 0);          // Stop the bot
  bot_move(2, MAX_SPEED);  // Make the bot turn clockwise at MAX_SPEED

  // Introduce a delay based on the input parameter 'in'
  if (in == 1) {
    delay(250);
  } else
    delay(turn_delay);

  // Keep updating the IR sensors and making the bot turn clockwise until the condition is met
  IR_Sensors_Update();
  while (!(!sensorValues[0] && !sensorValues[1] && sensorValues[2])) {  // sensorValues is an array that holds the values of the IR sensors
    IR_Sensors_Update();                                                // Update the IR sensors
    bot_move(2, MAX_SPEED - 15);                                        // Make the bot turn clockwise
  }

  bot_move(4, 0);  // Stop the bot
}

/* Function Name: turn_AClk
* Input: in - an integer value that determines the delay time
* Output: No return value (void function)
* Logic: This function turns the bot in an anti-clockwise direction.
* The bot first stops, then moves at maximum speed. Depending on the input value, it delays for a certain amount of time.
* It then enters a loop where it updates the IR sensors and continues to move at a slightly reduced speed until a sensor finds the line. Finally, it stops the bot.
* Example Call: turn_AClk(1);
*/

void turn_AClk(int in) {
  // Stop the bot
  bot_move(4, 0);
  // Move the bot at maximum speed
  bot_move(1, MAX_SPEED - 10);

  // Delay based on the input value
  if (in == 1) {
    delay(50);
  } else if (in == 2)
    delay(200);
  else
    delay(turn_delay);  // Delay for a default time if input is not 1 or 2

  // Continue moving until a certain sensor condition is met
  IR_Sensors_Update();
  while (!(sensorValues[0] && !sensorValues[1] && !sensorValues[2])) {
    // Update the IR sensors
    IR_Sensors_Update();

    // Move the bot at a slightly reduced speed
    bot_move(1, MAX_SPEED - 15);
  }
  // Stop the bot
  bot_move(4, 0);
}


/* Function Name: IR_Sensors_Update
* Input: No input parameters
* Output: No return value (void function)
* Logic: This function updates the IR sensors of the bot. 
* It iterates over each sensor, powers it on, waits for a short delay, reads the sensor value, waits for another short delay, and then powers off the sensor.
* Example Call: IR_Sensors_Update();
*/

void IR_Sensors_Update() {
  // Iterate over each sensor
  for (short i = 0; i <= 4; i++) {
    // Power on the sensor
    digitalWrite(irPowerPins[i], 1);

    // Wait for a short delay
    delayMicroseconds(850);

    // Read the sensor value and store it in the sensorValues array
    // The '!!' operator is used to convert the digitalRead value to a boolean (0 or 1)
    sensorValues[4 - i] = !!digitalRead(irSensorPins[i]);

    // Wait for another short delay
    delayMicroseconds(850);

    // Power off the sensor
    digitalWrite(irPowerPins[i], 0);
  }
}


/* Function Name: jumpJunction
* Input: No input parameters
* Output: No return value (void function)
* Logic: This function is used to make the bot jump a junction. 
* It first moves the bot forward at maximum speed, then delays for a certain amount of time, and finally stops the bot.
* Example Call: jumpJunction();
*/

void jumpJunction() {
  // The delay time to move forward after encountering a Junction
  const int moveforward_delay = 200;

  // Move the bot forward at maximum speed
  bot_move(3, MAX_SPEED);

  // Delay for the defined time
  delay(moveforward_delay);

  // Stop the bot
  bot_move(4, 0);
}


/* Function Name: connect_to_wifi
* Input: No input parameters
* Output: No return value (void function)
* Logic: This function connects the bot to the WiFi. It starts a WiFi server on port 80 and begins the WiFi connection with the provided SSID and password.
* It waits until the WiFi connection is established, then prints the connection status and the local IP address. It starts the server and waits for a client to connect. 
* When a client connects, it reads a string from the client until it encounters a carriage return ('\r'), sends a confirmation message to the client, and then disconnects the client. 
* It then waits for 10 seconds before ending the function.
* Example Call: connect_to_wifi();
*/

void connect_to_wifi() {
  const char* ssid = "moto g52_3180";
  const char* password = "tesla8yvgrvq22";
  // const char* ssid = "Pranav’s iPhone";
  // const char* password = "Pranav12032003";

  // Start a WiFi server on port 80
  WiFiServer server(80);

  // Begin the WiFi connection with the provided SSID and password
  WiFi.begin(ssid, password);

  // Wait until the WiFi connection is established
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // Print the connection status and the local IP address
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();

  // Wait for a client to connect
  while (1) {
    WiFiClient client = server.available();
    if (client) {
      // Print the connection status
      Serial.println("Bot connected");

      // Turn on the green LED
      digitalWrite(greenLed, 1);

      // Read a string from the client until a carriage return ('\r')
      inp = client.readStringUntil('\r');

      // Print the received string
      Serial.println("Received: " + inp);

      // Send a confirmation message to the client
      client.println("Instructions Recieved");

      // Turn off the green LED
      digitalWrite(greenLed, 0);

      // Disconnect the client
      client.stop();

      // Print the disconnection status
      Serial.println("Bot disconnected");

      // Break the loop
      break;
    }
  }

  // Wait for 10 seconds
  delay(5000);
}

const int U_Turn_delay = 1330;

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
    digitalWrite(motorPins[i], LOW);
  }

  IR_Sensors_Update();

  if (wifi_connect == 0)
    connect_to_wifi();
  else
    // inp = "l0r1l0a0l0f0l0r0Czu1l0r0l0a0l0c0El0f0l0f0l0f0l0a0Azf0l0a0l0r0Bzu1l0a0l0r0l0a2ss";
    inp = "l0r1l0a0l0f0l0r0Czu1l0f0l0r0l0f0El0f0l0f0l0f0l0a0Azf0l0a0l0r0Bzu1l0a0l0r0l0a2ss";
  //inp = "u1";
  // int start_time = millis();
  // while (millis() - start_time < 3000) {
  //   bot_move(3, MAX_SPEED);
  //   delay(10);
  // }
  // bot_move(4, 0);
  // delay(10000);
  for (int i = 0; i < inp.length() - 1; i++) {
    char instr1 = inp.charAt(i);
    char instr2 = inp.charAt(i + 1);
    Serial.print("instr1: ");
    Serial.println(instr1);
    Serial.print("instr2: ");
    Serial.println(instr2);
    int ack = 0;
    if (instr1 == 'l' && instr2 == '0') {
      Serial.println("line follow");
      while (ack == 0)
        ack = line_follow();
      // digitalWrite(redLed, 1);
      bot_move(4, 0);
      delay(250);
      // digitalWrite(redLed, 0);
    } else if (instr1 == 'r' && (instr2 == '0' || instr2 == '1')) {
      Serial.println("Clk");
      jumpJunction();
      turn_Clk(int(instr2) - 48);
    } else if (instr1 == 'a' && (instr2 == '0' || instr2 == '1' || instr2 == '2')) {
      Serial.println("AClk");
      jumpJunction();
      turn_AClk(int(instr2) - 48);
    } else if (instr1 == 'f' && instr2 == '0') {
      Serial.println("Forward");
      jumpJunction();
    } else if (instr1 == 'z') {
      analogWrite(buzzerPin, buzzer_loudness);
      delay(1100);
      analogWrite(buzzerPin, 1024);
      Serial.println("Buzzer");
    } else if (instr1 == 'E') {
      jumpJunction();
      int t = millis();
      while (millis() - t < 3000) {
        ack = line_follow();
      }
      bot_move(4, 0);
      analogWrite(buzzerPin, buzzer_loudness);
      digitalWrite(redLed, 1);
      delay(1100);
      analogWrite(buzzerPin, 1024);
      digitalWrite(redLed, 0);

      // bot_move(1, MAX_SPEED);
      // delay(U_Turn_delay - 200);
      // bot_move(4, 0);
      bot_move(1, MAX_SPEED - 20);
      delay(U_Turn_delay - 10);
      // IR_Sensors_Update();
      // while (!(!sensorValues[0] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
      //   bot_move(1, MAX_SPEED - 40);
      //   IR_Sensors_Update();
      // }
      bot_move(4, 0);
      delay(100);
      ack = 0;
      while (!ack) {
        ack = line_follow();
      }
      bot_move(4, 0);
      delay(300);
    }

    else if (instr1 == 'u' && instr2 == '0') {
      Serial.println("U-Turn Clockwise");
      bot_move(2, MAX_SPEED);
      delay(U_Turn_delay);
      bot_move(4, 0);
      delay(250);
    } else if (instr1 == 'u' && instr2 == '1') {
      Serial.println("U-Turn Anticlockwise");
      bot_move(1, MAX_SPEED - 20);
      delay(U_Turn_delay);
      bot_move(4, 0);
      delay(250);

      // else if (instr1 == 'u' && instr2 == '0') {
      //   Serial.println("U-Turn Clockwise");
      //   bot_move(2, MAX_SPEED);
      //   delay(U_Turn_delay);
      //   IR_Sensors_Update();
      //   while (!(!sensorValues[0] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
      //     bot_move(2, MAX_SPEED - 40);
      //     IR_Sensors_Update();
      //   }
      //   bot_move(4, 0);
      // } else if (instr1 == 'u' && instr2 == '1') {
      //   Serial.println("U-Turn Anticlockwise");
      //   bot_move(1, MAX_SPEED);
      //   delay(U_Turn_delay - 200);
      //   IR_Sensors_Update();
      //   while (!(!sensorValues[0] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
      //     bot_move(1, MAX_SPEED - 40);
      //     IR_Sensors_Update();
      //   }
      //   bot_move(4, 0);
      // }
    } else if (instr1 == 'A' || instr1 == 'D' || instr1 == 'b' || instr1 == 'c') {
      int cnt = 0;
      while (cnt < 15) {
        if (!sensorValues[1])
          cnt++;
        else
          cnt = 0;
        IR_Sensors_Update();
        ack = line_follow();
      }
      while (!sensorValues[4]) {
        bot_move(2, MAX_SPEED - 30);
        IR_Sensors_Update();
      }
      if (instr1 == 'c') {
        int init_time = millis();
        while (1) {
          int acC = line_follow();
          if (millis() - init_time > 800)
            break;
        }
      }
      bot_move(4, 0);
    } else if (instr1 == 'B' || instr1 == 'C' || instr1 == 'd' || instr1 == 'a') {
      int cnt = 0;
      while (cnt < 30) {
        IR_Sensors_Update();
        ack = line_follow();
        if (!sensorValues[1])
          cnt++;
        else
          cnt = 0;
      }
      bot_move(4, 0);
      delay(100);
      if (instr1 == 'B') {
        bot_move(1, MAX_SPEED);
        delay(100);
        bot_move(4, 0);
      }
      cnt = 0;
      if (instr1 == 'C')
        cnt = 3;
      while (cnt < 5) {
        IR_Sensors_Update();
        ack = line_follow();
        if (!sensorValues[0] && sensorValues[1] && !sensorValues[2])
          cnt++;
        else
          cnt = 0;
      }
      ack = 0;
      if (instr1 == 'C') {
        bot_move(0, MAX_SPEED - 30);
        delay(400);
        bot_move(4, 0);
      }
      bot_move(4, 0);

    } else if (instr1 == 's') {
      int cnt = 0;
      while (cnt < 10) {  // A to SP
        IR_Sensors_Update();
        sp_ack = line_follow();
        if (sensorValues[0] == 0 && sensorValues[1] == 0 && sensorValues[2] == 0 && sensorValues[3] == 0 && sensorValues[4] == 0)
          cnt++;
      }
      bot_move(3, MAX_SPEED);
      delay(250);
      bot_move(4, 0);
      // digitalWrite(redLed, 1);
      analogWrite(buzzerPin, buzzer_loudness);
      delay(5000);
      analogWrite(buzzerPin, 1024);
      // digitalWrite(redLed, 0);
    }
  }
}



int speed_decay = 20;  // Speed decay value for the bot

/* Function Name: line_follow
* Input: None
* Output: Returns 0 for line following operation and 1 when a junction is detected.
* Logic: This function is used for line following in a bot. It reads the sensor values and based on the sensor values, it decides the direction of the bot.
* If the bot is on the line, it moves forward. If the bot deviates from the line, it adjusts its direction to left or right. If the bot detects a junction, it stops.
* Example Call: line_follow();
*/
int line_follow() {
  // Update the IR sensor values
  IR_Sensors_Update();

  // If the bot is on the line, move forward
  if ((!sensorValues[0] && sensorValues[1] && !sensorValues[2]) ||  //
      (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && sensorValues[4])) {
    Serial.println("Moving Forward");
    speed_decay = 20;
    bot_move(3, MAX_SPEED);
    return 0;
  }
  // If the bot deviates to the right, adjust its direction to the right
  else if ((!sensorValues[0] && !sensorValues[1] && sensorValues[2]) ||  //
           (!sensorValues[0] && sensorValues[1] && sensorValues[2]) ||   //
           (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && !sensorValues[4])) {
    Serial.println("Moving Right");
    bot_move(2, MAX_SPEED - speed_decay);
    speed_decay = (speed_decay <= 80) ? speed_decay + 10 : 80;
    return 0;
  }
  // If the bot deviates to the left, adjust its direction to the left
  else if ((sensorValues[0] && !sensorValues[1] && !sensorValues[2]) ||                                          //
           (sensorValues[0] && sensorValues[1] && !sensorValues[2]) ||                                           //
           (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && !sensorValues[4]) ||  //
           (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && sensorValues[4])) {
    Serial.println("Moving Left");
    bot_move(1, MAX_SPEED - speed_decay);
    speed_decay = (speed_decay <= 80) ? speed_decay + 10 : 80;
    return 0;
  }
  // If the bot detects a junction, stop
  else if ((!sensorValues[0] && sensorValues[1] && sensorValues[2] && sensorValues[3] && sensorValues[4]) ||  //
           (sensorValues[0] && sensorValues[1] && sensorValues[2])) {
    bot_move(4, 0);
    Serial.println("Stop");
    Serial.println("Junction Detected");
    return (1);
  }
  // else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && !sensorValues[3] && !sensorValues[4]) {
  //   bot_move(2, MAX_SPEED - speed_decay);
  //   speed_decay = (speed_decay <= 80) ? speed_decay + 10 : speed_decay;
  //   // delay(10);
  //   return 0;
  // } else if (!sensorValues[0] && !sensorValues[1] && !sensorValues[2] && sensorValues[3] && sensorValues[4]) {
  //   bot_move(1, MAX_SPEED - speed_decay);
  //   speed_decay = (speed_decay <= 80) ? speed_decay + 10 : speed_decay;
  //   // delay(10);
  //   return 0;
  // }

  // If none of the above conditions are met, stop the bot
  else {
    bot_move(4, 0);
    return 0;
  }
}

void loop() {
}