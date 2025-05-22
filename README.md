# Team GG_1240(Pranav Lakshmanan , Srikar Bhardwaj) for E-Yantra Robotics competition(2023-2024), GeoGuide Theme . 

## About EYRc 

E-Yantra Robotics competition is a yearly **8-month Long** Task based robotics competition conducted by IIT Bombay sponosred by the Ministry of Education through NMEICT. This is one of the longest and largest running robotics competitions in the country where over 10,000 college(both undergraduate and masters) teams (each team comprising of 1-4 members) participate. The competition is split into 4-5 themes based on various robotic tech stacks and the partcipants are given timed tasks that are graded and completed within a set timeframe . It adopts a learning while competitng model where the initial tasks are meant to teach the participants about the underlying tech stack while the later stages aim at testing them. 

The best performing teams are given an chance to interview for the highly pretigious E-Yantra summer research internship . This is an 8-week full funded research scholarship at IIT Bombay where the interns build or help build robots along with their guides at various departments and Startups at IIT Bombay. The guides act as mentors and often the work done during these 8-weeks leads to research outputs such as research publications or for building the basis for future hardware based themes of EYRc or spin-offs as startups. 



## About GeoGuide theme 

TechStack - QGIS , CNNs(Transfer learning) , ESP32 & Socket Programming , Fusion360(Mechanical Design) , OpenCV , Circuit & PCB design(KiCAD) , Path Planning 

The GeoGuide Theme requires the participants to do the following - 

1) To design and build a 2-wheeled differential drive robot called Vanguard with BO Motors and an ESP32 acting as the fundamental building blocks. The robot additionally is required to have 5 IR-sensors that help the robot naviage the arena using line-following. The placement of these sensors is left to the descretion of the teams. The competition also gives liberty to participants to design their own PCBs but restricts them to use only competition sanctioned components and required passives. We came up with our own PCB design to house the entire ESP32 assembly including the buzzer , battery connections , the motor driver , the Start/Stop LEDs and the 5 IR sensors mounted on the front of the bot. We also designed our own bot using Fusion360 and 3D printed it using PLA. Addtionally we also came up with an circuit design to have over-current and over-voltage protections for the ESP32 while also incoperating safety features for the circuit to not short. We also had to come up with a way to supply power to the 5 mounted IR sensor separatelty during times of testing , while making sure the motors are not running.
   
2) This Robot is then supposed to navigate autonmously through a marked 9ft*9ft arena. The naviagtion through the arena is supposed to be continously tracked in real-time and it also involves the robot to stop at various different locations in the arena . This entire movement of the bot while it executes its tasks is a timed affair and it carries points. There are certain images placed on the arena that are of around 160 * 160 px resolution(as seen by our webcam setup mounted at the ceiling). These images depict various natural disasters such as humanitarian crisis , Military crisis , Fire in buildings etc etc. Hence each of these images has a certain priority order. The robot must start from its start position navigate to each of these images according their priority order while taking the shortest path , completely autonomously using line following aided by the signals of the IR sensors as the primary perception mechanism. Each time it reaches an important disaster it has stay there for 1 second and beep its buzzer before continuing. All of thie meant that the robot , as in our case , has to have the ability to take U-turns using a differential drive while taking the shortest path.

4) Both the robot and the arena are marked with Aruco markers that are then continously monitored by a overhead lenovo webcam. The use of the aruco markers are completely left to the descrtion of the teams. In our case these unique aruco IDs acted as location holders and helped us identify before the timed run started where are all the distasters and hence figure out the shortest path. The shortest path finding is implemented here by a weighted Dijkstras algorithm that takes into account certain paths that are not feasible for the robot to physically move in (such as path involving too many U-turns or paths where the line following where the line following was not accurate). (Cam(for windows).py file)
   
5) Addtionally since the Webcam was mounted on a ceiling with just a tube light illuminating the entire arena , this lead to the OpenCV algorithm mis-identifying certain aruco markers due to lens distortions , we had to come up with our own camera calibration algorithm to get rid of these pin and barrel distortions.

6) As mentioned earlier the entire movement of the bot is tracked in real-time. This is aided and required by the competition to be done in Geographics Information system software called QGIS. This meant that the entire arena had to be Georeferenced using Ground control control points. This meant that each pixel on the arena has a corresponding lat-long cordinate.

7) Additionally to decide which image has a priority for movement , first the images needed to be identified using OpenCV and the classified using a Convolutional neural network. We had only gotten about 100 images for each class(such as humanitarian aid , military , fire etc etc ) by the EYRc team. Each of these images were only 90 * 90 px , but the webcam feed gave us 160 * 160 px. Hence we had to perform large amounts of image augmentation and data-set preparation before feeding it for training . We also had to collect our own datasets for the model to train on due to the changing lighting conditions where the arena was laid. We used ResNet50 V2 as our base model for transfer learning.
The model Training and classification is one of the most important aspects of this theme.

9) On the hardware side we had to also complete code out the ESP32 to-do line following using its 5 IR sensors. This required constant calibration of the IR Sensors for each run and for executing turns we had to come up with the momentum decay function where the turning does not lead to the bot moving away from the marked roads on the arena , which would lead to penalty of points. We also had to make the bot execute complex U-turns in-order to save time , while communicating to the bot using socket-programming via the WiFi-module of the ESP32. The IR sensors combined with the Power hungry ESP32 led to constant voltage drops for the motors , hence we had to come up with a feedback mechanism that executes turns even when the battery voltage may not be ideal.

10) We were able to achieve AIR 9 and were selected for the summer internship at IITB. The final task was a 48 hour hackathon where we were supplied with a never before seen dataset for the images to be placed on the arena to check our entire algo stack. We had the normal task as well a bonus task(optional , but we still did it;)) that had more reward points. Both tasks had separate datasets and needed to be finished in 48 hours.

Our final task submissions with the complete code are found in the Task-6 folder , it contains 2 files where one where the main-processing is done (named Task_6.py) and the other is the ESP32 bot code(Follow_path.ino). 

Our CNN training code will be found in the Task 4a folder. 

--- 

## Youtube Video of Our final submission - 

Original Task -  https://youtu.be/YHy7Ekm6JQc
Bonus Task - https://youtu.be/sA-pHlsgXco

