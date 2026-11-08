#include <Servo.h>

Servo myServo1;
Servo myServo2;
int angle1;
int angle2;

//fali paljenje lasera

void setup() {
  myServo1.attach(3);
  myServo2.attach(5);
  //pocetni uglovi
  myServo1.write(90);
  myServo2.write(90);
}

void loop() {
  angle1 = ...;
  angle2 = ...;
  myServo1.write(angle1);
  myServo2.write(angle2);
  delay(15);
}
