#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERVOMIN "Your value" 
#define SERVOMAX "Your value" 
#define SERVO_FREQ 50 

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); 

bool state0[6] = {false, false, false, false, false, false};
bool state1[6] = {false, false, false, false, false, false};

bool change = false;

char sData;
String state;

int hall[5][3] = {{26, 0, 2200}, {27, 0, 2400}, {14, 0, 2300}, {25, 0, 2200}, {12, 0, 2300}};

int wrist = 0;
int thumb = 4;
int index = 1;
int middle = 2;
int ring = 3;
int pinky = 5;

int degToPwm(int degree) {
 return map(degree, 0, 320, SERVOMIN, SERVOMAX);
}

int deg = degToPwm(75);
int deg1 = degToPwm(95);
int deg2 = degToPwm(85);
int startDeg = degToPwm(180);

TaskHandle_t recieveData;

void recieveDataCode(void * parameter) {
 for(;;) {
  while(Serial.available()) {
   sData = Serial.read();
   if(sData == '\n') {
    for(int i = 0; i < 6; i++) {
     state0[i] = state.substring(i, i+1).toInt();
    }
    state = "";
    change = true;
    break;
   } else {
    state += sData;
   }
  }
  delay(10);
 }
}

void moveFinger(int fingerId, bool flex, int iteration) {
 if(fingerId != ring && fingerId != pinky) {
  if(flex) {
   if(fingerId == thumb) {
    float fPwm = SERVOMIN + (float(103)*float(iteration))/float(130);
    int iPwm = round(fPwm);
    pwm.setPWM(fingerId, 0, iPwm);
   } else {
    pwm.setPWM(fingerId, 0, SERVOMIN + iteration); 
   }
  } else {
   if(fingerId == thumb) {
    float fPwm = deg - (float(103)*float(iteration))/float(130);
    int iPwm = round(fPwm);
    pwm.setPWM(fingerId, 0, iPwm);
   } else {
    pwm.setPWM(fingerId, 0, deg1 - iteration); 
   }
  }
 } else /*if(fingerId == ring || fingerId == pinky)*/ {
  if(flex) {
   pwm.setPWM(fingerId, 0, startDeg - iteration);
  } else {
   pwm.setPWM(fingerId, 0, deg2 + iteration);
  }
 }
}

void setup() {
 Serial.begin(9600);
   
 for(int i = 0; i < 5; i++) {
  pinMode(hall[i][0], INPUT);
 }
  
 pwm.begin();
 pwm.setOscillatorFrequency(27000000);
 pwm.setPWMFreq(SERVO_FREQ);

 delay(10);
  
 xTaskCreatePinnedToCore(
  recieveDataCode,
  "recieveData",
  10000,
  NULL,
  0, 
  &recieveData,
  0);
 delay(500);
}

void loop() {
 if(change) {
  for(int i = 5; i < 135; i += 5) {
   for(int k = 0; k < 5; k++) {
    hall[k][1] = analogRead(hall[k][0]);
    if(hall[k][1] > hall[k][2]) {
     state1[k+1] = state0[k+1];
    }
   }
   for(int j = 0; j < 6; j++) {
    if(state0[j] != state1[j]) {
     moveFinger(j, state0[j], i);
    }
   } 
   delay(17);
  }

  for(int i = 0; i < 6; i++) {
   state1[i] = state0[i];
  }
 }
  
 delay(100);}