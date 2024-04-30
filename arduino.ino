#include "Wire.h"       
#include "I2Cdev.h"     
#include "MPU6050.h"    
#include "SD.h"
#define SD_ChipSelectPin 10
#include "TMRpcm.h"
#include "SPI.h"
TMRpcm tmrpcm;
MPU6050 mpu;
int16_t ax, ay, az;
int16_t gx, gy, gz;

const int buttonPin = 2;
int buttonState = 0;
int Flag = 0;

struct MyData {
  byte X;
  byte Y;
  byte Z;
};

MyData data;
MyData data2;

void setup()
{
  tmrpcm.speakerPin=9;
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();
  pinMode (buttonPin, INPUT);
  if(!SD.begin(SD_ChipSelectPin))
  {
    Serial.println("SD fail");
    return;
  }
  tmrpcm.setVolume(6);
}

void loop()
{
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  /* When Flag = 0 This Mean The Button Not Pressed */
  if (Flag == 0)
  {
      buttonState = digitalRead(buttonPin);
      if(buttonState == LOW) 
      {
        /* Start Value */
          Flag = 1;
          data.X = map(ax, -17000, 17000, 0, 255 ); // X axis data
          data.Y = map(ay, -17000, 17000, 0, 255);  // Y axis data
          delay(1000);
          Serial.print("Axis X = ");
          Serial.print(data.X);
          Serial.print("  ");
          Serial.print("Axis Y = ");
          Serial.print(data.Y);
          Serial.print("  ");
      }
  }
  /* When Flag = 1 This Mean The Button Pressed */
  else
  {
    /* Reading Values */
    data2.X = map(ax, -17000, 17000, 0, 255 ); // X axis data
    data2.Y = map(ay, -17000, 17000, 0, 255);  // Y axis data
    delay(1000);
    /* check Axis Change 
      if Condition is True This Mean X Axis Change */
    if(abs(data2.X - data.X) > abs(data2.Y - data.Y) && abs(data2.X - data.X) > 60)
    {
      Serial.println("X Axis Motion");
      /* Check Direction of X axis is Right or Left */
      if((data2.X - data.X) > 0 )
      {
        Serial.println("YES");
        tmrpcm.play("yes.wav");
        delay(1000);
      }
      else
      {  
        Serial.println("NO");
        tmrpcm.play("no.wav");
        delay(1000);
      }
    }
    /* This Mean Y Axis Change */
    else 
    {
      /* Check That if happen Change or not */
      if(abs(data2.Y - data.Y) > 60)
      {
        Serial.println("Y Axis Motion");
        /* Check Direction of Y axis is Right or Left */
        if((data2.Y - data.Y) > 0 )
        {
          Serial.println("HELLO");
        tmrpcm.play("hello.wav");
        delay(1000);
          
        }
        else
        {
          Serial.println("-ve Y Axis Motion");
        }
      }
    }
  }
}
