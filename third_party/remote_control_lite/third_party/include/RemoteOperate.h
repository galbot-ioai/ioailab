#ifndef _REMOTEOPERATE_H
#define _REMOTEOPERATE_H

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <mutex>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <map>
#include "Protocol.h"

class RemoteOperate : public Moudle{
public:
    RemoteOperate(std::string busName){
        mountBus(busName, RemoteOperateFun);
    }

    ~RemoteOperate(){};
    bool setReport(bool isEnable);

    enum{
        M_1,
        M_2,
        M_3,
        M_4,
        M_5,
        M_6,
        M_7,
        M_NUM,
    };

    enum{
        NONE_MODE,              //无模式
        OPERATE_MODE,           //遥操模式
    };


    typedef struct{
        float curPos;
        float curSpeed;
        float curTorque;
        uint32_t error;
        // uint32_t custom;
    }motorInfo;

    typedef struct{
        motorInfo m[M_NUM];
    }armInfo;

    armInfo armL, armR;

    typedef struct{
        uint16_t y;
        uint16_t x;
    }Pole;

    Pole poleL, poleR;
    Pole trigerL, trigerR;

    enum{
        KEY_DOWN    = 1,        //按下
        KEY_UP      = 2,        //抬起
        KEY_LONG    = 3,        //长按
        KEY_CLICK   = 4,        //点击
    };

    const char* downStatus(uint8_t status) {
        if(status == KEY_DOWN) {
            return "down";
        } else if(status == KEY_UP) {
            return "up";
        } else if(status == KEY_LONG) {
            return "long";
        } else if(status == KEY_CLICK) {
            return "click";            
        } else {
            return "null";
        }
    }
    uint8_t mode = NONE_MODE;

private:

    #define RemoteOperate_DEV_CODE    0xB0
    const devFunMapType RemoteOperateFun = {
        {
            RemoteOperate_DEV_CODE,{
                {"reportEnable",        0x01},
                {"armL",                0x02},
                {"armR",                0x03},

                {"poleL",               0x04},
                {"poleR",               0x05},  

                {"key1L",               0x06},  
                {"key2L",               0x07},  
                {"key3L",               0x08},
                {"key4L",               0x09},  
                {"key5L",               0x0A}, 
                {"key6L",               0x0B}, 
                {"resL",                0x0C},  
                {"key7L",               0x0D}, 
                {"key8L",               0x0E}, 
                {"key9L",               0x0F},  

                {"key1R",               0x10},  
                {"key2R",               0x11},  
                {"key3R",               0x12},  
                {"key4R",               0x13},  
                {"key5R",               0x14},  
                {"key6R",               0x15}, 
                {"resR",                0x16},  
                {"key7R",               0x17},  
                {"key8R",               0x18},  
                {"key9R",               0x19},  

                {"trigerL",               0x30},
                {"trigerR",               0x31},  

            }
        },
    };

};

#endif 

