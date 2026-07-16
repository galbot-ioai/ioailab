#ifndef _SYSTEM_H
#define _SYSTEM_H

#include "Protocol.h"

class System : public Moudle{
public:
    System(std::string busName){
        mountBus(busName, sysFun);
    }
    
    ~System(){};

    //产品 标识
    typedef struct{
        uint8_t statusCode;		//状态代号
        uint8_t res;			//保留
        uint16_t projectCode;	//项目编号
        uint8_t groupCode;		//组件代号
        uint8_t programmeCode;	//方案代号
        uint8_t customerCode;	//客户代号
        uint8_t data[5];		//流水码 或者 版本号(年月日时：20 24 04 17 19)
    }productIdentificationType;

    //组件
    enum{
        GROUP_ARM 			= 0x01,		//机械臂
        GROUP_HAND 			= 0x02,		//灵巧手
        GROUP_LOWER_BODY 	= 0x03,		//下半身
        GROUP_BODY 			= 0x04,		//躯干
        GROUP_HEAD 			= 0x05,		//头部
    };

    std::string getVersion();
    std::string getSN();
    typedef struct{
        uint16_t totalVoltage;          //总电压    0.01V
        int16_t totalCurrent;           //总电流    0.01A

        uint16_t averageVoltage;        //平均电压  0.001V
        uint16_t voltageDiff;           //压差      0.001V
        int16_t averageTempture;        //平均温度  0.1°
        int16_t temptureDiff;           //温差      0.1°

        uint16_t capacity;              //容量      0.01AH
        uint16_t loopCnt;               //循环次数
        uint16_t sysStatus;             //系统状态（0空闲、1充电、2放电）
        int16_t mosTempture;            //mos温度   0.1°

        uint8_t chargeStatus;           //充电状态
        uint8_t loadStatus;             //负载状态

        uint8_t warnStatus1;            //警告(每1bit代表一个警告)（单体过压、单体欠压、总压过压、总压欠压、放电过流、充电过流、放电高温、充电高温）
        uint8_t warnStatus2;            //警告(每1bit代表一个警告)（放电低温、充电低温、SOC过高、SOC过低、压差大、温差大、MOS温度过高、环境温度过高）
        uint8_t protectWord1;           //保护(每1bit代表一个保护)（单体过压、单体欠压、总压过压、总压欠压、放电过流、充电过流、放电高温、充电高温）
        uint8_t protectWord2;           //保护(每1bit代表一个保护)（放电低温、充电低温、SOC过高、SOC过低、压差大、温差大、MOS温度过高、环境温度过高）
    }BmsType;

    BmsType bms;

    void viewBmsInfo(void){
        RLOGI("totalVoltage:%.2f, totalCurrent:%.2f, averageVoltage:%.3f, voltageDiff:%.3f, averageTempture:%.2f, temptureDiff:%.2f \r\n",
        (float)bms.totalVoltage * 0.01f,
        (float)bms.totalCurrent * 0.01f,
        (float)bms.averageVoltage * 0.001f,
        (float)bms.voltageDiff * 0.001f,
        (float)bms.averageTempture * 0.1f,
        (float)bms.temptureDiff * 0.1f     );
        
        RLOGI("capacity:%.2f, loopCnt:%d, sysStatus:%d, mosTempture:%.2f, chargeStatus:%d, loadStatus:%d \r\n",
        (float)bms.capacity * 0.01f,
        bms.loopCnt,
        bms.sysStatus,
        (float)bms.mosTempture * 0.1f,
        bms.chargeStatus,
        bms.loadStatus);
        RLOGI("warnStatus: 0x%02X 0x%02X ,protectWord 0x%02X 0x%02X \r\n", bms.warnStatus1, bms.warnStatus2, bms.protectWord1, bms.protectWord2);
    }
    //心跳类型
    enum{
        HEART_NONE  = 0,
        HEART_APP 	= 1,
        HEART_BOOT 	= 2,
        HEART_NUM 	= 3,
    };
    const char* heartStr[HEART_NUM] = {       
        {"none"},
        {"app"},
        {"boot"},
    };
    uint8_t beartType = 0;

    //打开雷达电源
    bool setRadarPwr(bool isPower);
private:

    #define SYSTEM_CODE			0x20
    const devFunMapType sysFun = {
        {
            SYSTEM_CODE,{
                {"heartbeat",           0x01},
                {"version",             0x02},
                {"sncode",              0x03},
                {"bms",                 0x04},
                {"radar",               0x05},
            }
        },
    };
};

#endif 


