//
// Created by fbc on 2024/1/19.
//

#ifndef PROCOTOL_PROTOCOL_H
#define PROCOTOL_PROTOCOL_H
#include "log.h"

#include <thread>
#include <cstdlib>
#include <map>
#include <queue>
#include <string>
#include "SerialPort.h"
#include <signal.h>
#include <time.h>
#include <sys/time.h> 
#include <iomanip>
#include <chrono>
#include <ctime>
#include <functional>
#include <jsoncpp/json/json.h>

#include "log.h"


using namespace std::chrono;
using stdTimePoint = std::chrono::high_resolution_clock::time_point;

#define HEADH 0xDE
#define HEADL 0xED
#define TAILH 0xEA
#define TAILL 0xAE

#define  COMM_SEQ_POS       2
#define  COMM_TICK_POS      3
#define  COMM_LEN_POS       7
#define  COMM_PAYLOAD_START 8
#define  COMM_DEVTYPE_POS   8

#define  COMM_CRC_HIGH_POS  8
#define  COMM_CRC_LOW_POS   9

#define  COMM_END_HIGH_POS  10
#define  COMM_END_LOW_POS   11

#define  COMM_CTRL_LEN      12


//报文类型
typedef struct{
    uint8_t fIndex;             //帧索引
    uint8_t dev;
    uint8_t cmd;
    uint8_t fun;
    u8Vector data;              //对应命令字的内容
}msgType;

//应答报文结构
typedef struct{
    bool isAnswer;
    bool isReport;
    u8Vector answer;            //应答数据
    u8Vector report;            //上报数据
}answerMsgType;         

enum{
    SUB_PROT_DEV_POS,
    SUB_PROT_CMD_POS,
    SUB_PROT_FUN_POS,
    SUB_PROT_DATA_POS,
};

//命令属性(暂时没用到)
const std::map<std::string, uint8_t> cmd = {
        {"read",            0x20},
        {"write",           0x21},
        {"answer",          0x22},
        {"autoReport",      0x23}
};

//注册类
class Register{
public:
    using MemberFunctionPtr = std::function<void(uint8_t, uint8_t, std::vector<uint8_t>)>;    //函数指针

    std::map<uint8_t, MemberFunctionPtr> functionMap;

    // 注册函数，将其他类的成员函数注册到 map 中
    template<typename T, typename Func>
    void registerFunction(const uint8_t& dev, T& obj, Func func) {
        functionMap[dev] = std::bind(func, std::ref(obj), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    }

    // 调用函数，根据给定的名称调用注册的函数
    void callMapFunction(const uint8_t& dev, uint8_t cmd, uint8_t fun, std::vector<uint8_t> buf) {
        auto it = functionMap.find(dev);
        if (it != functionMap.end() && (it->second)) {
            it->second(cmd, fun, buf);
        } else {
            RLOGW("Register dev 0x%X map fun is not registered!!! \r\n", dev);
        }
    }

    //状态切换的函数指针
    using CallBackStatusFunctionPtr = std::function<void(std::string, uint8_t)>;    
    std::map<uint8_t, CallBackStatusFunctionPtr> openStatusFunctionMap;

    template<typename T, typename Func>
    void registerOpenStatusFunction(const uint8_t& dev, T& obj, Func func) {
        openStatusFunctionMap[dev] = std::bind(func, std::ref(obj), std::placeholders::_1, std::placeholders::_2);
    }
    //回调所有总线上的设备，并通知串口已打开
    void callMapOpenStatusFunction(std::string name) {
        if(openStatusFunctionMap.size() > 0){
            for(auto it = openStatusFunctionMap.begin(); it != openStatusFunctionMap.end(); it++){
                uint8_t dev = it->first;        //设备码
                if(it->second){ 
                    // RLOGW("dev 0x%X, open success callback()\r\n", dev);
                    it->second(name, dev);   
                }
            }            
        }
    }


    std::map<uint8_t, CallBackStatusFunctionPtr> errorStatusFunctionMap;
    template<typename T, typename Func>
    void registerErrorStatusFunction(const uint8_t& dev, T& obj, Func func) {
        errorStatusFunctionMap[dev] = std::bind(func, std::ref(obj), std::placeholders::_1, std::placeholders::_2);
    }
    void callMapErrorStatusFunction(std::string name) {
        if(errorStatusFunctionMap.size() > 0){
            for(auto it = errorStatusFunctionMap.begin(); it != errorStatusFunctionMap.end(); it++){
                uint8_t dev = it->first;        //设备码
                if(it->second){ 
                    // RLOGW("dev 0x%X, error occureed callback()\r\n", dev);
                    it->second(name, dev);   
                }
            }            
        }
    }


    using FunctionJsonPtr = std::function<void(Json::Value root)>;    
    FunctionJsonPtr funJsonPtr;
    // 注册函数，将其他类的成员函数注册到 map 中
    template<typename T, typename Func>
    void registerJsonFun(T& obj, Func func) {
        funJsonPtr = std::bind(func, std::ref(obj), std::placeholders::_1);
    }

    // 调用函数，根据给定的名称调用注册的函数
    void callJsonFunction(Json::Value root) {
        if(funJsonPtr != NULL){
            funJsonPtr(root);
        }
    }    

};


class Protocol : public Register{
public:
    Protocol(std::string name);
    ~Protocol();

    void registerFun(uint8_t dev, uint8_t fun);             //注册
    bool write(uint8_t dev, uint8_t cmd, uint8_t fun, u8Vector buf, uint16_t timeOut, bool isBlock);

    bool write(uint8_t dev, uint8_t fun, u8Vector buf, bool isBlock);                                    //写 重发次数为 1，超时 30ms
    bool write(uint8_t dev, uint8_t fun, u8Vector buf, uint16_t timeOut, bool isBlock);                   //写 重发次数为 1, 超时可以设置
    bool write(uint8_t dev, uint8_t fun, u8Vector buf, uint16_t timeOut, uint8_t maxCnt, bool isBlock);   //写
    
    u8Vector read(uint8_t dev, uint8_t fun, u8Vector buf, uint16_t timeOut, bool isBlock);
    u8Vector read(uint8_t dev, uint8_t fun, u8Vector buf, uint16_t timeOut, uint8_t maxCnt, bool isBlock);
    
    u8Vector readMap(uint8_t dev, uint8_t fun){                 //读数据
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        return devMap[dev].at(fun)->answer;
    }

    void clearMap(uint8_t dev, uint8_t fun){
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);        
        devMap[dev].at(fun)->answer.clear();
    }
    
    int revFrameCnt = 0;                                //接收总帧数、帧率
    int sendFrameCnt = 0;                               //发送总帧数、帧率  
    int frameFpsError = 0;

    bool open();  

    void reOpen(void){
        setStatus(OPEN_STATUS);
    }

    bool isWork(bool isBlock, int timeOut = 4000){
        std::unique_lock<std::mutex> lock(isWorkMutex);
        if(!isBlock) return(this->status == WORK_STATUS);   //非阻塞
        int cnt = 0;
        do{
            usleep(1*1000);
            if((cnt++) > timeOut){
                RLOGE("check is work timeout, time max is %d\r\n", timeOut);
                return false;
            }
        } while (this->status != WORK_STATUS);
        return true;
    }
    
    //超时函数封装
    bool isTimeOut(stdTimePoint &point, int period) {
        stdTimePoint curTime = high_resolution_clock::now();
        milliseconds deltaTime = duration_cast<milliseconds>(curTime - point);
        if (deltaTime.count() > period) {
            point = high_resolution_clock::now();
            return true;
        }
        return false;
    }

    using floatVector = std::vector<float>;
    using u32Vector = std::vector<uint32_t>;

    template<typename T>
    std::vector<uint8_t> convertToU8Ary(const std::vector<T>& inputVector) {
        std::vector<uint8_t> uint8Array;
        uint8Array.reserve(inputVector.size() * sizeof(T));

        for (const T& value : inputVector) {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
            for (size_t i = 0; i < sizeof(T); ++i) {
                uint8Array.push_back(bytes[i]);
            }
        }
        return uint8Array;
    }

    void viewAllFunCode(){                              //显示所有的命令码
        std::map<uint8_t, std::map<uint8_t, answerMsgType*>>::iterator devIt;    //一级迭代器
        
        for (devIt = devMap.begin(); devIt != devMap.end(); devIt++){
            uint8_t devCode = devIt->first;                                     //设备码
            printf("devCode:0x%X ,fun:", devCode);
            std::map<uint8_t, answerMsgType*> funMap = devIt->second;            //设备功能码字典
            std::map<uint8_t, answerMsgType*>::iterator funIt;                   //二级迭代器
            for(funIt = funMap.begin(); funIt != funMap.end(); funIt++){
                uint8_t funCode = funIt->first;
                printf(" 0x%X", funCode);
                // busDev->registerFun(devCode, funCode);                          //注册设备及功能码
            }
            printf("\r\n");
        }
    }

    void setCheckDataType(int type);

private:
    
    u8Vector tailRemain;             //尾剩余 buf len
    unsigned char *data;

    std::thread sendId;
  
    SerialPort *port;
    void sendThreadFun(int thdNum);                                 //线程处理函数
    bool isRunning;
    
    int extract(unsigned char *msg, int len);                       //报文协议解析
    void analysisMsg(u8Vector msg);                                 //报文解析
    void analysisMsg(unsigned char *msg, int len);                  //报文解析方法2
    void notifyProcess(msgType m);                                 //通知并处理

    void packageToBus(u8Vector buf);                            //数据打包
    void packageToBus(uint8_t dev, uint8_t cmd, uint8_t fun, u8Vector info);                            //数据打包    

    uint16_t crc16(uint8_t* buf, uint16_t len);         //crc计算

    std::queue<msgType> revMsgTable;                    //接收消息列表(待处理的消息)
    std::queue<msgType> sendMsgTable;                   //发送的消息列表
    std::mutex revMsgMutex, sendMsgMutex;               //列表对应的线程安全锁

    std::map<uint8_t, std::map<uint8_t, answerMsgType*>> devMap;     //（总线设备上的）功能字典（内存换时间）
    
    /* return 0: ok; -1: dev is not register; -2 : function is not registerd */
    int getDevFunState(uint8_t dev, uint8_t code){                         //判断 dev 和 键值是否存在
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        bool isDevExist = (devMap.find(dev) != devMap.end()) ? true : false;
        if(!isDevExist) return -1;
        return (devMap[dev].find(code) != devMap[dev].end()) ? 0 : -2;
    }

    const bool isDebugViewData = true;                 //调试 数据显示标志
    void viewData(char *tag, u8Vector m){               //显示当前数据
        if(!isDebugViewData)    return;
        printf("%s hex : ", tag);
        for(int i=0; i<m.size(); i++){
            printf(" %02X", m[i]);
        }
        printf("\r\n");
    }

    void viewData(char *tag, unsigned char *m, int len){               //显示当前数据方法2
        if(!isDebugViewData)    return;
        printf("%s hex : ", tag);
        for(int i=0; i<len; i++){
            printf(" %02X", m[i]);
        }
        printf("\r\n");
    }
    void viewMsg(char *tag, msgType m){                            //显示当前报文
        if(!isDebugViewData)    return;
        if(m.cmd == cmd.at("autoReport")) return;                   //不显示主动数据
        RLOGD("%s: fIndex:%d, dev:0x%X, cmd:0x%X, fun:0x%X, data:", tag, m.fIndex, m.dev, m.cmd, m.fun, m.fIndex);
        // for(int i = 0; i < m.data.size(); i++){
        //     printf("0x%02X,",m.data[i]);
        // }
        // printf("\r\n");
    }

    typedef enum{
        OPEN_STATUS = 0 ,			//打开状态
        WORK_STATUS   	,	    	//正常通信状态
        ERROR_STATUS    ,           //错误状态
        ENUM_STATUS_MAX				//枚举 最大值
    } FsmState;

    uint8_t status = OPEN_STATUS;
    uint8_t lastStatus = OPEN_STATUS;

    const char* stateName[ENUM_STATUS_MAX] = {       
        {"open"},
        {"work"},
        {"error"},
    };

    bool openPort(void);
    void work(stdTimePoint time);
    void error(stdTimePoint time);

    void setStatus(uint8_t s);
    
    bool isOpen(void){
        bool ret = port->isOpen();
        if(!ret){
            RLOGW("port %s is not open, please wait for opening before using ...!\r\n", port->name.c_str());
        }
        return ret;
    }
    std::mutex isWorkMutex;
    bool checkPortOnline(int len);
    bool openStatusSync = false;
    bool errorStatusSync = false;
    stdTimePoint checkRevCntTime;   
    int checkRevLenTotal = 0;

    int checkDataTimes;
};

extern std::map<std::string, Protocol*> proInstanceMap;  

class Device
{
public:
    //创建设备
    static Protocol* create(std::string name){
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        bool isDevExist = (proInstanceMap.find(name) != proInstanceMap.end()) ? true : false;
        if(!isDevExist){
            proInstanceMap[name] = new Protocol(name);
            RLOGI("Create protocol addr %p , %s \r\n", proInstanceMap[name], name.c_str());
        }
        return proInstanceMap[name];
    }

private:
 
};

//模块类
#define devFunMapType std::map<uint8_t, std::map<std::string, uint8_t>>
class Moudle
{
public:
    Protocol *busDev;                                               //总线设备
    uint8_t devCode;                                                //设备码
    Moudle(){};
    Moudle(std::string busName, devFunMapType map){                 //析构 支持挂载，也可以直接调用挂载函数
        mountBus(busName, map);                                     //挂载总线
    }

    void mountBus(std::string busName, devFunMapType map){          //挂载总线
        devFun = map;
        busDev = Device::create(busName);                           //创建总线设备

        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        devFunMapType::iterator devIt;                              //一级迭代器
        for (devIt = map.begin(); devIt != map.end(); devIt++){
            devCode = devIt->first;                                 //设备码
            std::map<std::string, uint8_t> funMap = devIt->second;  //设备功能码字典
            std::map<std::string, uint8_t>::iterator funIt;         //二级迭代器
            for(funIt = funMap.begin(); funIt != funMap.end(); funIt++){
                uint8_t funCode = funIt->second;
                busDev->registerFun(devCode, funCode);              //注册设备及功能码
            }
        }
        if (const char* debug_env = std::getenv("REMOTE_CONTROL_LITE_DEBUG_PROTOCOL")) {
            if (!(debug_env[0] == '0' && debug_env[1] == '\0')) {
                busDev->viewAllFunCode();
            }
        }
    }

    uint8_t getFun(std::string key){                                //通过 键值 获取功能码
        return devFun.at(devCode).at(key);
    }

    template<typename T, typename Func> 
    void setRevCallBack(T& obj, Func func){                         //设置接收回调函数
        busDev->registerFunction(devCode, obj, func);                    
    }

    template<typename T, typename Func> 
    void setOpenStatusCallBack(T& obj, Func func){                  //设置打开成功回调
        busDev->registerOpenStatusFunction(devCode, obj, func);                    
    }

    template<typename T, typename Func> 
    void setErrorStatusCallBack(T& obj, Func func){                  //设置发生错误回调
        busDev->registerErrorStatusFunction(devCode, obj, func);                    
    }

    void reOpen(){
        if(busDev){     busDev->reOpen();   
        }else{          RLOGE("protocol bus dev is null !!! \r\n"); }
    }

    bool isWork(bool isBlock, int timeOut = 4000){
        return busDev->isWork(isBlock, timeOut);
    }

    typedef enum{
        INIT_STATUS = 0 ,			//初始化
        RUN_STATUS   	,	    	//轨迹运动
        PAUSE_STATUS  	,			//暂停下电
        ERROR_STATUS	,			//错误_上下电尝试恢复
        POWER_ON_STATUS,			//开机状态
        SHUT_DOWN_STATUS,			//关机状态
        ENUM_STATUS_MAX				//枚举 最大值
    } fsmState;

    const char* stateName[ENUM_STATUS_MAX] = {       
        {"init"},
        {"run"},
        {"pause"},
        {"error"},
        {"poweron"},
        {"shutdown"},
    };

    ~Moudle(){}

private:
    devFunMapType devFun;                                           //仅模块的
};



#endif //PROCOTOL_PROTOCOL_H
