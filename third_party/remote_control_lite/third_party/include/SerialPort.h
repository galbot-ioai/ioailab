#ifndef SERIALCOM_SERIALPORT_H
#define SERIALCOM_SERIALPORT_H

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <mutex>
#include <stdio.h>
#include <vector>

// using u8Vector = std::vector<uint8_t>;
#define u8Vector  std::vector<uint8_t>

#define SERIAL_BUF_SIZE 1024

class SerialPort {
public:
    SerialPort();
    ~SerialPort(){};
    bool _open(std::string name, speed_t baudRate);
    void _close();

    bool reStart(); 

    bool isOpen() const;
    bool writeData(const char* data, size_t size);
    int readData(char *buffer);

    bool writeData(u8Vector v);
    u8Vector readData();

    void lseek(int offset);
    unsigned char* readData_sharememory(int *len);
    void viewData(char *tag, u8Vector v);
    // const char* name;
    std::string name;
    speed_t baudRate;

    int isAccess(void){
        return access(name.c_str(), 0);
    }
    
private:
    int fd = -1;
    const int revMaxLen = SERIAL_BUF_SIZE;
    pthread_mutex_t serial_mutex = PTHREAD_MUTEX_INITIALIZER;
    std::mutex mutex;

     
    char revBuf[SERIAL_BUF_SIZE];
    int wpos;
};

#endif //SERIALCOM_SERIALPORT_H
