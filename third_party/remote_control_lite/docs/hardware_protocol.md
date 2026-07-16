# 硬件协议与命令（Hardware Protocol & Commands）

本工程使用随包提供的 3rd 动态库与头文件完成串口协议：
- 头文件：third_party/include/Protocol.h, RemoteOperate.h, System.h, SerialPort.h
- 动态库：src/remote_control_lite/libs/libgalbot*.so

## 协议要点
- 报文头尾：HEADH=0xDE, HEADL=0xED, TAILH=0xEA, TAILL=0xAE（见 Protocol.h）
- 命令类别（cmd）：read=0x20, write=0x21, answer=0x22, autoReport=0x23
- 设备与功能注册：通过 Moudle::mountBus 将设备码与功能码映射注册到 Protocol 实例

## System 设备（System.h）
- 功能：heartbeat, version, sncode, bms, radar
- BMS 数据结构 BmsType：包括电压、电流、温度、容量、状态与告警等
- 本工程默认仅监听心跳（诊断），不对外暴露 System API

## RemoteOperate 设备（RemoteOperate.h）
- 模式：OPERATE_MODE（遥操模式）
- 左右臂：`armL`、`armR` -> 负载类型 armInfo（7×motorInfo，含角度/速度/力矩）
- 摇杆/扳机：`poleL/poleR`、`trigerL/trigerR` -> RemoteOperate::Pole（uint16 y/x）
- 按键事件：`key1L...key9L`、`key1R...key9R` -> 单字节状态（1=DOWN,2=UP,3=LONG,4=CLICK）
- 自动上报开关：`setReport(bool)`（本封装在启动/停止时启闭）

## 控制能力
- 公开头文件未提供直接“下发电机控制”指令，RemoteOperate 以输入上报为主
- 若需扩展写指令，可基于 `Protocol::write(dev, fun, payload, timeout, isBlock)` 发送
  - 请参照固件文档确认功能码与 payload 格式

## 在本工程中的使用
- 仅启用 autoReport 的关节/摇杆/扳机/按键通路
- 不做写指令，保持仅读以精简依赖与避免误操作
