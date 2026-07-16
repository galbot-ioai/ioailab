// Minimal, ROS-free device wrapper for Galbot remote teleop arm.
// Consumes auto-reported frames (armL/armR) from the MCU protocol
// and exposes latest per-arm joint samples to Python via pybind11.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "RemoteOperate.h"
#include "System.h"

namespace remote_arm {

constexpr std::size_t kArmDofs = 7;
constexpr std::size_t kButtonCount = 9;  // key1..key9

enum class ArmSide {
  Left = 0,
  Right = 1,
};

// Thread-safe sample container per arm.
struct ArmSample {
  std::uint64_t sequence;
  double timestamp;
  std::array<double, kArmDofs> position;
  std::array<double, kArmDofs> velocity;
  std::array<double, kArmDofs> torque;
};

struct JoystickSample {
  std::uint64_t sequence;
  double timestamp;
  uint16_t axis_x;
  uint16_t axis_y;
  uint16_t trigger_x;
  uint16_t trigger_y;
  std::array<uint8_t, kButtonCount> buttons;
};

// RemoteArmDevice owns protocol devices and receives armL/armR reports.
class RemoteArmDevice {
 public:
  RemoteArmDevice();
  ~RemoteArmDevice();

  RemoteArmDevice(const RemoteArmDevice&) = delete;
  RemoteArmDevice& operator=(const RemoteArmDevice&) = delete;

  // Configure per-arm joint offsets and scaling (applied to positions only).
  void configure(const std::array<double, kArmDofs>& left_init,
                 const std::array<double, kArmDofs>& right_init,
                 const std::array<double, kArmDofs>& left_scale,
                 const std::array<double, kArmDofs>& right_scale);

  // Start protocol and register callbacks on given serial port.
  void start(const std::string& port);
  void stop();
  bool isRunning() const;

  // Latest sample accessors (lock protected, non-blocking).
  bool hasSample(ArmSide side) const;
  ArmSample latestSample(ArmSide side) const;

  bool hasJoystick(ArmSide side) const;
  JoystickSample latestJoystick(ArmSide side) const;

 private:
  void initDevices(const std::string& port);
  void deinitDevices();

  void sysRevCallBack(uint8_t cmd_type, uint8_t function_code, std::vector<uint8_t> payload);
  // Protocol callbacks.
  void remoterRevCallBack(uint8_t cmd_type, uint8_t function_code, std::vector<uint8_t> payload);
  void handleArm(RemoteOperate::armInfo* info, ArmSide side);
  void handlePole(const RemoteOperate::Pole& info, ArmSide side, bool is_trigger);
  void handleButton(ArmSide side, std::size_t index, uint8_t status);
  int buttonIndex(ArmSide side, uint8_t function_code) const;
  void noopSystemCallback(uint8_t cmd_type, uint8_t function_code, std::vector<uint8_t> payload);
  void noopRemoteCallback(uint8_t cmd_type, uint8_t function_code, std::vector<uint8_t> payload);
  void disableReporting();

  struct SharedState {
    ArmSample sample{};
    bool ready{false};
  };

  struct JoystickSharedState {
    JoystickSample sample{};
    bool ready{false};
  };

  std::unique_ptr<System> system_;
  std::unique_ptr<RemoteOperate> remote_;

  std::array<std::array<double, kArmDofs>, 2> init_offsets_;
  std::array<std::array<double, kArmDofs>, 2> scales_;

  mutable std::mutex mutexes_[2];
  SharedState states_[2];

  mutable std::mutex joystick_mutexes_[2];
  JoystickSharedState joystick_states_[2];
  std::array<uint8_t, kButtonCount> left_button_functions_{};
  std::array<uint8_t, kButtonCount> right_button_functions_{};
  std::atomic<bool> running_;
};

}  // namespace remote_arm
