#include "remote_arm/device.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

using std::size_t;

namespace remote_arm {

namespace {
constexpr std::array<double, kArmDofs> kDefaultLeftInit{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
constexpr std::array<double, kArmDofs> kDefaultRightInit{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
constexpr std::array<double, kArmDofs> kDefaultLeftScale{1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
constexpr std::array<double, kArmDofs> kDefaultRightScale{-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
}  // namespace

// Initialize default scales and clear state.
RemoteArmDevice::RemoteArmDevice() : running_(false) {
  configure(kDefaultLeftInit, kDefaultRightInit, kDefaultLeftScale, kDefaultRightScale);
  for (auto& state : states_) {
    state.sample.sequence = 0;
    state.sample.timestamp = 0.0;
    state.sample.position.fill(0.0);
    state.sample.velocity.fill(0.0);
    state.sample.torque.fill(0.0);
    state.ready = false;
  }
  for (auto& joy : joystick_states_) {
    joy.sample.sequence = 0;
    joy.sample.timestamp = 0.0;
    joy.sample.axis_x = 0;
    joy.sample.axis_y = 0;
    joy.sample.trigger_x = 0;
    joy.sample.trigger_y = 0;
    joy.sample.buttons.fill(RemoteOperate::KEY_UP);
    joy.ready = false;
  }
}

RemoteArmDevice::~RemoteArmDevice() {
  stop();
}

void RemoteArmDevice::configure(const std::array<double, kArmDofs>& left_init,
                                const std::array<double, kArmDofs>& right_init,
                                const std::array<double, kArmDofs>& left_scale,
                                const std::array<double, kArmDofs>& right_scale) {
  init_offsets_[static_cast<size_t>(ArmSide::Left)] = left_init;
  init_offsets_[static_cast<size_t>(ArmSide::Right)] = right_init;
  scales_[static_cast<size_t>(ArmSide::Left)] = left_scale;
  scales_[static_cast<size_t>(ArmSide::Right)] = right_scale;
}

// Start protocol devices on the given serial port.
void RemoteArmDevice::start(const std::string& port) {
  if (running_.load()) {
    return;
  }
  initDevices(port);
  running_.store(true);
}

void RemoteArmDevice::stop() {
  if (!running_.load()) {
    return;
  }
  disableReporting();
  deinitDevices();
  running_.store(false);
}

bool RemoteArmDevice::isRunning() const {
  return running_.load();
}

bool RemoteArmDevice::hasSample(ArmSide side) const {
  const auto index = static_cast<size_t>(side);
  std::lock_guard<std::mutex> lock(mutexes_[index]);
  return states_[index].ready;
}

ArmSample RemoteArmDevice::latestSample(ArmSide side) const {
  const auto index = static_cast<size_t>(side);
  std::lock_guard<std::mutex> lock(mutexes_[index]);
  return states_[index].sample;
}

// Create System/RemoteOperate and hook callbacks.
void RemoteArmDevice::initDevices(const std::string& port) {
  if (!port.empty()) {
    system_ = std::make_unique<System>(port);
    if (system_) {
      system_->setRevCallBack(*this, &RemoteArmDevice::sysRevCallBack);
    }

    remote_ = std::make_unique<RemoteOperate>(port);
    if (!remote_) {
      throw std::runtime_error("Failed to create RemoteOperate instance");
    }
    remote_->setRevCallBack(*this, &RemoteArmDevice::remoterRevCallBack);
    const std::array<const char*, kButtonCount> left_keys{
        "key1L", "key2L", "key3L", "key4L", "key5L", "key6L", "key7L", "key8L", "key9L"};
    const std::array<const char*, kButtonCount> right_keys{
        "key1R", "key2R", "key3R", "key4R", "key5R", "key6R", "key7R", "key8R", "key9R"};
    for (std::size_t i = 0; i < kButtonCount; ++i) {
      left_button_functions_[i] = remote_->getFun(left_keys[i]);
      right_button_functions_[i] = remote_->getFun(right_keys[i]);
    }
    remote_->setReport(true);
  } else {
    throw std::invalid_argument("Port name must not be empty");
  }
}

void RemoteArmDevice::deinitDevices() {
  remote_.reset();
  system_.reset();
}

void RemoteArmDevice::disableReporting() {
  using namespace std::chrono_literals;
  if (remote_) {
    try {
      remote_->setReport(false);
    } catch (const std::exception&) {
      // Ignore shutdown failures; we are stopping anyway.
    }
    remote_->setRevCallBack(*this, &RemoteArmDevice::noopRemoteCallback);
  }
  if (system_) {
    system_->setRevCallBack(*this, &RemoteArmDevice::noopSystemCallback);
  }
  if (remote_) {
    std::this_thread::sleep_for(20ms);
  }
}

void RemoteArmDevice::sysRevCallBack(uint8_t cmd_type,
                                     uint8_t function_code,
                                     std::vector<uint8_t> payload) {
  (void)cmd_type;
  (void)function_code;
  (void)payload;
  // Diagnostics are ignored in standalone mode.
}

// Receive auto-reported frames and dispatch per arm.
void RemoteArmDevice::remoterRevCallBack(uint8_t cmd_type,
                                         uint8_t function_code,
                                         std::vector<uint8_t> payload) {
  if (cmd_type != cmd.at("autoReport")) {
    return;
  }

  if (!remote_) {
    return;
  }

  if (function_code == remote_->getFun("armL")) {
    if (payload.size() == sizeof(RemoteOperate::armInfo)) {
      RemoteOperate::armInfo info{};
      std::memcpy(&info, payload.data(), sizeof(RemoteOperate::armInfo));
      handleArm(&info, ArmSide::Left);
    }
  } else if (function_code == remote_->getFun("armR")) {
    if (payload.size() == sizeof(RemoteOperate::armInfo)) {
      RemoteOperate::armInfo info{};
      std::memcpy(&info, payload.data(), sizeof(RemoteOperate::armInfo));
      handleArm(&info, ArmSide::Right);
    }
  } else if (function_code == remote_->getFun("poleL")) {
    if (payload.size() == sizeof(RemoteOperate::Pole)) {
      RemoteOperate::Pole pole{};
      std::memcpy(&pole, payload.data(), sizeof(RemoteOperate::Pole));
      handlePole(pole, ArmSide::Left, /*is_trigger=*/false);
    }
  } else if (function_code == remote_->getFun("poleR")) {
    if (payload.size() == sizeof(RemoteOperate::Pole)) {
      RemoteOperate::Pole pole{};
      std::memcpy(&pole, payload.data(), sizeof(RemoteOperate::Pole));
      handlePole(pole, ArmSide::Right, /*is_trigger=*/false);
    }
  } else if (function_code == remote_->getFun("trigerL")) {
    if (payload.size() == sizeof(RemoteOperate::Pole)) {
      RemoteOperate::Pole trig{};
      std::memcpy(&trig, payload.data(), sizeof(RemoteOperate::Pole));
      handlePole(trig, ArmSide::Left, /*is_trigger=*/true);
    }
  } else if (function_code == remote_->getFun("trigerR")) {
    if (payload.size() == sizeof(RemoteOperate::Pole)) {
      RemoteOperate::Pole trig{};
      std::memcpy(&trig, payload.data(), sizeof(RemoteOperate::Pole));
      handlePole(trig, ArmSide::Right, /*is_trigger=*/true);
    }
  } else {
    int left_idx = buttonIndex(ArmSide::Left, function_code);
    if (left_idx >= 0) {
      uint8_t status = payload.size() > 1 ? payload[1] : (payload.empty() ? 0 : payload[0]);
      handleButton(ArmSide::Left, static_cast<std::size_t>(left_idx), status);
      return;
    }
    int right_idx = buttonIndex(ArmSide::Right, function_code);
    if (right_idx >= 0) {
      uint8_t status = payload.size() > 1 ? payload[1] : (payload.empty() ? 0 : payload[0]);
      handleButton(ArmSide::Right, static_cast<std::size_t>(right_idx), status);
      return;
    }
  }
}

// Convert raw armInfo to ArmSample applying offset/scale.
void RemoteArmDevice::handleArm(RemoteOperate::armInfo* info, ArmSide side) {
  if (!info) {
    return;
  }

  using clock = std::chrono::steady_clock;
  const auto now = clock::now();
  const double timestamp =
      std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();

  const auto index = static_cast<size_t>(side);
  ArmSample sample{};
  sample.timestamp = timestamp;

  const auto& offsets = init_offsets_[index];
  const auto& scales = scales_[index];

  for (size_t i = 0; i < kArmDofs; ++i) {
    const double raw_pos = static_cast<double>(info->m[i].curPos);
    const double raw_vel = static_cast<double>(info->m[i].curSpeed);
    const double raw_torque = static_cast<double>(info->m[i].curTorque);

    sample.position[i] = (raw_pos - offsets[i]) * scales[i];
    sample.velocity[i] = raw_vel;
    sample.torque[i] = raw_torque;
  }

  std::lock_guard<std::mutex> lock(mutexes_[index]);
  auto& state = states_[index];
  state.sample.sequence += 1;
  state.sample.timestamp = sample.timestamp;
  state.sample.position = sample.position;
  state.sample.velocity = sample.velocity;
  state.sample.torque = sample.torque;
  state.ready = true;
}

bool RemoteArmDevice::hasJoystick(ArmSide side) const {
  const auto index = static_cast<size_t>(side);
  std::lock_guard<std::mutex> lock(joystick_mutexes_[index]);
  return joystick_states_[index].ready;
}

JoystickSample RemoteArmDevice::latestJoystick(ArmSide side) const {
  const auto index = static_cast<size_t>(side);
  std::lock_guard<std::mutex> lock(joystick_mutexes_[index]);
  return joystick_states_[index].sample;
}

void RemoteArmDevice::handlePole(const RemoteOperate::Pole& info, ArmSide side, bool is_trigger) {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now();
  const double timestamp =
      std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();

  const auto index = static_cast<size_t>(side);
  std::lock_guard<std::mutex> lock(joystick_mutexes_[index]);
  auto& state = joystick_states_[index];
  state.sample.sequence += 1;
  state.sample.timestamp = timestamp;
  if (is_trigger) {
    state.sample.trigger_x = info.x;
    state.sample.trigger_y = info.y;
  } else {
    state.sample.axis_x = info.x;
    state.sample.axis_y = info.y;
  }
  state.ready = true;
}

void RemoteArmDevice::handleButton(ArmSide side, std::size_t index, uint8_t status) {
  if (index >= kButtonCount) {
    return;
  }
  using clock = std::chrono::steady_clock;
  const auto now = clock::now();
  const double timestamp =
      std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();

  const auto arm_index = static_cast<size_t>(side);
  std::lock_guard<std::mutex> lock(joystick_mutexes_[arm_index]);
  auto& state = joystick_states_[arm_index];
  state.sample.sequence += 1;
  state.sample.timestamp = timestamp;
  state.sample.buttons[index] = status;
  state.ready = true;
}

int RemoteArmDevice::buttonIndex(ArmSide side, uint8_t function_code) const {
  const auto& codes = (side == ArmSide::Left) ? left_button_functions_ : right_button_functions_;
  for (std::size_t i = 0; i < codes.size(); ++i) {
    if (codes[i] == function_code) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void RemoteArmDevice::noopSystemCallback(uint8_t, uint8_t, std::vector<uint8_t>) {}

void RemoteArmDevice::noopRemoteCallback(uint8_t, uint8_t, std::vector<uint8_t>) {}

}  // namespace remote_arm
