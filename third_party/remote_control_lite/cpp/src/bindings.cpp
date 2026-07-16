// Pybind11 bindings exposing RemoteArmDevice and data types.
#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "remote_arm/device.h"

namespace py = pybind11;
using namespace remote_arm;

namespace {
std::array<double, kArmDofs> AsArray(const py::sequence& seq) {
  if (py::len(seq) != static_cast<py::ssize_t>(kArmDofs)) {
    throw std::runtime_error("Expected sequence of length 7");
  }
  std::array<double, kArmDofs> out{};
  for (size_t i = 0; i < kArmDofs; ++i) {
    out[i] = seq[i].cast<double>();
  }
  return out;
}
}  // namespace

PYBIND11_MODULE(_remote_arm_cpp, m) {
  m.doc() = "Minimal bindings for Galbot remote arm reader";

  py::enum_<ArmSide>(m, "ArmSide")
      .value("Left", ArmSide::Left)
      .value("Right", ArmSide::Right)
      .export_values();

  py::class_<ArmSample>(m, "ArmSample")
      .def_property_readonly("sequence", [](const ArmSample& sample) { return sample.sequence; })
      .def_property_readonly("timestamp", [](const ArmSample& sample) { return sample.timestamp; })
      .def_property_readonly("position", [](const ArmSample& sample) { return sample.position; })
      .def_property_readonly("velocity", [](const ArmSample& sample) { return sample.velocity; })
      .def_property_readonly("torque", [](const ArmSample& sample) { return sample.torque; });

  py::class_<JoystickSample>(m, "JoystickSample")
      .def_property_readonly("sequence", [](const JoystickSample& sample) { return sample.sequence; })
      .def_property_readonly("timestamp", [](const JoystickSample& sample) { return sample.timestamp; })
      .def_property_readonly("axis", [](const JoystickSample& sample) {
        return py::make_tuple(sample.axis_x, sample.axis_y);
      })
      .def_property_readonly("trigger", [](const JoystickSample& sample) {
        return py::make_tuple(sample.trigger_x, sample.trigger_y);
      })
      .def_property_readonly("buttons", [](const JoystickSample& sample) {
        return std::vector<uint8_t>(sample.buttons.begin(), sample.buttons.end());
      });

  py::class_<RemoteArmDevice>(m, "RemoteArmDevice")
      .def(py::init<>())
      .def("configure",
           [](RemoteArmDevice& self,
              const py::sequence& left_init,
              const py::sequence& right_init,
              const py::sequence& left_scale,
              const py::sequence& right_scale) {
             self.configure(AsArray(left_init), AsArray(right_init), AsArray(left_scale),
                            AsArray(right_scale));
           })
      .def("start",
           [](RemoteArmDevice& self, const std::string& port) {
             py::gil_scoped_release release;
             self.start(port);
           },
           py::arg("port"))
      .def("stop",
           [](RemoteArmDevice& self) {
             py::gil_scoped_release release;
             self.stop();
           })
      .def("is_running", &RemoteArmDevice::isRunning)
      .def("has_sample", &RemoteArmDevice::hasSample, py::arg("side"))
      .def("latest_sample",
           [](const RemoteArmDevice& self, ArmSide side) { return self.latestSample(side); },
           py::arg("side"))
      .def("has_joystick", &RemoteArmDevice::hasJoystick, py::arg("side"))
      .def("latest_joystick",
           [](const RemoteArmDevice& self, ArmSide side) { return self.latestJoystick(side); },
           py::arg("side"));
}
