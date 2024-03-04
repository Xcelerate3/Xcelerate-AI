#pragma once

#include <functional>
#include <string>

namespace xcelerate {
namespace onnx {

enum class DeviceType { CPU = 0, CUDA = 1 };

struct Device {
  Device(const std::string& spec);
  DeviceType type;
  int device_id{-1};
};

} // namespace onnx
} // namespace xcelerate

namespace std {
template <>
struct hash<xcelerate::onnx::DeviceType> {
  std::size_t operator()(const xcelerate::onnx::DeviceType& k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
