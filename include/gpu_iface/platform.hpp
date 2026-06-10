// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "gpu_runtime_compat.hpp"
#include "macros.hpp"

namespace flashinfer {
namespace gpu_iface {

// Platform-agnostic stream type
#if defined(PLATFORM_CUDA_DEVICE)
constexpr int kWarpSize = 32;

#elif defined(PLATFORM_HIP_DEVICE)
// NOTE: __gfx1201__ is only defined in the device compilation pass.
// In the host compilation pass (amdclang -xhip host path), __gfx1201__ is NOT
// set, so the compile-time kWarpSize defaults to 64 for all HIP targets.
// Host-side kernel launch code (e.g. dim3 nthrs) must NOT rely on this constant.
// Use getHostWarpSize() at runtime instead.
#if defined(__gfx1201__)
constexpr int kWarpSize = 32;
#else
constexpr int kWarpSize = 64;
#endif

/// @brief Query the hardware warp size (wavefront size) of the current device at runtime.
/// @details This is the correct way to obtain the warp size on the HOST side because
///          architecture-specific macros like __gfx1201__ are only defined during the
///          device compilation pass and are unavailable in host code.
/// @return 32 for gfx1201 (RDNA4 / wave32) and 64 for CDNA3/CDNA4 (gfx942/gfx950).
inline int getHostWarpSize() {
  int device_id = 0;
  hipGetDevice(&device_id);
  int warp_size = 64;  // safe default
  hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device_id);
  return warp_size;
}

#endif

}  // namespace gpu_iface
}  // namespace flashinfer
