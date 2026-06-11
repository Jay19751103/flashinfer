// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_fp8.h>

#include <type_traits>

#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

namespace {
using f16 = _Float16;
using f16x4 = f16 __attribute__((ext_vector_type(4)));
using f16x8 = f16 __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using bf16x4 = short __attribute__((ext_vector_type(4)));
using bf16x8 = short __attribute__((ext_vector_type(8)));
using i32x2 = int __attribute__((ext_vector_type(2)));

}  // namespace

namespace flashinfer {
namespace gpu_iface {
namespace mma_impl {
namespace hip {

#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)

#if HIP_FP8_CVT_FAST_PATH && !HIP_FP8_TYPE_FNUZ
__device__ __forceinline__ float fp8_e4m3_fnuz_to_float(uint8_t x) {
  if (x == 0x80u) {
    float nan;
    uint32_t nan_bits = 0x7FC00000u;
    __builtin_memcpy(&nan, &nan_bits, sizeof(float));
    return nan;
  }
  uint32_t sign = (uint32_t)(x >> 7) & 1u;
  uint32_t exp8 = (uint32_t)(x >> 3) & 0xFu;
  uint32_t mant8 = (uint32_t)(x) & 0x7u;
  uint32_t f32;
  if (exp8 == 0u) {
    if (mant8 == 0u) {
      f32 = sign << 31u;
    } else {
      uint32_t p = 31u - __builtin_clz(mant8);
      f32 = (sign << 31u) | ((p + 117u) << 23u) | ((mant8 ^ (1u << p)) << (23u - p));
    }
  } else {
    f32 = (sign << 31u) | ((exp8 + 119u) << 23u) | (mant8 << 20u);
  }
  float result;
  __builtin_memcpy(&result, &f32, sizeof(float));
  return result;
}

__device__ __forceinline__ float fp8_e5m2_fnuz_to_float(uint8_t x) {
  if (x == 0x80u) {
    float nan;
    uint32_t nan_bits = 0x7FC00000u;
    __builtin_memcpy(&nan, &nan_bits, sizeof(float));
    return nan;
  }
  uint32_t sign = (uint32_t)(x >> 7) & 1u;
  uint32_t exp8 = (uint32_t)(x >> 2) & 0x1Fu;
  uint32_t mant8 = (uint32_t)(x) & 0x3u;
  uint32_t f32;
  if (exp8 == 0u) {
    if (mant8 == 0u) {
      f32 = sign << 31u;
    } else {
      uint32_t p = 31u - __builtin_clz(mant8);
      f32 = (sign << 31u) | ((p + 110u) << 23u) | ((mant8 ^ (1u << p)) << (23u - p));
    }
  } else {
    f32 = (sign << 31u) | ((exp8 + 111u) << 23u) | (mant8 << 21u);
  }
  float result;
  __builtin_memcpy(&result, &f32, sizeof(float));
  return result;
}
#endif

template <typename T>
__device__ __forceinline__ float fp8_to_float(T value) {
  if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz>) {
#if HIP_FP8_CVT_FAST_PATH && !HIP_FP8_TYPE_FNUZ
    return fp8_e4m3_fnuz_to_float(value.__x);
#else
    return static_cast<float>(value);
#endif
  } else if constexpr (std::is_same_v<T, __hip_fp8_e5m2_fnuz>) {
#if HIP_FP8_CVT_FAST_PATH && !HIP_FP8_TYPE_FNUZ
    return fp8_e5m2_fnuz_to_float(value.__x);
#else
    return static_cast<float>(value);
#endif
  } else {
    return static_cast<float>(value);
  }
}

/// @brief Transposes a 4x4 matrix of `half` values held across a quad of 4 threads.
/// @details This function operates on a group of 4 consecutive threads (a quad). It assumes
///          each thread holds 4 `half` values, which together form a 4x4 matrix where each
///          thread holds one row. The function permutes these values using a series of
///          `__shfl_xor` operations so that each thread ends up holding one column of the
///          original 4x4 matrix.
///
///          Visual Representation:
///          If `[a,b,c,d]` are the 4 `half` values in Thread 0's registers:
///
///          Before:                          After:
///          Thread 0: [a, b, c, d]           Thread 0: [a, e, i, m]
///          Thread 1: [e, f, g, h]   --->    Thread 1: [b, f, j, n]
///          Thread 2: [i, j, k, l]           Thread 2: [c, g, k, o]
///          Thread 3: [m, n, o, p]           Thread 3: [d, h, l, p]
///
/// @note    This function can be combined with `transpose_inter_quad_fragments` to perform a
///          full 16x16 in-register matrix transpose. This function handles the transposition
///          *within* each 4x4 data block.
__device__ __forceinline__ void transpose_intra_quad_fragments(uint32_t* R) {
  // Calculate lane within 4-thread group
  uint32_t lane_id = threadIdx.x % 32;
  uint32_t lane_in_group = lane_id % 4;

  // === ROUND 1: Exchange with neighbor (XOR with 1) ===
  // T0 <-> T1, T2 <-> T3 partial exchange
  uint32_t regid = (lane_in_group >> 1) & 0x1;
  uint32_t exchanged_val = __shfl_xor(R[regid], 0x1);
  uint32_t shift = (lane_in_group & 1) * 16;
  uint32_t keep_mask = 0x0000FFFF << shift;
  int left_shift_amount = 16 * (1 - (lane_in_group & 1));
  int right_shift_amount = 16 * (lane_in_group & 1);
  R[regid] = (R[regid] & keep_mask) | ((exchanged_val >> right_shift_amount) << left_shift_amount);

  // === ROUND 2: Exchange with one hop (XOR with 2) ===
  // T0 <-> T2, T1 <-> T3 exchange R[0] and R[1]
  // Swap entire registers based on thread position
  uint32_t is_top = 1 - regid;
  uint32_t temp0 = __shfl_xor(R[0], 0x2);
  uint32_t temp1 = __shfl_xor(R[1], 0x2);

  // Compute both possibilities and select
  R[0] = R[0] * is_top + temp1 * regid;
  R[1] = temp0 * is_top + R[1] * regid;

  // === ROUND 3: Exchange with neighbor again (XOR with 1) ===
  // T0 <-> T1, T2 <-> T3 exchange remaining parts

  regid = 1 - regid;
  exchanged_val = __shfl_xor(R[regid], 0x1);
  R[regid] = (R[regid] & keep_mask) | ((exchanged_val >> right_shift_amount) << left_shift_amount);
}

/// @brief Permutes matrix fragments between thread quads in a wavefront to perform a block-wise
///        transpose.
/// @details This function treats the 64-thread wavefront as a 4x4 grid of thread quads.
///          Each quad (4 consecutive threads) is considered to hold a 4x4 data fragment.
///          The function transposes this 4x4 grid of fragments by swapping the register
///          contents of threads in off-diagonal quads.
///
///          Visual Representation:
///          If B(r,c) is the 4x4 data fragment held by the quad at block-row 'r' and block-col 'c':
///
///          Before:                                          After:
///          +--------+--------+--------+--------+            +--------+--------+--------+--------+
///          | B(0,0) | B(0,1) | B(0,2) | B(0,3) |            | B(0,0) | B(1,0) | B(2,0) | B(3,0) |
///          +--------+--------+--------+--------+            +--------+--------+--------+--------+
///          | B(1,0) | B(1,1) | B(1,2) | B(1,3) |   --->     | B(0,1) | B(1,1) | B(2,1) | B(3,1) |
//          +--------+--------+--------+--------+            +--------+--------+--------+--------+
///          | B(2,0) | B(2,1) | B(2,2) | B(2,3) |            | B(0,2) | B(1,2) | B(2,2) | B(3,2) |
///          +--------+--------+--------+--------+            +--------+--------+--------+--------+
///          | B(3,0) | B(3,1) | B(3,2) | B(3,3) |            | B(0,3) | B(1,3) | B(2,3) | B(3,3) |
///          +--------+--------+--------+--------+            +--------+--------+--------+--------+
///
/// @note    This function can be combined with `transpose_intra_quad_fragments` (which transposes
///          the data *within* each fragment) to perform a full 16x16 in-register matrix transpose.
__device__ __forceinline__ void transpose_inter_quad_fragments(uint32_t* R) {
  uint32_t lane_id = threadIdx.x % 32;

  uint32_t block_row = (lane_id % 16) / 4;
  uint32_t block_col = (lane_id / 16);
  uint32_t thread_in_block = lane_id % 4;
  uint32_t partner_lane_id = (block_row * 16) + (block_col * 4) + thread_in_block;
  uint32_t xor_mask = lane_id ^ partner_lane_id;

  // Exchange both registers with the partner thread
  R[0] = __shfl_xor(R[0], xor_mask, 32);
  R[1] = __shfl_xor(R[1], xor_mask, 32);
}

/// @brief Performs a full 16x16 in-register matrix transpose by combining intra-quad and
///        inter-quad fragment transpositions.
/// @details This function converts between A-matrix layout (row-major) and B/C/D-matrix layout
///          (column-major) for CDNA3 MFMA operations. It applies both
///          transpose_intra_quad_fragments and transpose_inter_quad_fragments to fully transpose a
///          16x16 tile distributed across 64 threads.
///
///          For gfx1201 (RDNA4, wave32): WMMA D-output layout:
///            thread t, reg j → D[j*2 + (t/16)][t%16]
///          WMMA A-matrix input layout:
///            thread t, reg k → A[t%16][(t/16)*8 + k]
///          Transpose D→A means rearranging so that A[col_D][row_D] is held correctly.
///          Each group needs its own partial rows interleaved with the partner group's data
///          via __shfl_xor(R, 16).
///
/// @param R Pointer to fragment registers (2 uint32 on CDNA3, 4 uint32 on gfx1201)
__device__ __forceinline__ void transpose_mma_tile(uint32_t* R) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t dst_row = lane % 16;
  const uint32_t dst_k_base = (lane / 16) * 8;
  const uint32_t src_row_group = dst_row & 1u;
  const uint32_t src_reg = dst_row >> 1;
  const uint32_t src_word = src_reg >> 1;
  const uint32_t src_half = src_reg & 1u;
  uint32_t out[4];
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t packed = 0;
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      const uint32_t dst_k = dst_k_base + i * 2 + h;
      const uint32_t src_lane = src_row_group * 16 + dst_k;
      const uint32_t src0 = __shfl(R[0], src_lane, 32);
      const uint32_t src1 = __shfl(R[1], src_lane, 32);
      const uint32_t src2 = __shfl(R[2], src_lane, 32);
      const uint32_t src3 = __shfl(R[3], src_lane, 32);
      const uint32_t src =
          (src_word == 0u) ? src0 : ((src_word == 1u) ? src1 : ((src_word == 2u) ? src2 : src3));
      const uint32_t value = (src_half == 0) ? (src & 0xFFFFu) : (src >> 16);
      packed |= value << (16 * h);
    }
    out[i] = packed;
  }
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    R[i] = out[i];
  }
#else
  transpose_intra_quad_fragments(R);
  transpose_inter_quad_fragments(R);
#endif
}

// Single unified load function for all fragment types
/// @param R [in] pointer to the register file to load the fragment into
/// @param smem_ptr [in] pointer to the shared memory to load the fragment from
template <typename T>
__device__ __forceinline__ void load_fragment(uint32_t* R, const T* smem_ptr) {
  R[0] = reinterpret_cast<const uint32_t*>(smem_ptr)[0];
  R[1] = reinterpret_cast<const uint32_t*>(smem_ptr)[1];
}

// MMA operation for FP16 inputs with FP32 accumulator
template <typename T, mma::MMAMode mma_mode = mma::MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __hip_bfloat16>,
                "T must be __half or __hip_bfloat16");

  if constexpr (mma_mode == mma::MMAMode::kInit) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      C[i] = 0.0f;
    }
  }

  f32x8 C_fp32 = reinterpret_cast<f32x8*>(C)[0];
  if constexpr (std::is_same_v<T, __half>) {
    f16x8 A_fp16 = reinterpret_cast<f16x8*>(A)[0];
    f16x8 B_fp16 = reinterpret_cast<f16x8*>(B)[0];
    C_fp32 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(A_fp16, B_fp16, C_fp32);
  } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
    bf16x8 A_bf16 = reinterpret_cast<bf16x8*>(A)[0];
    bf16x8 B_bf16 = reinterpret_cast<bf16x8*>(B)[0];
    C_fp32 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(A_bf16, B_bf16, C_fp32);
  }

  reinterpret_cast<f32x8*>(C)[0] = C_fp32;
#elif defined(__HIP_DEVICE_COMPILE__) && (__gfx90a__ || __gfx908__ || __gfx942__ || __gfx950__)
  // Ensure T is either __half or __hip_bfloat16
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __hip_bfloat16>,
                "T must be __half or __hip_bfloat16");

  // Initialize C if requested
  if constexpr (mma_mode == mma::MMAMode::kInit) {
    C[0] = 0.0f;
    C[1] = 0.0f;
    C[2] = 0.0f;
    C[3] = 0.0f;
  }

  f16x4 B_fp16 = reinterpret_cast<f16x4*>(B)[0];
  f16x4 A_fp16 = reinterpret_cast<f16x4*>(A)[0];
  f32x4 C_fp32 = reinterpret_cast<f32x4*>(C)[0];

  if constexpr (std::is_same_v<T, __half>) {
    C_fp32 = __builtin_amdgcn_mfma_f32_16x16x16f16(A_fp16, B_fp16, C_fp32, 0, 0, 0);
  } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
    C_fp32 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A_fp16, B_fp16, C_fp32, 0, 0, 0);
  }

  reinterpret_cast<f32x4*>(C)[0] = C_fp32;
#elif defined(__HIP_DEVICE_COMPILE__)
#error "Unsupported GFX platform for HIP MMA ops."
#endif
}

/// @brief Loads a fragment from LDS to two 32bit registers and then transposes
/// the registers for a group of four consecuitive threads.
///
/// transposes the values in four adjacent threads. The function does the
/// following layout transformation:
/// Original data in registers for Threads 0-3 after fragment load
/// T0 : a b c d
/// T1 : e f g h
/// T2 : i j k l
/// T3 : m n o p
///
/// After transposition:
/// T0 : a e i m
/// T1 : b f j n
/// T2 : c g k o
/// T3 : d h l p
template <typename T>
__device__ __forceinline__ void load_quad_transposed_fragment(uint32_t* R, const T* smem_ptr) {
  static_assert(std::is_same_v<T, __half>, "Only half type is supported");
  load_fragment(R, smem_ptr);
  transpose_intra_quad_fragments(R);
}

// TODO: Verify correct matrix multiplication order for rowsum on CDNA3
// Current assumption: s_frag × ones_vector = row_sums
// Need to validate:
// 1. How compute_qk stores Q×K^T result in s_frag for CDNA3
// 2. Whether K is pre-transposed or transposed during fragment loading
// 3. If we need s_frag × M1 or M1 × s_frag for correct row sums
//
// Test with known input matrices to verify:
// - s_frag layout matches expected Q×K^T result
// - rowsum produces correct per-row sums
template <typename DType>
__device__ __forceinline__ void m16k16_rowsum_f16f16f32(float* d, DType* s_frag) {
  static_assert(sizeof(DType) == 2, "DType must be 16-bit type");
  static_assert(std::is_same_v<DType, __half> || std::is_same_v<DType, __hip_bfloat16>,
                "DType must be __half or __hip_bfloat16");

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
  // WMMA wave32: after transpose_mma_tile, s_frag is in A-matrix layout.
  // Use WMMA(A, ones, C) to compute row sums. C/D layout has 8 outputs per thread.
  f32x8 c = reinterpret_cast<f32x8*>(d)[0];
  f32x8 out;
  if constexpr (std::is_same_v<DType, __half>) {
    // B = all-ones 16x16 f16 matrix (packed into 4 uint32 = 8 f16)
    constexpr uint32_t fp16_one_pair = 0x3C003C00u;  // two f16 1.0 values
    f16x8 b;
    uint32_t* b_ptr = reinterpret_cast<uint32_t*>(&b);
    b_ptr[0] = fp16_one_pair; b_ptr[1] = fp16_one_pair;
    b_ptr[2] = fp16_one_pair; b_ptr[3] = fp16_one_pair;
    f16x8 a = reinterpret_cast<f16x8*>(s_frag)[0];
    out = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
  } else if constexpr (std::is_same_v<DType, __hip_bfloat16>) {
    constexpr uint32_t bf16_one_pair = 0x3F803F80u;  // two bf16 1.0 values
    bf16x8 b;
    uint32_t* b_ptr = reinterpret_cast<uint32_t*>(&b);
    b_ptr[0] = bf16_one_pair; b_ptr[1] = bf16_one_pair;
    b_ptr[2] = bf16_one_pair; b_ptr[3] = bf16_one_pair;
    bf16x8 a = reinterpret_cast<bf16x8*>(s_frag)[0];
    out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
  }
  reinterpret_cast<f32x8*>(d)[0] = out;
#elif defined(__HIP_DEVICE_COMPILE__)
  f32x4 c = {d[0], d[1], d[2], d[3]};
  f32x4 out;
  f16x4 a = reinterpret_cast<const f16x4*>(s_frag)[0];
  if constexpr (std::is_same_v<DType, __half>) {
    f16x4 b = {f16(1.0f), f16(1.0f), f16(1.0f), f16(1.0f)};
    out = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  } else if constexpr (std::is_same_v<DType, __hip_bfloat16>) {
    constexpr uint32_t bf16_one_pair = 0x3F803F80u;  // two bf16 1.0 values packed
    constexpr uint64_t bf16_ones = (uint64_t{bf16_one_pair} << 32) | bf16_one_pair;
    static_assert(sizeof(f16x4) == sizeof(bf16_ones), "f16x4 size mismatch");
    f16x4 b;
    __builtin_memcpy(&b, &bf16_ones, sizeof(f16x4));
    out = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  }
  d[0] = out.x;
  d[1] = out.y;
  d[2] = out.z;
  d[3] = out.w;
#endif
}

template <typename T, mma::MMAMode mma_mode = mma::MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* c_frag, T* a_frag,
                                                                   T* b_frag) {
  static_assert(sizeof(T) == 1, "DType must be 8-bit floating data type");

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
  if constexpr (mma_mode == mma::MMAMode::kInit) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      c_frag[i] = 0.0f;
    }
  }

  i32x2* a_i32 = reinterpret_cast<i32x2*>(a_frag);
  i32x2* b_i32 = reinterpret_cast<i32x2*>(b_frag);
  f32x8 c = reinterpret_cast<f32x8*>(c_frag)[0];

  if constexpr (std::is_same_v<T, __hip_fp8_e4m3_fnuz> || std::is_same_v<T, __hip_fp8_e4m3>) {
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_i32[0], b_i32[0], c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_i32[1], b_i32[1], c);
  } else if constexpr (std::is_same_v<T, __hip_fp8_e5m2_fnuz> || std::is_same_v<T, __hip_fp8_e5m2>) {
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a_i32[0], b_i32[0], c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a_i32[1], b_i32[1], c);
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported AMD FP8 MMA dtype");
  }

  reinterpret_cast<f32x8*>(c_frag)[0] = c;
#elif defined(__HIP_DEVICE_COMPILE__)
  FLASHINFER_RUNTIME_ASSERT("FP8 MMA is implemented for AMD gfx1201 WMMA only");
#endif
}

template <typename DType>
__device__ __forceinline__ void m16k32_rowsum_f8f8f32(float* d_frag, DType* s_frag) {
  static_assert(sizeof(DType) == 1, "DType must be 8-bit floating data type");

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
  d_frag[0] += fp8_to_float(s_frag[0]) + fp8_to_float(s_frag[1]) +
               fp8_to_float(s_frag[4]) + fp8_to_float(s_frag[5]) +
               fp8_to_float(s_frag[8]) + fp8_to_float(s_frag[9]) +
               fp8_to_float(s_frag[12]) + fp8_to_float(s_frag[13]);
  d_frag[1] += fp8_to_float(s_frag[2]) + fp8_to_float(s_frag[3]) +
               fp8_to_float(s_frag[6]) + fp8_to_float(s_frag[7]) +
               fp8_to_float(s_frag[10]) + fp8_to_float(s_frag[11]) +
               fp8_to_float(s_frag[14]) + fp8_to_float(s_frag[15]);
#elif defined(__HIP_DEVICE_COMPILE__)
  FLASHINFER_RUNTIME_ASSERT("FP8 rowsum is implemented for AMD gfx1201 only");
#endif
}

}  // namespace hip
}  // namespace mma_impl
}  // namespace gpu_iface
}  // namespace flashinfer
