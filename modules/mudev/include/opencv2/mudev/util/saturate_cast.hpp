/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#pragma once

#ifndef OPENCV_MUDEV_UTIL_SATURATE_CAST_HPP
#define OPENCV_MUDEV_UTIL_SATURATE_CAST_HPP

#include "../common.hpp"
#include<algorithm>
#include <musa_fp16.h>

namespace cv {
namespace mudev {

//! @addtogroup mudev
//! @{

template <typename T>
__device__ __forceinline__ T saturate_cast(uchar v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(schar v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(ushort v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(short v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(uint v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(int v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(float v) {
  return T(v);
}
template <typename T>
__device__ __forceinline__ T saturate_cast(double v) {
  return T(v);
}

template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(schar v) {
  // uint res = 0;
  // int vi = v;
  // asm("cvt.sat.u8.s8 %0, %1;" : "=r"(res) : "r"(vi));
  // return res;
  constexpr schar lb = 0;
  return static_cast<uchar>(std::max(lb, v));
}
template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(short v) {
  // uint res = 0;
  // asm("cvt.sat.u8.s16 %0, %1;" : "=r"(res) : "h"(v));
  // return res;
  constexpr short lb = 0;
  constexpr short ub = 255;
  return static_cast<uchar>(std::min(std::max(v, lb), ub));
}
template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(ushort v) {
  // uint res = 0;
  // asm("cvt.sat.u8.u16 %0, %1;" : "=r"(res) : "h"(v));
  // return res;
  constexpr ushort ub = 255;
  return static_cast<uchar>(std::min(ub, v));
}
template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(int v) {
  // uint res = 0;
  // asm("cvt.sat.u8.s32 %0, %1;" : "=r"(res) : "r"(v));
  // return res;
  return static_cast<uchar>(std::min(std::max(v, 0), 255));
}
template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(uint v) {
  // uint res = 0;
  // asm("cvt.sat.u8.u32 %0, %1;" : "=r"(res) : "r"(v));
  // return res;
  constexpr uint ub = 255;
  return static_cast<uchar>(std::min(ub, v));
}
template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(float v) {
  // uint res = 0;
  // asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(v));
  // return res; 
  constexpr float round_val = 0.5f;
  return __float2uchar_rz_sat(v + round_val);
}
template <>
__device__ __forceinline__ uchar saturate_cast<uchar>(double v) {
  // uint res = 0;
  // asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(res) : "d"(v));
  // return res;
  return saturate_cast<uchar>(static_cast<float>(v));
}

template <>
__device__ __forceinline__ schar saturate_cast<schar>(uchar v) {
  // uint res = 0;
  // uint vi = v;
  // asm("cvt.sat.s8.u8 %0, %1;" : "=r"(res) : "r"(vi));
  // return res;
  constexpr uchar ub = 127;
  return static_cast<schar>(std::min(v, ub));
}
template <>
__device__ __forceinline__ schar saturate_cast<schar>(short v) {
  // uint res = 0;
  // asm("cvt.sat.s8.s16 %0, %1;" : "=r"(res) : "h"(v));
  // return res;
  constexpr short lb = -128;
  constexpr short ub = 127;
  return static_cast<schar>(std::min(std::max(v, lb), ub));
}
template <>
__device__ __forceinline__ schar saturate_cast<schar>(ushort v) {
  // uint res = 0;
  // asm("cvt.sat.s8.u16 %0, %1;" : "=r"(res) : "h"(v));
  // return res;
  constexpr ushort ub = 127;
  return static_cast<schar>(std::min(v, ub));
}
template <>
__device__ __forceinline__ schar saturate_cast<schar>(int v) {
  // uint res = 0;
  // asm("cvt.sat.s8.s32 %0, %1;" : "=r"(res) : "r"(v));
  // return res;
  return static_cast<schar>(std::min(std::max(v, -128), 127));
}
template <>
__device__ __forceinline__ schar saturate_cast<schar>(uint v) {
  // uint res = 0;
  // asm("cvt.sat.s8.u32 %0, %1;" : "=r"(res) : "r"(v));
  // return res;
  constexpr uint ub = 127;
  return static_cast<schar>(std::min(v, ub));
}
template <>
__device__ __forceinline__ schar saturate_cast<schar>(float v) {
  // uint res = 0;
  // asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(v));
  // return res;
  float round_val = v >= 0.0f ? 0.5f : -0.5f;
  return __float2char_rz_sat(v + round_val);
}
template <>
__device__ __forceinline__ schar saturate_cast<schar>(double v) {
  // uint res = 0;
  // asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(res) : "d"(v));
  // return res;
  return saturate_cast<schar>(static_cast<float>(v));
}

template <>
__device__ __forceinline__ ushort saturate_cast<ushort>(schar v) {
  // ushort res = 0;
  // int vi = v;
  // asm("cvt.sat.u16.s8 %0, %1;" : "=h"(res) : "r"(vi));
  // return res;
  constexpr schar lb = 0;
  return static_cast<ushort>(std::max(v, lb));
}
template <>
__device__ __forceinline__ ushort saturate_cast<ushort>(short v) {
  // ushort res = 0;
  // asm("cvt.sat.u16.s16 %0, %1;" : "=h"(res) : "h"(v));
  // return res;
  constexpr short lb = 0;
  return static_cast<ushort>(std::max(v, lb));
}
template <>
__device__ __forceinline__ ushort saturate_cast<ushort>(int v) {
  // ushort res = 0;
  // asm("cvt.sat.u16.s32 %0, %1;" : "=h"(res) : "r"(v));
  // return res;
  return static_cast<ushort>(std::min(std::max(v, 0), 65535));
}
template <>
__device__ __forceinline__ ushort saturate_cast<ushort>(uint v) {
  // ushort res = 0;
  // asm("cvt.sat.u16.u32 %0, %1;" : "=h"(res) : "r"(v));
  // return res;
  constexpr uint ub = 65535;
  return static_cast<ushort>(std::min(v, ub));
}
template <>
__device__ __forceinline__ ushort saturate_cast<ushort>(float v) {
  // ushort res = 0;
  // asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(res) : "f"(v));
  // return res;

  // return __float2ushort_rz_sat(v);
  constexpr float lb = 0.0f;
  constexpr float ub = 65535.0f;
  constexpr float round_val = 0.5f;
  return static_cast<ushort>(std::min(std::max(v + round_val, lb), ub));
}
template <>
__device__ __forceinline__ ushort saturate_cast<ushort>(double v) {
  // ushort res = 0;
  // asm("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(res) : "d"(v));
  // return res;
  return saturate_cast<ushort>(static_cast<float>(v));
}

template <>
__device__ __forceinline__ short saturate_cast<short>(ushort v) {
  // short res = 0;
  // asm("cvt.sat.s16.u16 %0, %1;" : "=h"(res) : "h"(v));
  // return res;
  constexpr ushort ub = 32767;
  return static_cast<short>(std::min(v, ub));
}
template <>
__device__ __forceinline__ short saturate_cast<short>(int v) {
  // short res = 0;
  // asm("cvt.sat.s16.s32 %0, %1;" : "=h"(res) : "r"(v));
  // return res;
  constexpr int lb = -32768;
  constexpr int ub = 32767;
  return static_cast<short>(std::min(std::max(v, lb), ub));
}
template <>
__device__ __forceinline__ short saturate_cast<short>(uint v) {
  // short res = 0;
  // asm("cvt.sat.s16.u32 %0, %1;" : "=h"(res) : "r"(v));
  // return res;
  constexpr uint ub = 32767;
  return static_cast<short>(std::min(v, ub));
}
template <>
__device__ __forceinline__ short saturate_cast<short>(float v) {
  // short res = 0;
  // asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(res) : "f"(v));
  // return res;

  // return __float2short_rz_sat(v);
  constexpr float lb = -32768.0f;
  constexpr float ub = 32767.0f;
  float round_val = v >= 0.0f ? 0.5f : -0.5f;
  return static_cast<short>(std::min(std::max(v + round_val, lb), ub));
}
template <>
__device__ __forceinline__ short saturate_cast<short>(double v) {
  // short res = 0;
  // asm("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(res) : "d"(v));
  // return res;
  return saturate_cast<short>(static_cast<float>(v));
}

template <>
__device__ __forceinline__ int saturate_cast<int>(uint v) {
  // int res = 0;
  // asm("cvt.sat.s32.u32 %0, %1;" : "=r"(res) : "r"(v));
  // return res;
  constexpr uint ub = 2147483647;
  return static_cast<int>(std::min(v, ub));
}
template <>
__device__ __forceinline__ int saturate_cast<int>(float v) {
  return __float2int_rn(v);
}
template <>
__device__ __forceinline__ int saturate_cast<int>(double v) {
  // return __double2int_rn(v);
  return saturate_cast<int>(static_cast<float>(v));
}

template <>
__device__ __forceinline__ uint saturate_cast<uint>(schar v) {
  // uint res = 0;
  // int vi = v;
  // asm("cvt.sat.u32.s8 %0, %1;" : "=r"(res) : "r"(vi));
  // return res;
  constexpr schar lb = 0;
  return static_cast<uint>(std::max(v, lb));
}
template <>
__device__ __forceinline__ uint saturate_cast<uint>(short v) {
  // uint res = 0;
  // asm("cvt.sat.u32.s16 %0, %1;" : "=r"(res) : "h"(v));
  // return res;
  constexpr short lb = 0;
  return static_cast<uint>(std::max(v, lb));
}
template <>
__device__ __forceinline__ uint saturate_cast<uint>(int v) {
  // uint res = 0;
  // asm("cvt.sat.u32.s32 %0, %1;" : "=r"(res) : "r"(v));
  // return res;
  constexpr int lb = 0;
  return static_cast<uint>(std::max(v, lb));
}
template <>
__device__ __forceinline__ uint saturate_cast<uint>(float v) {
  return __float2uint_rn(v);
}
template <>
__device__ __forceinline__ uint saturate_cast<uint>(double v) {
  // return __double2uint_rn(v);
  return saturate_cast<uint>(static_cast<float>(v));
}

template <typename T, typename D>
__device__ __forceinline__ D cast_fp16(T v);

template <>
__device__ __forceinline__ float cast_fp16<short, float>(short v) {
  // #if __MUSACC_VER_MAJOR__ >= 9
  return float(*(__half*)&v);
  // #else
  //   return __half2float(v);
  // #endif
}

template <>
__device__ __forceinline__ short cast_fp16<float, short>(float v) {
  // #if __MUSACC_VER_MAJOR__ >= 9
  __half h(v);
  return *(short*)&h;
  // #else
  //   return (short)__float2half_rn(v);
  // #endif
}
//! @}

}  // namespace mudev
}  // namespace cv

#endif
