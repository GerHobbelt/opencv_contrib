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

#ifndef OPENCV_MUDEV_WARP_SHUFFLE_HPP
#define OPENCV_MUDEV_WARP_SHUFFLE_HPP

#include "../common.hpp"
#include "../util/vec_traits.hpp"
#include "../block/block.hpp"
#include "warp.hpp"

namespace cv {
namespace mudev {

//! @addtogroup mudev
//! @{


#define __shfl(x, y, z) __shfl_sync(0xFFFFFFFFU, x, y, z)
#define __shfl_xor(x, y, z) __shfl_xor_sync(0xFFFFFFFFU, x, y, z)
#define __shfl_down(x, y, z) __shfl_down_sync(0xFFFFFFFFU, x, y, z)

// shfl
__device__ __forceinline__ uchar shfl(uchar val, int srcLane,
                                      int width = warpSize) {
  return (uchar)__shfl((int)val, srcLane, width);
}

__device__ __forceinline__ schar shfl(schar val, int srcLane,
                                      int width = warpSize) {
  return (schar)__shfl((int)val, srcLane, width);
}

__device__ __forceinline__ ushort shfl(ushort val, int srcLane,
                                       int width = warpSize) {
  return (ushort)__shfl((int)val, srcLane, width);
}

__device__ __forceinline__ short shfl(short val, int srcLane,
                                      int width = warpSize) {
  return (short)__shfl((int)val, srcLane, width);
}

__device__ __forceinline__ int shfl(int val, int srcLane,
                                    int width = warpSize) {
  return __shfl(val, srcLane, width);
}

__device__ __forceinline__ uint shfl(uint val, int srcLane,
                                     int width = warpSize) {
  return (uint)__shfl((int)val, srcLane, width);
}

__device__ __forceinline__ float shfl(float val, int srcLane,
                                      int width = warpSize) {
  return __shfl(val, srcLane, width);
}

__device__ double shfl(double val, int srcLane, int width = warpSize) {
  int lo = __double2loint(val);
  int hi = __double2hiint(val);

  lo = __shfl(lo, srcLane, width);
  hi = __shfl(hi, srcLane, width);

  return __hiloint2double(hi, lo);
}

#define CV_MUDEV_SHFL_VEC_INST(input_type)                              \
  __device__ __forceinline__ input_type##1 shfl(                        \
      const input_type##1 & val, int srcLane, int width = warpSize) {   \
    return VecTraits<input_type##1>::make(shfl(val.x, srcLane, width)); \
  }                                                                     \
  __device__ __forceinline__ input_type##2 shfl(                        \
      const input_type##2 & val, int srcLane, int width = warpSize) {   \
    return VecTraits<input_type##2>::make(shfl(val.x, srcLane, width),  \
                                          shfl(val.y, srcLane, width)); \
  }                                                                     \
  __device__ __forceinline__ input_type##3 shfl(                        \
      const input_type##3 & val, int srcLane, int width = warpSize) {   \
    return VecTraits<input_type##3>::make(shfl(val.x, srcLane, width),  \
                                          shfl(val.y, srcLane, width),  \
                                          shfl(val.z, srcLane, width)); \
  }                                                                     \
  __device__ __forceinline__ input_type##4 shfl(                        \
      const input_type##4 & val, int srcLane, int width = warpSize) {   \
    return VecTraits<input_type##4>::make(                              \
        shfl(val.x, srcLane, width), shfl(val.y, srcLane, width),       \
        shfl(val.z, srcLane, width), shfl(val.w, srcLane, width));      \
  }

CV_MUDEV_SHFL_VEC_INST(uchar)
CV_MUDEV_SHFL_VEC_INST(char)
CV_MUDEV_SHFL_VEC_INST(ushort)
CV_MUDEV_SHFL_VEC_INST(short)
CV_MUDEV_SHFL_VEC_INST(uint)
CV_MUDEV_SHFL_VEC_INST(int)
CV_MUDEV_SHFL_VEC_INST(float)
CV_MUDEV_SHFL_VEC_INST(double)

#undef CV_MUDEV_SHFL_VEC_INST

// shfl_up

template <typename T>
__device__ __forceinline__ T shfl_up_sync(uint mask, T val, uint delta,
                                          int width = warpSize) {
  return (T)__shfl_up_sync(mask, val, delta, width);
}

template <typename T>
__device__ __forceinline__ T compatible_shfl_up(T val, uint delta,
                                                int width = warpSize) {
  return shfl_up_sync(0xFFFFFFFFU, val, delta, width);
}

// shfl_down

__device__ __forceinline__ uchar shfl_down(uchar val, uint delta,
                                           int width = warpSize) {
  return (uchar)__shfl_down((int)val, delta, width);
}

__device__ __forceinline__ schar shfl_down(schar val, uint delta,
                                           int width = warpSize) {
  return (schar)__shfl_down((int)val, delta, width);
}

__device__ __forceinline__ ushort shfl_down(ushort val, uint delta,
                                            int width = warpSize) {
  return (ushort)__shfl_down((int)val, delta, width);
}

__device__ __forceinline__ short shfl_down(short val, uint delta,
                                           int width = warpSize) {
  return (short)__shfl_down((int)val, delta, width);
}

__device__ __forceinline__ int shfl_down(int val, uint delta,
                                         int width = warpSize) {
  return __shfl_down(val, delta, width);
}

__device__ __forceinline__ uint shfl_down(uint val, uint delta,
                                          int width = warpSize) {
  return (uint)__shfl_down((int)val, delta, width);
}

__device__ __forceinline__ float shfl_down(float val, uint delta,
                                           int width = warpSize) {
  return __shfl_down(val, delta, width);
}

__device__ double shfl_down(double val, uint delta, int width = warpSize) {
  int lo = __double2loint(val);
  int hi = __double2hiint(val);

  lo = __shfl_down(lo, delta, width);
  hi = __shfl_down(hi, delta, width);

  return __hiloint2double(hi, lo);
}

#define CV_MUDEV_SHFL_DOWN_VEC_INST(input_type)                            \
  __device__ __forceinline__ input_type##1 shfl_down(                      \
      const input_type##1 & val, uint delta, int width = warpSize) {       \
    return VecTraits<input_type##1>::make(shfl_down(val.x, delta, width)); \
  }                                                                        \
  __device__ __forceinline__ input_type##2 shfl_down(                      \
      const input_type##2 & val, uint delta, int width = warpSize) {       \
    return VecTraits<input_type##2>::make(shfl_down(val.x, delta, width),  \
                                          shfl_down(val.y, delta, width)); \
  }                                                                        \
  __device__ __forceinline__ input_type##3 shfl_down(                      \
      const input_type##3 & val, uint delta, int width = warpSize) {       \
    return VecTraits<input_type##3>::make(shfl_down(val.x, delta, width),  \
                                          shfl_down(val.y, delta, width),  \
                                          shfl_down(val.z, delta, width)); \
  }                                                                        \
  __device__ __forceinline__ input_type##4 shfl_down(                      \
      const input_type##4 & val, uint delta, int width = warpSize) {       \
    return VecTraits<input_type##4>::make(                                 \
        shfl_down(val.x, delta, width), shfl_down(val.y, delta, width),    \
        shfl_down(val.z, delta, width), shfl_down(val.w, delta, width));   \
  }

CV_MUDEV_SHFL_DOWN_VEC_INST(uchar)
CV_MUDEV_SHFL_DOWN_VEC_INST(char)
CV_MUDEV_SHFL_DOWN_VEC_INST(ushort)
CV_MUDEV_SHFL_DOWN_VEC_INST(short)
CV_MUDEV_SHFL_DOWN_VEC_INST(uint)
CV_MUDEV_SHFL_DOWN_VEC_INST(int)
CV_MUDEV_SHFL_DOWN_VEC_INST(float)
CV_MUDEV_SHFL_DOWN_VEC_INST(double)

#undef CV_MUDEV_SHFL_DOWN_VEC_INST

// shfl_xor

__device__ __forceinline__ uchar shfl_xor(uchar val, int laneMask,
                                          int width = warpSize) {
  return (uchar)__shfl_xor((int)val, laneMask, width);
}

__device__ __forceinline__ schar shfl_xor(schar val, int laneMask,
                                          int width = warpSize) {
  return (schar)__shfl_xor((int)val, laneMask, width);
}

__device__ __forceinline__ ushort shfl_xor(ushort val, int laneMask,
                                           int width = warpSize) {
  return (ushort)__shfl_xor((int)val, laneMask, width);
}

__device__ __forceinline__ short shfl_xor(short val, int laneMask,
                                          int width = warpSize) {
  return (short)__shfl_xor((int)val, laneMask, width);
}

__device__ __forceinline__ int shfl_xor(int val, int laneMask,
                                        int width = warpSize) {
  return __shfl_xor(val, laneMask, width);
}

__device__ __forceinline__ uint shfl_xor(uint val, int laneMask,
                                         int width = warpSize) {
  return (uint)__shfl_xor((int)val, laneMask, width);
}

__device__ __forceinline__ float shfl_xor(float val, int laneMask,
                                          int width = warpSize) {
  return __shfl_xor(val, laneMask, width);
}

__device__ double shfl_xor(double val, int laneMask, int width = warpSize) {
  int lo = __double2loint(val);
  int hi = __double2hiint(val);

  lo = __shfl_xor(lo, laneMask, width);
  hi = __shfl_xor(hi, laneMask, width);

  return __hiloint2double(hi, lo);
}

#define CV_MUDEV_SHFL_XOR_VEC_INST(input_type)                               \
  __device__ __forceinline__ input_type##1 shfl_xor(                         \
      const input_type##1 & val, int laneMask, int width = warpSize) {       \
    return VecTraits<input_type##1>::make(shfl_xor(val.x, laneMask, width)); \
  }                                                                          \
  __device__ __forceinline__ input_type##2 shfl_xor(                         \
      const input_type##2 & val, int laneMask, int width = warpSize) {       \
    return VecTraits<input_type##2>::make(shfl_xor(val.x, laneMask, width),  \
                                          shfl_xor(val.y, laneMask, width)); \
  }                                                                          \
  __device__ __forceinline__ input_type##3 shfl_xor(                         \
      const input_type##3 & val, int laneMask, int width = warpSize) {       \
    return VecTraits<input_type##3>::make(shfl_xor(val.x, laneMask, width),  \
                                          shfl_xor(val.y, laneMask, width),  \
                                          shfl_xor(val.z, laneMask, width)); \
  }                                                                          \
  __device__ __forceinline__ input_type##4 shfl_xor(                         \
      const input_type##4 & val, int laneMask, int width = warpSize) {       \
    return VecTraits<input_type##4>::make(                                   \
        shfl_xor(val.x, laneMask, width), shfl_xor(val.y, laneMask, width),  \
        shfl_xor(val.z, laneMask, width), shfl_xor(val.w, laneMask, width)); \
  }

CV_MUDEV_SHFL_XOR_VEC_INST(uchar)
CV_MUDEV_SHFL_XOR_VEC_INST(char)
CV_MUDEV_SHFL_XOR_VEC_INST(ushort)
CV_MUDEV_SHFL_XOR_VEC_INST(short)
CV_MUDEV_SHFL_XOR_VEC_INST(uint)
CV_MUDEV_SHFL_XOR_VEC_INST(int)
CV_MUDEV_SHFL_XOR_VEC_INST(float)
CV_MUDEV_SHFL_XOR_VEC_INST(double)

#undef CV_MUDEV_SHFL_XOR_VEC_INST
#undef __shfl
#undef __shfl_xor
#undef __shfl_up
#undef __shfl_down

//! @}
}  // namespace mudev
}  // namespace cv

#endif
