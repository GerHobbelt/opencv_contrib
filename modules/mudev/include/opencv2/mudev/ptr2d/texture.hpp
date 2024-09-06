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

#ifndef OPENCV_MUDEV_PTR2D_TEXTURE_HPP
#define OPENCV_MUDEV_PTR2D_TEXTURE_HPP

#include <cstring>
#include "../common.hpp"
#include "glob.hpp"
#include "gpumat.hpp"
#include "traits.hpp"

#if 1 // MUSART_VERSION >= 5050

namespace {
template <typename T>
struct CvMudevTextureRef {
  typedef texture<T, musaTextureType2D, musaReadModeElementType> TexRef;

  static TexRef ref;

  __host__ static void bind(
      const cv::mudev::GlobPtrSz<T>& mat, bool normalizedCoords = false,
      musaTextureFilterMode filterMode = musaFilterModePoint,
      musaTextureAddressMode addressMode = musaAddressModeClamp) {
    ref.normalized = normalizedCoords;
    ref.filterMode = filterMode;
    ref.addressMode[0] = addressMode;
    ref.addressMode[1] = addressMode;
    ref.addressMode[2] = addressMode;

    musaChannelFormatDesc desc = musaCreateChannelDesc<T>();

    CV_MUDEV_SAFE_CALL(musaBindTexture2D(0, &ref, mat.data, &desc, mat.cols,
                                         mat.rows, mat.step));
  }

  __host__ static void unbind() { musaUnbindTexture(ref); }
};

template <typename T>
typename CvMudevTextureRef<T>::TexRef CvMudevTextureRef<T>::ref;
}  // namespace

#endif

namespace cv {
namespace mudev {

//! @addtogroup mudev
//! @{

#if 1 // MUSART_VERSION >= 5050

template <typename T>
struct TexturePtr {
  typedef T value_type;
  typedef float index_type;

  musaTextureObject_t texObj;

  __device__ __forceinline__ T operator()(float y, float x) const {
    // Use the texture object
    return tex2D<T>(texObj, x, y);
  }
};

template <typename T>
struct Texture : TexturePtr<T> {
  int rows, cols;

  __host__ explicit Texture(
      const GlobPtrSz<T>& mat, bool normalizedCoords = false,
      musaTextureFilterMode filterMode = musaFilterModePoint,
      musaTextureAddressMode addressMode = musaAddressModeClamp) {

    rows = mat.rows;
    cols = mat.cols;

    // Use the texture object
    musaResourceDesc texRes;
    std::memset(&texRes, 0, sizeof(texRes));
    texRes.resType = musaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = mat.data;
    texRes.res.pitch2D.height = mat.rows;
    texRes.res.pitch2D.width = mat.cols;
    texRes.res.pitch2D.pitchInBytes = mat.step;
    texRes.res.pitch2D.desc = musaCreateChannelDesc<T>();

    musaTextureDesc texDescr;
    std::memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = normalizedCoords;
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.addressMode[1] = addressMode;
    texDescr.addressMode[2] = addressMode;
    texDescr.readMode = musaReadModeElementType;

    CV_MUDEV_SAFE_CALL(
        musaCreateTextureObject(&this->texObj, &texRes, &texDescr, 0));
  }

  __host__ ~Texture() {
    // Use the texture object
    musaDestroyTextureObject(this->texObj);
  }
};

template <typename T>
struct PtrTraits<Texture<T> > : PtrTraitsBase<Texture<T>, TexturePtr<T> > {};

#else

template <typename T>
struct TexturePtr {
  typedef T value_type;
  typedef float index_type;

  musaTextureObject_t texObj;

  __device__ __forceinline__ T operator()(float y, float x) const {
    // Use the texture object
    return tex2D<T>(texObj, x, y);
  }
};

template <typename T>
struct Texture : TexturePtr<T> {
  int rows, cols;

  __host__ explicit Texture(
      const GlobPtrSz<T>& mat, bool normalizedCoords = false,
      musaTextureFilterMode filterMode = musaFilterModePoint,
      musaTextureAddressMode addressMode = musaAddressModeClamp) {
    rows = mat.rows;
    cols = mat.cols;

    // Use the texture object
    musaResourceDesc texRes;
    std::memset(&texRes, 0, sizeof(texRes));
    texRes.resType = musaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = mat.data;
    texRes.res.pitch2D.height = mat.rows;
    texRes.res.pitch2D.width = mat.cols;
    texRes.res.pitch2D.pitchInBytes = mat.step;
    texRes.res.pitch2D.desc = musaCreateChannelDesc<T>();

    musaTextureDesc texDescr;
    std::memset(&texDescr, 0, sizeof(texDescr));
    texDescr.normalizedCoords = normalizedCoords;
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.addressMode[1] = addressMode;
    texDescr.addressMode[2] = addressMode;
    texDescr.readMode = musaReadModeElementType;

    CV_MUDEV_SAFE_CALL(
        musaCreateTextureObject(&this->texObj, &texRes, &texDescr, 0));
  }

  __host__ ~Texture() {
    // Use the texture object
    musaDestroyTextureObject(this->texObj);
  }
};

template <typename T>
struct PtrTraits<Texture<T> > : PtrTraitsBase<Texture<T>, TexturePtr<T> > {};

#endif

//! @}

}  // namespace mudev
}  // namespace cv

#endif
