/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_MUDEV

#error "opencv_mudev is required"

#else

#include "../lut.hpp"

#include "opencv2/musaarithm.hpp"
#include "opencv2/mudev.hpp"
#include "opencv2/core/private.musa.hpp"

using namespace cv;
using namespace cv::musa;
using namespace cv::mudev;

namespace cv {
namespace musa {

texture<uchar, musaTextureType1D, musaReadModeElementType> texLutTable;

LookUpTableImpl::LookUpTableImpl(InputArray _lut) {
  if (_lut.kind() == _InputArray::MUSA_GPU_MAT) {
    d_lut = _lut.getMUSAGpuMat();
  } else {
    Mat h_lut = _lut.getMat();
    d_lut.upload(Mat(1, 256, h_lut.type(), h_lut.data));
  }

  CV_Assert(d_lut.depth() == CV_8U);
  CV_Assert(d_lut.rows == 1 && d_lut.cols == 256);

  // Use the texture object
  musaResourceDesc texRes;
  std::memset(&texRes, 0, sizeof(texRes));
  texRes.resType = musaResourceTypeLinear;
  texRes.res.linear.devPtr = d_lut.data;
  texRes.res.linear.desc = musaCreateChannelDesc<uchar>();
  texRes.res.linear.sizeInBytes = 256 * d_lut.channels() * sizeof(uchar);

  musaTextureDesc texDescr;
  std::memset(&texDescr, 0, sizeof(texDescr));

  CV_MUDEV_SAFE_CALL(
      musaCreateTextureObject(&texLutTableObj, &texRes, &texDescr, 0));
}

LookUpTableImpl::~LookUpTableImpl() {
  // Use the texture object
  musaDestroyTextureObject(texLutTableObj);
}

struct LutTablePtrC1 {
  typedef uchar value_type;
  typedef uchar index_type;

  musaTextureObject_t texLutTableObj;

  __device__ __forceinline__ uchar operator()(uchar, uchar x) const {
    // Use the texture object
    return tex1Dfetch<uchar>(texLutTableObj, x);
  }
};
struct LutTablePtrC3 {
  typedef uchar3 value_type;
  typedef uchar3 index_type;

  musaTextureObject_t texLutTableObj;

  __device__ __forceinline__ uchar3 operator()(const uchar3&,
                                               const uchar3& x) const {
    // Use the texture object
    return make_uchar3(tex1Dfetch<uchar>(texLutTableObj, x.x * 3),
                       tex1Dfetch<uchar>(texLutTableObj, x.y * 3 + 1),
                       tex1Dfetch<uchar>(texLutTableObj, x.z * 3 + 2));
  }
};

void LookUpTableImpl::transform(InputArray _src, OutputArray _dst,
                                Stream& stream) {
  GpuMat src = getInputMat(_src, stream);

  const int cn = src.channels();
  const int lut_cn = d_lut.channels();

  CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
  CV_Assert(lut_cn == 1 || lut_cn == cn);

  GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

  if (lut_cn == 1) {
    GpuMat_<uchar> src1(src.reshape(1));
    GpuMat_<uchar> dst1(dst.reshape(1));

    LutTablePtrC1 tbl;
    tbl.texLutTableObj = texLutTableObj;

    dst1.assign(lut_(src1, tbl), stream);
  } else if (lut_cn == 3) {
    GpuMat_<uchar3>& src3 = (GpuMat_<uchar3>&)src;
    GpuMat_<uchar3>& dst3 = (GpuMat_<uchar3>&)dst;

    LutTablePtrC3 tbl;
    tbl.texLutTableObj = texLutTableObj;

    dst3.assign(lut_(src3, tbl), stream);
  }

  syncOutput(dst, _dst, stream);
}

}  // namespace musa
}  // namespace cv

#endif
