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

#include "precomp.hpp"

using namespace cv;
using namespace cv::musa;

#if !defined(HAVE_MUSA) || defined(MUSA_DISABLER)

void cv::musa::add(InputArray, InputArray, OutputArray, InputArray, int,
                   Stream&) {
  throw_no_musa();
}
void cv::musa::subtract(InputArray, InputArray, OutputArray, InputArray, int,
                        Stream&) {
  throw_no_musa();
}
void cv::musa::multiply(InputArray, InputArray, OutputArray, double, int,
                        Stream&) {
  throw_no_musa();
}
void cv::musa::divide(InputArray, InputArray, OutputArray, double, int,
                      Stream&) {
  throw_no_musa();
}
void cv::musa::absdiff(InputArray, InputArray, OutputArray, Stream&) {
  throw_no_musa();
}

void cv::musa::abs(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::sqr(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::sqrt(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::exp(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::log(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::pow(InputArray, double, OutputArray, Stream&) {
  throw_no_musa();
}

void cv::musa::compare(InputArray, InputArray, OutputArray, int, Stream&) {
  throw_no_musa();
}

void cv::musa::bitwise_not(InputArray, OutputArray, InputArray, Stream&) {
  throw_no_musa();
}
void cv::musa::bitwise_or(InputArray, InputArray, OutputArray, InputArray,
                          Stream&) {
  throw_no_musa();
}
void cv::musa::bitwise_and(InputArray, InputArray, OutputArray, InputArray,
                           Stream&) {
  throw_no_musa();
}
void cv::musa::bitwise_xor(InputArray, InputArray, OutputArray, InputArray,
                           Stream&) {
  throw_no_musa();
}

void cv::musa::rshift(InputArray, Scalar_<int>, OutputArray, Stream&) {
  throw_no_musa();
}
void cv::musa::lshift(InputArray, Scalar_<int>, OutputArray, Stream&) {
  throw_no_musa();
}

void cv::musa::min(InputArray, InputArray, OutputArray, Stream&) {
  throw_no_musa();
}
void cv::musa::max(InputArray, InputArray, OutputArray, Stream&) {
  throw_no_musa();
}

void cv::musa::addWeighted(InputArray, double, InputArray, double, double,
                           OutputArray, int, Stream&) {
  throw_no_musa();
}

double cv::musa::threshold(InputArray, OutputArray, double, double, int,
                           Stream&) {
  throw_no_musa();
  return 0.0;
}

void cv::musa::inRange(InputArray, const Scalar&, const Scalar&, OutputArray,
                       Stream&) {
  throw_no_musa();
}

void cv::musa::magnitude(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::magnitude(InputArray, InputArray, OutputArray, Stream&) {
  throw_no_musa();
}
void cv::musa::magnitudeSqr(InputArray, OutputArray, Stream&) {
  throw_no_musa();
}
void cv::musa::magnitudeSqr(InputArray, InputArray, OutputArray, Stream&) {
  throw_no_musa();
}
void cv::musa::phase(InputArray, InputArray, OutputArray, bool, Stream&) {
  throw_no_musa();
}
void cv::musa::cartToPolar(InputArray, InputArray, OutputArray, OutputArray,
                           bool, Stream&) {
  throw_no_musa();
}
void cv::musa::polarToCart(InputArray, InputArray, OutputArray, OutputArray,
                           bool, Stream&) {
  throw_no_musa();
}

#else

////////////////////////////////////////////////////////////////////////
// arithm_op

namespace {
typedef void (*mat_mat_func_t)(const GpuMat& src1, const GpuMat& src2,
                               GpuMat& dst, const GpuMat& mask, double scale,
                               Stream& stream, int op);
typedef void (*mat_scalar_func_t)(const GpuMat& src, Scalar val, bool inv,
                                  GpuMat& dst, const GpuMat& mask, double scale,
                                  Stream& stream, int op);

void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst,
               InputArray _mask, double scale, int dtype, Stream& stream,
               mat_mat_func_t mat_mat_func, mat_scalar_func_t mat_scalar_func,
               int op = 0) {
  const int kind1 = _src1.kind();
  const int kind2 = _src2.kind();

  const bool isScalar1 = (kind1 == _InputArray::MATX);
  const bool isScalar2 = (kind2 == _InputArray::MATX);
  CV_Assert(!isScalar1 || !isScalar2);

  GpuMat src1;
  if (!isScalar1) src1 = getInputMat(_src1, stream);

  GpuMat src2;
  if (!isScalar2) src2 = getInputMat(_src2, stream);

  Mat scalar;
  if (isScalar1)
    scalar = _src1.getMat();
  else if (isScalar2)
    scalar = _src2.getMat();

  Scalar val;
  if (!scalar.empty()) {
    CV_Assert(scalar.total() <= 4);
    scalar.convertTo(Mat_<double>(scalar.rows, scalar.cols, &val[0]), CV_64F);
  }

  GpuMat mask = getInputMat(_mask, stream);

  const int sdepth = src1.empty() ? src2.depth() : src1.depth();
  const int cn = src1.empty() ? src2.channels() : src1.channels();
  const Size size = src1.empty() ? src2.size() : src1.size();

  if (dtype < 0) dtype = sdepth;

  const int ddepth = CV_MAT_DEPTH(dtype);

  CV_Assert(sdepth <= CV_64F && ddepth <= CV_64F);
  CV_Assert(!scalar.empty() ||
            (src2.type() == src1.type() && src2.size() == src1.size()));
  CV_Assert(mask.empty() ||
            (cn == 1 && mask.size() == size && mask.type() == CV_8UC1));

  if (sdepth == CV_64F || ddepth == CV_64F) {
    if (!deviceSupports(NATIVE_DOUBLE))
      CV_Error(Error::StsUnsupportedFormat,
               "The device doesn't support double");
  }

  GpuMat dst = getOutputMat(_dst, size, CV_MAKE_TYPE(ddepth, cn), stream);

  if (isScalar1)
    mat_scalar_func(src2, val, true, dst, mask, scale, stream, op);
  else if (isScalar2)
    mat_scalar_func(src1, val, false, dst, mask, scale, stream, op);
  else
    mat_mat_func(src1, src2, dst, mask, scale, stream, op);

  syncOutput(dst, _dst, stream);
}
}  // namespace

////////////////////////////////////////////////////////////////////////
// add

void addMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
            const GpuMat& mask, double, Stream& _stream, int);

void addScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst,
               const GpuMat& mask, double, Stream& stream, int);

void cv::musa::add(InputArray src1, InputArray src2, OutputArray dst,
                   InputArray mask, int dtype, Stream& stream) {
  arithm_op(src1, src2, dst, mask, 1.0, dtype, stream, addMat, addScalar);
}

////////////////////////////////////////////////////////////////////////
// subtract

void subMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
            const GpuMat& mask, double, Stream& _stream, int);

void subScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst,
               const GpuMat& mask, double, Stream& stream, int);

void cv::musa::subtract(InputArray src1, InputArray src2, OutputArray dst,
                        InputArray mask, int dtype, Stream& stream) {
  arithm_op(src1, src2, dst, mask, 1.0, dtype, stream, subMat, subScalar);
}

////////////////////////////////////////////////////////////////////////
// multiply

void mulMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&,
            double scale, Stream& stream, int);
void mulMat_8uc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                     Stream& stream);
void mulMat_16sc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                      Stream& stream);

void mulScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst,
               const GpuMat& mask, double scale, Stream& stream, int);

void cv::musa::multiply(InputArray _src1, InputArray _src2, OutputArray _dst,
                        double scale, int dtype, Stream& stream) {
  if (_src1.type() == CV_8UC4 && _src2.type() == CV_32FC1) {
    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert(src1.size() == src2.size());

    GpuMat dst = getOutputMat(_dst, src1.size(), src1.type(), stream);

    mulMat_8uc4_32f(src1, src2, dst, stream);

    syncOutput(dst, _dst, stream);
  } else if (_src1.type() == CV_16SC4 && _src2.type() == CV_32FC1) {
    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert(src1.size() == src2.size());

    GpuMat dst = getOutputMat(_dst, src1.size(), src1.type(), stream);

    mulMat_16sc4_32f(src1, src2, dst, stream);

    syncOutput(dst, _dst, stream);
  } else {
    arithm_op(_src1, _src2, _dst, GpuMat(), scale, dtype, stream, mulMat,
              mulScalar);
  }
}

////////////////////////////////////////////////////////////////////////
// divide

void divMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&,
            double scale, Stream& stream, int);
void divMat_8uc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                     Stream& stream);
void divMat_16sc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                      Stream& stream);

void divScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst,
               const GpuMat& mask, double scale, Stream& stream, int);

void cv::musa::divide(InputArray _src1, InputArray _src2, OutputArray _dst,
                      double scale, int dtype, Stream& stream) {
  if (_src1.type() == CV_8UC4 && _src2.type() == CV_32FC1) {
    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert(src1.size() == src2.size());

    GpuMat dst = getOutputMat(_dst, src1.size(), src1.type(), stream);

    divMat_8uc4_32f(src1, src2, dst, stream);

    syncOutput(dst, _dst, stream);
  } else if (_src1.type() == CV_16SC4 && _src2.type() == CV_32FC1) {
    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert(src1.size() == src2.size());

    GpuMat dst = getOutputMat(_dst, src1.size(), src1.type(), stream);

    divMat_16sc4_32f(src1, src2, dst, stream);

    syncOutput(dst, _dst, stream);
  } else {
    arithm_op(_src1, _src2, _dst, GpuMat(), scale, dtype, stream, divMat,
              divScalar);
  }
}

//////////////////////////////////////////////////////////////////////////////
// absdiff

void absDiffMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                const GpuMat&, double, Stream& stream, int);

void absDiffScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst,
                   const GpuMat&, double, Stream& stream, int);

void cv::musa::absdiff(InputArray src1, InputArray src2, OutputArray dst,
                       Stream& stream) {
  arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, absDiffMat,
            absDiffScalar);
}

//////////////////////////////////////////////////////////////////////////////
// compare

void cmpMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&,
            double, Stream& stream, int cmpop);

void cmpScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst,
               const GpuMat&, double, Stream& stream, int cmpop);

void cv::musa::compare(InputArray src1, InputArray src2, OutputArray dst,
                       int cmpop, Stream& stream) {
  arithm_op(src1, src2, dst, noArray(), 1.0, CV_8U, stream, cmpMat, cmpScalar,
            cmpop);
}

//////////////////////////////////////////////////////////////////////////////
// Binary bitwise logical operations

namespace {
enum { BIT_OP_AND, BIT_OP_OR, BIT_OP_XOR };
}

void bitMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
            const GpuMat& mask, double, Stream& stream, int op);

void bitScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst,
               const GpuMat& mask, double, Stream& stream, int op);

void cv::musa::bitwise_or(InputArray src1, InputArray src2, OutputArray dst,
                          InputArray mask, Stream& stream) {
  arithm_op(src1, src2, dst, mask, 1.0, -1, stream, bitMat, bitScalar,
            BIT_OP_OR);
}

void cv::musa::bitwise_and(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask, Stream& stream) {
  arithm_op(src1, src2, dst, mask, 1.0, -1, stream, bitMat, bitScalar,
            BIT_OP_AND);
}

void cv::musa::bitwise_xor(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask, Stream& stream) {
  arithm_op(src1, src2, dst, mask, 1.0, -1, stream, bitMat, bitScalar,
            BIT_OP_XOR);
}

//////////////////////////////////////////////////////////////////////////////
// shift

namespace {
template <int DEPTH, int cn>
struct MUppShiftFunc {
  typedef typename MUPPTypeTraits<DEPTH>::mupp_type mupp_type;

  typedef MUppStatus (*func_t)(const mupp_type* pSrc1, int nSrc1Step,
                               const MUpp32u* pConstants, mupp_type* pDst,
                               int nDstStep, MUppiSize oSizeROI);
};
template <int DEPTH>
struct MUppShiftFunc<DEPTH, 1> {
  typedef typename MUPPTypeTraits<DEPTH>::mupp_type mupp_type;

  typedef MUppStatus (*func_t)(const mupp_type* pSrc1, int nSrc1Step,
                               const MUpp32u pConstants, mupp_type* pDst,
                               int nDstStep, MUppiSize oSizeROI);
};

template <int DEPTH, int cn, typename MUppShiftFunc<DEPTH, cn>::func_t func>
struct MUppShift {
  typedef typename MUPPTypeTraits<DEPTH>::mupp_type mupp_type;

  static void call(const GpuMat& src, Scalar_<MUpp32u> sc, GpuMat& dst,
                   musaStream_t stream) {
    MUppStreamHandler h(stream);

    MUppiSize oSizeROI;
    oSizeROI.width = src.cols;
    oSizeROI.height = src.rows;

    muppSafeCall(func(src.ptr<mupp_type>(), static_cast<int>(src.step), sc.val,
                      dst.ptr<mupp_type>(), static_cast<int>(dst.step),
                      oSizeROI));

    if (stream == 0) musaSafeCall(musaDeviceSynchronize());
  }
};
template <int DEPTH, typename MUppShiftFunc<DEPTH, 1>::func_t func>
struct MUppShift<DEPTH, 1, func> {
  typedef typename MUPPTypeTraits<DEPTH>::mupp_type mupp_type;

  static void call(const GpuMat& src, Scalar_<MUpp32u> sc, GpuMat& dst,
                   musaStream_t stream) {
    MUppStreamHandler h(stream);

    MUppiSize oSizeROI;
    oSizeROI.width = src.cols;
    oSizeROI.height = src.rows;

    muppSafeCall(func(src.ptr<mupp_type>(), static_cast<int>(src.step),
                      sc.val[0], dst.ptr<mupp_type>(),
                      static_cast<int>(dst.step), oSizeROI));

    if (stream == 0) musaSafeCall(musaDeviceSynchronize());
  }
};
}  // namespace

void cv::musa::rshift(InputArray _src, Scalar_<int> val, OutputArray _dst,
                      Stream& stream) {
  typedef void (*func_t)(const GpuMat& src, Scalar_<MUpp32u> sc, GpuMat& dst,
                         musaStream_t stream);
  static const func_t funcs[5][4] = {
      {MUppShift<CV_8U, 1, muppiRShiftC_8u_C1R>::call, 0,
       MUppShift<CV_8U, 3, muppiRShiftC_8u_C3R>::call,
       MUppShift<CV_8U, 4, muppiRShiftC_8u_C4R>::call},
      {MUppShift<CV_8S, 1, muppiRShiftC_8s_C1R>::call, 0,
       MUppShift<CV_8S, 3, muppiRShiftC_8s_C3R>::call,
       MUppShift<CV_8S, 4, muppiRShiftC_8s_C4R>::call},
      {MUppShift<CV_16U, 1, muppiRShiftC_16u_C1R>::call, 0,
       MUppShift<CV_16U, 3, muppiRShiftC_16u_C3R>::call,
       MUppShift<CV_16U, 4, muppiRShiftC_16u_C4R>::call},
      {MUppShift<CV_16S, 1, muppiRShiftC_16s_C1R>::call, 0,
       MUppShift<CV_16S, 3, muppiRShiftC_16s_C3R>::call,
       MUppShift<CV_16S, 4, muppiRShiftC_16s_C4R>::call},
      {MUppShift<CV_32S, 1, muppiRShiftC_32s_C1R>::call, 0,
       MUppShift<CV_32S, 3, muppiRShiftC_32s_C3R>::call,
       MUppShift<CV_32S, 4, muppiRShiftC_32s_C4R>::call},
  };

  GpuMat src = getInputMat(_src, stream);

  CV_Assert(src.depth() < CV_32F);
  CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

  GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

  funcs[src.depth()][src.channels() - 1](src, val, dst,
                                         StreamAccessor::getStream(stream));

  syncOutput(dst, _dst, stream);
}

void cv::musa::lshift(InputArray _src, Scalar_<int> val, OutputArray _dst,
                      Stream& stream) {
  typedef void (*func_t)(const GpuMat& src, Scalar_<MUpp32u> sc, GpuMat& dst,
                         musaStream_t stream);
  static const func_t funcs[5][4] = {
      {MUppShift<CV_8U, 1, muppiLShiftC_8u_C1R>::call, 0,
       MUppShift<CV_8U, 3, muppiLShiftC_8u_C3R>::call,
       MUppShift<CV_8U, 4, muppiLShiftC_8u_C4R>::call},
      {0, 0, 0, 0},
      {MUppShift<CV_16U, 1, muppiLShiftC_16u_C1R>::call, 0,
       MUppShift<CV_16U, 3, muppiLShiftC_16u_C3R>::call,
       MUppShift<CV_16U, 4, muppiLShiftC_16u_C4R>::call},
      {0, 0, 0, 0},
      {MUppShift<CV_32S, 1, muppiLShiftC_32s_C1R>::call, 0,
       MUppShift<CV_32S, 3, muppiLShiftC_32s_C3R>::call,
       MUppShift<CV_32S, 4, muppiLShiftC_32s_C4R>::call},
  };

  GpuMat src = getInputMat(_src, stream);

  CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U ||
            src.depth() == CV_32S);
  CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

  GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

  funcs[src.depth()][src.channels() - 1](src, val, dst,
                                         StreamAccessor::getStream(stream));

  syncOutput(dst, _dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
// Minimum and maximum operations

namespace {
enum { MIN_OP, MAX_OP };
}

void minMaxMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
               const GpuMat&, double, Stream& stream, int op);

void minMaxScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst,
                  const GpuMat&, double, Stream& stream, int op);

void cv::musa::min(InputArray src1, InputArray src2, OutputArray dst,
                   Stream& stream) {
  arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, minMaxMat,
            minMaxScalar, MIN_OP);
}

void cv::musa::max(InputArray src1, InputArray src2, OutputArray dst,
                   Stream& stream) {
  arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, minMaxMat,
            minMaxScalar, MAX_OP);
}

////////////////////////////////////////////////////////////////////////
// MUPP magnitide

namespace {
typedef MUppStatus (*muppMagnitude_t)(const MUpp32fc* pSrc, int nSrcStep,
                                      MUpp32f* pDst, int nDstStep,
                                      MUppiSize oSizeROI);

void mupp_magnitude(const GpuMat& src, GpuMat& dst, muppMagnitude_t func,
                    musaStream_t stream) {
  CV_Assert(src.type() == CV_32FC2);

  MUppiSize sz;
  sz.width = src.cols;
  sz.height = src.rows;

  MUppStreamHandler h(stream);

  muppSafeCall(func(src.ptr<MUpp32fc>(), static_cast<int>(src.step),
                    dst.ptr<MUpp32f>(), static_cast<int>(dst.step), sz));

  if (stream == 0) musaSafeCall(musaDeviceSynchronize());
}
}  // namespace

void cv::musa::magnitude(InputArray _src, OutputArray _dst, Stream& stream) {
  GpuMat src = getInputMat(_src, stream);

  GpuMat dst = getOutputMat(_dst, src.size(), CV_32FC1, stream);

  mupp_magnitude(src, dst, muppiMagnitude_32fc32f_C1R,
                 StreamAccessor::getStream(stream));

  syncOutput(dst, _dst, stream);
}

void cv::musa::magnitudeSqr(InputArray _src, OutputArray _dst, Stream& stream) {
  GpuMat src = getInputMat(_src, stream);

  GpuMat dst = getOutputMat(_dst, src.size(), CV_32FC1, stream);

  mupp_magnitude(src, dst, muppiMagnitudeSqr_32fc32f_C1R,
                 StreamAccessor::getStream(stream));

  syncOutput(dst, _dst, stream);
}

#endif
