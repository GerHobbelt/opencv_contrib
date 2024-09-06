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

#if !defined (HAVE_MUSA) || defined (MUSA_DISABLER)

double cv::musa::norm(InputArray, int, InputArray) { throw_no_musa(); return 0.0; }
void cv::musa::calcNorm(InputArray, OutputArray, int, InputArray, Stream&) { throw_no_musa(); }
double cv::musa::norm(InputArray, InputArray, int) { throw_no_musa(); return 0.0; }
void cv::musa::calcNormDiff(InputArray, InputArray, OutputArray, int, Stream&) { throw_no_musa(); }

Scalar cv::musa::sum(InputArray, InputArray) { throw_no_musa(); return Scalar(); }
void cv::musa::calcSum(InputArray, OutputArray, InputArray, Stream&) { throw_no_musa(); }
Scalar cv::musa::absSum(InputArray, InputArray) { throw_no_musa(); return Scalar(); }
void cv::musa::calcAbsSum(InputArray, OutputArray, InputArray, Stream&) { throw_no_musa(); }
Scalar cv::musa::sqrSum(InputArray, InputArray) { throw_no_musa(); return Scalar(); }
void cv::musa::calcSqrSum(InputArray, OutputArray, InputArray, Stream&) { throw_no_musa(); }

void cv::musa::minMax(InputArray, double*, double*, InputArray) { throw_no_musa(); }
void cv::musa::findMinMax(InputArray, OutputArray, InputArray, Stream&) { throw_no_musa(); }
void cv::musa::minMaxLoc(InputArray, double*, double*, Point*, Point*, InputArray) { throw_no_musa(); }
void cv::musa::findMinMaxLoc(InputArray, OutputArray, OutputArray, InputArray, Stream&) { throw_no_musa(); }

int cv::musa::countNonZero(InputArray) { throw_no_musa(); return 0; }
void cv::musa::countNonZero(InputArray, OutputArray, Stream&) { throw_no_musa(); }

void cv::musa::reduce(InputArray, OutputArray, int, int, int, Stream&) { throw_no_musa(); }

void cv::musa::meanStdDev(InputArray, Scalar&, Scalar&) { throw_no_musa(); }
void cv::musa::meanStdDev(InputArray, OutputArray, Stream&) { throw_no_musa(); }

void cv::musa::rectStdDev(InputArray, InputArray, OutputArray, Rect, Stream&) { throw_no_musa(); }

void cv::musa::normalize(InputArray, OutputArray, double, double, int, int, InputArray, Stream&) { throw_no_musa(); }

void cv::musa::integral(InputArray, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::sqrIntegral(InputArray, OutputArray, Stream&) { throw_no_musa(); }

#else

////////////////////////////////////////////////////////////////////////
// norm

namespace cv { namespace musa { namespace device {

void normL2(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask, Stream& stream);

void findMaxAbs(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask, Stream& stream);

}}}

void cv::musa::calcNorm(InputArray _src, OutputArray dst, int normType, InputArray mask, Stream& stream)
{
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    GpuMat src = getInputMat(_src, stream);

    GpuMat src_single_channel = src.reshape(1);

    if (normType == NORM_L1)
    {
        calcAbsSum(src_single_channel, dst, mask, stream);
    }
    else if (normType == NORM_L2)
    {
        cv::musa::device::normL2(src_single_channel, dst, mask, stream);
    }
    else // NORM_INF
    {
        cv::musa::device::findMaxAbs(src_single_channel, dst, mask, stream);
    }
}

double cv::musa::norm(InputArray _src, int normType, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    calcNorm(_src, dst, normType, _mask, stream);

    stream.waitForCompletion();

    double val;
    dst.createMatHeader().convertTo(Mat(1, 1, CV_64FC1, &val), CV_64F);

    return val;
}

////////////////////////////////////////////////////////////////////////
// meanStdDev

void cv::musa::meanStdDev(InputArray src, OutputArray dst, Stream& stream)
{
    const GpuMat gsrc = getInputMat(src, stream);

    CV_Assert( (gsrc.type() == CV_8UC1) || (gsrc.type() == CV_32FC1) );
    GpuMat gdst = getOutputMat(dst, 1, 2, CV_64FC1, stream);

    MUppiSize sz;
    sz.width  = gsrc.cols;
    sz.height = gsrc.rows;

    int bufSize;

if (gsrc.type() == CV_8UC1)
    muppSafeCall( muppiMeanStdDevGetBufferHostSize_8u_C1R(sz, &bufSize) );
else
    muppSafeCall( muppiMeanStdDevGetBufferHostSize_32f_C1R(sz, &bufSize) );

    BufferPool pool(stream);
    GpuMat buf = pool.getBuffer(1, bufSize, gsrc.type());

    // detail: https://github.com/opencv/opencv/issues/11063
    //MUppStreamHandler h(StreamAccessor::getStream(stream));

    if(gsrc.type() == CV_8UC1)
        muppSafeCall( muppiMean_StdDev_8u_C1R(gsrc.ptr<MUpp8u>(), static_cast<int>(gsrc.step), sz, buf.ptr<MUpp8u>(), gdst.ptr<MUpp64f>(), gdst.ptr<MUpp64f>() + 1) );
    else
        muppSafeCall( muppiMean_StdDev_32f_C1R(gsrc.ptr<MUpp32f>(), static_cast<int>(gsrc.step), sz, buf.ptr<MUpp8u>(), gdst.ptr<MUpp64f>(), gdst.ptr<MUpp64f>() + 1) );

    syncOutput(gdst, dst, stream);
}

void cv::musa::meanStdDev(InputArray src, Scalar& mean, Scalar& stddev)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    meanStdDev(src, dst, stream);

    stream.waitForCompletion();

    double vals[2];
    dst.createMatHeader().copyTo(Mat(1, 2, CV_64FC1, &vals[0]));

    mean = Scalar(vals[0]);
    stddev = Scalar(vals[1]);
}

void cv::musa::meanStdDev(InputArray _src, Scalar& mean, Scalar& stddev, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    meanStdDev(_src, dst, _mask, stream);
    stream.waitForCompletion();

    double vals[2];
    dst.createMatHeader().copyTo(Mat(1, 2, CV_64FC1, &vals[0]));

    mean = Scalar(vals[0]);
    stddev = Scalar(vals[1]);
}

void cv::musa::meanStdDev(InputArray src, OutputArray dst, InputArray mask, Stream& stream)
{
    const GpuMat gsrc = getInputMat(src, stream);
    const GpuMat gmask = getInputMat(mask, stream);

    CV_Assert( (gsrc.type() == CV_8UC1) || (gsrc.type() == CV_32FC1) );
    GpuMat gdst = getOutputMat(dst, 1, 2, CV_64FC1, stream);

    MUppiSize sz;
    sz.width  = gsrc.cols;
    sz.height = gsrc.rows;

    int bufSize;
if (gsrc.type() == CV_8UC1)
    muppSafeCall( muppiMeanStdDevGetBufferHostSize_8u_C1MR(sz, &bufSize) );
else
    muppSafeCall( muppiMeanStdDevGetBufferHostSize_32f_C1MR(sz, &bufSize) );

    BufferPool pool(stream);
    GpuMat buf = pool.getBuffer(1, bufSize, gsrc.type());

    if(gsrc.type() == CV_8UC1)
        muppSafeCall( muppiMean_StdDev_8u_C1MR(gsrc.ptr<MUpp8u>(), static_cast<int>(gsrc.step), gmask.ptr<MUpp8u>(), static_cast<int>(gmask.step),
                                             sz, buf.ptr<MUpp8u>(), gdst.ptr<MUpp64f>(), gdst.ptr<MUpp64f>() + 1) );
    else
        muppSafeCall( muppiMean_StdDev_32f_C1MR(gsrc.ptr<MUpp32f>(), static_cast<int>(gsrc.step), gmask.ptr<MUpp8u>(), static_cast<int>(gmask.step),
                                              sz, buf.ptr<MUpp8u>(), gdst.ptr<MUpp64f>(), gdst.ptr<MUpp64f>() + 1) );

    syncOutput(gdst, dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
// rectStdDev

void cv::musa::rectStdDev(InputArray _src, InputArray _sqr, OutputArray _dst, Rect rect, Stream& _stream)
{
    GpuMat src = getInputMat(_src, _stream);
    GpuMat sqr = getInputMat(_sqr, _stream);

    CV_Assert( src.type() == CV_32SC1 && sqr.type() == CV_64FC1 );

    GpuMat dst = getOutputMat(_dst, src.size(), CV_32FC1, _stream);

    MUppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    MUppiRect muppRect;
    muppRect.height = rect.height;
    muppRect.width = rect.width;
    muppRect.x = rect.x;
    muppRect.y = rect.y;

    musaStream_t stream = StreamAccessor::getStream(_stream);

    MUppStreamHandler h(stream);

    muppSafeCall( muppiRectStdDev_32s32f_C1R(src.ptr<MUpp32s>(), static_cast<int>(src.step), sqr.ptr<MUpp64f>(), static_cast<int>(sqr.step),
                dst.ptr<MUpp32f>(), static_cast<int>(dst.step), sz, muppRect) );

    if (stream == 0)
        musaSafeCall( musaDeviceSynchronize() );

    syncOutput(dst, _dst, _stream);
}

#endif
