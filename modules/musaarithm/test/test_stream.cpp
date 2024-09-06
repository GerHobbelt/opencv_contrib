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

#include "test_precomp.hpp"

#ifdef HAVE_MUSA

#include <musa_runtime.h>

#include "opencv2/core/musa.hpp"
#include "opencv2/core/musa_stream_accessor.hpp"
#include "opencv2/ts/musa_test.hpp"

namespace opencv_test { namespace {

struct Async : testing::TestWithParam<cv::musa::DeviceInfo>
{
    cv::musa::HostMem src;
    cv::musa::GpuMat d_src;

    cv::musa::HostMem dst;
    cv::musa::GpuMat d_dst;

    virtual void SetUp()
    {
        cv::musa::DeviceInfo devInfo = GetParam();
        cv::musa::setDevice(devInfo.deviceID());

        src = cv::musa::HostMem(cv::musa::HostMem::PAGE_LOCKED);

        cv::Mat m = randomMatMusa(cv::Size(128, 128), CV_8UC1);
        m.copyTo(src);
    }
};

void checkMemSet(int status, void* userData)
{
    ASSERT_EQ(musaSuccess, status);

    Async* test = reinterpret_cast<Async*>(userData);

    cv::musa::HostMem src = test->src;
    cv::musa::HostMem dst = test->dst;

    cv::Mat dst_gold = cv::Mat::zeros(src.size(), src.type());

    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

MUSA_TEST_P(Async, MemSet)
{
    cv::musa::Stream stream;

    d_dst.upload(src);

    d_dst.setTo(cv::Scalar::all(0), stream);
    d_dst.download(dst, stream);

    Async* test = this;
    stream.enqueueHostCallback(checkMemSet, test);

    stream.waitForCompletion();
}

void checkConvert(int status, void* userData)
{
    ASSERT_EQ(musaSuccess, status);

    Async* test = reinterpret_cast<Async*>(userData);

    cv::musa::HostMem src = test->src;
    cv::musa::HostMem dst = test->dst;

    cv::Mat dst_gold;
    src.createMatHeader().convertTo(dst_gold, CV_32S);

    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

MUSA_TEST_P(Async, Convert)
{
    cv::musa::Stream stream;

    d_src.upload(src, stream);
    d_src.convertTo(d_dst, CV_32S, stream);
    d_dst.download(dst, stream);

    Async* test = this;
    stream.enqueueHostCallback(checkConvert, test);

    stream.waitForCompletion();
}

MUSA_TEST_P(Async, WrapStream)
{
    musaStream_t musa_stream = NULL;
    ASSERT_EQ(musaSuccess, musaStreamCreate(&musa_stream));

    {
        cv::musa::Stream stream = cv::musa::StreamAccessor::wrapStream(musa_stream);

        d_src.upload(src, stream);
        d_src.convertTo(d_dst, CV_32S, stream);
        d_dst.download(dst, stream);

        Async* test = this;
        stream.enqueueHostCallback(checkConvert, test);

        stream.waitForCompletion();
    }

    ASSERT_EQ(musaSuccess, musaStreamDestroy(musa_stream));
}

MUSA_TEST_P(Async, HostMemAllocator)
{
    cv::musa::Stream stream;

    cv::Mat h_dst;
    h_dst.allocator = cv::musa::HostMem::getAllocator();

    d_src.upload(src, stream);
    d_src.convertTo(d_dst, CV_32S, stream);
    d_dst.download(h_dst, stream);

    stream.waitForCompletion();

    cv::Mat dst_gold;
    src.createMatHeader().convertTo(dst_gold, CV_32S);

    ASSERT_MAT_NEAR(dst_gold, h_dst, 0);
}

INSTANTIATE_TEST_CASE_P(MUSA_Stream, Async, ALL_DEVICES);

}} // namespace
#endif // HAVE_MUSA
