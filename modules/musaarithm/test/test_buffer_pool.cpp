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

#include "opencv2/core/musa.hpp"
#include "opencv2/core/private.musa.hpp"
#include "opencv2/ts/musa_test.hpp"

namespace opencv_test { namespace {

struct BufferPoolTest : TestWithParam<DeviceInfo>
{
    void RunSimpleTest(cv::musa::Stream& stream, cv::musa::HostMem& dst_1, cv::musa::HostMem& dst_2)
    {
        BufferPool pool(stream);

        {
            cv::musa::GpuMat buf0 = pool.getBuffer(Size(640, 480), CV_8UC1);
            EXPECT_FALSE( buf0.empty() );

            buf0.setTo(Scalar::all(0), stream);

            cv::musa::GpuMat buf1 = pool.getBuffer(Size(640, 480), CV_8UC1);
            EXPECT_FALSE( buf1.empty() );

            buf0.convertTo(buf1, buf1.type(), 1.0, 1.0, stream);

            buf1.download(dst_1, stream);
        }

        {
            cv::musa::GpuMat buf2 = pool.getBuffer(Size(1280, 1024), CV_32SC1);
            EXPECT_FALSE( buf2.empty() );

            buf2.setTo(Scalar::all(2), stream);

            buf2.download(dst_2, stream);
        }
    }

    void CheckSimpleTest(cv::musa::HostMem& dst_1, cv::musa::HostMem& dst_2)
    {
        EXPECT_MAT_NEAR(Mat(Size(640, 480), CV_8UC1, Scalar::all(1)), dst_1, 0.0);
        EXPECT_MAT_NEAR(Mat(Size(1280, 1024), CV_32SC1, Scalar::all(2)), dst_2, 0.0);
    }
};

MUSA_TEST_P(BufferPoolTest, FromNullStream)
{
    cv::musa::HostMem dst_1, dst_2;

    RunSimpleTest(cv::musa::Stream::Null(), dst_1, dst_2);

    musaSafeCall(musaDeviceSynchronize());

    CheckSimpleTest(dst_1, dst_2);
}

MUSA_TEST_P(BufferPoolTest, From2Streams)
{
    cv::musa::HostMem dst1_1, dst1_2;
    cv::musa::HostMem dst2_1, dst2_2;

    cv::musa::Stream stream1, stream2;
    RunSimpleTest(stream1, dst1_1, dst1_2);
    RunSimpleTest(stream2, dst2_1, dst2_2);

    stream1.waitForCompletion();
    stream2.waitForCompletion();

    CheckSimpleTest(dst1_1, dst1_2);
    CheckSimpleTest(dst2_1, dst2_2);
}

INSTANTIATE_TEST_CASE_P(MUSA_Stream, BufferPoolTest, ALL_DEVICES);

}} // namespace
#endif // HAVE_MUSA
