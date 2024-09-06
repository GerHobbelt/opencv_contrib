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

namespace opencv_test { namespace {

namespace
{
    IMPLEMENT_PARAM_CLASS(KSize, cv::Size)
    IMPLEMENT_PARAM_CLASS(Anchor, cv::Point)
    IMPLEMENT_PARAM_CLASS(Deriv_X, int)
    IMPLEMENT_PARAM_CLASS(Deriv_Y, int)
    IMPLEMENT_PARAM_CLASS(Iterations, int)
    IMPLEMENT_PARAM_CLASS(KernelSize, int)

    cv::Mat getInnerROI(cv::InputArray m_, cv::Size ksize)
    {
        cv::Mat m = getGpuMat(m_);
        cv::Rect roi(ksize.width, ksize.height, m.cols - 2 * ksize.width, m.rows - 2 * ksize.height);
        return m(roi);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Blur

PARAM_TEST_CASE(Blur, cv::musa::DeviceInfo, cv::Size, MatType, KSize, Anchor, BorderType, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        anchor = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(Blur, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);

    cv::Ptr<cv::musa::Filter> blurFilter = cv::musa::createBoxFilter(src.type(), -1, ksize, anchor, borderType);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    blurFilter->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::blur(src, dst_gold, ksize, anchor, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Blur, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_32FC1)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

PARAM_TEST_CASE(Filter2D, cv::musa::DeviceInfo, cv::Size, MatType, KSize, Anchor, BorderType, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        anchor = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(Filter2D, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);
    cv::Mat kernel = randomMatMusa(cv::Size(ksize.width, ksize.height), CV_32FC1, 0.0, 1.0);

    cv::Ptr<cv::musa::Filter> filter2D = cv::musa::createLinearFilter(src.type(), -1, kernel, anchor, borderType);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    filter2D->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::filter2D(src, dst_gold, -1, kernel, anchor, 0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, CV_MAT_DEPTH(type) == CV_32F ? 1e-1 : 1.0);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Filter2D, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC4)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7)), KSize(cv::Size(11, 11)), KSize(cv::Size(13, 13)), KSize(cv::Size(15, 15))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

PARAM_TEST_CASE(Laplacian, cv::musa::DeviceInfo, cv::Size, MatType, KSize, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(Laplacian, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);

    cv::Ptr<cv::musa::Filter> laplacian = cv::musa::createLaplacianFilter(src.type(), -1, ksize.width);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    laplacian->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::Laplacian(src, dst_gold, -1, ksize.width);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 0.0 : 1e-3);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Laplacian, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_32FC1)),
    testing::Values(KSize(cv::Size(1, 1)), KSize(cv::Size(3, 3))),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// SeparableLinearFilter

PARAM_TEST_CASE(SeparableLinearFilter, cv::musa::DeviceInfo, cv::Size, MatDepth, Channels, KSize, Anchor, BorderType, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        ksize = GET_PARAM(4);
        anchor = GET_PARAM(5);
        borderType = GET_PARAM(6);
        useRoi = GET_PARAM(7);

        cv::musa::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

MUSA_TEST_P(SeparableLinearFilter, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);
    cv::Mat rowKernel = randomMatMusa(Size(ksize.width, 1), CV_32FC1, 0.0, 1.0);
    cv::Mat columnKernel = randomMatMusa(Size(ksize.height, 1), CV_32FC1, 0.0, 1.0);

    cv::Ptr<cv::musa::Filter> filter = cv::musa::createSeparableLinearFilter(src.type(), -1, rowKernel, columnKernel, anchor, borderType);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    filter->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::sepFilter2D(src, dst_gold, -1, rowKernel, columnKernel, anchor, 0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 1.0 : 1e-2);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, SeparableLinearFilter, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(KSize(cv::Size(3, 3)),
                    KSize(cv::Size(7, 7)),
                    KSize(cv::Size(13, 13)),
                    KSize(cv::Size(15, 15)),
                    KSize(cv::Size(17, 17)),
                    KSize(cv::Size(23, 15)),
                    KSize(cv::Size(31, 3))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel

PARAM_TEST_CASE(Sobel, cv::musa::DeviceInfo, cv::Size, MatDepth, Channels, KSize, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    cv::Size ksize;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        ksize = GET_PARAM(4);
        dx = GET_PARAM(5);
        dy = GET_PARAM(6);
        borderType = GET_PARAM(7);
        useRoi = GET_PARAM(8);

        cv::musa::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

MUSA_TEST_P(Sobel, Accuracy)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat src = randomMatMusa(size, type);

    cv::Ptr<cv::musa::Filter> sobel = cv::musa::createSobelFilter(src.type(), -1, dx, dy, ksize.width, 1.0, borderType);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    sobel->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::Sobel(src, dst_gold, -1, dx, dy, ksize.width, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 0.0 : 0.1);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Sobel, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7))),
    testing::Values(Deriv_X(0), Deriv_X(1), Deriv_X(2)),
    testing::Values(Deriv_Y(0), Deriv_Y(1), Deriv_Y(2)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr

PARAM_TEST_CASE(Scharr, cv::musa::DeviceInfo, cv::Size, MatDepth, Channels, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        dx = GET_PARAM(4);
        dy = GET_PARAM(5);
        borderType = GET_PARAM(6);
        useRoi = GET_PARAM(7);

        cv::musa::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

MUSA_TEST_P(Scharr, Accuracy)
{
    if (dx + dy != 1)
        return;

    cv::Mat src = randomMatMusa(size, type);

    cv::Ptr<cv::musa::Filter> scharr = cv::musa::createScharrFilter(src.type(), -1, dx, dy, 1.0, borderType);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    scharr->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::Scharr(src, dst_gold, -1, dx, dy, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 0.0 : 0.1);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Scharr, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(Deriv_X(0), Deriv_X(1)),
    testing::Values(Deriv_Y(0), Deriv_Y(1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

PARAM_TEST_CASE(GaussianBlur, cv::musa::DeviceInfo, cv::Size, MatDepth, Channels, KSize, BorderType, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    cv::Size ksize;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        ksize = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::musa::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

MUSA_TEST_P(GaussianBlur, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);
    double sigma1 = randomDoubleMusa(0.0, 1.0);
    double sigma2 = randomDoubleMusa(0.0, 1.0);

    cv::Ptr<cv::musa::Filter> gauss = cv::musa::createGaussianFilter(src.type(), -1, ksize, sigma1, sigma2, borderType);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    gauss->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::GaussianBlur(src, dst_gold, ksize, sigma1, sigma2, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 4.0 : 1e-4);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, GaussianBlur, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(KSize(cv::Size(3, 3)),
                    KSize(cv::Size(5, 5)),
                    KSize(cv::Size(7, 7)),
                    KSize(cv::Size(9, 9)),
                    KSize(cv::Size(11, 11)),
                    KSize(cv::Size(13, 13)),
                    KSize(cv::Size(15, 15)),
                    KSize(cv::Size(17, 17)),
                    KSize(cv::Size(19, 19)),
                    KSize(cv::Size(21, 21)),
                    KSize(cv::Size(23, 23)),
                    KSize(cv::Size(25, 25)),
                    KSize(cv::Size(27, 27)),
                    KSize(cv::Size(29, 29)),
                    KSize(cv::Size(31, 31))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Erode

PARAM_TEST_CASE(Erode, cv::musa::DeviceInfo, cv::Size, MatType, Anchor, Iterations, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        anchor = GET_PARAM(3);
        iterations = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(Erode, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::Ptr<cv::musa::Filter> erode = cv::musa::createMorphologyFilter(cv::MORPH_ERODE, src.type(), kernel, anchor, iterations);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    erode->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::erode(src, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Erode, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate

PARAM_TEST_CASE(Dilate, cv::musa::DeviceInfo, cv::Size, MatType, Anchor, Iterations, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        anchor = GET_PARAM(3);
        iterations = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(Dilate, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::Ptr<cv::musa::Filter> dilate = cv::musa::createMorphologyFilter(cv::MORPH_DILATE, src.type(), kernel, anchor, iterations);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    dilate->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::dilate(src, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Dilate, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// MorphEx

CV_ENUM(MorphOp, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT)

PARAM_TEST_CASE(MorphEx, cv::musa::DeviceInfo, cv::Size, MatType, MorphOp, Anchor, Iterations, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int morphOp;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        morphOp = GET_PARAM(3);
        anchor = GET_PARAM(4);
        iterations = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(MorphEx, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::Ptr<cv::musa::Filter> morph = cv::musa::createMorphologyFilter(morphOp, src.type(), kernel, anchor, iterations);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    morph->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::morphologyEx(src, dst_gold, morphOp, kernel, anchor, iterations);

    cv::Size border = cv::Size(kernel.cols + (iterations + 1) * kernel.cols + 2, kernel.rows + (iterations + 1) * kernel.rows + 2);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, border), getInnerROI(dst, border), 0.0);
}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, MorphEx, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    MorphOp::all(),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Median


PARAM_TEST_CASE(Median, cv::musa::DeviceInfo, cv::Size, MatDepth,  KernelSize, UseRoi)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int kernel;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        kernel = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::musa::setDevice(devInfo.deviceID());
    }
};



MUSA_TEST_P(Median, Accuracy)
{
    cv::Mat src = randomMatMusa(size, type);

    cv::Ptr<cv::musa::Filter> median = cv::musa::createMedianFilter(src.type(), kernel);

    cv::musa::GpuMat dst = createGpuMat(size, type, useRoi);
    median->apply(musaLoadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::medianBlur(src,dst_gold,kernel);

    cv::Rect rect(kernel+1,0,src.cols-(2*kernel+1),src.rows);
    cv::Mat dst_gold_no_border = dst_gold(rect);
    cv::musa::GpuMat dst_no_border = cv::musa::GpuMat(dst, rect);

    EXPECT_MAT_NEAR(dst_gold_no_border, dst_no_border, 1);

}

INSTANTIATE_TEST_CASE_P(MUSA_Filters, Median, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U)),
    testing::Values(KernelSize(3),
                    KernelSize(5),
                    KernelSize(7),
                    KernelSize(9),
                    KernelSize(11),
                    KernelSize(13),
                    KernelSize(15)),
    WHOLE_SUBMAT)
    );

}} // namespace

#endif // HAVE_MUSA
