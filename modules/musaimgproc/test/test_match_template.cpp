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

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_CCOEFF_NORMED))

namespace
{
    IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size);
}

PARAM_TEST_CASE(MatchTemplate8U, cv::musa::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        templ_size = GET_PARAM(2);
        cn = GET_PARAM(3);
        method = GET_PARAM(4);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(MatchTemplate8U, Accuracy)
{
    cv::Mat image = randomMatMusa(size, CV_MAKETYPE(CV_8U, cn));
    cv::Mat templ = randomMatMusa(templ_size, CV_MAKETYPE(CV_8U, cn));

    cv::Ptr<cv::musa::TemplateMatching> alg = cv::musa::createTemplateMatching(image.type(), method);

    cv::musa::GpuMat dst;
    alg->match(musaLoadMat(image), musaLoadMat(templ), dst);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    cv::Mat h_dst(dst);
    ASSERT_EQ(dst_gold.size(), h_dst.size());
    ASSERT_EQ(dst_gold.type(), h_dst.type());
    for (int y = 0; y < h_dst.rows; ++y)
    {
        for (int x = 0; x < h_dst.cols; ++x)
        {
            float gold_val = dst_gold.at<float>(y, x);
            float actual_val = dst_gold.at<float>(y, x);
            ASSERT_FLOAT_EQ(gold_val, actual_val) << y << ", " << x;
        }
    }
}

INSTANTIATE_TEST_CASE_P(MUSA_ImgProc, MatchTemplate8U, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    ALL_TEMPLATE_METHODS));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate32F

PARAM_TEST_CASE(MatchTemplate32F, cv::musa::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::musa::DeviceInfo devInfo;
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    int n, m, h, w;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        templ_size = GET_PARAM(2);
        cn = GET_PARAM(3);
        method = GET_PARAM(4);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(MatchTemplate32F, Regression)
{
    cv::Mat image = randomMatMusa(size, CV_MAKETYPE(CV_32F, cn));
    cv::Mat templ = randomMatMusa(templ_size, CV_MAKETYPE(CV_32F, cn));

    cv::Ptr<cv::musa::TemplateMatching> alg = cv::musa::createTemplateMatching(image.type(), method);

    cv::musa::GpuMat dst;
    alg->match(musaLoadMat(image), musaLoadMat(templ), dst);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    cv::Mat h_dst(dst);
    ASSERT_EQ(dst_gold.size(), h_dst.size());
    ASSERT_EQ(dst_gold.type(), h_dst.type());
    for (int y = 0; y < h_dst.rows; ++y)
    {
        for (int x = 0; x < h_dst.cols; ++x)
        {
            float gold_val = dst_gold.at<float>(y, x);
            float actual_val = dst_gold.at<float>(y, x);
            ASSERT_FLOAT_EQ(gold_val, actual_val) << y << ", " << x;
        }
    }
}

INSTANTIATE_TEST_CASE_P(MUSA_ImgProc, MatchTemplate32F, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplateBlackSource

PARAM_TEST_CASE(MatchTemplateBlackSource, cv::musa::DeviceInfo, TemplateMethod)
{
    cv::musa::DeviceInfo devInfo;
    int method;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        method = GET_PARAM(1);

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(MatchTemplateBlackSource, Accuracy)
{
    cv::Mat image = readGpuImage("matchtemplate/black.png");
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readGpuImage("matchtemplate/cat.png");
    ASSERT_FALSE(pattern.empty());

    cv::Ptr<cv::musa::TemplateMatching> alg = cv::musa::createTemplateMatching(image.type(), method);

    cv::musa::GpuMat d_dst;
    alg->match(musaLoadMat(image), musaLoadMat(pattern), d_dst);

    cv::Mat dst(d_dst);

    double maxValue;
    cv::Point maxLoc;
    cv::minMaxLoc(dst, NULL, &maxValue, NULL, &maxLoc);

    cv::Point maxLocGold = cv::Point(284, 12);

    ASSERT_EQ(maxLocGold, maxLoc);
}

INSTANTIATE_TEST_CASE_P(MUSA_ImgProc, MatchTemplateBlackSource, testing::Combine(
    ALL_DEVICES,
    testing::Values(TemplateMethod(cv::TM_CCOEFF_NORMED), TemplateMethod(cv::TM_CCORR_NORMED))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_CCOEF_NORMED

PARAM_TEST_CASE(MatchTemplate_CCOEF_NORMED, cv::musa::DeviceInfo, std::pair<std::string, std::string>)
{
    cv::musa::DeviceInfo devInfo;
    std::string imageName;
    std::string patternName;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        imageName = GET_PARAM(1).first;
        patternName = GET_PARAM(1).second;

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(MatchTemplate_CCOEF_NORMED, Accuracy)
{
    cv::Mat image = readGpuImage(imageName);
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readGpuImage(patternName);
    ASSERT_FALSE(pattern.empty());

    cv::Ptr<cv::musa::TemplateMatching> alg = cv::musa::createTemplateMatching(image.type(), cv::TM_CCOEFF_NORMED);

    cv::musa::GpuMat d_dst;
    alg->match(musaLoadMat(image), musaLoadMat(pattern), d_dst);

    cv::Mat dst(d_dst);

    cv::Point minLoc, maxLoc;
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Mat dstGold;
    cv::matchTemplate(image, pattern, dstGold, cv::TM_CCOEFF_NORMED);

    double minValGold, maxValGold;
    cv::Point minLocGold, maxLocGold;
    cv::minMaxLoc(dstGold, &minValGold, &maxValGold, &minLocGold, &maxLocGold);

    ASSERT_EQ(minLocGold, minLoc);
    ASSERT_EQ(maxLocGold, maxLoc);
    ASSERT_LE(maxVal, 1.0);
    ASSERT_GE(minVal, -1.0);
}

INSTANTIATE_TEST_CASE_P(MUSA_ImgProc, MatchTemplate_CCOEF_NORMED, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::make_pair(std::string("matchtemplate/source-0.png"), std::string("matchtemplate/target-0.png")))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_CanFindBigTemplate

struct MatchTemplate_CanFindBigTemplate : testing::TestWithParam<cv::musa::DeviceInfo>
{
    cv::musa::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::musa::setDevice(devInfo.deviceID());
    }
};

MUSA_TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF_NORMED)
{
    cv::Mat scene = readGpuImage("matchtemplate/scene.png");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readGpuImage("matchtemplate/template.png");
    ASSERT_FALSE(templ.empty());

    cv::Ptr<cv::musa::TemplateMatching> alg = cv::musa::createTemplateMatching(scene.type(), cv::TM_SQDIFF_NORMED);

    cv::musa::GpuMat d_result;
    alg->match(musaLoadMat(scene), musaLoadMat(templ), d_result);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_LT(minVal, 1e-3);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

MUSA_TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF)
{
    cv::Mat scene = readGpuImage("matchtemplate/scene.png");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readGpuImage("matchtemplate/template.png");
    ASSERT_FALSE(templ.empty());

    cv::Ptr<cv::musa::TemplateMatching> alg = cv::musa::createTemplateMatching(scene.type(), cv::TM_SQDIFF);

    cv::musa::GpuMat d_result;
    alg->match(musaLoadMat(scene), musaLoadMat(templ), d_result);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

INSTANTIATE_TEST_CASE_P(MUSA_ImgProc, MatchTemplate_CanFindBigTemplate, ALL_DEVICES);


}} // namespace
#endif // HAVE_MUSA
