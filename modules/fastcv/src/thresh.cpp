/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "precomp.hpp"

namespace cv {
namespace fastcv {

void thresholdRange(InputArray _src, OutputArray _dst, uint8_t lowThresh, uint8_t highThresh, uint8_t trueValue, uint8_t falseValue)
{
    INITIALIZATION_CHECK;

    CV_Assert(lowThresh <= highThresh);

    CV_Assert(!_src.empty() && _src.type() == CV_8UC1);
    CV_Assert(_src.cols() % 8 == 0);
    CV_Assert(_src.step() % 8 == 0);
    Mat src = _src.getMat();

    _dst.create(_src.size(), CV_8UC1);
    // in case of fixed layout array we cannot fix this on our side, can only fail if false
    CV_Assert(_dst.step() % 8 == 0);
    Mat dst = _dst.getMat();

    fcvStatus status = fcvFilterThresholdRangeu8_v2(src.data, src.cols, src.rows, src.step,
                                                    dst.data, dst.step,
                                                    lowThresh, highThresh, trueValue, falseValue);

    if (status != FASTCV_SUCCESS)
    {
        std::string s = fcvStatusStrings.count(status) ? fcvStatusStrings.at(status) : "unknown";
        CV_Error( cv::Error::StsInternal, "FastCV error: " + s);
    }
}

} // fastcv::
} // cv::
