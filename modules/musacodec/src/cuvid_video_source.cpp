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

#include "precomp.hpp"

#ifdef HAVE_NVCUVID

using namespace cv;
using namespace cv::musacodec;
using namespace cv::musacodec::detail;

cv::musacodec::detail::CuvidVideoSource::CuvidVideoSource(const String& fname)
{
    CUVIDSOURCEPARAMS params;
    std::memset(&params, 0, sizeof(CUVIDSOURCEPARAMS));

    // Fill parameter struct
    params.pUserData = this;                        // will be passed to data handlers
    params.pfnVideoDataHandler = HandleVideoData;   // our local video-handler callback
    params.pfnAudioDataHandler = 0;

    // now create the actual source
    MUresult cuRes = cuvidCreateVideoSource(&videoSource_, fname.c_str(), &params);
    if (cuRes == MUSA_ERROR_INVALID_SOURCE)
        CV_Error(Error::StsUnsupportedFormat, "Unsupported video source");
    muSafeCall( cuRes );

    CUVIDEOFORMAT vidfmt;
    muSafeCall( cuvidGetSourceVideoFormat(videoSource_, &vidfmt, 0) );

    CV_Assert(Codec::NumCodecs == musaVideoCodec::musaVideoCodec_NumCodecs);
    format_.codec = static_cast<Codec>(vidfmt.codec);
    format_.chromaFormat = static_cast<ChromaFormat>(vidfmt.chroma_format);
    format_.nBitDepthMinus8 = vidfmt.bit_depth_luma_minus8;
    format_.width = vidfmt.coded_width;
    format_.height = vidfmt.coded_height;
    format_.displayArea = Rect(Point(vidfmt.display_area.left, vidfmt.display_area.top), Point(vidfmt.display_area.right, vidfmt.display_area.bottom));
    format_.valid = true;
    if (vidfmt.frame_rate.numerator != 0 && vidfmt.frame_rate.denominator != 0)
        format_.fps = vidfmt.frame_rate.numerator / (double)vidfmt.frame_rate.denominator;
}

cv::musacodec::detail::CuvidVideoSource::~CuvidVideoSource()
{
    cuvidDestroyVideoSource(videoSource_);
}

FormatInfo cv::musacodec::detail::CuvidVideoSource::format() const
{
    return format_;
}

void cv::musacodec::detail::CuvidVideoSource::updateFormat(const FormatInfo& videoFormat)
{
    format_ = videoFormat;
    format_.valid = true;
}

void cv::musacodec::detail::CuvidVideoSource::start()
{
    muSafeCall( cuvidSetVideoSourceState(videoSource_, musaVideoState_Started) );
}

void cv::musacodec::detail::CuvidVideoSource::stop()
{
    muSafeCall( cuvidSetVideoSourceState(videoSource_, musaVideoState_Stopped) );
}

bool cv::musacodec::detail::CuvidVideoSource::isStarted() const
{
    return (cuvidGetVideoSourceState(videoSource_) == musaVideoState_Started);
}

bool cv::musacodec::detail::CuvidVideoSource::hasError() const
{
    return (cuvidGetVideoSourceState(videoSource_) == musaVideoState_Error);
}

int MUSAAPI cv::musacodec::detail::CuvidVideoSource::HandleVideoData(void* userData, CUVIDSOURCEDATAPACKET* packet)
{
    CuvidVideoSource* thiz = static_cast<CuvidVideoSource*>(userData);

    return thiz->parseVideoData(packet->payload, packet->payload_size, thiz->RawModeEnabled(), false, (packet->flags & CUVID_PKT_ENDOFSTREAM) != 0);
}

#endif // HAVE_NVCUVID
