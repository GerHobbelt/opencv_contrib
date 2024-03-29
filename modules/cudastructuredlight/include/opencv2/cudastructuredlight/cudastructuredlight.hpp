/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
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
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CUDA_STRUCTURED_LIGHT_HPP__
#define __OPENCV_CUDA_STRUCTURED_LIGHT_HPP__

#ifndef __cplusplus
#error cudastructuredlight.hpp header must be complier as C++
#endif

#include "opencv2/core/cuda.hpp"
#include "opencv2/structured_light.hpp"

/**
  @addtogroup cuda
  @{
    @defgroup cudastructuredlight Structured Light Algorithm
  @}
 */

namespace cv {
namespace cuda {
//! @addtogroup structured_light
//! @{

//! Type of the decoding algorithm
// other algorithms can be implemented
enum {
    SINUSOIDAL_COMPLEMENTARY_GRAY_CODE =
        0, //!< Zhang Q, Su X, Xiang L, et al. 3-D shape measurement based on
           //!< complementary Gray-code light[J]. Optics and Lasers in
           //!< Engineering, 2012, 50(4): 574-579.>
    MULTI_VIEW_STEREO_GEOMETRY, // Willomitzer, F. and G. Häusler (2017).
                                // "Single-shot 3D motion picture camera with a
                                // dense point cloud(Article)." Optics Express
                                // Vol.25(No.19): 23451-23464. An, Y., et al.
                                // (2016). "Pixel-wise absolute phase unwrapping
                                // using geometric constraints of structured
                                // light system." Opt Express 24(16):
                                // 18445-18459. Li, Z., et al. (2013).
                                // "Multiview phase shifting: a full-resolution
                                // and high-speed 3D measurement framework for
                                // arbitrary shape dynamic objects." Opt Lett
                                // 38(9): 1389-1391. Willomitzer, F., et al.
                                // (2015). "Single-shot three-dimensional
                                // sensing with improved data density." Applied
                                // Optics 54(3).

};

/** @brief Abstract base class for generating and decoding structured light
 * patterns.
 */
class CV_EXPORTS_W StructuredLightPattern : public virtual Algorithm {
  public:
    /** @brief Generates the structured light pattern to project.

     @param patternImages The generated pattern: a vector<Mat>, in which each
     image is a CV_8U Mat at projector's resolution.
     */
    CV_WRAP
    virtual bool generate(std::vector<cv::Mat> &patternImages) = 0;

    /** @brief Decodes the structured light pattern, generating a disparity map

     @param patternImages The acquired pattern images to decode
     (vector<vector<GpuMat>>), loaded as grayscale and previously rectified.
     @param disparityMap The decoding result: a CV_64F Mat at image resolution,
     storing the computed disparity map.
     @param flags Flags setting decoding algorithms. Default:
     SINUSOIDAL_COMPLEMENTARY_GRAY_CODE.
     @note All the images must be at the same resolution.
     @param stream CUDA asynchronous streams.
     */
    CV_WRAP
    virtual bool decode(const std::vector<std::vector<Mat>> &patternImages,
                        Mat &disparityMap,
                        const int flags = SINUSOIDAL_COMPLEMENTARY_GRAY_CODE,
                        Stream &stream = Stream::Null()) const = 0;
};

//! @}

} // namespace cuda
} // namespace cv

#endif //!__OPENCV_CUDA_STRUCTURED_LIGHT_HPP__
