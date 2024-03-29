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
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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
 //     this list of conditions and the following disclaimer in the
 documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote
 products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is"
 and
 // any express or implied warranties, including, but not limited to, the
 implied
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

#ifndef __OPENCV_CUDA_MUTIL_VIEW_STEREO_GEOMETRY_HPP__
#define __OPENCV_CUDA_MUTIL_VIEW_STEREO_GEOMETRY_HPP__

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudastructuredlight/cudastructuredlight.hpp"

#include <Eigen/Eigen>

namespace cv {
namespace cuda {
//! @addtogroup cudastructuredlight
//! @{

/** @brief Class implementing the Multi-view Stereo Geometry pattern,
 based on @cite Willomitzer.
 *
 *  The resulting pattern consists of a sinusoidal pattern.
 *
 *  The equipment is mainly composed of a camera 1 with a very small angle
 with the projector and two other cameras 2 and 3 with a large angle with the
 projector.
 *
 *  The entire projection sequence contains the sine fringe sequence.
 For an image with a format (WIDTH, HEIGHT), a vertical sinusoidal fringe
 with a period of N has a period width of w = WIDTH / N.Firstly, the algorithm
 uses the reference phase solution method of An, Y et al. to quickly dephase
 the wrapping phase, then uses the polynomial fitting method to quickly recover
 the depth, and finally uses the multi-view stereo geometric constraint to
 optimize the coarser depth again: removing the background or improving the
 accuracy.
 *
 */
class CV_EXPORTS_W MultiViewStereoGeometryPattern
    : public StructuredLightPattern {
  public:
    /** @brief Parameters of StructuredLightPattern constructor.
     *  @param width Projector's width. Default value is 1280.
     *  @param height Projector's height. Default value is 720.
     */
    struct CV_EXPORTS Params {
        Params();
        bool horizontal;
        int width;
        int height;
        int nbrOfPeriods;
        int shiftTime;
        int minDisparity;
        int maxDisparity;
        float confidenceThreshold;
        float maxCost;
        float costMinDiff;
        float costMaxDiff;
        float minDepth;
        float maxDepth;
        GpuMat refUnwrappedMap;
        Eigen::Matrix4f PL1;
        Eigen::Matrix4f PR4;
        Eigen::Matrix4f PR2;
        Eigen::Matrix3f M1;
        Eigen::Matrix3f M2;
        Eigen::Matrix3f M3;
        Eigen::Matrix3f R12;
        Eigen::Matrix3f R13;
        Eigen::Vector3f T12;
        Eigen::Vector3f T13;
    };

    /** @brief Constructor
     @param parameters MultiViewStereoGeometryPattern parameters
     MultiViewStereoGeometryPattern::Params: the width and the height of the
     projector.
     */
    static Ptr<MultiViewStereoGeometryPattern>
    create(const MultiViewStereoGeometryPattern::Params &parameters =
               MultiViewStereoGeometryPattern::Params());
    /**
     * @brief Compute a wrapped phase map from sinusoidal patterns.
     * @param patternImages Input data to compute the wrapped phase map.
     * @param wrappedPhaseMap Wrapped phase map obtained through PSP.
     * @param confidenceMap Phase modulation diagram.
     * @param stream CUDA asynchronous streams.
     */
    CV_WRAP
    virtual void computeWrappedAndConfidenceMap(
        const GpuMat &patternImages, GpuMat &wrappedPhaseMap,
        GpuMat &confidenceMap, Stream &stream = Stream::Null()) const = 0;
    /**
     * @brief Unwrap the wrapped phase map to remove phase ambiguities.
     * @param refUnwrapMap Refer to the absolute phase
     * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
     * @param confidenceMap Phase modulation diagram.
     * @param unwrappedPhaseMap Noisy and coarse absolute phase map.
     * @param stream CUDA asynchronous streams.
     */
    CV_WRAP
    virtual void unwrapPhaseMap(const GpuMat &refUnwapMap,
                                const GpuMat &wrappedPhaseMap,
                                const GpuMat &confidenceMap,
                                GpuMat &unwrappedPhaseMap,
                                Stream &stream = Stream::Null()) const = 0;
    /**
     * @brief Use polynomials to fit the structured light system model and
     * recover the rough depth from the absolute phase map.
     *
     * @param unwrappedPhaseMap Noisy and coarse absolute phase map.
     * @param coarseDepthMap Noisy and coarse depth map.
     * @param stream CUDA asynchronous streams.
     * @return CV_WRAP
     */
    CV_WRAP
    virtual void polynomialFitting(const GpuMat &unwrappedPhaseMap,
                                   GpuMat &coarseDepthMap,
                                   Stream &stream = Stream::Null()) const = 0;

    /**
     * @brief Use multi-view stereo geometry constraints to remove rough depth
     * map noise or optimize accuracy.
     * @param coarseDepthMap Noisy and coarse depth map.
     * @param wrappedMap1 Wrap phase of camera 1.
     * @param wrappedMap2 Wrap phase of camera 2.
     * @param wrappedMap3 Wrap phase of camera 3.
     * @param refineDepthMap Extremely low-noise and accuracy-optimized depth
     * maps.
     * @param stream CUDA asynchronous streams.
     */
    CV_WRAP
    virtual void multiViewStereoRefineDepth(
        const GpuMat &coarseDepthMap, const GpuMat &wrappedMap1,
        const GpuMat &confidenceMap1, const GpuMat &wrappedMap2,
        const GpuMat &confidenceMap2, const GpuMat &wrappedMap3,
        const GpuMat &confidenceMap3, GpuMat &refineDepthMap,
        Stream &stream = Stream::Null()) const = 0;
};

//! @}

} // namespace cuda
} // namespace cv
#endif //!__OPENCV_CUDA_MUTIL_VIEW_STEREO_GEOMETRY_HPP__
