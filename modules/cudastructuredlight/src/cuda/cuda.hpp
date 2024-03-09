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

#ifndef __CUDASTRUCTUREDLIGHT_CUDA_HPP__
#define __CUDASTRUCTUREDLIGHT_CUDA_HPP__

#include "common.hpp"

#include <Eigen/Eigen>

namespace cv {
namespace cuda {
namespace device {
namespace cudastructuredlight {
void calcPSPWrappedAndConfidenceMap(const GpuMat &phaseImgs, GpuMat &wrappedMap,
                                    GpuMat &confidenceMap,
                                    Stream &stream = Stream::Null());

void unwrapSinusCompleGraycodeMap(const GpuMat &grayImgs,
                                  const GpuMat &wrappedMap,
                                  const GpuMat &confidenceMap,
                                  GpuMat &unwrappedMap,
                                  const float confidenceThreshold = 0.f,
                                  Stream &stream = Stream::Null());

void stereoMatch(const GpuMat &left, const GpuMat &right,
                 const StereoMatchParams &params, GpuMat &dispMap,
                 Stream &stream = Stream::Null());

void polynomialFitting(const GpuMat &phase, const Eigen::Matrix3f &intrinsic,
                       const Eigen::Vector<float, 8> &params,
                       const float minDepth, const float maxDepth,
                       GpuMat &depth, Stream &stream = Stream::Null());

void unwrapWithRefUnwrappedMap(const GpuMat &wrappedMap,
                               const GpuMat &confidenceMap,
                               const GpuMat &refUnwrappedMap, GpuMat &unwrapped,
                               const float confidenceThresholdVal = 5.f,
                               Stream &stream = Stream::Null());

void fromDepthGetTexture(const GpuMat &depth, const GpuMat &texture,
                         const Eigen::Matrix3f &M, const Eigen::Matrix3f &R,
                         const Eigen::Vector3f &T, GpuMat &mappedTexture,
                         Stream &stream = Stream::Null());

void generateCloud(const GpuMat &depth, const Eigen::Matrix3f &M,
                   const float minDepth, const float maxDepth, float3 *cloud,
                   Stream &stream = Stream::Null());

void reverseCamera(const GpuMat &phase, const Eigen::Matrix4f &PL,
                   const Eigen::Matrix4f &PR, const float minDepth,
                   const float maxDepth, const float pitch, GpuMat &depth,
                   const bool isHonrizon = false,
                   Stream &stream = Stream::Null());

void multiViewStereoGeometry(
    const GpuMat &coarseDepthMap, const const Eigen::Matrix3f &M1,
    const GpuMat &wrappedMap1, const GpuMat &confidenceMap1,
    const Eigen::Matrix3f &M2, const Eigen::Matrix3f &R12,
    const Eigen::Vector3f &T12, const GpuMat &wrappedMap2,
    const GpuMat &confidenceMap2, const Eigen::Matrix3f &M3,
    const Eigen::Matrix3f &R13, const Eigen::Vector3f &T13,
    const GpuMat &wrappedMap3, const GpuMat &confidenceMap3,
    const Eigen::Matrix4f &PL, const Eigen::Matrix4f &PR, GpuMat &fineDepthMap,
    const float confidenceThreshold = 5.f, const float maxCost = 0.01f,
    const bool isHonrizon = false, Stream &stream = Stream::Null());
} // namespace cudastructuredlight
} // namespace device
} // namespace cuda
} // namespace cv

#endif // !__CUDASTRUCTUREDLIGHT_CUDA_HPP__
