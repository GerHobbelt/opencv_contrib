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

#include "opencv2/cudastructuredlight/cuda_multi_view_stereo_geometry_pattern.hpp"

#include "cuda/cuda.hpp"

namespace cv {
namespace cuda {

using namespace device::cudastructuredlight;

class CV_EXPORTS_W MultiViewStereoGeometryPattern_Impl CV_FINAL
    : public MultiViewStereoGeometryPattern {
  public:
    // Constructor
    explicit MultiViewStereoGeometryPattern_Impl(
        const MultiViewStereoGeometryPattern::Params parameters =
            MultiViewStereoGeometryPattern::Params());
    // Destructor
    virtual ~MultiViewStereoGeometryPattern_Impl() CV_OVERRIDE{};

    // Generate psp sinusoidal patterns
    bool generate(std::vector<cv::Mat> &patternImages) CV_OVERRIDE;

    // decode patterns and compute disparity map.
    bool decode(const std::vector<std::vector<Mat>> &patternImages,
                Mat &disparityMap, const int flags = 0,
                Stream &stream = Stream::Null()) const CV_OVERRIDE;

    // Compute a wrapped phase map from the sinusoidal patterns
    void computeWrappedAndConfidenceMap(
        const GpuMat &patternImages, GpuMat &wrappedPhaseMap,
        GpuMat &confidenceMap,
        Stream &stream = Stream::Null()) const CV_OVERRIDE;

    // Unwrap the wrapped phase map to remove phase ambiguities
    void unwrapPhaseMap(const GpuMat &refUnwapMap,
                        const GpuMat &wrappedPhaseMap,
                        const GpuMat &confidenceMap, GpuMat &unwrappedPhaseMap,
                        Stream &stream = Stream::Null()) const CV_OVERRIDE;

    // Use polynomials to fit the structured light system model and recover the
    // rough depth from the absolute phase map.
    void polynomialFitting(const GpuMat &unwrappedPhaseMap,
                           GpuMat &coarseDepthMap,
                           Stream &stream = Stream::Null()) const CV_OVERRIDE;

    // Use multi-view stereo geometry constraints to remove rough depth map
    // noise or optimize accuracy.
    void multiViewStereoRefineDepth(
        const GpuMat &coarseDepthMap, const GpuMat &wrappedMap1,
        const GpuMat &confidenceMap1, const GpuMat &wrappedMap2,
        const GpuMat &confidenceMap2, const GpuMat &wrappedMap3,
        const GpuMat &confidenceMap3, GpuMat &refineDepthMap,
        Stream &stream = Stream::Null()) const CV_OVERRIDE;

  private:
    Params params;
};

// Default parameters value
MultiViewStereoGeometryPattern_Impl::Params::Params() {
    width = 1280;
    height = 720;
    nbrOfPeriods = 40;
    shiftTime = 4;
    minDisparity = 0;
    maxDisparity = 320;
    horizontal = false;
    confidenceThreshold = 5.f;
    maxCost = 0.1f;
    costMinDiff = 0.0001f;
    costMaxDiff = 1.f;
}

MultiViewStereoGeometryPattern_Impl::MultiViewStereoGeometryPattern_Impl(
    const MultiViewStereoGeometryPattern_Impl::Params parameters)
    : params(parameters) {}

void MultiViewStereoGeometryPattern_Impl::computeWrappedAndConfidenceMap(
    const GpuMat &patternImages, GpuMat &wrappedPhaseMap, GpuMat &confidenceMap,
    Stream &stream) const {

    calcPSPWrappedAndConfidenceMap(patternImages, wrappedPhaseMap,
                                   confidenceMap, stream);
}

void MultiViewStereoGeometryPattern_Impl::unwrapPhaseMap(
    const GpuMat &refUnwapMap, const GpuMat &wrappedPhaseMap,
    const GpuMat &confidenceMap, GpuMat &unwrappedPhaseMap,
    Stream &stream) const {
    unwrapWithRefUnwrappedMap(wrappedPhaseMap, confidenceMap, refUnwapMap,
                              unwrappedPhaseMap, params.confidenceThreshold,
                              stream);
}

bool MultiViewStereoGeometryPattern_Impl::generate(
    std::vector<cv::Mat> &pattern) {
    pattern.clear();

    const int height = params.horizontal ? params.width : params.height;
    const int width = params.horizontal ? params.height : params.width;
    const int pixelsPerPeriod = width / params.nbrOfPeriods;
    // generate phase-shift imgs.
    for (int i = 0; i < params.shiftTime; ++i) {
        Mat intensityMap = Mat::zeros(height, width, CV_8UC1);
        const float shiftVal =
            static_cast<float>(CV_2PI) / params.shiftTime * i;

        for (int j = 0; j < height; ++j) {
            auto intensityMapPtr = intensityMap.ptr<uchar>(j);
            for (int k = 0; k < width; ++k) {
                // Set the fringe starting intensity to 0 so that it corresponds
                // to the complementary graycode interval.
                const float wrappedPhaseVal =
                    (k % pixelsPerPeriod) /
                        static_cast<float>(pixelsPerPeriod) *
                        static_cast<float>(CV_2PI) -
                    static_cast<float>(CV_PI);
                intensityMapPtr[k] = static_cast<uchar>(
                    127.5 + 127.5 * cos(wrappedPhaseVal + shiftVal));
            }
        }

        intensityMap = params.horizontal ? intensityMap.t() : intensityMap;
        pattern.push_back(intensityMap);
    }

    return true;
}

void MultiViewStereoGeometryPattern_Impl::polynomialFitting(
    const GpuMat &unwrappedPhaseMap, GpuMat &coarseDepthMap,
    Stream &stream) const {
    /*
    device::cudastructuredlight::polynomialFitting(unwrappedPhaseMap, params.M1,
                               params.K, params.minDepth, params.maxDepth,
                               coarseDepthMap, stream);
    */
    reverseCamera(
        unwrappedPhaseMap, params.PL1, params.PR4, params.minDepth,
        params.maxDepth,
        static_cast<float>(params.horizontal ? params.height : params.width) /
            params.nbrOfPeriods,
        coarseDepthMap, params.horizontal, stream);
}

void MultiViewStereoGeometryPattern_Impl::multiViewStereoRefineDepth(
    const GpuMat &coarseDepthMap, const GpuMat &wrappedMap1,
    const GpuMat &confidenceMap1, const GpuMat &wrappedMap2,
    const GpuMat &confidenceMap2, const GpuMat &wrappedMap3,
    const GpuMat &confidenceMap3, GpuMat &refineDepthMap,
    Stream &stream) const {
    // TODO@LiuYunhuang:测试垂直相位情况
    multiViewStereoGeometry(
        coarseDepthMap, params.M1, wrappedMap1, confidenceMap1, params.M2,
        params.R12, params.T12, wrappedMap2, confidenceMap2, params.M3,
        params.R13, params.T13, wrappedMap3, confidenceMap3, params.PL1,
        params.PR2, refineDepthMap, params.confidenceThreshold, params.maxCost,
        params.horizontal, stream);
}

bool MultiViewStereoGeometryPattern_Impl::decode(
    const std::vector<std::vector<Mat>> &patternImages, Mat &depthMap,
    const int flags, Stream &stream) const {
    CV_Assert(!patternImages.empty() && patternImages.size() == 3);

    if (flags == MULTI_VIEW_STEREO_GEOMETRY) {
        std::vector<Mat> imgsDividedMerged(
            3); // PhaseImgs-Camera1, PhaseImgs-Camera2, PhaseImgs-Camera3
        parallel_for_(Range(0, 3), [&](const Range &range) {
            for (int i = range.start; i < range.end; ++i) {
                merge(patternImages[i], imgsDividedMerged[i]);
            }
        });

        std::vector<Stream> streams(3, Stream(cudaStreamNonBlocking));
        std::vector<GpuMat> imgsDividedMergedDev(3);
        std::vector<GpuMat> wrappedMapDev(3), confidenceMap(3);
        GpuMat unwrappedMap, coarseDepthMap, refineDepthMap;
        imgsDividedMergedDev[0].upload(imgsDividedMerged[0], streams[0]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[0],
                                       wrappedMapDev[0], confidenceMap[0],
                                       streams[0]);
        imgsDividedMergedDev[1].upload(imgsDividedMerged[1], streams[1]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[1],
                                       wrappedMapDev[1], confidenceMap[1],
                                       streams[1]);
        imgsDividedMergedDev[2].upload(imgsDividedMerged[2], streams[2]);
        computeWrappedAndConfidenceMap(imgsDividedMergedDev[2],
                                       wrappedMapDev[2], confidenceMap[2],
                                       streams[2]);
        unwrapPhaseMap(params.refUnwrappedMap, wrappedMapDev[0],
                       confidenceMap[0], unwrappedMap, streams[0]);
        polynomialFitting(unwrappedMap, coarseDepthMap, streams[0]);

        streams[0].waitForCompletion();
        streams[1].waitForCompletion();
        streams[2].waitForCompletion();

        multiViewStereoRefineDepth(coarseDepthMap, wrappedMapDev[0],
                                   confidenceMap[0], wrappedMapDev[1],
                                   confidenceMap[1], wrappedMapDev[2],
                                   confidenceMap[2], refineDepthMap);

        refineDepthMap.download(depthMap);
    }

    return true;
}

Ptr<MultiViewStereoGeometryPattern> MultiViewStereoGeometryPattern::create(
    const MultiViewStereoGeometryPattern::Params &params) {
    return makePtr<MultiViewStereoGeometryPattern_Impl>(params);
}
} // namespace cuda
} // namespace cv
