// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::musa;

#if !defined (HAVE_MUSA) || defined (MUSA_DISABLER)

void cv::musa::connectedComponents(InputArray img_, OutputArray labels_, int connectivity,
    int ltype, ConnectedComponentsAlgorithmsTypes ccltype) { throw_no_musa(); }

#else /* !defined (HAVE_MUSA) */

namespace cv { namespace musa { namespace device { namespace imgproc {
        void BlockBasedKomuraEquivalence(const cv::musa::GpuMat& img, cv::musa::GpuMat& labels);
}}}}


void cv::musa::connectedComponents(InputArray img_, OutputArray labels_, int connectivity,
    int ltype, ConnectedComponentsAlgorithmsTypes ccltype) {
    const cv::musa::GpuMat img = img_.getMUSAGpuMat();
    cv::musa::GpuMat& labels = labels_.getMUSAGpuMatRef();

    CV_Assert(img.channels() == 1);
    CV_Assert(connectivity == 8);
    CV_Assert(ltype == CV_32S);
    CV_Assert(ccltype == CCL_BKE || ccltype == CCL_DEFAULT);

    int iDepth = img_.depth();
    CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

    labels.create(img.size(), CV_MAT_DEPTH(ltype));

    if ((ccltype == CCL_BKE || ccltype == CCL_DEFAULT) && connectivity == 8 && ltype == CV_32S) {
        using cv::musa::device::imgproc::BlockBasedKomuraEquivalence;
        BlockBasedKomuraEquivalence(img, labels);
    }

}

void cv::musa::connectedComponents(InputArray img_, OutputArray labels_, int connectivity, int ltype) {
    cv::musa::connectedComponents(img_, labels_, connectivity, ltype, CCL_DEFAULT);
}


#endif /* !defined (HAVE_MUSA) */
