// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::musa;

#if !defined (HAVE_MUSA) || defined (MUSA_DISABLER)

Ptr<LookUpTable> cv::musa::createLookUpTable(InputArray) { throw_no_musa(); return Ptr<LookUpTable>(); }

#else /* !defined (HAVE_MUSA) || defined (MUSA_DISABLER) */

// lut.hpp includes musa_runtime.h and can only be included when we have MUSA
#include "lut.hpp"

Ptr<LookUpTable> cv::musa::createLookUpTable(InputArray lut)
{
    return makePtr<LookUpTableImpl>(lut);
}

#endif
