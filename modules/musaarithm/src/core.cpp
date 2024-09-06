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

#include "precomp.hpp"

using namespace cv;
using namespace cv::musa;

#if !defined (HAVE_MUSA) || defined (MUSA_DISABLER)

void cv::musa::merge(const GpuMat*, size_t, OutputArray, Stream&) { throw_no_musa(); }
void cv::musa::merge(const std::vector<GpuMat>&, OutputArray, Stream&) { throw_no_musa(); }

void cv::musa::split(InputArray, GpuMat*, Stream&) { throw_no_musa(); }
void cv::musa::split(InputArray, std::vector<GpuMat>&, Stream&) { throw_no_musa(); }

void cv::musa::transpose(InputArray, OutputArray, Stream&) { throw_no_musa(); }

void cv::musa::flip(InputArray, OutputArray, int, Stream&) { throw_no_musa(); }

void cv::musa::copyMakeBorder(InputArray, OutputArray, int, int, int, int, int, Scalar, Stream&) { throw_no_musa(); }

#else /* !defined (HAVE_MUSA) */

////////////////////////////////////////////////////////////////////////
// flip

namespace
{
    template<int DEPTH> struct muppTypeTraits;
    template<> struct muppTypeTraits<CV_8U>  { typedef MUpp8u mupp_t; };
    template<> struct muppTypeTraits<CV_8S>  { typedef MUpp8s mupp_t; };
    template<> struct muppTypeTraits<CV_16U> { typedef MUpp16u mupp_t; };
    template<> struct muppTypeTraits<CV_16S> { typedef MUpp16s mupp_t; };
    template<> struct muppTypeTraits<CV_32S> { typedef MUpp32s mupp_t; };
    template<> struct muppTypeTraits<CV_32F> { typedef MUpp32f mupp_t; };
    template<> struct muppTypeTraits<CV_64F> { typedef MUpp64f mupp_t; };

    template <int DEPTH> struct MUppMirrorFunc
    {
        typedef typename muppTypeTraits<DEPTH>::mupp_t mupp_t;

        typedef MUppStatus (*func_t)(const mupp_t* pSrc, int nSrcStep, mupp_t* pDst, int nDstStep, MUppiSize oROI, MUppiAxis flip);
    };

    template <int DEPTH, typename MUppMirrorFunc<DEPTH>::func_t func> struct MUppMirror
    {
        typedef typename MUppMirrorFunc<DEPTH>::mupp_t mupp_t;

        static void call(const GpuMat& src, GpuMat& dst, int flipCode, musaStream_t stream)
        {
            MUppStreamHandler h(stream);

            MUppiSize sz;
            sz.width  = src.cols;
            sz.height = src.rows;

            muppSafeCall( func(src.ptr<mupp_t>(), static_cast<int>(src.step),
                dst.ptr<mupp_t>(), static_cast<int>(dst.step), sz,
                (flipCode == 0 ? MUPP_HORIZONTAL_AXIS : (flipCode > 0 ? MUPP_VERTICAL_AXIS : MUPP_BOTH_AXIS))) );

            if (stream == 0)
                musaSafeCall( musaDeviceSynchronize() );
        }
    };

    template <int DEPTH> struct MUppMirrorIFunc
    {
        typedef typename muppTypeTraits<DEPTH>::mupp_t mupp_t;

        typedef MUppStatus (*func_t)(mupp_t* pSrcDst, int nSrcDstStep, MUppiSize oROI, MUppiAxis flip);
    };

    template <int DEPTH, typename MUppMirrorIFunc<DEPTH>::func_t func> struct MUppMirrorI
    {
        typedef typename MUppMirrorIFunc<DEPTH>::mupp_t mupp_t;

        static void call(GpuMat& srcDst, int flipCode, musaStream_t stream)
        {
            MUppStreamHandler h(stream);

            MUppiSize sz;
            sz.width  = srcDst.cols;
            sz.height = srcDst.rows;

            muppSafeCall( func(srcDst.ptr<mupp_t>(), static_cast<int>(srcDst.step),
                sz,
                (flipCode == 0 ? MUPP_HORIZONTAL_AXIS : (flipCode > 0 ? MUPP_VERTICAL_AXIS : MUPP_BOTH_AXIS))) );

            if (stream == 0)
                musaSafeCall( musaDeviceSynchronize() );
        }
    };
}

void cv::musa::flip(InputArray _src, OutputArray _dst, int flipCode, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int flipCode, musaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {MUppMirror<CV_8U, muppiMirror_8u_C1R>::call, 0, MUppMirror<CV_8U, muppiMirror_8u_C3R>::call, MUppMirror<CV_8U, muppiMirror_8u_C4R>::call},
        {0,0,0,0},
        {MUppMirror<CV_16U, muppiMirror_16u_C1R>::call, 0, MUppMirror<CV_16U, muppiMirror_16u_C3R>::call, MUppMirror<CV_16U, muppiMirror_16u_C4R>::call},
        {0,0,0,0},
        {MUppMirror<CV_32S, muppiMirror_32s_C1R>::call, 0, MUppMirror<CV_32S, muppiMirror_32s_C3R>::call, MUppMirror<CV_32S, muppiMirror_32s_C4R>::call},
        {MUppMirror<CV_32F, muppiMirror_32f_C1R>::call, 0, MUppMirror<CV_32F, muppiMirror_32f_C3R>::call, MUppMirror<CV_32F, muppiMirror_32f_C4R>::call}
    };

    typedef void (*ifunc_t)(GpuMat& srcDst, int flipCode, musaStream_t stream);
    static const ifunc_t ifuncs[6][4] =
    {
        {MUppMirrorI<CV_8U, muppiMirror_8u_C1IR>::call, 0, MUppMirrorI<CV_8U, muppiMirror_8u_C3IR>::call, MUppMirrorI<CV_8U, muppiMirror_8u_C4IR>::call},
        {0,0,0,0},
        {MUppMirrorI<CV_16U, muppiMirror_16u_C1IR>::call, 0, MUppMirrorI<CV_16U, muppiMirror_16u_C3IR>::call, MUppMirrorI<CV_16U, muppiMirror_16u_C4IR>::call},
        {0,0,0,0},
        {MUppMirrorI<CV_32S, muppiMirror_32s_C1IR>::call, 0, MUppMirrorI<CV_32S, muppiMirror_32s_C3IR>::call, MUppMirrorI<CV_32S, muppiMirror_32s_C4IR>::call},
        {MUppMirrorI<CV_32F, muppiMirror_32f_C1IR>::call, 0, MUppMirrorI<CV_32F, muppiMirror_32f_C3IR>::call, MUppMirrorI<CV_32F, muppiMirror_32f_C4IR>::call}
    };

    GpuMat src = getInputMat(_src, stream);

    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

    _dst.create(src.size(), src.type());
    GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);
    bool isInplace = (src.data == dst.data);
    bool isSizeOdd = (src.cols & 1) == 1 || (src.rows & 1) == 1;
    if (isInplace && isSizeOdd)
        CV_Error(Error::BadROISize, "In-place version of flip only accepts even width/height");

    if (isInplace == false)
        funcs[src.depth()][src.channels() - 1](src, dst, flipCode, StreamAccessor::getStream(stream));
    else // in-place
        ifuncs[src.depth()][src.channels() - 1](src, flipCode, StreamAccessor::getStream(stream));

    syncOutput(dst, _dst, stream);
}

#endif /* !defined (HAVE_MUSA) */
