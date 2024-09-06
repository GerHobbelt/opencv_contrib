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

void cv::musa::calcHist(InputArray, OutputArray, Stream&) { throw_no_musa(); }

void cv::musa::equalizeHist(InputArray, OutputArray, Stream&) { throw_no_musa(); }

cv::Ptr<cv::musa::CLAHE> cv::musa::createCLAHE(double, cv::Size) { throw_no_musa(); return cv::Ptr<cv::musa::CLAHE>(); }

void cv::musa::evenLevels(OutputArray, int, int, int, Stream&) { throw_no_musa(); }

void cv::musa::histEven(InputArray, OutputArray, int, int, int, Stream&) { throw_no_musa(); }
void cv::musa::histEven(InputArray, GpuMat*, int*, int*, int*, Stream&) { throw_no_musa(); }

void cv::musa::histRange(InputArray, OutputArray, InputArray, Stream&) { throw_no_musa(); }
void cv::musa::histRange(InputArray, GpuMat*, const GpuMat*, Stream&) { throw_no_musa(); }

#else /* !defined (HAVE_MUSA) */

////////////////////////////////////////////////////////////////////////
// calcHist

namespace hist
{
    void histogram256(PtrStepSzb src, int* hist, musaStream_t stream);
    void histogram256(PtrStepSzb src, PtrStepSzb mask, int* hist, musaStream_t stream);
}

void cv::musa::calcHist(InputArray _src, OutputArray _hist, Stream& stream)
{
    calcHist(_src, cv::musa::GpuMat(), _hist, stream);
}

void cv::musa::calcHist(InputArray _src, InputArray _mask, OutputArray _hist, Stream& stream)
{
    GpuMat src = _src.getMUSAGpuMat();
    GpuMat mask = _mask.getMUSAGpuMat();

    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( mask.empty() || mask.type() == CV_8UC1 );
    CV_Assert( mask.empty() || mask.size() == src.size() );

    _hist.create(1, 256, CV_32SC1);
    GpuMat hist = _hist.getMUSAGpuMat();

    hist.setTo(Scalar::all(0), stream);

    if (mask.empty())
        hist::histogram256(src, hist.ptr<int>(), StreamAccessor::getStream(stream));
    else
        hist::histogram256(src, mask, hist.ptr<int>(), StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// equalizeHist

namespace hist
{
    void equalizeHist(PtrStepSzb src, PtrStepSzb dst, const uchar* lut, musaStream_t stream);
    void buildLut(PtrStepSzi hist, PtrStepSzb lut, int size, musaStream_t stream);
}

void cv::musa::equalizeHist(InputArray _src, OutputArray _dst, Stream& _stream)
{
    GpuMat src = getInputMat(_src, _stream);

    CV_Assert( src.type() == CV_8UC1 );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getMUSAGpuMat();

    size_t bufSize = 256 * sizeof(int) + 256 * sizeof(uchar);

    BufferPool pool(_stream);
    GpuMat buf = pool.getBuffer(1, static_cast<int>(bufSize), CV_8UC1);

    GpuMat hist(1, 256, CV_32SC1, buf.data);
    GpuMat lut(1, 256, CV_8UC1, buf.data + 256 * sizeof(int));

    musa::calcHist(src, hist, _stream);

    musaStream_t stream = StreamAccessor::getStream(_stream);

    hist::buildLut(hist, lut, src.rows * src.cols, stream);

    hist::equalizeHist(src, dst, lut.data, stream);
}

////////////////////////////////////////////////////////////////////////
// CLAHE

namespace clahe
{
    void calcLut_8U(PtrStepSzb src, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, musaStream_t stream);
    void calcLut_16U(PtrStepSzus src, PtrStepus lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, PtrStepSzi hist, musaStream_t stream);
    template <typename T> void transform(PtrStepSz<T> src, PtrStepSz<T> dst, PtrStep<T> lut, int tilesX, int tilesY, int2 tileSize, musaStream_t stream);
}

namespace
{
    class CLAHE_Impl : public cv::musa::CLAHE
    {
    public:
        CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

        void apply(cv::InputArray src, cv::OutputArray dst);
        void apply(InputArray src, OutputArray dst, Stream& stream);

        void setClipLimit(double clipLimit);
        double getClipLimit() const;

        void setTilesGridSize(cv::Size tileGridSize);
        cv::Size getTilesGridSize() const;

        void collectGarbage();

    private:
        double clipLimit_;
        int tilesX_;
        int tilesY_;

        GpuMat srcExt_;
        GpuMat lut_;
        GpuMat hist_; // histogram on global memory for CV_16UC1 case
    };

    CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
        clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
    {
    }

    void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst)
    {
        apply(_src, _dst, Stream::Null());
    }

    void CLAHE_Impl::apply(InputArray _src, OutputArray _dst, Stream& s)
    {
        GpuMat src = _src.getMUSAGpuMat();

        const int type = src.type();

        CV_Assert( type == CV_8UC1 || type == CV_16UC1 );

        _dst.create( src.size(), type );
        GpuMat dst = _dst.getMUSAGpuMat();

        const int histSize = type == CV_8UC1 ? 256 : 65536;

        ensureSizeIsEnough(tilesX_ * tilesY_, histSize, type, lut_);

        musaStream_t stream = StreamAccessor::getStream(s);

        cv::Size tileSize;
        GpuMat srcForLut;

        if (src.cols % tilesX_ == 0 && src.rows % tilesY_ == 0)
        {
            tileSize = cv::Size(src.cols / tilesX_, src.rows / tilesY_);
            srcForLut = src;
        }
        else
        {
#ifndef HAVE_OPENCV_MUSAARITHM
            throw_no_musa();
#else
            cv::musa::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0, tilesX_ - (src.cols % tilesX_), cv::BORDER_REFLECT_101, cv::Scalar(), s);
#endif

            tileSize = cv::Size(srcExt_.cols / tilesX_, srcExt_.rows / tilesY_);
            srcForLut = srcExt_;
        }

        const int tileSizeTotal = tileSize.area();
        const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

        int clipLimit = 0;
        if (clipLimit_ > 0.0)
        {
            clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
            clipLimit = std::max(clipLimit, 1);
        }

        if (type == CV_8UC1)
            clahe::calcLut_8U(srcForLut, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), clipLimit, lutScale, stream);
        else // type == CV_16UC1
        {
            ensureSizeIsEnough(tilesX_ * tilesY_, histSize, CV_32SC1, hist_);
            clahe::calcLut_16U(srcForLut, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), clipLimit, lutScale, hist_, stream);
        }

        if (type == CV_8UC1)
            clahe::transform<uchar>(src, dst, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), stream);
        else // type == CV_16UC1
            clahe::transform<ushort>(src, dst, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), stream);
    }

    void CLAHE_Impl::setClipLimit(double clipLimit)
    {
        clipLimit_ = clipLimit;
    }

    double CLAHE_Impl::getClipLimit() const
    {
        return clipLimit_;
    }

    void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
    {
        tilesX_ = tileGridSize.width;
        tilesY_ = tileGridSize.height;
    }

    cv::Size CLAHE_Impl::getTilesGridSize() const
    {
        return cv::Size(tilesX_, tilesY_);
    }

    void CLAHE_Impl::collectGarbage()
    {
        srcExt_.release();
        lut_.release();
    }
}

cv::Ptr<cv::musa::CLAHE> cv::musa::createCLAHE(double clipLimit, cv::Size tileGridSize)
{
    return makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
}

////////////////////////////////////////////////////////////////////////
// MUPP Histogram

namespace
{
    typedef MUppStatus (*get_buf_size_c1_t)(MUppiSize oSizeROI, int nLevels, int* hpBufferSize);
    typedef MUppStatus (*get_buf_size_c4_t)(MUppiSize oSizeROI, int nLevels[], int* hpBufferSize);

    template<int SDEPTH> struct MUppHistogramEvenFuncC1
    {
        typedef typename MUPPTypeTraits<SDEPTH>::mupp_type src_t;

    typedef MUppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, MUppiSize oSizeROI, MUpp32s * pHist,
            int nLevels, MUpp32s nLowerLevel, MUpp32s nUpperLevel, MUpp8u * pBuffer);
    };
    template<int SDEPTH> struct MUppHistogramEvenFuncC4
    {
        typedef typename MUPPTypeTraits<SDEPTH>::mupp_type src_t;

        typedef MUppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, MUppiSize oSizeROI,
            MUpp32s * pHist[4], int nLevels[4], MUpp32s nLowerLevel[4], MUpp32s nUpperLevel[4], MUpp8u * pBuffer);
    };

    template<int SDEPTH, typename MUppHistogramEvenFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct MUppHistogramEvenC1
    {
        typedef typename MUppHistogramEvenFuncC1<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, OutputArray _hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
        {
            const int levels = histSize + 1;

            _hist.create(1, histSize, CV_32S);
            GpuMat hist = _hist.getMUSAGpuMat();

            MUppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            MUppStreamHandler h(stream);

            muppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<MUpp32s>(), levels,
                lowerLevel, upperLevel, buf.ptr<MUpp8u>()) );

            if (!stream)
                musaSafeCall( musaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename MUppHistogramEvenFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct MUppHistogramEvenC4
    {
        typedef typename MUppHistogramEvenFuncC4<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
        {
            int levels[] = {histSize[0] + 1, histSize[1] + 1, histSize[2] + 1, histSize[3] + 1};
            hist[0].create(1, histSize[0], CV_32S);
            hist[1].create(1, histSize[1], CV_32S);
            hist[2].create(1, histSize[2], CV_32S);
            hist[3].create(1, histSize[3], CV_32S);

            MUppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            MUpp32s* pHist[] = {hist[0].ptr<MUpp32s>(), hist[1].ptr<MUpp32s>(), hist[2].ptr<MUpp32s>(), hist[3].ptr<MUpp32s>()};

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            MUppStreamHandler h(stream);

            muppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, levels, lowerLevel, upperLevel, buf.ptr<MUpp8u>()) );

            if (!stream)
                musaSafeCall( musaDeviceSynchronize() );
        }
    };

    template<int SDEPTH> struct MUppHistogramRangeFuncC1
    {
        typedef typename MUPPTypeTraits<SDEPTH>::mupp_type src_t;
        typedef MUpp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef MUppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, MUppiSize oSizeROI, MUpp32s* pHist,
            const MUpp32s* pLevels, int nLevels, MUpp8u* pBuffer);
    };
    template<> struct MUppHistogramRangeFuncC1<CV_32F>
    {
        typedef MUpp32f src_t;
        typedef MUpp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef MUppStatus (*func_ptr)(const MUpp32f* pSrc, int nSrcStep, MUppiSize oSizeROI, MUpp32s* pHist,
            const MUpp32f* pLevels, int nLevels, MUpp8u* pBuffer);
    };
    template<int SDEPTH> struct MUppHistogramRangeFuncC4
    {
        typedef typename MUPPTypeTraits<SDEPTH>::mupp_type src_t;
        typedef MUpp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef MUppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, MUppiSize oSizeROI, MUpp32s* pHist[4],
            const MUpp32s* pLevels[4], int nLevels[4], MUpp8u* pBuffer);
    };
    template<> struct MUppHistogramRangeFuncC4<CV_32F>
    {
        typedef MUpp32f src_t;
        typedef MUpp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef MUppStatus (*func_ptr)(const MUpp32f* pSrc, int nSrcStep, MUppiSize oSizeROI, MUpp32s* pHist[4],
            const MUpp32f* pLevels[4], int nLevels[4], MUpp8u* pBuffer);
    };

    template<int SDEPTH, typename MUppHistogramRangeFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct MUppHistogramRangeC1
    {
        typedef typename MUppHistogramRangeFuncC1<SDEPTH>::src_t src_t;
        typedef typename MUppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=MUppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, OutputArray _hist, const GpuMat& levels, Stream& stream)
        {
            CV_Assert( levels.type() == LEVEL_TYPE_CODE && levels.rows == 1 );

            _hist.create(1, levels.cols - 1, CV_32S);
            GpuMat hist = _hist.getMUSAGpuMat();

            MUppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels.cols, &buf_size);

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            MUppStreamHandler h(stream);

            muppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<MUpp32s>(), levels.ptr<level_t>(), levels.cols, buf.ptr<MUpp8u>()) );

            if (stream == 0)
                musaSafeCall( musaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename MUppHistogramRangeFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct MUppHistogramRangeC4
    {
        typedef typename MUppHistogramRangeFuncC4<SDEPTH>::src_t src_t;
        typedef typename MUppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=MUppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
        {
            CV_Assert( levels[0].type() == LEVEL_TYPE_CODE && levels[0].rows == 1 );
            CV_Assert( levels[1].type() == LEVEL_TYPE_CODE && levels[1].rows == 1 );
            CV_Assert( levels[2].type() == LEVEL_TYPE_CODE && levels[2].rows == 1 );
            CV_Assert( levels[3].type() == LEVEL_TYPE_CODE && levels[3].rows == 1 );

            hist[0].create(1, levels[0].cols - 1, CV_32S);
            hist[1].create(1, levels[1].cols - 1, CV_32S);
            hist[2].create(1, levels[2].cols - 1, CV_32S);
            hist[3].create(1, levels[3].cols - 1, CV_32S);

            MUpp32s* pHist[] = {hist[0].ptr<MUpp32s>(), hist[1].ptr<MUpp32s>(), hist[2].ptr<MUpp32s>(), hist[3].ptr<MUpp32s>()};
            int nLevels[] = {levels[0].cols, levels[1].cols, levels[2].cols, levels[3].cols};
            const level_t* pLevels[] = {levels[0].ptr<level_t>(), levels[1].ptr<level_t>(), levels[2].ptr<level_t>(), levels[3].ptr<level_t>()};

            MUppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, nLevels, &buf_size);

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            MUppStreamHandler h(stream);

            muppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, pLevels, nLevels, buf.ptr<MUpp8u>()) );

            if (stream == 0)
                musaSafeCall( musaDeviceSynchronize() );
        }
    };
}

void cv::musa::evenLevels(OutputArray _levels, int nLevels, int lowerLevel, int upperLevel, Stream& stream)
{
    const int kind = _levels.kind();

    _levels.create(1, nLevels, CV_32SC1);

    Mat host_levels;
    if (kind == _InputArray::MUSA_GPU_MAT)
        host_levels.create(1, nLevels, CV_32SC1);
    else
        host_levels = _levels.getMat();

    muppSafeCall( muppiEvenLevelsHost_32s(host_levels.ptr<MUpp32s>(), nLevels, lowerLevel, upperLevel) );

    if (kind == _InputArray::MUSA_GPU_MAT)
        _levels.getMUSAGpuMatRef().upload(host_levels, stream);
}

namespace hist
{
    void histEven8u(PtrStepSzb src, int* hist, int binCount, int lowerLevel, int upperLevel, musaStream_t stream);
}

namespace
{
    void histEven8u(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, musaStream_t stream)
    {
        hist.create(1, histSize, CV_32S);
        musaSafeCall( musaMemsetAsync(hist.data, 0, histSize * sizeof(int), stream) );
        hist::histEven8u(src, hist.ptr<int>(), histSize, lowerLevel, upperLevel, stream);
    }
}

void cv::musa::histEven(InputArray _src, OutputArray hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, OutputArray hist, int levels, int lowerLevel, int upperLevel, Stream& stream);
    static const hist_t hist_callers[] =
    {
        MUppHistogramEvenC1<CV_8U , muppiHistogramEven_8u_C1R , muppiHistogramEvenGetBufferSize_8u_C1R >::hist,
        0,
        MUppHistogramEvenC1<CV_16U, muppiHistogramEven_16u_C1R, muppiHistogramEvenGetBufferSize_16u_C1R>::hist,
        MUppHistogramEvenC1<CV_16S, muppiHistogramEven_16s_C1R, muppiHistogramEvenGetBufferSize_16s_C1R>::hist
    };

    GpuMat src = _src.getMUSAGpuMat();

    if (src.depth() == CV_8U)
    {
        histEven8u(src, hist.getMUSAGpuMatRef(), histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
        return;
    }

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 );

    hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel, stream);
}

void cv::musa::histEven(InputArray _src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], int levels[4], int lowerLevel[4], int upperLevel[4], Stream& stream);
    static const hist_t hist_callers[] =
    {
        MUppHistogramEvenC4<CV_8U , muppiHistogramEven_8u_C4R , muppiHistogramEvenGetBufferSize_8u_C4R >::hist,
        0,
        MUppHistogramEvenC4<CV_16U, muppiHistogramEven_16u_C4R, muppiHistogramEvenGetBufferSize_16u_C4R>::hist,
        MUppHistogramEvenC4<CV_16S, muppiHistogramEven_16s_C4R, muppiHistogramEvenGetBufferSize_16s_C4R>::hist
    };

    GpuMat src = _src.getMUSAGpuMat();

    CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 );

    hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel, stream);
}

void cv::musa::histRange(InputArray _src, OutputArray hist, InputArray _levels, Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, OutputArray hist, const GpuMat& levels, Stream& stream);
    static const hist_t hist_callers[] =
    {
        MUppHistogramRangeC1<CV_8U , muppiHistogramRange_8u_C1R , muppiHistogramRangeGetBufferSize_8u_C1R >::hist,
        0,
        MUppHistogramRangeC1<CV_16U, muppiHistogramRange_16u_C1R, muppiHistogramRangeGetBufferSize_16u_C1R>::hist,
        MUppHistogramRangeC1<CV_16S, muppiHistogramRange_16s_C1R, muppiHistogramRangeGetBufferSize_16s_C1R>::hist,
        0,
        MUppHistogramRangeC1<CV_32F, muppiHistogramRange_32f_C1R, muppiHistogramRangeGetBufferSize_32f_C1R>::hist
    };

    GpuMat src = _src.getMUSAGpuMat();
    GpuMat levels = _levels.getMUSAGpuMat();

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 || src.type() == CV_32FC1 );

    hist_callers[src.depth()](src, hist, levels, stream);
}

void cv::musa::histRange(InputArray _src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream);
    static const hist_t hist_callers[] =
    {
        MUppHistogramRangeC4<CV_8U , muppiHistogramRange_8u_C4R , muppiHistogramRangeGetBufferSize_8u_C4R >::hist,
        0,
        MUppHistogramRangeC4<CV_16U, muppiHistogramRange_16u_C4R, muppiHistogramRangeGetBufferSize_16u_C4R>::hist,
        MUppHistogramRangeC4<CV_16S, muppiHistogramRange_16s_C4R, muppiHistogramRangeGetBufferSize_16s_C4R>::hist,
        0,
        MUppHistogramRangeC4<CV_32F, muppiHistogramRange_32f_C4R, muppiHistogramRangeGetBufferSize_32f_C4R>::hist
    };

    GpuMat src = _src.getMUSAGpuMat();

    CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 || src.type() == CV_32FC4 );

    hist_callers[src.depth()](src, hist, levels, stream);
}

#endif /* !defined (HAVE_MUSA) */
