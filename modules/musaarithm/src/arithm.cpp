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

void cv::musa::gemm(InputArray, InputArray, double, InputArray, double, OutputArray, int, Stream&) { throw_no_musa(); }

void cv::musa::mulSpectrums(InputArray, InputArray, OutputArray, int, bool, Stream&) { throw_no_musa(); }
void cv::musa::mulAndScaleSpectrums(InputArray, InputArray, OutputArray, int, float, bool, Stream&) { throw_no_musa(); }

void cv::musa::dft(InputArray, OutputArray, Size, int, Stream&) { throw_no_musa(); }

Ptr<Convolution> cv::musa::createConvolution(Size) { throw_no_musa(); return Ptr<Convolution>(); }

#else /* !defined (HAVE_MUSA) */

namespace
{
    #define error_entry(entry)  { entry, #entry }

    struct ErrorEntry
    {
        int code;
        const char* str;
    };

    struct ErrorEntryComparer
    {
        int code;
        ErrorEntryComparer(int code_) : code(code_) {}
        bool operator()(const ErrorEntry& e) const { return e.code == code; }
    };

    String getErrorString(int code, const ErrorEntry* errors, size_t n)
    {
        size_t idx = std::find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

        const char* msg = (idx != n) ? errors[idx].str : "Unknown error code";
        String str = cv::format("%s [Code = %d]", msg, code);

        return str;
    }
}

#ifdef HAVE_MUBLAS
    namespace
    {
        const ErrorEntry mublas_errors[] =
        {
            error_entry( MUBLAS_STATUS_SUCCESS ),
            // error_entry( MUBLAS_STATUS_NOT_INITIALIZED ),
            // error_entry( MUBLAS_STATUS_ALLOC_FAILED ),
            error_entry( MUBLAS_STATUS_INVALID_VALUE ),
            // error_entry( MUBLAS_STATUS_ARCH_MISMATCH ),
            // error_entry( MUBLAS_STATUS_MAPPING_ERROR ),
            // error_entry( MUBLAS_STATUS_EXECUTION_FAILED ),
            // error_entry( MUBLAS_STATUS_INTERNAL_ERROR )
        };

        const size_t mublas_error_num = sizeof(mublas_errors) / sizeof(mublas_errors[0]);

        static inline void ___mublasSafeCall(mublasStatus_t err, const char* file, const int line, const char* func)
        {
            if (MUBLAS_STATUS_SUCCESS != err)
            {
                String msg = getErrorString(err, mublas_errors, mublas_error_num);
                cv::error(cv::Error::GpuApiCallError, msg, func, file, line);
            }
        }
    }

    #define mublasSafeCall(expr)  ___mublasSafeCall(expr, __FILE__, __LINE__, CV_Func)
#endif // HAVE_MUBLAS

#ifdef HAVE_MUFFT
    namespace
    {
        //////////////////////////////////////////////////////////////////////////
        // MUFFT errors

        const ErrorEntry mufft_errors[] =
        {
            error_entry( MUFFT_INVALID_PLAN ),
            error_entry( MUFFT_ALLOC_FAILED ),
            error_entry( MUFFT_INVALID_TYPE ),
            error_entry( MUFFT_INVALID_VALUE ),
            error_entry( MUFFT_INTERNAL_ERROR ),
            error_entry( MUFFT_EXEC_FAILED ),
            error_entry( MUFFT_SETUP_FAILED ),
            error_entry( MUFFT_INVALID_SIZE ),
            error_entry( MUFFT_UNALIGNED_DATA )
        };

        const int mufft_error_num = sizeof(mufft_errors) / sizeof(mufft_errors[0]);

        void ___mufftSafeCall(int err, const char* file, const int line, const char* func)
        {
            if (MUFFT_SUCCESS != err)
            {
                String msg = getErrorString(err, mufft_errors, mufft_error_num);
                cv::error(cv::Error::GpuApiCallError, msg, func, file, line);
            }
        }
    }

    #define mufftSafeCall(expr)  ___mufftSafeCall(expr, __FILE__, __LINE__, CV_Func)

#endif

////////////////////////////////////////////////////////////////////////
// gemm

void cv::musa::gemm(InputArray _src1, InputArray _src2, double alpha, InputArray _src3, double beta, OutputArray _dst, int flags, Stream& stream)
{
#ifndef HAVE_MUBLAS
    CV_UNUSED(_src1);
    CV_UNUSED(_src2);
    CV_UNUSED(alpha);
    CV_UNUSED(_src3);
    CV_UNUSED(beta);
    CV_UNUSED(_dst);
    CV_UNUSED(flags);
    CV_UNUSED(stream);
    CV_Error(Error::StsNotImplemented, "The library was build without MUBLAS");
#else
    // MUBLAS works with column-major matrices

    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);
    GpuMat src3 = getInputMat(_src3, stream);

    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_32FC2 || src1.type() == CV_64FC1 || src1.type() == CV_64FC2 );
    CV_Assert( src2.type() == src1.type() && (src3.empty() || src3.type() == src1.type()) );

    if (src1.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    bool tr1 = (flags & GEMM_1_T) != 0;
    bool tr2 = (flags & GEMM_2_T) != 0;
    bool tr3 = (flags & GEMM_3_T) != 0;

    if (src1.type() == CV_64FC2)
    {
        if (tr1 || tr2 || tr3)
            CV_Error(cv::Error::StsNotImplemented, "transpose operation doesn't implemented for CV_64FC2 type");
    }

    Size src1Size = tr1 ? Size(src1.rows, src1.cols) : src1.size();
    Size src2Size = tr2 ? Size(src2.rows, src2.cols) : src2.size();
    Size src3Size = tr3 ? Size(src3.rows, src3.cols) : src3.size();
    Size dstSize(src2Size.width, src1Size.height);

    CV_Assert( src1Size.width == src2Size.height );
    CV_Assert( src3.empty() || src3Size == dstSize );

    GpuMat dst = getOutputMat(_dst, dstSize, src1.type(), stream);

    if (beta != 0)
    {
        if (src3.empty())
        {
            dst.setTo(Scalar::all(0), stream);
        }
        else
        {
            if (tr3)
            {
                musa::transpose(src3, dst, stream);
            }
            else
            {
                src3.copyTo(dst, stream);
            }
        }
    }

    mublasHandle_t handle;
    mublasSafeCall( mublasCreate(&handle) );

    mublasSafeCall( mublasSetStream(handle, StreamAccessor::getStream(stream)) );

    mublasSafeCall( mublasSetPointerMode(handle, MUBLAS_POINTER_MODE_HOST) );

    const float alphaf = static_cast<float>(alpha);
    const float betaf = static_cast<float>(beta);

    const muComplex alphacf = make_muComplex(alphaf, 0);
    const muComplex betacf = make_muComplex(betaf, 0);

    const muDoubleComplex alphac = make_muDoubleComplex(alpha, 0);
    const muDoubleComplex betac = make_muDoubleComplex(beta, 0);

    mublasOperation_t transa = tr2 ? MUBLAS_OP_T : MUBLAS_OP_N;
    mublasOperation_t transb = tr1 ? MUBLAS_OP_T : MUBLAS_OP_N;

    switch (src1.type())
    {
    case CV_32FC1:
        mublasSafeCall( mublasSgemm(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphaf,
            src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
            src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
            &betaf,
            dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
        break;

    case CV_64FC1:
        mublasSafeCall( mublasDgemm(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alpha,
            src2.ptr<double>(), static_cast<int>(src2.step / sizeof(double)),
            src1.ptr<double>(), static_cast<int>(src1.step / sizeof(double)),
            &beta,
            dst.ptr<double>(), static_cast<int>(dst.step / sizeof(double))) );
        break;

    case CV_32FC2:
        mublasSafeCall( mublasCgemm(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            reinterpret_cast<const muComplex*>(&alphacf),
            src2.ptr<muComplex>(), static_cast<int>(src2.step / sizeof(muComplex)),
            src1.ptr<muComplex>(), static_cast<int>(src1.step / sizeof(muComplex)),
            reinterpret_cast<const muComplex*>(&betacf),
            dst.ptr<muComplex>(), static_cast<int>(dst.step / sizeof(muComplex))) );
        break;

    case CV_64FC2:
        mublasSafeCall( mublasZgemm(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            reinterpret_cast<const muDoubleComplex*>(&alphac),
            src2.ptr<muDoubleComplex>(), static_cast<int>(src2.step / sizeof(muDoubleComplex)),
            src1.ptr<muDoubleComplex>(), static_cast<int>(src1.step / sizeof(muDoubleComplex)),
            reinterpret_cast<const muDoubleComplex*>(&betac),
            dst.ptr<muDoubleComplex>(), static_cast<int>(dst.step / sizeof(muDoubleComplex))) );
        break;
    }

    mublasSafeCall( mublasDestroy(handle) );

    syncOutput(dst, _dst, stream);
#endif
}

//////////////////////////////////////////////////////////////////////////////
// DFT function

void cv::musa::dft(InputArray _src, OutputArray _dst, Size dft_size, int flags, Stream& stream)
{
    if (getInputMat(_src, stream).channels() == 2)
        flags |= DFT_COMPLEX_INPUT;

    Ptr<DFT> dft = createDFT(dft_size, flags);
    dft->compute(_src, _dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
// DFT algorithm

#ifdef HAVE_MUFFT

namespace
{

    class DFTImpl : public DFT
    {
        Size dft_size, dft_size_opt;
        bool is_1d_input, is_row_dft, is_scaled_dft, is_inverse, is_complex_input, is_complex_output;

        mufftType dft_type;
        mufftHandle plan;

    public:
        DFTImpl(Size dft_size, int flags)
            : dft_size(dft_size),
              dft_size_opt(dft_size),
              is_1d_input((dft_size.height == 1) || (dft_size.width == 1)),
              is_row_dft((flags & DFT_ROWS) != 0),
              is_scaled_dft((flags & DFT_SCALE) != 0),
              is_inverse((flags & DFT_INVERSE) != 0),
              is_complex_input((flags & DFT_COMPLEX_INPUT) != 0),
              is_complex_output(!(flags & DFT_REAL_OUTPUT)),
              dft_type(!is_complex_input ? MUFFT_R2C : (is_complex_output ? MUFFT_C2C : MUFFT_C2R))
        {
            // We don't support unpacked output (in the case of real input)
            CV_Assert( !(flags & DFT_COMPLEX_OUTPUT) );

            // We don't support real-to-real transform
            CV_Assert( is_complex_input || is_complex_output );

            if (is_1d_input && !is_row_dft)
            {
                // If the source matrix is single column handle it as single row
                dft_size_opt.width = std::max(dft_size.width, dft_size.height);
                dft_size_opt.height = std::min(dft_size.width, dft_size.height);
            }

            CV_Assert( dft_size_opt.width > 1 );

            if (is_1d_input || is_row_dft)
                mufftSafeCall( mufftPlan1d(&plan, dft_size_opt.width, dft_type, dft_size_opt.height) );
            else
                mufftSafeCall( mufftPlan2d(&plan, dft_size_opt.height, dft_size_opt.width, dft_type) );
        }

        ~DFTImpl()
        {
            mufftSafeCall( mufftDestroy(plan) );
        }

        void compute(InputArray _src, OutputArray _dst, Stream& stream)
        {
            GpuMat src = getInputMat(_src, stream);

            CV_Assert( src.type() == CV_32FC1 || src.type() == CV_32FC2 );
            CV_Assert( is_complex_input == (src.channels() == 2) );

            // Make sure here we work with the continuous input,
            // as MUFFT can't handle gaps
            GpuMat src_cont;
            if (src.isContinuous())
            {
                src_cont = src;
            }
            else
            {
                BufferPool pool(stream);
                src_cont.allocator = pool.getAllocator();
                createContinuous(src.rows, src.cols, src.type(), src_cont);
                src.copyTo(src_cont, stream);
            }

            mufftSafeCall( mufftSetStream(plan, StreamAccessor::getStream(stream)) );

            if (is_complex_input)
            {
                if (is_complex_output)
                {
                    createContinuous(dft_size, CV_32FC2, _dst);
                    GpuMat dst = _dst.getMUSAGpuMat();

                    mufftSafeCall(mufftExecC2C(
                            plan, src_cont.ptr<mufftComplex>(), dst.ptr<mufftComplex>(),
                            is_inverse ? MUFFT_INVERSE : MUFFT_FORWARD));
                }
                else
                {
                    createContinuous(dft_size, CV_32F, _dst);
                    GpuMat dst = _dst.getMUSAGpuMat();

                    mufftSafeCall(mufftExecC2R(
                            plan, src_cont.ptr<mufftComplex>(), dst.ptr<mufftReal>()));
                }
            }
            else
            {
                // We could swap dft_size for efficiency. Here we must reflect it
                if (dft_size == dft_size_opt)
                    createContinuous(Size(dft_size.width / 2 + 1, dft_size.height), CV_32FC2, _dst);
                else
                    createContinuous(Size(dft_size.width, dft_size.height / 2 + 1), CV_32FC2, _dst);

                GpuMat dst = _dst.getMUSAGpuMat();

                mufftSafeCall(mufftExecR2C(
                                  plan, src_cont.ptr<mufftReal>(), dst.ptr<mufftComplex>()));
            }

            if (is_scaled_dft)
                musa::multiply(_dst, Scalar::all(1. / dft_size.area()), _dst, 1, -1, stream);
        }
    };
}

#endif

Ptr<DFT> cv::musa::createDFT(Size dft_size, int flags)
{
#ifndef HAVE_MUFFT
    CV_UNUSED(dft_size);
    CV_UNUSED(flags);
    CV_Error(Error::StsNotImplemented, "The library was build without MUFFT");
    return Ptr<DFT>();
#else
    return makePtr<DFTImpl>(dft_size, flags);
#endif
}

//////////////////////////////////////////////////////////////////////////////
// Convolution

#ifdef HAVE_MUFFT

namespace
{
    class ConvolutionImpl : public Convolution
    {
    public:
        explicit ConvolutionImpl(Size user_block_size_) : user_block_size(user_block_size_) {}

        void convolve(InputArray image, InputArray templ, OutputArray result, bool ccorr = false, Stream& stream = Stream::Null());

    private:
        void create(Size image_size, Size templ_size);
        static Size estimateBlockSize(Size result_size);

        Size result_size;
        Size block_size;
        Size user_block_size;
        Size dft_size;

        GpuMat image_spect, templ_spect, result_spect;
        GpuMat image_block, templ_block, result_data;
    };

    void ConvolutionImpl::create(Size image_size, Size templ_size)
    {
        result_size = Size(image_size.width - templ_size.width + 1,
                           image_size.height - templ_size.height + 1);

        block_size = user_block_size;
        if (user_block_size.width == 0 || user_block_size.height == 0)
            block_size = estimateBlockSize(result_size);

        dft_size.width = 1 << int(ceil(std::log(block_size.width + templ_size.width - 1.) / std::log(2.)));
        dft_size.height = 1 << int(ceil(std::log(block_size.height + templ_size.height - 1.) / std::log(2.)));

        // MUFFT has hard-coded kernels for power-of-2 sizes (up to 8192),
        // see MUSA Toolkit 4.1 MUFFT Library Programming Guide
        if (dft_size.width > 8192)
            dft_size.width = getOptimalDFTSize(block_size.width + templ_size.width - 1);
        if (dft_size.height > 8192)
            dft_size.height = getOptimalDFTSize(block_size.height + templ_size.height - 1);

        // To avoid wasting time doing small DFTs
        dft_size.width = std::max(dft_size.width, 512);
        dft_size.height = std::max(dft_size.height, 512);

        createContinuous(dft_size, CV_32F, image_block);
        createContinuous(dft_size, CV_32F, templ_block);
        createContinuous(dft_size, CV_32F, result_data);

        int spect_len = dft_size.height * (dft_size.width / 2 + 1);
        createContinuous(1, spect_len, CV_32FC2, image_spect);
        createContinuous(1, spect_len, CV_32FC2, templ_spect);
        createContinuous(1, spect_len, CV_32FC2, result_spect);

        // Use maximum result matrix block size for the estimated DFT block size
        block_size.width = std::min(dft_size.width - templ_size.width + 1, result_size.width);
        block_size.height = std::min(dft_size.height - templ_size.height + 1, result_size.height);
    }

    Size ConvolutionImpl::estimateBlockSize(Size result_size)
    {
        int width = (result_size.width + 2) / 3;
        int height = (result_size.height + 2) / 3;
        width = std::min(width, result_size.width);
        height = std::min(height, result_size.height);
        return Size(width, height);
    }

    void ConvolutionImpl::convolve(InputArray _image, InputArray _templ, OutputArray _result, bool ccorr, Stream& _stream)
    {
        GpuMat image = getInputMat(_image, _stream);
        GpuMat templ = getInputMat(_templ, _stream);

        CV_Assert( image.type() == CV_32FC1 );
        CV_Assert( templ.type() == CV_32FC1 );

        create(image.size(), templ.size());

        GpuMat result = getOutputMat(_result, result_size, CV_32FC1, _stream);

        musaStream_t stream = StreamAccessor::getStream(_stream);

        mufftHandle planR2C, planC2R;
        mufftSafeCall( mufftPlan2d(&planC2R, dft_size.height, dft_size.width, MUFFT_C2R) );
        mufftSafeCall( mufftPlan2d(&planR2C, dft_size.height, dft_size.width, MUFFT_R2C) );

        mufftSafeCall( mufftSetStream(planR2C, stream) );
        mufftSafeCall( mufftSetStream(planC2R, stream) );

        GpuMat templ_roi(templ.size(), CV_32FC1, templ.data, templ.step);
        musa::copyMakeBorder(templ_roi, templ_block, 0, templ_block.rows - templ_roi.rows, 0,
                            templ_block.cols - templ_roi.cols, 0, Scalar(), _stream);

        mufftSafeCall( mufftExecR2C(planR2C, templ_block.ptr<mufftReal>(), templ_spect.ptr<mufftComplex>()) );

        // Process all blocks of the result matrix
        for (int y = 0; y < result.rows; y += block_size.height)
        {
            for (int x = 0; x < result.cols; x += block_size.width)
            {
                Size image_roi_size(std::min(x + dft_size.width, image.cols) - x,
                                    std::min(y + dft_size.height, image.rows) - y);
                GpuMat image_roi(image_roi_size, CV_32F, (void*)(image.ptr<float>(y) + x),
                                 image.step);
                musa::copyMakeBorder(image_roi, image_block, 0, image_block.rows - image_roi.rows,
                                    0, image_block.cols - image_roi.cols, 0, Scalar(), _stream);

                mufftSafeCall(mufftExecR2C(planR2C, image_block.ptr<mufftReal>(),
                                           image_spect.ptr<mufftComplex>()));
                musa::mulAndScaleSpectrums(image_spect, templ_spect, result_spect, 0,
                                          1.f / dft_size.area(), ccorr, _stream);
                mufftSafeCall(mufftExecC2R(planC2R, result_spect.ptr<mufftComplex>(),
                                           result_data.ptr<mufftReal>()));

                Size result_roi_size(std::min(x + block_size.width, result.cols) - x,
                                     std::min(y + block_size.height, result.rows) - y);
                GpuMat result_roi(result_roi_size, result.type(),
                                  (void*)(result.ptr<float>(y) + x), result.step);
                GpuMat result_block(result_roi_size, result_data.type(),
                                    result_data.ptr(), result_data.step);

                result_block.copyTo(result_roi, _stream);
            }
        }

        mufftSafeCall( mufftDestroy(planR2C) );
        mufftSafeCall( mufftDestroy(planC2R) );

        syncOutput(result, _result, _stream);
    }
}

#endif

Ptr<Convolution> cv::musa::createConvolution(Size user_block_size)
{
#ifndef HAVE_MUFFT
    CV_UNUSED(user_block_size);
    CV_Error(Error::StsNotImplemented, "The library was build without MUFFT");
    return Ptr<Convolution>();
#else
    return makePtr<ConvolutionImpl>(user_block_size);
#endif
}

#endif /* !defined (HAVE_MUSA) */
