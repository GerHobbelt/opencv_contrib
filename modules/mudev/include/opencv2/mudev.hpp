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

#pragma once

#ifndef OPENCV_MUDEV_HPP
#define OPENCV_MUDEV_HPP

#include "mudev/common.hpp"

#include "mudev/util/atomic.hpp"
#include "mudev/util/limits.hpp"
#include "mudev/util/saturate_cast.hpp"
#include "mudev/util/simd_functions.hpp"
#include "mudev/util/tuple.hpp"
#include "mudev/util/type_traits.hpp"
#include "mudev/util/vec_math.hpp"
#include "mudev/util/vec_traits.hpp"

#include "mudev/functional/color_cvt.hpp"
#include "mudev/functional/functional.hpp"
#include "mudev/functional/tuple_adapter.hpp"

#include "mudev/warp/reduce.hpp"
#include "mudev/warp/scan.hpp"
#include "mudev/warp/shuffle.hpp"
#include "mudev/warp/warp.hpp"

#include "mudev/block/block.hpp"
#include "mudev/block/dynamic_smem.hpp"
#include "mudev/block/reduce.hpp"
#include "mudev/block/scan.hpp"
#include "mudev/block/vec_distance.hpp"

#include "mudev/grid/copy.hpp"
#include "mudev/grid/reduce.hpp"
#include "mudev/grid/histogram.hpp"
#include "mudev/grid/integral.hpp"
#include "mudev/grid/pyramids.hpp"
#include "mudev/grid/reduce_to_vec.hpp"
#include "mudev/grid/split_merge.hpp"
#include "mudev/grid/transform.hpp"
#include "mudev/grid/transpose.hpp"

#include "mudev/ptr2d/constant.hpp"
#include "mudev/ptr2d/deriv.hpp"
#include "mudev/ptr2d/extrapolation.hpp"
#include "mudev/ptr2d/glob.hpp"
#include "mudev/ptr2d/gpumat.hpp"
#include "mudev/ptr2d/interpolation.hpp"
#include "mudev/ptr2d/lut.hpp"
#include "mudev/ptr2d/mask.hpp"
#include "mudev/ptr2d/remap.hpp"
#include "mudev/ptr2d/resize.hpp"
#include "mudev/ptr2d/texture.hpp"
#include "mudev/ptr2d/traits.hpp"
#include "mudev/ptr2d/transform.hpp"
#include "mudev/ptr2d/warping.hpp"
#include "mudev/ptr2d/zip.hpp"

#include "mudev/expr/binary_func.hpp"
#include "mudev/expr/binary_op.hpp"
#include "mudev/expr/color.hpp"
#include "mudev/expr/deriv.hpp"
#include "mudev/expr/expr.hpp"
#include "mudev/expr/per_element_func.hpp"
#include "mudev/expr/reduction.hpp"
#include "mudev/expr/unary_func.hpp"
#include "mudev/expr/unary_op.hpp"
#include "mudev/expr/warping.hpp"

/**
  @addtogroup musa
  @{
    @defgroup mudev Device layer
  @}
*/

#endif
