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
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
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

#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"
#include <vector>

namespace cv
{
namespace omnidir
{
    //! @addtogroup ccalib
    //! @{

    enum {
        CALIB_USE_GUESS             = 1, // currently not used
        CALIB_FIX_SKEW              = 2,
        CALIB_FIX_K1                = 4,
        CALIB_FIX_K2                = 8,
        CALIB_FIX_P1                = 16,
        CALIB_FIX_P2                = 32,
        CALIB_FIX_XI                = 64,
        CALIB_FIX_GAMMA             = 128,
        CALIB_FIX_CENTER            = 256
    };

    enum{
        RECTIFY_PERSPECTIVE         = 1,
        RECTIFY_CYLINDRICAL         = 2,
        RECTIFY_LONGLATI            = 3,
        RECTIFY_STEREOGRAPHIC       = 4
    };

    enum{
        XYZRGB  = 1,
        XYZ     = 2
    };
/**
 * This module was accepted as a GSoC 2015 project for OpenCV, authored by
 * Baisheng Lai, mentored by Bo Li.
 */

    /** @brief Projects points for omnidirectional camera using CMei's model

    @param objectPoints Object points in world coordinate, vector of vector of Vec3f or Mat of
    1xN/Nx1 3-channel of type CV_32F and N is the number of points. 64F is also acceptable.
    @param imagePoints Output array of image points, vector of vector of Vec2f or
    1xN/Nx1 2-channel of type CV_32F. 64F is also acceptable.
    @param rvec vector of rotation between world coordinate and camera coordinate, i.e., om
    @param tvec vector of translation between pattern coordinate and camera coordinate
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param jacobian Optional output 2Nx16 of type CV_64F jacobian matrix, contains the derivatives of
    image pixel points wrt parameters including \f$om, T, f_x, f_y, s, c_x, c_y, xi, k_1, k_2, p_1, p_2\f$.
    This matrix will be used in calibration by optimization.

    The function projects object 3D points of world coordinate to image pixels, parameter by intrinsic
    and extrinsic parameters. Also, it optionally compute a by-product: the jacobian matrix containing
    contains the derivatives of image pixel points wrt intrinsic and extrinsic parameters.
     */
    CV_EXPORTS_W void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec,
                       InputArray K, double xi, InputArray D, OutputArray jacobian = noArray());

    /** @overload */
    CV_EXPORTS void projectPoints(InputArray objectPoints, OutputArray imagePoints, const Affine3d& affine,
                        InputArray K, double xi, InputArray D, OutputArray jacobian = noArray());

    /** @brief Undistort 2D image points for omnidirectional camera using CMei's model

    @param distorted Array of distorted image points, vector of Vec2f
    or 1xN/Nx1 2-channel Mat of type CV_32F, 64F depth is also acceptable
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param R Rotation trainsform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param undistorted array of normalized object points, vector of Vec2f/Vec2d or 1xN/Nx1 2-channel Mat with the same
    depth of distorted points.
     */
    CV_EXPORTS_W void undistortPoints(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray xi, InputArray R);

    /** @brief Computes undistortion and rectification maps for omnidirectional camera image transform by a rotation R.
    It output two maps that are used for cv::remap(). If D is empty then zero distortion is used,
    if R or P is empty then identity matrices are used.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$, with depth CV_32F or CV_64F
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$, with depth CV_32F or CV_64F
    @param xi The parameter xi for CMei's model
    @param R Rotation transform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3, with depth CV_32F or CV_64F
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
    for details.
    @param map1 The first output map.
    @param map2 The second output map.
    @param flags Flags indicates the rectification type,  RECTIFY_PERSPECTIVE, RECTIFY_CYLINDRICAL, RECTIFY_LONGLATI and RECTIFY_STEREOGRAPHIC
    are supported.
     */
    CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, InputArray xi, InputArray R, InputArray P, const cv::Size& size,
        int m1type, OutputArray map1, OutputArray map2, int flags);

    /** @brief Undistort omnidirectional images to perspective images

    @param distorted The input omnidirectional image.
    @param undistorted The output undistorted image.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model.
    @param flags Flags indicates the rectification type,  RECTIFY_PERSPECTIVE, RECTIFY_CYLINDRICAL, RECTIFY_LONGLATI and RECTIFY_STEREOGRAPHIC
    @param Knew Camera matrix of the distorted image. If it is not assigned, it is just K.
    @param newSize The new image size. By default, it is the size of distorted.
    @param R Rotation matrix between the input and output images. By default, it is identity matrix.
    */
    CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray xi, int flags,
        InputArray Knew = cv::noArray(), const Size& newSize = Size(), InputArray R = Mat::eye(3, 3, CV_64F));

    /** @brief Estimates new camera intrinsic matrix for undistortion or rectification. Function is optimized for perspective (RECTIFY_PERSPECTIVE)
       and spherical (RECTIFY_LONGLATI) projection. For all other projection types supported by the omnidirectional model the new camera matrix is
       estimated as for the spherical projection.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model.
    @param imageSize Size of the image
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3)
    @param rectificationType Flag indicates the rectification type for the output, possibilities:
    RECTIFY_PERSPECTIVE, RECTIFY_CYLINDRICAL, RECTIFY_LONGLATI and RECTIFY_STEREOGRAPHIC. There are two different estimation modes: one for
    RECTIFY_PERSPECTIVE and the other one for all other projection types.
    @param scale0 If rectificationType is RECTIFY_PERSPECTIVE, this parameter sets the new focal length in the range between the min focal
    length and the max focal length (needs to be in the range of [0, 1]). For other rectificationTypes, this parameter is used to scale the
    horizontal field of view (fov) (> 1: increase fov, < 1: decrease fov, needs to be >0).
    @param scale1 If rectificationType is RECTIFY_PERSPECTIVE, this parameter is used as divisor for the new focal length. For other
    rectificationTypes, this parameter is used to scale the vertical field of view (fov) (> 1: increase fov, < 1: decrease fov, needs to be >0).
    @param newSize the new size of the image after undistortion or rectification

    Estimates new camera intrinsic matrix for undistortion or rectification depending on the rectification/projection type (rectificationType):

    - RECTIFY_PERSPECTIVE:
    Output matrix is calculated by estimating the focal length (\f$f_{new}\f$) and the camera center (\f$c_{new}\f$) of the undistorted image.
    For that, points on the border of the distorted image plane are undistorted. From these the min and max focal distance as well as the center
    point of the undistorted image can be calculated. Finally, \f$f_{new}\f$ is determined by scaling between the min and max focal length using
    scale0, dividing by scale1 and by applying a scale depending on newSize. \f$c_{new}\f$ is obtained from the center point and the focal length.
    The final output matrix looks as follows: \f[P = \vecthreethree{f_{new}(x)}{0}{c_{new}(x)}{0}{f_{new}(y)}{c_{new}(y)}{0}{0}{1}.\f]
    This is the exact same procedure as done in the fisheye model function @ref cv::fisheye::estimateNewCameraMatrixForUndistortRectify.

    - RECTIFY_LONGLATI:
    Output matrix P contains scaling parameters to convert from pixels to angles (horizontal and vertical) and the offset of the
    image center in pixels. For that, points on the border of the distorted image plane are undistorted. The minimum and maximum x and
    y values of the undistorted points are used to define points on the unit sphere. From these the horizontal and vertical field of view (fov)
    angles \f$\theta\f$ (longitude) and \f$\phi\f$ (latitude) are calculated. \f$\theta\f$ is then scaled by scale0 and \f$\phi\f$ by scale1.
    Then, the horizontal and vertical pixels per radiant ratios (\f$r_x\f$, \f$r_y\f$) are calculated as follows, using the width and height
    (\f$w_n\f$, \f$h_n\f$) of the output image (newSize):
    \f[\begin{bmatrix}r_x \\ r_y\end{bmatrix} = \begin{bmatrix}\frac{w_n}{\theta} \\ \frac{h_n}{\phi}\end{bmatrix}.\f]
    Further, the longitude and latitude angles (\f$\theta_c\f$, \f$\phi_c\f$) between the minimum points (for x and y) and the
    center point (x=0, y=0) on the unit sphere are calculated to estimate the angle offsets of the principal point. For the LONGLATI projection,
    the default fov is assumed to be \f$\pi\f$, with the center point at \f$\frac{\pi}{2}\f$. Therefore, the angle offset is calculated as
    follows: \f[\begin{bmatrix}\alpha_x \\ \alpha_y\end{bmatrix} = \begin{bmatrix}\frac{\pi}{2} - \theta_c \\ \frac{\pi}{2} - \phi_c\end{bmatrix}.\f]
    The pixel offsets of the principal point (\f$o_x\f$, \f$o_y\f$) are obtained as following:
    \f[\begin{bmatrix}c_x \\ c_y\end{bmatrix} = \begin{bmatrix}\alpha_x \cdot r_x \\ \alpha_y \cdot r_y\end{bmatrix}.\f]
    The final output matrix is then defined as follows: \f[P = \vecthreethree{r_x}{0}{-o_x}{0}{r_y}{-o_y}{0}{0}{1}.\f]
    */
    CV_EXPORTS_W void estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, InputArray xi, const Size &imageSize, InputArray R,
        OutputArray P, int rectificationType, double scale0 = 0.0, double scale1 = 1.0, const Size& newSize = Size());

    /** @brief Perform omnidirectional camera calibration, the default depth of outputs is CV_64F.

    @param objectPoints Vector of vectors of Vec3f holding object points in the object (pattern) coordinate system.
    It also can be vector of Mat with size 1xN/Nx1 and type CV_32FC3. Data with depth of 64_F is also acceptable.
    @param imagePoints Vector of vectors of Vec2f holding image points corresponding to the objectPoints.
    Must be the same size and the same type as objectPoints.
    @param size Image size of calibration images.
    @param K Output calibrated camera matrix.
    @param xi Output parameter xi for CMei's model
    @param D Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$
    @param rvecs Output rotation for each calibration image
    @param tvecs Output translation for each calibration image
    @param flags The flags that control calibrate
    @param criteria Termination criteria for optimization
    @param idx Indices of images that pass initialization and which are really used for calibration. So the size of rvecs
    and tvecs is the same as idx.total().
    */
    CV_EXPORTS_W double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size size,
        InputOutputArray K, InputOutputArray xi, InputOutputArray D, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
        int flags, TermCriteria criteria, OutputArray idx=noArray());

    /** @brief Stereo calibration for omnidirectional camera model. It computes the intrinsic parameters for two
    cameras and the extrinsic parameters between the two cameras. The default depth of outputs is CV_64F.

    @param objectPoints Object points in the object (pattern) coordinate sytem. Its type is vector<vector<Vec3f> >.
    It also can be vector of Mat with size 1xN/Nx1 and type CV_32FC3. Data with depth of 64_F is also acceptable.
    @param imagePoints1 The image points of the first camera corresponding to objectPoints, with type vector<vector<Vec2f> >.
    It must be the same size and the same type as objectPoints.
    @param imagePoints2 The image points of the second camera corresponding to objectPoints, with type vector<vector<Vec2f> >.
    It must be the same size and the same type as objectPoints.
    @param imageSize1 Image size of calibration images of the first camera.
    @param imageSize2 Image size of calibration images of the second camera.
    @param K1 Output camera matrix for the first camera.
    @param xi1 Output parameter xi of Mei's model for the first camera
    @param D1 Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the first camera
    @param K2 Output camera matrix for the second camera.
    @param xi2 Output parameter xi of CMei's model for the second camera
    @param D2 Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the second camera
    @param rvec Output rotation between the first and second camera
    @param tvec Output translation between the first and second camera
    @param rvecsL Output rotation for each image of the first camera
    @param tvecsL Output translation for each image of the first camera
    @param flags The flags that control stereoCalibrate
    @param criteria Termination criteria for optimization
    @param idx Indices of image pairs that pass initialization and which are really used for calibration. So the size of rvecsL
    and tvecsL is the same as idx.total().
    @
    */
    CV_EXPORTS_W double stereoCalibrate(InputOutputArrayOfArrays objectPoints, InputOutputArrayOfArrays imagePoints1, InputOutputArrayOfArrays imagePoints2,
        const Size& imageSize1, const Size& imageSize2, InputOutputArray K1, InputOutputArray xi1, InputOutputArray D1, InputOutputArray K2, InputOutputArray xi2,
        InputOutputArray D2, OutputArray rvec, OutputArray tvec, OutputArrayOfArrays rvecsL, OutputArrayOfArrays tvecsL, int flags, TermCriteria criteria, OutputArray idx=noArray());

    /** @brief Stereo rectification for omnidirectional camera model. It computes only the rectification rotations for
    the two cameras of the stereo pair.

    @param R Rotation between the first and second camera (rotation of second camera into first one)
    @param T Translation between the first and second camera (translation of second camera into first one)
    @param R1 Output 3x3 rotation matrix for the first camera
    @param R2 Output 3x3 rotation matrix for the second camera
    */
    CV_EXPORTS_W void stereoRectify(InputArray R, InputArray T, OutputArray R1, OutputArray R2);

    /** @brief Stereo rectification for omnidirectional camera model. Estimates the rectification rotations and new projection matrices for
    the two cameras of the stereo pair.

    @param K1 First camera intrinsic matrix.
    @param D1 First camera distortion parameters.
    @param xi1 First camera parameter xi of Mei's model.
    @param K2 Second camera intrinsic matrix.
    @param D2 Second camera distortion parameters.
    @param xi2 Second camera parameter xi of Mei's model.
    @param imageSize Size of the image used for stereo calibration.
    @param R Rotation matrix that rotates the coordinate systems of the second camera into the coordinate system of the first
    camera (see @ref stereoCalibrate ).
    @param tvec Translation vector that translates the coordinate system of the second camera into the coordinate system of the
    second camera (see @ref stereoCalibrate ).
    @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    @param P1 Output 3x4 (rectificationType=RECTIFY_PERSPECTIVE) or 3x3 (rectificationType=RECTIFY_LONGLATI) projection matrix in
    the new (rectified) coordinate systems for the first camera. See detailed function description for more info.
    @param P2 Output 3x4 (rectificationType=RECTIFY_PERSPECTIVE) or 3x3 (rectificationType=RECTIFY_LONGLATI) projection matrix in
    the new (rectified) coordinate systems for the second camera. See detailed function description for more info.
    @param Q Output 4x4 disparity-to-depth mapping matrix (see reprojectImageTo3D) if rectificationType=RECTIFY_PERSPECTIVE,
    otherwise Q is cleared (empty Mat).
    @param flags Operation flags that may be zero or @ref CALIB_ZERO_DISPARITY . If the flag is set,
    the function makes the principal points of each camera have the same pixel coordinates in the
    rectified views. And if the flag is not set, the function may still shift the images in the
    horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
    useful image area.
    @param rectificationType Flag indicates the rectification type for the output, possibilities:
    RECTIFY_PERSPECTIVE, RECTIFY_LONGLATI. Other projections are not supported for stereo rectification.
    @param newSize New image resolution after rectification. The same size should be passed to @ref initUndistortRectifyMap .
    When (0,0) is passed (default), it is set to the original imageSize. Setting it to larger value can help you preserve details in
    the original image, especially when there is a big radial distortion.
    @param scale0 If rectificationType=RECTIFY_PERSPECTIVE, this parameter sets the new focal length in the range between the min focal
    length and the max focal length (needs to be in the range of [0, 1]). For rectificationType=RECTIFY_LONGLATI, this parameter is used
    to scale the horizontal field of view (fov) (> 1: increase fov, < 1: decrease fov, needs to be >0).
    @param scale1 If rectificationType=RECTIFY_PERSPECTIVE, this parameter is used as divisor for the new focal length. For
    rectificationType=RECTIFY_LONGLATI, this parameter is used to scale the vertical field of view (fov) (> 1: increase fov,
    < 1: decrease fov, needs to be >0).

    The rectification transformations R1 and R2 are calculated calling @ref stereoRectify(InputArray R, InputArray T, OutputArray R1, OutputArray R2) .
    In addition, the new projection matrices P1 and P2 are calculated from the new camera matrices (\f$K1_{new}\f$, \f$K2_{new}\f$) estimated with
    @ref estimateNewCameraMatrixForUndistortRectify . Independent of the rectificationType, the following parameters are calculated the same:
    - \f$f_{rect}\f$: the minimum vertical focal length of \f$K1_{new}\f$ and \f$K2_{new}\f$
    - \f$c1_{rect} = c2_{rect}\f$: the average of the principal points of \f$K1_{new}\f$ and \f$K2_{new}\f$ if flags == CALIB_ZERO_DISPARITY,
    otherwise only the \f$y\f$-values of \f$c1_{rect}\f$ and \f$c2_{rect}\f$ are set to the average and the \f$x\f$-values are set to the
    \f$x\f$-values of the principal points of \f$K1_{new}\f$ and \f$K2_{new}\f$, respectively.

    Using the above parameters, P1 and P2 are defined as follows depending on the rectificationType:

    - RECTIFY_PERSPECTIVE:
    P1 and P2 are 3x4 projection matrices and they are calculated the same way as for the fisheye camera model (see @ref cv::fisheye::stereoRectify ).
    \f[P1 = \begin{bmatrix}
    f_{rect} & 0 & c1_{rect}(x) & 0 \\
    0 & f_{rect} & c1_{rect}(y) & 0 \\
    0 & 0 & 1 & 0
    \end{bmatrix},
    P2 = \begin{bmatrix}
    f_{rect} & 0 & c2_{rect}(x) & b \cdot f_{rect} \\
    0 & f_{rect} & c2_{rect}(y) & 0 \\
    0 & 0 & 1 & 0
    \end{bmatrix},\f]
    with \f$b\f$ being the baseline between the two rectified cameras.

    - RECTIFY_LONGLATI:
    \f[P1 = \begin{bmatrix}
    f_{rect} & 0 & c1_{rect}(x)\\
    0 & f_{rect} & c1_{rect}(y)\\
    0 & 0 & 1
    \end{bmatrix},
    P2 = \begin{bmatrix}
    f_{rect} & 0 & c2_{rect}(x)\\
    0 & f_{rect} & c2_{rect}(y)\\
    0 & 0 & 1
    \end{bmatrix}.\f]
    */

    CV_EXPORTS_W void stereoRectify(cv::InputArray K1, cv::InputArray D1, cv::InputArray xi1, cv::InputArray K2, cv::InputArray D2, cv::InputArray xi2, const cv::Size &imageSize, cv::InputArray R, cv::InputArray tvec,
        cv::OutputArray R1, cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2, cv::OutputArray Q, int flags, int rectificationType, const cv::Size &newSize,
        double scale0, double scale1);

    /** @brief Reprojects a disparity image created from a image pair that was stereo rectified with the omni-directional camera
    model to 3D space.

    @param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
    floating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no
    fractional bits. If the disparity is 16-bit signed format, as computed by @ref StereoBM or
    @ref StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before
    being used here.
    @param _3dImage Output 3-channel floating-point image of the same size as disparity. Each element of
    _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. Depending
    on T or Q the returned points are represented in the rectified coordinate system of the first or second
    camera.
    @param P Projection matrix of disparity image (see P1 or P2 estimated with @ref stereoRectify(cv::InputArray K1, cv::InputArray D1, cv::InputArray xi1,
    cv::InputArray K2, cv::InputArray D2, cv::InputArray xi2, const cv::Size &imageSize, cv::InputArray R, cv::InputArray tvec, cv::OutputArray R1,
    cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2, cv::OutputArray Q, int flags, int rectificationType, const cv::Size &newSize,
    double scale0, double scale1) )
    @param T Translation between the first and second camera
    @param Q \f$4 \times 4\f$ perspective transformation matrix that can be obtained with @ref stereoRectify(cv::InputArray K1, cv::InputArray D1,
    cv::InputArray xi1, cv::InputArray K2, cv::InputArray D2, cv::InputArray xi2, const cv::Size &imageSize, cv::InputArray R, cv::InputArray tvec,
    cv::OutputArray R1, cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2, cv::OutputArray Q, int flags, int rectificationType,
    const cv::Size &newSize, double scale0, double scale1).
    Only used for rectificationType=RECTIFY_PERSPECTIVE. If not provided, P and T are used to obtain Q
    (rectificationType=RECTIFY_PERSPECTIVE). If one uses Q obtained by @ref stereoRectify(cv::InputArray K1, cv::InputArray D1, cv::InputArray xi1,
    cv::InputArray K2, cv::InputArray D2, cv::InputArray xi2, const cv::Size &imageSize, cv::InputArray R, cv::InputArray tvec, cv::OutputArray R1,
    cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2, cv::OutputArray Q, int flags, int rectificationType, const cv::Size &newSize,
    double scale0, double scale1), the returned points in _3dImage are represented in the first camera's rectified coordinate system. Not used if
    rectificationType=RECTIFY_LONGLATI. By default it is @ref cv::noArray.
    @param rectificationType Flag indicates the rectification type used for stereo rectification. Supported
    are RECTIFY_PERSPECTIVE and RECTIFY_LONGLATI.
    @param handleMissingValues Indicates, whether the function should handle missing values (i.e.
    points where the disparity was not computed). If handleMissingValues=true, then pixels with the
    minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
    to 3D points with a very large Z value (currently set to 10000).
    @param ddepth The optional output array depth. If it is -1, the output image will have CV_32F
    depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.

    The function transforms a single-channel disparity map to a 3-channel image representing a 3D
    surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
    computes 3D points depending on the rectification type:

    - reftification_type=RECTIFY_PERSPECTIVE:
    \f[\begin{bmatrix}
    X \\
    Y \\
    Z \\
    W
    \end{bmatrix} = Q \begin{bmatrix}
    x \\
    y \\
    \texttt{disparity} (x,y) \\
    1
    \end{bmatrix}.\f]
    The final 3D points are obtained as follows
    \f[\begin{bmatrix}
    p_x \\
    p_y \\
    p_z
    \end{bmatrix} = \begin{bmatrix}
    X/W \\
    Y/W \\
    Z/W
    \end{bmatrix}.\f]

    - rectificationType=RECTIFY_LONGLATI:
    If rectification was done using spherical projection, pixel coordinates indicate latitude and longitude angles.
    To transform from pixel coordinates to angles, the new projection matrix P has to contain the required scaling factors
    and offsets (see @ref estimateNewCameraMatrixForUndistortRectify ) to compute theta and phi as follows:
    \f[\begin{bmatrix}
    \theta \\
    \phi
    \end{bmatrix} = \begin{bmatrix}
    x/P_{00} - P_{02}/P_{00} \\
    y/P_{11} - P_{12}/P_{11}
    \end{bmatrix}.\f]
    Finally, 3D coordinates are calculated using spherical coordinates:
    \f[\begin{bmatrix}
    p_x \\
    p_y \\
    p_z
    \end{bmatrix} = \begin{bmatrix}
    -\cos(\theta) \\
    -\sin(\theta) \cdot \cos(\phi) \\
    \sin(\theta) \cdot \sin(\phi)
    \end{bmatrix} \cdot \texttt{depth},\f]
    with \f[\texttt{depth} = |T| \cdot \frac{P_{00}}{\texttt{disparity} (x,y)}.\f]
    */
    CV_EXPORTS_W void reprojectImageTo3D( InputArray disparity,
                                          OutputArray _3dImage, InputArray P, InputArray T,
                                          InputArray Q = cv::noArray(),
                                          int rectificationType = RECTIFY_PERSPECTIVE,
                                          bool handleMissingValues = false,
                                          int ddepth = -1 );

    /** @brief Stereo 3D reconstruction from a pair of images

    @param image1 The first input image
    @param image2 The second input image
    @param K1 Input camera matrix of the first camera
    @param D1 Input distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the first camera
    @param xi1 Input parameter xi for the first camera for CMei's model
    @param K2 Input camera matrix of the second camera
    @param D2 Input distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the second camera
    @param xi2 Input parameter xi for the second camera for CMei's model
    @param R Rotation between the first and second camera
    @param T Translation between the first and second camera
    @param flag Flag of rectification type, RECTIFY_PERSPECTIVE or RECTIFY_LONGLATI
    @param numDisparities The parameter 'numDisparities' in StereoSGBM, see StereoSGBM for details.
    @param SADWindowSize The parameter 'SADWindowSize' in StereoSGBM, see StereoSGBM for details.
    @param disparity Disparity map generated by stereo matching
    @param image1Rec Rectified image of the first image
    @param image2Rec rectified image of the second image
    @param newSize Image size of rectified image, see omnidir::undistortImage
    @param Knew New camera matrix of rectified image, see omnidir::undistortImage
    @param pointCloud Point cloud of 3D reconstruction, with type CV_64FC3
    @param pointType Point cloud type, it can be XYZRGB or XYZ
    */
    CV_EXPORTS_W void stereoReconstruct(InputArray image1, InputArray image2, InputArray K1, InputArray D1, InputArray xi1,
        InputArray K2, InputArray D2, InputArray xi2, InputArray R, InputArray T, int flag, int numDisparities, int SADWindowSize,
        OutputArray disparity, OutputArray image1Rec, OutputArray image2Rec, const Size& newSize = Size(), InputArray Knew = cv::noArray(),
        OutputArray pointCloud = cv::noArray(), int pointType = XYZRGB);

namespace internal
{
    void initializeCalibration(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size size, OutputArrayOfArrays omAll,
        OutputArrayOfArrays tAll, OutputArray K, double& xi, OutputArray idx = noArray());

    void initializeStereoCalibration(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
        const Size& size1, const Size& size2, OutputArray om, OutputArray T, OutputArrayOfArrays omL, OutputArrayOfArrays tL, OutputArray K1, OutputArray D1, OutputArray K2, OutputArray D2,
        double &xi1, double &xi2, int flags, OutputArray idx);

    void computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters, Mat& JTJ_inv, Mat& JTE, int flags,
							double epsilon);

    void computeJacobianStereo(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
        InputArray parameters, Mat& JTJ_inv, Mat& JTE, int flags, double epsilon);

    void encodeParameters(InputArray K, InputArrayOfArrays omAll, InputArrayOfArrays tAll, InputArray distoaration, double xi, OutputArray parameters);

    void encodeParametersStereo(InputArray K1, InputArray K2, InputArray om, InputArray T, InputArrayOfArrays omL, InputArrayOfArrays tL,
        InputArray D1, InputArray D2, double xi1, double xi2, OutputArray parameters);

    void decodeParameters(InputArray paramsters, OutputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray distoration, double& xi);

    void decodeParametersStereo(InputArray parameters, OutputArray K1, OutputArray K2, OutputArray om, OutputArray T, OutputArrayOfArrays omL,
        OutputArrayOfArrays tL, OutputArray D1, OutputArray D2, double& xi1, double& xi2);

    void estimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters, Mat& errors, Vec2d& std_error, double& rms, int flags);

    void estimateUncertaintiesStereo(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputArray parameters, Mat& errors,
        Vec2d& std_error, double& rms, int flags);

    double computeMeanReproErr(InputArrayOfArrays imagePoints, InputArrayOfArrays proImagePoints);

    double computeMeanReproErr(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray K, InputArray D, double xi, InputArrayOfArrays omAll,
        InputArrayOfArrays tAll);

    double computeMeanReproErrStereo(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputArray K1, InputArray K2,
        InputArray D1, InputArray D2, double xi1, double xi2, InputArray om, InputArray T, InputArrayOfArrays omL, InputArrayOfArrays TL);

    void subMatrix(const Mat& src, Mat& dst, const std::vector<int>& cols, const std::vector<int>& rows);

    void flags2idx(int flags, std::vector<int>& idx, int n);

    void flags2idxStereo(int flags, std::vector<int>& idx, int n);

    void fillFixed(Mat&G, int flags, int n);

    void fillFixedStereo(Mat& G, int flags, int n);

    double findMedian(const Mat& row);

    Vec3d findMedian3(InputArray mat);

    void getInterset(InputArray idx1, InputArray idx2, OutputArray inter1, OutputArray inter2, OutputArray inter_ori);

    void compose_motion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2, Mat& om3, Mat& T3, Mat& dom3dom1,
        Mat& dom3dT1, Mat& dom3dom2, Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2);

    //void JRodriguesMatlab(const Mat& src, Mat& dst);

    //void dAB(InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB);
} // internal

//! @}

} // omnidir

} //cv
#endif
