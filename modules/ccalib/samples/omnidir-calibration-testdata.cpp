#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    const std::string keys = "{help h usage ?||print this message}"
                                 "{scale0 |0.0| scaling parameter for estimating new camera instrinsics for undistort rectify}"
                                 "{scale1 |1.0| scaling parameter for estimating new camera instrinsics for undistort rectify}"
                                 "{raxis |""| rotation axis for estimating new camera instrinsics for undistort rectify}"
                                 "{rangmul |0.0| used by multiplying pi with multiplier to get rotation angle used for estimating new camera instrinsics for undistort rectify}"
                                 "{help||show help}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    double scale0 = parser.get<double>("scale0");
    double scale1 = parser.get<double>("scale1");
    std::string rotAx = parser.get<std::string>("raxis");
    double rotAngMul = parser.get<double>("rangmul");

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    cv::Size imageSize(1280, 800);
    const int n_images = 34;
    const std::string datasetRepositoryPath = "/home/stefan/Documents/DifferentProjects/OpenCVPullRequests/opencv_extra/testdata/cv/cameracalibration/fisheye";

    std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    const std::string folder = datasetRepositoryPath + "/calib-3_stereo_from_JY";
    const std::string leftImageList = folder + "/left.xml";

    cv::FileStorage fs_left(leftImageList, cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> imagePoints[i];
    fs_left.release();
    std::cout << "Loaded image points from: " << leftImageList << std::endl;
    std::cout << imagePoints.size() << std::endl;

    const std::string objectPointList = folder + "/object.xml";

    cv::FileStorage fs_object(objectPointList, cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    std::cout << "Loaded object points from: " << objectPointList << std::endl;
    std::cout << objectPoints.size() << std::endl;

    int flags = 0;
    flags |= cv::omnidir::CALIB_FIX_SKEW;

    cv::Matx33d K;
    cv::Matx14d D;
    cv::Vec<double, 1> xi;

    std::cout << "Everything loaded, starting calibration" << std::endl;

    double rms = cv::omnidir::calibrate(objectPoints, imagePoints, imageSize, K, xi, D,
                               cv::noArray(), cv::noArray(), flags, cv::TermCriteria(3, 20, 1e-6), cv::noArray());

    std::cout << "RMS of calibration: " << rms << std::endl;
    std::cout << "K: " << std::endl << K << std::endl;
    std::cout << "Xi: " << xi << std::endl;
    std::cout << "D: " << D << std::endl;

    int flagf = 0;
    flagf |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flagf |= cv::fisheye::CALIB_CHECK_COND;
    flagf |= cv::fisheye::CALIB_FIX_SKEW;

    cv::Matx33d Kf;
    cv::Matx14d Df;
    double rmsf = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, Kf, Df, cv::noArray(), cv::noArray(), flagf, cv::TermCriteria(3, 20, 1e-6));
    std::cout << "RMS of calibration of fisheye: " << rmsf << std::endl;

    cv::Mat R = cv::Mat::eye(3, 3, cv::DataType<double>::type);
    double rotAng = CV_PI * rotAngMul;

    if (rotAx == "x") {
        cv::Rodrigues(cv::Vec3d(rotAng, 0, 0), R);
    } else if (rotAx == "y") {
        cv::Rodrigues(cv::Vec3d(0, rotAng, 0), R);
    } else if (rotAx == "z") {
        cv::Rodrigues(cv::Vec3d(0, 0, rotAng), R);
    }

    cv::Mat K_new_p(3, 3, cv::DataType<double>::type);
    cv::Mat K_new_l(3, 3, cv::DataType<double>::type);

    cv::omnidir::estimateNewCameraMatrixForUndistortRectify(K, D, xi, imageSize, R, K_new_p, cv::omnidir::RECTIFY_PERSPECTIVE, scale0, scale1);
    cv::omnidir::estimateNewCameraMatrixForUndistortRectify(K, D, xi, imageSize, R, K_new_l, cv::omnidir::RECTIFY_LONGLATI, scale0, scale1);
    cv::Mat K_new_f;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, imageSize, R, K_new_f, scale0, imageSize, scale1);

    std::cout << "K_new perspective: " << std::endl << K_new_p << std::endl;
    std::cout << "K_new longilati: " << std::endl << K_new_l << std::endl;

    const std::string imageFilename = folder + "/left/stereo_pair_014.jpg";
    cv::Mat img = cv::imread(imageFilename);
    cv::Mat undistorted_p, undistorted_l, undistorted_f;

    cv::omnidir::undistortImage(img, undistorted_p, K, D, xi, cv::omnidir::RECTIFY_PERSPECTIVE, K_new_p, imageSize, R);
    cv::omnidir::undistortImage(img, undistorted_l, K, D, xi, cv::omnidir::RECTIFY_LONGLATI, K_new_l, imageSize, R);

    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(Kf, Df, R, K_new_f, imageSize, CV_16SC2, map1, map2 );
    cv::remap(img, undistorted_f, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::imshow("distorted", img);
    cv::imshow("undistorted perspective", undistorted_p);
    cv::imshow("undistorted longilati", undistorted_l);
    cv::imshow("undistorted fisheye", undistorted_f);
    cv::waitKey();

    std::string outputFilename = "calibration_results_omni_single_lens_scale0" + std::to_string(scale0) + "_scale1" + std::to_string(scale1) + ((rotAx == "x" || rotAx == "y" || rotAx == "z") ? ("_rotAx" + rotAx + "_rotAng" + std::to_string(rotAng)) : "") + ".yaml";
    cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "flags" << flags;
    fs << "K" << K;
    fs << "D" << D;
    fs << "xi" << xi;
    fs << "rms" << rms;

    fs << "scale0" << scale0;
    fs << "scale1" << scale1;
    fs << "R" << R;
    fs << "K_undist_perspective" << K_new_p;
    fs << "K_undist_longilati" << K_new_l;

    return 0;
}
