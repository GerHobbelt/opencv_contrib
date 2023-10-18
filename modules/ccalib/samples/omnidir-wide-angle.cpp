#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void loadCameraParameters(const std::string& parameterFilePath, cv::Size& imageSize, cv::Mat& camMat, cv::Mat& distCoeff, cv::Mat& xi)
{
    cv::FileStorage fs(parameterFilePath, cv::FileStorage::READ);

    imageSize = cv::Size(fs["image_height"], fs["image_width"]);
    std::cout << imageSize << std::endl;
    if (imageSize.empty()) {
        cv::error(cv::Error::StsBadArg, "Could not load camera parameters from file, imageSize is empty.", __func__,
            __FILE__, __LINE__);
    }

    fs["distortion_coefficients"] >> distCoeff;
    if (distCoeff.empty()) {
        cv::error(cv::Error::StsBadArg, "Could not load camera parameters from file, distortion coefficients empty.",
            __func__, __FILE__, __LINE__);
    }

    fs["camera_matrix"] >> camMat;
    if (camMat.empty()) {
        cv::error(cv::Error::StsBadArg, "Could not load camera parameters from file, camera matrix empty.", __func__,
            __FILE__, __LINE__);
    }

    fs["xi"] >> xi;
    if (xi.empty()) {
        cv::error(cv::Error::StsBadArg,
            "Could not load camera parameters from file, xi empty, but needed for omnidirectional camera model.",
            __func__, __FILE__, __LINE__);
    }
}

int main(int argc, char** argv)
{
    const std::string keys = "{help h usage ?||print this message}"
                                 "{@config_file_path |<none>| path to omni-dir config file}"
                                 "{@input_image_path |<none>| path to input image}"
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

    std::string configFilePath = parser.get<std::string>(0);
    std::string imagePath = parser.get<std::string>(1);
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

    cv::Size imageSize;
    cv::Mat K, D, xi;
    loadCameraParameters(configFilePath, imageSize, K, D, xi);

    std::cout << "K: " << std::endl << K << std::endl;
    std::cout << "Xi: " << xi << std::endl;
    std::cout << "D: " << D << std::endl;

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

    std::cout << "K_new perspective: " << std::endl << K_new_p << std::endl;
    std::cout << "K_new longilati: " << std::endl << K_new_l << std::endl;

    cv::Mat img = cv::imread(imagePath);
    cv::Mat undistorted_p, undistorted_l;

    cv::omnidir::undistortImage(img, undistorted_p, K, D, xi, cv::omnidir::RECTIFY_PERSPECTIVE, K_new_p, imageSize, R);
    cv::omnidir::undistortImage(img, undistorted_l, K, D, xi, cv::omnidir::RECTIFY_LONGLATI, K_new_l, imageSize, R);

    cv::imshow("distorted", img);
    cv::imshow("undistorted perspective", undistorted_p);
    cv::imshow("undistorted longilati", undistorted_l);
    cv::waitKey();

    return 0;
}
