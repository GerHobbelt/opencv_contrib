// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR

namespace opencv_test {
namespace {

    class OmnidirTest : public cvtest::BaseTest {
    public:
        inline OmnidirTest()
            /* :
               datasetRepositoryPath("/home/stefan/Documents/DifferentProjects/OpenCVPullRequests/opencv_extra/testdata/cv/cameracalibration/fisheye")
             */
            : datasetRepositoryPath(combine(ts->get_data_path().c_str(), "/cameracalibration/fisheye"))
            , imageSize(1280, 800)
            // Estimated using default compile flags in DEBUG mode
            , K(9.3427053703903516e+02, 0., 6.1854878705698127e+02, 0., 9.3683593862086059e+02, 3.7989519649147530e+02,
                  0., 0., 1.)
            , D(-3.3409746491669956e-01, 1.2265996091178646e-01, 1.4917459281375525e-03, 9.2091740179076498e-04)
            , Xi(7.0243186914355871e-01)
        {
        }
        inline ~OmnidirTest() { }

    protected:
        const std::string datasetRepositoryPath;
        const cv::Size imageSize;
        const cv::Matx33d K;
        const cv::Matx14d D;
        const cv::Vec<double, 1> Xi;

    protected:
        inline std::string combine(const std::string& _item1, const std::string& _item2)
        {
            std::string item1 = _item1, item2 = _item2;
            std::replace(item1.begin(), item1.end(), '\\', '/');
            std::replace(item2.begin(), item2.end(), '\\', '/');

            if (item1.empty())
                return item2;

            if (item2.empty())
                return item1;

            char last = item1[item1.size() - 1];
            return item1 + (last != '/' ? "/" : "") + item2;
        }
    };

    class CV_CalibrationOmni : public OmnidirTest {
    public:
        inline CV_CalibrationOmni() { }
        inline ~CV_CalibrationOmni() { }

    protected:
        void run(int) override
        {
            ts->set_failed_test_info(cvtest::TS::OK);
            test_case_count = 1;
            const int n_images = 34;
            std::vector<std::vector<cv::Point2d>> imagePoints(n_images);
            std::vector<std::vector<cv::Point3d>> objectPoints(n_images);

            const std::string folder = combine(datasetRepositoryPath, "calib-3_stereo_from_JY");
            cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
            CV_Assert(fs_left.isOpened());
            for (int i = 0; i < n_images; ++i)
                fs_left[cv::format("image_%d", i)] >> imagePoints[i];
            fs_left.release();

            cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
            CV_Assert(fs_object.isOpened());
            for (int i = 0; i < n_images; ++i)
                fs_object[cv::format("image_%d", i)] >> objectPoints[i];
            fs_object.release();

            int flag = 0;
            flag |= cv::omnidir::CALIB_FIX_SKEW;

            cv::Matx33d estK;
            cv::Matx14d estD;
            cv::Vec<double, 1> estXi;

            cv::omnidir::calibrate(objectPoints, imagePoints, imageSize, estK, estXi, estD, cv::noArray(),
                cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));

            EXPECT_MAT_NEAR(estK, this->K, 1e-10);
            EXPECT_MAT_NEAR(estD, this->D, 1e-10);
        }
    };

    class CV_EstimateNewCameraMatrixForUndistortRectify : public OmnidirTest {
    public:
        CV_EstimateNewCameraMatrixForUndistortRectify() { }
        ~CV_EstimateNewCameraMatrixForUndistortRectify() { }

    protected:
        void run(int) override
        {
            ts->set_failed_test_info(cvtest::TS::OK);
            int progress = 0;
            int caseId = 0;

            cv::Mat R_eye = cv::Mat::eye(3, 3, cv::DataType<double>::type);
            cv::Mat R_xp;

            // scale0 = 0, scale1 = 1
            progress = update_progress(progress, 1, 27, 0);
            cv::Matx33d K_new_truth = cv::Matx33d(4.5288365257994622e+02, 0., 5.8291094241366773e+02, 0.,
                4.5412722003996015e+02, 3.7066458811112551e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.0, 1.0);
            // scale0 = -1, scale1 = -1
            progress = update_progress(progress, 2, 27, 0);
            runCasePerspective(++caseId, K_new_truth, R_eye, -1.0, -1.0);
            // scale0 = 0, scale1 = 0
            progress = update_progress(progress, 3, 27, 0);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.0, 0.0);
            // scale0 = 1, scale1 = 1
            progress = update_progress(progress, 4, 27, 0);
            K_new_truth = cv::Matx33d(2.7199168328349276e+02, 0., 6.0571360264050668e+02, 0., 2.7273854178633098e+02,
                3.8238181481266639e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 1.0, 1.0);
            // scale0 = 1, scale1 = 0
            progress = update_progress(progress, 5, 27, 0);
            runCasePerspective(++caseId, K_new_truth, R_eye, 1.0, 0.0);
            // scale0 = 0.5, scale1 = 0.5
            progress = update_progress(progress, 6, 27, 0);
            K_new_truth = cv::Matx33d(7.2487533586343898e+02, 0., 5.4862454505417429e+02, 0., 7.2686576182629119e+02,
                3.5304640292379185e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.5, 0.5);
            // scale0 = 0.0, scale1 = 0.5
            progress = update_progress(progress, 7, 27, 0);
            K_new_truth = cv::Matx33d(9.0576730515989243e+02, 0., 5.2582188482733534e+02, 0., 9.0825444007992030e+02,
                3.4132917622225096e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.0, 0.5);
            // scale0 = 1, scale1 = 0.5
            progress = update_progress(progress, 8, 27, 0);
            K_new_truth = cv::Matx33d(5.4398336656698552e+02, 0., 5.7142720528101324e+02, 0., 5.4547708357266197e+02,
                3.6476362962533273e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 1.0, 0.5);
            // scale0 = 0.5, scale1 = 1.0
            progress = update_progress(progress, 9, 27, 0);
            K_new_truth = cv::Matx33d(3.6243766793171949e+02, 0., 5.9431227252708709e+02, 0., 3.6343288091314560e+02,
                3.7652320146189595e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.5, 1.0);
            // scale0 = 0.0, scale1 = 2
            progress = update_progress(progress, 10, 27, 0);
            K_new_truth = cv::Matx33d(2.2644182628997311e+02, 0., 6.1145547120683386e+02, 0., 2.2706361001998008e+02,
                3.8533229405556278e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.0, 2.0);
            // scale0 = 1, scale1 = 2
            progress = update_progress(progress, 11, 27, 0);
            K_new_truth = cv::Matx33d(1.3599584164174638e+02, 0., 6.2285680132025334e+02, 0., 1.3636927089316549e+02,
                3.9119090740633317e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 1.0, 2.0);
            // scale0 = 2, scale1 = 2
            progress = update_progress(progress, 12, 27, 0);
            runCasePerspective(++caseId, K_new_truth, R_eye, 2.0, 2.0);
            // scale0 = 0.5, scale1 = 2
            progress = update_progress(progress, 13, 27, 0);
            K_new_truth = cv::Matx33d(1.8121883396585974e+02, 0., 6.1715613626354354e+02, 0., 1.8171644045657280e+02,
                3.8826160073094798e+02, 0., 0., 1.);
            runCasePerspective(++caseId, K_new_truth, R_eye, 0.5, 2.0);

            // scale0 = 0, scale1 = 1
            progress = update_progress(progress, 14, 27, 0);
            K_new_truth = cv::Matx33d(5.5459064803950560e+02, 0., -2.4965993983377021e+02, 0., 5.5347894441196308e+02,
                -4.8857633821260163e+02, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 0.0, 1.0);
            // scale0 = -1, scale1 = -1
            progress = update_progress(progress, 15, 27, 0);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, -1.0, -1.0);
            // scale0 = 0, scale1 = 0
            progress = update_progress(progress, 16, 27, 0);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 0.0, 0.0);
            // scale0 = 1, scale1 = 1
            progress = update_progress(progress, 17, 27, 0);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 1.0, 1.0);
            // scale0 = 1, scale1 = 0
            progress = update_progress(progress, 18, 27, 0);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 1.0, 0.0);
            // scale0 = 0, scale1 = 1
            progress = update_progress(progress, 18, 27, 0);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 0.0, 1.0);
            // scale0 = 0.5, scale1 = 0.5
            progress = update_progress(progress, 19, 27, 0);
            K_new_truth = cv::Matx33d(1.1091812960790112e+03, 0., -1.1208088926490268e+03, 0., 1.1069578888239262e+03,
                -1.3579790310532298e+03, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 0.5, 0.5);
            // scale0 = 1.0, scale1 = 0.5
            progress = update_progress(progress, 20, 27, 0);
            K_new_truth = cv::Matx33d(5.5459064803950560e+02, 0., -2.4965993983377021e+02, 0., 1.1069578888239262e+03,
                -1.3579790310532298e+03, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 1.0, 0.5);
            // scale0 = 0.5, scale1 = 1.0
            progress = update_progress(progress, 21, 27, 0);
            K_new_truth = cv::Matx33d(1.1091812960790112e+03, 0., -1.1208088926490268e+03, 0., 5.5347894441196308e+02,
                -4.8857633821260163e+02, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 0.5, 1.0);
            // scale0 = 1.0, scale1 = 2.0
            progress = update_progress(progress, 22, 27, 0);
            K_new_truth = cv::Matx33d(5.5459064803950560e+02, 0., -2.4965993983377021e+02, 0., 2.7673947220598154e+02,
                -5.3874991792287453e+01, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 1.0, 2.0);
            // scale0 = 2.0, scale1 = 1.0
            progress = update_progress(progress, 23, 27, 0);
            K_new_truth = cv::Matx33d(2.7729532401975280e+02, 0., 1.8591453657385813e+02, 0., 5.5347894441196308e+02,
                -4.8857633821260163e+02, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 2.0, 1.0);
            // scale0 = 2.0, scale1 = 2.0
            progress = update_progress(progress, 24, 27, 0);
            K_new_truth = cv::Matx33d(2.7729532401975280e+02, 0., 1.8591453657385813e+02, 0., 2.7673947220598154e+02,
                -5.3874991792287453e+01, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 2.0, 2.0);
            // scale0 = 0.5, scale1 = 2.0
            progress = update_progress(progress, 25, 27, 0);
            K_new_truth = cv::Matx33d(1.1091812960790112e+03, 0., -1.1208088926490268e+03, 0., 2.7673947220598154e+02,
                -5.3874991792287453e+01, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 0.5, 2.0);
            // scale0 = 0.5, scale1 = 2.0
            progress = update_progress(progress, 26, 27, 0);
            K_new_truth = cv::Matx33d(2.7729532401975280e+02, 0., 1.8591453657385813e+02, 0., 1.1069578888239262e+03,
                -1.3579790310532298e+03, 0., 0., 1.);
            runCaseLongiLati(++caseId, K_new_truth, R_eye, 2.0, 0.5);
            progress = update_progress(progress, 27, 27, 0);
        }

        // test accuracy of calculating new camera matrix for undistort rectify width different rotations, and scaling
        // parameters for perspective rectification
        void runCasePerspective(int caseId, cv::Matx33d K_new_truth, cv::Mat R, double scale0, double scale1)
        {
            cv::Mat K_new(3, 3, cv::DataType<double>::type);

            cv::omnidir::estimateNewCameraMatrixForUndistortRectify(this->K, this->D, this->Xi, this->imageSize, R,
                K_new, cv::omnidir::RECTIFY_PERSPECTIVE, scale0, scale1);
            auto result = cvtest::assertMatNear("K_new", "K_new_truth", "1e-6", K_new, K_new_truth, 1e-6);
            if (!result) {
                ts->printf(cvtest::TS::LOG, "case %d: rectification: perspective, scale0=%lf, scale1=%lf: %s\n", caseId,
                    scale0, scale1, result.message());
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }
        }

        // test accuracy of calculating new camera matrix for undistort rectify width different rotations, and scaling
        // parameters for longitute latitude rectification
        void runCaseLongiLati(int caseId, cv::Matx33d K_new_truth, cv::Mat R, double scale0, double scale1)
        {
            cv::Mat K_new(3, 3, cv::DataType<double>::type);

            cv::omnidir::estimateNewCameraMatrixForUndistortRectify(
                this->K, this->D, this->Xi, this->imageSize, R, K_new, cv::omnidir::RECTIFY_LONGLATI, scale0, scale1);
            auto result = cvtest::assertMatNear("K_new", "K_new_truth", "1e-6", K_new, K_new_truth, 1e-6);

            if (!result) {
                ts->printf(cvtest::TS::LOG, "case %d: rectification: longilati, scale0=%lf, scale1=%lf: %s\n", caseId,
                    scale0, scale1, result.message());
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }
        }
    };

    template <class T> double thres() { return 1.0; }
    template <> double thres<float>() { return 1e-5; }

    class CV_ReprojectImageTo3DTest : public cvtest::BaseTest {
    public:
        CV_ReprojectImageTo3DTest() { }
        ~CV_ReprojectImageTo3DTest() { }

    protected:
        void run(int)
        {
            ts->set_failed_test_info(cvtest::TS::OK);
            int progress = 0;
            int caseId = 0;

            // Stereo rectification with perspective projection
            progress = update_progress(progress, 1, 25, 0);
            runCasePerspective<float, float>(++caseId, -100.f, 100.f);
            progress = update_progress(progress, 2, 25, 0);
            runCasePerspective<int, float>(++caseId, -100, 100);
            progress = update_progress(progress, 3, 25, 0);
            runCasePerspective<short, float>(++caseId, -100, 100);
            progress = update_progress(progress, 4, 25, 0);
            runCasePerspective<unsigned char, float>(++caseId, 10, 100);

            progress = update_progress(progress, 5, 25, 0);
            runCasePerspective<float, int>(++caseId, -100.f, 100.f);
            progress = update_progress(progress, 6, 25, 0);
            runCasePerspective<int, int>(++caseId, -100, 100);
            progress = update_progress(progress, 7, 25, 0);
            runCasePerspective<short, int>(++caseId, -100, 100);
            progress = update_progress(progress, 8, 25, 0);
            runCasePerspective<unsigned char, int>(++caseId, 10, 100);

            progress = update_progress(progress, 9, 25, 0);
            runCasePerspective<float, short>(++caseId, -100.f, 100.f);
            progress = update_progress(progress, 10, 25, 0);
            runCasePerspective<int, short>(++caseId, -100, 100);
            progress = update_progress(progress, 11, 25, 0);
            runCasePerspective<short, short>(++caseId, -100, 100);
            progress = update_progress(progress, 12, 25, 0);
            runCasePerspective<unsigned char, short>(++caseId, 10, 100);

            // Stereo rectification with longitude-latitude projection
            progress = update_progress(progress, 13, 25, 0);
            runCaseLongiLati<float, float>(++caseId, -100.f, 100.f);
            progress = update_progress(progress, 14, 25, 0);
            runCaseLongiLati<int, float>(++caseId, -100, 100);
            progress = update_progress(progress, 15, 25, 0);
            runCaseLongiLati<short, float>(++caseId, -100, 100);
            progress = update_progress(progress, 16, 25, 0);
            runCaseLongiLati<unsigned char, float>(++caseId, 10, 100);

            progress = update_progress(progress, 17, 25, 0);
            runCaseLongiLati<float, int>(++caseId, -100.f, 100.f);
            progress = update_progress(progress, 18, 25, 0);
            runCaseLongiLati<int, int>(++caseId, -100, 100);
            progress = update_progress(progress, 19, 25, 0);
            runCaseLongiLati<short, int>(++caseId, -100, 100);
            progress = update_progress(progress, 20, 25, 0);
            runCaseLongiLati<unsigned char, int>(++caseId, 10, 100);

            progress = update_progress(progress, 21, 25, 0);
            runCaseLongiLati<float, short>(++caseId, -100.f, 100.f);
            progress = update_progress(progress, 22, 25, 0);
            runCaseLongiLati<int, short>(++caseId, -100, 100);
            progress = update_progress(progress, 23, 25, 0);
            runCaseLongiLati<short, short>(++caseId, -100, 100);
            progress = update_progress(progress, 24, 25, 0);
            runCaseLongiLati<unsigned char, short>(++caseId, 10, 100);

            progress = update_progress(progress, 25, 25, 0);
        }

        template <class U, class V> double error(const Vec<U, 3>& v1, const Vec<V, 3>& v2) const
        {
            double tmp, sum = 0;
            double nsum = 0;
            for (int i = 0; i < 3; ++i) {
                tmp = v1[i];
                nsum += tmp * tmp;

                tmp = tmp - v2[i];
                sum += tmp * tmp;
            }
            return sqrt(sum) / (sqrt(nsum) + 1.);
        }

        // test accuracy and different input modalities (Q or P and T) of projecting disparity to 3D in case of
        // perspective rectification
        template <class InT, class OutT> void runCasePerspective(int caseId, InT min, InT max)
        {
            typedef Vec<OutT, 3> out3d_t;

            bool handleMissingValues = (unsigned)theRNG() % 2 == 0;

            Mat_<InT> disp(Size(320, 240));
            randu(disp, Scalar(min), Scalar(max));

            if (handleMissingValues)
                disp(disp.rows / 2, disp.cols / 2) = min - 1;

            Mat_<double> Q
                = Mat_<double>({ 1, 0, 0, theRNG().uniform(-5.0, 5.0), 0, 1, 0, theRNG().uniform(-5.0, 5.0), 0, 0, 0,
                                   theRNG().uniform(-5.0, 5.0), 0, 0, theRNG().uniform(0.0, 5.0) + 0.00001, 0 })
                      .reshape(1, 4);

            Mat_<out3d_t> _3dImgQ(disp.size());
            omnidir::reprojectImageTo3D(
                disp, _3dImgQ, noArray(), noArray(), Q, omnidir::RECTIFY_PERSPECTIVE, handleMissingValues);

            Mat_<double> P = cv::Mat_<double>({
                                                  Q(2, 3),
                                                  0.0,
                                                  -Q(0, 3),
                                                  0.0,
                                                  1.0,
                                                  -Q(1, 3),
                                                  0.0,
                                                  0.0,
                                                  1.0,
                                              })
                                 .reshape(1, 3);

            Mat_<double> T = cv::Mat_<double>({ 1.0 / Q(3, 2), 0.0, 0.0 });

            Mat_<out3d_t> _3dImgPT(disp.size());
            omnidir::reprojectImageTo3D(
                disp, _3dImgPT, P, T, noArray(), omnidir::RECTIFY_PERSPECTIVE, handleMissingValues);

            for (int y = 0; y < disp.rows; ++y) {
                for (int x = 0; x < disp.cols; ++x) {
                    InT d = disp(y, x);

                    double from[4] = {
                        static_cast<double>(x),
                        static_cast<double>(y),
                        static_cast<double>(d),
                        1.0,
                    };
                    Mat_<double> res = Q * Mat_<double>(4, 1, from);
                    res /= res(3, 0);

                    out3d_t pixel_exp = *res.ptr<Vec3d>();
                    out3d_t pixel_out = _3dImgQ(y, x);
                    out3d_t pixel_out_PT = _3dImgPT(y, x);

                    const int largeZValue = 10000; /* see documentation */

                    if (handleMissingValues && y == disp.rows / 2 && x == disp.cols / 2) {
                        if (pixel_out[2] == largeZValue)
                            continue;

                        ts->printf(cvtest::TS::LOG, "Missing values are handled improperly\n");
                        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                        return;
                    } else {
                        double err = error(pixel_out, pixel_exp), t = thres<OutT>();
                        if (err > t) {
                            ts->printf(cvtest::TS::LOG,
                                "case %d. too big error at (%d, %d): %g vs expected %g: res = (%g, %g, %g, w=%g) vs "
                                "pixel_out = (%g, %g, %g)\n",
                                caseId, x, y, err, t, res(0, 0), res(1, 0), res(2, 0), res(3, 0), (double)pixel_out[0],
                                (double)pixel_out[1], (double)pixel_out[2]);
                            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                            return;
                        }
                    }

                    double err = error(pixel_out, pixel_out_PT), t = thres<OutT>();
                    if (err > t) {
                        ts->printf(cvtest::TS::LOG,
                            "case %d. too big error at (%d, %d): %g vs expected %g: pixel_out from Q = (%g, %g, %g) vs "
                            "pixel_out from P and T = (%g, %g, %g)\n",
                            caseId, x, y, err, t, (double)pixel_out[0], (double)pixel_out[1], (double)pixel_out[2],
                            (double)pixel_out_PT[0], (double)pixel_out_PT[1], (double)pixel_out_PT[2]);
                        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                        return;
                    }
                }
            }
        }

        // test accuracy of projecting disparity to 3D in case of longitude latitude rectification
        template <class InT, class OutT> void runCaseLongiLati(int caseId, InT min, InT max)
        {
            // tested field of view from pi/8 to 2*pi
            double maxFov = CV_2PI;
            double minFov = CV_PI * 0.125;
            Size imgSize(320, 240);

            typedef Vec<OutT, 3> out3d_t;

            bool handleMissingValues = (unsigned)theRNG() % 2 == 0;

            Mat_<InT> disp(imgSize);
            randu(disp, Scalar(min), Scalar(max));

            if (handleMissingValues)
                disp(disp.rows / 2, disp.cols / 2) = min - 1;

            // Setup random projection matrix
            Vec2d fovAng = { theRNG().uniform(minFov, maxFov), theRNG().uniform(minFov, maxFov) };
            Vec2d cAng = { theRNG().uniform(minFov, fovAng[0]), theRNG().uniform(minFov, fovAng[1]) };
            Vec2d pixPerRad
                = { static_cast<double>(imgSize.width) / fovAng[0], static_cast<double>(imgSize.height) / fovAng[1] };
            Vec2d cOffset = (Vec2d::all(CV_PI * 0.5) - cAng).mul(pixPerRad);
            Mat_<double> P = cv::Mat_<double>({
                                                  pixPerRad[0],
                                                  0.0,
                                                  cOffset[0],
                                                  0.0,
                                                  pixPerRad[1],
                                                  cOffset[1],
                                                  0.0,
                                                  0.0,
                                                  1.0,
                                              })
                                 .reshape(1, 3);
            Mat_<double> T = cv::Mat_<double>({ theRNG().uniform(-5.0, 5.0) + 0.00001, 0.0, 0.0 });

            Mat_<out3d_t> _3DImg(disp.size());
            omnidir::reprojectImageTo3D(disp, _3DImg, P, T, noArray(), omnidir::RECTIFY_LONGLATI, handleMissingValues);

            Mat_<double> Pinv = P.inv();

            double baseline = cv::norm(T);
            double f = P(0, 0);

            for (int y = 0; y < disp.rows; ++y) {
                for (int x = 0; x < disp.cols; ++x) {
                    InT d = disp(y, x);
                    double depth = baseline * f / d;
                    double xPixel
                        = Pinv(0, 0) * static_cast<double>(x) + Pinv(0, 1) * static_cast<double>(y) + Pinv(0, 2);
                    double yPixel
                        = Pinv(1, 0) * static_cast<double>(x) + Pinv(1, 1) * static_cast<double>(y) + Pinv(1, 2);
                    Mat_<double> res = cv::Mat_<double>({ -std::cos(xPixel), -std::sin(xPixel) * std::cos(yPixel),
                                           std::sin(xPixel) * std::sin(yPixel) })
                        * depth;

                    out3d_t pixel_exp = *res.ptr<Vec3d>();
                    out3d_t pixel_out = _3DImg(y, x);

                    const int largeZValue = 10000; /* see documentation */

                    if (handleMissingValues && y == disp.rows / 2 && x == disp.cols / 2) {
                        if (pixel_out[2] == largeZValue)
                            continue;

                        ts->printf(cvtest::TS::LOG, "case %d. Missing values are handled improperly\n", caseId);
                        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                        return;
                    } else {
                        double err = error(pixel_out, pixel_exp), t = thres<OutT>();
                        if (err > t) {
                            ts->printf(cvtest::TS::LOG,
                                "case %d. too big error at (%d, %d): %g vs expected %g: res = (%g, %g, %g, w=%g) vs "
                                "pixel_out = (%g, %g, %g)\n",
                                caseId, x, y, err, t, res(0, 0), res(1, 0), res(2, 0), res(3, 0), (double)pixel_out[0],
                                (double)pixel_out[1], (double)pixel_out[2]);
                            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                            return;
                        }
                    }
                }
            }
        }
    };

    TEST(CV_CalibrationOmni, omnidir)
    {
        CV_CalibrationOmni test;
        test.safe_run();
    }
    TEST(CV_EstimateNewCameraMatrixForUndistortRectify, omnidir)
    {
        CV_EstimateNewCameraMatrixForUndistortRectify test;
        test.safe_run();
    }
    TEST(Calib3d_ReprojectImageTo3D, accuracy)
    {
        CV_ReprojectImageTo3DTest test;
        test.safe_run();
    }

}
} // namespace
