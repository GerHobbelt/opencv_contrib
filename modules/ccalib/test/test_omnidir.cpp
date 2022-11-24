// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/* class OmnidirTest : public cvtest::BaseTest */
/* { */
/* public: */
/*     OmnidirTest(); */
/*     ~OmnidirTest(); */
/*     void */
/* }; */

template<class T> double thres() { return 1.0; }
template<> double thres<float>() { return 1e-5; }

class CV_ReprojectImageTo3DTest : public cvtest::BaseTest
{
public:
    CV_ReprojectImageTo3DTest() {}
    ~CV_ReprojectImageTo3DTest() {}
protected:


    void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);
        int progress = 0;
        int caseId = 0;

        // Stereo rectification with perspective projection
        progress = update_progress( progress, 1, 14, 0 );
        runCasePerspective<float, float>(++caseId, -100.f, 100.f);
        progress = update_progress( progress, 2, 14, 0 );
        runCasePerspective<int, float>(++caseId, -100, 100);
        progress = update_progress( progress, 3, 14, 0 );
        runCasePerspective<short, float>(++caseId, -100, 100);
        progress = update_progress( progress, 4, 14, 0 );
        runCasePerspective<unsigned char, float>(++caseId, 10, 100);
        progress = update_progress( progress, 5, 14, 0 );

        runCasePerspective<float, int>(++caseId, -100.f, 100.f);
        progress = update_progress( progress, 6, 14, 0 );
        runCasePerspective<int, int>(++caseId, -100, 100);
        progress = update_progress( progress, 7, 14, 0 );
        runCasePerspective<short, int>(++caseId, -100, 100);
        progress = update_progress( progress, 8, 14, 0 );
        runCasePerspective<unsigned char, int>(++caseId, 10, 100);
        progress = update_progress( progress, 10, 14, 0 );

        runCasePerspective<float, short>(++caseId, -100.f, 100.f);
        progress = update_progress( progress, 11, 14, 0 );
        runCasePerspective<int, short>(++caseId, -100, 100);
        progress = update_progress( progress, 12, 14, 0 );
        runCasePerspective<short, short>(++caseId, -100, 100);
        progress = update_progress( progress, 13, 14, 0 );
        runCasePerspective<unsigned char, short>(++caseId, 10, 100);
        progress = update_progress( progress, 14, 14, 0 );

        // Stereo rectification with longitude-latitude projection
        runCaseLongiLati<float, float>(++caseId, -100.f, 100.f);
        progress = update_progress( progress, 2, 14, 0 );
        runCaseLongiLati<int, float>(++caseId, -100, 100);
        progress = update_progress( progress, 3, 14, 0 );
        runCaseLongiLati<short, float>(++caseId, -100, 100);
        progress = update_progress( progress, 4, 14, 0 );
        runCaseLongiLati<unsigned char, float>(++caseId, 10, 100);
        progress = update_progress( progress, 5, 14, 0 );

        runCaseLongiLati<float, int>(++caseId, -100.f, 100.f);
        progress = update_progress( progress, 6, 14, 0 );
        runCaseLongiLati<int, int>(++caseId, -100, 100);
        progress = update_progress( progress, 7, 14, 0 );
        runCaseLongiLati<short, int>(++caseId, -100, 100);
        progress = update_progress( progress, 8, 14, 0 );
        runCaseLongiLati<unsigned char, int>(++caseId, 10, 100);
        progress = update_progress( progress, 10, 14, 0 );

        runCaseLongiLati<float, short>(++caseId, -100.f, 100.f);
        progress = update_progress( progress, 11, 14, 0 );
        runCaseLongiLati<int, short>(++caseId, -100, 100);
        progress = update_progress( progress, 12, 14, 0 );
        runCaseLongiLati<short, short>(++caseId, -100, 100);
        progress = update_progress( progress, 13, 14, 0 );
        runCaseLongiLati<unsigned char, short>(++caseId, 10, 100);
        progress = update_progress( progress, 14, 14, 0 );
    }

    template<class U, class V> double error(const Vec<U, 3>& v1, const Vec<V, 3>& v2) const
    {
        double tmp, sum = 0;
        double nsum = 0;
        for(int i = 0; i < 3; ++i)
        {
            tmp = v1[i];
            nsum +=  tmp * tmp;

            tmp = tmp - v2[i];
            sum += tmp * tmp;

        }
        return sqrt(sum)/(sqrt(nsum)+1.);
    }

    // test accuracy and different input modalities (Q or P and T) of projecting disparity to 3D in case of perspective rectification
    template<class InT, class OutT> void runCasePerspective(int caseId, InT min, InT max)
    {
        typedef Vec<OutT, 3> out3d_t;

        bool handleMissingValues = (unsigned)theRNG() % 2 == 0;

        Mat_<InT> disp(Size(320, 240));
        randu(disp, Scalar(min), Scalar(max));

        if (handleMissingValues)
            disp(disp.rows/2, disp.cols/2) = min - 1;

        Mat_<double> Q = Mat_<double>({1, 0, 0, theRNG().uniform(-5.0, 5.0),
                          0, 1, 0, theRNG().uniform(-5.0, 5.0),
                          0, 0, 0, theRNG().uniform(-5.0, 5.0),
                          0, 0, theRNG().uniform(0.0, 5.0)+0.00001, 0}).reshape(1, 4);

        Mat_<out3d_t> _3dImgQ(disp.size());
        omnidir::reprojectImageTo3D(disp, _3dImgQ, noArray(), noArray(), Q, omnidir::RECTIFY_PERSPECTIVE, handleMissingValues);

        Mat_<double> P = cv::Mat_<double>({Q(2, 3), 0.0, -Q(0, 3),
                                               0.0, 1.0, -Q(1, 3),
                                               0.0, 0.0,      1.0,
                                            }).reshape(1, 3);

        Mat_<double> T = cv::Mat_<double>({1.0/Q(3, 2), 0.0, 0.0});

        Mat_<out3d_t> _3dImgPT(disp.size());
        omnidir::reprojectImageTo3D(disp, _3dImgPT, P, T, noArray(), omnidir::RECTIFY_PERSPECTIVE, handleMissingValues);

        for(int y = 0; y < disp.rows; ++y)
        {
            for(int x = 0; x < disp.cols; ++x)
            {
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

                if (handleMissingValues && y == disp.rows/2 && x == disp.cols/2)
                {
                    if (pixel_out[2] == largeZValue)
                        continue;

                    ts->printf(cvtest::TS::LOG, "Missing values are handled improperly\n");
                    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                    return;
                }
                else
                {
                    double err = error(pixel_out, pixel_exp), t = thres<OutT>();
                    if ( err > t )
                    {
                        ts->printf(cvtest::TS::LOG, "case %d. too big error at (%d, %d): %g vs expected %g: res = (%g, %g, %g, w=%g) vs pixel_out = (%g, %g, %g)\n",
                                caseId, x, y, err, t, res(0,0), res(1,0), res(2,0), res(3,0),
                                (double)pixel_out[0], (double)pixel_out[1], (double)pixel_out[2]);
                        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                        return;
                    }
                }

                double err = error(pixel_out, pixel_out_PT), t = thres<OutT>();
                if ( err > t )
                {
                    ts->printf(cvtest::TS::LOG, "case %d. too big error at (%d, %d): %g vs expected %g: pixel_out from Q = (%g, %g, %g) vs pixel_out from P and T = (%g, %g, %g)\n",
                            caseId, x, y, err, t, (double)pixel_out[0], (double)pixel_out[1], (double)pixel_out[2],
                            (double)pixel_out_PT[0], (double)pixel_out_PT[1], (double)pixel_out_PT[2]);
                    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                    return;
                }
            }
        }
    }

    // test accuracy of projecting disparity to 3D in case of longitude latitude rectification
    template<class InT, class OutT> void runCaseLongiLati(int caseId, InT min, InT max)
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
            disp(disp.rows/2, disp.cols/2) = min - 1;

        // Setup random projection matrix
        Vec2d fovAng = {theRNG().uniform(minFov, maxFov), theRNG().uniform(minFov, maxFov)};
        Vec2d cAng = {theRNG().uniform(minFov, fovAng[0]), theRNG().uniform(minFov, fovAng[1])};
        Vec2d pixPerRad = {static_cast<double>(imgSize.width)/fovAng[0], static_cast<double>(imgSize.height)/fovAng[1]};
        Vec2d cOffset = (Vec2d::all(CV_PI * 0.5) - cAng).mul(pixPerRad);
        Mat_<double> P = cv::Mat_<double>({pixPerRad[0], 0.0, cOffset[0],
                                               0.0, pixPerRad[1], cOffset[1],
                                               0.0, 0.0,      1.0,
                                            }).reshape(1, 3);
        Mat_<double> T = cv::Mat_<double>({theRNG().uniform(-5.0, 5.0) + 0.00001, 0.0, 0.0});

        Mat_<out3d_t> _3DImg(disp.size());
        omnidir::reprojectImageTo3D(disp, _3DImg, P, T, noArray(), omnidir::RECTIFY_LONGLATI, handleMissingValues);

        Mat_<double> Pinv = P.inv();

        double baseline = cv::norm(T);
        double f = P(0, 0);

        for(int y = 0; y < disp.rows; ++y)
        {
            for(int x = 0; x < disp.cols; ++x)
            {
                InT d = disp(y, x);
                double depth = baseline * f / d;
                double xPixel = Pinv(0, 0) * static_cast<double>(x) + Pinv(0, 1) * static_cast<double>(y) + Pinv(0, 2);
                double yPixel = Pinv(1, 0) * static_cast<double>(x) + Pinv(1, 1) * static_cast<double>(y) + Pinv(1, 2);
                Mat_<double> res = cv::Mat_<double>({-std::cos(xPixel), -std::sin(xPixel) * std::cos(yPixel), std::sin(xPixel) * std::sin(yPixel)}) * depth;

                out3d_t pixel_exp = *res.ptr<Vec3d>();
                out3d_t pixel_out = _3DImg(y, x);

                const int largeZValue = 10000; /* see documentation */

                if (handleMissingValues && y == disp.rows/2 && x == disp.cols/2)
                {
                    if (pixel_out[2] == largeZValue)
                        continue;

                    ts->printf(cvtest::TS::LOG, "case %d. Missing values are handled improperly\n", caseId);
                    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                    return;
                }
                else
                {
                    double err = error(pixel_out, pixel_exp), t = thres<OutT>();
                    if ( err > t )
                    {
                        ts->printf(cvtest::TS::LOG, "case %d. too big error at (%d, %d): %g vs expected %g: res = (%g, %g, %g, w=%g) vs pixel_out = (%g, %g, %g)\n",
                                caseId, x, y, err, t, res(0,0), res(1,0), res(2,0), res(3,0),
                                (double)pixel_out[0], (double)pixel_out[1], (double)pixel_out[2]);
                        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
                        return;
                    }
                }
            }
        }
    }
};

TEST(Calib3d_ReprojectImageTo3D, accuracy) { CV_ReprojectImageTo3DTest test; test.safe_run(); }


}} // namespace
