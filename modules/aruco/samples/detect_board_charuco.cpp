#include <opencv2/highgui.hpp>
//! [charucohdr]
#include <opencv2/objdetect/charuco_detector.hpp>
//! [charucohdr]
#include <vector>
#include <iostream>
#include "aruco_samples_utility.hpp"

using namespace std;
using namespace cv;

static aruco::DetectorParameters readDetectorParamsFromCommandLine(CommandLineParser &parser) {
    aruco::DetectorParameters detectorParams;
    if (parser.has("dp")) {
        FileStorage fs(parser.get<string>("dp"), FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            throw -1;
        }
    }
    return detectorParams;
}

static void readCameraParamsFromCommandLine(CommandLineParser &parser, Mat& camMatrix, Mat& distCoeffs) {
    //! [camDistCoeffs]
    if(parser.has("c")) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            throw -1;
        }
    }
    //! [camDistCoeffs]
}

static aruco::Dictionary readDictionatyFromCommandLine(CommandLineParser &parser) {
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    if (parser.has("d")) {
        int dictionaryId = parser.get<int>("d");
        dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));
    }
    else if (parser.has("cd")) {
        FileStorage fs(parser.get<string>("cd"), FileStorage::READ);
        bool readOk = dictionary.readDictionary(fs.root());
        if(!readOk) {
            cerr << "Invalid dictionary file" << endl;
            throw -1;
        }
    }
    else {
        cerr << "Dictionary not specified" << endl;
        throw -1;
    }
    return dictionary;
}

namespace {
const char* about = "Pose estimation using a ChArUco board";
const char* keys  =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{sl       |       | Square side length (in meters) }"
        "{ml       |       | Marker side length (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{c        |       | Output file with calibrated camera parameters }"
        "{v        |       | Input from video or image file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       |       | Apply refind strategy }";
}


int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 6) {
        parser.printMessage();
        return 0;
    }

    //! [charuco_detect_board_full_sample]
    int squaresX = parser.get<int>("w");
    int squaresY = parser.get<int>("h");
    float squareLength = parser.get<float>("sl");
    float markerLength = parser.get<float>("ml");
    bool refine = parser.has("rs");
    int camId = parser.get<int>("ci");

    string video;
    if(parser.has("v")) {
        video = parser.get<string>("v");
    }

    Mat camMatrix, distCoeffs;
    readCameraParamsFromCommandLine(parser, camMatrix, distCoeffs);

    aruco::DetectorParameters detectorParams = readDetectorParamsFromCommandLine(parser);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    aruco::Dictionary dictionary = readDictionatyFromCommandLine(parser);

    VideoCapture inputVideo;
    int waitTime = 0;
    if(!video.empty()) {
        inputVideo.open(video);
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    float axisLength = 0.5f * ((float)min(squaresX, squaresY) * (squareLength));

    // create charuco board object
    aruco::CharucoBoard charucoboard(Size(squaresX, squaresY), squareLength, markerLength, dictionary);

    // create charuco detector
    aruco::CharucoParameters charucoParams;
    charucoParams.tryRefineMarkers = refine; // if tryRefineMarkers, refineDetectedMarkers() will be used in detectBoard()
    charucoParams.cameraMatrix = camMatrix; // cameraMatrix can be used in detectBoard()
    charucoParams.distCoeffs = distCoeffs; // distCoeffs can be used in detectBoard()
    aruco::CharucoDetector charucoDetector(charucoboard, charucoParams, detectorParams);

    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        //! [inputImg]
        Mat image, imageCopy;
        inputVideo.retrieve(image);
        //! [inputImg]

        double tick = (double)getTickCount();

        vector<int> markerIds, charucoIds;
        vector<vector<Point2f> > markerCorners;
        vector<Point2f> charucoCorners;
        Vec3d rvec, tvec;

        //! [interpolateCornersCharuco]
        // detect markers and charuco corners
        charucoDetector.detectBoard(image, charucoCorners, charucoIds, markerCorners, markerIds);
        //! [interpolateCornersCharuco]

        //! [poseCharuco]
        // estimate charuco board pose
        bool validPose = false;
        if(camMatrix.total() != 0 && distCoeffs.total() != 0 && charucoIds.size() >= 4) {
            Mat objPoints, imgPoints;
            charucoboard.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);
            validPose = solvePnP(objPoints, imgPoints, camMatrix, distCoeffs, rvec, tvec);
        }
        //! [poseCharuco]

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // draw results
        image.copyTo(imageCopy);
        if(markerIds.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, markerCorners);
        }

        if(charucoIds.size() > 0) {
            //! [drawDetectedCornersCharuco]
            aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
            //! [drawDetectedCornersCharuco]
        }

        if(validPose)
            cv::drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }
    //! [charuco_detect_board_full_sample]
    return 0;
}
