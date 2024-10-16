// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <ctime>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/3d.hpp>
#include <opencv2/calib.hpp>
#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "aruco_samples_utility.hpp"

using namespace std;
using namespace cv;


namespace {
const char* about =
        "Calibration using a ArUco Planar Grid board\n"
        "  To capture a frame for calibration, press 'c',\n"
        "  If input comes from video, press any key for next frame\n"
        "  To finish capturing, press 'ESC' key and calibration starts.\n";
const char* keys  =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{l        |       | Marker side length (in meters) }"
        "{s        |       | Separation between two consecutive markers in the grid (in meters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{@outfile |<none> | Output file with calibrated camera parameters }"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       | false | Apply refind strategy }"
        "{zt       | false | Assume zero tangential distortion }"
        "{a        |       | Fix aspect ratio (fx/fy) to this value }"
        "{pc       | false | Fix the principal point at the center }";
}


int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 6) {
        parser.printMessage();
        return 0;
    }

    int markersX = parser.get<int>("w");
    int markersY = parser.get<int>("h");
    float markerLength = parser.get<float>("l");
    float markerSeparation = parser.get<float>("s");
    string outputFile = parser.get<String>(0);

    int calibrationFlags = 0;
    float aspectRatio = 1;
    if(parser.has("a")) {
        calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if(parser.get<bool>("zt")) calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
    if(parser.get<bool>("pc")) calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;

    aruco::DetectorParameters detectorParams;
    if(parser.has("dp")) {
        FileStorage fs(parser.get<string>("dp"), FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    bool refindStrategy = parser.get<bool>("rs");
    int camId = parser.get<int>("ci");
    String video;

    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    VideoCapture inputVideo;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(0);
    if (parser.has("d")) {
        int dictionaryId = parser.get<int>("d");
        dictionary = aruco::getPredefinedDictionary(
            aruco::PredefinedDictionaryType(dictionaryId)
        );
    }
    else if (parser.has("cd")) {
        FileStorage fs(parser.get<std::string>("cd"), FileStorage::READ);
        bool readOk = dictionary.aruco::Dictionary::readDictionary(fs.root());
        if(!readOk) {
            cerr << "Invalid dictionary file" << endl;
            return 0;
        }
    }
    else {
        cerr << "Dictionary not specified" << endl;
        return 0;
    }

    // Create board object
    aruco::GridBoard gridboard(Size(markersX, markersY), markerLength, markerSeparation, dictionary);

    // Collected frames for calibration
    vector<vector<vector<Point2f>>> allMarkerCorners;
    vector<vector<int>> allMarkerIds;
    Size imageSize;

    aruco::ArucoDetector detector(dictionary, detectorParams);

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedMarkers;

        // Detect markers
        detector.detectMarkers(image, markerCorners, markerIds, rejectedMarkers);

        // Refind strategy to detect more markers
        if(refindStrategy) {
            detector.refineDetectedMarkers(
                image, gridboard, markerCorners, markerIds, rejectedMarkers
            );
        }

        // Draw results
        image.copyTo(imageCopy);

        if(!markerIds.empty()) {
            aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
        }

        putText(
            imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
            Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2
        );
        imshow("out", imageCopy);

        // Wait for key pressed
        char key = (char)waitKey(waitTime);

        if(key == 27) {
             break;
        }

        if(key == 'c' && !markerIds.empty()) {
            cout << "Frame captured" << endl;
            allMarkerCorners.push_back(markerCorners);
            allMarkerIds.push_back(markerIds);
            imageSize = image.size();
        }
    }

    if(allMarkerIds.empty()) {
        cerr << "Not enough captures for calibration" << endl;
        return 0;
    }

    Mat cameraMatrix, distCoeffs;

    if(calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = aspectRatio;
    }

    // Prepare data for calibration
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoints;
    vector<Mat> processedObjectPoints, processedImagePoints;
    size_t nFrames = allMarkerCorners.size();

    for(size_t frame = 0; frame < nFrames; frame++) {
        Mat currentImgPoints, currentObjPoints;

        gridboard.matchImagePoints(
            allMarkerCorners[frame], allMarkerIds[frame],
            currentObjPoints, currentImgPoints
        );

        if(currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }

    // Calibrate camera
    double repError = calibrateCamera(
        processedObjectPoints, processedImagePoints, imageSize,
        cameraMatrix, distCoeffs, noArray(), noArray(), noArray(),
        noArray(), noArray(), calibrationFlags
    );

    bool saveOk = saveCameraParams(
        outputFile, imageSize, aspectRatio, calibrationFlags, cameraMatrix,
        distCoeffs, repError
    );

    if(!saveOk) {
        cerr << "Cannot save output file" << endl;
        return 0;
    }

    cout << "Rep Error: " << repError << endl;
    cout << "Calibration saved to " << outputFile << endl;

    return 0;
}
