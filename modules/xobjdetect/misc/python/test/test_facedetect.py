#!/usr/bin/env python

'''
face detection using haar cascades
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.275, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

from tests_common import NewOpenCVTests, intersectionRate

class facedetect_test(NewOpenCVTests):

    def test_facedetect(self):
        cascade_fn = self.extraTestDataPath + '/cv/cascadeandhog/cascades/haarcascade_frontalface_alt.xml'
        nested_fn  = self.extraTestDataPath + '/cv/cascadeandhog/cascades/haarcascade_eye.xml'

        cascade = cv.CascadeClassifier(cascade_fn)
        nested = cv.CascadeClassifier(nested_fn)

        samples = ['samples/data/lena.jpg', 'cv/cascadeandhog/images/mona-lisa.png']

        faces = []
        eyes = []

        testFaces = [
        #lena
        [[218, 200, 389, 371],
        [ 244, 240, 294, 290],
        [ 309, 246, 352, 289]],

        #lisa
        [[167, 119, 307, 259],
        [188, 153, 229, 194],
        [236, 153, 277, 194]]
        ]

        for sample in samples:

            img = self.get_sample(  sample)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (5, 5), 0)

            rects = detect(gray, cascade)
            faces.append(rects)

            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    subrects = detect(roi.copy(), nested)

                    for rect in subrects:
                        rect[0] += x1
                        rect[2] += x1
                        rect[1] += y1
                        rect[3] += y1

                    eyes.append(subrects)

        faces_matches = 0
        eyes_matches = 0

        eps = 0.8

        for i in range(len(faces)):
            for j in range(len(testFaces)):
                if intersectionRate(faces[i][0], testFaces[j][0]) > eps:
                    faces_matches += 1
                    #check eyes
                    if len(eyes[i]) == 2:
                        if intersectionRate(eyes[i][0], testFaces[j][1]) > eps and intersectionRate(eyes[i][1] , testFaces[j][2]) > eps:
                            eyes_matches += 1
                        elif intersectionRate(eyes[i][1], testFaces[j][1]) > eps and intersectionRate(eyes[i][0], testFaces[j][2]) > eps:
                            eyes_matches += 1

        self.assertEqual(faces_matches, 2)
        self.assertEqual(eyes_matches, 2)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
