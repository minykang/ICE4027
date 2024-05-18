#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


void DogGrabCut() {
    Mat img = imread("MyDog.jpg", 1);
    imshow("src_img", img);

    Rect rect = Rect(Point(29.3, 55.56), Point(719.3, 952.77));
    Mat result, bg_model, fg_model;
    grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    Mat mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    img.copyTo(mask, result);

    imshow("mask", mask);
    imshow("result", result);

    waitKey(0);
}


void HamgrabCut() {
    Mat img = imread("hamster.jpg", 1);
    imshow("src_img", img);

    Rect rect = Rect(Point(374, 76), Point(802, 433));
    Mat result, bg_model, fg_model;
    grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    Mat mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    img.copyTo(mask, result);

    imshow("mask", mask);
    imshow("result", result);

    waitKey(0);
}


void CatgrabCut() {
    Mat img = imread("cat1.png", 1);
    imshow("src_img", img);

    Rect rect = Rect(Point(375, 30), Point(632, 464));
    Mat result, bg_model, fg_model;
    grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    Mat mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    img.copyTo(mask, result);

    imshow("mask", mask);
    imshow("result", result);

    waitKey(0);
}





int main() {

   
    //DogGrabCut();
    //HamgrabCut();
    CatgrabCut();
    return 0;
}