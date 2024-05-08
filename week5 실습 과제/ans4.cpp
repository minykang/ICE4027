#include <iostream>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using std::string;
using namespace std;


int myKernerConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }
	else return sum;
}

Mat mySobelFilter(Mat srcImg) {
	int kernel45[3][3] = { -2,-1,0,
						   -1,0,1,
						   0,1,2 }; 
	// 45도 대각 에지 검출
	int kernel135[3][3] = { 0,1,2,
						  -1,0,1,
						  -2,-1,0 }; 
	// 135도 대각 에지 검출

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernerConv3x3(srcData, kernel45, x, y, width, height))
				+ abs(myKernerConv3x3(srcData, kernel135, x, y, width, height))) / 2;
			// 45도와 135도의 대각 에지를 검출하기 위해 두 에지 결과의 절댓값 합 형태로 해서 최종결과를 도출
		}
	}

	return dstImg;

}

int main()
{
	Mat src_img = imread("gear.jpg", 0);

	Mat img = mySobelFilter(src_img);
	imshow("SobelImg", img);
	waitKey(0);
	destroyAllWindows();
}