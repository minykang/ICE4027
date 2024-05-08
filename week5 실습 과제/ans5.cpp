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


//3x3 커널의 컨볼루션 연산 함수
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

//컬러 영상에서 Gaussian filter에서 활용하는 마스크 배열 함수
int myKernerConv3x3(uchar* arr, int kernel[][3], int col, int row, int k, int width, int height)
{
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1; i <= 1; i++)
		{
			if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
			{
				//RGB 컬러 영상에서 row, col 항에 3을 곱한 값 사용
				int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
				sum += color * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	return sum / sumKernel;
}


Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[3][3] = { 1,2,1,
						 2,4,2,
						 1,2,1 };

	Mat dstImg(srcImg.size(), CV_8UC3); //컬러 영상에는 CV_8UC3를 사용
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = y * width * 3 + x * 3; //RGB 컬러 영상이므로 data 값에 3을 곱하기
			int index2 = (y * 2) * (width * 2) * 3 + (x * 2) * 3;

			for (int k = 0; k < 3; k++)
			{
				dstData[index + k] = myKernerConv3x3(srcData, kernel, x, y, k, width, height);
			}
		}
	}

	return dstImg;
}

//영상을 다운 샘플링하는 함수
Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC3);
	//컬러 영상이므로 CV_8UC3를 사용
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			//컬러 영상이므로 data 값에 3을 곱한거사용
			int i1 = y * width * 3 + x * 3;
			int i2 = (y * 2) * (width * 2) * 3 + (x * 2) * 3;

			dstData[i1 + 0] = srcData[i2 + 0];
			dstData[i1 + 1] = srcData[i2 + 1];
			dstData[i1 + 2] = srcData[i2 + 2];
		}
	}

	return dstImg;
}


//Gaussian 피라미드 생성 함수
vector<Mat> myGaussianPyramid(Mat src_img) {
	vector<Mat> Vec;

	Vec.push_back(src_img);
	for (int i = 0; i < 4; i++) {
#if USE_OPENCV
		pyrDown(src_img, src_img, Size(src_img.cols / 2, src_img.rows / 2));
#else
		src_img = mySampling(src_img);
		src_img = myGaussianFilter(src_img);//가우시안 필터링
#endif
		Vec.push_back(src_img);
	}

	return Vec;
}


int main()
{
	Mat src_img = imread("gear.jpg", 1);

	vector<Mat> pyramid = myGaussianPyramid(src_img);

	imshow("1", pyramid[0]);
	imshow("2", pyramid[1]);
	imshow("3", pyramid[2]);
	imshow("4", pyramid[3]);

	waitKey(0);
	destroyAllWindows();

	return 0;
}