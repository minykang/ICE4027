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


#include <iostream>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


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

vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) {
		if (i != 3) {
			Mat highImg = srcImg; // 수행하기 이전 영상을 백업

			srcImg = mySampling(srcImg); // down sampling
			srcImg = myGaussianFilter(srcImg); // gaussian filtering

			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size()); // 작아진 영상을 백업한 영상의 크기로 확대
			Vec.push_back(highImg - lowImg + 128);
			// 차 연산을 벡터 배열에 삽입
			// 0~255 범위를 벗어나는 것을 방지하기 위해 128을 더함
		}
		else {
			Vec.push_back(srcImg);
		}
	}

	return Vec;
}

int main()
{
	Mat src_img = imread("gear.jpg", 1);
	Mat dst_img;

	vector<Mat> Vecpyra = myLaplacianPyramid(src_img);

	imshow("1", Vecpyra[0]);
	imshow("2", Vecpyra[1]);
	imshow("3", Vecpyra[2]);
	imshow("4", Vecpyra[3]);

	waitKey(0);
	destroyAllWindows();

	reverse(Vecpyra.begin(), Vecpyra.end()); // 작은 영상부터 처리하기 위해 vector의 순서를 반대로 함

	for (int i = 0; i < Vecpyra.size(); i++) { // Vector의 크기만큼 반복
		if (i == 0) { // 가장 작은 영상은 차 영상이 아니므로 바로 출력
			dst_img = Vecpyra[i];

			imshow("Test window1", dst_img);
			waitKey(0);
			destroyWindow("Test window1");
		}
		else {
			resize(dst_img, dst_img, Vecpyra[i].size()); // 작은 영상을 확대
			dst_img = dst_img + Vecpyra[i] - 128; // 차 영상을 다시 더해 큰 영상으로 복원
			// 앞서 더했던 128을 다시 빼준다.

			imshow("Test window2", dst_img);
			waitKey(0);
			destroyWindow("Test window2");
		}
	}
	waitKey(0);
	destroyAllWindows();

	return 0;
}

