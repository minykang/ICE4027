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

Mat MyCopy(Mat srcImg)
{
    int width = srcImg.cols;
    int height = srcImg.rows;
    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dstData[y * width + x] = srcData[y * width + x];

        }
    }
    return dstImg;
}

int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height)
{
    int sum = 0;
    int sumKernel = 0;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++)
        {
            if ((y + j) >= 0 && (y + j) < height && (x + j) >= 0 && (x + i) < width) {
                sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
                sumKernel += kernel[i + 1][j + 1];
            }
        }
    }
    if (sumKernel != 0)
    {
        return sum / sumKernel;
    }
    else
        return sum;
}



Mat myGaussianFilter(Mat srcImg) {
    int width = srcImg.cols;
    int height = srcImg.rows;
    int kernel[9][9] = {
        { 1,2,3,4,5,4,3,2,1 },
        { 2,3,4,5,6,5,4,3,2 },
        { 3,4,5,6,7,6,5,4,3 },
        { 4,5,6,7,8,7,6,5,4 },
        { 5,6,7,8,9,8,7,6,5 },
        { 4,5,6,7,8,7,6,5,4 },
        { 3,4,5,6,7,6,5,4,3 },
        { 2,3,4,5,6,5,4,3,2 },
        { 1,2,3,4,5,4,3,2,1 }
    };
    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            dstData[y * width + x] = myKernelConv9x9(srcData, kernel, x, y, width, height);
        }
    }
    return dstImg;
}


Mat saltAndPepper(Mat img, int num) {
	//찍을 점의 개수를 num으로 설정
	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols; //이미지의 폭 정보저장
		int y = rand() % img.rows; //이미지의 높이 정보저장
		

		if (img.channels() == 1) {
			//이미지 채널수 반환
			if (n % 2 == 0) {
				img.at<char>(y, x) = 255; //단일 채널 접근
			}
			else {
				img.at<char>(y, x) = 0;
			}
		}

	}

	return img;
}


int main()
{
    Mat src_img = imread("gear.jpg", 0);

    Mat img = saltAndPepper(src_img, 500);
    //saltpepper함수 적용한 이미지 불러오기
    imshow("Saltpepper noise", img);

    Mat gaussianImg = myGaussianFilter(img);
    imshow("Gaussian Filter", img);
    //가우시안 필터 씌운 이미지 불러오기
    waitKey(0);
    destroyAllWindows();

  
    return 0;
}