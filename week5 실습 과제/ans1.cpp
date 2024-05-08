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
			dstData[y * width + x] = srcData[y * width + x];// dst data에 src data를 복사

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





int main()
{
    Mat result;
    Mat temp = imread("gear.jpg", 0);
    Mat src_img = imread("gear.jpg", 0);
    
    result = myGaussianFilter(temp);//가우시안 필터 실행

    imshow("before", src_img);
    imshow("after", result);
    waitKey(0);
    destroyAllWindows();
}