#include <iostream>
#include "opencv2/core/core.hpp"// Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

void BlueSpreadSalts(Mat img, int num) //여기에서 img는 Blue를 뿌리는 이미지의 배경이고 num은 그 픽셀의 수를 나타낸다. 

{
	for (int n = 0; n < num; n++)
	{
		int x = rand() % img.cols; //이미지의 폭 정보 저장
		int y = rand() % img.rows; //이미지의 높이 정보 저장

		if (img.channels() == 1)//이미지의 채널 수 반환
		{
			img.at<uchar>(y, x) = 255;//단일 채널 접근
		}
		else
		{
			img.at<Vec3b>(y, x)[0] = 255; //Blue의 픽셀의 가도를 max를 설정해야잘보인다.

			img.at<Vec3b>(y, x)[2] = 0;//순수 blue의 값을 확인하기 위해서는 Blue의 채널이 아닌건 0으로 한다. 
			img.at<Vec3b>(y, x)[1] = 0;
		}
	}
}

void GreenSpreadSalts(Mat img, int num)
{
	for (int n = 0; n < num; n++)
	{
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		if (img.channels() == 1)
		{
			img.at<uchar>(y, x) = 255;
		}
		else
		{
			img.at<Vec3b>(y, x)[1] = 255;
			img.at<Vec3b>(y, x)[0] = 0;
			img.at<Vec3b>(y, x)[2] = 0;
		}
	}
}

void RedSpreadSalts(Mat img, int num)
{
	for (int n = 0; n < num; n++)
	{
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		if (img.channels() == 1)
		{
			img.at<uchar>(y, x) = 255;
		}
		else
		{
			img.at<Vec3b>(y, x)[2] = 255;
			img.at<Vec3b>(y, x)[0] = 0;
			img.at<Vec3b>(y, x)[1] = 0;
		}
	}
}

int countingBlue(Mat img)
{
	int count = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (img.at<Vec3b>(y, x)[0] == 255 && img.at<Vec3b>(y, x)[1] == 0 && img.at<Vec3b>(y, x)[2] == 0)
				count++;
		}
	}

	return count;
}
int countingGreen(Mat img)
{
	int count = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (img.at<Vec3b>(y, x)[1] == 255 && img.at<Vec3b>(y, x)[0] == 0 && img.at<Vec3b>(y, x)[2] == 0)
				count++;
		}
	}

	return count;
}

int countingRed(Mat img)
{
	int count = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (img.at<Vec3b>(y, x)[2] == 255 && img.at<Vec3b>(y, x)[0] == 0 && img.at<Vec3b>(y, x)[1] == 0)
				count++;
		}
	}

	return count;
}
int main() {
	namedWindow("window", 1);
	Mat src_img = imread("img1.jpg", -1); // 이미지 읽기
	// int flags = IMREAD_COLOR 또는 int flags = 1 -> 컬러 영상으로 읽음
	// int flags = IMREAD_GRAYSCALE 또는 int flags = 0 -> 흑백 영상으로 읽음
	// int flags = IMREAD_UNCHANGED 또는 int flags = -1 -> 원본 영상의 형식대로 읽음
	BlueSpreadSalts(src_img, 200);
	GreenSpreadSalts(src_img, 400);
	RedSpreadSalts(src_img, 500);
	imshow("Test window", src_img); // 이미지 출력
	waitKey(0); // 키 입력 대기(0: 키가 입력될 때 까지 프로그램 멈춤)
	destroyWindow("Test window"); // 이미지 출력창 종료
	imwrite("langding_gray.png", src_img); // 이미지 쓰기
	/*
	cout << "Blue : " << countingBlue(src_img) << "\n";
	cout << "Green : " << countingGreen(src_img) << "\n";
	cout << "Red : " << countingRed(src_img) << "\n";
	*/

	return 0;
}
