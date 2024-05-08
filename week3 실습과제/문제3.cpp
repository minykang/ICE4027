#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat GetHistogram(Mat& src)
{
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	return histImage;
}

void bright(Mat& img)
{
	float k = 0.0;
	for (int i = 0; i < img.rows; i++)
	{
		if (k > 255)
			k = 255.0;

		for (int j = 0; j < img.cols; j++)
		{
			img.at<uchar>(i, j) = k;
		}
		k += 0.6;
	}
}



int main() {
	namedWindow("window", 1);
	Mat src_img = imread("img2.jpg", 0); // 이미지 읽기
	Mat bef = imread("img2.jpg", 0);
	bright(src_img);
	Mat gr;
	bitwise_not(src_img, gr);
	Mat result;
	//result = bef - gr;
	result = bef - src_img;

	//Mat histo = GetHistogram(result);
	

	imshow("Test window", result); // 이미지 출력
	waitKey(0); // 키 입력 대기(0: 키가 입력될 때 까지 프로그램 멈춤)
	destroyWindow("Test window"); // 이미지 출력창 종료
	

}
