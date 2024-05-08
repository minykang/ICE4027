#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define USE_OPENCV false;


using namespace cv;
using namespace std;

clock_t start, finish;

// 직접 salt_pepper noise를 없애기 위해 만든 함수 
// 중앙값 필터링을 수행하는 함수
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; // 이미지의 너비
	int hg = src_img.rows; // 이미지의 높이
	int kwd = kn_size.width; // 커널 너비
	int khg = kn_size.height; // 커널 높이
	int rad_w = kwd / 2; // 반경 너비
	int rad_h = khg / 2; // 반경 높이

	uchar* src_data = (uchar*)src_img.data;// 입력 이미지 데이터
	uchar* dst_data = (uchar*)dst_img.data;// 출력결과 이미지 데이터

	float* table = new float[kwd * khg]; // 커널 데이터를 동적할당한다
	float tmp;

	// 커널 안에서 각각의 픽셀에 대해서 수행
	for (int c = rad_w; c < wd - rad_w; c++) {
		for (int r = rad_h; r < hg - rad_h; r++) {
			tmp = 0.0f;
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					tmp = (float)src_data[(r + kr) * wd + (c + kc)];
					table[(kr + rad_h) * kwd + (kc + rad_w)] = tmp;
				}
			}
			sort(table, table + kwd * khg); // 커널 데이터를 정렬
			dst_data[r * wd + c] = (uchar)table[(kwd * khg) / 2]; // 중간값을 결과 이미지에 적용
		}
	}

	delete[] table; // 동적 할당된 메모리를 해제한다
}

//median함수를 만든것을 수행하게 하는 함수
void ex1() {
	cout << "--- doMedianEx() ---\n" << endl;

	// 이미지 불러오기
	Mat src_img = imread("salt_pepper.png", 0);
	if (!src_img.data) {
		printf("No image data \n");
	}

	// Median 필터링함수를 수행 수행
	Mat dst_img;
#ifdef USE_OPENCV
	medianBlur(src_img, dst_img, 5);
#else
	myMedian(src_img, dst_img, Size(5, 5));
#endif

	// 결과 출력
	Mat result_img;
	hconcat(src_img, dst_img, result_img); // 원본과 결과를 가로로 연결한다
	imshow("doMedianEx()", result_img);
	waitKey(0);
}





int main() {
	
	ex1();



	return 0;
}