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


void followEdges(int x, int y, Mat& magnitude, int tUpper, int tLower, Mat& edges) {
	edges.at<float>(y, x) = 255;

	// 이웃 픽셀 에지 따는 과정
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			if ((i == 0) && (j == 0) && (x + i >= 0) && (x + j >= 0) &&
				(x + i < magnitude.cols) && (y + j < magnitude.rows)) {
				if ((magnitude.at<float>(y + j, x + i) > tLower) &&
					(edges.at<float>(y + j, x + i) != 255)) {
					followEdges(x + i, y + j, magnitude, tUpper, tLower, edges);
					// 재귀적 방법으로 이웃 픽셀에서 확실한 edge를 찾아 edge를 구성
				}
			}
		}
	}
}

// 에지 감지 함수
void edgeDetect(Mat& magnitude, int tUpper, int tLower, Mat& edges) {
	int rows = magnitude.rows;
	int cols = magnitude.cols;

	edges = Mat(magnitude.size(), CV_32F, 0.0);

	// 픽셀 에지따기
	for (int x = 0; x < cols; x++) {
		for (int y = 0; y < rows; y++) {
			if (magnitude.at<float>(y, x) > tUpper) {
				followEdges(x, y, magnitude, tUpper, tLower, edges);
				// edge가 확실하면 이웃 픽셀을 따라확실한 edge를 탐색
			}
		}
	}
}

// 최대가 아닌 부분은 억제하는 함수
// 최대가 아니면 없애서 thinning과정됨
void nonMaximumSuppression(Mat& magnitudeImage, Mat& directionImage) {
	Mat checkImage = Mat(magnitudeImage.rows, magnitudeImage.cols, CV_8U);
	MatIterator_<float >itMag = magnitudeImage.begin<float>();
	MatIterator_<float >itDirection = directionImage.begin<float>();
	MatIterator_<unsigned char>itRet = checkImage.begin<unsigned char >();
	MatIterator_<float > itEnd = magnitudeImage.end<float>();

	for (; itMag != itEnd; ++itDirection, ++itRet, ++itMag) {
		const Point pos = itRet.pos();
		float currentDirection = atan(*itDirection) * (180 / 3.142);
		while (currentDirection < 0) currentDirection += 180;
		*itDirection = currentDirection;
		if (currentDirection > 22.5 && currentDirection <= 67.5) {
			if (pos.y > 0 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x - 1)) {
				magnitudeImage.at<float >(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float >(pos.y + 1, pos.x + 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 67.5 && currentDirection <= 112.5) {
			if (pos.y > 0 && *itMag <= magnitudeImage.at<float >(pos.y - 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && *itMag <= magnitudeImage.at<float >(pos.y + 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 112.5 && currentDirection <= 157.5) {
			if (pos.y > 0 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float >(pos.y - 1, pos.x + 1)) {
				magnitudeImage.at<float >(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x>0 && *itMag <= magnitudeImage.at<float >(pos.y + 1, pos.x - 1)) {
				magnitudeImage.at<float >(pos.y, pos.x) = 0;
			}
		}
		else {
			if (pos.x > 0 && *itMag <= magnitudeImage.at<float >(pos.y, pos.x - 1)) {
				magnitudeImage.at<float >(pos.y, pos.x) = 0;
			}
			if (pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float >(pos.y, pos.x + 1)) {
				magnitudeImage.at<float >(pos.y, pos.x) = 0;
			}
		}
	}

}


void ex3() {
	cout << "--- doCannyEx() ---\n" << endl;

	// 이미지 로드 
	Mat src_img = imread("rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	//Mat dst_img;
	Mat dst_img1, dst_img2, dst_img3, dst_img4, dst_img5;
#if USE_OPENCV
	// < Canny 에지 탐색 수행 >
	Canny(src_img, dst_img, 180, 240);
#else
	// Gaussian 필터 기반 노이즈 제거 
	Mat blur_img;
	GaussianBlur(src_img, blur_img, Size(3, 3), 1.5);

	// Sobel필터를 통한 에지 검출 
	Mat magX = Mat(src_img.rows, src_img.cols, CV_32F);
	Mat magY = Mat(src_img.rows, src_img.cols, CV_32F);
	Sobel(blur_img, magX, CV_32F, 1, 0, 3);
	Sobel(blur_img, magY, CV_32F, 0, 1, 3);

	// sobel 결과로 에지 방향 계산
	Mat sum = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodX = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodY = Mat(src_img.rows, src_img.cols, CV_64F);
	multiply(magX, magX, prodX);
	multiply(magY, magY, prodY);
	sum = prodX + prodY;
	sqrt(sum, sum);

	// < 에지 크기 계산 >
	Mat magnitude = sum.clone();


	//magnitude.convertTo(magnitude, CV_32F);  // 데이터 타입 변환

	// Non-maximum suppression을 적용하여 엣지 변환
	Mat slopes = Mat(src_img.rows, src_img.cols, CV_32F);
	divide(magY, magX, slopes);
	// gradient의 방향 계산
	nonMaximumSuppression(magnitude, slopes);

	// < Edge tracking by hysteresis >
	/*edgeDetect(magnitude, 200, 100, dst_img);*/
	/*dst_img.convertTo(dst_img, CV_8UC1);*/
#endif

	//edge tracking by hysteresis
	start = clock();
	edgeDetect(magnitude, 200, 100, dst_img1);
	finish = clock();
	cout << (float)(finish - start) << "ms" << endl;
	start = clock();
	edgeDetect(magnitude, 150, 100, dst_img2);
	finish = clock();
	cout << (float)(finish - start) << "ms" << endl;
	start = clock();
	edgeDetect(magnitude, 100, 50, dst_img3);
	finish = clock();
	cout << (float)(finish - start) << "ms" << endl;
	start = clock();
	edgeDetect(magnitude, 200, 30, dst_img4);
	finish = clock();
	cout << (float)(finish - start) << "ms" << endl;
	start = clock();
	edgeDetect(magnitude, 150, 50, dst_img5);
	finish = clock();
	cout << (float)(finish - start) << "ms" << endl;

	dst_img1.convertTo(dst_img1, CV_8UC1);
	dst_img2.convertTo(dst_img2, CV_8UC1);
	dst_img3.convertTo(dst_img3, CV_8UC1);
	dst_img4.convertTo(dst_img4, CV_8UC1);
	dst_img5.convertTo(dst_img5, CV_8UC1);




	// 결과 이미지 생성 세로로 합침
	Mat result_img, result_img2, result_img2_2, result_img3;
	hconcat(src_img, dst_img1, result_img);
	hconcat(dst_img2, dst_img3, result_img2);
	hconcat(dst_img4, dst_img5, result_img3);
	// 가로로 합침
	vconcat(result_img, result_img2, result_img2_2);
	vconcat(result_img2, result_img3, result_img3);


	// 각 결과 이미지를 따로 표시
	imshow("Result Image 1", result_img);
	waitKey(0); // 사용자가 키를 누를 때까지 기다림

	imshow("Result Image 2", result_img2_2);
	waitKey(0); // 사용자가 키를 누를 때까지 기다림

	imshow("Result Image 3", result_img3);
	waitKey(0); // 사용자가 키를 누를 때까지 기다림

}


int main() {
	

	
	ex3();



	return 0;
}