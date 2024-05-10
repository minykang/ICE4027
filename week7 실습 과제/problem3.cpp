#include <iostream>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"// Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더

using namespace cv;
using std::string;
using namespace std;


void CvColorModels(Mat bgr_img) {
	Mat gray_img, rgb_img, hsv_img, yuv_img, xyz_img;

	cvtColor(bgr_img, gray_img, COLOR_BGR2GRAY);
	cvtColor(bgr_img, rgb_img, COLOR_BGR2RGB);
	cvtColor(bgr_img, hsv_img, COLOR_BGR2HSV);
	cvtColor(bgr_img, yuv_img, COLOR_BGR2YCrCb);
	cvtColor(bgr_img, xyz_img, COLOR_BGR2XYZ);

	Mat print_img;
	bgr_img.copyTo(print_img);
	cvtColor(gray_img, gray_img, COLOR_GRAY2BGR);
	hconcat(print_img, gray_img, print_img);
	hconcat(print_img, rgb_img, print_img);
	hconcat(print_img, hsv_img, print_img);
	hconcat(print_img, yuv_img, print_img);
	hconcat(print_img, xyz_img, print_img);

	imshow("results", print_img);
	imwrite("CvColorModels.png", print_img);

	waitKey(0);
}

Mat GetYCbCr(Mat src_img) {
	double b, g, r, y, cb, cr;
	Mat dst_img;
	src_img.copyTo(dst_img);

	for (int row = 0; row < dst_img.rows; row++) {
		for (int col = 0; col < dst_img.cols; col++) {
			b = (double)dst_img.at<Vec3b>(row, col)[0];
			g = (double)dst_img.at<Vec3b>(row, col)[1];
			r = (double)dst_img.at<Vec3b>(row, col)[2];

			y = 0.2627 * r + 0.678 * g + 0.0593 * b;
			cb = -0.13963 * r - 0.36037 * g + 0.5 * b;
			cr = 0.5 * r - 0.45979 * g - 0.04021 * b;

			y = y > 255.0 ? 255.0 : y < 0 ? 0 : y;
			cb = cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
			cr = cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;

			dst_img.at<Vec3b>(row, col)[0] = (uchar)y;
			dst_img.at<Vec3b>(row, col)[1] = (uchar)cb;
			dst_img.at<Vec3b>(row, col)[2] = (uchar)cr;
		}
	}
	return dst_img;
}


Mat CvKMeans(Mat src_img, int k) {
	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) = (float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x * src_img.rows) = (float)src_img.at<uchar>(y, x);
			}
		}
	}

	Mat labels;
	Mat centers;
	int attempts = 5;
	kmeans(samples, k, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001),
		attempts, KMEANS_PP_CENTERS, centers);

	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] =
						(uchar)centers.at<float>(cluster_idx, z);
				}
			}
			else {
				dst_img.at<uchar>(y, x) =
					(uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}
	return dst_img;
}

void createClustersInfo(Mat imgInput, int n_cluster,
	vector<Scalar>& clustersCenters, vector<vector<Point>>& ptlnClusters) {
	RNG random(getTickCount());

	for (int k = 0; k < n_cluster; k++) {

		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);

		clustersCenters.push_back(centerK);

		vector<Point>ptlnClusterK;
		ptlnClusters.push_back(ptlnClusterK);
	}
}
double computeColorDistance(Scalar pixel, Scalar clusterPixel) {
	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));

	return distance;
}

void findAssociatedCluster(Mat imgInput, int n_cluster,
	vector<Scalar> clustersCenters, vector<vector<Point>>& ptlnClusters) {
	for (int r = 0; r < imgInput.rows; r++) {
		for (int c = 0; c < imgInput.cols; c++) {
			double minDistance = INFINITY;
			int closestClusterIndex = 0;
			Scalar pixel = imgInput.at<Vec3b>(r, c);

			for (int k = 0; k < n_cluster; k++) {
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);

				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}

			ptlnClusters[closestClusterIndex].push_back(Point(c, r));
		}
	}
}



double adjustClusterCenters(Mat src_img, int n_cluster,
	vector<Scalar>& clustersCenters, vector<vector<Point>> ptlnClusters,
	double& oldCenter, double newCenter) {

	double diffChange;

	for (int k = 0; k < n_cluster; k++) {
		vector<Point> ptlnCluster = ptlnClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		for (int i = 0; i < ptlnCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptlnCluster[i].y, ptlnCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptlnCluster.size();
		newGreen /= ptlnCluster.size();
		newRed /= ptlnCluster.size();

		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);

		clustersCenters[k] = newPixel;
	}

	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);

	oldCenter = newCenter;

	return diffChange;
}

Mat applyFinalClusterToImage(Mat src_img, int n_cluster,
	vector<vector<Point>> ptlnClusters,
	vector<Scalar>clustersCenters) {
	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++) {
		vector<Point>ptlnCluster = ptlnClusters[k];
		for (int j = 0; j < ptlnCluster.size(); j++) {
			dst_img.at<Vec3b>(ptlnCluster[j])[0] = clustersCenters[k].val[0];
			dst_img.at<Vec3b>(ptlnCluster[j])[1] = clustersCenters[k].val[1];
			dst_img.at<Vec3b>(ptlnCluster[j])[2] = clustersCenters[k].val[2];
		}
	}

	return dst_img;
}

Mat MyKMeans(Mat src_img, int n_cluster) {
	// 클러스터 중심과 포인트 정보를 저장하는 벡터를 선언
	vector <Scalar> clustersCenters;
	vector <vector<Point>> ptlnClusters;
	// 알고리즘 종료 기준을 설정
	double threshold = 0.001;
	// 초기 중심의 차이를 계산하기 위한 변수들을 초기화
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter;

	// 이미지를 기반으로 클러스터 정보를 생성
	createClustersInfo(src_img, n_cluster, clustersCenters, ptlnClusters);

	// 중심의 변화가 설정한 임계값보다 큰 동안 반복
	while (diffChange > threshold) {

		// 중심과 포인트 클러스터를 초기화
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) {
			ptlnClusters[k].clear();
		}

		// 각 포인트를 가장 가까운 클러스터에 연결
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptlnClusters);

		// 클러스터 중심을 조정하고 중심의 변화량을 계산
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptlnClusters,
			oldCenter, newCenter);

	}
	// 최종 클러스터링을 적용하여 이미지를 반환
	Mat dst_img = applyFinalClusterToImage(src_img, n_cluster, ptlnClusters, clustersCenters);

	return dst_img;
}



Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h = 0.0, s, v;// 파랑, 초록, 빨강, 색상, 채도, 밝기 변수

	Mat dst_img(src_img.size(), src_img.type());

	// 이미지 각 픽셀을 순회
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {

			// 현재 픽셀의 파랑, 초록, 빨강 성분을 추출
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			// 0에서 1 범위로 정규화
			b /= 255.0;
			g /= 255.0;
			r /= 255.0;

			//큰 값 작은 값을 찾기
			double cMax = max({ b, g, r });
			double cMin = min({ b, g, r });

			double delta = cMax - cMin;

			v = cMax;

			//채도 계산
			if (v == 0) s = 0;
			else s = (delta / cMax);

			//색상계산
			if (delta == 0) h = 0;
			else if (cMax == r) h = 60 * (g - b) / (v - cMin);
			else if (cMax == g) h = 120 + 60 * (b - r) / (v - cMin);
			else if (cMax == b) h = 240 + 60 * (r - g) / (v - cMin);

			if (h < 0) h += 360;

			//성분을 0에서 255
			v *= 255;
			s *= 255;
			h /= 2;


			h = h > 255.0 ? 255.0 : h < 0 ? 0 : h;
			s = s > 255.0 ? 255.0 : s < 0 ? 0 : s;
			v = v > 255.0 ? 255.0 : v < 0 ? 0 : v;

			// 새로운 HSV 값 설정
			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;

		}
	}

	return dst_img;
}
int main() {

	Mat src_img = imread("fruits.png", -1);
	Mat dst_img1 = MyBgr2Hsv(src_img);
	Mat dst_img2 = MyKMeans(src_img, 2);
	Mat dst_img3 = MyKMeans(src_img, 3);
	Mat dst_img4 = MyKMeans(src_img, 4);
	Mat dst_img5 = MyKMeans(src_img, 5);
	imshow("original", src_img);
	imshow("HSV", dst_img1);
	imshow("KMeans1", dst_img2);
	imshow("KMeans2", dst_img3);
	imshow("KMeans3", dst_img4);
	imshow("KMeans4", dst_img5);
	waitKey(0);

	return 0;
}