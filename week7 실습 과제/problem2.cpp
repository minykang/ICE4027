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

	// Gray 이미지를 BGR로 변환하여 채널을 맞춥니다.
	cvtColor(gray_img, gray_img, COLOR_GRAY2BGR);

	// 첫 번째 행의 이미지들을 합칩니다.
	Mat first_row;
	vector<Mat> imgs_1 = { bgr_img, gray_img, rgb_img };
	hconcat(imgs_1, first_row);

	// 두 번째 행의 이미지들을 합칩니다.
	Mat second_row;
	vector<Mat> imgs_2 = { hsv_img, yuv_img, xyz_img };
	hconcat(imgs_2, second_row);

	// 첫 번째 행을 디스플레이합니다.
	imshow("First Row", first_row);
	waitKey(0);

	// 두 번째 행을 디스플레이합니다.
	imshow("Second Row", second_row);
	waitKey(0);

	// 첫 번째 행과 두 번째 행을 수직으로 합칩니다.
	Mat combined;
	vconcat(first_row, second_row, combined);

	// 최종 결과를 디스플레이합니다.
	imshow("Combined Results", combined);
	waitKey(0);

	// 최종 결과를 이미지 파일로 저장합니다.
	imwrite("CvColorModels.png", combined);
}




Mat GetYCbCr(Mat src_img) {
	double b, g, r, y, cb, cr;
	Mat dst_img;
	src_img.copyTo(dst_img);

	//화소 인덱싱
	for (int row = 0; row < dst_img.rows; row++) {
		for (int col = 0; col < dst_img.cols; col++) {
			//BGR취득
			//openCV의 Mat은 BGR의 순서를 가짐에 유의
			b = (double)dst_img.at<Vec3b>(row, col)[0];
			g = (double)dst_img.at<Vec3b>(row, col)[1];
			r = (double)dst_img.at<Vec3b>(row, col)[2];

			//색상 변환 계산
			//정확한 계산을 위해 double 자료형 사용
			y = 0.2627 * r + 0.678 * g + 0.0593 * b;
			cb = -0.13963 * r - 0.36037 * g + 0.5 * b;
			cr = 0.5 * r - 0.45979 * g - 0.04021 * b;

			//오버플로우 방지
			y > 255.0 ? 255.0 : y < 0 ? 0 : y;
			cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
			cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;

			//변환된 색상 대입
			//double 자료형의 값을 본래 자료형으로 반환
			dst_img.at<Vec3b>(row, col)[0] = (uchar)y;
			dst_img.at<Vec3b>(row, col)[1] = (uchar)cb;
			dst_img.at<Vec3b>(row, col)[2] = (uchar)cr;


		}
	}
	imshow("results", dst_img);
	waitKey(0);

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

void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters,
	vector<vector<Point>>& ptInClusters) {

	RNG random(cv::getTickCount());

	for (int k = 0; k < n_cluster; k++) {
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point>ptInClustersK;
		ptInClusters.push_back(ptInClustersK);
	}


}

double computeColorDistance(Scalar pixel, Scalar clusterPixel) {

	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));

	return distance;

}
void findAssociatedCluster(Mat imgInput, int n_cluster, vector<Scalar> clustersCenters,
	vector<vector<Point>>& ptInClusters) {

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
			ptInClusters[closestClusterIndex].push_back(Point(c, r));
		}
	}
}

double adjustClusterCenters(Mat src_img, int n_cluster, vector<Scalar>& clustersCenters,
	vector<vector<Point>>ptInClusters, double& oldCenter, double newCenter) {

	double diffChange;

	for (int k = 0; k < n_cluster; k++) {

		vector<Point>ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		if (ptInCluster.size() > 0) { // 포인트가 있을 때만 계산
			for (int i = 0; i < ptInCluster.size(); i++) {
				Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
				newBlue += pixel.val[0];
				newGreen += pixel.val[1];
				newRed += pixel.val[2];
			}
			newBlue /= ptInCluster.size();
			newGreen /= ptInCluster.size();
			newRed /= ptInCluster.size();
		}
		else {
			continue; // 포인트가 없는 클러스터는 건너뜁니다
		}

		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		clustersCenters[k] = newPixel;
	}
	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);

	oldCenter = newCenter;

	return diffChange;
}


Mat applyFinalClusterTolmage(Mat src_img, int n_cluster, vector<vector<Point>>ptInClusters,
	vector<Scalar>clustersCenters) {
	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++) {

		vector<Point>ptInCluster = ptInClusters[k];

		for (int j = 0; j < ptInCluster.size(); j++) {
			dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
			dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
			dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];

		}
	}

	return dst_img;
}


Mat MyKmeans(Mat src_img, int n_cluster) {
	vector<Scalar>clustersCenters;
	vector<vector<Point>>ptInClusters;
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter;

	createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

	while (diffChange > threshold) {

		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }

		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);

		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);


	}
	Mat dst_img = applyFinalClusterTolmage(src_img, n_cluster, ptInClusters, clustersCenters);

	//imshow("results", dst_img);
	//waitKey(0);


	return dst_img;
}

double myMax(double a, double b, double c) {
	if (a > b && a > c)
		return a;
	if (b > c && b > a)
		return b;
	return c;
}

double myMin(double a, double b, double c) {
	if (a < b && a < c)
		return a;
	if (b < c && b < a)
		return b;
	return c;
}

double InRange(double b, double g, double r, double v, double min_val) {
	double h;
	if (v == r) {
		h = 60 * (0 + (g - b) / (v - min_val));
	}
	else if (v == g) {
		h = 60 * (2 + (b - r) / (v - min_val));
	}
	else if (v == b)
	{
		h = 60 * (4 + (r - g) / (v - min_val));
	}
	if (h < 0)
		h += 360;

	return h;
}

Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h, s, v;
	double min;
	Mat dst_img(src_img.size(), src_img.type());
	// 각 픽셀에 대해 반복수행
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			// 현재 픽셀의 파랑, 녹색, 빨강 값을 추출
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			v = myMax(b, g, r);
			min = myMin(b, g, r);// 파랑, 녹색, 빨강 중 최솟값을 찾기
			if (v == 0)
				s = 0;
			else
				s = (v - min) / v;

			// BGR을 HSV로 변환
			h = InRange(b, g, r, v, min);

			// 변환된 HSV 값을 대상 이미지에 할당
			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;
		}
	}
	return dst_img;
}



int main() {

	Mat src_img = imread("beach.jpg", -1);
	Mat dst_img1 = CvKMeans(src_img, 2);
	Mat dst_img2 = CvKMeans(src_img, 3);
	imshow("test1", src_img);
	imshow("test2", dst_img1);
	imshow("test3", dst_img2);
	waitKey(0);


	return 0;
}