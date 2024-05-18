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

Mat CvKmeans(Mat src_img, int k) {
	
	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) = (float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x + src_img.rows) = (float)src_img.at<uchar>(y, x);
			}
		}
	}

	
	Mat labels; // 군집 판별 결과가 담길 1차원 벡터
	Mat centers; // 각 군집의 중앙값(대표값)
	int attemps = 5;

	kmeans(samples, k, labels,
		TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001),
		attemps, KMEANS_PP_CENTERS, centers);

	//1차원 벡터 => 2차원 영상
	Mat dst_img(src_img.size(), src_img.type());

	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] = (uchar)centers.at<float>(cluster_idx, z);
					//군집판별 결과에 따라 각 군집의 중앙값으로 결과 생성
				}
			}
			else {
				dst_img.at<uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}

	imshow("results - CVkMeans", dst_img);
	waitKey(0);

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


Mat myInRange(Mat hsv_img)
{
	Mat dst_img(hsv_img.size(), hsv_img.type());// 출력 이미지 생성

	int img_color[5] = { 0, 0, 0, 0 };// 각 색상별 픽셀 수 저장하는 배열

	// HSV 이미지를 순회하면서 색상별로 픽셀 수를 계산
	for (int y = 0; y < hsv_img.rows; y++)
	{
		for (int x = 0; x < hsv_img.cols; x++)
		{
			// 색깔에 따라 픽셀 수 증가함
			if (0 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 20) //빨강
			{
				img_color[0]++;
			}
			else if (20 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 40) //노랑
			{
				img_color[1]++;
			}
			else if (40 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 70) //그린
			{
				img_color[2]++;
			}
			else if (70 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 120) //파랑
			{
				img_color[3]++;
			}
			
			
		}
	}
	//max 픽셀수 찾기
	int max_color = max({ img_color[0], img_color[1], img_color[2], img_color[3] });

	if (max_color == img_color[0])
	{
		cout << "Color is Red." << endl;
		for (int y = 0; y < hsv_img.rows; y++)
		{
			for (int x = 0; x < hsv_img.cols; x++)
			{
				//빨강이면 흰색
				if (0 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 20)
				{
					dst_img.at<Vec3b>(y, x)[0] = 255;
					dst_img.at<Vec3b>(y, x)[1] = 255;
					dst_img.at<Vec3b>(y, x)[2] = 255;
				}
				else
				{
					//그 외 검은색
					dst_img.at<Vec3b>(y, x)[0] = 0;
					dst_img.at<Vec3b>(y, x)[1] = 0;
					dst_img.at<Vec3b>(y, x)[2] = 0;
				}

				
			}
		}
	}
	//노랑
	else if (max_color == img_color[1])
	{
		cout << "Color is Yellow." << endl;
		for (int y = 0; y < hsv_img.rows; y++)
		{
			for (int x = 0; x < hsv_img.cols; x++)
			{
				if (20 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 40)
				{
					dst_img.at<Vec3b>(y, x)[0] = 255;
					dst_img.at<Vec3b>(y, x)[1] = 255;
					dst_img.at<Vec3b>(y, x)[2] = 255;
				}
				else
				{
					dst_img.at<Vec3b>(y, x)[0] = 0;
					dst_img.at<Vec3b>(y, x)[1] = 0;
					dst_img.at<Vec3b>(y, x)[2] = 0;
				}
			}
		}
	}
	//그린
	else if (max_color == img_color[2])
	{
		cout << "Color is Green." << endl;
		for (int y = 0; y < hsv_img.rows; y++)
		{
			for (int x = 0; x < hsv_img.cols; x++)
			{
				if (40 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 70)
				{
					dst_img.at<Vec3b>(y, x)[0] = 255;
					dst_img.at<Vec3b>(y, x)[1] = 255;
					dst_img.at<Vec3b>(y, x)[2] = 255;
				}
				else
				{
					dst_img.at<Vec3b>(y, x)[0] = 0;
					dst_img.at<Vec3b>(y, x)[1] = 0;
					dst_img.at<Vec3b>(y, x)[2] = 0;
				}
			}
		}
	}
	//파랑
	else if (max_color == img_color[3])
	{
		cout << "Color is Blue." << endl;
		for (int y = 0; y < hsv_img.rows; y++)
		{
			for (int x = 0; x < hsv_img.cols; x++)
			{
				if (70 < hsv_img.at<Vec3b>(y, x)[0] && hsv_img.at<Vec3b>(y, x)[0] <= 120)
				{
					dst_img.at<Vec3b>(y, x)[0] = 255;
					dst_img.at<Vec3b>(y, x)[1] = 255;
					dst_img.at<Vec3b>(y, x)[2] = 255;
				}
				else
				{
					dst_img.at<Vec3b>(y, x)[0] = 0;
					dst_img.at<Vec3b>(y, x)[1] = 0;
					dst_img.at<Vec3b>(y, x)[2] = 0;
				}
			}
		}
	}

	

	return dst_img;
}

int main()
{
	Mat src_img, hsv_img, dst_img;

	src_img = imread("fruits.png", 1);
	cout << "fruits : ";
	hsv_img = MyBgr2Hsv(src_img);
	dst_img = myInRange(hsv_img);

	//imshow("fruits : red", hsv_img);
	imshow("fruits : red", dst_img);


	waitKey(0);
	destroyAllWindows();
	return 0;
}
