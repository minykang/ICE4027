#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class Point5D {
public:
    float x, y, l, u, v;

     //Point5D();
     //~Point5D();

    void accumPt(Point5D);
    void copyPt(Point5D);
    float getColorDist(Point5D);
    float getSpatialDist(Point5D);
    void scalePt(float);
    void setPt(float, float, float, float, float);
    void printPt();
};

void Point5D::accumPt(Point5D Pt) {
    x += Pt.x;
    y += Pt.y;
    l += Pt.l;
    u += Pt.u;
    v += Pt.v;
}

void Point5D::copyPt(Point5D Pt) {
    x = Pt.x;
    y = Pt.y;
    l = Pt.l;
    u = Pt.u;
    v = Pt.v;
}

float Point5D::getColorDist(Point5D Pt) {
    return sqrt(pow(l - Pt.l, 2) + pow(u - Pt.u, 2) + pow(v - Pt.v, 2));
}

float Point5D::getSpatialDist(Point5D Pt) {
    return sqrt(pow(x - Pt.x, 2) + pow(y - Pt.y, 2));
}

void Point5D::scalePt(float scale) {
    x *= scale;
    y *= scale;
    l *= scale;
    u *= scale;
    v *= scale;
}

void Point5D::setPt(float px, float py, float pl, float pa, float pb) {
    x = px;
    y = py;
    l = pl;
    u = pa;
    v = pb;
}

void Point5D::printPt() {
    cout << x << " " << y << " " << l << " " << u << " " << v << endl;
}

// Mean Shift 클래스 정의
class MeanShift {
public:
    float bw_spatial = 8;
    float bw_color = 16;
    float min_shift_color = 0.1;
    float min_shift_spatial = 0.1;
    int max_steps = 10;
    vector<Mat> img_split;
    MeanShift(float, float, float, float, int);
    void doFiltering(Mat&);
};

// Mean Shift 생성자
MeanShift::MeanShift(float bs, float bc, float msc, float mss, int ms) {
    bw_spatial = bs;
    bw_color = bc;
    max_steps = ms;
    min_shift_color = msc;
    min_shift_spatial = mss;
}

// Mean Shift 필터링 수행
void MeanShift::doFiltering(Mat& img) {
    int height = img.rows;
    int width = img.cols;
    split(img, img_split);

    Point5D pt, pt_prev, pt_cur, pt_sum;

    int pad_left, pad_right, pad_top, pad_bottom;
    size_t n_pt, step;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            pad_left = (col - bw_spatial) > 0 ? (col - bw_spatial) : 0;
            pad_right = (col + bw_spatial) < width ? (col + bw_spatial) : width;
            pad_top = (row - bw_spatial) > 0 ? (row - bw_spatial) : 0;
            pad_bottom = (row + bw_spatial) < height ? (row + bw_spatial) : height;

            pt_cur.setPt(row, col, (float)img_split[0].at<uchar>(row, col), (float)img_split[1].at<uchar>(row, col), (float)img_split[2].at<uchar>(row, col));
            step = 0;
            do {
                pt_prev.copyPt(pt_cur);
                pt_sum.setPt(0, 0, 0, 0, 0);
                n_pt = 0;
                for (int hx = pad_top; hx < pad_bottom; hx++) {
                    for (int hy = pad_left; hy < pad_right; hy++) {
                        pt.setPt(hx, hy, (float)img_split[0].at<uchar>(hx, hy), (float)img_split[1].at<uchar>(hx, hy), (float)img_split[2].at<uchar>(hx, hy));
                        if (pt.getColorDist(pt_cur) < bw_color) {
                            pt_sum.accumPt(pt);
                            n_pt++;
                        }
                    }
                }

                pt_sum.scalePt(1.0 / n_pt);
                pt_cur.copyPt(pt_sum);
                step++;
            } while ((pt_cur.getColorDist(pt_prev) > min_shift_color) &&
                (pt_cur.getSpatialDist(pt_prev) > min_shift_spatial) &&
                (step < max_steps));
            img.at<Vec3b>(row, col) = Vec3b(pt_cur.l, pt_cur.u, pt_cur.v);
        }
    }
}


void exCVMeanShift() {
    Mat img = imread("fruits.png");// fruits 이미지 읽어오기
    if (img.empty()) exit(-1);
    cout << " exCvMeanShift() " << endl;//함수 호출 메시지를 출력

    resize(img, img, Size(256, 256), 0, 0, cv::INTER_AREA);// 이미지를 크기 256x256으로 사이즈 재조정
    imshow("src", img);
    imwrite("exCVMeanShift.jpeg", img);

    pyrMeanShiftFiltering(img, img, 8, 16);
    // OpenCV에서 있는 pyrMeanShiftFiltering 함수를 사용하여 Mean Shift 필터링을 수행
    // 여기서 8은 공간 대역폭(spatial bandwidth)이고 16은 색상 대역폭(color bandwidth)이다

    imshow("Dst", img);
    waitKey();
    destroyAllWindows();
    imwrite("exCVMeanShift_dst.jpeg", img);
}

//lower level의 My Mean-Shifting 함수
void exMyMeanShift() {
    Mat img = imread("fruits.png");// fruits 이미지 읽어오기
    if (img.empty()) exit(-1);
    cout << " exMyMeanShift() " << endl;//함수 호출 메시지를 출력

    resize(img, img, Size(256, 256), 0, 0, cv::INTER_AREA);// 이미지를 크기 256x256으로 사이즈 재조정
    imshow("src", img);
    imwrite("exMyMeanShift.jpeg", img);

    cvtColor(img, img, cv::COLOR_RGB2Luv);// RGB 색상 공간에서 Luv 색상 공간으로 변환

    MeanShift MSProc(8, 16, 0.1, 0.1, 10);
    // Mean Shift 객체를 생성
    // 공간 대역폭 8, 색상 대역폭 16, 최소 이동값 0.1, 최대 반복 횟수 10으로 설정
   
    MSProc.doFiltering(img);// Mean Shift 필터링을 수행

    cvtColor(img, img, cv::COLOR_Luv2RGB);// 필터링된 이미지를 Luv 색상 공간에서 다시 RGB 색상 공간으로 변환

    imshow("Dst", img);
    waitKey();
    destroyAllWindows();
    imwrite("exMyMeanShift_dst.jpeg", img);
}


int main(int argc, const char* argv[]) {
    exCVMeanShift();
    exMyMeanShift();

    return 0;
}
