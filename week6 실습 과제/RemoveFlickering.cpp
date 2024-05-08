#include <iostream>
using namespace std;

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

Mat doIdft(Mat complexImg);
Mat setComplex(Mat magImg, Mat phaImg);
Mat centralize(Mat complex);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat getMagnitude(Mat complexImg);
Mat doDft(Mat srcImg);
Mat padding(Mat img);

void showMagnitude(Mat srcImg);
Mat removeFlicker(Mat srcImg);


// Flicker를 제거하는 함수
Mat removeFlicker(Mat srcImg)
{
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);// 크기 정보의 최솟값과 최댓값 계산
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);// 크기 정보를 0에서 1 범위로 정규화한다

    Mat maskImg = Mat::ones(magImg.size(), CV_32F);
    rectangle(maskImg, Rect(Point(maskImg.cols / 2 - 20, 0), Point(maskImg.cols / 2 + 20, maskImg.rows)), Scalar::all(0), -1);
    circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 5, Scalar::all(1), -1, -1, 0);
    imshow("Mask Image", maskImg);
    // 크기 정보와 같은 크기의 모두 1로 이루어진 행렬 생성하고
    // 중심 영역을 검은색으로 채우고 중심점을 흰색으로 채운다

    Mat maskedMagImg;
    multiply(magImg, maskImg, maskedMagImg);// 크기 정보에 마스크를 적용해서 중심 영역을 제외한 영역을 0으로 만든다
    imshow("Masked Magnitude Image", maskedMagImg);



    normalize(maskedMagImg, maskedMagImg, (float)minVal, (float)maxVal, NORM_MINMAX);
    // 정규화된 크기 정보를 원래 범위로 변환한다

    Mat complexImg2 = setComplex(maskedMagImg, phaImg);
    Mat dstImg = doIdft(complexImg2);
    return myNormalize(dstImg);
    // 주파수 도메인 역변환 수행하고 생성된 이미지를 정규화해서 반환한다.
}


void showMagnitude(Mat srcImg)
{
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);

    // 크기 정보 정규화하고 크기정보를 시각화하여 보여준다
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);
    imshow("Magnitude Image", magImg);
}

Mat padding(Mat img)
{
    int dftRows = getOptimalDFTSize(img.rows);
    int dftCols = getOptimalDFTSize(img.cols);

    Mat padded;
    copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
    return padded;
}


Mat doDft(Mat srcImg)
{
    Mat floatImg;
    srcImg.convertTo(floatImg, CV_32F);

    Mat complexImg;
    dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

    return complexImg;
}
// Magnitude 영상 취득
Mat getMagnitude(Mat complexImg)
{
    Mat planes[2];
    split(complexImg, planes);
    // 실수부, 허수부 분리

    Mat magImg;
    magnitude(planes[0], planes[1], magImg);
    magImg += Scalar::all(1);
    log(magImg, magImg);
    // magnitude 취득
    // log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

    return magImg;
}

Mat myNormalize(Mat src)
{
    Mat dst;
    src.copyTo(dst);
    normalize(dst, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

// Phase 영상 취득
Mat getPhase(Mat complexImg)
{
    Mat planes[2];
    split(complexImg, planes);
    // 실수부, 허수부 분리

    Mat phaImg;
    phase(planes[0], planes[1], phaImg);
    // phase 취득

    return phaImg;
}

// 좌표계 중앙 이동 (centralize)
Mat centralize(Mat complex)
{
    Mat planes[2];
    split(complex, planes);
    int cx = planes[0].cols / 2;
    int cy = planes[1].rows / 2;

    Mat qORe(planes[0], Rect(0, 0, cx, cy));
    Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
    Mat q2Re(planes[0], Rect(0, cy, cx, cy));
    Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

    Mat tmp;
    qORe.copyTo(tmp);
    q3Re.copyTo(qORe);
    tmp.copyTo(q3Re);
    q1Re.copyTo(tmp);
    q2Re.copyTo(q1Re);
    tmp.copyTo(q2Re);

    Mat q0Im(planes[1], Rect(0, 0, cx, cy));
    Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
    Mat q2Im(planes[1], Rect(0, cy, cx, cy));
    Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

    q0Im.copyTo(tmp);
    q3Im.copyTo(q0Im);
    tmp.copyTo(q3Im);
    q1Im.copyTo(tmp);
    q2Im.copyTo(q1Im);
    tmp.copyTo(q2Im);

    Mat centerComplex;
    merge(planes, 2, centerComplex);

    return centerComplex;
}

Mat setComplex(Mat magImg, Mat phaImg)
{
    exp(magImg, magImg);
    magImg -= Scalar::all(1);
    // magnitude 계산을 반대로 수행

    Mat planes[2];
    polarToCart(magImg, phaImg, planes[0], planes[1]);
    // 극 좌표계 -> 직교 좌표계 (각도와 크기로부터 2차원 좌표)

    Mat complexImg;
    merge(planes, 2, complexImg);
    // 실수부, 허수부 합체

    return complexImg;
}
Mat doIdft(Mat complexImg)
{
    Mat idftcvt;
    idft(complexImg, idftcvt);
    // IDFT를 이용한 원본 영상 취득

    Mat planes[2];
    split(idftcvt, planes);

    Mat dstImg;
    magnitude(planes[0], planes[1], dstImg);
    normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
    dstImg.convertTo(dstImg, CV_8UC1);
    // 일반 영상의 type과 표현범위로 변환

    return dstImg;
}


int main()
{
    Mat srcImg = imread("img3.jpg", 0);// 이미지 불러오기
    imshow("Original Image", srcImg);
    showMagnitude(srcImg);// 주파수 도메인 영상의 크기 정보를 시각화

    Mat dstImg = removeFlicker(srcImg);// 플리커가 제거된 이미지 생성
    imshow("Flicker Removed Image", dstImg);

    waitKey(0);
    destroyAllWindows();
}