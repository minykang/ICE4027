#include <iostream>
using namespace std;

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;


Mat doBPF(Mat srcImg);


Mat doIdft(Mat complexImg);
Mat setComplex(Mat magImg, Mat phaImg);
Mat centralize(Mat complex);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat getMagnitude(Mat complexImg);
Mat doDft(Mat srcImg);

Mat padding(Mat img);




// Band Pass Filtering (BPF)라고함
Mat doBPF(Mat srcImg)
{
    
    Mat padImg = padding(srcImg);// 원본 이미지를 입력으로 받아 패딩된 이미지를 생성
    Mat complexImg = doDft(padImg);// 패딩된 이미지를 주파수 도메인으로 변환시킴
    Mat centerComplexImg = centralize(complexImg);//centralize를 통해 band pass filter의 모양을 확인할 수 있다.
    Mat magImg = getMagnitude(centerComplexImg);//앞서 centralize한 사진의 magnitude를 구함
    Mat phaImg = getPhase(centerComplexImg);//앞서 centralize한 사진의 phase를 구함

    // BPF 결과에서 최소값과 최대값을 찾는다
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
    normalize(magImg, magImg, 0, 1, NORM_MINMAX);
    // magnitude의 사진에서 최대와 최소와 각각의 location을 구함
    // 이때 magnitude의 최대 최소의 범위를 0-1로 정규화함.

    Mat maskImg = Mat::zeros(magImg.size(), CV_32F);//선언한 이미지의 크기를 magImg와 동일하게 설정하고 초기값은 0으로 함.
    circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(1), -1, -1, 0);//반지름 50
    circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 25, Scalar::all(0), -1, -1, 0);//반지름 25
    imshow("Band Pass Filter", maskImg);
    waitKey(0);
    destroyAllWindows();
    //25에서 50사이의 모양을 가진 band pass filter을 보여줌

    Mat magImg2;
    multiply(magImg, maskImg, magImg2);//원래의 magnitude이미지와 filter이미지를 곱함.

    
    normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);//정규화
    Mat complexImg2 = setComplex(magImg2, phaImg);//두 이미지를 곱한 주파수 도메인에서의 새로운 영상
    Mat dstImg = doIdft(complexImg2);//공간 도메인으로 변환

    return myNormalize(dstImg);//정규화한 이미지를 return
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
    Mat srcImg = imread("img1.jpg", 0);//원본사진
    Mat dstImg = doBPF(srcImg);//BPF 거친 사진

    imshow("Original Image", srcImg);
    imshow("After BPF", dstImg);

    waitKey(0);
    destroyAllWindows();

    return 0;
}