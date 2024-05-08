#include <iostream>
using namespace std;

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

// 주파수 도메인
Mat mySobelFilterInFrequencyDomain(Mat srcImg);
Mat doIdft(Mat complexImg);
Mat setComplex(Mat magImg, Mat phaImg);
Mat centralize(Mat complex);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat getMagnitude(Mat complexImg);
Mat doDft(Mat srcImg);
Mat padding(Mat img);

// 공간 도메인
Mat mySobelFilter(Mat srcImg);
int myKernelConv3x3(uchar* arr, int kernel[3][3], int x, int y, int width, int height);


Mat mySobelFilterInFrequencyDomain(Mat srcImg) {

    // X 성분 커널 정의
    int kernelX[3][3] = { -1,0,1,
                         -2,0,2,
                         -1,0,1 };

    // Y 성분 커널 정의
    int kernelY[3][3] = { -1, -2, -1,
                           0, 0, 0,
                           1, 2, 1 };

    // 원본 이미지를 주파수 도메인으로 변환
    Mat complexImg = doDft(srcImg);
    // 주파수 도메인 이미지를 중앙 이동
    Mat centerComplexImg = centralize(complexImg);
    // 이미지의 magnitude 구하기
    Mat srcmag = getMagnitude(centerComplexImg);
    // 이미지의 phase 구하기
    Mat phaImg = getPhase(centerComplexImg);



    // X, Y 성분 커널을 주파수 도메인 형식으로 변환
    Mat sobelX = Mat(Size(3, 3), CV_8UC1, kernelX);
    Mat sobelY = Mat(Size(3, 3), CV_8UC1, kernelY);

    // X, Y 성분 커널을 주파수 도메인으로 변환
    Mat complexX = doDft(sobelX);
    Mat complexY = doDft(sobelY);

    // X , Y 성분 커널의 size를 이미지와 일치하도록 조절
    resize(complexX, complexX, Size(complexImg.cols, complexImg.rows));
    resize(complexY, complexY, Size(complexImg.cols, complexImg.rows));

    // X 성분 주파수 도메인 이미지를 중앙으로 이동
    Mat centerComplexImgX = centralize(complexX);
    // Y 성분 주파수 도메인 이미지를 중앙으로 이동
    Mat centerComplexImgY = centralize(complexY);

    // X, Y 성분 중앙화한 이미지의 magnitude 구하기
    Mat sobelMagX = getMagnitude(centerComplexImgX);
    Mat sobelMagY = getMagnitude(centerComplexImgY);

    // X, Y 성분의 최소값과 최대값을 변수에 저장
    double minValX, maxValX;
    double minValY, maxValY;

    // X, Y 성분의 최소값과 최대값을 찾고 0부터 1까지의 범위로 정규화
    minMaxLoc(sobelMagX, &minValX, &maxValX);
    normalize(sobelMagX, sobelMagX, 0, 1, NORM_MINMAX);
    minMaxLoc(sobelMagY, &minValY, &maxValY);
    normalize(sobelMagY, sobelMagY, 0, 1, NORM_MINMAX);

    // 결과 이미지 저장 변수
    Mat X;
    Mat Y;
    // 행렬 곱셈을 위해 이미지를 CV_32F 형식으로 변환
    Mat srcmat;
    Mat Xmat;
    Mat Ymat;

    // 이미지를 CV_32F 형식으로 변환
    srcmag.convertTo(srcmat, CV_32F);
    sobelMagX.convertTo(Xmat, CV_32F);
    sobelMagY.convertTo(Ymat, CV_32F);

    // X,Y 성분을 이미지의 magnitude와 곱
    multiply(Xmat, srcmat, X);
    multiply(Ymat, srcmat, Y);

    // X와 Y 성분의 결과 이미지를 X와 Y 성분의 최소 및 최대 값으로 정규화
    normalize(X, X, (float)minValX, (float)maxValX, NORM_MINMAX);
    normalize(Y, Y, (float)minValY, (float)maxValY, NORM_MINMAX);

    // X, Y 성분 이미지를 원본의 phase 이미지와 병합
    Mat getX = setComplex(X, phaImg);
    Mat getY = setComplex(Y, phaImg);

    // 병합된 이미지를 공간 도메인으로 변환하고 0에서 255까지 정규화
    Mat resultX = myNormalize(doIdft(getX));
    Mat resultY = myNormalize(doIdft(getY));

    // X와 Y 성분 이미지의 합 반환
    return (resultX + resultY);
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

Mat mySobelFilter(Mat srcImg)
{
    int kernelX[3][3] = { -1, 0, 1,
                         -2, 0, 2,
                         -1, 0, 1 };
    int kernelY[3][3] = { -1, -2, -1,
                         0, 0, 0,
                         1, 2, 1 };
    //x성분과 y성분의 커널을 선언한다

    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    int width = srcImg.cols;
    int height = srcImg.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
                abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) /
                2;
        }
    }

    return dstImg;
}
int myKernelConv3x3(uchar* arr, int kernel[3][3], int x, int y, int width, int height)
{
    int sum = 0;
    int sumKernel = 0;

    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width)
            {
                sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
                sumKernel += kernel[i + 1][j + 1];
            }
        }
        //시그마 계산을 할 때 컨볼루션 식을 사용하여 계산
        //각 원소들의 총합을 구해서 최종적으로 행렬의 총 합을 구해 나눈다.
    }
    if (sumKernel != 0)
    {
        return sum / sumKernel;
    }
    else
    {
        return sum;
    }
}

int main()
{
    Mat srcImg = imread("img2.jpg", 0);
    Mat frequencyDstImg = mySobelFilterInFrequencyDomain(srcImg);
    Mat spatialDstImg = mySobelFilter(srcImg);

    imshow("Original Image", srcImg);
    imshow("Sobel Filter in Frequency Domain", frequencyDstImg);
    imshow("Sobel Filter in Spatial Domain", spatialDstImg);

    waitKey(0);
    destroyAllWindows();

    return 0;
}