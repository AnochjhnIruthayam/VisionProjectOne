/*
 * functionsProject.hpp
 *
 *  Created on: Oct 15, 2013
 *      Author: anoch
 */

#ifndef FUNCTIONSPROJECT_HPP_
#define FUNCTIONSPROJECT_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

enum
{
	MIN_FILTER = 0,
	MEDIAN_FILTER = 4,
	MAX_FILTER = 8
};

Mat FourierMagnitudeSpectrum(const Mat &img);
Mat makeHistogram(const Mat &img);
Mat orderStatisticFilter(const Mat &inputImage, int percentile);
Mat adaptiveFilter(const Mat &inputImage, int maxWindowSize);
double adaptiveFilterStageB(double z_xy, double z_min, double z_med, double z_max);
double adaptiveFilterStageA(const Mat &image_pad, int WindowSize, int maxWindowSize, int x, int y);
void windowScan(const Mat &inputImage, int windowSize, int maxWindowSize, int x, int y, double& z_min, double& z_med, double& z_max);
Mat notchFilter(double d0, double n, int wy, int wx, double u_k, double v_k);
Mat notchProcess(const Mat &img, double d0, double n1);

#endif /* FUNCTIONSPROJECT_HPP_ */
