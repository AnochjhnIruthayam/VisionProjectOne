/*
 * mainProject.cpp
 *
 *  Created on: Oct 14, 2013
 *      Author: anoch
 */


#include "functionsProject.hpp"

void peppersOne(string filename)
{
	//open image
	Mat imgPepOrig = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//make histogram and output
	Mat imgPepHist = makeHistogram(imgPepOrig);
    namedWindow("Histogram for " + filename, CV_WINDOW_AUTOSIZE );
    imshow("Histogram for " + filename, imgPepHist );
    //imwrite( "/home/anoch/Documents/VisionPics/Histogram"+filename, imgPepHist );


    //filter noise out and output
    Mat imgPepMedian = orderStatisticFilter(imgPepOrig, MIN_FILTER);
    namedWindow("FilterMedian for " + filename, CV_WINDOW_AUTOSIZE );
    imshow("FilterMedian for " + filename, imgPepMedian );
    //imwrite( "/home/anoch/Documents/EclipseProject/VisionProjectOne/After"+filename, imgPepOneMedian );

    //make histogram of the restored image and output
	Mat imgPepAfterHist = makeHistogram(imgPepMedian);
    namedWindow("Histogram for " + filename, CV_WINDOW_AUTOSIZE );
    imshow("Histogram for " + filename, imgPepAfterHist );
    //imwrite( "/home/anoch/Documents/VisionPics/HistogramAfter"+filename, imgPepAfterHist );

}
void peppersTwo(string filename)
{
	//open image
	Mat imgPepOrig = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//make histogram and output
	Mat imgPepHist = makeHistogram(imgPepOrig);
	namedWindow("Histogram for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("Histogram for " + filename, imgPepHist );
	//imwrite( "/home/anoch/Documents/VisionPics/Hist"+filename, imgPepHist );


	//filter noise out and output
	Mat imgPepMedian = orderStatisticFilter(imgPepOrig, MEDIAN_FILTER);
	Mat imgPepMedian2 = orderStatisticFilter(imgPepMedian, MEDIAN_FILTER);
	namedWindow("FilterMedian for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("FilterMedian for " + filename, imgPepMedian2 );
	//imwrite( "/home/anoch/Documents/VisionPics/After2Median"+filename, imgPepMedian2 );

	//make histogram and output
	Mat imgPepHist2 = makeHistogram(imgPepMedian2);
	namedWindow("Histogram for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("Histogram for " + filename, imgPepHist2 );
	//imwrite( "/home/anoch/Documents/VisionPics/After2MedianHist"+filename, imgPepHist2 );

}
void peppersThree(string filename)
{
	//open image
	Mat imgPepOrig = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//make histogram and output
	Mat imgPepHist = makeHistogram(imgPepOrig);
	namedWindow("Histogram for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("Histogram for " + filename, imgPepHist );
	//imwrite( "/home/anoch/Documents/VisionPics/HistAMF60"+filename, imgPepHist );

	//filter noise out and output
	Mat imgPepAdaptiveMedian = adaptiveFilter(imgPepOrig, 60);
	namedWindow("FilterAdaptiveMedian60 for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("FilterAdaptiveMedian60 for " + filename, imgPepAdaptiveMedian );
	//imwrite( "/home/anoch/Documents/VisionPics/FilterAdaptiveMedian60"+filename, imgPepAdaptiveMedian );
}

void peppersFour(string filename)
{
	Mat imgPepOrig = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//filter out
	Mat imgPepNotch= notchProcess(imgPepOrig, 5, 2);
	namedWindow("Notch filter for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("Notch filter for " + filename, imgPepNotch );
	//imwrite( "/home/anoch/Documents/VisionPics/NotchFilter"+filename, imgPepNotch );

	//fourier after
	Mat imgPepSpec = FourierMagnitudeSpectrum(imgPepNotch);
	namedWindow("Fourier Magnitude Spectrum after filter for " + filename, CV_WINDOW_AUTOSIZE );
	imshow("Fourier Magnitude Spectrum after filter for " + filename, imgPepSpec );
}

int main( int argc, char* argv[] )
{
	peppersOne("peppers_1.png");
	waitKey(0);
	peppersTwo("peppers_2.png");
	waitKey(0);
	peppersThree("peppers_3.png");
	waitKey(0);
	peppersFour("peppers_4.png");
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}



