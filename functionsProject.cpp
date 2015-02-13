/*
 * functionsProject.cpp
 *
 *  Created on: Oct 15, 2013
 *      Author: anoch
 */

#include "functionsProject.hpp"





template<class ImgT>
void dftshift(ImgT& img) {
   const int cx = img.cols/2;
   const int cy = img.rows/2;

   ImgT tmp;
   ImgT topLeft(img, cv::Rect(0, 0, cx, cy));
   ImgT topRight(img, cv::Rect(cx, 0, cx, cy));
   ImgT bottomLeft(img, cv::Rect(0, cy, cx, cy));
   ImgT bottomRight(img, cv::Rect(cx, cy, cx, cy));

   topLeft.copyTo(tmp);
   bottomRight.copyTo(topLeft);
   tmp.copyTo(bottomRight);

   topRight.copyTo(tmp);
   bottomLeft.copyTo(topRight);
   tmp.copyTo(bottomLeft);
}

Mat notchFilter(double d0, double n, int wy, int wx, double u_k, double v_k)
{
	int cx = wx/2;
	int cy = wy/2;
    cv::Mat_<cv::Vec2f> hpf(wy, wx);
    for(int y = 0; y < wy; ++y)
    {
	   for(int x = 0; x < wx; ++x)
	   {
		  // Real part
		  const double d_k = sqrt( double((x-cx-u_k)*(x-cx-u_k)) + double((y-cy-v_k)*(y-cy-v_k)) );
		  const double d_mk = sqrt( double((x-cx+u_k)*(x-cx+u_k)) + double((y-cy+v_k)*(y-cy+v_k)) );

		  if(d_k==0 || d_mk == 0) // Avoid division by zero
			  hpf(y,x)[0] = 0;
		  else
			  hpf(y,x)[0] = (1.0 / (1.0 + std::pow(d0/d_k, 2.0*n)))*(1.0 / (1.0 + std::pow(d0/d_mk, 2.0*n)));
		  // Imaginary part
		  hpf(y,x)[1] = 0;
	   }
    }
    return hpf;
}


Mat notchProcess(const Mat &img, double d0, double n1)
{

	// Get original size
	int wxOrig = img.cols;
	int wyOrig = img.rows;

	int m = getOptimalDFTSize( wyOrig );
	int n = getOptimalDFTSize( wxOrig );

	copyMakeBorder(img, img, 0, m - wyOrig, 0, n - wxOrig, BORDER_CONSTANT, Scalar::all(0));

	// Get padded image size
	const int wx = img.cols;
	const int wy = img.rows;

   // Print image sizes
   cout << "Original image size (rows,cols): (" << wyOrig << "," << wxOrig << ")" << std::endl;
   cout << "Padded image size (rows,cols): (" << wy << "," << wx << ")" << std::endl;

   // Compute DFT of image
   Mat_<float> imgs[] = {img.clone(), cv::Mat_<float>::zeros(wy, wx)};
   Mat_<Vec2f> img_dft;
   merge(imgs, 2, img_dft);
   dft(img_dft, img_dft);

   // Shift to center
   dftshift(img_dft);

   //Notch filter
   Mat notchfilterOutput;
   mulSpectrums(notchFilter(d0, n1, wx, wy, -79, 79), notchFilter(d0, n1, wx, wy, -31, -31),notchfilterOutput, DFT_ROWS );

   // Multiply and shift back
   mulSpectrums(notchfilterOutput, img_dft, img_dft, cv::DFT_ROWS);
   dftshift(img_dft);

   //or you can do like this, then you dont need to split
   Mat_<float> output;
   dft(img_dft, output, cv::DFT_INVERSE| cv::DFT_REAL_OUTPUT);

   normalize(output, output, 0, 1, CV_MINMAX);

   return output;
}

Mat FourierMagnitudeSpectrum(const Mat &img)
{

    Mat padded;
    int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols);
	copyMakeBorder(img, padded, 0, m-img.rows, 0, n-img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);

	split(complexI, planes);  // spilt to planes[0]=Re and planes[1]=Im part

	Mat_<float> mag, phase;
	cartToPolar(planes[0],planes[1],mag, phase, false);

    dftshift(mag);

    mag += Scalar::all(1);                    // switch to logarithmic scale
    log(mag, mag);

    normalize(mag, mag, 0, 1, CV_MINMAX); // Transform the matrix with float values into a

    return mag;

}

Mat makeHistogram(const Mat &img)
{
    int histSize = 256;

    /// Set the ranges)
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat imgHist;

    calcHist( &img, 1, 0, Mat(), imgHist, 1, &histSize, &histRange, uniform, accumulate );
    Mat histImage( 512, 512, CV_8UC3, Scalar( 255,255,255) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(imgHist, imgHist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


    int bin_w = cvRound( (double) 512/histSize);
    int hist_h = 512;

    for( int i = 1; i < histSize; i++ )
	 {

		 line( histImage, Point( bin_w*(i-1), hist_h - cvRound(imgHist.at<float>(i-1)) ) ,
						  Point( bin_w*(i), hist_h - cvRound(imgHist.at<float>(i)) ),
						  Scalar( 255, 0, 0), 2, 8, 0  );

	 }
    return histImage;

}

Mat orderStatisticFilter(const Mat &inputImage, int percentile)
{

	Mat imgCopy;
	inputImage.copyTo(imgCopy);
	double N8[9];
	//make offset, so it does not reach boarders
	for(int x = 1; x < inputImage.rows-1; x++)
	{
		for(int y = 1; y < inputImage.cols-1; y++)
		{
			//load values for the 8 neighbour values incl. the pixel itself
			N8[0] =  inputImage.at<uchar>(x-1,y-1);
			N8[1] =  inputImage.at<uchar>(x,y-1);
			N8[2] =  inputImage.at<uchar>(x+1,y-1);

			N8[3] =  inputImage.at<uchar>(x-1,y);
			N8[4] =  inputImage.at<uchar>(x,y);
			N8[5] =  inputImage.at<uchar>(x+1,y);

			N8[6] =  inputImage.at<uchar>(x-1,y+1);
			N8[7] =  inputImage.at<uchar>(x,y+1);
			N8[8] =  inputImage.at<uchar>(x+1,y+1);

			//sort from min to max
			int rowsize = 9;
			double tmp = 0;
			for (int i = 0; i < rowsize; i++)
			{
				for (int j = i+1; j < rowsize; j++)
				{
					if (N8[i] > N8[j])
					{
						tmp = N8[i];
						N8[i] = N8[j];
						N8[j] = tmp;
					}
				}
			}
			//SORT END

			// FILTER
			imgCopy.at<uchar>(x,y) = N8[percentile];

		}
	}
	return imgCopy;

}

void windowScan(const Mat &inputImage, int windowSize, int maxWindowSize, int x, int y, double& z_min, double& z_med, double& z_max)
{
	//declare the maximum possible size for the array
	double window[maxWindowSize*maxWindowSize];
	int windowHalf = windowSize/2;

	int currentPosition = 0;

	//save the pixels in the window S_xy
	for(int i = 0; i <= windowHalf; i++)
	{
		for(int j = 0; j <= windowHalf; j++)
		{
			window[currentPosition++] = inputImage.at<uchar>(x-i,y-j);
		}
	}
	for(int i = 1; i <= windowHalf; i++)
	{
		for(int j = 0; j <= windowHalf; j++)
		{
			window[currentPosition++] = inputImage.at<uchar>(x+i,y-j);
		}
	}
	for(int i = 0; i <= windowHalf; i++)
	{
		for(int j = 1; j <= windowHalf; j++)
		{
			window[currentPosition++] = inputImage.at<uchar>(x-i,y+j);
		}
	}
	for(int i = 1; i <= windowHalf; i++)
	{
		for(int j = 1; j <= windowHalf; j++)
		{
			window[currentPosition++] = inputImage.at<uchar>(x+i,y+j);
		}
	}
	//sorting from min to max
	int rowsize = windowSize*windowSize;
	double tmp = 0;
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = i+1; j < rowsize; j++)
		{
			if (window[i] > window[j])
			{
				tmp = window[i];
				window[i] = window[j];
				window[j] = tmp;
			}
		}
	}

	//calculate minimum, median and maximum
	z_min = window[0];
	z_med = window[(rowsize-1)/2];
	z_max = window[rowsize-1];

}

double adaptiveFilterStageA(const Mat &image_pad, int WindowSize, int maxWindowSize, int x, int y)
{
	double z_min = 0, z_med = 0, z_max = 0, z_xy=0;
	windowScan(image_pad, WindowSize,maxWindowSize , x, y,z_min, z_med, z_max);
	z_xy = image_pad.at<uchar>(x,y);
	double A1 = z_med - z_min;
	double A2 = z_med - z_max;
	if((A1 > 0) && (A2 < 0))
	{
		//Stage B
		return adaptiveFilterStageB(z_xy, z_min,z_med,z_max);
	}
	else
	{
		//increase window size
		WindowSize = WindowSize + 2; // always increase as a odd number
		if(WindowSize <= maxWindowSize)
		{
			//Stage A
			return adaptiveFilterStageA(image_pad, WindowSize, maxWindowSize, x, y);
		}
		else
		{
			return z_med;
		}
	}
}

double adaptiveFilterStageB(double z_xy, double z_min, double z_med, double z_max)
{
	//Stage B
	double B1 = z_xy - z_min;
	double B2 = z_xy - z_max;
	if((B1 > 0) && (B2 < 0))
	{
		return z_xy;
	}
	else
	{
		return z_med;
	}
}

Mat adaptiveFilter(const Mat &inputImage, int maxWindowSize)
{
	int padSize = maxWindowSize/2;
	//declare window size
	int WindowSize = 3;
	//make a copy
	Mat imgCopy;
	inputImage.copyTo(imgCopy);

	//pad the image according to the maxWindowSize
	Mat image_pad;
	copyMakeBorder(inputImage,image_pad,padSize,padSize,padSize,padSize,BORDER_CONSTANT,Scalar(0));
	double status = 1./(imgCopy.cols*imgCopy.rows);
	int currentState = 0;
	for(int x = padSize; x < image_pad.rows-padSize; x++)
	{
		for(int y = padSize; y < image_pad.cols-padSize; y++)
		{
			if(currentState == 0)
					currentState++;
			cout << (currentState*status)*100 << "%"<< endl;
			currentState++;

			double value = adaptiveFilterStageA(image_pad, WindowSize, maxWindowSize, x, y);
			imgCopy.at<uchar>(x-padSize,y-padSize) = value;
			WindowSize = 3;

		}
	}


	return imgCopy;
}
