/*image_processing
 This project will be used for study image processing.
 The first we will learn to using opencv library. All project use C++ to do all task.
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/cvconfig.h>
#include <math.h>

using namespace std;
using namespace cv;


/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

//defind parameter of canny edge
int edgeThresh = 1;
int lowThreshold = 30;
int const max_lowThreshold = 100;
int ratio = 2;
int kernel_size = 3;
char* window_name = "Edge Map";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}
//Edge detection


int main() {

	//Read image for file
	string refFilename("lena5.jpg");
	cout << "Reading reference image : " << refFilename << endl;

	//Mat is using to store the reference image, vector
	Mat imProcess_Pure = imread(refFilename);
	src = imProcess_Pure;

	// Check the image already opened
	if (src.empty())
		std::cout << "failed to open lena5.jpg" << std::endl;
	else
		std::cout << "lena5.jpg loaded OK" << std::endl;

	// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	//Convert to gray
	Mat imProcess_Gray;
	cvtColor(imProcess_Pure, imProcess_Gray, CV_BGR2GRAY);

	//src is the imProcess_Gray
	src_gray = imProcess_Gray;

	// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	// Show the image
	CannyThreshold(100, window_name);

	//Display the image sec
	waitKey(0);
	system("pause");
}