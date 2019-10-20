/*image_processing
 This project will be used for study image processing.
 The first we will learn to using opencv library. All project use C++ to do all task.
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/cvconfig.h>
#include <math.h>
#include <typeinfo>

using namespace std;
using namespace cv;

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
// Global variables

Mat src1, src_gray1, src2, src_gray2;
Mat srcCalib1, srcCalib_gray1, srcCalib2, srcCalib_gray2;
//image with edge detection
Mat dst1, detected_edges1, dst2, detected_edges2;
Mat dstCalib1, detected_edgesCalib1, dstCalib2, detected_edgesCalib2;

//defind parameter of canny edge
int edgeThresh = 1;
int lowThreshold = 20;
int const max_lowThreshold = 100;
int ratio_canny = 2;
int kernel_size = 3;

/*Main function of canny detect edge
* 
	param1 (int) : Using for threshold of the image
	im (Mat): the source image
	imEdge (Mat): Edge map of the image
*/
void CannyThreshold(int)
{
	// Reduce noise with a kernel 3x3
	blur(src_gray1, detected_edges1, Size(3, 3));
	blur(src_gray2, detected_edges2, Size(3, 3));

	// Canny detector
	Canny(detected_edges1, detected_edges1, lowThreshold, lowThreshold* ratio_canny, kernel_size);
	Canny(detected_edges2, detected_edges2, lowThreshold, lowThreshold* ratio_canny, kernel_size);
	// Using Canny's output as a mask, we display our result
	dst1 = Scalar::all(0);
	dst2 = Scalar::all(0);

	src1.copyTo(dst1, detected_edges1);
	src2.copyTo(dst2, detected_edges2);
}

//Canny for calib image
void CannyThresholdCalib(int)
{
	// Reduce noise with a kernel 3x3
	blur(srcCalib_gray1, detected_edgesCalib1, Size(3, 3));
	blur(srcCalib_gray2, detected_edgesCalib2, Size(3, 3));

	// Canny detector
	Canny(detected_edgesCalib1, detected_edgesCalib1, lowThreshold, lowThreshold* ratio_canny, kernel_size);
	Canny(detected_edgesCalib2, detected_edgesCalib2, lowThreshold, lowThreshold* ratio_canny, kernel_size);
	// Using Canny's output as a mask, we display our result
	dstCalib1 = Scalar::all(0);
	dstCalib2 = Scalar::all(0);

	srcCalib1.copyTo(dstCalib1, detected_edgesCalib1);
	srcCalib2.copyTo(dstCalib2, detected_edgesCalib2);
}

//**End Canny detect edge**//





/*
Homograph detect misalign image
*/

//Max features of 2 image
const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;
//POR homorgraphy 


//Function of misalignment detect 2 picture of output of canny detect edge
//Processing on gray image only
void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)

{
	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	cvtColor(im2, im2Gray, CV_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());


	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	// Draw top matches
	Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	imwrite("matches.jpg", imMatches);


	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	// Find homography
	h = findHomography(points1, points2, RANSAC);

	// Use homography to warp image
	warpPerspective(im1, im1Reg, h, im2.size());

}
//End of homograph detection.



/*
*	This is main function of Image_processing
	Function using canny detect edge and compare homograph for detect
	missalignment
*
*/
int main(int argc, char **argv)
{
	////////////////////////////////////////////////////////
	//                                                   ///
	//**       Calibration homography                  **///
	//                                                   ///
	////////////////////////////////////////////////////////
	//Reding calibration image
	string calib1Filename("calib1.jpg");
	cout << "Reading calib1 image: " << calib1Filename << endl;
	Mat imCalib1 = imread(calib1Filename);

	string calib2Filename("calib2.jpg");
	cout << "Reading calib1 image: " << calib2Filename << endl;
	Mat imCalib2 = imread(calib2Filename);

	// Create a matrix of the same type and size as src (for dst)
	dstCalib1.create(srcCalib1.size(), srcCalib1.type());
	dstCalib2.create(srcCalib2.size(), srcCalib2.type());

	//Detect edge using canny algorimths
	//store image
	srcCalib1 = imCalib1;
	srcCalib2 = imCalib2;

	//Transfer image to gray
	cvtColor(srcCalib1, srcCalib_gray1, CV_BGR2GRAY);
	cvtColor(srcCalib2, srcCalib_gray2, CV_BGR2GRAY);

	//imReference convert to edge detect
	CannyThresholdCalib(100);

	// Registered image will be resotred in imReg. 
	// The estimated homography will be stored in h. 
	Mat imRegCalib, hCalib;

	// Align images
	cout << "Calibration images ..." << endl;
	alignImages(dstCalib1, dstCalib2, imRegCalib, hCalib);


	// Print estimated homography
	cout << "Estimated calibra-homography : \n" << hCalib << endl;
	////////////////////////////////////////////////////////
	//                                                   ///
    //**       Calculate homograph with real image     **///
	//                                                   ///
	////////////////////////////////////////////////////////
	// Read reference image
	string refFilename("form.jpg");
	cout << "Reading reference image : " << refFilename << endl;
	Mat imReference = imread(refFilename);


	// Read image to be aligned
	string imFilename("scanned-form.jpg");
	cout << "Reading image to align : " << imFilename << endl;
	Mat im = imread(imFilename);

	// Create a matrix of the same type and size as src (for dst)
	dst1.create(src1.size(), src1.type());
	dst2.create(src2.size(), src2.type());

	//Detect edge using canny algorimths
	//store image
	src1 = imReference;
	src2 = im;

	//Transfer image to gray
	cvtColor(src1, src_gray1, CV_BGR2GRAY);
	cvtColor(src2, src_gray2, CV_BGR2GRAY);

	//imReference convert to edge detect
	CannyThreshold(100);

	// Registered image will be resotred in imReg. 
	// The estimated homography will be stored in h. 
	Mat imReg, h;

	// Align images
	cout << "Aligning images ..." << endl;
	alignImages(dst1, dst2, imReg, h);

	// Write aligned image to disk. 
	string outFilename("aligned.jpg");
	cout << "Saving aligned image : " << outFilename << endl;
	imwrite(outFilename, imReg);


	// Print estimated homography
	cout << "Estimated homography : \n" << h << endl;
	cout << "Type of homograph matrix : " << typeid(h).name() << endl;
	//Display the image sec
	waitKey(0);

	system ("pause");
	return 0;


}