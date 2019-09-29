/*image_processing
# This project will be used for study image processing. 
# The first we will learn to using opencv library. All project use C++ to do all task.
*/


#include<iostream>
#include"opencv2/core.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/imgproc.hpp"
#include"opencv2/objdetect.hpp"
#include"opencv2/video.hpp"
#include"opencv2/highgui/highgui_c.h"
#include<math.h>

using namespace std;
using namespace cv;




int main(){

	//Read image for file
	string refFilename("POR1.jpg");
	cout << "Reading reference image : " << refFilename << endl;

	//Mat is using to store the reference image, vector
	Mat imProcess_Pure = imread(refFilename);

	// Check the image already opened
	if (imProcess_Pure.empty())
		std::cout << "failed to open POR1..jpg" << std::endl;
	else
		std::cout << "POR1.jpg loaded OK" << std::endl;


	//Convert to gray
	Mat imProcess_Gray;
	cvtColor(imProcess_Pure, imProcess_Gray, CV_BGR2GRAY);

	namedWindow("Immage_processed", WINDOW_AUTOSIZE);

	//show the image
	imshow("Immage_processed", imProcess_Gray);


	//Display the image in 5 sec
	waitKey(0);
	system("pause");
	
