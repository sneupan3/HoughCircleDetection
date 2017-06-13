#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <wiringPi.h>

using namespace cv;
using namespace std;
static bool circleFound = false;
int readPin = 15;
void readPixel(Mat, Point, int);
void blobDetection(Mat);
Rect houghDetection(Mat src){
	cout << "Hough" << endl;
	Mat src_gray;
	//Convert it to gray
	cvtColor( src, src_gray, CV_BGR2GRAY );
	// Reduce the noise
	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
	vector<Vec3f> circles;
	// Apply the Hough Transform to find the circles
	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 100, 100, 60, 80, 350);
	Rect roi;
	for( size_t i = 0; i < circles.size(); i++ )
	{   
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		cout << "Circle Detected!\n Radius: " << radius << endl;
		cout << "Center: " << center.x <<", "<< center.y<< endl;
		readPixel(src, center, radius);
		int x = center.x;
		int y = center.y;
		roi = Rect(x-radius, y-radius, 2 * radius, 2* radius);
		circleFound = true;
//		if(roi != NULL){
//			cout<< "Not NULL" <<endl;
//		}
		//circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// Drawing the circle outline
		circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}   
	return roi;
}

int main(int args, char** argv){
wiringPiSetup();
pinMode (readPin, INPUT);

cout<<"Inside Main\n"<< endl;
//while (!digitalRead(readPin)) {cout<<"Waiting";}
	VideoCapture cap(0);
	if(cap.isOpened() == false){
		return -1;
	}

	Mat edges;
	Rect roi;
	namedWindow("houghCircle", 1);

	while(true){
		Mat frame;
		cap >> frame;
		roi = houghDetection(frame);
		//cvtColor(frame, edges, COLOR_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
		//Canny(edges, edges, 70, 30, 3);
		imshow("houghCircle", frame);
		if(circleFound == true){
			Mat croppedImage = frame(roi);
			blobDetection(croppedImage);
			imwrite("./houghImg.png", croppedImage);
			circleFound = false;
		}
		if(waitKey(30) >= 0) break;
	
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}


void readPixel(Mat img, Point p, int radius){
	
	Vec3b intensity = img.at<Vec3b>(p.y,p.x);
	int blue = intensity.val[0];
	int green = intensity.val[1];
	int red = intensity.val[2];
	cout << blue << " " << green << " " << red << endl;

}

void blobDetection(Mat im)
{

    // Read image
    //Mat im = imread( argv[1], IMREAD_GRAYSCALE );

    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold =80;
    params.maxThreshold = 200;

    params.filterByColor = true;
    params.blobColor = 0;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 25;
    params.maxArea = 60;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;
    //params.maxCircularity = 3;

    // Filter by Convexity
    //params.filterByConvexity = true;
    //params.minConvexity = 0.87;
	
	params.minDistBetweenBlobs = 5;
    // Filter by Inertia
    //params.filterByInertia = true;
    //params.minInertiaRatio = 0.01;


    // Storage for blobs
    vector<KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

    // Set up detector with params
    SimpleBlobDetector detector(params);

    // Detect blobs
    detector.detect( im, keypoints);
#else 

    // Set up detector with params
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

    // Detect blobs
    detector->detect( im, keypoints);
#endif 

    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
    // the size of the circle corresponds to the size of blob

    Mat im_with_keypoints;
    drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    cout << keypoints.size() << endl;
    // Show blobs
    imshow("keypoints", im_with_keypoints );
    //waitKey(0);

}
