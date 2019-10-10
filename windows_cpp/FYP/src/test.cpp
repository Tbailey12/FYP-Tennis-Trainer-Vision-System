/*
 * test.cpp
 *
 *  Created on: 10 Oct 2019
 *      Author: tom
 */

#include <test.hpp>
#include <iostream>

//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void MyFunc(){
	cv::Mat image;
	image = cv::imread("img/fish.jpg");

	std::cout << image.at<cv::Vec3b>(10,10) << std::endl;
	cv::Mat one = cv::Mat::ones(5,5,CV_8UC1);
	cv::cvtColor(image,image, cv::COLOR_BGR2GRAY);
	std::cout << image.at<cv::Vec3b>(4,4) << std::endl;

	cv::namedWindow("DisplayWindow",cv::WINDOW_AUTOSIZE);
	cv::imshow("DisplayWindow",image);
	cv::waitKey(0);
}
