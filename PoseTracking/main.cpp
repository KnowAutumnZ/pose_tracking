#include <iostream>
#include "orbDetector.h"

using namespace PoseTracking;

int main()
{
	orbDetector orb(1000, 1.2, 8, 20, 12);

	std::vector<std::string> vimpath;
	cv::glob("./data/rgbd_dataset_freiburg2_desk/rgb/*.png", vimpath);

	for (size_t i=0; i<vimpath.size(); i++)
	{
		cv::Mat im = cv::imread(vimpath[i]);

		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		(orb)(im, keypoints, descriptors);

		cv::Mat imdraw = im.clone();
		cv::drawKeypoints(im, keypoints, imdraw, cv::Scalar(0, 255, 0));
	}

	//cv::Mat im1_ = cv::imread("./data/5.jpg");
	//cv::Mat im2_ = cv::imread("./data/7.jpg");

	////cv::resize(im2_, im2_, cv::Size(1280, 720));

	//cv::Mat im1 = im1_.clone();
	//cv::Mat im2 = im2_.clone();

	//cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);

	//cv::Mat im = cv::Mat(im1.rows, im1.cols + im2.cols, CV_8UC3);

	//im1_.copyTo(im(cv::Rect(0, 0, im1.cols, im1.rows)));
	//im2_.copyTo(im(cv::Rect(im1.cols, 0, im2.cols, im2.rows)));

	//std::vector<cv::Point> pts1;
	//for (size_t i = 0; i < im1_.rows; i+=15)
	//{
	//	for (size_t j = 0; j < im1_.cols; j+=15)
	//	{
	//		if (im1.ptr<uchar>(i)[j] > 130)
	//		{
	//			int x = j;
	//			int y = i;

	//			cv::circle(im1_, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
	//			//cv::line(im, cv::Point(x, y), cv::Point(x + im1.cols, y), cv::Scalar(0, 255, 255));

	//			pts1.push_back(cv::Point(x, y));
	//		}
	//	}
	//}

	//std::vector<cv::Point> pts2;
	//for (size_t i = 0; i < im2_.rows; i += 15)
	//{
	//	for (size_t j = 0; j < im2_.cols; j += 15)
	//	{
	//		if (im2.ptr<uchar>(i)[j] > 130)
	//		{
	//			int x = j;
	//			int y = i;

	//			cv::circle(im2_, cv::Point(x, y), 2, cv::Scalar(0, 0, 222), -1);
	//			pts2.push_back(cv::Point(x, y));
	//			//cv::line(im, cv::Point(x, y), cv::Point(x + im1.cols, y), cv::Scalar(0, 255, 255));
	//		}
	//	}
	//}

	//cv::imwrite("im1.jpg", im1_);
	//cv::imwrite("im2.jpg", im2_);

	//for (int i=0; i< pts2.size(); i++)
	//{
	//	cv::line(im, pts1[i], cv::Point(pts2[i].x +im1.cols, pts2[i].y), cv::Scalar(0, 255, 0));
	//}


	//cv::Mat im = abs(im2 - im1);
	//cv::imwrite("im.jpg", im);

	//cv::Mat im = cv::imread("./data/1.jpg", 0);
	//cv::Mat image_binary;
	//cv::threshold(im, image_binary, 30, 255, cv::THRESH_BINARY);

	//std::vector<std::vector<cv::Point> > vcontours;
	//std::vector<cv::Vec4i> hierarchy;
	//cv::findContours(image_binary, vcontours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

	//std::vector<std::vector<cv::Point> > vcontours2;
	//for (int i=0; i< vcontours.size(); i++)
	//{
	//	if (vcontours[i].size() < 20) continue;

	//	vcontours2.push_back(vcontours[i]);
	//}
	//cv::drawContours(im, vcontours2, -1, cv::Scalar::all(255));
	//cv::imwrite("im.jpg", im);

	return 0;
}