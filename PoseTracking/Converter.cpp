#include "Converter.h"

namespace PoseTracking
{
	//将变换矩阵转换为李代数se3：cv:Mat->g2o::SE3Quat
	g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
	{
		//首先将旋转矩阵提取出来
		Eigen::Matrix<double, 3, 3> R;
		R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
			cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
			cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

		//然后将平移向量提取出来
		Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

		//构造g2o::SE3Quat类型并返回
		return g2o::SE3Quat(R, t);
	}

	//李代数se3转换为变换矩阵：g2o::SE3Quat->cv::Mat
	cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
	{
		//在实际操作上，首先转化成为Eigen中的矩阵形式，然后转换成为cv::Mat的矩阵形式。
		Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
		//然后再由Eigen::Matrix->cv::Mat
		return toCvMat(eigMat);
	}

	//Eigen::Matrix<double,4,4> -> cv::Mat，用于变换矩阵T的中间转换
	cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
	{
		//首先定义存储计算结果的变量
		cv::Mat cvMat(4, 4, CV_32F);
		//然后逐个元素赋值
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				cvMat.at<float>(i, j) = m(i, j);

		//返回计算结果，还是用深拷贝函数
		return cvMat.clone();
	}
}