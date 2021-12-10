#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace PoseTracking
{
	class Initializer
	{
	public:
		/**
		 * @brief 根据参考帧构造初始化器
		 *
		 * @param[in] ReferenceFrame        参考帧
		 * @param[in] sigma                 测量误差
		 * @param[in] iterations            RANSAC迭代次数
		 */
		Initializer(const cv::Mat& K, const Frame &ReferenceFrame,
			float sigma = 1.0,
			int iterations = 200);

	private:
		/** 相机内参 */
		cv::Mat mK;

		/** 测量误差 */
		float mSigma, mSigma2;

		/** 算Fundamental和Homography矩阵时RANSAC迭代次数  */
		int mMaxIterations;

		/** 二维容器，外层容器的大小为迭代次数，内层容器大小为每次迭代算H或F矩阵需要的点,实际上是八对 */
		std::vector<std::vector<size_t> > mvSets;

		// (Frame 1)
		/** 存储Reference Frame中的特征点 */
		std::vector<cv::KeyPoint> mvKeys1;

		// (Frame 2)
		/** 存储Current Frame中的特征点 */
		std::vector<cv::KeyPoint> mvKeys2;
	};
}
