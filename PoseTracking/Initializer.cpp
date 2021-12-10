#include "Initializer.h"

namespace PoseTracking
{
	/**
	 * @brief 根据参考帧构造初始化器
	 *
	 * @param[in] ReferenceFrame        参考帧
	 * @param[in] sigma                 测量误差
	 * @param[in] iterations            RANSAC迭代次数
	 */
	Initializer::Initializer(const cv::Mat& K, const Frame &ReferenceFrame, float sigma, int iterations)
	{
		//从参考帧中获取相机的内参数矩阵
		mK = K.clone();

		// 从参考帧中获取去畸变后的特征点
		mvKeys1 = ReferenceFrame.mvKeys;

		//获取估计误差
		mSigma = sigma;
		mSigma2 = sigma * sigma;

		//最大迭代次数
		mMaxIterations = iterations;
	}
}