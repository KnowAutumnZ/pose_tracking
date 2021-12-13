#pragma once

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace PoseTracking
{
	typedef std::pair<int, int> Match;

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
			int iterations = 10);

		/**
		 * @brief 计算基础矩阵和单应性矩阵，选取最佳的来恢复出最开始两帧之间的相对姿态，并进行三角化得到初始地图点
		 * Step 1 重新记录特征点对的匹配关系
		 * Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
		 * Step 3 计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算
		 * Step 4 计算得分比例来判断选取哪个模型来求位姿R,t
		 *
		 * @param[in] CurrentFrame          当前帧，也就是SLAM意义上的第二帧
		 * @param[in] vMatches12            当前帧（2）和参考帧（1）图像中特征点的匹配关系
		 *                                  vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
		 *                                  没有匹配关系的话，vMatches12[i]值为 -1
		 * @param[in & out] R21                   相机从参考帧到当前帧的旋转
		 * @param[in & out] t21                   相机从参考帧到当前帧的平移
		 * @param[in & out] vP3D                  三角化测量之后的三维地图点
		 * @param[in & out] vbTriangulated        标记三角化点是否有效，有效为true
		 * @return true                     该帧可以成功初始化，返回true
		 * @return false                    该帧不满足初始化条件，返回false
		 */
		bool Initialize(const Frame &CurrentFrame,
			const std::vector<int> &vMatches12,
			cv::Mat &R21, cv::Mat &t21,
			std::vector<cv::Point3f> &vP3D,
			std::vector<bool> &vbTriangulated);

	private:
		inline void SeedRandOnce(int seed)
		{
			srand(seed);
		}

		inline int RandomInt(int min, int max) {
			int d = max - min + 1;
			return int(((double)rand() / ((double)RAND_MAX + 1.0)) * d) + min;
		}
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

		/** Match的数据结构是pair,mvMatches12只记录Reference到Current匹配上的特征点对  */
		std::vector<Match> mvMatches12;
		/** 记录Reference Frame的每个特征点在Current Frame是否有匹配的特征点 */
		std::vector<bool> mvbMatched1;
	};
}
