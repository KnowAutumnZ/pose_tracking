#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace PoseTracking
{
	class Initializer
	{
	public:
		/**
		 * @brief ���ݲο�֡�����ʼ����
		 *
		 * @param[in] ReferenceFrame        �ο�֡
		 * @param[in] sigma                 �������
		 * @param[in] iterations            RANSAC��������
		 */
		Initializer(const cv::Mat& K, const Frame &ReferenceFrame,
			float sigma = 1.0,
			int iterations = 200);

	private:
		/** ����ڲ� */
		cv::Mat mK;

		/** ������� */
		float mSigma, mSigma2;

		/** ��Fundamental��Homography����ʱRANSAC��������  */
		int mMaxIterations;

		/** ��ά��������������Ĵ�СΪ�����������ڲ�������СΪÿ�ε�����H��F������Ҫ�ĵ�,ʵ�����ǰ˶� */
		std::vector<std::vector<size_t> > mvSets;

		// (Frame 1)
		/** �洢Reference Frame�е������� */
		std::vector<cv::KeyPoint> mvKeys1;

		// (Frame 2)
		/** �洢Current Frame�е������� */
		std::vector<cv::KeyPoint> mvKeys2;
	};
}
