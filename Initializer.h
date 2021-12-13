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
		 * @brief ���ݲο�֡�����ʼ����
		 *
		 * @param[in] ReferenceFrame        �ο�֡
		 * @param[in] sigma                 �������
		 * @param[in] iterations            RANSAC��������
		 */
		Initializer(const cv::Mat& K, const Frame &ReferenceFrame,
			float sigma = 1.0,
			int iterations = 10);

		/**
		 * @brief �����������͵�Ӧ�Ծ���ѡȡ��ѵ����ָ����ʼ��֮֡��������̬�����������ǻ��õ���ʼ��ͼ��
		 * Step 1 ���¼�¼������Ե�ƥ���ϵ
		 * Step 2 ������ƥ��������������ѡ��8��ƥ��������Ϊһ�飬���ڹ���H�����F����
		 * Step 3 ����fundamental ���� ��homography ����Ϊ�˼��ٷֱ����̼߳���
		 * Step 4 ����÷ֱ������ж�ѡȡ�ĸ�ģ������λ��R,t
		 *
		 * @param[in] CurrentFrame          ��ǰ֡��Ҳ����SLAM�����ϵĵڶ�֡
		 * @param[in] vMatches12            ��ǰ֡��2���Ͳο�֡��1��ͼ�����������ƥ���ϵ
		 *                                  vMatches12[i]���ͣ�i��ʾ֡1�йؼ��������ֵ��vMatches12[i]��ֵΪ֡2�Ĺؼ�������ֵ
		 *                                  û��ƥ���ϵ�Ļ���vMatches12[i]ֵΪ -1
		 * @param[in & out] R21                   ����Ӳο�֡����ǰ֡����ת
		 * @param[in & out] t21                   ����Ӳο�֡����ǰ֡��ƽ��
		 * @param[in & out] vP3D                  ���ǻ�����֮�����ά��ͼ��
		 * @param[in & out] vbTriangulated        ������ǻ����Ƿ���Ч����ЧΪtrue
		 * @return true                     ��֡���Գɹ���ʼ��������true
		 * @return false                    ��֡�������ʼ������������false
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

		/** Match�����ݽṹ��pair,mvMatches12ֻ��¼Reference��Currentƥ���ϵ��������  */
		std::vector<Match> mvMatches12;
		/** ��¼Reference Frame��ÿ����������Current Frame�Ƿ���ƥ��������� */
		std::vector<bool> mvbMatched1;
	};
}
