#include "Initializer.h"

namespace PoseTracking
{
	/**
	 * @brief ���ݲο�֡�����ʼ����
	 *
	 * @param[in] ReferenceFrame        �ο�֡
	 * @param[in] sigma                 �������
	 * @param[in] iterations            RANSAC��������
	 */
	Initializer::Initializer(const cv::Mat& K, const Frame &ReferenceFrame, float sigma, int iterations)
	{
		//�Ӳο�֡�л�ȡ������ڲ�������
		mK = K.clone();

		// �Ӳο�֡�л�ȡȥ������������
		mvKeys1 = ReferenceFrame.mvKeys;

		//��ȡ�������
		mSigma = sigma;
		mSigma2 = sigma * sigma;

		//����������
		mMaxIterations = iterations;
	}
}