#pragma once

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/types/types_seven_dof_expmap.h"

namespace PoseTracking 
{
	/**
	 * @brief ʵ���� ORB-SLAM2�е�һЩ���õ�ת����
	 * @details ע������һ����ȫ�ľ�̬�࣬û�г�Ա���������еĳ�Ա������Ϊ��̬�ġ�
	 */
	class Converter
	{
	public:
		/**
		 * @brief ����cv::Mat��ʽ�洢��λ��ת����Ϊg2o::SE3Quat����
		 *
		 * @param[in] ��cv::Mat��ʽ�洢��λ��
		 * @return g2o::SE3Quat ����g2o::SE3Quat��ʽ�洢��λ��
		 */
		static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);

		/**
		 * @brief ����g2o::SE3Quat��ʽ�洢��λ��ת����Ϊcv::Mat��ʽ
		 *
		 * @param[in] SE3 �����g2o::SE3Quat��ʽ�洢�ġ���ת����λ��
		 * @return cv::Mat ת�����
		 * @remark
		 */
		static cv::Mat toCvMat(const g2o::SE3Quat &SE3);

		/**
		 * @brief ��4x4 double��Eigen����洢��λ��ת����Ϊcv::Mat��ʽ
		 *
		 * @paramp[in] m ����Eigen����
		 * @return cv::Mat ת�����
		 * @remark
		 */
		static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
	};
}