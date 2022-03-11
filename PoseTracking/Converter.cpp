#include "Converter.h"

namespace PoseTracking
{
	//���任����ת��Ϊ�����se3��cv:Mat->g2o::SE3Quat
	g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
	{
		//���Ƚ���ת������ȡ����
		Eigen::Matrix<double, 3, 3> R;
		R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
			cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
			cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

		//Ȼ��ƽ��������ȡ����
		Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

		//����g2o::SE3Quat���Ͳ�����
		return g2o::SE3Quat(R, t);
	}

	//�����se3ת��Ϊ�任����g2o::SE3Quat->cv::Mat
	cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
	{
		//��ʵ�ʲ����ϣ�����ת����ΪEigen�еľ�����ʽ��Ȼ��ת����Ϊcv::Mat�ľ�����ʽ��
		Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
		//Ȼ������Eigen::Matrix->cv::Mat
		return toCvMat(eigMat);
	}

	//Eigen::Matrix<double,4,4> -> cv::Mat�����ڱ任����T���м�ת��
	cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
	{
		//���ȶ���洢�������ı���
		cv::Mat cvMat(4, 4, CV_32F);
		//Ȼ�����Ԫ�ظ�ֵ
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				cvMat.at<float>(i, j) = m(i, j);

		//���ؼ��������������������
		return cvMat.clone();
	}
}