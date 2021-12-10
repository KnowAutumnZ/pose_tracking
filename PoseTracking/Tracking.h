#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include "log.h"
#include "RRConfig.h"
#include "orbDetector.h"
#include "Frame.h"
#include "Initializer.h"

namespace PoseTracking
{
	//���ö���������� ��ʾ��ϵͳ��ʹ�õĴ���������
	enum eSensor {
		MONOCULAR = 0,
		STEREO = 1,
		RGBD = 2
	};

	class Tracking
	{
	public:
		/**
		 * @brief ���캯��
		 *
		 * @param[in] pSys              ϵͳʵ��
		 * @param[in] pVoc              �ֵ�ָ��
		 * @param[in] pFrameDrawer      ֡������
		 * @param[in] pMapDrawer        ��ͼ������
		 * @param[in] pMap              ��ͼ���
		 * @param[in] pKFDB             �ؼ�֡���ݿ���
		 * @param[in] strSettingPath    �����ļ�·��
		 * @param[in] sensor            ����������
		 */
		Tracking(const std::string &strSettingPath, eSensor sensor);


		/**
		 * @brief ����Ŀ����ͼ��
		 *
		 * @param[in] im            ͼ��
		 * @param[in] timestamp     ʱ���
		 * @return cv::Mat          ��������ϵ����֡�������ϵ�ı任����
		 */
		cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

		/** @brief ��׷�ٽ��� */
		void Track();

		/** @brief ��Ŀ�����ʱ�������еĳ�ʼ������ */
		void MonocularInitialization();

	public:
		//����״̬����
		enum eTrackingState {
			NO_IMAGES_YET = 0,            //<��ǰ��ͼ��
			NOT_INITIALIZED = 1,          //<��ͼ����û����ɳ�ʼ��
			OK = 2,                       //<����ʱ��Ĺ���״̬
			LOST = 3                      //<ϵͳ�Ѿ������˵�״̬
		};

		//����״̬
		eTrackingState mState;
		//��һ֡�ĸ���״̬.��������ڻ��Ƶ�ǰ֡��ʱ��ᱻʹ�õ�
		eTrackingState mLastProcessedState;

	private:
		// ORB
		// orb������ȡ�������ܵ�Ŀ����˫Ŀ��mpORBextractorLeft��Ҫ�õ�
		// �����˫Ŀ����Ҫ�õ�mpORBextractorRight
		// NOTICE ����ǵ�Ŀ���ڳ�ʼ����ʱ��ʹ��mpIniORBextractor������mpORBextractorLeft��
		// mpIniORBextractor��������ȡ�������������mpORBextractorLeft������

		//ORB��������ȡ��
		orbDetector* mpORBextractorLeft, *mpORBextractorRight;
		//�ڳ�ʼ����ʱ��ʹ�õ���������ȡ��,����ȡ������������������
		orbDetector* mpIniORBextractor;

		//����������
		eSensor mSensor;

		//�ڲΡ�����
		cv::Mat mK, mDistort;

		//��ʼ�������еĲο�֡
		Frame mInitialFrame;
		//׷���߳�����һ����ǰ֡
		Frame mCurrentFrame;
		//��һ֡
		Frame mLastFrame;

		//��Ŀ��ʼ��
		Initializer* mpInitializer{nullptr};

		//���ٳ�ʼ��ʱǰ��֮֡���ƥ��
		std::vector<int> mvIniMatches;
	};
}