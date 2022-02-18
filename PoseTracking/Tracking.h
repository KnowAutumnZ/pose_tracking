#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include "log.h"
#include "RRConfig.h"

#include "Frame.h"
#include "orbmatcher.h"
#include "orbDetector.h"
#include "Initializer.h"
#include "Map.h"
#include "KeyFrameDatabase.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Viewer.h"
#include "system.h"

namespace PoseTracking
{
	enum eSensor;

	//��System�����õ���Tracking��Tracking�������õ���System���������������ǰ��
	class System;
	class Viewer;
	class FrameDrawer;

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
		Tracking(const std::string &strSettingPath, FrameDrawer *pFrameDrawer, Map* pMap, MapDrawer* pMapDrawer, eSensor sensor);

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

		/** @brief ��Ŀ�����ʱ�����ɳ�ʼ��ͼ */
		void CreateInitialMapMonocular();

	public:
		//����״̬����
		enum eTrackingState {
			NOT_INITIALIZED = 1,          //<��ͼ����û����ɳ�ʼ��
			OK = 2,                       //<����ʱ��Ĺ���״̬
			LOST = 3                      //<ϵͳ�Ѿ������˵�״̬
		};

		//����״̬
		eTrackingState mState;
		//��һ֡�ĸ���״̬.��������ڻ��Ƶ�ǰ֡��ʱ��ᱻʹ�õ�
		eTrackingState mLastProcessedState;

		//��ǵ�ǰϵͳ�Ǵ���SLAM״̬���Ǵ���λ״̬
		bool mbOnlyTracking;

	public:
		// ORB
		// orb������ȡ�������ܵ�Ŀ����˫Ŀ��mpORBextractorLeft��Ҫ�õ�
		// �����˫Ŀ����Ҫ�õ�mpORBextractorRight
		// NOTICE ����ǵ�Ŀ���ڳ�ʼ����ʱ��ʹ��mpIniORBextractor������mpORBextractorLeft��
		// mpIniORBextractor��������ȡ�������������mpORBextractorLeft������

		//ORB��������ȡ��
		orbDetector* mpORBextractorLeft, *mpORBextractorRight;
		//�ڳ�ʼ����ʱ��ʹ�õ���������ȡ��,����ȡ������������������
		orbDetector* mpIniORBextractor;

		//��ǰϵͳ���е�ʱ��,�ؼ�֡�����������ݿ�
		KeyFrameDatabase* mpKeyFrameDB;

		//Map
		//(ȫ��)��ͼ���
		Map* mpMap;

		//Drawers  ���ӻ��鿴�����
		///�鿴��������
		Viewer* mpViewer;
		///֡���������
		FrameDrawer* mpFrameDrawer;
		///��ͼ���������
		MapDrawer* mpMapDrawer;

		//����������
		eSensor mSensor;

		//�ڲΡ�����
		cv::Mat mK, mDistort;

		//ͼ��
		cv::Mat mIm;

		//��ʼ�������еĲο�֡
		Frame mInitialFrame;
		//׷���߳�����һ����ǰ֡
		Frame mCurrentFrame;

		// ��һ�ؼ�֡
		KeyFrame* mpLastKeyFrame;
		// ��һ֡
		Frame mLastFrame;
		// ��һ���ؼ�֡��ID
		unsigned int mnLastKeyFrameId;
		// ��һ���ض�λ����һ֡��ID
		unsigned int mnLastRelocFrameId;

		//��Ŀ��ʼ��
		Initializer* mpInitializer{nullptr};

		//Local Map �ֲ���ͼ���
		//�ο��ؼ�֡
		KeyFrame* mpReferenceKF;// ��ǰ�ؼ�֡���ǲο�֡
		//�ֲ��ؼ�֡����
		std::vector<KeyFrame*> mvpLocalKeyFrames;
		//�ֲ���ͼ��ļ���
		std::vector<MapPoint*> mvpLocalMapPoints;

		//���ٳ�ʼ��ʱǰ��֮֡���ƥ��
		std::vector<int> mvIniMatches;

		//��ʼ��������ƥ���������ǻ��õ��Ŀռ��
		std::vector<cv::Point3f> mvIniP3D;
	};
}