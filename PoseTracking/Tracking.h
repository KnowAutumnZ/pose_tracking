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
#include "Optimizer.h"
#include "LocalMapping.h"

namespace PoseTracking
{
	//�ڲΡ�����
	static cv::Mat mK, mDistort;
	static float fx, fy, cx, cy;

	//ϵͳ��ʹ�õĴ���������
	enum eSensor;

	//��System�����õ���Tracking��Tracking�������õ���System���������������ǰ��
	class System;
	class Viewer;
	class Map;
	class FrameDrawer;
	class LocalMapping;
	class Initializer;

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

		/**
		 * @brief ���þֲ���ͼ���
		 *
		 * @param[in] pLocalMapper �ֲ���ͼ��
		 */
		void SetLocalMapper(LocalMapping* pLocalMapper);

		/**
		 * @brief ���ÿ��ӻ��鿴�����
		 *
		 * @param[in] pViewer ���ӻ��鿴��
		 */
		void SetViewer(Viewer* pViewer);

	public:
		/** @brief ��Ŀ�����ʱ�������еĳ�ʼ������ */
		void MonocularInitialization();

		/** @brief ��Ŀ�����ʱ�����ɳ�ʼ��ͼ */
		void CreateInitialMapMonocular();

		/**
		 * @brief �Բο��ؼ�֡��MapPoints���и���
		 *
		 * 1. ���㵱ǰ֡�Ĵʰ�������ǰ֡��������ֵ��ض����nodes��
		 * 2. ������ͬһnode�������ӽ���ƥ��
		 * 3. ����ƥ��Թ��Ƶ�ǰ֡����̬
		 * 4. ������̬�޳���ƥ��
		 * @return ���ƥ��������10������true
		 */
		bool TrackReferenceKeyFrame();

		/**
		 * @brief �������ٶ�ģ�Ͷ���һ֡��MapPoints���и���
		 *
		 * 1. �ǵ�Ŀ�������Ҫ����һ֡����һЩ�µ�MapPoints����ʱ��
		 * 2. ����һ֡��MapPointsͶӰ����ǰ֡��ͼ��ƽ���ϣ���ͶӰ��λ�ý�������ƥ��
		 * 3. ����ƥ��Թ��Ƶ�ǰ֡����̬
		 * 4. ������̬�޳���ƥ��
		 * @return ���ƥ��������10������true
		 * @see V-B Initial Pose Estimation From Previous Frame
		 */
		bool TrackWithMotionModel();

		/**
		 * @brief ˫Ŀ��rgbd����ͷ�������ֵΪ��һ֡�����µ�MapPoints
		 *
		 * ��˫Ŀ��rgbd����£�ѡȡһЩ���СһЩ�ĵ㣨�ɿ�һЩ�� \n
		 * ����ͨ�����ֵ����һЩ�µ�MapPoints
		 */
		void UpdateLastFrame();

		/**
		 * @brief ��Local Map��MapPoints���и���
		 * Step 1�����¾ֲ��ؼ�֡ mvpLocalKeyFrames �;ֲ���ͼ�� mvpLocalMapPoints
		 * Step 2���ھֲ���ͼ�в����뵱ǰ֡ƥ���MapPoints, ��ʵҲ���ǶԾֲ���ͼ����и���
		 * Step 3�����¾ֲ�����MapPoints���λ���ٴ��Ż�
		 * Step 4�����µ�ǰ֡��MapPoints���۲�̶ȣ���ͳ�Ƹ��پֲ���ͼ��Ч��
		 * Step 5�����ݸ���ƥ����Ŀ���ػ���������Ƿ���ٳɹ�
		 * @return true         ���ٳɹ�
		 * @return false        ����ʧ��
		 */
		bool TrackLocalMap();

		/**
		 * @brief ���¾ֲ���ͼ LocalMap
		 *
		 * �ֲ���ͼ���������ӹؼ�֡���ٽ��ؼ�֡�����Ӹ��ؼ�֡������Щ�ؼ�֡�۲⵽��MapPoints
		 */
		void UpdateLocalMap();

		/**
		 * @brief ���¾ֲ���ͼ�㣨���Ծֲ��ؼ�֡��
		 *
		 */
		void UpdateLocalPoints();

		/**
		* @brief ���¾ֲ��ؼ�֡
		* �����Ǳ�����ǰ֡��MapPoints�����۲⵽��ЩMapPoints�Ĺؼ�֡�����ڵĹؼ�֡���丸�ӹؼ�֡����ΪmvpLocalKeyFrames
		* Step 1��������ǰ֡��MapPoints����¼�����ܹ۲⵽��ǰ֡MapPoints�Ĺؼ�֡
		* Step 2�����¾ֲ��ؼ�֡��mvpLocalKeyFrames������Ӿֲ��ؼ�֡����������
		* Step 2.1 ����1���ܹ۲⵽��ǰ֡MapPoints�Ĺؼ�֡��Ϊ�ֲ��ؼ�֡ �����ھ���£��
		* Step 2.2 ����2����������1�õ��ľֲ��ؼ�֡�ﹲ�ӳ̶ȺܸߵĹؼ�֡�������ǵļ��˺��ھ���Ϊ�ֲ��ؼ�֡
		* Step 3�����µ�ǰ֡�Ĳο��ؼ�֡�����Լ����ӳ̶���ߵĹؼ�֡��Ϊ�ο��ؼ�֡
		*/
		void UpdateLocalKeyFrames();

		/**
		 * @brief �� Local MapPoints ���и���
		 *
		 * �ھֲ���ͼ�в����ڵ�ǰ֡��Ұ��Χ�ڵĵ㣬����Ұ��Χ�ڵĵ�͵�ǰ֡�����������ͶӰƥ��
		 */
		void SearchLocalPoints();

		/**
		 * @brief �ϵ�ǰ֡�Ƿ�Ϊ�ؼ�֡
		 * @return true if needed
		 */
		bool NeedNewKeyFrame();
		/**
		 * @brief �����µĹؼ�֡
		 *
		 * ���ڷǵ�Ŀ�������ͬʱ�����µ�MapPoints
		 */
		void CreateNewKeyFrame();
	public:
		//����״̬����
		enum eTrackingState {
			NOT_INITIALIZED = 1,          //<��ͼ����û����ɳ�ʼ��
			OK = 2,                       //<����ʱ��Ĺ���״̬
			LOST = 3                      //<ϵͳ�Ѿ������˵�״̬
		};

		//��ǵ�ǰϵͳ�Ǵ���SLAM״̬���Ǵ���λ״̬
		bool mbOnlyTracking;

		//����״̬
		eTrackingState mState;
		//��һ֡�ĸ���״̬.��������ڻ��Ƶ�ǰ֡��ʱ��ᱻʹ�õ�
		eTrackingState mLastProcessedState;

		//����������
		eSensor mSensor;

		//ͼ��
		cv::Mat mIm;

		//Motion Model
		cv::Mat mVelocity;

		//�ֲ��ؼ�֡����
		std::vector<KeyFrame*> mvpLocalKeyFrames;
		//�ֲ���ͼ��ļ���
		std::vector<MapPoint*> mvpLocalMapPoints;

		//���ٳ�ʼ��ʱǰ��֮֡���ƥ��
		std::vector<int> mvIniMatches;

		//��ʼ��������ƥ���������ǻ��õ��Ŀռ��
		std::vector<cv::Point3f> mvIniP3D;

		//���еĲο��ؼ�֡����ǰ֡��λ��;������ע�͵���˼,����洢��Ҳ�����λ��
		list<cv::Mat> mlRelativeFramePoses;

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

		//��ʼ�������еĲο�֡
		Frame* mpInitialFrame;
		//׷���߳�����һ����ǰ֡
		Frame* mpCurrentFrame;
		// ��һ֡
		Frame* mpLastFrame;

		//��Ŀ��ʼ��
		Initializer* mpInitializer{nullptr};

		// ��һ�ؼ�֡
		KeyFrame* mpLastKeyFrame;

		// ��һ���ؼ�֡��ID
		unsigned int mnLastKeyFrameId;
		// ��һ���ض�λ����һ֡��ID
		unsigned int mnLastRelocFrameId;

		//Local Map �ֲ���ͼ���
		//�ο��ؼ�֡
		KeyFrame* mpReferenceKF;// ��ǰ�ؼ�֡���ǲο�֡

	public:
		//��ǰ֡�еĽ���ƥ����ڵ�,���ᱻ��ͬ�ĺ�������ʹ��
		int mnMatchesInliers;

		// �½��ؼ�֡���ض�λ�������ж���С���ʱ��������֡���й�
		int mMinFrames;
		int mMaxFrames;

		//�ο��ؼ�֡
		std::list<KeyFrame*> mlpReferences;
		//����֡��ʱ���
		std::list<double> mlFrameTimes;
		//�Ƿ�����ı�־
		std::list<bool> mlbLost;

		//�ֲ���ͼ�����
		LocalMapping* mpLocalMapper;
	};
}