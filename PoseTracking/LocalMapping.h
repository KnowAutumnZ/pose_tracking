#pragma once

#include "KeyFrame.h"
#include "Map.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>

namespace PoseTracking
{
	class Map;
	class Tracking;

	/** @brief �ֲ���ͼ�߳��� */
	class LocalMapping
	{
	public:
		/**
		 * @brief ���캯��
		 * @param[in] pMap          �ֲ���ͼ�ľ���� //?
		 * @param[in] bMonocular    ��ǰϵͳ�Ƿ��ǵ�Ŀ����
		 */
		LocalMapping(Map* pMap, const float bMonocular);

		/** @brief �߳������� */
		void Run();

		/**
		 * @brief ����׷���߳̾��
		 * @param[in] pTracker ׷���߳̾��
		 */
		void SetTracker(Tracking* pTracker);

		/**
		 * @brief ����ؼ�֡,���ⲿ�̵߳���
		 * @details ���ؼ�֡���뵽��ͼ�У��Ա㽫�����оֲ���ͼ�Ż� \n
		 * NOTICE ��������ǽ��ؼ�֡���뵽�б��н��еȴ�
		 * @param pKF KeyFrame
		 */
		void InsertKeyFrame(KeyFrame* pKF);

		/** @brief �ⲿ�̵߳���,����ֹͣ��ǰ�̵߳Ĺ��� */
		void RequestStop();

		/** @brief ����ǰ�̸߳�λ,���ⲿ�̵߳���,������ */
		void RequestReset();

		/**
		 * @brief ����Ƿ�Ҫ�ѵ�ǰ�ľֲ���ͼ�߳�ֹͣ,�����ǰ�߳�û����ô��������־,��������־����λ��ô������Ϊֹͣ����.��run��������
		 * @return true
		 * @return false
		 */
		bool Stop();

		/** @brief �ͷŵ�ǰ���ڻ������еĹؼ�ָ֡��  */
		void Release();

		/** @brief ���mbStopped�Ƿ���λ�� */
		bool isStopped();

		/** @brief �Ƿ�����ֹ��ǰ�̵߳����� */
		bool stopRequested();

		/** @brief �鿴��ǰ�Ƿ�������ܹؼ�֡ */
		bool AcceptKeyFrames();

		/**
		 * @brief ����"������ܹؼ�֡"��״̬��־
		 * @param[in] flag �ǻ��߷�
		 */
		void SetAcceptKeyFrames(bool flag);

		/** @brief ���� mbnotStop��־��״̬ */
		bool SetNotStop(bool flag);

		/** @brief �ⲿ�̵߳���,��ֹBA */
		void InterruptBA();

		/** @brief ������ֹ��ǰ�߳� */

		void RequestFinish();
		/** @brief ��ǰ�̵߳�run�����Ƿ��Ѿ���ֹ */

		bool isFinished();

		//�鿴�����еȴ�����Ĺؼ�֡��Ŀ
		int KeyframesInQueue() {
			std::unique_lock<std::mutex> lock(mMutexNewKFs);
			return mlNewKeyFrames.size();
		}
	private:
		/**
		 * @brief �鿴�б����Ƿ��еȴ�������Ĺؼ�֡
		 * @return ������ڣ�����true
		 */
		bool CheckNewKeyFrames();

		/**
		 * @brief �����б��еĹؼ�֡
		 *
		 * - ����Bow���������ǻ��µ�MapPoints
		 * - ������ǰ�ؼ�֡��MapPoints��������MapPoints��ƽ���۲ⷽ��͹۲���뷶Χ
		 * - ����ؼ�֡������Covisibilityͼ��Essentialͼ
		 * @see VI-A keyframe insertion
		 */
		void ProcessNewKeyFrame();

		/** @brief ����˶������к͹��ӳ̶ȱȽϸߵĹؼ�֡ͨ�����ǻ��ָ���һЩMapPoints */
		void CreateNewMapPoints();

		/**
		 * @brief �޳�ProcessNewKeyFrame��CreateNewMapPoints������������������õ�MapPoints
		 * @see VI-B recent map points culling
		 */
		void MapPointCulling();

		/** @brief ��鲢�ںϵ�ǰ�ؼ�֡������֡���������ڣ��ظ���MapPoints */
		void SearchInNeighbors();

		/**
		 * @brief �ؼ�֡�޳�
		 * @detials ��Covisibility Graph�еĹؼ�֡����90%���ϵ�MapPoints�ܱ������ؼ�֡������3�����۲⵽������Ϊ�ùؼ�֡Ϊ����ؼ�֡��
		 * @see VI-E Local Keyframe Culling
		 */
		void KeyFrameCulling();

		/**
		 * �������ؼ�֡����̬���������ؼ�֮֡��Ļ�������
		 * @param  pKF1 �ؼ�֡1
		 * @param  pKF2 �ؼ�֡2
		 * @return      ��������
		 */
		cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

		/**
		 * @brief ������ά����v�ķ��Գƾ���
		 * @param[in] v     ��ά����
		 * @return cv::Mat  ���Գƾ���
		 */
		cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

		/** @brief ��鵱ǰ�Ƿ��и�λ�̵߳����� */
		void ResetIfRequested();

		/** @brief ����Ƿ��Ѿ����ⲿ�߳�������ֹ��ǰ�߳� */
		bool CheckFinish();

		/** @brief ���õ�ǰ�߳��Ѿ������ؽ�����,�ɱ��߳�run�������� */
		void SetFinish();

		// Tracking�߳���LocalMapping�в���ؼ�֡���Ȳ��뵽�ö�����
		std::list<KeyFrame*> mlNewKeyFrames; ///< �ȴ�����Ĺؼ�֡�б�
		// ��ǰ���ڴ���Ĺؼ�֡
		KeyFrame* mpCurrentKeyFrame;
		// �洢��ǰ�ؼ�֡���ɵĵ�ͼ��,Ҳ�ǵȴ����ĵ�ͼ���б�
		std::list<MapPoint*> mlpRecentAddedMapPoints;
		// ����ֹ�߳���صĻ�����
		std::mutex mMutexStop;
		// �����ؼ�֡�б�ʱʹ�õĻ����� 
		std::mutex mMutexNewKFs;

	private:
		// ��ǰϵͳ��������Ŀ����˫ĿRGB-D�ı�־
		bool mbMonocular;
		// ��ǰϵͳ�Ƿ��յ�������λ���ź�
		bool mbResetRequested;
		// ��ǰ�߳��Ƿ��յ���������ֹ���ź�
		bool mbFinishRequested;
		// ��ǰ�̵߳��������Ƿ��Ѿ���ֹ
		bool mbFinished;
		// ��ֹBA�ı�־
		bool mbAbortBA;
		// ��ǰ�߳��Ƿ��Ѿ���������ֹ��
		bool mbStopped;
		// ��ֹ��ǰ�̵߳�����
		bool mbStopRequested;
		// ��־�⵱ǰ�̻߳����ܹ�ֹͣ����,���ȼ����Ǹ�"mbStopRequested"Ҫ��.ֻ�������mbStopRequested������Ҫ���ʱ��,�̲߳Ż����һϵ�е���ֹ����
		bool mbNotStop;
		// ��ǰ�ֲ���ͼ�߳��Ƿ�����ؼ�֡����
		bool mbAcceptKeyFrames;

		float fx, fy, cx, cy, invfx, invfy;

		// ��"�߳���������"�йصĻ�����
		std::mutex mMutexFinish;
		// �͸�λ�ź��йصĻ�����
		std::mutex mMutexReset;
		// �Ͳ���������������йصĻ�����
		std::mutex mMutexAccept;

		// ָ��ֲ���ͼ�ľ��
		Map* mpMap;
		// ׷���߳̾��
		Tracking* mpTracker;
	};
}