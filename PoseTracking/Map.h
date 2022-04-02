#pragma once

#include <set>
#include <mutex>

#include "MapPoint.h"
#include "KeyFrame.h"

namespace PoseTracking
{
	class MapPoint;
	class KeyFrame;

	class Map
	{
	public:
		/** @brief ���캯�� */
		Map();

		/**
		 * @brief ���ͼ����ӹؼ�֡
		 *
		 * @param[in] pKF �ؼ�֡
		 */
		void AddKeyFrame(KeyFrame* pKF);

		/**
		 * @brief �ӵ�ͼ��ɾ���ؼ�֡
		 * @detials ʵ�������������Ŀǰ������ɾ������std::set�б���ĵ�ͼ���ָ��,����ɾ����
		 * ֮ǰ�ĵ�ͼ����ռ�õ��ڴ���ʵ��û�еõ��ͷ�
		 * @param[in] pKF �ؼ�֡
		 */
		void EraseKeyFrame(KeyFrame* pKF);

		/**
		 * @brief ���ͼ����ӵ�ͼ��
		 *
		 * @param[in] pMP ��ͼ��
		 */
		void AddMapPoint(MapPoint* pMP);

		/**
		 * @brief �ӵ�ͼ�в�����ͼ��
		 *
		 * @param[in] pMP ��ͼ��
		 */
		void EraseMapPoint(MapPoint* pMP);

		/**
		 * @brief ��ȡ��ͼ�е����е�ͼ��
		 *
		 * @return std::vector<MapPoint*> ��õĵ�ͼ������
		 */
		std::vector<MapPoint*> GetAllMapPoints();

		/**
		 * @brief ��ȡ��ͼ�е����йؼ�֡
		 *
		 * @return std::vector<KeyFrame*> ��õĹؼ�֡����
		 */
		std::vector<KeyFrame*> GetAllKeyFrames();

		/**
		 * @brief ���òο���ͼ��
		 * @detials һ����ָ,���õ�ǰ֡�еĲο���ͼ��; ��Щ�㽫����DrawMapPoints������ͼ
		 *
		 * @param[in] vpMPs ��ͼ����
		 */
		void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

		/**
		 * @brief ��ȡ��ͼ�е����вο���ͼ��
		 *
		 * @return std::vector<MapPoint*> ��õĲο���ͼ������
		 */
		std::vector<MapPoint*> GetReferenceMapPoints();

		/**
		 * @brief ��õ�ǰ��ͼ�еĵ�ͼ�����
		 *
		 * @return long unsigned int ����
		 */
		long unsigned int MapPointsInMap();

		/**
		 * @brief ��ȡ��ǰ��ͼ�еĹؼ�֡����
		 *
		 * @return long unsigned �ؼ�֡����
		 */
		long unsigned  KeyFramesInMap();

		 //Ϊ�˱����ͼ��id��ͻ��ƵĻ�����
		std::mutex mMutexPointCreation;

	public:
		// �洢���еĵ�ͼ��
		std::set<MapPoint*> mspMapPoints;

		//�ο���ͼ��
		std::vector<MapPoint*> mvpReferenceMapPoints;

		// �洢���еĹؼ�֡
		std::set<KeyFrame*> mspKeyFrames;

		// ���������ʼ�Ĺؼ�֡
		std::vector<KeyFrame*> mvpKeyFrameOrigins;

		//��ǰ��ͼ�о������ID�Ĺؼ�֡
		long unsigned int mnMaxKFid;

		//��ĳ�Ա�����ڶ����Ա�������в�����ʱ��,��ֹ��ͻ�Ļ�����
		std::mutex mMutexMap;

		//�����µ�ͼʱ�Ļ�����.�ػ�����к;ֲ�BA�����ȫ�ֵ�ͼ��ʱ����õ����
		std::mutex mMutexMapUpdate;
	};
}