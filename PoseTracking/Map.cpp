#include "Map.h"

namespace PoseTracking
{
	//���캯��,��ͼ�������ؼ�֡id��0
	Map::Map() :mnMaxKFid(0)
	{
	}

	/*
	 * @brief Insert KeyFrame in the map
	 * @param pKF KeyFrame
	 */
	 //�ڵ�ͼ�в���ؼ�֡,ͬʱ���¹ؼ�֡�����id
	void Map::AddKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mspKeyFrames.insert(pKF);
		if (pKF->mnId > mnMaxKFid)
			mnMaxKFid = pKF->mnId;
	}

	/*
	 * @brief Insert MapPoint in the map
	 * @param pMP MapPoint
	 */
	 //���ͼ�в����ͼ��
	void Map::AddMapPoint(MapPoint *pMP)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mspMapPoints.insert(pMP);
	}

	//��ȡ��ͼ�е����е�ͼ��
	std::vector<MapPoint*> Map::GetAllMapPoints()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
	}

	//��ȡ��ͼ�е����йؼ�֡
	std::vector<KeyFrame*> Map::GetAllKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
	}

	/*
	 * @brief ���òο�MapPoints��������DrawMapPoints������ͼ
	 * @param vpMPs Local MapPoints
	 */
	 // ���òο���ͼ�����ڻ�ͼ��ʾ�ֲ���ͼ�㣨��ɫ��
	void Map::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mvpReferenceMapPoints = vpMPs;
	}

	//��ȡ�ο���ͼ��
	std::vector<MapPoint*> Map::GetReferenceMapPoints()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mvpReferenceMapPoints;
	}

	//��ȡ��ͼ����Ŀ
	long unsigned int Map::MapPointsInMap()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mspMapPoints.size();
	}

	//��ȡ��ͼ�еĹؼ�֡��Ŀ
	long unsigned int Map::KeyFramesInMap()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mspKeyFrames.size();
	}

}