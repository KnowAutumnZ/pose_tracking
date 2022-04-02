#include "Map.h"

namespace PoseTracking
{
	//构造函数,地图点中最大关键帧id归0
	Map::Map() :mnMaxKFid(0)
	{
	}

	/*
	 * @brief Insert KeyFrame in the map
	 * @param pKF KeyFrame
	 */
	 //在地图中插入关键帧,同时更新关键帧的最大id
	void Map::AddKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mspKeyFrames.insert(pKF);
		if (pKF->mnId > mnMaxKFid)
			mnMaxKFid = pKF->mnId;
	}

	/**
	 * @brief Erase KeyFrame from the map
	 * @param pKF KeyFrame
	 */
	void Map::EraseKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mspKeyFrames.erase(pKF);
	}

	/*
	 * @brief Insert MapPoint in the map
	 * @param pMP MapPoint
	 */
	 //向地图中插入地图点
	void Map::AddMapPoint(MapPoint *pMP)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mspMapPoints.insert(pMP);
	}

	//获取地图中的所有地图点
	std::vector<MapPoint*> Map::GetAllMapPoints()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
	}

	//获取地图中的所有关键帧
	std::vector<KeyFrame*> Map::GetAllKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
	}

	/*
	 * @brief 设置参考MapPoints，将用于DrawMapPoints函数画图
	 * @param vpMPs Local MapPoints
	 */
	 // 设置参考地图点用于绘图显示局部地图点（红色）
	void Map::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mvpReferenceMapPoints = vpMPs;
	}

	//获取参考地图点
	std::vector<MapPoint*> Map::GetReferenceMapPoints()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mvpReferenceMapPoints;
	}

	//获取地图点数目
	long unsigned int Map::MapPointsInMap()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mspMapPoints.size();
	}

	/**
	 * @brief 从地图中删除地图点,但是其实这个地图点所占用的内存空间并没有被释放
	 *
	 * @param[in] pMP
	 */
	void Map::EraseMapPoint(MapPoint *pMP)
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		mspMapPoints.erase(pMP);

		//下面是作者加入的注释. 实际上只是从std::set中删除了地图点的指针, 原先地图点
		//占用的内存区域并没有得到释放
		// TODO: This only erase the pointer.
		// Delete the MapPoint
	}

	//获取地图中的关键帧数目
	long unsigned int Map::KeyFramesInMap()
	{
		std::unique_lock<std::mutex> lock(mMutexMap);
		return mspKeyFrames.size();
	}

}