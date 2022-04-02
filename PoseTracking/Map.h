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
		/** @brief 构造函数 */
		Map();

		/**
		 * @brief 向地图中添加关键帧
		 *
		 * @param[in] pKF 关键帧
		 */
		void AddKeyFrame(KeyFrame* pKF);

		/**
		 * @brief 从地图中删除关键帧
		 * @detials 实际上这个函数中目前仅仅是删除了在std::set中保存的地图点的指针,并且删除后
		 * 之前的地图点所占用的内存其实并没有得到释放
		 * @param[in] pKF 关键帧
		 */
		void EraseKeyFrame(KeyFrame* pKF);

		/**
		 * @brief 向地图中添加地图点
		 *
		 * @param[in] pMP 地图点
		 */
		void AddMapPoint(MapPoint* pMP);

		/**
		 * @brief 从地图中擦除地图点
		 *
		 * @param[in] pMP 地图点
		 */
		void EraseMapPoint(MapPoint* pMP);

		/**
		 * @brief 获取地图中的所有地图点
		 *
		 * @return std::vector<MapPoint*> 获得的地图点序列
		 */
		std::vector<MapPoint*> GetAllMapPoints();

		/**
		 * @brief 获取地图中的所有关键帧
		 *
		 * @return std::vector<KeyFrame*> 获得的关键帧序列
		 */
		std::vector<KeyFrame*> GetAllKeyFrames();

		/**
		 * @brief 设置参考地图点
		 * @detials 一般是指,设置当前帧中的参考地图点; 这些点将用于DrawMapPoints函数画图
		 *
		 * @param[in] vpMPs 地图点们
		 */
		void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

		/**
		 * @brief 获取地图中的所有参考地图点
		 *
		 * @return std::vector<MapPoint*> 获得的参考地图点序列
		 */
		std::vector<MapPoint*> GetReferenceMapPoints();

		/**
		 * @brief 获得当前地图中的地图点个数
		 *
		 * @return long unsigned int 个数
		 */
		long unsigned int MapPointsInMap();

		/**
		 * @brief 获取当前地图中的关键帧个数
		 *
		 * @return long unsigned 关键帧个数
		 */
		long unsigned  KeyFramesInMap();

		 //为了避免地图点id冲突设计的互斥量
		std::mutex mMutexPointCreation;

	public:
		// 存储所有的地图点
		std::set<MapPoint*> mspMapPoints;

		//参考地图点
		std::vector<MapPoint*> mvpReferenceMapPoints;

		// 存储所有的关键帧
		std::set<KeyFrame*> mspKeyFrames;

		// 保存了最初始的关键帧
		std::vector<KeyFrame*> mvpKeyFrameOrigins;

		//当前地图中具有最大ID的关键帧
		long unsigned int mnMaxKFid;

		//类的成员函数在对类成员变量进行操作的时候,防止冲突的互斥量
		std::mutex mMutexMap;

		//当更新地图时的互斥量.回环检测中和局部BA后更新全局地图的时候会用到这个
		std::mutex mMutexMapUpdate;
	};
}