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

	//在System类中用到了Tracking，Tracking类中又用到了System，必须添加这样的前置
	class System;
	class Viewer;
	class FrameDrawer;

	class Tracking
	{
	public:
		/**
		 * @brief 构造函数
		 *
		 * @param[in] pSys              系统实例
		 * @param[in] pVoc              字典指针
		 * @param[in] pFrameDrawer      帧绘制器
		 * @param[in] pMapDrawer        地图绘制器
		 * @param[in] pMap              地图句柄
		 * @param[in] pKFDB             关键帧数据库句柄
		 * @param[in] strSettingPath    配置文件路径
		 * @param[in] sensor            传感器类型
		 */
		Tracking(const std::string &strSettingPath, FrameDrawer *pFrameDrawer, Map* pMap, MapDrawer* pMapDrawer, eSensor sensor);

		/**
		 * @brief 处理单目输入图像
		 *
		 * @param[in] im            图像
		 * @param[in] timestamp     时间戳
		 * @return cv::Mat          世界坐标系到该帧相机坐标系的变换矩阵
		 */
		cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

		/** @brief 主追踪进程 */
		void Track();

		/** @brief 单目输入的时候所进行的初始化操作 */
		void MonocularInitialization();

		/** @brief 单目输入的时候生成初始地图 */
		void CreateInitialMapMonocular();

	public:
		//跟踪状态类型
		enum eTrackingState {
			NOT_INITIALIZED = 1,          //<有图像但是没有完成初始化
			OK = 2,                       //<正常时候的工作状态
			LOST = 3                      //<系统已经跟丢了的状态
		};

		//跟踪状态
		eTrackingState mState;
		//上一帧的跟踪状态.这个变量在绘制当前帧的时候会被使用到
		eTrackingState mLastProcessedState;

		//标记当前系统是处于SLAM状态还是纯定位状态
		bool mbOnlyTracking;

	public:
		// ORB
		// orb特征提取器，不管单目还是双目，mpORBextractorLeft都要用到
		// 如果是双目，则要用到mpORBextractorRight
		// NOTICE 如果是单目，在初始化的时候使用mpIniORBextractor而不是mpORBextractorLeft，
		// mpIniORBextractor属性中提取的特征点个数是mpORBextractorLeft的两倍

		//ORB特征点提取器
		orbDetector* mpORBextractorLeft, *mpORBextractorRight;
		//在初始化的时候使用的特征点提取器,其提取到的特征点个数会更多
		orbDetector* mpIniORBextractor;

		//当前系统运行的时候,关键帧所产生的数据库
		KeyFrameDatabase* mpKeyFrameDB;

		//Map
		//(全局)地图句柄
		Map* mpMap;

		//Drawers  可视化查看器相关
		///查看器对象句柄
		Viewer* mpViewer;
		///帧绘制器句柄
		FrameDrawer* mpFrameDrawer;
		///地图绘制器句柄
		MapDrawer* mpMapDrawer;

		//传感器类型
		eSensor mSensor;

		//内参、畸变
		cv::Mat mK, mDistort;

		//图像
		cv::Mat mIm;

		//初始化过程中的参考帧
		Frame mInitialFrame;
		//追踪线程中有一个当前帧
		Frame mCurrentFrame;

		// 上一关键帧
		KeyFrame* mpLastKeyFrame;
		// 上一帧
		Frame mLastFrame;
		// 上一个关键帧的ID
		unsigned int mnLastKeyFrameId;
		// 上一次重定位的那一帧的ID
		unsigned int mnLastRelocFrameId;

		//单目初始器
		Initializer* mpInitializer{nullptr};

		//Local Map 局部地图相关
		//参考关键帧
		KeyFrame* mpReferenceKF;// 当前关键帧就是参考帧
		//局部关键帧集合
		std::vector<KeyFrame*> mvpLocalKeyFrames;
		//局部地图点的集合
		std::vector<MapPoint*> mvpLocalMapPoints;

		//跟踪初始化时前两帧之间的匹配
		std::vector<int> mvIniMatches;

		//初始化过程中匹配后进行三角化得到的空间点
		std::vector<cv::Point3f> mvIniP3D;
	};
}