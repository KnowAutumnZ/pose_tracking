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
	//这个枚举类型用于 表示本系统所使用的传感器类型
	enum eSensor {
		MONOCULAR = 0,
		STEREO = 1,
		RGBD = 2
	};

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
		Tracking(const std::string &strSettingPath, eSensor sensor);


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

	public:
		//跟踪状态类型
		enum eTrackingState {
			NO_IMAGES_YET = 0,            //<当前无图像
			NOT_INITIALIZED = 1,          //<有图像但是没有完成初始化
			OK = 2,                       //<正常时候的工作状态
			LOST = 3                      //<系统已经跟丢了的状态
		};

		//跟踪状态
		eTrackingState mState;
		//上一帧的跟踪状态.这个变量在绘制当前帧的时候会被使用到
		eTrackingState mLastProcessedState;

	private:
		// ORB
		// orb特征提取器，不管单目还是双目，mpORBextractorLeft都要用到
		// 如果是双目，则要用到mpORBextractorRight
		// NOTICE 如果是单目，在初始化的时候使用mpIniORBextractor而不是mpORBextractorLeft，
		// mpIniORBextractor属性中提取的特征点个数是mpORBextractorLeft的两倍

		//ORB特征点提取器
		orbDetector* mpORBextractorLeft, *mpORBextractorRight;
		//在初始化的时候使用的特征点提取器,其提取到的特征点个数会更多
		orbDetector* mpIniORBextractor;

		//传感器类型
		eSensor mSensor;

		//内参、畸变
		cv::Mat mK, mDistort;

		//初始化过程中的参考帧
		Frame mInitialFrame;
		//追踪线程中有一个当前帧
		Frame mCurrentFrame;
		//上一帧
		Frame mLastFrame;

		//单目初始器
		Initializer* mpInitializer{nullptr};

		//跟踪初始化时前两帧之间的匹配
		std::vector<int> mvIniMatches;
	};
}