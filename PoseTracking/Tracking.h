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
	//内参、畸变
	static cv::Mat mK, mDistort;
	static float fx, fy, cx, cy;

	//系统所使用的传感器类型
	enum eSensor;

	//在System类中用到了Tracking，Tracking类中又用到了System，必须添加这样的前置
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

		/**
		 * @brief 设置局部地图句柄
		 *
		 * @param[in] pLocalMapper 局部建图器
		 */
		void SetLocalMapper(LocalMapping* pLocalMapper);

		/**
		 * @brief 设置可视化查看器句柄
		 *
		 * @param[in] pViewer 可视化查看器
		 */
		void SetViewer(Viewer* pViewer);

	public:
		/** @brief 单目输入的时候所进行的初始化操作 */
		void MonocularInitialization();

		/** @brief 单目输入的时候生成初始地图 */
		void CreateInitialMapMonocular();

		/**
		 * @brief 对参考关键帧的MapPoints进行跟踪
		 *
		 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
		 * 2. 对属于同一node的描述子进行匹配
		 * 3. 根据匹配对估计当前帧的姿态
		 * 4. 根据姿态剔除误匹配
		 * @return 如果匹配数大于10，返回true
		 */
		bool TrackReferenceKeyFrame();

		/**
		 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
		 *
		 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
		 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
		 * 3. 根据匹配对估计当前帧的姿态
		 * 4. 根据姿态剔除误匹配
		 * @return 如果匹配数大于10，返回true
		 * @see V-B Initial Pose Estimation From Previous Frame
		 */
		bool TrackWithMotionModel();

		/**
		 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
		 *
		 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
		 * 可以通过深度值产生一些新的MapPoints
		 */
		void UpdateLastFrame();

		/**
		 * @brief 对Local Map的MapPoints进行跟踪
		 * Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
		 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
		 * Step 3：更新局部所有MapPoints后对位姿再次优化
		 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
		 * Step 5：根据跟踪匹配数目及回环情况决定是否跟踪成功
		 * @return true         跟踪成功
		 * @return false        跟踪失败
		 */
		bool TrackLocalMap();

		/**
		 * @brief 更新局部地图 LocalMap
		 *
		 * 局部地图包括：共视关键帧、临近关键帧及其子父关键帧，由这些关键帧观测到的MapPoints
		 */
		void UpdateLocalMap();

		/**
		 * @brief 更新局部地图点（来自局部关键帧）
		 *
		 */
		void UpdateLocalPoints();

		/**
		* @brief 更新局部关键帧
		* 方法是遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
		* Step 1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
		* Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
		* Step 2.1 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧 （将邻居拉拢入伙）
		* Step 2.2 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们的家人和邻居作为局部关键帧
		* Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
		*/
		void UpdateLocalKeyFrames();

		/**
		 * @brief 对 Local MapPoints 进行跟踪
		 *
		 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
		 */
		void SearchLocalPoints();

		/**
		 * @brief 断当前帧是否为关键帧
		 * @return true if needed
		 */
		bool NeedNewKeyFrame();
		/**
		 * @brief 创建新的关键帧
		 *
		 * 对于非单目的情况，同时创建新的MapPoints
		 */
		void CreateNewKeyFrame();
	public:
		//跟踪状态类型
		enum eTrackingState {
			NOT_INITIALIZED = 1,          //<有图像但是没有完成初始化
			OK = 2,                       //<正常时候的工作状态
			LOST = 3                      //<系统已经跟丢了的状态
		};

		//标记当前系统是处于SLAM状态还是纯定位状态
		bool mbOnlyTracking;

		//跟踪状态
		eTrackingState mState;
		//上一帧的跟踪状态.这个变量在绘制当前帧的时候会被使用到
		eTrackingState mLastProcessedState;

		//传感器类型
		eSensor mSensor;

		//图像
		cv::Mat mIm;

		//Motion Model
		cv::Mat mVelocity;

		//局部关键帧集合
		std::vector<KeyFrame*> mvpLocalKeyFrames;
		//局部地图点的集合
		std::vector<MapPoint*> mvpLocalMapPoints;

		//跟踪初始化时前两帧之间的匹配
		std::vector<int> mvIniMatches;

		//初始化过程中匹配后进行三角化得到的空间点
		std::vector<cv::Point3f> mvIniP3D;

		//所有的参考关键帧到当前帧的位姿;看上面注释的意思,这里存储的也是相对位姿
		list<cv::Mat> mlRelativeFramePoses;

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

		//初始化过程中的参考帧
		Frame* mpInitialFrame;
		//追踪线程中有一个当前帧
		Frame* mpCurrentFrame;
		// 上一帧
		Frame* mpLastFrame;

		//单目初始器
		Initializer* mpInitializer{nullptr};

		// 上一关键帧
		KeyFrame* mpLastKeyFrame;

		// 上一个关键帧的ID
		unsigned int mnLastKeyFrameId;
		// 上一次重定位的那一帧的ID
		unsigned int mnLastRelocFrameId;

		//Local Map 局部地图相关
		//参考关键帧
		KeyFrame* mpReferenceKF;// 当前关键帧就是参考帧

	public:
		//当前帧中的进行匹配的内点,将会被不同的函数反复使用
		int mnMatchesInliers;

		// 新建关键帧和重定位中用来判断最小最大时间间隔，和帧率有关
		int mMinFrames;
		int mMaxFrames;

		//参考关键帧
		std::list<KeyFrame*> mlpReferences;
		//所有帧的时间戳
		std::list<double> mlFrameTimes;
		//是否跟丢的标志
		std::list<bool> mlbLost;

		//局部建图器句柄
		LocalMapping* mpLocalMapper;
	};
}