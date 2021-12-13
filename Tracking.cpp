#include "Tracking.h"

namespace PoseTracking
{
	Tracking::Tracking(const std::string &strSettingPath, eSensor sensor):mSensor(sensor)/*, mState(NO_IMAGES_YET)*/
	{
		std::string TrackingCFG = strSettingPath + "TrackingCFG.ini";

		rr::RrConfig config;
		config.ReadConfig(TrackingCFG);

		int nfeatures = config.ReadInt("PoseTracking", "nfeatures", 500);
		float scaleFactor = config.ReadFloat("PoseTracking", "scaleFactor", 1.2);
		int nlevels = config.ReadInt("PoseTracking", "nlevels", 8);
		int iniThFAST = config.ReadInt("PoseTracking", "iniThFAST", 20);
		int minThFAST = config.ReadInt("PoseTracking", "minThFAST", 12);

		// 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
		if (mSensor == STEREO)
			mpORBextractorRight = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		// 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
		if (mSensor == MONOCULAR)
			mpIniORBextractor = new orbDetector(nfeatures * 2, scaleFactor, nlevels, iniThFAST, minThFAST);

		mpORBextractorLeft = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		float fx = config.ReadFloat("PoseTracking", "fx", 0.0);
		float fy = config.ReadFloat("PoseTracking", "fy", 0.0);
		float cx = config.ReadFloat("PoseTracking", "cx", 0.0);
		float cy = config.ReadFloat("PoseTracking", "cy", 0.0);

		mK = cv::Mat_<float>(3, 3) << (fx, 0, cx, 0, fy, cy, 0, 0, 1);

		float k1 = config.ReadFloat("PoseTracking", "k1", 0.0);
		float k2 = config.ReadFloat("PoseTracking", "k2", 0.0);
		float p1 = config.ReadFloat("PoseTracking", "p1", 0.0);
		float p2 = config.ReadFloat("PoseTracking", "p2", 0.0);
		float k3 = config.ReadFloat("PoseTracking", "k3", 0.0);

		mDistort = cv::Mat_<float>(5, 1) << (k1, k2, p1, p2, k3);
	}

	/**
	 * @brief 处理单目输入图像
	 *
	 * @param[in] im            图像
	 * @param[in] timestamp     时间戳
	 * @return cv::Mat          世界坐标系到该帧相机坐标系的变换矩阵
	 */
	cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
	{
		if (im.empty()) return cv::Mat();

		cv::Mat imGray;
		if (im.channels() == 3) cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);

		if (mState == NO_IMAGES_YET || mState == NOT_INITIALIZED)
			mCurrentFrame = Frame(imGray, timestamp, mpIniORBextractor, mK, mDistort);
		else
			mCurrentFrame = Frame(imGray, timestamp, mpORBextractorLeft, mK, mDistort);

		// Step 3 ：跟踪
		Track();

		return mCurrentFrame.mTcw;
	}

	/** @brief 主追踪进程 */
	void Tracking::Track()
	{
		// 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
		if (mState == NO_IMAGES_YET) mState = NOT_INITIALIZED;

		// mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
		mLastProcessedState = mState;

		if (mState == NOT_INITIALIZED)
		{
			MonocularInitialization();

			//更新帧绘制器中存储的最新状态
			mpFrameDrawer->Update(this);

			//这个状态量在上面的初始化函数中被更新
			if (mState != OK) return;
		}
		else
		{

		}

	}

	/*
	 * @brief 单目的地图初始化
	 *
	 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
	 * 得到初始两帧的匹配、相对运动、初始MapPoints
	 *
	 * Step 1：（未创建）得到用于初始化的第一帧，初始化需要两帧
	 * Step 2：（已创建）如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
	 * Step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
	 * Step 4：如果初始化的两帧之间的匹配点太少，重新初始化
	 * Step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
	 * Step 6：删除那些无法进行三角化的匹配点
	 * Step 7：将三角化得到的3D点包装成MapPoints
	 */
	void Tracking::MonocularInitialization()
	{
		if (!mpInitializer)
		{
			// 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
			mInitialFrame = Frame(mCurrentFrame);

			// 由当前帧构造初始器 sigma:1.0 iterations:200
			mpInitializer = new Initializer(mK, mCurrentFrame, 1.0, 200);

			// 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
			std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
		}
		else
		{
			// Step 2 如果当前帧特征点数太少（不超过100），则重新构造初始器
			// NOTICE 只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
			if ((int)mCurrentFrame.mvKeys.size() <= 100)
			{
				delete mpInitializer;
				mpInitializer = static_cast<Initializer*>(NULL);
				fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
				return;
			}

			// Step 3 在mInitialFrame与mCurrentFrame中找匹配的特征点对
			ORBmatcher matcher(
				0.9,        //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
				true);      //检查特征点的方向

			// 对 mInitialFrame,mCurrentFrame 进行特征点匹配
			// mvbPrevMatched为参考帧的特征点坐标，初始化存储的是mInitialFrame中特征点坐标，匹配后存储的是匹配好的当前帧的特征点坐标
			// mvIniMatches 保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
			int nmatches = matcher.SearchForInitialization(
				mInitialFrame, mCurrentFrame,    //初始化时的参考帧和当前帧
				mvIniMatches,                    //保存匹配关系
				20);                             //搜索窗口大小

			// Step 4 验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
			if (nmatches < 16)
			{
				delete mpInitializer;
				mpInitializer = static_cast<Initializer*>(NULL);
				return;
			}

			cv::Mat Rcw; // Current Camera Rotation
			cv::Mat tcw; // Current Camera Translation
			std::vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)



		}





	}



}