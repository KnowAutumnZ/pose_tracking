#include "Tracking.h"

namespace PoseTracking
{
	Tracking::Tracking(const std::string &strSettingPath, FrameDrawer *pFrameDrawer, Map* pMap, MapDrawer* pMapDrawer, eSensor sensor):
		mSensor(sensor), mbOnlyTracking(false), mState(NOT_INITIALIZED),
		mpFrameDrawer(pFrameDrawer), mpMap(pMap), mpMapDrawer(pMapDrawer)
	{
		std::string TrackingCFG = strSettingPath + "TrackingCFG1.ini";

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
			mpIniORBextractor = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		mpORBextractorLeft = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		float fx = config.ReadFloat("PoseTracking", "fx", 0.0);
		float fy = config.ReadFloat("PoseTracking", "fy", 0.0);
		float cx = config.ReadFloat("PoseTracking", "cx", 0.0);
		float cy = config.ReadFloat("PoseTracking", "cy", 0.0);

		//必须给初值
		mK = cv::Mat::zeros(3, 3, CV_32F);
		mK.at<float>(0, 0) = fx;
		mK.at<float>(0, 2) = cx;
		mK.at<float>(1, 1) = fy;
		mK.at<float>(1, 2) = cy;
		mK.at<float>(2, 2) = 1.0;

		float k1 = config.ReadFloat("PoseTracking", "k1", 0.0);
		float k2 = config.ReadFloat("PoseTracking", "k2", 0.0);
		float p1 = config.ReadFloat("PoseTracking", "p1", 0.0);
		float p2 = config.ReadFloat("PoseTracking", "p2", 0.0);
		float k3 = config.ReadFloat("PoseTracking", "k3", 0.0);

		mDistort = cv::Mat::zeros(5, 1, CV_32F);
		mDistort.at<float>(0, 0) = k1;
		mDistort.at<float>(1, 0) = k2;
		mDistort.at<float>(2, 0) = p1;
		mDistort.at<float>(3, 0) = p2;
		mDistort.at<float>(4, 0) = k3;
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

		mIm = im;

		cv::Mat imGray;
		if (im.channels() == 3) cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);

		if (mState == NOT_INITIALIZED)
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
			mpInitializer = new Initializer(mK, mCurrentFrame, 1.0, 10);

			// 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
			std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
		}
		else
		{
			// Step 2 如果当前帧特征点数太少（不超过20），则重新构造初始器
			// NOTICE 只有连续两帧的特征点个数都大于20时，才能继续进行初始化过程
			if ((int)mCurrentFrame.mvKeys.size() <= 20)
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

			// Step 5 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
			if (mpInitializer->Initialize(
				mCurrentFrame,      //当前帧
				mvIniMatches,       //当前帧和参考帧的特征点的匹配关系
				Rcw, tcw,           //初始化得到的相机的位姿
				mvIniP3D,           //进行三角化得到的空间点集合
				vbTriangulated))    //以及对应于mvIniMatches来讲,其中哪些点被三角化了
			{
				// Step 6 初始化成功后，删除那些无法进行三角化的匹配点
				for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
				{
					if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
					{
						mvIniMatches[i] = -1;
						nmatches--;
					}
				}

				// Step 7 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
				mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
				// 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
				cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
				Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
				tcw.copyTo(Tcw.rowRange(0, 3).col(3));
				mCurrentFrame.SetPose(Tcw);

				// Step 8 创建初始化地图点MapPoints
				// Initialize函数会得到mvIniP3D，
				// mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
				// CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
				CreateInitialMapMonocular();
			}//当初始化成功的时候进行
		}//如果单目初始化器已经被创建
	}

	/**
	 * @brief 单目相机成功初始化后用三角化得到的点生成MapPoints
	 *
	 */
	void Tracking::CreateInitialMapMonocular()
	{
		// Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
		KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);  // 第一帧
		KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);  // 第二帧
  
		// Step 2 将关键帧插入到地图
		mpMap->AddKeyFrame(pKFini);
		mpMap->AddKeyFrame(pKFcur);

		// Step 3 用初始化得到的3D点来生成地图点MapPoints
		//  mvIniMatches[i] 表示初始化两帧特征点匹配关系。
		//  具体解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值,没有匹配关系的话，vMatches12[i]值为 -1
		for (size_t i = 0; i < mvIniMatches.size(); i++)
		{
			// 没有匹配，跳过
			if (mvIniMatches[i] < 0)
				continue;

			//Create MapPoint.
			// 用三角化点初始化为空间点的世界坐标
			cv::Mat worldPos(mvIniP3D[i]);

			// Step 3.1 用3D点构造MapPoint
			MapPoint* pMP = new MapPoint(
				worldPos,
				pKFcur,
				mpMap);

			// Step 3.2 为该MapPoint添加属性：
			// a.观测到该MapPoint的关键帧
			// b.该MapPoint的描述子
			// c.该MapPoint的平均观测方向和深度范围

			// 表示该KeyFrame的2D特征点和对应的3D地图点
			pKFini->AddMapPoint(pMP, i);
			pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

			// a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
			pMP->AddObservation(pKFini, i);
			pMP->AddObservation(pKFcur, mvIniMatches[i]);

			// b.从众多观测到该MapPoint的特征点中挑选最有代表性的描述子
			pMP->ComputeDistinctiveDescriptors();
			// c.更新该MapPoint平均观测方向以及观测距离的范围
			pMP->UpdateNormalAndDepth();

			//mvIniMatches下标i表示在初始化参考帧中的特征点的序号
			//mvIniMatches[i]是初始化当前帧中的特征点的序号
			mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
			mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

			//Add to Map
			mpMap->AddMapPoint(pMP);
		}

		// Step 5 取场景的中值深度，用于尺度归一化 
		// 为什么是 pKFini 而不是 pKCur ? 答：都可以的，内部做了位姿变换了
		float medianDepth = pKFini->ComputeSceneMedianDepth(2);
		float invMedianDepth = 1.0f / medianDepth;

		// Step 6 将两帧之间的变换归一化到平均深度1的尺度下
		cv::Mat Tc2w = pKFcur->GetPose();
		// x/z y/z 将z归一化到1 
		Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
		pKFcur->SetPose(Tc2w);

		// Step 7 把3D点的尺度也归一化到1
		// 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的? 答：是的，因为是同样的三维点
		vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
		for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
		{
			if (vpAllMapPoints[iMP])
			{
				MapPoint* pMP = vpAllMapPoints[iMP];
				pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
			}
		}

		mCurrentFrame.SetPose(pKFcur->GetPose());
		mnLastKeyFrameId = mCurrentFrame.mnId;
		mpLastKeyFrame = pKFcur;

		// 单目初始化之后，得到的初始地图中的所有点都是局部地图点
		mvpLocalMapPoints = mpMap->GetAllMapPoints();
		mpReferenceKF = pKFcur;

		mCurrentFrame.mpReferenceKF = pKFcur;
		mLastFrame = Frame(mCurrentFrame);

		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

		mpMap->mvpKeyFrameOrigins.push_back(pKFini);

		// 初始化成功，至此，初始化过程完成
		mState = OK;
	}


}