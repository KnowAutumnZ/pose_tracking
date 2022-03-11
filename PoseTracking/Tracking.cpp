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
			bool bOK;
			if (mState == OK)
			{
				// Step 2.2 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
				// 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
				// 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
				// mnLastRelocFrameId 上一次重定位的那一帧
				if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
				{
					// 用最近的关键帧来跟踪当前的普通帧
					// 通过BoW的方式在参考帧中找当前帧特征点的匹配点
					// 优化每个特征点都对应3D点重投影误差即可得到位姿
					bOK = TrackReferenceKeyFrame();
				}
				else
				{
					// 用最近的普通帧来跟踪当前的普通帧
					// 根据恒速模型设定当前帧的初始位姿
					// 通过投影的方式在参考帧中找当前帧特征点的匹配点
					// 优化每个特征点所对应3D点的投影误差即可得到位姿
					bOK = TrackWithMotionModel();
					if (!bOK)
						//根据恒速模型失败了，只能根据参考关键帧来跟踪
						bOK = TrackReferenceKeyFrame();
				}
			}

			// 将最新的关键帧作为当前帧的参考关键帧
			mCurrentFrame.mpReferenceKF = mpReferenceKF;

			// Step 3：在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
			// 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
			if (!mbOnlyTracking)
			{
				if (bOK)
					bOK = TrackLocalMap();
			}




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

	/*
	 * @brief 用参考关键帧的地图点来对当前普通帧进行跟踪
	 *
	 * Step 1：将当前普通帧的描述子转化为BoW向量
	 * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
	 * Step 3: 将上一帧的位姿态作为当前帧位姿的初始值
	 * Step 4: 通过优化3D-2D的重投影误差来获得位姿
	 * Step 5：剔除优化后的匹配点中的外点
	 * @return 如果匹配数超10，返回true
	 *
	 */
	bool Tracking::TrackReferenceKeyFrame()
	{
		ORBmatcher matcher(0.7, true);
		std::vector<MapPoint*> vpMapPointMatches;

		int nmatches = matcher.SearchForRefModel(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
		// 匹配数目小于15，认为跟踪失败
		if (nmatches < 15)
			return false;

		// Step 3:将上一帧的位姿态作为当前帧位姿的初始值
		mCurrentFrame.mvpMapPoints = vpMapPointMatches;
		mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

		// Step 4:通过优化3D-2D的重投影误差来获得位姿
		Optimizer::PoseOptimization(this, &mCurrentFrame);

		// Step 5：剔除优化后的匹配点中的外点
		//之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
		int nmatchesMap = 0;
		for (int i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++)
		{
			if (mCurrentFrame.mvpMapPoints[i])
			{
				//如果对应到的某个特征点是外点c
				if (mCurrentFrame.mvbOutlier[i])
				{
					//清除它在当前帧中存在过的痕迹
					MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
					pMP->mbTrackInView = false;
					pMP->mnLastFrameSeen = mCurrentFrame.mnId;
					nmatches--;
				}
				else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
					nmatchesMap++;
			}
		}
		// 跟踪成功的数目超过10才认为跟踪成功，否则跟踪失败
		return nmatchesMap >= 10;
	}

	/**
	 * @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪
	 * Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
	 * Step 2：根据上一帧特征点对应地图点进行投影匹配
	 * Step 3：优化当前帧位姿
	 * Step 4：剔除地图点中外点
	 * @return 如果匹配数大于10，认为跟踪成功，返回true
	 */
	bool Tracking::TrackWithMotionModel()
	{
		// 最小距离 < 0.9*次小距离 匹配成功，检查旋转
		ORBmatcher matcher(0.7, true);

		// Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
		UpdateLastFrame();

		// Step 2：根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
		mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

		// 清空当前帧的地图点
		fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

		// 设置特征匹配过程中的搜索半径
		int th;
		if (mSensor != STEREO)
			th = 15;//单目
		else
			th = 7;//双目

		// Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
		int nmatches = matcher.SearchByProjection(this, mCurrentFrame, mLastFrame, th, mSensor == MONOCULAR);

		// 如果匹配点太少，则扩大搜索半径再来一次
		if (nmatches < 20)
		{
			fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
			nmatches = matcher.SearchByProjection(this, mCurrentFrame, mLastFrame, 2 * th, mSensor == MONOCULAR); // 2*th
		}

		// 如果还是不能够获得足够的匹配点,那么就认为跟踪失败
		if (nmatches < 20)
			return false;

		// Step 4：利用3D-2D投影关系，优化当前帧位姿
		Optimizer::PoseOptimization(this, &mCurrentFrame);

		// Step 5：剔除地图点中外点
		int nmatchesMap = 0;
		for (int i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++)
		{
			if (mCurrentFrame.mvpMapPoints[i])
			{
				if (mCurrentFrame.mvbOutlier[i])
				{
					// 如果优化后判断某个地图点是外点，清除它的所有关系
					MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

					mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
					pMP->mbTrackInView = false;
					pMP->mnLastFrameSeen = mCurrentFrame.mnId;
					nmatches--;
				}
				else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
					// 累加成功匹配到的地图点数目
					nmatchesMap++;
			}
		}
		// Step 6：匹配超过10个点就认为跟踪成功
		return nmatchesMap >= 10;
	}

	/**
	 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
	 *
	 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
	 * 可以通过深度值产生一些新的MapPoints
	 */
	void Tracking::UpdateLastFrame()
	{
		// Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
		// 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
		KeyFrame* pRef = mLastFrame.mpReferenceKF;

		// ref_keyframe 到 lastframe的位姿变换
		cv::Mat Tlr = mlRelativeFramePoses.back();

		// 将上一帧的世界坐标系下的位姿计算出来
		// l:last, r:reference, w:world
		// Tlw = Tlr*Trw 
		mLastFrame.SetPose(Tlr*pRef->GetPose());

		// 如果上一帧为关键帧，或者单目的情况，则退出
		if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == MONOCULAR)
			return;
	}

	/**
	 * @brief 用局部地图进行跟踪，进一步优化位姿
	 *
	 * 1. 更新局部地图，包括局部关键帧和关键点
	 * 2. 对局部MapPoints进行投影匹配
	 * 3. 根据匹配对估计当前帧的姿态
	 * 4. 根据姿态剔除误匹配
	 * @return true if success
	 *
	 * Step 1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
	 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
	 * Step 3：更新局部所有MapPoints后对位姿再次优化
	 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
	 * Step 5：决定是否跟踪成功
	 */
	bool Tracking::TrackLocalMap()
	{
		// Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
		UpdateLocalMap();

		// Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
		SearchLocalPoints();








	}

	/**
	 * @brief 更新LocalMap
	 *
	 * 局部地图包括：
	 * 1、K1个关键帧、K2个临近关键帧和参考关键帧
	 * 2、由这些关键帧观测到的MapPoints
	 */
	void Tracking::UpdateLocalMap()
	{
		// This is for visualization
		// 设置参考地图点用于绘图显示局部地图点（红色）
		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		// Update
		// 用共视图来更新局部关键帧和局部地图点,但这个函数每次都要遍历共视关键帧,当关键帧很多时，程序执行很慢
		UpdateLocalKeyFrames();
		UpdateLocalPoints();
	}

	/**
	 * @brief 跟踪局部地图函数里，更新局部关键帧
	 * 方法是遍历当前帧的地图点，将观测到这些地图点的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
	 * Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
	 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧包括以下3种类型
	 *      类型1：能观测到当前帧地图点的关键帧，也称一级共视关键帧
	 *      类型2：一级共视关键帧的共视关键帧，称为二级共视关键帧
	 *      类型3：一级共视关键帧的子关键帧、父关键帧
	 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
	 */
	void Tracking::UpdateLocalKeyFrames()
	{
		// Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
		std::map<KeyFrame*, int> keyframeCounter;
		for (int i = 0; i < mCurrentFrame.mvKeys.size(); i++)
		{
			if (mCurrentFrame.mvpMapPoints[i])
			{
				MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
				if (!pMP->isBad())
				{
					// 得到观测到该地图点的关键帧和该地图点在关键帧中的索引
					const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

					// 由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都对观测到这个地图点的关键帧进行累计投票
					// 这里的操作非常精彩！
					// map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。如果键不存在，则添加一组键值对
					// it->first 是地图点看到的关键帧，同一个关键帧看到的地图点会累加到该关键帧计数
					// 所以最后keyframeCounter 第一个参数表示某个关键帧，第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
					for (map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
						keyframeCounter[it->first]++;
				}
				else
					mCurrentFrame.mvpMapPoints[i] = NULL;
			}
		}

		// 没有当前帧没有共视关键帧，返回
		if (keyframeCounter.empty())
			return;

		// 存储具有最多观测次数（max）的关键帧
		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

		// Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
		// 先清空局部关键帧
		mvpLocalKeyFrames.clear();
		// 先申请3倍内存，不够后面再加
		mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

		// Step 2.1 类型1：能观测到当前帧地图点的关键帧作为局部关键帧 （将邻居拉拢入伙）（一级共视关键帧） 
		for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			// 如果设定为要删除的，跳过
			if (pKF->isBad())
				continue;

			// 寻找具有最大观测数目的关键帧
			if (it->second > max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			// 添加到局部关键帧的列表里
			mvpLocalKeyFrames.push_back(it->first);

			// 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
			// 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
			pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
		}

		// Step 2.2 遍历一级共视关键帧，寻找更多的局部关键帧 
		for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
		{
			// 处理的局部关键帧不超过80帧
			if (mvpLocalKeyFrames.size() > 80)
				break;

			KeyFrame* pKF = *itKF;

			// 类型2:一级共视关键帧的共视（前10个）关键帧，称为二级共视关键帧（将邻居的邻居拉拢入伙）
			// 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧
			const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
		}






	}






}