#include "MapPoint.h"

namespace PoseTracking
{
	long unsigned int MapPoint::nNextId = 0;
	std::mutex MapPoint::mGlobalMutex;

	/**
	 * @brief Construct a new Map Point:: Map Point object
	 *
	 * @param[in] Pos           MapPoint的坐标（世界坐标系）
	 * @param[in] pRefKF        关键帧
	 * @param[in] pMap          地图
	 */
	MapPoint::MapPoint(const cv::Mat &Pos,  //地图点的世界坐标
		KeyFrame *pRefKF,					//生成地图点的关键帧
		Map* pMap) :						//地图点所存在的地图
		mnFirstKFid(pRefKF->mnId),              //第一次观测/生成它的关键帧 id
		mnFirstFrame(pRefKF->mnFrameId),        //创建该地图点的帧ID(因为关键帧也是帧啊)
		nObs(0),                                //被观测次数
		mnTrackReferenceForFrame(0),            //放置被重复添加到局部地图点的标记
		mnLastFrameSeen(0),                     //是否决定判断在某个帧视野中的变量
		mnBALocalForKF(0),                      //
		mnFuseCandidateForKF(0),                //
		mnLoopPointForKF(0),                    //
		mnCorrectedByKF(0),                     //
		mnCorrectedReference(0),                //
		mnBAGlobalForKF(0),                     //
		mpRefKF(pRefKF),                        //
		mnVisible(1),                           //在帧中的可视次数
		mnFound(1),                             //被找到的次数 和上面的相比要求能够匹配上
		mbBad(false),                           //坏点标记
		mpReplaced(static_cast<MapPoint*>(NULL)), //替换掉当前地图点的点
		mfMinDistance(0),                       //当前地图点在某帧下,可信赖的被找到时其到关键帧光心距离的下界
		mfMaxDistance(0),                       //上界
		mpMap(pMap)                             //从属地图
	{
		Pos.copyTo(mWorldPos);
		//平均观测方向初始化为0
		mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

		// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
		std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
		mnId = nNextId++;
	}

	/*
	 * @brief 给定坐标与frame构造MapPoint
	 *
	 * 双目：UpdateLastFrame()
	 * @param Pos    MapPoint的坐标（世界坐标系）
	 * @param pMap   Map
	 * @param pFrame Frame
	 * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
	 */
	MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF) :
		mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
		mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
		mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
		mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
	{
		Pos.copyTo(mWorldPos);
		cv::Mat Ow = pFrame->GetCameraCenter();
		mNormalVector = mWorldPos - Ow;// 世界坐标系下相机到3D点的向量 (当前关键帧的观测方向)
		mNormalVector = mNormalVector / cv::norm(mNormalVector);// 单位化

		//这个算重了吧
		cv::Mat PC = Pos - Ow;
		const float dist = cv::norm(PC);    //到相机的距离
		const int level = pFrame->mvKeys[idxF].octave;
		const float levelScaleFactor = pFrame->mvScaleFactors[level];
		const int nLevels = pFrame->mnScaleLevels;

		// 另见 PredictScale 函数前的注释
		/* 666,因为在提取特征点的时候, 考虑到了图像的尺度问题,因此在不同图层上提取得到的特征点,对应着特征点距离相机的远近
		   不同, 所以在这里生成地图点的时候,也要再对其进行确认
		   虽然我们拿不到每个图层之间确定的尺度信息,但是我们有缩放比例这个相对的信息哇
		*/
		mfMaxDistance = dist * levelScaleFactor;                              //当前图层的"深度"
		mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];    //该特征点上一个图层的"深度""

		// 见 mDescriptor 在MapPoint.h中的注释 ==> 其实就是获取这个地图点的描述子
		pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

		// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
		// TODO 不太懂,怎么个冲突法? 
		std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
		mnId = nNextId++;
	}

	//获取地图点在世界坐标系下的坐标
	cv::Mat MapPoint::GetWorldPos()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mWorldPos.clone();
	}

	//设置地图点在世界坐标系下的坐标
	void MapPoint::SetWorldPos(const cv::Mat &Pos)
	{
		//TODO 为什么这里多了个线程锁
		std::unique_lock<std::mutex> lock2(mGlobalMutex);
		std::unique_lock<std::mutex> lock(mMutexPos);
		Pos.copyTo(mWorldPos);
	}

	/**
	 * @brief 给地图点添加观测
	 *
	 * 记录哪些 KeyFrame 的那个特征点能观测到该 地图点
	 * 并增加观测的相机数目nObs，单目+1，双目或者rgbd+2
	 * 这个函数是建立关键帧共视关系的核心函数，能共同观测到某些地图点的关键帧是共视关键帧
	 * @param pKF KeyFrame
	 * @param idx MapPoint在KeyFrame中的索引
	 */
	void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		// mObservations:观测到该MapPoint的关键帧KF和该MapPoint在KF中的索引
		// 如果已经添加过观测，返回
		if (mObservations.count(pKF))
			return;
		// 如果没有添加过观测，记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
		mObservations[pKF] = idx;

		//if (pKF->mvuRight[idx] >= 0)
		//	nObs += 2; // 双目或者rgbd
		//else
			nObs++; // 单目
	}

	// 被观测到的相机数目，单目+1，双目或RGB-D则+2
	int MapPoint::Observations()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return nObs;
	}

	/**
	 * @brief 计算地图点最具代表性的描述子
	 *
	 * 由于一个地图点会被许多相机观测到，因此在插入关键帧后，需要判断是否更新代表当前点的描述子
	 * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
	 */
	void MapPoint::ComputeDistinctiveDescriptors()
	{
		// Retrieve all observed descriptors
		std::vector<cv::Mat> vDescriptors;
		std::map<KeyFrame*, size_t> observations;

		// Step 1 获取该地图点所有有效的观测关键帧信息
		{
			std::unique_lock<std::mutex> lock1(mMutexFeatures);
			if (mbBad)
				return;
			observations = mObservations;
		}

		if (observations.empty())
			return;

		vDescriptors.reserve(observations.size());

		for (auto& mit: observations)
		{
			// mit->first取观测到该地图点的关键帧
			// mit->second取该地图点在关键帧中的索引
			KeyFrame* pKF = mit.first;

			if (!pKF->isBad())
			{
				// 取对应的描述子向量                                               
				vDescriptors.push_back(pKF->mDescriptors.row(mit.second));
			}
		}

		if (vDescriptors.empty())
			return;

		// Step 3 计算这些描述子两两之间的距离
		// N表示为一共多少个描述子
		const size_t N = vDescriptors.size();

		// 将Distances表述成一个对称的矩阵
		// float Distances[N][N];
		std::vector<std::vector<float> > Distances;
		Distances.resize(N, std::vector<float>(N, 0));
		for (size_t i = 0; i < N; i++)
		{
			// 和自己的距离当然是0
			Distances[i][i] = 0;
			// 计算并记录不同描述子距离
			for (size_t j = i + 1; j < N; j++)
			{
				int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
				Distances[i][j] = distij;
				Distances[j][i] = distij;
			}
		}

		// Step 4 选择最有代表性的描述子，它与其他描述子应该具有最小的距离中值
		int BestMedian = INT_MAX;   // 记录最小的中值
		int BestIdx = 0;            // 最小中值对应的索引

		for (size_t i = 0; i < N; i++)
		{
			// 第i个描述子到其它所有描述子之间的距离
			// vector<int> vDists(Distances[i],Distances[i]+N);
			std::vector<int> vDists(Distances[i].begin(), Distances[i].end());
			std::sort(vDists.begin(), vDists.end());

			// 获得中值
			int median = vDists[0.5*(N - 1)];

			// 寻找最小的中值
			if (median < BestMedian)
			{
				BestMedian = median;
				BestIdx = i;
			}
		}

		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			mDescriptor = vDescriptors[BestIdx].clone();
		}
	}

	/**
	 * @brief 更新地图点的平均观测方向、观测距离范围
	 *
	 */
	void MapPoint::UpdateNormalAndDepth()
	{
		// Step 1 获得观测到该地图点的所有关键帧、坐标等信息
		std::map<KeyFrame*, size_t> observations;
		KeyFrame* pRefKF;
		cv::Mat Pos;
		{
			std::unique_lock<std::mutex> lock1(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexPos);
			if (mbBad)
				return;

			observations = mObservations; // 获得观测到该地图点的所有关键帧
			pRefKF = mpRefKF;             // 观测到该点的参考关键帧（第一次创建时的关键帧）
			Pos = mWorldPos.clone();      // 地图点在世界坐标系中的位置
		}

		if (observations.empty())
			return;

		// Step 2 计算该地图点的平均观测方向
		// 能观测到该地图点的所有关键帧，对该点的观测方向归一化为单位向量，然后进行求和得到该地图点的朝向
		// 初始值为0向量，累加为归一化向量，最后除以总数n
		cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
		int n = 0;

		for (auto& mit : observations)
		{
			KeyFrame* pKF = mit.first;
			cv::Mat Owi = pKF->GetCameraCenter();
			// 获得地图点和观测到它关键帧的向量并归一化
			cv::Mat normali = mWorldPos - Owi;
			normal = normal + normali / cv::norm(normali);
			n++;
		}

		cv::Mat PC = Pos - pRefKF->GetCameraCenter();                           // 参考关键帧相机指向地图点的向量（在世界坐标系下的表示）
		const float dist = cv::norm(PC);                                        // 该点到参考关键帧相机的距离
		const int level = pRefKF->mvKeys[observations[pRefKF]].octave;          // 观测到该地图点的当前帧的特征点在金字塔的第几层
		const float levelScaleFactor = pRefKF->mvScaleFactors[level];           // 当前金字塔层对应的尺度因子，scale^n，scale=1.2，n为层数
		const int nLevels = pRefKF->mnScaleLevels;                              // 金字塔总层数，默认为8

		{
			std::unique_lock<std::mutex> lock3(mMutexPos);
			// 使用方法见PredictScale函数前的注释
			mfMaxDistance = dist * levelScaleFactor;                              // 观测到该点的距离上限
			mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];  // 观测到该点的距离下限
			mNormalVector = normal / n;                                           // 获得地图点平均的观测方向
		}
	}

	// 没有经过 MapPointCulling 检测的MapPoints, 认为是坏掉的点
	bool MapPoint::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexPos);
		return mbBad;
	}

}