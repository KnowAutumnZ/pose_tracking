#include "KeyFrame.h"

namespace PoseTracking
{
	// 下一个关键帧的id
	long unsigned int KeyFrame::nNextId = 0;

	//关键帧的构造函数
	KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB) :mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mnFrameId(F.mnId),
		mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv), mvKeys(F.mvKeys), mDescriptors(F.mDescriptors.clone()),
		mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), 
		mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), mvpMapPoints(F.mvpMapPoints), mbNotErase(false),mbToBeErased(false), mbBad(false)
	{
		// 获取id
		mnId = nNextId++;

		// 根据指定的普通帧, 初始化用于加速匹配的网格对象信息; 其实就把每个网格中有的特征点的索引复制过来
		mGrid.resize(mnGridCols);
		for (int i = 0; i < mnGridCols; i++)
		{
			mGrid[i].resize(mnGridRows);
			for (int j = 0; j < mnGridRows; j++)
				mGrid[i][j] = F.mGrid[i][j];
		}

		// 设置当前关键帧的位姿
		SetPose(F.mTcw);
	}

	// 设置当前关键帧的位姿
	void KeyFrame::SetPose(const cv::Mat &Tcw_)
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		Tcw_.copyTo(Tcw);
		cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
		cv::Mat Rwc = Rcw.t();
		// 和普通帧中进行的操作相同
		Ow = -Rwc * tcw;

		// 计算当前位姿的逆
		Twc = cv::Mat::eye(4, 4, Tcw.type());
		Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
		Ow.copyTo(Twc.rowRange(0, 3).col(3));

		// center为相机坐标系（左目）下，立体相机中心的坐标
		// 立体相机中心点坐标与左目相机坐标之间只是在x轴上相差mHalfBaseline,
		// 因此可以看出，立体相机中两个摄像头的连线为x轴，正方向为左目相机指向右目相机 (齐次坐标)
		cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
		// 世界坐标系下，左目相机中心到立体相机中心的向量，方向由左目相机指向立体相机中心
		Cw = Twc * center;
	}

	// 获取位姿
	cv::Mat KeyFrame::GetPose()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.clone();
	}

	// 获取位姿的逆
	cv::Mat KeyFrame::GetPoseInverse()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Twc.clone();
	}

	// 获取双目相机的中心,这个只有在可视化的时候才会用到
	cv::Mat KeyFrame::GetStereoCenter()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Cw.clone();
	}

	// 获取姿态
	cv::Mat KeyFrame::GetRotation()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.rowRange(0, 3).colRange(0, 3).clone();
	}

	// 获取位置
	cv::Mat KeyFrame::GetTranslation()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.rowRange(0, 3).col(3).clone();
	}

	// 获取(左目)相机的中心在世界坐标系下的坐标
	cv::Mat KeyFrame::GetCameraCenter()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Ow.clone();
	}

	// Add MapPoint to KeyFrame
	void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mvpMapPoints[idx] = pMP;
	}

	/**
	 * @brief 由于其他的原因,导致当前关键帧观测到的某个地图点被删除(bad==true)了,将该地图点置为NULL
	 *
	 * @param[in] idx   地图点在该关键帧中的id
	 */
	void KeyFrame::EraseMapPointMatch(const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		// NOTE 使用这种方式表示其中的某个地图点被删除
		mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
	}

	// 返回当前关键帧是否已经完蛋了
	bool KeyFrame::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mbBad;
	}

	// Compute Scene Depth (q=2 median). Used in monocular. 评估当前关键帧场景深度，q=2表示中值. 只是在单目情况下才会使用
	// 其实过程就是对当前关键帧下所有地图点的深度进行从小到大排序,返回距离头部其中1/q处的深度值作为当前场景的平均深度
	float KeyFrame::ComputeSceneMedianDepth(const int q)
	{
		cv::Mat Tcw_;
		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexPose);
			Tcw_ = Tcw.clone();
		}

		std::vector<float> vDepths;
		cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
		Rcw2 = Rcw2.t();
		float zcw = Tcw_.at<float>(2, 3);
		// 遍历每一个地图点,计算并保存其在当前关键帧下的深度
		for (int i = 0; i < mvpMapPoints.size(); i++)
		{
			if (mvpMapPoints[i])
			{
				MapPoint* pMP = mvpMapPoints[i];
				cv::Mat x3Dw = pMP->GetWorldPos();
				float z = Rcw2.dot(x3Dw) + zcw; // (R*x3Dw+t)的第三行，即z
				vDepths.push_back(z);
			}
		}

		sort(vDepths.begin(), vDepths.end());
		return vDepths[(vDepths.size() - 1) / q];
	}

	// 获取当前关键帧的具体的地图点
	std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints;
	}



}