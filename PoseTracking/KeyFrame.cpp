#include "KeyFrame.h"

namespace PoseTracking
{
	// ��һ���ؼ�֡��id
	long unsigned int KeyFrame::nNextId = 0;

	//�ؼ�֡�Ĺ��캯��
	KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB) :mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mnFrameId(F.mnId),
		mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv), mvKeys(F.mvKeys), mDescriptors(F.mDescriptors.clone()),
		mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), 
		mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), mvpMapPoints(F.mvpMapPoints), mbNotErase(false),mbToBeErased(false), mbBad(false)
	{
		// ��ȡid
		mnId = nNextId++;

		// ����ָ������ͨ֡, ��ʼ�����ڼ���ƥ������������Ϣ; ��ʵ�Ͱ�ÿ���������е���������������ƹ���
		mGrid.resize(mnGridCols);
		for (int i = 0; i < mnGridCols; i++)
		{
			mGrid[i].resize(mnGridRows);
			for (int j = 0; j < mnGridRows; j++)
				mGrid[i][j] = F.mGrid[i][j];
		}

		// ���õ�ǰ�ؼ�֡��λ��
		SetPose(F.mTcw);
	}

	// ���õ�ǰ�ؼ�֡��λ��
	void KeyFrame::SetPose(const cv::Mat &Tcw_)
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		Tcw_.copyTo(Tcw);
		cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
		cv::Mat Rwc = Rcw.t();
		// ����ͨ֡�н��еĲ�����ͬ
		Ow = -Rwc * tcw;

		// ���㵱ǰλ�˵���
		Twc = cv::Mat::eye(4, 4, Tcw.type());
		Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
		Ow.copyTo(Twc.rowRange(0, 3).col(3));

		// centerΪ�������ϵ����Ŀ���£�����������ĵ�����
		// ����������ĵ���������Ŀ�������֮��ֻ����x�������mHalfBaseline,
		// ��˿��Կ����������������������ͷ������Ϊx�ᣬ������Ϊ��Ŀ���ָ����Ŀ��� (�������)
		cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
		// ��������ϵ�£���Ŀ������ĵ�����������ĵ���������������Ŀ���ָ�������������
		Cw = Twc * center;
	}

	// ��ȡλ��
	cv::Mat KeyFrame::GetPose()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.clone();
	}

	// ��ȡλ�˵���
	cv::Mat KeyFrame::GetPoseInverse()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Twc.clone();
	}

	// ��ȡ˫Ŀ���������,���ֻ���ڿ��ӻ���ʱ��Ż��õ�
	cv::Mat KeyFrame::GetStereoCenter()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Cw.clone();
	}

	// ��ȡ��̬
	cv::Mat KeyFrame::GetRotation()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.rowRange(0, 3).colRange(0, 3).clone();
	}

	// ��ȡλ��
	cv::Mat KeyFrame::GetTranslation()
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.rowRange(0, 3).col(3).clone();
	}

	// ��ȡ(��Ŀ)�������������������ϵ�µ�����
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
	 * @brief ����������ԭ��,���µ�ǰ�ؼ�֡�۲⵽��ĳ����ͼ�㱻ɾ��(bad==true)��,���õ�ͼ����ΪNULL
	 *
	 * @param[in] idx   ��ͼ���ڸùؼ�֡�е�id
	 */
	void KeyFrame::EraseMapPointMatch(const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		// NOTE ʹ�����ַ�ʽ��ʾ���е�ĳ����ͼ�㱻ɾ��
		mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
	}

	// ���ص�ǰ�ؼ�֡�Ƿ��Ѿ��군��
	bool KeyFrame::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mbBad;
	}

	// Compute Scene Depth (q=2 median). Used in monocular. ������ǰ�ؼ�֡������ȣ�q=2��ʾ��ֵ. ֻ���ڵ�Ŀ����²Ż�ʹ��
	// ��ʵ���̾��ǶԵ�ǰ�ؼ�֡�����е�ͼ�����Ƚ��д�С��������,���ؾ���ͷ������1/q�������ֵ��Ϊ��ǰ������ƽ�����
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
		// ����ÿһ����ͼ��,���㲢�������ڵ�ǰ�ؼ�֡�µ����
		for (int i = 0; i < mvpMapPoints.size(); i++)
		{
			if (mvpMapPoints[i])
			{
				MapPoint* pMP = mvpMapPoints[i];
				cv::Mat x3Dw = pMP->GetWorldPos();
				float z = Rcw2.dot(x3Dw) + zcw; // (R*x3Dw+t)�ĵ����У���z
				vDepths.push_back(z);
			}
		}

		sort(vDepths.begin(), vDepths.end());
		return vDepths[(vDepths.size() - 1) / q];
	}

	// ��ȡ��ǰ�ؼ�֡�ľ���ĵ�ͼ��
	std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints;
	}



}