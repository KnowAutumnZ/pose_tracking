#include "KeyFrame.h"

namespace PoseTracking
{
	// ��һ���ؼ�֡��id
	long unsigned int KeyFrame::nNextId = 0;

	//�ؼ�֡�Ĺ��캯��
	KeyFrame::KeyFrame(Frame* F, Map *pMap, KeyFrameDatabase *pKFDB) :mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS), mnFrameId(F->mnId),
		mfGridElementWidthInv(F->mfGridElementWidthInv), mfGridElementHeightInv(F->mfGridElementHeightInv), mvKeys(F->mvKeys), mDescriptors(F->mDescriptors.clone()),
		mnScaleLevels(F->mnScaleLevels), mfScaleFactor(F->mfScaleFactor),mfLogScaleFactor(F->mfLogScaleFactor), mvScaleFactors(F->mvScaleFactors),
		mnMinX(F->mnMinX), mnMinY(F->mnMinY), mnMaxX(F->mnMaxX), mnMaxY(F->mnMaxY),
		mvLevelSigma2(F->mvLevelSigma2), mvInvLevelSigma2(F->mvInvLevelSigma2), mvpMapPoints(F->mvpMapPoints), mbNotErase(false),mbToBeErased(false), mbBad(false),
		mpMap(pMap)
	{
		// ��ȡid
		mnId = nNextId++;

		// ����ָ������ͨ֡, ��ʼ�����ڼ���ƥ������������Ϣ; ��ʵ�Ͱ�ÿ���������е���������������ƹ���
		mGrid.resize(mnGridCols);
		for (int i = 0; i < mnGridCols; i++)
		{
			mGrid[i].resize(mnGridRows);
			for (int j = 0; j < mnGridRows; j++)
				mGrid[i][j] = F->mGrid[i][j];
		}

		// ���õ�ǰ�ؼ�֡��λ��
		SetPose(F->mTcw);
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

	// ͬ��
	void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
	{
		//��ȡ��ǰ��ͼ����ĳ���ؼ�֡�Ĺ۲��У���Ӧ������������������û�й۲⣬����Ϊ-1
		int idx = pMP->GetIndexInKeyFrame(this);
		if (idx >= 0)
			mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
	}

	// ��ͼ����滻
	void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
	{
		mvpMapPoints[idx] = pMP;
	}

	// ���ص�ǰ�ؼ�֡�Ƿ��Ѿ��군��
	bool KeyFrame::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mbBad;
	}

	// �ж�ĳ�����Ƿ��ڵ�ǰ�ؼ�֡��ͼ����
	bool KeyFrame::IsInImage(const float &x, const float &y) const
	{
		return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
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

	// �ؼ�֡�У����ڵ������ٹ۲���ĿminObs��MapPoints������.��Щ�����㱻��Ϊ׷�ٵ���
	int KeyFrame::TrackedMapPoints(const int &minObs)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);

		int nPoints = 0;
		// �Ƿ�����Ŀ
		const bool bCheckObs = minObs > 0;
		// N�ǵ�ǰ֡��������ĸ���
		for (int i = 0; i < mvKeys.size(); i++)
		{
			MapPoint* pMP = mvpMapPoints[i];
			if (pMP)     //û�б�ɾ��
			{
				if (!pMP->isBad())   //���Ҳ��ǻ���
				{
					if (bCheckObs)
					{
						// ����������ֵҪ��ĵ�ͼ�������1
						if (mvpMapPoints[i]->Observations() >= minObs)
							nPoints++;
					}
					else
						nPoints++;
				}
			}
		}

		return nPoints;
	}

	// ��ȡ��ǰ�ؼ�֡�ľ����ĳ����ͼ��
	MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints[idx];
	}

	// ��ȡ��ǰ�ؼ�֡�ľ���ĵ�ͼ��
	std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints;
	}

	//��ȡ��ǰ�ؼ�֡���ӹؼ�֡
	std::set<KeyFrame*> KeyFrame::GetChilds()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens;
	}

	//��ȡ��ǰ�ؼ�֡�ĸ��ؼ�֡
	KeyFrame* KeyFrame::GetParent()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mpParent;
	}

	// ɾ��ĳ���ӹؼ�֡
	void KeyFrame::EraseChild(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.erase(pKF);
	}

	// �ı䵱ǰ�ؼ�֡�ĸ��ؼ�֡
	void KeyFrame::ChangeParent(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		// ���˫�����ӹ�ϵ
		mpParent = pKF;
		pKF->AddChild(this);
	}

	// �ж�ĳ���ؼ�֡�Ƿ��ǵ�ǰ�ؼ�֡���ӹؼ�֡
	bool KeyFrame::hasChild(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens.count(pKF);
	}

	// �õ���ùؼ�֡���ӵĹؼ�֡(�Ѱ�Ȩֵ����)
	std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mvpOrderedConnectedKeyFrames;
	}

	/**
	 * @brief �õ���ùؼ�֡���ӵ�ǰN����ǿ���ӹؼ�֡(�Ѱ�Ȩֵ����)
	 *
	 * @param[in] N                 �趨Ҫȡ���Ĺؼ�֡��Ŀ
	 * @return vector<KeyFrame*>    ����Ȩ�������Ĺؼ�֡����
	 */
	std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);

		if ((int)mvpOrderedConnectedKeyFrames.size() < N)
			// ��������������ͷ������еĹؼ�֡
			return mvpOrderedConnectedKeyFrames;
		else
			// ȡǰN����ǿ���ӹؼ�֡
			return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
	}

	// �õ��ùؼ�֡��pKF��Ȩ��
	int KeyFrame::GetWeight(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);

		if (mConnectedKeyFrameWeights.count(pKF))
			return mConnectedKeyFrameWeights[pKF];
		else
			// û�����ӵĻ�Ȩ��Ҳ���ǹ��ӵ��������0
			return 0;
	}

	void KeyFrame::AddConnection(KeyFrame* pKF, const int &weight)
	{
		{
			// ����������ֹͬʱ�����������ݲ�����ͻ
			std::unique_lock<mutex> lock(mMutexConnections);

			// �½����������Ȩ��
			if (!mConnectedKeyFrameWeights.count(pKF))
				// count��������0��˵��mConnectedKeyFrameWeights��û��pKF���½�����
				mConnectedKeyFrameWeights[pKF] = weight;
			else if (mConnectedKeyFrameWeights[pKF] != weight)
				// ֮ǰ���ӵ�Ȩ�ز�һ���ˣ���Ҫ����
				mConnectedKeyFrameWeights[pKF] = weight;
			else
				return;
		}

		// ���ӹ�ϵ�仯��Ҫ������ѹ��ӣ���Ҫ�����½�������
		UpdateBestCovisibles();
	}

	/**
	 * @brief ����Ȩ�شӴ�С�����ӣ����ӣ��Ĺؼ�֡��������
	 *
	 * ���º�ı����洢��mvpOrderedConnectedKeyFrames��mvOrderedWeights��
	 */
	void KeyFrame::UpdateBestCovisibles()
	{
		// ����������ֹͬʱ�����������ݲ�����ͻ
		std::unique_lock<mutex> lock(mMutexConnections);
		// http://stackoverflow.com/questions/3389648/difference-between-stdliststdpair-and-stdmap-in-c-stl (std::map �� std::list<std::pair>������)

		std::vector<std::pair<int, KeyFrame*> > vPairs;
		vPairs.reserve(mConnectedKeyFrameWeights.size());
		// ȡ���������ӵĹؼ�֡��mConnectedKeyFrameWeights������Ϊstd::map<KeyFrame*,int>����vPairs���������ӵĵ�ͼ��������ǰ�棬��������
		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			vPairs.push_back(make_pair(mit->second, mit->first));

		// ΪʲôҪ�������棿��Ϊ�����ɾ���������㣬ֻ��Ҫ�޸���һ�ڵ�λ�ã�����Ҫ�ƶ�����Ԫ��
		std::list<KeyFrame*> lKFs;   // �������ӹؼ�֡
		std::list<int> lWs;          // �������ӹؼ�֡��Ӧ��Ȩ�أ����ӵ�ͼ����Ŀ��
		for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
		{
			// push_front ���ɴӴ�С
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		// Ȩ�شӴ�С���е����ӹؼ�֡
		mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
		// �Ӵ�С���е�Ȩ�أ���mvpOrderedConnectedKeyFramesһһ��Ӧ
		mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
	}

	/*
	 * ���¹ؼ�֮֡�������ͼ
	 *
	 * 1. ���Ȼ�øùؼ�֡������MapPoint�㣬ͳ�ƹ۲⵽��Щ3d���ÿ���ؼ�֡���������йؼ�֮֡��Ĺ��ӳ̶�
	 *    ��ÿһ���ҵ��Ĺؼ�֡������һ���ߣ��ߵ�Ȩ���Ǹùؼ�֡�뵱ǰ�ؼ�֡����3d��ĸ�����
	 * 2. ���Ҹ�Ȩ�ر������һ����ֵ�����û�г�������ֵ��Ȩ�أ���ô��ֻ����Ȩ�����ıߣ��������ؼ�֡�Ĺ��ӳ̶ȱȽϸߣ�
	 * 3. ����Щ���Ӱ���Ȩ�شӴ�С���������Է��㽫���Ĵ���
	 *    ������covisibilityͼ֮�����û�г�ʼ���������ʼ��Ϊ����Ȩ�����ıߣ��������ؼ�֡���ӳ̶���ߵ��Ǹ��ؼ�֡�������������������
	 */
	void KeyFrame::UpdateConnections()
	{
		// �ؼ�֡-Ȩ�أ�Ȩ��Ϊ�����ؼ�֡�뵱ǰ�ؼ�֡���ӵ�ͼ��ĸ�����Ҳ��Ϊ���ӳ̶�
		std::map<KeyFrame*, int> KFcounter;
		std::vector<MapPoint*> vpMP;

		{
			// ��øùؼ�֡�����е�ͼ��
			std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
			vpMP = mvpMapPoints;
		}

		// Step 1 ͨ����ͼ�㱻�ؼ�֡�۲������ͳ�ƹؼ�֮֡��Ĺ��ӳ̶�
		// ͳ��ÿһ����ͼ�㶼�ж��ٹؼ�֡�뵱ǰ�ؼ�֡���ڹ��ӹ�ϵ��ͳ�ƽ������KFcounter
		for (std::vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;

			if (!pMP)
				continue;

			if (pMP->isBad())
				continue;

			// ����ÿһ����ͼ�㣬observations��¼�˿��Թ۲⵽�õ�ͼ������йؼ�֡
			std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			// ����һ����ͼ����Ա�����ؼ�֡�۲⵽,��˶���ÿһ�ι۲�,���Թ۲⵽�����ͼ��Ĺؼ�֡�����ۼ�ͶƱ
			// ����Ĳ����ǳ����ʣ�
			// map[key] = value����Ҫ����ļ�����ʱ���Ḳ�Ǽ���Ӧ��ԭ����ֵ������������ڣ������һ���ֵ��
			// it->first �ǵ�ͼ�㿴���Ĺؼ�֡��ͬһ���ؼ�֡�����ĵ�ͼ����ۼӵ��ùؼ�֡����
			// �������KFcounter ��һ��������ʾĳ���ؼ�֡����2��������ʾ�ùؼ�֡�����˶��ٵ�ǰ֡(mCurrentFrame)�ĵ�ͼ�㣬Ҳ���ǹ��ӳ̶�
			for (std::map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
				KFcounter[it->first]++;
		}

		// û�й��ӹ�ϵ��ֱ���˳� 
		if (KFcounter.empty())
			return;

		int nmax = 0; // ��¼��ߵĹ��ӳ̶�
		KeyFrame* pKFmax = NULL;
		// ������15�����ӵ�ͼ��Ż���ӹ��ӹ�ϵ
		int th = 15;

		// vPairs��¼�������ؼ�֡����֡������th�Ĺؼ�֡
		// pair<int,KeyFrame*>���ؼ�֡��Ȩ��д��ǰ�棬�ؼ�֡д�ں��淽���������
		std::vector<std::pair<int, KeyFrame*> > vPairs;
		vPairs.reserve(KFcounter.size());
		// Step 2 �ҵ���ӦȨ�����Ĺؼ�֡�����ӳ̶���ߵĹؼ�֡��
		for (std::map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
		{
			if (mit->second > nmax)
			{
				nmax = mit->second;
				pKFmax = mit->first;
			}

			// �������ӹ�ϵ������Ҫ���ڵ���th�����ӵ�ͼ��
			if (mit->second >= th)
			{
				// ��ӦȨ����Ҫ������ֵ������Щ�ؼ�֡��������
				vPairs.push_back(make_pair(mit->second, mit->first));
				// �Է��ؼ�֡ҲҪ��������Ϣ
				// ����KFcounter�иùؼ�֡��mConnectedKeyFrameWeights
				// ��������KeyFrame��mConnectedKeyFrameWeights�����������ؼ�֡�뵱ǰ֡������Ȩ��
				(mit->first)->AddConnection(this, mit->second);
			}
		}

		//  Step 3 ���û�г�����ֵ��Ȩ�أ����Ȩ�����Ĺؼ�֡��������
		if (vPairs.empty())
		{
			// ���ÿ���ؼ�֡�������ӵĹؼ�֡�ĸ���������th��
			// �Ǿ�ֻ�����������ؼ�֡���ӳ̶���ߵĹؼ�֡��mConnectedKeyFrameWeights
			// ���Ƕ�֮ǰth�����ֵ���ܹ��ߵ�һ������
			vPairs.push_back(make_pair(nmax, pKFmax));
			pKFmax->AddConnection(this, nmax);
		}

		// Step 4 �����㹲�ӳ̶ȵĹؼ�֡�Ը������ӹ�ϵ��Ȩ�أ��Ӵ�С��
		// vPairs���Ķ����໥���ӳ̶ȱȽϸߵĹؼ�֡�͹���Ȩ�أ��������ɴ�С��������
		std::sort(vPairs.begin(), vPairs.end());                // sort����Ĭ����������
		// �������Ľ���ֱ���֯��Ϊ������������
		std::list<KeyFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i < vPairs.size(); i++)
		{
			// push_front �����˴Ӵ�С˳��
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		{
			std::unique_lock<std::mutex> lockCon(mMutexConnections);

			// ���µ�ǰ֡�������ؼ�֡������Ȩ��
			mConnectedKeyFrameWeights = KFcounter;
			mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
			mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

			// Step 5 ����������������
			if (mbFirstConnection && mnId != 0)
			{
				// ��ʼ���ùؼ�֡�ĸ��ؼ�֡Ϊ���ӳ̶���ߵ��Ǹ��ؼ�֡
				mpParent = mvpOrderedConnectedKeyFrames.front();
				// ����˫�����ӹ�ϵ������ǰ�ؼ�֡��Ϊ���ӹؼ�֡
				mpParent->AddChild(this);
				mbFirstConnection = false;
			}
		}
	}

	// ����ӹؼ�֡�������ӹؼ�֡��������ӹ�ϵ�Ĺؼ�֡���ǵ�ǰ�ؼ�֡��
	void KeyFrame::AddChild(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.insert(pKF);
	}

	/**
	 * @brief ������ִ��ɾ���ؼ�֡�Ĳ���
	 * ��Ҫɾ�����Ǹùؼ�֡����������֡����ͼ��֮������ӹ�ϵ
	 *
	 * mbNotErase���ã���ʾҪɾ���ùؼ�֡�������ӹ�ϵ��������ؼ�֡�п������ڻػ������߼���sim3��������ʱ����Ȼ����ؼ�֡���࣬����ȴ����ɾ����
	 * ������mbNotEraseΪtrue����ʱ�����setbadflag����ʱ�����Ὣ����ؼ�֡ɾ����ֻ���mbTobeErase���true����������ؼ�֡����ɾ��������ʱ��,�ȼ������Ժ���
	 * �ڱջ��߳������ SetErase()�����mbToBeErased ��ɾ��֮ǰ����ɾ����ûɾ����֡��
	 */
	void KeyFrame::SetBadFlag()
	{
		// Step 1 ���ȴ���һ��ɾ�����˵��������
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);

			// ��0�ؼ�֡������ɾ��
			if (mnId == 0)
				return;
			else if (mbNotErase)
			{
				// mbNotErase��ʾ��Ӧ��ɾ�������ǰ�mbToBeErased��Ϊtrue����װ�Ѿ�ɾ������ʵû��ɾ��
				mbToBeErased = true;
				return;
			}
		}

		// Step 2 �������к͵�ǰ�ؼ�֡�����Ĺؼ�֡��ɾ�������뵱ǰ�ؼ�֡����ϵ
		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			mit->first->EraseConnection(this); // �������Ĺؼ�֡ɾ�����Լ�����ϵ

		// Step 3 ����ÿһ����ǰ�ؼ�֡�ĵ�ͼ�㣬ɾ��ÿһ����ͼ��͵�ǰ�ؼ�֡����ϵ
		for (size_t i = 0; i < mvpMapPoints.size(); i++)
			if (mvpMapPoints[i])
				mvpMapPoints[i]->EraseObservation(this);

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			std::unique_lock<std::mutex> lock1(mMutexFeatures);

			// ����Լ��������ؼ�֮֡�����ϵ
			mConnectedKeyFrameWeights.clear();
			mvpOrderedConnectedKeyFrames.clear();

			// Step 4 ��������������Ҫ�Ǵ���ø��ӹؼ�֡����Ȼ����������ؼ�֡ά����ͼ���ѣ����߻���
			// ��ѡ���ؼ�֡
			std::set<KeyFrame*> sParentCandidates;
			// ����ǰ֡�ĸ��ؼ�֡�����ѡ���ؼ�֡
			sParentCandidates.insert(mpParent);

			// ÿ����һ�ξ�Ϊ����һ���ӹؼ�֡Ѱ�Ҹ��ؼ�֡����߹��ӳ̶ȣ����ҵ������ӹؼ�֡������Ϊ�����ӹؼ�֡�ĺ�ѡ���ؼ�֡
			while (!mspChildrens.empty())
			{
				bool bContinue = false;

				int max = -1;
				KeyFrame* pC;
				KeyFrame* pP;

				// Step 4.1 ����ÿһ���ӹؼ�֡�������Ǹ�������ָ��ĸ��ؼ�֡
				for (std::set<KeyFrame*>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
				{
					KeyFrame* pKF = *sit;
					// ������Ч���ӹؼ�֡
					if (pKF->isBad())
						continue;

					// Step 4.2 �ӹؼ�֡����ÿһ���������ӵĹؼ�֡    
					std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();

					for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
					{
						// sParentCandidates �иտ�ʼ����������ӹؼ�֡�ġ�үү����Ҳ�ǵ�ǰ�ؼ�֡�ĺ�ѡ���ؼ�֡
						for (set<KeyFrame*>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
						{
							// Step 4.3 ������Ӻ�sParentCandidates���й��ӣ�ѡ������ǿ���Ǹ���Ϊ�µĸ�
							if (vpConnected[i]->mnId == (*spcit)->mnId)
							{
								int w = pKF->GetWeight(vpConnected[i]);
								// Ѱ�Ҳ�����Ȩֵ�����Ǹ����ӹ�ϵ
								if (w > max)
								{
									pC = pKF;                   //�ӹؼ�֡
									pP = vpConnected[i];        //Ŀǰ���ӹؼ�֡�������Ȩֵ�Ĺؼ�֡�������ĸ��ؼ�֡�� 
									max = w;                    //�������Ȩֵ
									bContinue = true;           //˵���ӽڵ��ҵ��˿�����Ϊ���¸��ؼ�֡��֡
								}
							}
						}
					}
				}

				// Step 4.4 ���������Ĺ������ҵ����µĸ��ڵ�
				// �������Ӧ�÷ŵ������ӹؼ�֡ѭ����?
				// �ش𣺲���Ҫ������whileѭ����û�˳�����ʹ�ø��µ�sParentCandidates
				if (bContinue)
				{
					// ��Ϊ���ڵ����ˣ������ӽڵ��ҵ����µĸ��ڵ㣬�Ͱ�������Ϊ�Լ��ĸ��ڵ�
					pC->ChangeParent(pP);
					// ��Ϊ�ӽڵ��ҵ����µĸ��ڵ㲢�����˸��ڵ㣬��ô���ӽڵ���������Ϊ�����ӽڵ�ı�ѡ���ڵ�
					sParentCandidates.insert(pC);
					// ���ӽڵ㴦����ϣ�ɾ��
					mspChildrens.erase(pC);
				}
				else
					break;

			}

			// Step 4.5 ��������ӽڵ�û���ҵ��µĸ��ڵ�
			if (!mspChildrens.empty())
				for (set<KeyFrame*>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
				{
					// ֱ�ӰѸ��ڵ�ĸ��ڵ���Ϊ�Լ��ĸ��ڵ� ��������Щ�ӽڵ���˵,���ǵ��µĸ��ڵ���ʵ�����Լ���үү�ڵ�
					(*sit)->ChangeParent(mpParent);
				}

			mpParent->EraseChild(this);
			// mTcp ��ʾԭ���ؼ�֡����ǰ�ؼ�֡��λ�˱任���ڱ���λ�˵�ʱ��ʹ��
			mTcp = Tcw * mpParent->GetPoseInverse();
			// ��ǵ�ǰ�ؼ�֡�Ѿ�����
			mbBad = true;
		}

		// ��ͼ�͹ؼ�֡���ݿ���ɾ���ùؼ�֡
		mpMap->EraseKeyFrame(this);
	}

	// ɾ����ǰ�ؼ�֡��ָ���ؼ�֮֡��Ĺ��ӹ�ϵ
	void KeyFrame::EraseConnection(KeyFrame* pKF)
	{
		// ��ʵ���Ӧ�ñ�ʾ�Ƿ�������й��ӹ�ϵ
		bool bUpdate = false;

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mConnectedKeyFrameWeights.count(pKF))
			{
				mConnectedKeyFrameWeights.erase(pKF);
				bUpdate = true;
			}
		}

		// ���������й��ӹ�ϵ,��ôɾ��֮���Ҫ���¹��ӹ�ϵ
		if (bUpdate)
			UpdateBestCovisibles();
	}

	// ��ȡĳ��������������е�������id,��ʵ����� Frame.cc �е��Ǹ����������϶���һ�µ�; rΪ�߳����뾶��
	std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
	{
		std::vector<size_t> vIndices;
		vIndices.reserve(mvKeys.size());

		// ����Ҫ������cell�ķ�Χ

		// floor����ȡ����mfGridElementWidthInv Ϊÿ������ռ���ٸ�����
		const int nMinCellX = max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
		if (nMinCellX >= mnGridCols)
			return vIndices;

		// ceil����ȡ��
		const int nMaxCellX = min((int)mnGridCols - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
		if (nMaxCellX < 0)
			return vIndices;

		const int nMinCellY = max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
		if (nMinCellY >= mnGridRows)
			return vIndices;

		const int nMaxCellY = min((int)mnGridRows - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
		if (nMaxCellY < 0)
			return vIndices;

		// ����ÿ��cell,ȡ������ÿ��cell�еĵ�,����ÿ���㶼Ҫ�����Ƿ���������
		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				const vector<size_t> vCell = mGrid[ix][iy];
				for (size_t j = 0, jend = vCell.size(); j < jend; j++)
				{
					const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					if (fabs(distx) < r && fabs(disty) < r)
						vIndices.push_back(vCell[j]);
				}
			}
		}

		return vIndices;
	}

}