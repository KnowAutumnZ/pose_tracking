#include "MapPoint.h"

namespace PoseTracking
{
	long unsigned int MapPoint::nNextId = 0;
	std::mutex MapPoint::mGlobalMutex;

	/**
	 * @brief Construct a new Map Point:: Map Point object
	 *
	 * @param[in] Pos           MapPoint�����꣨��������ϵ��
	 * @param[in] pRefKF        �ؼ�֡
	 * @param[in] pMap          ��ͼ
	 */
	MapPoint::MapPoint(const cv::Mat &Pos,  //��ͼ�����������
		KeyFrame *pRefKF,					//���ɵ�ͼ��Ĺؼ�֡
		Map* pMap) :						//��ͼ�������ڵĵ�ͼ
		mnFirstKFid(pRefKF->mnId),              //��һ�ι۲�/�������Ĺؼ�֡ id
		mnFirstFrame(pRefKF->mnFrameId),        //�����õ�ͼ���֡ID(��Ϊ�ؼ�֡Ҳ��֡��)
		nObs(0),                                //���۲����
		mnTrackReferenceForFrame(0),            //���ñ��ظ���ӵ��ֲ���ͼ��ı��
		mnLastFrameSeen(0),                     //�Ƿ�����ж���ĳ��֡��Ұ�еı���
		mnBALocalForKF(0),                      //
		mnFuseCandidateForKF(0),                //
		mnLoopPointForKF(0),                    //
		mnCorrectedByKF(0),                     //
		mnCorrectedReference(0),                //
		mnBAGlobalForKF(0),                     //
		mpRefKF(pRefKF),                        //
		mnVisible(1),                           //��֡�еĿ��Ӵ���
		mnFound(1),                             //���ҵ��Ĵ��� ����������Ҫ���ܹ�ƥ����
		mbBad(false),                           //������
		mpReplaced(static_cast<MapPoint*>(NULL)), //�滻����ǰ��ͼ��ĵ�
		mfMinDistance(0),                       //��ǰ��ͼ����ĳ֡��,�������ı��ҵ�ʱ�䵽�ؼ�֡���ľ�����½�
		mfMaxDistance(0),                       //�Ͻ�
		mpMap(pMap)                             //������ͼ
	{
		Pos.copyTo(mWorldPos);
		//ƽ���۲ⷽ���ʼ��Ϊ0
		mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

		// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
		std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
		mnId = nNextId++;
	}

	/*
	 * @brief ����������frame����MapPoint
	 *
	 * ˫Ŀ��UpdateLastFrame()
	 * @param Pos    MapPoint�����꣨��������ϵ��
	 * @param pMap   Map
	 * @param pFrame Frame
	 * @param idxF   MapPoint��Frame�е�����������Ӧ��������ı��
	 */
	MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF) :
		mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
		mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
		mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
		mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
	{
		Pos.copyTo(mWorldPos);
		cv::Mat Ow = pFrame->GetCameraCenter();
		mNormalVector = mWorldPos - Ow;// ��������ϵ�������3D������� (��ǰ�ؼ�֡�Ĺ۲ⷽ��)
		mNormalVector = mNormalVector / cv::norm(mNormalVector);// ��λ��

		//��������˰�
		cv::Mat PC = Pos - Ow;
		const float dist = cv::norm(PC);    //������ľ���
		const int level = pFrame->mvKeys[idxF].octave;
		const float levelScaleFactor = pFrame->mvScaleFactors[level];
		const int nLevels = pFrame->mnScaleLevels;

		// ��� PredictScale ����ǰ��ע��
		/* 666,��Ϊ����ȡ�������ʱ��, ���ǵ���ͼ��ĳ߶�����,����ڲ�ͬͼ������ȡ�õ���������,��Ӧ����������������Զ��
		   ��ͬ, �������������ɵ�ͼ���ʱ��,ҲҪ�ٶ������ȷ��
		   ��Ȼ�����ò���ÿ��ͼ��֮��ȷ���ĳ߶���Ϣ,�������������ű��������Ե���Ϣ��
		*/
		mfMaxDistance = dist * levelScaleFactor;                              //��ǰͼ���"���"
		mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];    //����������һ��ͼ���"���""

		// �� mDescriptor ��MapPoint.h�е�ע�� ==> ��ʵ���ǻ�ȡ�����ͼ���������
		pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

		// MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
		// TODO ��̫��,��ô����ͻ��? 
		std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
		mnId = nNextId++;
	}

	//��ȡ��ͼ������������ϵ�µ�����
	cv::Mat MapPoint::GetWorldPos()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mWorldPos.clone();
	}

	//���õ�ͼ������������ϵ�µ�����
	void MapPoint::SetWorldPos(const cv::Mat &Pos)
	{
		//TODO Ϊʲô������˸��߳���
		std::unique_lock<std::mutex> lock2(mGlobalMutex);
		std::unique_lock<std::mutex> lock(mMutexPos);
		Pos.copyTo(mWorldPos);
	}

	//��������ϵ�µ�ͼ�㱻�������۲��ƽ���۲ⷽ��
	cv::Mat MapPoint::GetNormal()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return mNormalVector.clone();
	}

	//��ȡ��ͼ��Ĳο��ؼ�֡
	KeyFrame* MapPoint::GetReferenceKeyFrame()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mpRefKF;
	}

	// �ܹ��۲⵽��ǰ��ͼ������йؼ�֡���õ�ͼ����KF�е�����
	std::map<KeyFrame*, size_t> MapPoint::GetObservations()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mObservations;
	}

	/**
	 * @brief ���õ�ͼ���Ƿ��ڹؼ�֡�У��ж�Ӧ�Ķ�ά�����㣩
	 *
	 * @param[in] pKF       �ؼ�֡
	 * @return true         ����ܹ��۲⵽������true
	 * @return false        ����۲ⲻ��������false
	 */
	bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		// ���ڷ���true�������ڷ���false
		// std::map.count �÷�����http://www.cplusplus.com/reference/map/map/count/
		return (mObservations.count(pKF));
	}

	//��ȡ��ǰ��ͼ����ĳ���ؼ�֡�Ĺ۲��У���Ӧ���������ID
	int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		if (mObservations.count(pKF))
			return mObservations[pKF];
		else
			return -1;
	}

	/**
	 * @brief ����ͼ����ӹ۲�
	 *
	 * ��¼��Щ KeyFrame ���Ǹ��������ܹ۲⵽�� ��ͼ��
	 * �����ӹ۲�������ĿnObs����Ŀ+1��˫Ŀ����rgbd+2
	 * ��������ǽ����ؼ�֡���ӹ�ϵ�ĺ��ĺ������ܹ�ͬ�۲⵽ĳЩ��ͼ��Ĺؼ�֡�ǹ��ӹؼ�֡
	 * @param pKF KeyFrame
	 * @param idx MapPoint��KeyFrame�е�����
	 */
	void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		// mObservations:�۲⵽��MapPoint�Ĺؼ�֡KF�͸�MapPoint��KF�е�����
		// ����Ѿ���ӹ��۲⣬����
		if (mObservations.count(pKF))
			return;
		// ���û����ӹ��۲⣬��¼���ܹ۲⵽��MapPoint��KF�͸�MapPoint��KF�е�����
		mObservations[pKF] = idx;

		//if (pKF->mvuRight[idx] >= 0)
		//	nObs += 2; // ˫Ŀ����rgbd
		//else
			nObs++; // ��Ŀ
	}

	// ���۲⵽�������Ŀ����Ŀ+1��˫Ŀ��RGB-D��+2
	int MapPoint::Observations()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return nObs;
	}

	/**
	 * @brief �����ͼ����ߴ����Ե�������
	 *
	 * ����һ����ͼ��ᱻ�������۲⵽������ڲ���ؼ�֡����Ҫ�ж��Ƿ���´���ǰ���������
	 * �Ȼ�õ�ǰ������������ӣ�Ȼ�����������֮����������룬��õ�������������������Ӧ�þ�����С�ľ�����ֵ
	 */
	void MapPoint::ComputeDistinctiveDescriptors()
	{
		// Retrieve all observed descriptors
		std::vector<cv::Mat> vDescriptors;
		std::map<KeyFrame*, size_t> observations;

		// Step 1 ��ȡ�õ�ͼ��������Ч�Ĺ۲�ؼ�֡��Ϣ
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
			// mit->firstȡ�۲⵽�õ�ͼ��Ĺؼ�֡
			// mit->secondȡ�õ�ͼ���ڹؼ�֡�е�����
			KeyFrame* pKF = mit.first;

			if (!pKF->isBad())
			{
				// ȡ��Ӧ������������                                               
				vDescriptors.push_back(pKF->mDescriptors.row(mit.second));
			}
		}

		if (vDescriptors.empty())
			return;

		// Step 3 ������Щ����������֮��ľ���
		// N��ʾΪһ�����ٸ�������
		const size_t N = vDescriptors.size();

		// ��Distances������һ���ԳƵľ���
		// float Distances[N][N];
		std::vector<std::vector<float> > Distances;
		Distances.resize(N, std::vector<float>(N, 0));
		for (size_t i = 0; i < N; i++)
		{
			// ���Լ��ľ��뵱Ȼ��0
			Distances[i][i] = 0;
			// ���㲢��¼��ͬ�����Ӿ���
			for (size_t j = i + 1; j < N; j++)
			{
				int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
				Distances[i][j] = distij;
				Distances[j][i] = distij;
			}
		}

		// Step 4 ѡ�����д����Ե������ӣ���������������Ӧ�þ�����С�ľ�����ֵ
		int BestMedian = INT_MAX;   // ��¼��С����ֵ
		int BestIdx = 0;            // ��С��ֵ��Ӧ������

		for (size_t i = 0; i < N; i++)
		{
			// ��i�������ӵ���������������֮��ľ���
			// vector<int> vDists(Distances[i],Distances[i]+N);
			std::vector<int> vDists(Distances[i].begin(), Distances[i].end());
			std::sort(vDists.begin(), vDists.end());

			// �����ֵ
			int median = vDists[0.5*(N - 1)];

			// Ѱ����С����ֵ
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

	// ��ȡ��ǰ��ͼ���������
	cv::Mat MapPoint::GetDescriptor()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mDescriptor.clone();
	}

	/**
	 * @brief ���µ�ͼ���ƽ���۲ⷽ�򡢹۲���뷶Χ
	 *
	 */
	void MapPoint::UpdateNormalAndDepth()
	{
		// Step 1 ��ù۲⵽�õ�ͼ������йؼ�֡���������Ϣ
		std::map<KeyFrame*, size_t> observations;
		KeyFrame* pRefKF;
		cv::Mat Pos;
		{
			std::unique_lock<std::mutex> lock1(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexPos);
			if (mbBad)
				return;

			observations = mObservations; // ��ù۲⵽�õ�ͼ������йؼ�֡
			pRefKF = mpRefKF;             // �۲⵽�õ�Ĳο��ؼ�֡����һ�δ���ʱ�Ĺؼ�֡��
			Pos = mWorldPos.clone();      // ��ͼ������������ϵ�е�λ��
		}

		if (observations.empty())
			return;

		// Step 2 ����õ�ͼ���ƽ���۲ⷽ��
		// �ܹ۲⵽�õ�ͼ������йؼ�֡���Ըõ�Ĺ۲ⷽ���һ��Ϊ��λ������Ȼ�������͵õ��õ�ͼ��ĳ���
		// ��ʼֵΪ0�������ۼ�Ϊ��һ������������������n
		cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
		int n = 0;

		for (auto& mit : observations)
		{
			KeyFrame* pKF = mit.first;
			cv::Mat Owi = pKF->GetCameraCenter();
			// ��õ�ͼ��͹۲⵽���ؼ�֡����������һ��
			cv::Mat normali = mWorldPos - Owi;
			normal = normal + normali / cv::norm(normali);
			n++;
		}

		cv::Mat PC = Pos - pRefKF->GetCameraCenter();                           // �ο��ؼ�֡���ָ���ͼ�������������������ϵ�µı�ʾ��
		const float dist = cv::norm(PC);                                        // �õ㵽�ο��ؼ�֡����ľ���
		const int level = pRefKF->mvKeys[observations[pRefKF]].octave;          // �۲⵽�õ�ͼ��ĵ�ǰ֡���������ڽ������ĵڼ���
		const float levelScaleFactor = pRefKF->mvScaleFactors[level];           // ��ǰ���������Ӧ�ĳ߶����ӣ�scale^n��scale=1.2��nΪ����
		const int nLevels = pRefKF->mnScaleLevels;                              // �������ܲ�����Ĭ��Ϊ8

		{
			std::unique_lock<std::mutex> lock3(mMutexPos);
			// ʹ�÷�����PredictScale����ǰ��ע��
			mfMaxDistance = dist * levelScaleFactor;                              // �۲⵽�õ�ľ�������
			mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];  // �۲⵽�õ�ľ�������
			mNormalVector = normal / n;                                           // ��õ�ͼ��ƽ���Ĺ۲ⷽ��
		}
	}

	float MapPoint::GetMinDistanceInvariance()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return 0.8f*mfMinDistance;
	}

	float MapPoint::GetMaxDistanceInvariance()
	{
		std::unique_lock<std::mutex> lock(mMutexPos);
		return 1.2f*mfMaxDistance;
	}

	/**
	 * @brief Increase Visible
	 *
	 * Visible��ʾ��
	 * 1. ��MapPoint��ĳЩ֡����Ұ��Χ�ڣ�ͨ��Frame::isInFrustum()�����ж�
	 * 2. ��MapPoint����Щ֡�۲⵽��������һ���ܺ���Щ֡��������ƥ����
	 *    ���磺��һ��MapPoint����ΪM������ĳһ֡F����Ұ��Χ�ڣ�
	 *    �����������õ�M���Ժ�F��һ֡��ĳ����������ƥ����
	 * TODO  ����˵��found ���Ǳ�ʾƥ�������
	 */
	void MapPoint::IncreaseVisible(int n)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mnVisible += n;
	}

	/**
	 * @brief Increase Found
	 *
	 * ���ҵ��õ��֡��+n��nĬ��Ϊ1
	 * @see Tracking::TrackLocalMap()
	 */
	void MapPoint::IncreaseFound(int n)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mnFound += n;
	}

	// ���㱻�ҵ��ı���
	float MapPoint::GetFoundRatio()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return static_cast<float>(mnFound) / mnVisible;
	}

	// û�о��� MapPointCulling ����MapPoints, ��Ϊ�ǻ����ĵ�
	bool MapPoint::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexPos);
		return mbBad;
	}

	// ɾ��ĳ���ؼ�֡�Ե�ǰ��ͼ��Ĺ۲�
	void MapPoint::EraseObservation(KeyFrame* pKF)
	{
		bool bBad = false;
		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			// �������Ҫɾ���Ĺ۲�,���ݵ�Ŀ��˫Ŀ���͵Ĳ�ͬ������ɾ����ǰ��ͼ��ı��۲����
			if (mObservations.count(pKF))
			{
				int idx = mObservations[pKF];

				nObs--;
				mObservations.erase(pKF);

				// �����keyFrame�ǲο�֡����Frame��ɾ��������ָ��RefFrame
				if (mpRefKF == pKF)
					mpRefKF = mObservations.begin()->first;

				// If only 2 observations or less, discard point
				// ���۲⵽�õ�������Ŀ����2ʱ�������õ�
				if (nObs <= 2)
					bBad = true;
			}
		}

		if (bBad)
			// ��֪���Թ۲⵽��MapPoint��Frame����MapPoint�ѱ�ɾ��
			SetBadFlag();
	}

	/**
	 * @brief ��֪���Թ۲⵽��MapPoint��Frame����MapPoint�ѱ�ɾ��
	 *
	 */
	void MapPoint::SetBadFlag()
	{
		std::map<KeyFrame*, size_t> obs;
		{
			std::unique_lock<std::mutex> lock1(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexPos);
			mbBad = true;
			// ��mObservationsת�浽obs��obs��mObservations������ָ�룬��ֵ����Ϊǳ����
			obs = mObservations;
			// ��mObservationsָ����ڴ��ͷţ�obs��Ϊ�ֲ�����֮���Զ�ɾ��
			mObservations.clear();
		}
		for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			KeyFrame* pKF = mit->first;
			// ���߿��Թ۲⵽��MapPoint��KeyFrame����MapPoint��ɾ��
			pKF->EraseMapPointMatch(mit->second);
		}
		// ������MapPoint������ڴ�
		mpMap->EraseMapPoint(this);
	}

	// ��ͼ�к��ߵĴ�С��ʾ��ͬͼ��ͼ���ϵ�һ�����ر�ʾ����ʵ����ռ��еĴ�С
	//              ____
	// Nearer      /____\     level:n-1 --> dmin
	//            /______\                       d/dmin = 1.2^(n-1-m)
	//           /________\   level:m   --> d
	//          /__________\                     dmax/d = 1.2^m
	// Farther /____________\ level:0   --> dmax
	//
	//           log(dmax/d)
	// m = ceil(------------)
	//            log(1.2)
	// �������������:
	// �ڽ���ͶӰƥ���ʱ�������������������Χ,���ǵ����ڲ�ͬ�߶�(Ҳ���Ǿ������Զ��,λ��ͼ��������в�ͬͼ��)���������ܵ������ת��Ӱ�첻ͬ,
	// ��˻�ϣ������������ĵ��������Χ����һ��,���������Զ�ĵ��������Χ��Сһ��,����Ҫ������,���ݵ㵽�ؼ�֡/֡�ľ������������ڵ�ǰ�Ĺؼ�֡/֡��,
	// ���Ŵ����ĸ��߶�
	/**
	 * @brief Ԥ���ͼ���Ӧ���������ڵ�ͼ��������߶Ȳ���
	 *
	 * @param[in] currentDist   ������ľ����ͼ�����
	 * @param[in] pKF           �ؼ�֡
	 * @return int              Ԥ��Ľ������߶�
	 */
	int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
	{
		float ratio;
		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			// mfMaxDistance = ref_dist*levelScaleFactor Ϊ�ο�֡�����ϳ߶Ⱥ�ľ���
			// ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
			ratio = mfMaxDistance / currentDist;
		}

		// ȡ����
		int nScale = std::ceil(log(ratio) / pKF->mfLogScaleFactor);
		if (nScale < 0)
			nScale = 0;
		else if (nScale >= pKF->mnScaleLevels)
			nScale = pKF->mnScaleLevels - 1;

		return nScale;
	}

	/**
	 * @brief ���ݵ�ͼ�㵽���ĵľ�����Ԥ��һ�����������������ĳ߶�
	 *
	 * @param[in] currentDist       ��ͼ�㵽���ĵľ���
	 * @param[in] pF                ��ǰ֡
	 * @return int                  �߶�
	 */
	int MapPoint::PredictScale(const float &currentDist, Frame* pF)
	{
		float ratio;
		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			ratio = mfMaxDistance / currentDist;
		}

		int nScale = std::ceil(log(ratio) / pF->mfLogScaleFactor);
		if (nScale < 0)
			nScale = 0;
		else if (nScale >= pF->mnScaleLevels)
			nScale = pF->mnScaleLevels - 1;

		return nScale;
	}

	/**
	 * @brief �滻��ͼ�㣬���¹۲��ϵ
	 *
	 * @param[in] pMP       �øõ�ͼ�����滻��ǰ��ͼ��
	 */
	void MapPoint::Replace(MapPoint* pMP)
	{
		// ͬһ����ͼ��������
		if (pMP->mnId == this->mnId)
			return;

		//Ҫ�滻��ǰ��ͼ��,����������:
		// 1. ����ǰ��ͼ��Ĺ۲����ݵ��������ݶ�"����"���µĵ�ͼ����
		// 2. ���۲⵽��ǰ��ͼ��Ĺؼ�֡����Ϣ���и���

		// �����ǰ��ͼ�����Ϣ����һ�κ�SetBadFlag������ͬ
		int nvisible, nfound;
		std::map<KeyFrame*, size_t> obs;
		{
			std::unique_lock<std::mutex> lock1(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexPos);
			obs = mObservations;
			//�����ǰ��ͼ���ԭ�й۲�
			mObservations.clear();
			//��ǰ�ĵ�ͼ�㱻ɾ����
			mbBad = true;
			//�ݴ浱ǰ��ͼ��Ŀ��Ӵ����ͱ��ҵ��Ĵ���
			nvisible = mnVisible;
			nfound = mnFound;
			//ָ����ǰ��ͼ���Ѿ���ָ���ĵ�ͼ���滻��
			mpReplaced = pMP;
		}

		// �����ܹ۲⵽ԭ��ͼ��Ĺؼ�֡��Ҫ���Ƶ��滻�ĵ�ͼ����
		//- ���۲⵽��ǰ��ͼ�ĵĹؼ�֡����Ϣ���и���
		for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			KeyFrame* pKF = mit->first;
			if (!pMP->IsInKeyFrame(pKF))
			{
				// �ùؼ�֡��û�ж�"Ҫ�滻����ͼ��ĵ�ͼ��"�Ĺ۲�
				pKF->ReplaceMapPointMatch(mit->second, pMP);	// ��KeyFrame��pMP�滻��ԭ����MapPoint
				pMP->AddObservation(pKF, mit->second);			// ��MapPoint�滻����Ӧ��KeyFrame
			}
			else
			{
				// ����ؼ�֡�Ե�ǰ�ĵ�ͼ���"Ҫ�滻����ͼ��ĵ�ͼ��"�����й۲�
				// ������ͻ����pKF��������������a,b����������������������ǽ�����ͬ�ģ����������������Ӧ���� MapPoint Ϊthis,pMP
				// Ȼ����fuse�Ĺ�����pMP�Ĺ۲���࣬��Ҫ�滻this����˱���b��pMP����ϵ��ȥ��a��this����ϵ
				//˵����,��Ȼ���öԷ����Ǹ���ͼ�������浱ǰ�ĵ�ͼ��,����˵���Է�����,����ɾ������ؼ�֡�Ե�ǰ֡�Ĺ۲�
				pKF->EraseMapPointMatch(mit->second);
			}
		}

		//- ����ǰ��ͼ��Ĺ۲����ݵ��������ݶ�"����"���µĵ�ͼ����
		pMP->IncreaseFound(nfound);
		pMP->IncreaseVisible(nvisible);
		//�����Ӹ���
		pMP->ComputeDistinctiveDescriptors();

		//��֪��ͼ,ɾ����
		mpMap->EraseMapPoint(this);
	}
}