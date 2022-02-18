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

	// û�о��� MapPointCulling ����MapPoints, ��Ϊ�ǻ����ĵ�
	bool MapPoint::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexPos);
		return mbBad;
	}

}