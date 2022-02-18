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

		// �����˫Ŀ��tracking�����л������õ�mpORBextractorRight��Ϊ��Ŀ��������ȡ��
		if (mSensor == STEREO)
			mpORBextractorRight = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		// �ڵ�Ŀ��ʼ����ʱ�򣬻���mpIniORBextractor����Ϊ��������ȡ��
		if (mSensor == MONOCULAR)
			mpIniORBextractor = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		mpORBextractorLeft = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		float fx = config.ReadFloat("PoseTracking", "fx", 0.0);
		float fy = config.ReadFloat("PoseTracking", "fy", 0.0);
		float cx = config.ReadFloat("PoseTracking", "cx", 0.0);
		float cy = config.ReadFloat("PoseTracking", "cy", 0.0);

		//�������ֵ
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
	 * @brief ����Ŀ����ͼ��
	 *
	 * @param[in] im            ͼ��
	 * @param[in] timestamp     ʱ���
	 * @return cv::Mat          ��������ϵ����֡�������ϵ�ı任����
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

		// Step 3 ������
		Track();

		return mCurrentFrame.mTcw;
	}

	/** @brief ��׷�ٽ��� */
	void Tracking::Track()
	{
		// mLastProcessedState �洢��Tracking���µ�״̬������FrameDrawer�еĻ���
		mLastProcessedState = mState;

		if (mState == NOT_INITIALIZED)
		{
			MonocularInitialization();

			//����֡�������д洢������״̬
			mpFrameDrawer->Update(this);

			//���״̬��������ĳ�ʼ�������б�����
			if (mState != OK) return;
		}
		else
		{

		}

	}

	/*
	 * @brief ��Ŀ�ĵ�ͼ��ʼ��
	 *
	 * ���еؼ����������͵�Ӧ�Ծ���ѡȡ����һ��ģ�ͣ��ָ����ʼ��֮֡��������̬�Լ�����
	 * �õ���ʼ��֡��ƥ�䡢����˶�����ʼMapPoints
	 *
	 * Step 1����δ�������õ����ڳ�ʼ���ĵ�һ֡����ʼ����Ҫ��֡
	 * Step 2�����Ѵ����������ǰ֡������������100����õ����ڵ�Ŀ��ʼ���ĵڶ�֡
	 * Step 3����mInitialFrame��mCurrentFrame����ƥ����������
	 * Step 4�������ʼ������֮֡���ƥ���̫�٣����³�ʼ��
	 * Step 5��ͨ��Hģ�ͻ�Fģ�ͽ��е�Ŀ��ʼ�����õ���֡������˶�����ʼMapPoints
	 * Step 6��ɾ����Щ�޷��������ǻ���ƥ���
	 * Step 7�������ǻ��õ���3D���װ��MapPoints
	 */
	void Tracking::MonocularInitialization()
	{
		if (!mpInitializer)
		{
			// ��ʼ����Ҫ��֡���ֱ���mInitialFrame��mCurrentFrame
			mInitialFrame = Frame(mCurrentFrame);

			// �ɵ�ǰ֡�����ʼ�� sigma:1.0 iterations:200
			mpInitializer = new Initializer(mK, mCurrentFrame, 1.0, 10);

			// ��ʼ��Ϊ-1 ��ʾû���κ�ƥ�䡣������洢����ƥ��ĵ��id
			std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
		}
		else
		{
			// Step 2 �����ǰ֡��������̫�٣�������20���������¹����ʼ��
			// NOTICE ֻ��������֡�����������������20ʱ�����ܼ������г�ʼ������
			if ((int)mCurrentFrame.mvKeys.size() <= 20)
			{
				delete mpInitializer;
				mpInitializer = static_cast<Initializer*>(NULL);
				fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
				return;
			}

			// Step 3 ��mInitialFrame��mCurrentFrame����ƥ����������
			ORBmatcher matcher(
				0.9,        //��ѵĺʹμ����������ֵı�ֵ��ֵ�������ǱȽϿ��ɵģ�����ʱһ����0.7
				true);      //���������ķ���

			// �� mInitialFrame,mCurrentFrame ����������ƥ��
			// mvbPrevMatchedΪ�ο�֡�����������꣬��ʼ���洢����mInitialFrame�����������꣬ƥ���洢����ƥ��õĵ�ǰ֡������������
			// mvIniMatches ����ο�֡F1���������Ƿ�ƥ���ϣ�index������F1��Ӧ������������ֵ�������ƥ��õ�F2����������
			int nmatches = matcher.SearchForInitialization(
				mInitialFrame, mCurrentFrame,    //��ʼ��ʱ�Ĳο�֡�͵�ǰ֡
				mvIniMatches,                    //����ƥ���ϵ
				20);                             //�������ڴ�С

			// Step 4 ��֤ƥ�����������ʼ������֮֡���ƥ���̫�٣����³�ʼ��
			if (nmatches < 16)
			{
				delete mpInitializer;
				mpInitializer = static_cast<Initializer*>(NULL);
				return;
			}

			cv::Mat Rcw; // Current Camera Rotation
			cv::Mat tcw; // Current Camera Translation
			std::vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

			// Step 5 ͨ��Hģ�ͻ�Fģ�ͽ��е�Ŀ��ʼ�����õ���֡������˶�����ʼMapPoints
			if (mpInitializer->Initialize(
				mCurrentFrame,      //��ǰ֡
				mvIniMatches,       //��ǰ֡�Ͳο�֡���������ƥ���ϵ
				Rcw, tcw,           //��ʼ���õ��������λ��
				mvIniP3D,           //�������ǻ��õ��Ŀռ�㼯��
				vbTriangulated))    //�Լ���Ӧ��mvIniMatches����,������Щ�㱻���ǻ���
			{
				// Step 6 ��ʼ���ɹ���ɾ����Щ�޷��������ǻ���ƥ���
				for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
				{
					if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
					{
						mvIniMatches[i] = -1;
						nmatches--;
					}
				}

				// Step 7 ����ʼ���ĵ�һ֡��Ϊ��������ϵ����˵�һ֡�任����Ϊ��λ����
				mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
				// ��Rcw��tcw����Tcw,����ֵ��mTcw��mTcwΪ��������ϵ���������ϵ�ı任����
				cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
				Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
				tcw.copyTo(Tcw.rowRange(0, 3).col(3));
				mCurrentFrame.SetPose(Tcw);

				// Step 8 ������ʼ����ͼ��MapPoints
				// Initialize������õ�mvIniP3D��
				// mvIniP3D��cv::Point3f���͵�һ���������Ǹ����3D�����ʱ������
				// CreateInitialMapMonocular��3D���װ��MapPoint���ʹ���KeyFrame��Map��
				CreateInitialMapMonocular();
			}//����ʼ���ɹ���ʱ�����
		}//�����Ŀ��ʼ�����Ѿ�������
	}

	/**
	 * @brief ��Ŀ����ɹ���ʼ���������ǻ��õ��ĵ�����MapPoints
	 *
	 */
	void Tracking::CreateInitialMapMonocular()
	{
		// Create KeyFrames ��Ϊ��Ŀ��ʼ��ʱ��Ĳο�֡�͵�ǰ֡���ǹؼ�֡
		KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);  // ��һ֡
		KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);  // �ڶ�֡
  
		// Step 2 ���ؼ�֡���뵽��ͼ
		mpMap->AddKeyFrame(pKFini);
		mpMap->AddKeyFrame(pKFcur);

		// Step 3 �ó�ʼ���õ���3D�������ɵ�ͼ��MapPoints
		//  mvIniMatches[i] ��ʾ��ʼ����֡������ƥ���ϵ��
		//  ������ͣ�i��ʾ֡1�йؼ��������ֵ��vMatches12[i]��ֵΪ֡2�Ĺؼ�������ֵ,û��ƥ���ϵ�Ļ���vMatches12[i]ֵΪ -1
		for (size_t i = 0; i < mvIniMatches.size(); i++)
		{
			// û��ƥ�䣬����
			if (mvIniMatches[i] < 0)
				continue;

			//Create MapPoint.
			// �����ǻ����ʼ��Ϊ�ռ�����������
			cv::Mat worldPos(mvIniP3D[i]);

			// Step 3.1 ��3D�㹹��MapPoint
			MapPoint* pMP = new MapPoint(
				worldPos,
				pKFcur,
				mpMap);

			// Step 3.2 Ϊ��MapPoint������ԣ�
			// a.�۲⵽��MapPoint�Ĺؼ�֡
			// b.��MapPoint��������
			// c.��MapPoint��ƽ���۲ⷽ�����ȷ�Χ

			// ��ʾ��KeyFrame��2D������Ͷ�Ӧ��3D��ͼ��
			pKFini->AddMapPoint(pMP, i);
			pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

			// a.��ʾ��MapPoint���Ա��ĸ�KeyFrame���ĸ�������۲⵽
			pMP->AddObservation(pKFini, i);
			pMP->AddObservation(pKFcur, mvIniMatches[i]);

			// b.���ڶ�۲⵽��MapPoint������������ѡ���д����Ե�������
			pMP->ComputeDistinctiveDescriptors();
			// c.���¸�MapPointƽ���۲ⷽ���Լ��۲����ķ�Χ
			pMP->UpdateNormalAndDepth();

			//mvIniMatches�±�i��ʾ�ڳ�ʼ���ο�֡�е�����������
			//mvIniMatches[i]�ǳ�ʼ����ǰ֡�е�����������
			mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
			mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

			//Add to Map
			mpMap->AddMapPoint(pMP);
		}

		// Step 5 ȡ��������ֵ��ȣ����ڳ߶ȹ�һ�� 
		// Ϊʲô�� pKFini ������ pKCur ? �𣺶����Եģ��ڲ�����λ�˱任��
		float medianDepth = pKFini->ComputeSceneMedianDepth(2);
		float invMedianDepth = 1.0f / medianDepth;

		// Step 6 ����֮֡��ı任��һ����ƽ�����1�ĳ߶���
		cv::Mat Tc2w = pKFcur->GetPose();
		// x/z y/z ��z��һ����1 
		Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
		pKFcur->SetPose(Tc2w);

		// Step 7 ��3D��ĳ߶�Ҳ��һ����1
		// Ϊʲô��pKFini? �ǲ��Ǿ�����ʹ�� pKFcur �õ��Ľ��Ҳ����ͬ��? ���ǵģ���Ϊ��ͬ������ά��
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

		// ��Ŀ��ʼ��֮�󣬵õ��ĳ�ʼ��ͼ�е����е㶼�Ǿֲ���ͼ��
		mvpLocalMapPoints = mpMap->GetAllMapPoints();
		mpReferenceKF = pKFcur;

		mCurrentFrame.mpReferenceKF = pKFcur;
		mLastFrame = Frame(mCurrentFrame);

		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

		mpMap->mvpKeyFrameOrigins.push_back(pKFini);

		// ��ʼ���ɹ������ˣ���ʼ���������
		mState = OK;
	}


}