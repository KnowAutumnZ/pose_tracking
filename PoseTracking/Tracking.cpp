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

		fx = config.ReadFloat("PoseTracking", "fx", 0.0);
		fy = config.ReadFloat("PoseTracking", "fy", 0.0);
		cx = config.ReadFloat("PoseTracking", "cx", 0.0);
		cy = config.ReadFloat("PoseTracking", "cy", 0.0);

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

		// Max/Min Frames to insert keyframes and to check relocalisation
		mMinFrames = 0;
		mMaxFrames = 30;
	}

	//���þֲ���ͼ��
	void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
	{
		mpLocalMapper = pLocalMapper;
	}
	//���ÿ��ӻ��鿴��
	void Tracking::SetViewer(Viewer *pViewer)
	{
		mpViewer = pViewer;
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
			mpCurrentFrame = new Frame(imGray, timestamp, mpIniORBextractor, mK, mDistort);
		else
			mpCurrentFrame = new Frame(imGray, timestamp, mpORBextractorLeft, mK, mDistort);

		// Step 3 ������
		Track();

		cv::Mat Tcw = mpCurrentFrame->mTcw.clone();
		delete mpCurrentFrame;

		return Tcw;
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
			bool bOK;
			if (mState == OK)
			{
				// Step 2.2 �˶�ģ���ǿյĻ������ض�λ�����ٲο��ؼ�֡���������ģ�͸���
				// ��һ������,����˶�ģ��Ϊ��,˵���Ǹճ�ʼ����ʼ�������Ѿ�������
				// �ڶ�������,�����ǰ֡�����ظ������ض�λ��֡�ĺ��棬���ǽ��ض�λ֡���ָ�λ��
				// mnLastRelocFrameId ��һ���ض�λ����һ֡
				if (mVelocity.empty() || mpCurrentFrame->mnId < mnLastRelocFrameId + 2)
				{
					// ������Ĺؼ�֡�����ٵ�ǰ����ͨ֡
					// ͨ��BoW�ķ�ʽ�ڲο�֡���ҵ�ǰ֡�������ƥ���
					// �Ż�ÿ�������㶼��Ӧ3D����ͶӰ���ɵõ�λ��
					bOK = TrackReferenceKeyFrame();
				}
				else
				{
					// ���������ͨ֡�����ٵ�ǰ����ͨ֡
					// ���ݺ���ģ���趨��ǰ֡�ĳ�ʼλ��
					// ͨ��ͶӰ�ķ�ʽ�ڲο�֡���ҵ�ǰ֡�������ƥ���
					// �Ż�ÿ������������Ӧ3D���ͶӰ���ɵõ�λ��
					bOK = TrackWithMotionModel();
					if (!bOK)
						//���ݺ���ģ��ʧ���ˣ�ֻ�ܸ��ݲο��ؼ�֡������
						bOK = TrackReferenceKeyFrame();
				}
			}

			// �����µĹؼ�֡��Ϊ��ǰ֡�Ĳο��ؼ�֡
			mpCurrentFrame->mpReferenceKF = mpReferenceKF;

			// Step 3���ڸ��ٵõ���ǰ֡��ʼ��̬�����ڶ�local map���и��ٵõ������ƥ�䣬���Ż���ǰλ��
			// ǰ��ֻ�Ǹ���һ֡�õ���ʼλ�ˣ����������ֲ��ؼ�֡���ֲ���ͼ�㣬�͵�ǰ֡����ͶӰƥ�䣬�õ�����ƥ���MapPoints�����Pose�Ż�
			if (!mbOnlyTracking)
			{
				if (bOK)
					bOK = TrackLocalMap();
			}

			//��������Ĳ������ж��Ƿ�׷�ٳɹ�
			if (bOK)
				mState = OK;
			else
				mState = LOST;

			// Step 4��������ʾ�߳��е�ͼ�������㡢��ͼ�����Ϣ
			mpFrameDrawer->Update(this);

			//ֻ���ڳɹ�׷��ʱ�ſ������ɹؼ�֡������
			if (bOK)
			{
				// Step 5�����ٳɹ������º����˶�ģ��
				if (!mpLastFrame->mTcw.empty())
				{                
					// ���º����˶�ģ�� TrackWithMotionModel �е�mVelocity
					cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
					mpLastFrame->GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
					mpLastFrame->GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
					// mVelocity = Tcl = Tcw * Twl,��ʾ��һ֡����ǰ֡�ı任�� ���� Twl = LastTwc
					mVelocity = mpCurrentFrame->mTcw*LastTwc;
				}
				else
					//�����ٶ�Ϊ��
					mVelocity = cv::Mat();

				//������ʾ�е�λ��
				mpMapDrawer->SetCurrentCameraPose(mpCurrentFrame->mTcw);

				// Step 6������۲ⲻ���ĵ�ͼ��   
				for (int i = 0; i < mpCurrentFrame->mvKeys.size(); i++)
				{
					MapPoint* pMP = mpCurrentFrame->mvpMapPoints[i];
					if (pMP)
						if (pMP->Observations() < 1)
						{
							mpCurrentFrame->mvbOutlier[i] = false;
							mpCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
						}
				}

				// Step 8����Ⲣ����ؼ�֡������˫Ŀ��RGB-D������µĵ�ͼ��
				if (NeedNewKeyFrame())
					CreateNewKeyFrame();

				// ��������˵������BA�б�Huber�˺����ж�Ϊ���Ĵ����µĹؼ�֡�У��ú�����BA�����������ǲ������������
				// ���ǹ�����һ֡λ�˵�ʱ�����ǲ�������Щ��㣬����ɾ��

				//  Step 9 ɾ����Щ��bundle adjustment�м��Ϊoutlier�ĵ�ͼ��
				for (int i = 0; i < mpCurrentFrame->mvKeys.size(); i++)
				{
					// �����һ��������Ҫִ���ж�����Ϊ, ǰ��Ĳ����п���ɾ�������еĵ�ͼ��
					if (mpCurrentFrame->mvpMapPoints[i] && mpCurrentFrame->mvbOutlier[i])
						mpCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
				}
			}

			//ȷ���Ѿ������˲ο��ؼ�֡
			if (!mpCurrentFrame->mpReferenceKF)
				mpCurrentFrame->mpReferenceKF = mpReferenceKF;

			// ������һ֡������,��ǰ֡����һ֡
			mpLastFrame = new Frame(mpCurrentFrame);
		}

		// Step 11����¼λ����Ϣ��������󱣴����еĹ켣
		if (!mpCurrentFrame->mTcw.empty())
		{
			// ���������̬Tcr = Tcw * Twr, Twr = Trw^-1
			cv::Mat Tcr = mpCurrentFrame->mTcw*mpCurrentFrame->mpReferenceKF->GetPoseInverse();
			//�������״̬
			mlRelativeFramePoses.push_back(Tcr);
			mlpReferences.push_back(mpReferenceKF);
			mlFrameTimes.push_back(mpCurrentFrame->mTimeStamp);
			mlbLost.push_back(mState == LOST);
		}
		else
		{
			// �������ʧ�ܣ������λ��ʹ����һ��ֵ
			mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
			mlpReferences.push_back(mlpReferences.back());
			mlFrameTimes.push_back(mlFrameTimes.back());
			mlbLost.push_back(mState == LOST);
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
			mpInitialFrame = mpCurrentFrame;

			// �ɵ�ǰ֡�����ʼ�� sigma:1.0 iterations:200
			mpInitializer = new Initializer(mpCurrentFrame, 1.0, 10);

			// ��ʼ��Ϊ-1 ��ʾû���κ�ƥ�䡣������洢����ƥ��ĵ��id
			std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
		}
		else
		{
			// Step 2 �����ǰ֡��������̫�٣�������20���������¹����ʼ��
			// NOTICE ֻ��������֡�����������������20ʱ�����ܼ������г�ʼ������
			if ((int)mpCurrentFrame->mvKeys.size() <= 20)
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
				mpInitialFrame, mpCurrentFrame,    //��ʼ��ʱ�Ĳο�֡�͵�ǰ֡
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
				mpCurrentFrame,      //��ǰ֡
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
				mpInitialFrame->SetPose(cv::Mat::eye(4, 4, CV_32F));
				// ��Rcw��tcw����Tcw,����ֵ��mTcw��mTcwΪ��������ϵ���������ϵ�ı任����
				cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
				Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
				tcw.copyTo(Tcw.rowRange(0, 3).col(3));
				mpCurrentFrame->SetPose(Tcw);

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
		KeyFrame* pKFini = new KeyFrame(mpInitialFrame, mpMap, mpKeyFrameDB);  // ��һ֡
		KeyFrame* pKFcur = new KeyFrame(mpCurrentFrame, mpMap, mpKeyFrameDB);  // �ڶ�֡
  
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
			mpCurrentFrame->mvpMapPoints[mvIniMatches[i]] = pMP;
			mpCurrentFrame->mvbOutlier[mvIniMatches[i]] = false;

			//Add to Map
			mpMap->AddMapPoint(pMP);
		}

		// Step 3.3 ���¹ؼ�֡������ӹ�ϵ
		// ��3D��͹ؼ�֮֡�佨���ߣ�ÿ������һ��Ȩ�أ��ߵ�Ȩ���Ǹùؼ�֡�뵱ǰ֡����3D��ĸ���
		pKFini->UpdateConnections();
		pKFcur->UpdateConnections();

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
		std::vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
		for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
		{
			if (vpAllMapPoints[iMP])
			{
				MapPoint* pMP = vpAllMapPoints[iMP];
				pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
			}
		}

		mpCurrentFrame->SetPose(pKFcur->GetPose());
		mnLastKeyFrameId = mpCurrentFrame->mnId;
		mpLastKeyFrame = pKFcur;

		// ��Ŀ��ʼ��֮�󣬵õ��ĳ�ʼ��ͼ�е����е㶼�Ǿֲ���ͼ��
		mvpLocalMapPoints = mpMap->GetAllMapPoints();
		mpReferenceKF = pKFcur;

		mpCurrentFrame->mpReferenceKF = pKFcur;
		mpLastFrame = mpCurrentFrame;

		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

		mpMap->mvpKeyFrameOrigins.push_back(pKFini);

		// ��ʼ���ɹ������ˣ���ʼ���������
		mState = OK;
	}

	/*
	 * @brief �òο��ؼ�֡�ĵ�ͼ�����Ե�ǰ��ͨ֡���и���
	 *
	 * Step 1������ǰ��ͨ֡��������ת��ΪBoW����
	 * Step 2��ͨ���ʴ�BoW���ٵ�ǰ֡��ο�֮֡���������ƥ��
	 * Step 3: ����һ֡��λ��̬��Ϊ��ǰ֡λ�˵ĳ�ʼֵ
	 * Step 4: ͨ���Ż�3D-2D����ͶӰ��������λ��
	 * Step 5���޳��Ż����ƥ����е����
	 * @return ���ƥ������10������true
	 *
	 */
	bool Tracking::TrackReferenceKeyFrame()
	{
		ORBmatcher matcher(0.7, true);
		std::vector<MapPoint*> vpMapPointMatches;

		int nmatches = matcher.SearchForRefModel(mpReferenceKF, mpCurrentFrame, vpMapPointMatches);
		// ƥ����ĿС��15����Ϊ����ʧ��
		if (nmatches < 15)
			return false;

		// Step 3:����һ֡��λ��̬��Ϊ��ǰ֡λ�˵ĳ�ʼֵ
		mpCurrentFrame->mvpMapPoints = vpMapPointMatches;
		mpCurrentFrame->SetPose(mpLastFrame->mTcw); // ����һ�ε�Tcw���ó�ֵ����PoseOptimization����������һЩ

		// Step 4:ͨ���Ż�3D-2D����ͶӰ��������λ��
		Optimizer::PoseOptimization(mpCurrentFrame);

		// Step 5���޳��Ż����ƥ����е����
		//֮�������Ż�֮����޳���㣬����Ϊ���Ż��Ĺ����о����˶���Щ���ı��
		int nmatchesMap = 0;
		for (int i = 0; i < mpCurrentFrame->mvpMapPoints.size(); i++)
		{
			if (mpCurrentFrame->mvpMapPoints[i])
			{
				//�����Ӧ����ĳ�������������c
				if (mpCurrentFrame->mvbOutlier[i])
				{
					//������ڵ�ǰ֡�д��ڹ��ĺۼ�
					MapPoint* pMP = mpCurrentFrame->mvpMapPoints[i];

					mpCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
					pMP->mbTrackInView = false;
					pMP->mnLastFrameSeen = mpCurrentFrame->mnId;
					nmatches--;
				}
				else if (mpCurrentFrame->mvpMapPoints[i]->Observations() > 0)
					nmatchesMap++;
			}
		}
		// ���ٳɹ�����Ŀ����10����Ϊ���ٳɹ����������ʧ��
		return nmatchesMap >= 10;
	}

	/**
	 * @brief ���ݺ㶨�ٶ�ģ������һ֡��ͼ�����Ե�ǰ֡���и���
	 * Step 1��������һ֡��λ�ˣ�����˫Ŀ��RGB-D���������������ֵ������ʱ��ͼ��
	 * Step 2��������һ֡�������Ӧ��ͼ�����ͶӰƥ��
	 * Step 3���Ż���ǰ֡λ��
	 * Step 4���޳���ͼ�������
	 * @return ���ƥ��������10����Ϊ���ٳɹ�������true
	 */
	bool Tracking::TrackWithMotionModel()
	{
		// ��С���� < 0.9*��С���� ƥ��ɹ��������ת
		ORBmatcher matcher(0.7, true);

		// Step 1��������һ֡��λ�ˣ�����˫Ŀ��RGB-D���������������ֵ������ʱ��ͼ��
		UpdateLastFrame();

		// Step 2������֮ǰ���Ƶ��ٶȣ��ú���ģ�͵õ���ǰ֡�ĳ�ʼλ�ˡ�
		mpCurrentFrame->SetPose(mVelocity*mpLastFrame->mTcw);

		// ��յ�ǰ֡�ĵ�ͼ��
		fill(mpCurrentFrame->mvpMapPoints.begin(), mpCurrentFrame->mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

		// ��������ƥ������е������뾶
		int th;
		if (mSensor != STEREO)
			th = 15;//��Ŀ
		else
			th = 7;//˫Ŀ

		// Step 3������һ֡��ͼ�����ͶӰƥ�䣬���ƥ��㲻���������������뾶����һ��
		int nmatches = matcher.SearchByProjection(mpCurrentFrame, mpLastFrame, th, mSensor == MONOCULAR);

		// ���ƥ���̫�٣������������뾶����һ��
		if (nmatches < 20)
		{
			fill(mpCurrentFrame->mvpMapPoints.begin(), mpCurrentFrame->mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
			nmatches = matcher.SearchByProjection(mpCurrentFrame, mpLastFrame, 2 * th, mSensor == MONOCULAR); // 2*th
		}

		// ������ǲ��ܹ�����㹻��ƥ���,��ô����Ϊ����ʧ��
		if (nmatches < 20)
			return false;

		// Step 4������3D-2DͶӰ��ϵ���Ż���ǰ֡λ��
		Optimizer::PoseOptimization(mpCurrentFrame);

		// Step 5���޳���ͼ�������
		int nmatchesMap = 0;
		for (int i = 0; i < mpCurrentFrame->mvpMapPoints.size(); i++)
		{
			if (mpCurrentFrame->mvpMapPoints[i])
			{
				if (mpCurrentFrame->mvbOutlier[i])
				{
					// ����Ż����ж�ĳ����ͼ������㣬����������й�ϵ
					MapPoint* pMP = mpCurrentFrame->mvpMapPoints[i];

					mpCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
					pMP->mbTrackInView = false;
					pMP->mnLastFrameSeen = mpCurrentFrame->mnId;
					nmatches--;
				}
				else if (mpCurrentFrame->mvpMapPoints[i]->Observations() > 0)
					// �ۼӳɹ�ƥ�䵽�ĵ�ͼ����Ŀ
					nmatchesMap++;
			}
		}
		// Step 6��ƥ�䳬��10�������Ϊ���ٳɹ�
		return nmatchesMap >= 10;
	}

	/**
	 * @brief ˫Ŀ��rgbd����ͷ�������ֵΪ��һ֡�����µ�MapPoints
	 *
	 * ��˫Ŀ��rgbd����£�ѡȡһЩ���СһЩ�ĵ㣨�ɿ�һЩ�� \n
	 * ����ͨ�����ֵ����һЩ�µ�MapPoints
	 */
	void Tracking::UpdateLastFrame()
	{
		// Step 1�����òο��ؼ�֡������һ֡����������ϵ�µ�λ��
		// ��һ��ͨ֡�Ĳο��ؼ�֡��ע�������õ��ǲο��ؼ�֡��λ��׼������������һ֡����ͨ֡
		KeyFrame* pRef = mpLastFrame->mpReferenceKF;

		// ref_keyframe �� lastframe��λ�˱任
		cv::Mat Tlr = mlRelativeFramePoses.back();

		// ����һ֡����������ϵ�µ�λ�˼������
		// l:last, r:reference, w:world
		// Tlw = Tlr*Trw 
		mpLastFrame->SetPose(Tlr*pRef->GetPose());

		// �����һ֡Ϊ�ؼ�֡�����ߵ�Ŀ����������˳�
		if (mnLastKeyFrameId == mpLastFrame->mnId || mSensor == MONOCULAR)
			return;
	}

	/**
	 * @brief �þֲ���ͼ���и��٣���һ���Ż�λ��
	 *
	 * 1. ���¾ֲ���ͼ�������ֲ��ؼ�֡�͹ؼ���
	 * 2. �Ծֲ�MapPoints����ͶӰƥ��
	 * 3. ����ƥ��Թ��Ƶ�ǰ֡����̬
	 * 4. ������̬�޳���ƥ��
	 * @return true if success
	 *
	 * Step 1�����¾ֲ��ؼ�֡mvpLocalKeyFrames�;ֲ���ͼ��mvpLocalMapPoints
	 * Step 2���ھֲ���ͼ�в����뵱ǰ֡ƥ���MapPoints, ��ʵҲ���ǶԾֲ���ͼ����и���
	 * Step 3�����¾ֲ�����MapPoints���λ���ٴ��Ż�
	 * Step 4�����µ�ǰ֡��MapPoints���۲�̶ȣ���ͳ�Ƹ��پֲ���ͼ��Ч��
	 * Step 5�������Ƿ���ٳɹ�
	 */
	bool Tracking::TrackLocalMap()
	{
		// Step 1�����¾ֲ��ؼ�֡ mvpLocalKeyFrames �;ֲ���ͼ�� mvpLocalMapPoints
		UpdateLocalMap();

		// Step 2��ɸѡ�ֲ���ͼ������������Ұ��Χ�ڵĵ�ͼ�㣬ͶӰ����ǰ֡����ƥ�䣬�õ������ƥ���ϵ
		SearchLocalPoints();

		// ���������֮ǰ���� Relocalization��TrackReferenceKeyFrame��TrackWithMotionModel �ж���λ���Ż���
		// Step 3��ǰ�������˸����ƥ���ϵ��BA�Ż��õ���׼ȷ��λ��
		Optimizer::PoseOptimization(mpCurrentFrame);
		mnMatchesInliers = 0;

		// Step 4�����µ�ǰ֡�ĵ�ͼ�㱻�۲�̶ȣ���ͳ�Ƹ��پֲ���ͼ��ƥ����Ŀ
		for (int i = 0; i < mpCurrentFrame->mvKeys.size(); i++)
		{
			if (mpCurrentFrame->mvpMapPoints[i])
			{
				// ���ڵ�ǰ֡�ĵ�ͼ����Ա���ǰ֡�۲⵽���䱻�۲�ͳ������1
				if (!mpCurrentFrame->mvbOutlier[i])
				{
					// �ҵ��õ��֡��mnFound �� 1
					mpCurrentFrame->mvpMapPoints[i]->IncreaseFound();
					//�鿴��ǰ�Ƿ����ڴ���λ����
					if (!mbOnlyTracking)
					{
						// ����õ�ͼ�㱻����۲���ĿnObs����0��ƥ���ڵ����+1
						// nObs�� ���۲⵽�������Ŀ����Ŀ+1��˫Ŀ��RGB-D��+2
						if (mpCurrentFrame->mvpMapPoints[i]->Observations() > 0)
							mnMatchesInliers++;
					}
					else
						// ��¼��ǰ֡���ٵ��ĵ�ͼ����Ŀ������ͳ�Ƹ���Ч��
						mnMatchesInliers++;
				}
				// ��������ͼ�������,���ҵ�ǰ������뻹��˫Ŀ��ʱ��,��ɾ�������
				else
					mpCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
			}
		}

		// Step 5�����ݸ���ƥ����Ŀ���ض�λ��������Ƿ���ٳɹ�
		// �������ոշ������ض�λ,��ô���ٳɹ�ƥ��20�������Ϊ�ǳɹ�����
		if (mpCurrentFrame->mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 20)
			return false;

		//�����������״̬��ֻҪ���ٵĵ�ͼ�����15������Ϊ�ɹ���
		if (mnMatchesInliers < 15)
			return false;
		else
			return true;
	}

	/**
	 * @brief ����LocalMap
	 *
	 * �ֲ���ͼ������
	 * 1��K1���ؼ�֡��K2���ٽ��ؼ�֡�Ͳο��ؼ�֡
	 * 2������Щ�ؼ�֡�۲⵽��MapPoints
	 */
	void Tracking::UpdateLocalMap()
	{
		// ���òο���ͼ�����ڻ�ͼ��ʾ�ֲ���ͼ�㣨��ɫ��
		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

		// �ù���ͼ�����¾ֲ��ؼ�֡�;ֲ���ͼ��,���������ÿ�ζ�Ҫ�������ӹؼ�֡,���ؼ�֡�ܶ�ʱ������ִ�к���
		UpdateLocalKeyFrames();
		UpdateLocalPoints();
	}

	/**
	 * @brief ���پֲ���ͼ��������¾ֲ��ؼ�֡
	 * �����Ǳ�����ǰ֡�ĵ�ͼ�㣬���۲⵽��Щ��ͼ��Ĺؼ�֡�����ڵĹؼ�֡���丸�ӹؼ�֡����ΪmvpLocalKeyFrames
	 * Step 1��������ǰ֡�ĵ�ͼ�㣬��¼�����ܹ۲⵽��ǰ֡��ͼ��Ĺؼ�֡
	 * Step 2�����¾ֲ��ؼ�֡��mvpLocalKeyFrames������Ӿֲ��ؼ�֡��������3������
	 *      ����1���ܹ۲⵽��ǰ֡��ͼ��Ĺؼ�֡��Ҳ��һ�����ӹؼ�֡
	 *      ����2��һ�����ӹؼ�֡�Ĺ��ӹؼ�֡����Ϊ�������ӹؼ�֡
	 *      ����3��һ�����ӹؼ�֡���ӹؼ�֡�����ؼ�֡
	 * Step 3�����µ�ǰ֡�Ĳο��ؼ�֡�����Լ����ӳ̶���ߵĹؼ�֡��Ϊ�ο��ؼ�֡
	 */
	void Tracking::UpdateLocalKeyFrames()
	{
		// Step 1��������ǰ֡�ĵ�ͼ�㣬��¼�����ܹ۲⵽��ǰ֡��ͼ��Ĺؼ�֡
		std::map<KeyFrame*, int> keyframeCounter;
		for (int i = 0; i < mpCurrentFrame->mvKeys.size(); i++)
		{
			if (mpCurrentFrame->mvpMapPoints[i])
			{
				MapPoint* pMP = mpCurrentFrame->mvpMapPoints[i];
				if (!pMP->isBad())
				{
					// �õ��۲⵽�õ�ͼ��Ĺؼ�֡�͸õ�ͼ���ڹؼ�֡�е�����
					const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

					// ����һ����ͼ����Ա�����ؼ�֡�۲⵽,��˶���ÿһ�ι۲�,���Թ۲⵽�����ͼ��Ĺؼ�֡�����ۼ�ͶƱ
					// ����Ĳ����ǳ����ʣ�
					// map[key] = value����Ҫ����ļ�����ʱ���Ḳ�Ǽ���Ӧ��ԭ����ֵ������������ڣ������һ���ֵ��
					// it->first �ǵ�ͼ�㿴���Ĺؼ�֡��ͬһ���ؼ�֡�����ĵ�ͼ����ۼӵ��ùؼ�֡����
					// �������keyframeCounter ��һ��������ʾĳ���ؼ�֡����2��������ʾ�ùؼ�֡�����˶��ٵ�ǰ֡(mCurrentFrame)�ĵ�ͼ�㣬Ҳ���ǹ��ӳ̶�
					for (std::map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
						keyframeCounter[it->first]++;
				}
				else
					mpCurrentFrame->mvpMapPoints[i] = NULL;
			}
		}

		// û�е�ǰ֡û�й��ӹؼ�֡������
		if (keyframeCounter.empty())
			return;

		// �洢�������۲������max���Ĺؼ�֡
		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

		// Step 2�����¾ֲ��ؼ�֡��mvpLocalKeyFrames������Ӿֲ��ؼ�֡��3������
		// ����վֲ��ؼ�֡
		mvpLocalKeyFrames.clear();
		// ������3���ڴ棬���������ټ�
		mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

		// Step 2.1 ����1���ܹ۲⵽��ǰ֡��ͼ��Ĺؼ�֡��Ϊ�ֲ��ؼ�֡ �����ھ���£����һ�����ӹؼ�֡�� 
		for (std::map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			// ����趨ΪҪɾ���ģ�����
			if (pKF->isBad())
				continue;

			// Ѱ�Ҿ������۲���Ŀ�Ĺؼ�֡
			if (it->second > max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			// ��ӵ��ֲ��ؼ�֡���б���
			mvpLocalKeyFrames.push_back(it->first);

			// �øùؼ�֡�ĳ�Ա����mnTrackReferenceForFrame ��¼��ǰ֡��id
			// ��ʾ���Ѿ��ǵ�ǰ֡�ľֲ��ؼ�֡�ˣ����Է�ֹ�ظ���Ӿֲ��ؼ�֡
			pKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
		}

		// Step 2.2 ����һ�����ӹؼ�֡��Ѱ�Ҹ���ľֲ��ؼ�֡ 
		for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
		{
			// ����ľֲ��ؼ�֡������30֡
			if (mvpLocalKeyFrames.size() > 30)
				break;

			KeyFrame* pKF = *itKF;

			// ����2:һ�����ӹؼ�֡�Ĺ��ӣ�ǰ10�����ؼ�֡����Ϊ�������ӹؼ�֡�����ھӵ��ھ���£��
			// �������֡����10֡,��ô�ͷ������о��й��ӹ�ϵ�Ĺؼ�֡
			const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

			// vNeighs �ǰ��չ��ӳ̶ȴӴ�С����
			for (std::vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				KeyFrame* pNeighKF = *itNeighKF;
				if (!pNeighKF->isBad())
				{
					// mnTrackReferenceForFrame��ֹ�ظ���Ӿֲ��ؼ�֡
					if (pNeighKF->mnTrackReferenceForFrame != mpCurrentFrame->mnId)
					{
						mvpLocalKeyFrames.push_back(pNeighKF);
						pNeighKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
						break;
					}
				}
			}

			// ����3:��һ�����ӹؼ�֡���ӹؼ�֡��Ϊ�ֲ��ؼ�֡�����ھӵĺ�������£��
			const std::set<KeyFrame*> spChilds = pKF->GetChilds();
			for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				KeyFrame* pChildKF = *sit;
				if (!pChildKF->isBad())
				{
					if (pChildKF->mnTrackReferenceForFrame != mpCurrentFrame->mnId)
					{
						mvpLocalKeyFrames.push_back(pChildKF);
						pChildKF->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
						break;
					}
				}
			}

			// ����3:��һ�����ӹؼ�֡�ĸ��ؼ�֡�����ھӵĸ�ĸ����£��
			KeyFrame* pParent = pKF->GetParent();
			if (pParent)
			{
				// mnTrackReferenceForFrame��ֹ�ظ���Ӿֲ��ؼ�֡
				if (pParent->mnTrackReferenceForFrame != mpCurrentFrame->mnId)
				{
					mvpLocalKeyFrames.push_back(pParent);
					pParent->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
				}
			}
		}

		// Step 3�����µ�ǰ֡�Ĳο��ؼ�֡�����Լ����ӳ̶���ߵĹؼ�֡��Ϊ�ο��ؼ�֡
		if (pKFmax)
		{
			mpReferenceKF = pKFmax;
			mpCurrentFrame->mpReferenceKF = mpReferenceKF;
		}
	}

	/*
	 * @brief ���¾ֲ��ؼ��㡣�ȰѾֲ���ͼ��գ�Ȼ�󽫾ֲ��ؼ�֡����Ч��ͼ����ӵ��ֲ���ͼ��
	 */
	void Tracking::UpdateLocalPoints()
	{
		// Step 1����վֲ���ͼ��
		mvpLocalMapPoints.clear();

		// Step 2�������ֲ��ؼ�֡ mvpLocalKeyFrames
		for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
		{
			KeyFrame* pKF = *itKF;
			const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

			// step 2�����ֲ��ؼ�֡�ĵ�ͼ����ӵ�mvpLocalMapPoints
			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP)
					continue;
				// �øõ�ͼ��ĳ�Ա����mnTrackReferenceForFrame ��¼��ǰ֡��id
				// ��ʾ���Ѿ��ǵ�ǰ֡�ľֲ���ͼ���ˣ����Է�ֹ�ظ���Ӿֲ���ͼ��
				if (pMP->mnTrackReferenceForFrame == mpCurrentFrame->mnId)
					continue;
				if (!pMP->isBad())
				{
					mvpLocalMapPoints.push_back(pMP);
					pMP->mnTrackReferenceForFrame = mpCurrentFrame->mnId;
				}
			}
		}
	}

	/**
	 * @brief �þֲ���ͼ�����ͶӰƥ�䣬�õ������ƥ���ϵ
	 * ע�⣺�ֲ���ͼ�����Ѿ��ǵ�ǰ֡��ͼ��Ĳ���Ҫ��ͶӰ��ֻ��Ҫ������Ĳ�������Ұ��Χ�ڵĵ�͵�ǰ֡����ͶӰƥ��
	 */
	void Tracking::SearchLocalPoints()
	{
		// Step 1��������ǰ֡�ĵ�ͼ�㣬�����Щ��ͼ�㲻����֮���ͶӰ����ƥ��
		for (vector<MapPoint*>::iterator vit = mpCurrentFrame->mvpMapPoints.begin(), vend = mpCurrentFrame->mvpMapPoints.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;
			if (pMP)
			{
				if (pMP->isBad())
				{
					*vit = static_cast<MapPoint*>(NULL);
				}
				else
				{
					// �����ܹ۲⵽�õ��֡����1(����ǰ֡�۲���)
					pMP->IncreaseVisible();
					// ��Ǹõ㱻��ǰ֡�۲⵽
					pMP->mnLastFrameSeen = mpCurrentFrame->mnId;
					// ��Ǹõ��ں�������ƥ��ʱ����ͶӰ����Ϊ�Ѿ���ƥ����
					pMP->mbTrackInView = false;
				}
			}
		}

		// ׼������ͶӰƥ��ĵ����Ŀ
		int nToMatch = 0;

		// Step 2���ж����оֲ���ͼ���г���ǰ֡��ͼ����ĵ㣬�Ƿ��ڵ�ǰ֡��Ұ��Χ��
		for (std::vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;

			// �Ѿ�����ǰ֡�۲⵽�ĵ�ͼ��϶�����Ұ��Χ�ڣ�����
			if (pMP->mnLastFrameSeen == mpCurrentFrame->mnId)
				continue;
			// ��������
			if (pMP->isBad())
				continue;

			// �жϵ�ͼ���Ƿ����ڵ�ǰ֡��Ұ��
			if (mpCurrentFrame->isInFrustum(pMP, 0.5))
			{
				// �۲⵽�õ��֡����1
				pMP->IncreaseVisible();
				// ֻ������Ұ��Χ�ڵĵ�ͼ��Ų���֮���ͶӰƥ��
				nToMatch++;
			}
		}

		// Step 3�������Ҫ����ͶӰƥ��ĵ����Ŀ����0���ͽ���ͶӰƥ�䣬���Ӹ����ƥ���ϵ
		if (nToMatch > 0)
		{
			ORBmatcher matcher(0.8);
			int th = 1;

			// �������ǰ���й��ض�λ����ô����һ�����ӿ�����������ֵ��Ҫ����
			if (mpCurrentFrame->mnId < mnLastRelocFrameId + 2)
				th = 5;

			// ͶӰƥ��õ������ƥ���ϵ
			matcher.SearchByProjection(mpCurrentFrame, mvpLocalMapPoints, th);
		}
	}

	/**
	 * @brief �жϵ�ǰ֡�Ƿ���Ҫ����ؼ�֡
	 *
	 * Step 1����VOģʽ�²�����ؼ�֡������ֲ���ͼ���ջ����ʹ�ã��򲻲���ؼ�֡
	 * Step 2�����������һ���ض�λ�ȽϽ������߹ؼ�֡��Ŀ����������ƣ�������ؼ�֡
	 * Step 3���õ��ο��ؼ�֡���ٵ��ĵ�ͼ������
	 * Step 4����ѯ�ֲ���ͼ�������Ƿ�æ,Ҳ���ǵ�ǰ�ܷ�����µĹؼ�֡
	 * Step 5������˫Ŀ��RGBD����ͷ��ͳ�ƿ�����ӵ���Ч��ͼ������ �� ���ٵ��ĵ�ͼ������
	 * Step 6�������Ƿ���Ҫ����ؼ�֡
	 * @return true         ��Ҫ
	 * @return false        ����Ҫ
	 */
	bool Tracking::NeedNewKeyFrame()
	{
		// Step 1����VOģʽ�²�����ؼ�֡
		if (mbOnlyTracking)
			return false;

		// ��ȡ��ǰ��ͼ�еĹؼ�֡��Ŀ
		const int nKFs = mpMap->KeyFramesInMap();

		// Step 4���õ��ο��ؼ�֡���ٵ��ĵ�ͼ������
		// UpdateLocalKeyFrames �����лὫ�뵱ǰ�ؼ�֡���ӳ̶���ߵĹؼ�֡�趨Ϊ��ǰ֡�Ĳο��ؼ�֡ 
		// ��ͼ�����С�۲����
		int nMinObs = 3;
		if (nKFs <= 2)
			nMinObs = 2;

		// �ο��ؼ�֡��ͼ���й۲����Ŀ>= nMinObs�ĵ�ͼ����Ŀ
		int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

		// Step 5����ѯ�ֲ���ͼ�߳��Ƿ�æ����ǰ�ܷ�����µĹؼ�֡
		bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

		// Step 7�������Ƿ���Ҫ����ؼ�֡
		// Step 7.1���趨������ֵ����ǰ֡�Ͳο��ؼ�֡���ٵ���ı���������Խ��Խ���������ӹؼ�֡
		float thRefRatio = 0.75f;

		// �ؼ�ֻ֡��һ֡����ô����ؼ�֡����ֵ���õĵ�һ�㣬����Ƶ�ʽϵ�
		if (nKFs < 2)
			thRefRatio = 0.4f;

		//��Ŀ����²���ؼ�֡��Ƶ�ʺܸ�    
		if (mSensor == MONOCULAR)
			thRefRatio = 0.9f;

		// Step 7.2���ܳ�ʱ��û�в���ؼ�֡�����Բ���
		const bool c1a = mpCurrentFrame->mnId >= mnLastKeyFrameId + mMaxFrames;

		// Step 7.3���������ؼ�֡����С�������localMapper���ڿ���״̬�����Բ���
		const bool c1b = (mpCurrentFrame->mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);

		// Step 7.5���Ͳο�֡��ȵ�ǰ���ٵ��ĵ�̫��, ͬʱ���ٵ����ڵ㻹����̫��
		const bool c2 = ((mnMatchesInliers < nRefMatches*thRefRatio) && mnMatchesInliers > 15);

		if ((c1a || c1b) && c2)
		{
			// Step 7.6��local mapping����ʱ����ֱ�Ӳ��룬�����е�ʱ��Ҫ�����������
			if (bLocalMappingIdle)
			{
				//���Բ���ؼ�֡
				return true;
			}
			else
			{
				mpLocalMapper->InterruptBA();
				// �����ﲻ������̫��ؼ�֡
				// tracking����ؼ�֡����ֱ�Ӳ��룬�����Ȳ��뵽mlNewKeyFrames�У�
				// Ȼ��localmapper�����pop�������뵽mspKeyFrames
				if (mpLocalMapper->KeyframesInQueue() < 3)
					//�����еĹؼ�֡��Ŀ���Ǻܶ�,���Բ���
					return true;
				else
					//�����л���Ĺؼ�֡��Ŀ̫��,��ʱ���ܲ���
					return false;
			}
		}
		else
			return false;
	}

	/**
	 * @brief �����µĹؼ�֡
	 * ���ڷǵ�Ŀ�������ͬʱ�����µ�MapPoints
	 *
	 * Step 1������ǰ֡����ɹؼ�֡
	 * Step 2������ǰ�ؼ�֡����Ϊ��ǰ֡�Ĳο��ؼ�֡
	 * Step 3������˫Ŀ��rgbd����ͷ��Ϊ��ǰ֡�����µ�MapPoints
	 */
	void Tracking::CreateNewKeyFrame()
	{
		// Step 1������ǰ֡����ɹؼ�֡
		KeyFrame* pKF = new KeyFrame(mpCurrentFrame, mpMap, mpKeyFrameDB);

		// Step 2������ǰ�ؼ�֡����Ϊ��ǰ֡�Ĳο��ؼ�֡
		// ��UpdateLocalKeyFrames�����лὫ�뵱ǰ�ؼ�֡���ӳ̶���ߵĹؼ�֡�趨Ϊ��ǰ֡�Ĳο��ؼ�֡
		mpReferenceKF = pKF;
		mpCurrentFrame->mpReferenceKF = pKF;

		// Step 4������ؼ�֡
		// �ؼ�֡���뵽�б� mlNewKeyFrames�У��ȴ�local mapping�߳�����
		mpLocalMapper->InsertKeyFrame(pKF);

		// ��ǰ֡��Ϊ�µĹؼ�֡������
		mnLastKeyFrameId = mpCurrentFrame->mnId;
		mpLastKeyFrame = pKF;
	}
}