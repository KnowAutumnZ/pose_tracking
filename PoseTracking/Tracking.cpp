#include "Tracking.h"

namespace PoseTracking
{
	Tracking::Tracking(const std::string &strSettingPath, eSensor sensor):mSensor(sensor)/*, mState(NO_IMAGES_YET)*/
	{
		std::string TrackingCFG = strSettingPath + "TrackingCFG.ini";

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
			mpIniORBextractor = new orbDetector(nfeatures * 2, scaleFactor, nlevels, iniThFAST, minThFAST);

		mpORBextractorLeft = new orbDetector(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

		float fx = config.ReadFloat("PoseTracking", "fx", 0.0);
		float fy = config.ReadFloat("PoseTracking", "fy", 0.0);
		float cx = config.ReadFloat("PoseTracking", "cx", 0.0);
		float cy = config.ReadFloat("PoseTracking", "cy", 0.0);

		mK = cv::Mat_<float>(3, 3) << (fx, 0, cx, 0, fy, cy, 0, 0, 1);

		float k1 = config.ReadFloat("PoseTracking", "k1", 0.0);
		float k2 = config.ReadFloat("PoseTracking", "k2", 0.0);
		float p1 = config.ReadFloat("PoseTracking", "p1", 0.0);
		float p2 = config.ReadFloat("PoseTracking", "p2", 0.0);
		float k3 = config.ReadFloat("PoseTracking", "k3", 0.0);

		mDistort = cv::Mat_<float>(5, 1) << (k1, k2, p1, p2, k3);
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

		cv::Mat imGray;
		if (im.channels() == 3) cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);

		if (mState == NO_IMAGES_YET || mState == NOT_INITIALIZED)
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
		// ���ͼ��λ�������ߵ�һ�����У���ΪNO_IMAGE_YET״̬
		if (mState == NO_IMAGES_YET) mState = NOT_INITIALIZED;

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
			mpInitializer = new Initializer(mK, mCurrentFrame, 1.0, 200);

			// ��ʼ��Ϊ-1 ��ʾû���κ�ƥ�䡣������洢����ƥ��ĵ��id
			std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
		}
		else
		{
			// Step 2 �����ǰ֡��������̫�٣�������100���������¹����ʼ��
			// NOTICE ֻ��������֡�����������������100ʱ�����ܼ������г�ʼ������
			if ((int)mCurrentFrame.mvKeys.size() <= 100)
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



		}





	}



}