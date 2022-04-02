#include "LocalMapping.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

namespace PoseTracking
{
	// ���캯��
	LocalMapping::LocalMapping(Map *pMap, const float bMonocular) :
		mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
		mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
	{
		/*
		 * mbStopRequested��    �ⲿ�̵߳��ã�Ϊtrue����ʾ�ⲿ�߳�����ֹͣ local mapping
		 * mbStopped��          Ϊtrue��ʾ���Բ���ֹlocalmapping �߳�
		 * mbNotStop��          true����ʾ��Ҫֹͣ localmapping �̣߳���ΪҪ����ؼ�֡�ˡ���Ҫ�� mbStopped ���ʹ��
		 * mbAcceptKeyFrames��  true��������ܹؼ�֡��tracking ��local mapping ֮��Ĺؼ�֡����
		 * mbAbortBA��          �Ƿ�����BA�Ż��ı�־λ
		 * mbFinishRequested��  ������ֹ��ǰ�̵߳ı�־��ע��ֻ�����󣬲�һ����ֹ����ֹҪ�� mbFinished
		 * mbResetRequested��   ����ǰ�̸߳�λ�ı�־��true����ʾһֱ����λ������λ��δ��ɣ���ʾ��λ���Ϊfalse
		 * mbFinished��         �ж�����LocalMapping::Run() �Ƿ���ɵı�־��
		 */
	}

	// ����׷���߳̾��
	void LocalMapping::SetTracker(Tracking *pTracker)
	{
		mpTracker = pTracker;

		fx = mK.at<float>(0, 0);
		fy = mK.at<float>(1, 1);
		cx = mK.at<float>(0, 2);
		cy = mK.at<float>(1, 2);

		invfx = 1.0 / fx;
		invfy = 1.0 / fy;
	}

	// �鿴��ǰ�Ƿ�������ܹؼ�֡
	bool LocalMapping::AcceptKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexAccept);
		return mbAcceptKeyFrames;
	}

	// ����"������ܹؼ�֡"��״̬��־
	void LocalMapping::SetAcceptKeyFrames(bool flag)
	{
		std::unique_lock<std::mutex> lock(mMutexAccept);
		mbAcceptKeyFrames = flag;
	}

	// ��ֹBA
	void LocalMapping::InterruptBA()
	{
		mbAbortBA = true;
	}

	// ����ؼ�֡,���ⲿ��Tracking���̵߳���;����ֻ�ǲ��뵽�б���,�ȴ��߳�������������д���
	void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		// ���ؼ�֡���뵽�б���
		mlNewKeyFrames.push_back(pKF);
		mbAbortBA = true;
	}

	// �鿴�б����Ƿ��еȴ�������Ĺؼ�֡,
	bool LocalMapping::CheckNewKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		return(!mlNewKeyFrames.empty());
	}

	// �߳�������
	void LocalMapping::Run()
	{
		// ���״̬����ʾ��ǰrun�����������У���δ����
		mbFinished = false;
		// ��ѭ��
		while (1)
		{
			// Step 1 ����Tracking��LocalMapping�����ڷ�æ״̬���벻Ҫ���ҷ��͹ؼ�֡������
			// LocalMapping�̴߳���Ĺؼ�֡����Tracking�̷߳�����
			SetAcceptKeyFrames(false);

			// �ȴ�����Ĺؼ�֡�б�Ϊ��
			if (CheckNewKeyFrames())
			{
				// Step 2 �����б��еĹؼ�֡����������BoW�����¹۲⡢�����ӡ�����ͼ�����뵽��ͼ��
				ProcessNewKeyFrame();

				// Step 3 ���ݵ�ͼ��Ĺ۲�����޳��������õĵ�ͼ��
				MapPointCulling();

				// Step 4 ��ǰ�ؼ�֡�����ڹؼ�֡ͨ�����ǻ������µĵ�ͼ�㣬ʹ�ø��ٸ���
				CreateNewMapPoints();

				// �Ѿ�����������е�����һ���ؼ�֡
				if (!CheckNewKeyFrames())
				{
					//  Step 5 ��鲢�ںϵ�ǰ�ؼ�֡�����ڹؼ�֡֡���������ڣ����ظ��ĵ�ͼ��
					SearchInNeighbors();
				}

				// ��ֹBA�ı�־
				mbAbortBA = false;
				// �Ѿ�����������е�����һ���ؼ�֡�����ұջ����û������ֹͣLocalMapping
				if (!CheckNewKeyFrames())
				{
					// Local BA
					// Step 6 ���ֲ���ͼ�еĹؼ�֡����2����ʱ����оֲ���ͼ��BA
					if (mpMap->KeyFramesInMap() > 2)
						// ע������ĵڶ��������ǰ���ַ���ݵ�,������� mbAbortBA ״̬�����仯ʱ���ܹ���ʱִ��/ֹͣBA
						Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

					// Check redundant local Keyframes
					// Step 7 ��Ⲣ�޳���ǰ֡���ڵĹؼ�֡������Ĺؼ�֡
					// ������ж����ùؼ�֡��90%�ĵ�ͼ����Ա������ؼ�֡�۲⵽
					KeyFrameCulling();
				}
			}

			// Tracking will see that Local Mapping is not busy
			SetAcceptKeyFrames(true);
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
	}

	/**
	 * @brief �����б��еĹؼ�֡����������BoW�����¹۲⡢�����ӡ�����ͼ�����뵽��ͼ��
	 *
	 */
	void LocalMapping::ProcessNewKeyFrame()
	{
		// Step 1���ӻ��������ȡ��һ֡�ؼ�֡
		// �ùؼ�֡������Tracking�߳���LocalMapping�в���Ĺؼ�֡���
		{
			std::unique_lock<std::mutex> lock(mMutexNewKFs);
			// ȡ���б�����ǰ��Ĺؼ�֡����Ϊ��ǰҪ����Ĺؼ�֡
			mpCurrentKeyFrame = mlNewKeyFrames.front();
			// ȡ����ǰ��Ĺؼ�֡����ԭ�����б���ɾ���ùؼ�֡
			mlNewKeyFrames.pop_front();
		}

		// Step 3����ǰ����ؼ�֡����Ч�ĵ�ͼ�㣬����normal�������ӵ���Ϣ
		// TrackLocalMap�к͵�ǰ֡��ƥ���ϵĵ�ͼ��͵�ǰ�ؼ�֡���й�����
		const std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
		// �Ե�ǰ���������ؼ�֡�е����еĵ�ͼ��չ������
		for (size_t i = 0; i < vpMapPointMatches.size(); i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
					{
						// �����ͼ�㲻�����Ե�ǰ֡�Ĺ۲⣨�������Ծֲ���ͼ�㣩��Ϊ��ǰ��ͼ����ӹ۲�
						pMP->AddObservation(mpCurrentKeyFrame, i);
						// ��øõ��ƽ���۲ⷽ��͹۲���뷶Χ
						pMP->UpdateNormalAndDepth();
						// ���µ�ͼ������������
						pMP->ComputeDistinctiveDescriptors();
					}
					else // this can only happen for new stereo points inserted by the Tracking
					{
						// �����ǰ֡���Ѿ������������ͼ��,���������ͼ����ȴû�а�������ؼ�֡����Ϣ
						// ��Щ��ͼ���������˫Ŀ��RGBD���ٹ����������ɵĵ�ͼ�㣬������CreateNewMapPoints ��ͨ�����ǻ�����
						// ��������ͼ�����mlpRecentAddedMapPoints���ȴ�����MapPointCulling�����ļ���
						mlpRecentAddedMapPoints.push_back(pMP);
					}
				}
			}
		}

		// Step 4�����¹ؼ�֡������ӹ�ϵ������ͼ��
		mpCurrentKeyFrame->UpdateConnections();

		// Step 5�����ùؼ�֡���뵽��ͼ��
		mpMap->AddKeyFrame(mpCurrentKeyFrame);
	}

	/**
	 * @brief ���������ͼ�㣬���ݵ�ͼ��Ĺ۲�����޳��������õ������ĵ�ͼ��
	 * mlpRecentAddedMapPoints���洢�����ĵ�ͼ�㣬������Ҫɾ�����в����׵�
	 */
	void LocalMapping::MapPointCulling()
	{
		std::list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
		const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

		// Step 1����������������ò�ͬ�Ĺ۲���ֵ
		int nThObs;
		if (mbMonocular)
			nThObs = 2;
		else
			nThObs = 3;
		const int cnThObs = nThObs;

		// Step 2�������������ӵĵ�ͼ��
		while (lit != mlpRecentAddedMapPoints.end())
		{
			MapPoint* pMP = *lit;
			if (pMP->isBad())
			{
				// Step 2.1���Ѿ��ǻ���ĵ�ͼ����Ӷ�����ɾ��
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if (pMP->GetFoundRatio() < 0.25f)
			{
				// Step 2.2�����ٵ��õ�ͼ���֡�����Ԥ�ƿɹ۲⵽�õ�ͼ���֡���ı���С��25%���ӵ�ͼ��ɾ��
				// (mnFound/mnVisible�� < 25%
				// mnFound ����ͼ�㱻����֡��������ͨ֡������������Խ��Խ��
				// mnVisible����ͼ��Ӧ�ñ������Ĵ���
				// (mnFound/mnVisible�������ڴ�FOV��ͷ���������ߣ�����խFOV��ͷ����������
				pMP->SetBadFlag();
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
			{
				// Step 2.3���Ӹõ㽨����ʼ���������Ѿ����˲�С��2���ؼ�֡
				// ���ǹ۲⵽�õ�������ȴ��������ֵcnThObs���ӵ�ͼ��ɾ��
				pMP->SetBadFlag();
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
				// Step 2.4���ӽ����õ㿪ʼ���Ѿ�����3���ؼ�֡��û�б��޳�������Ϊ�������ߵĵ�
				// ���û��SetBadFlag()�����Ӷ�����ɾ��
				lit = mlpRecentAddedMapPoints.erase(lit);
			else
				lit++;
		}
	}

	/**
	 * @brief �õ�ǰ�ؼ�֡�����ڹؼ�֡ͨ�����ǻ������µĵ�ͼ�㣬ʹ�ø��ٸ���
	 *
	 */
	void LocalMapping::CreateNewMapPoints()
	{
		// nn��ʾ������ѹ��ӹؼ�֡����Ŀ
		// ��ͬ��������Ҫ��һ��,��Ŀ��ʱ����Ҫ�и���ľ��нϺù��ӹ�ϵ�Ĺؼ�֡��������ͼ
		int nn = 10;
		if (mbMonocular)
			nn = 20;

		// Step 1���ڵ�ǰ�ؼ�֡�Ĺ��ӹؼ�֡���ҵ����ӳ̶���ߵ�nn֡���ڹؼ�֡
		const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

		// ������ƥ������ ��Ѿ��� < 0.6*�μѾ��룬�ȽϿ����ˡ��������ת
		ORBmatcher matcher(0.6, false);

		// ȡ����ǰ֡����������ϵ���������ϵ�ı任����
		cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));

		// �õ���ǰ�ؼ�֡����Ŀ����������������ϵ�е����ꡢ�ڲ�
		cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

		// mfScaleFactor = 1.2
		const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
		// ��¼���ǻ��ɹ��ĵ�ͼ����Ŀ
		int nnew = 0;
		// Step 2���������ڹؼ�֡������ƥ�䲢�ü���Լ���޳���ƥ�䣬�������ǻ�
		for (size_t i = 0; i < vpNeighKFs.size(); i++)
		{
			KeyFrame* pKF2 = vpNeighKFs[i];
			// ���ڵĹؼ�֡��������������ϵ�е�����
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			// ���������������ؼ�֡������λ��
			cv::Mat vBaseline = Ow2 - Ow1;
			// ���߳���
			const float baseline = cv::norm(vBaseline);

			// ��Ŀ������
			// ���ڹؼ�֡�ĳ��������ֵ
			const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
			// �����뾰��ı���
			const float ratioBaselineDepth = baseline / medianDepthKF2;
			// ��������ر�С������̫�ָ̻�3D�㲻׼����ô������ǰ�ڽӵĹؼ�֡��������3D��
			if (ratioBaselineDepth < 0.01)
				continue;

			// Step 4�����������ؼ�֡��λ�˼�������֮��Ļ�������
			cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);
			// Step 5��ͨ���ʴ������ؼ�֡��δƥ������������ƥ�䣬�ü���Լ��������Ⱥ�㣬�����µ�ƥ����
			std::vector<std::pair<size_t, size_t> > vMatchedIndices;
			matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

			cv::Mat Rcw2 = pKF2->GetRotation();
			cv::Mat Rwc2 = Rcw2.t();
			cv::Mat tcw2 = pKF2->GetTranslation();
			cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			// Step 6����ÿ��ƥ��ͨ�����ǻ�����3D��,�� Triangulate�������
			const int nmatches = vMatchedIndices.size();
			for (int ikp = 0; ikp < nmatches; ikp++)
			{
				// Step 6.1��ȡ��ƥ��������
				// ��ǰƥ����ڵ�ǰ�ؼ�֡�е�����
				const int &idx1 = vMatchedIndices[ikp].first;
				// ��ǰƥ������ڽӹؼ�֡�е�����
				const int &idx2 = vMatchedIndices[ikp].second;

				// ��ǰƥ���ڵ�ǰ�ؼ�֡�е�������
				const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeys[idx1];
				// ��ǰƥ�����ڽӹؼ�֡�е�������
				const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];

				// Step 6.2������ƥ��㷴ͶӰ�õ��Ӳ��
				// �����㷴ͶӰ,��ʵ�õ������ڸ����������ϵ�µ�һ���ǹ�һ���ķ�������,�������ķ�ͶӰ�����غ�
				cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx)*invfx, (kp1.pt.y - cy)*invfy, 1.0);
				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx)*invfx, (kp2.pt.y - cy)*invfy, 1.0);

				// ���������ϵת����������ϵ(�õ�����������ͶӰ���ߵ�һ��ͬ����������������ϵ�µı�ʾ,����ֻ�ܹ���ʾ����)���õ��Ӳ������ֵ
				cv::Mat ray1 = Rwc1 * xn1;
				cv::Mat ray2 = Rwc2 * xn2;

				// ƥ������߼н�����ֵ
				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

				// Step 6.4�����ǻ��ָ�3D��
				cv::Mat x3D;
				// cosParallaxRays > 0 && cosParallaxRays<0.9998�����Ӳ������,0.9998 ��Ӧ1��
				// ƥ���ԼнǴ������Ƿ��ָ�3D��
				// �ο���https://github.com/raulmur/ORB_SLAM2/issues/345
				if (cosParallaxRays > 0 && cosParallaxRays < 0.9998)
				{
					// Linear Triangulation Method
					// ��Initializer.cc�� Triangulate ����,ʵ����һ����,������ǰ�ͶӰ���󻻳��˱任����
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					x3D = vt.row(3).t();
					// ��һ��֮ǰ�ļ��
					if (x3D.at<float>(3) == 0)
						continue;
					// ��һ����Ϊ�������,Ȼ����ȡǰ������ά����Ϊŷʽ����
					x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
				}
				else
					continue; //No stereo and very low parallax, ����

				// Ϊ����������㣬ת����Ϊ��������
				cv::Mat x3Dt = x3D.t();

				// Step 6.5��������ɵ�3D���Ƿ������ǰ��,���ڵĻ��ͷ��������
				float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
				if (z1 <= 0)
					continue;

				float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
				if (z2 <= 0)
					continue;

				// Step 6.6������3D���ڵ�ǰ�ؼ�֡�µ���ͶӰ���
				const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
				const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
				const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
				const float invz1 = 1.0 / z1;

				// ��Ŀ�����
				float u1 = fx * x1*invz1 + cx;
				float v1 = fy * y1*invz1 + cy;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				// ���������һ�����ص�ƫ�2���ɶȿ���������ֵ��5.991
				if ((errX1*errX1 + errY1 * errY1) > 5.991*sigmaSquare1)
					continue;

				// ����3D������һ���ؼ�֡�µ���ͶӰ������ͬ��
				const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
				const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
				const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
				const float invz2 = 1.0 / z2;

				float u2 = fx * x2*invz2 + cx;
				float v2 = fy * y2*invz2 + cy;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				if ((errX2*errX2 + errY2 * errY2) > 5.991*sigmaSquare2)
					continue;

				// Step 6.7�����߶�������
				// ��������ϵ�£�3D�������������������������ָ��3D��
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = cv::norm(normal2);

				if (dist1 == 0 || dist2 == 0)
					continue;

				// ratioDist�ǲ����ǽ������߶��µľ������
				const float ratioDist = dist2 / dist1;
				// �������߶����ӵı���
				const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

				// ����ı�����ͼ��������ı�����Ӧ�ò�̫�࣬���������
				if (ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
					continue;

				// Step 6.8�����ǻ�����3D��ɹ��������MapPoint
				MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

				// Step 6.9��Ϊ��MapPoint������ԣ�
				// a.�۲⵽��MapPoint�Ĺؼ�֡
				pMP->AddObservation(mpCurrentKeyFrame, idx1);
				pMP->AddObservation(pKF2, idx2);

				mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
				pKF2->AddMapPoint(pMP, idx2);

				// b.��MapPoint��������
				pMP->ComputeDistinctiveDescriptors();

				// c.��MapPoint��ƽ���۲ⷽ�����ȷ�Χ
				pMP->UpdateNormalAndDepth();

				mpMap->AddMapPoint(pMP);

				// Step 6.10�����²����ĵ���������
				// ��ЩMapPoints���ᾭ��MapPointCulling�����ļ���
				mlpRecentAddedMapPoints.push_back(pMP);
				nnew++;
			}
		}
	}

	// �������ؼ�֡����̬���������ؼ�֮֡��Ļ�������
	cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
	{
		// �ȹ�����֮֡���R12,t12
		cv::Mat R1w = pKF1->GetRotation();
		cv::Mat t1w = pKF1->GetTranslation();
		cv::Mat R2w = pKF2->GetRotation();
		cv::Mat t2w = pKF2->GetTranslation();

		cv::Mat R12 = R1w * R2w.t();

		cv::Mat t12 = -R1w * R2w.t()*t2w + t1w;

		// �õ� t12 �ķ��Գƾ���
		cv::Mat t12x = SkewSymmetricMatrix(t12);

		const cv::Mat &K1 = mK;
		const cv::Mat &K2 = mK;

		// Essential Matrix: t12���R12
		// Fundamental Matrix: inv(K1)*E*inv(K2)
		return K1.t().inv()*t12x*R12*K2.inv();
	}

	// ������ά����v�ķ��Գƾ���
	cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
	{
		return (cv::Mat_<float>(3, 3) <<
			0, -v.at<float>(2), v.at<float>(1),
			v.at<float>(2), 0, -v.at<float>(0),
			-v.at<float>(1), v.at<float>(0), 0);
	}

	/**
	 * @brief ��鲢�ںϵ�ǰ�ؼ�֡������֡���������ڣ��ظ��ĵ�ͼ��
	 *
	 */
	void LocalMapping::SearchInNeighbors()
	{
		// Step 1����õ�ǰ�ؼ�֡�ڹ���ͼ��Ȩ������ǰnn���ڽӹؼ�֡
		// ��ʼ֮ǰ�ȶ��弸������
		// ��ǰ�ؼ�֡���ڽӹؼ�֡����Ϊһ�����ڹؼ�֡��Ҳ�����ھ�
		// ��һ�����ڹؼ�֡���ڵĹؼ�֡����Ϊ�������ڹؼ�֡��Ҳ�����ھӵ��ھ�

		// ��Ŀ���Ҫ20���ڽӹؼ�֡��˫Ŀ����RGBD��Ҫ10��
		int nn = 10;
		if (mbMonocular)
			nn = 20;

		// �͵�ǰ�ؼ�֡���ڵĹؼ�֡��Ҳ����һ�����ڹؼ�֡
		const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

		// Step 2���洢һ�����ڹؼ�֡����������ڹؼ�֡
		std::vector<KeyFrame*> vpTargetKFs;
		// ��ʼ�����к�ѡ��һ���ؼ�֡չ��������
		for (std::vector<KeyFrame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			// û�к͵�ǰ֡���й��ںϵĲ���
			if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
				continue;
			// ����һ�����ڹؼ�֡    
			vpTargetKFs.push_back(pKFi);
			// ����Ѿ�����
			pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

			// ��һ�����ڹؼ�֡�Ĺ��ӹ�ϵ��õ�5�����ڹؼ�֡ ��Ϊ�������ڹؼ�֡
			const std::vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
			// �����õ��Ķ������ڹؼ�֡
			for (std::vector<KeyFrame*>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
			{
				KeyFrame* pKFi2 = *vit2;
				// ��Ȼ����������ڹؼ�֡Ҫ��û�к͵�ǰ�ؼ�֡�����ں�,��������������ڹؼ�֡Ҳ���ǵ�ǰ�ؼ�֡
				if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
					continue;
				// ����������ڹؼ�֡    
				vpTargetKFs.push_back(pKFi2);
			}
		}

		// ʹ��Ĭ�ϲ���, ���źʹ��ű���0.6,ƥ��ʱ������������ת
		ORBmatcher matcher;
		// Step 3������ǰ֡�ĵ�ͼ��ֱ�ͶӰ���������ڹؼ�֡��Ѱ��ƥ����Ӧ�ĵ�ͼ������ںϣ���Ϊ����ͶӰ�ں�
		std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
		for (std::vector<KeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;

			// ����ͼ��ͶӰ���ؼ�֡�н���ƥ����ںϣ��ںϲ�������
			// 1.�����ͼ����ƥ��ؼ�֡�������㣬���Ҹõ��ж�Ӧ�ĵ�ͼ�㣬��ôѡ��۲���Ŀ����滻������ͼ��
			// 2.�����ͼ����ƥ��ؼ�֡�������㣬���Ҹõ�û�ж�Ӧ�ĵ�ͼ�㣬��ôΪ�õ���Ӹ�ͶӰ��ͼ��
			// ע�����ʱ��Ե�ͼ���ںϵĲ�����������Ч��
			matcher.Fuse(pKFi, vpMapPointMatches);
		}

		// Step 4�����������ڹؼ�֡��ͼ��ֱ�ͶӰ����ǰ�ؼ�֡��Ѱ��ƥ����Ӧ�ĵ�ͼ������ںϣ���Ϊ����ͶӰ�ں�
		// ���ڽ��д洢Ҫ�ںϵ�һ���ڽӺͶ����ڽӹؼ�֡����MapPoints�ļ���
		std::vector<MapPoint*> vpFuseCandidates;
		vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

		//  Step 4.1������ÿһ��һ���ڽӺͶ����ڽӹؼ�֡���ռ����ǵĵ�ͼ��洢�� vpFuseCandidates
		for (std::vector<KeyFrame*>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
		{
			KeyFrame* pKFi = *vitKF;
			std::vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

			// ������ǰһ���ڽӺͶ����ڽӹؼ�֡�����е�MapPoints,�ҳ���Ҫ�����ںϵĲ��Ҽ��뵽������
			for (std::vector<MapPoint*>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
			{
				MapPoint* pMP = *vitMP;
				if (!pMP)
					continue;

				// �����ͼ���ǻ��㣬�����Ѿ��ӽ�����vpFuseCandidates������
				if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
					continue;

				// ���뼯�ϣ�������Ѿ�����
				pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
				vpFuseCandidates.push_back(pMP);
			}
		}

		// Step 4.2�����е�ͼ��ͶӰ�ں�,�������ںϲ�������ȫ��ͬ��
		// ��ͬ�������������"ÿ���ؼ�֡�͵�ǰ�ؼ�֡�ĵ�ͼ������ں�",���������"��ǰ�ؼ�֡�������ڽӹؼ�֡�ĵ�ͼ������ں�"
		matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

		// Step 5�����µ�ǰ֡��ͼ��������ӡ���ȡ�ƽ���۲ⷽ�������
		vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
		for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					// �������ҵ�pMP�Ĺؼ�֡�У������ѵ�������
					pMP->ComputeDistinctiveDescriptors();

					// ����ƽ���۲ⷽ��͹۲����
					pMP->UpdateNormalAndDepth();
				}
			}
		}

		// Step 6�����µ�ǰ֡������֡�Ĺ������ӹ�ϵ
		mpCurrentKeyFrame->UpdateConnections();
	}

	/**
	 * @brief ��⵱ǰ�ؼ�֡�ڹ���ͼ�еĹؼ�֡�����ݵ�ͼ���ڹ���ͼ�е�����̶��޳��ù��ӹؼ�֡
	 * ����ؼ�֡���ж���90%���ϵĵ�ͼ���ܱ������ؼ�֡������3�����۲⵽
	 */
	void LocalMapping::KeyFrameCulling()
	{
		// �ú��������������룬������һ�£�
		// mpCurrentKeyFrame����ǰ�ؼ�֡������������ж����Ƿ���Ҫɾ��
		// pKF�� mpCurrentKeyFrame��ĳһ�����ӹؼ�֡
		// vpMapPoints��pKF��Ӧ�����е�ͼ��
		// pMP��vpMapPoints�е�ĳ����ͼ��
		// observations�������ܹ۲⵽pMP�Ĺؼ�֡
		// pKFi��observations�е�ĳ���ؼ�֡
		// scaleLeveli��pKFi�Ľ������߶�
		// scaleLevel��pKF�Ľ������߶�

		// Step 1�����ݹ���ͼ��ȡ��ǰ�ؼ�֡�����й��ӹؼ�֡
		std::vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

		// �����еĹ��ӹؼ�֡���б���
		for (std::vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
		{
			KeyFrame* pKF = *vit;
			// ��1���ؼ�֡����ɾ��������
			if (pKF->mnId == 0)
				continue;

			// Step 2����ȡÿ�����ӹؼ�֡�ĵ�ͼ��
			const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

			// �۲������ֵ��Ĭ��Ϊ3
			const int thObs = 3;

			// ��¼����۲�����Ŀ
			int nRedundantObservations = 0;

			int nMPs = 0;
			// Step 3�������ù��ӹؼ�֡�����е�ͼ�㣬�����ܱ���������3���ؼ�֡�۲⵽�ĵ�ͼ��Ϊ�����ͼ��
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (!pMP->isBad())
				{
					nMPs++;
					// pMP->Observations() �ǹ۲⵽�õ�ͼ����������Ŀ����Ŀ1��˫Ŀ2��
					if (pMP->Observations() > thObs)
					{
						const int &scaleLevel = pKF->mvKeys[i].octave;
						// Observation�洢���ǿ��Կ����õ�ͼ������йؼ�֡�ļ���
						const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

						int nObs = 0;
						// �����۲⵽�õ�ͼ��Ĺؼ�֡
						for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
						{
							KeyFrame* pKFi = mit->first;
							if (pKFi == pKF)
								continue;
							const int &scaleLeveli = pKFi->mvKeys[mit->second].octave;

							// �߶�Լ����ΪʲôpKF �߶�+1 Ҫ���ڵ��� pKFi �߶ȣ�
							// �ش���Ϊͬ������ͽ������㼶�ĵ�ͼ���׼ȷ
							if (scaleLeveli <= scaleLevel + 1)
							{
								nObs++;
								// �Ѿ��ҵ�3�����������Ĺؼ�֡����ֹͣ������
								if (nObs >= thObs)
									break;
							}
						}
						// ��ͼ�����ٱ�3���ؼ�֡�۲⵽���ͼ�¼Ϊ����㣬��������������Ŀ
						if (nObs >= thObs)
						{
							nRedundantObservations++;
						}
					}
				}

			}
			// Step 4������ùؼ�֡90%���ϵ���Ч��ͼ�㱻�ж�Ϊ����ģ�����Ϊ�ùؼ�֡������ģ���Ҫɾ���ùؼ�֡
			if (nRedundantObservations > 0.9*nMPs)
				pKF->SetBadFlag();
		}
	}
}