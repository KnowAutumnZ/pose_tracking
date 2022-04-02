#include "orbmatcher.h"

namespace PoseTracking
{
	// Ҫ�õ���һЩ��ֵ
	const int TH_HIGH = 100;
	const int TH_LOW = 50;
	const int HISTO_LENGTH = 30;

	// ���캯��,����Ĭ��ֵΪ0.6,true
	ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
	{

	}

	/**
	 * @brief ����ͼ��ͶӰ���ؼ�֡�н���ƥ����ںϣ��ںϲ�������
	 * 1.�����ͼ����ƥ��ؼ�֡�������㣬���Ҹõ��ж�Ӧ�ĵ�ͼ�㣬��ôѡ��۲���Ŀ����滻������ͼ��
	 * 2.�����ͼ����ƥ��ؼ�֡�������㣬���Ҹõ�û�ж�Ӧ�ĵ�ͼ�㣬��ôΪ�õ���Ӹ�ͶӰ��ͼ��

	 * @param[in] pKF           �ؼ�֡
	 * @param[in] vpMapPoints   ��ͶӰ�ĵ�ͼ��
	 * @param[in] th            �������ڵ���ֵ��Ĭ��Ϊ3
	 * @return int              ���µ�ͼ�������
	 */
	int ORBmatcher::Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float th)
	{
		// ȡ����ǰ֡λ�ˡ��ڲΡ���������������ϵ������
		cv::Mat Rcw = pKF->GetRotation();
		cv::Mat tcw = pKF->GetTranslation();

		const float &fx = mK.at<float>(0, 0);
		const float &fy = mK.at<float>(1, 1);
		const float &cx = mK.at<float>(0, 2);
		const float &cy = mK.at<float>(1, 2);

		cv::Mat Ow = pKF->GetCameraCenter();

		int nFused = 0;
		const int nMPs = vpMapPoints.size();
		// �������еĴ�ͶӰ��ͼ��
		for (int i = 0; i < nMPs; i++)
		{
			MapPoint* pMP = vpMapPoints[i];
			// Step 1 �жϵ�ͼ�����Ч�� 
			if (!pMP)
				continue;
			// ��ͼ����Ч �� �Ѿ��Ǹ�֡�ĵ�ͼ�㣨�����ںϣ�������
			if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
				continue;

			// ����ͼ��任���ؼ�֡���������ϵ��
			cv::Mat p3Dw = pMP->GetWorldPos();
			cv::Mat p3Dc = Rcw * p3Dw + tcw;

			// ���ֵΪ��������
			if (p3Dc.at<float>(2) < 0.0f)
				continue;

			// Step 2 �õ���ͼ��ͶӰ���ؼ�֡��ͼ������
			const float invz = 1 / p3Dc.at<float>(2);
			const float x = p3Dc.at<float>(0)*invz;
			const float y = p3Dc.at<float>(1)*invz;

			const float u = fx * x + cx;
			const float v = fy * y + cy;

			// ͶӰ����Ҫ����Ч��Χ��
			if (!pKF->IsInImage(u, v))
				continue;

			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			cv::Mat PO = p3Dw - Ow;
			const float dist3D = cv::norm(PO);

			// Step 3 ��ͼ�㵽�ؼ�֡������ľ�������������Ч��Χ��
			if (dist3D<minDistance || dist3D>maxDistance)
				continue;

			// Step 4 ��ͼ�㵽���ĵ�������õ�ͼ���ƽ���۲�����֮��н�ҪС��60��
			cv::Mat Pn = pMP->GetNormal();
			if (PO.dot(Pn) < 0.5*dist3D)
				continue;

			// ���ݵ�ͼ�㵽������ľ���Ԥ��ƥ������ڵĽ������߶�
			int nPredictedLevel = pMP->PredictScale(dist3D, pKF);
			// ȷ��������Χ
			const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
			// Step 5 ��ͶӰ�㸽�������������ҵ���ѡƥ��������
			const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

			if (vIndices.empty())
				continue;

			// Step 6 ����Ѱ�����ƥ���
			const cv::Mat dMP = pMP->GetDescriptor();
			int bestDist = 256;
			int bestIdx = -1;
			for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)// ����3������������Χ�ڵ�features
			{
				const size_t idx = *vit;
				const cv::KeyPoint &kp = pKF->mvKeys[idx];

				const int &kpLevel = kp.octave;
				// �������㼶Ҫ�ӽ���ͬһ���Сһ�㣩����������
				if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
					continue;

				// ����ͶӰ�����ѡƥ��������ľ��룬���ƫ��ܴ�ֱ������
				// ��Ŀ���
				const float &kpx = kp.pt.x;
				const float &kpy = kp.pt.y;
				const float ex = u - kpx;
				const float ey = v - kpy;
				const float e2 = ex * ex + ey * ey;

				// ���ɶ�Ϊ2�ģ�����������ֵ5.99�����������һ�����ص�ƫ�
				if (e2*pKF->mvInvLevelSigma2[kpLevel] > 5.99)
					continue;

				const cv::Mat &dKF = pKF->mDescriptors.row(idx);
				const int dist = DescriptorDistance(dMP, dKF);

				// ��ͶӰ��������Ӿ�����С
				if (dist < bestDist)
				{
					bestDist = dist;
					bestIdx = idx;
				}
			}

			// Step 7 �ҵ�ͶӰ���Ӧ�����ƥ�������㣬�����Ƿ���ڵ�ͼ�����ںϻ�����
			// ���ƥ�����ҪС����ֵ
			if (bestDist <= TH_LOW)
			{
				MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
				if (pMPinKF)
				{
					// ������ƥ����ж�Ӧ��Ч��ͼ�㣬ѡ�񱻹۲���������Ǹ��滻
					if (!pMPinKF->isBad())
					{
						if (pMPinKF->Observations() > pMP->Observations())
							pMP->Replace(pMPinKF);
						else
							pMPinKF->Replace(pMP);
					}
				}
				else
				{
					// ������ƥ���û�ж�Ӧ��ͼ�㣬��ӹ۲���Ϣ
					pMP->AddObservation(pKF, bestIdx);
					pKF->AddMapPoint(pMP, bestIdx);
				}
				nFused++;
			}
		}
		return nFused;
	}

	/*
	 * @brief ���û�������F12����Լ������BoW����ƥ�������ؼ�֡��δƥ��������㣬�����µ�ƥ����
	 * ������˵��pKF1ͼ���ÿ����������pKF2ͼ��ͬһnode�ڵ����������������ƥ�䣬�ж��Ƿ�����Լ�����Լ��������Լ������ƥ���������
	 * @param pKF1          �ؼ�֡1
	 * @param pKF2          �ؼ�֡2
	 * @param F12           ��2��1�Ļ�������
	 * @param vMatchedPairs �洢ƥ��������ԣ������������ڹؼ�֡�е�������ʾ
	 * @param bOnlyStereo   ��˫Ŀ��rgbd����£��Ƿ�Ҫ������������ͼ����ƥ��
	 * @return              �ɹ�ƥ�������
	 */
	int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, std::vector<std::pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
	{
		cv::Mat d1 = pKF1->mDescriptors;
		cv::Mat d2 = pKF2->mDescriptors;

		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

		std::vector<cv::DMatch> matches;
		matcher->match(d1, d2, matches);

		//�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
		double min_dist = 10000, max_dist = 0;
		for (int i = 0; i < d1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		int nmatches = 0;
		for (int i = 0; i < d1.rows; i++)
		{
			// ͨ������������idx1��pKF1��ȡ����Ӧ��MapPoint
			MapPoint* pMP1 = pKF1->GetMapPoint(matches[i].queryIdx);
			// ͨ������������idx2��pKF2��ȡ����Ӧ��MapPoint
			MapPoint* pMP2 = pKF2->GetMapPoint(matches[i].trainIdx);

			// ����Ѱ�ҵ���δƥ��������㣬����pMP1Ӧ��ΪNULL
			if (matches[i].distance > mfNNratio * max_dist || pMP1 || pMP2)
			{
				continue;
			}
			else
			{
				vMatchedPairs.push_back(std::make_pair(matches[i].queryIdx, matches[i].trainIdx));
				nmatches++;
			}
		}

		return nmatches;
	}

	/**
	 * @brief ͨ��ͶӰ��ͼ�㵽��ǰ֡����Local MapPoint���и���
	 * ����
	 * Step 1 ������Ч�ľֲ���ͼ��
	 * Step 2 �趨�����������ڵĴ�С��ȡ�����ӽ�, ����ǰ�ӽǺ�ƽ���ӽǼнǽ�Сʱ, rȡһ����С��ֵ
	 * Step 3 ͨ��ͶӰ���Լ��������ں�Ԥ��ĳ߶Ƚ�������, �ҳ������뾶�ڵĺ�ѡƥ�������
	 * Step 4 Ѱ�Һ�ѡƥ����е���Ѻʹμ�ƥ���
	 * Step 5 ɸѡ���ƥ���
	 * @param[in] F                         ��ǰ֡
	 * @param[in] vpMapPoints               �ֲ���ͼ�㣬���Ծֲ��ؼ�֡
	 * @param[in] th                        ������Χ
	 * @return int                          �ɹ�ƥ�����Ŀ
	 */
	int ORBmatcher::SearchByProjection(Frame* F, const std::vector<MapPoint*> &vpMapPoints, const float th)
	{
		int nmatches = 0;

		// ��� th��=1 (RGBD ������߸ոս��й��ض�λ), ��Ҫ����Χ����
		const bool bFactor = th != 1.0;

		// Step 1 ������Ч�ľֲ���ͼ��
		for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
		{
			MapPoint* pMP = vpMapPoints[iMP];

			// �жϸõ��Ƿ�ҪͶӰ
			if (!pMP->mbTrackInView)
				continue;

			if (pMP->isBad())
				continue;

			// ͨ������Ԥ��Ľ������������ò�������ڵ�ǰ��֡
			const int &nPredictedLevel = pMP->mnTrackScaleLevel;

			// The size of the window will depend on the viewing direction
			// Step 2 �趨�����������ڵĴ�С��ȡ�����ӽ�, ����ǰ�ӽǺ�ƽ���ӽǼнǽ�Сʱ, rȡһ����С��ֵ
			float r = RadiusByViewingCos(pMP->mTrackViewCos);

			// �����Ҫ����Χ�������������ֵth
			if (bFactor)
				r *= th;

			// Step 3 ͨ��ͶӰ���Լ��������ں�Ԥ��ĳ߶Ƚ�������, �ҳ������뾶�ڵĺ�ѡƥ�������
			const std::vector<size_t> vIndices =
				GetFeaturesInArea(F, pMP->mTrackProjX, pMP->mTrackProjY,        // �õ�ͼ��ͶӰ��һ֡�ϵ�����
					r*F->mvScaleFactors[nPredictedLevel],						// ��Ϊ�������ڵĴ�С�͸������㱻׷�ٵ�ʱ�����ĳ߶�Ҳ�й�ϵ
					nPredictedLevel - 1, nPredictedLevel);						// ������ͼ�㷶Χ

			// û�ҵ���ѡ��,�ͷ����Ե�ǰ���ƥ��
			if (vIndices.empty())
				continue;

			const cv::Mat MPdescriptor = pMP->GetDescriptor();

			// ���ŵĴ��ŵ������Ӿ����index
			int bestDist = 256;
			int bestLevel = -1;
			int bestDist2 = 256;
			int bestLevel2 = -1;
			int bestIdx = -1;

			// Get best and second matches with near keypoints
			// Step 4 Ѱ�Һ�ѡƥ����е���Ѻʹμ�ƥ���
			for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;

				// ���Frame�еĸ���Ȥ���Ѿ��ж�Ӧ��MapPoint��,���˳��ô�ѭ��
				if (F->mvpMapPoints[idx])
					if (F->mvpMapPoints[idx]->Observations() > 0)
						continue;

				const cv::Mat &d = F->mDescriptors.row(idx);

				// �����ͼ��ͺ�ѡͶӰ��������Ӿ���
				const int dist = DescriptorDistance(MPdescriptor, d);

				// Ѱ�������Ӿ�����С�ʹ�С�������������
				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestLevel2 = bestLevel;
					bestLevel = F->mvKeys[idx].octave;
					bestIdx = idx;
				}
				else if (dist < bestDist2)
				{
					bestLevel2 = F->mvKeys[idx].octave;
					bestDist2 = dist;
				}
			}

			// Step 5 ɸѡ���ƥ���
			// ���ƥ����뻹��Ҫ�������趨��ֵ��
			if (bestDist <= TH_HIGH)
			{
				// ����1��bestLevel==bestLevel2 ��ʾ ��Ѻʹμ���ͬһ�������㼶
				// ����2��bestDist>mfNNratio*bestDist2 ��ʾ��ѺʹμѾ��벻������ֵ������������˵ bestDist/bestDist2 ԽСԽ��
				if (bestLevel == bestLevel2 && bestDist > mfNNratio*bestDist2)
					continue;

				//������: ΪFrame�е����������Ӷ�Ӧ��MapPoint
				F->mvpMapPoints[bestIdx] = pMP;
				nmatches++;
			}
		}

		return nmatches;
	}

	// ���ݹ۲���ӽ�������ƥ���ʱ���������ڴ�С
	float ORBmatcher::RadiusByViewingCos(const float &viewCos)
	{
		// ���ӽ����С��3.6�㣬��Ӧcos(3.6��)=0.998��������Χ��2.5��������4
		if (viewCos > 0.998)
			return 2.5;
		else
			return 4.0;
	}

	/**
	 * @brief ����һ֡���ٵĵ�ͼ��ͶӰ����ǰ֡����������ƥ��㡣���ڸ���ǰһ֡
	 * ����
	 * Step 1 ������תֱ��ͼ�����ڼ����תһ����
	 * Step 2 ���㵱ǰ֡��ǰһ֡��ƽ������
	 * Step 3 ����ǰһ֡��ÿһ����ͼ�㣬ͨ�����ͶӰģ�ͣ��õ�ͶӰ����ǰ֡����������
	 * Step 4 ���������ǰ��ǰ���������ж������߶ȷ�Χ
	 * Step 5 ������ѡƥ��㣬Ѱ�Ҿ�����С�����ƥ���
	 * Step 6 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
	 * Step 7 ������תһ�¼�⣬�޳���һ�µ�ƥ��
	 * @param[in] CurrentFrame          ��ǰ֡
	 * @param[in] LastFrame             ��һ֡
	 * @param[in] th                    ������Χ��ֵ��Ĭ�ϵ�ĿΪ7��˫Ŀ15
	 * @param[in] bMono                 �Ƿ�Ϊ��Ŀ
	 * @return int                      �ɹ�ƥ�������
	 */
	int ORBmatcher::SearchByProjection(Frame* CurrentFrame, const Frame* LastFrame, const float th, const bool bMono)
	{
		int nmatches = 0;

		const float fx = mK.at<float>(0, 0);
		const float fy = mK.at<float>(1, 1);
		const float cx = mK.at<float>(0, 2);
		const float cy = mK.at<float>(1, 2);

		// Step 1 ������תֱ��ͼ�����ڼ����תһ����
		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i < HISTO_LENGTH; i++)
			rotHist[i].reserve(500);

		//! ԭ���ߴ����� const float factor = 1.0f/HISTO_LENGTH; �Ǵ���ģ�����Ϊ�������
		const float factor = HISTO_LENGTH / 360.0f;

		// Step 2 ���㵱ǰ֡��ǰһ֡��ƽ������
		//��ǰ֡�����λ��
		const cv::Mat Rcw = CurrentFrame->mTcw.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tcw = CurrentFrame->mTcw.rowRange(0, 3).col(3);

		//��ǰ�������ϵ����������ϵ��ƽ���������������ϵ��
		const cv::Mat twc = -Rcw.t()*tcw;

		//  Step 3 ����ǰһ֡��ÿһ����ͼ�㣬ͨ�����ͶӰģ�ͣ��õ�ͶӰ����ǰ֡����������
		for (int i = 0; i < LastFrame->mvKeys.size(); i++)
		{
			MapPoint* pMP = LastFrame->mvpMapPoints[i];

			if(!pMP) continue;

			if (!LastFrame->mvbOutlier[i])
			{
				// ����һ֡��Ч��MapPointsͶӰ����ǰ֡����ϵ
				cv::Mat x3Dw = pMP->GetWorldPos();
				cv::Mat x3Dc = Rcw * x3Dw + tcw;

				const float xc = x3Dc.at<float>(0);
				const float yc = x3Dc.at<float>(1);
				const float invzc = 1.0 / x3Dc.at<float>(2);

				if (invzc < 0)
					continue;

				// ͶӰ����ǰ֡��
				float u = fx * xc*invzc + cx;
				float v = fy * yc*invzc + cy;

				if (u<CurrentFrame->mnMinX || u>CurrentFrame->mnMaxX)
					continue;
				if (v<CurrentFrame->mnMinY || v>CurrentFrame->mnMaxY)
					continue;

				// ��һ֡�е�ͼ���Ӧ��ά���������ڵĽ������㼶
				int nLastOctave = LastFrame->mvKeys[i].octave;

				// Search in a window. Size depends on scale
				// ��Ŀ��th = 15��˫Ŀ��th = 7
				float radius = th * CurrentFrame->mvScaleFactors[nLastOctave]; // �߶�Խ��������ΧԽ��

				// ��¼��ѡƥ����id
				std::vector<size_t> vIndices2;

				// Step 4 ���������ǰ��ǰ���������ж������߶ȷ�Χ��
				vIndices2 = GetFeaturesInArea(CurrentFrame, u, v, radius, nLastOctave - 1, nLastOctave + 1);

				if (vIndices2.empty())
					continue;

				const cv::Mat dMP = pMP->GetDescriptor();

				int bestDist = 256;
				int bestIdx2 = -1;

				// Step 5 ������ѡƥ��㣬Ѱ�Ҿ�����С�����ƥ��� 
				for (std::vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
				{
					const size_t i2 = *vit;

					// ������������Ѿ��ж�Ӧ��MapPoint��,���˳��ô�ѭ��
					if (CurrentFrame->mvpMapPoints[i2])
						if (CurrentFrame->mvpMapPoints[i2]->Observations() > 0)
							continue;

					const cv::Mat &d = CurrentFrame->mDescriptors.row(i2);
					const int dist = DescriptorDistance(dMP, d);

					if (dist < bestDist)
					{
						bestDist = dist;
						bestIdx2 = i2;
					}
				}

				// ���ƥ�����ҪС���趨��ֵ
				if (bestDist <= TH_HIGH)
				{
					CurrentFrame->mvpMapPoints[bestIdx2] = pMP;
					nmatches++;

					// Step 6 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
					if (mbCheckOrientation)
					{
						float rot = LastFrame->mvKeys[i].angle - CurrentFrame->mvKeys[bestIdx2].angle;
						if (rot < 0.0)
							rot += 360.0f;
						int bin = round(rot*factor);
						if (bin == HISTO_LENGTH)
							bin = 0;
						assert(bin >= 0 && bin < HISTO_LENGTH);
						rotHist[bin].push_back(bestIdx2);
					}
				}
			}
		}

		//  Step 7 ������תһ�¼�⣬�޳���һ�µ�ƥ��
		if (mbCheckOrientation)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				// ������������ǰ3���ĵ�ԣ��޳�
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
					{
						CurrentFrame->mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
						nmatches--;
					}
				}
			}
		}
		return nmatches;
	}

	int ORBmatcher::SearchForRefModel(KeyFrame *pKF, Frame* F, std::vector<MapPoint*> &vpMapPointMatches)
	{
		// ��ȡ�ùؼ�֡�ĵ�ͼ��
		const std::vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
		// ����ͨ֡F�����������һ��
		vpMapPointMatches = std::vector<MapPoint*>(F->mvpMapPoints.size(), static_cast<MapPoint*>(NULL));

		cv::Mat d1 = pKF->mDescriptors;
		cv::Mat d2 = F->mDescriptors;

		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		
		std::vector<cv::DMatch> matches;
		//BFMatcher matcher ( NORM_HAMMING );
		matcher->match(d1, d2, matches);

		//�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
		double min_dist = 10000, max_dist = 0;
		for (int i = 0; i < d1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
		int nmatches = 0;
		for (int i = 0; i < d1.rows; i++)
		{
			if (matches[i].distance > mfNNratio * max_dist || !vpMapPointsKF[matches[i].queryIdx])
				continue;
			else
			{
				vpMapPointMatches[matches[i].trainIdx] = vpMapPointsKF[matches[i].queryIdx];
				nmatches++;
			}

		}
		return nmatches;
	}

	/**
	 * @brief ��Ŀ��ʼ�������ڲο�֡�͵�ǰ֡��������ƥ��
	 * ����
	 * Step 1 ������תֱ��ͼ
	 * Step 2 �ڰ뾶������������ǰ֡F2�����еĺ�ѡƥ��������
	 * Step 3 �����������������е�����Ǳ�ڵ�ƥ���ѡ�㣬�ҵ����ŵĺʹ��ŵ�
	 * Step 4 �����Ŵ��Ž�����м�飬������ֵ������/���ű�����ɾ���ظ�ƥ��
	 * Step 5 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
	 * Step 6 ɸ����תֱ��ͼ�С�������������
	 * Step 7 �����ͨ��ɸѡ��ƥ��õ������㱣��
	 * @param[in] F1                        ��ʼ���ο�֡
	 * @param[in] F2                        ��ǰ֡
	 * @param[in & out] vbPrevMatched       �����洢���ǲο�֡���������������꣬�ú�������Ϊƥ��õĵ�ǰ֡������������
	 * @param[in & out] vnMatches12         ����ο�֡F1���������Ƿ�ƥ���ϣ�index������F1��Ӧ������������ֵ�������ƥ��õ�F2����������
	 * @param[in] windowSize                ��������
	 * @return int                          ���سɹ�ƥ�����������Ŀ
	 */
	int ORBmatcher::SearchForInitialization(Frame* F1, Frame* F2, std::vector<int> &vnMatches12, int windowSize)
	{
		int nmatches = 0;
		// F1���������F2��ƥ���ϵ��ע���ǰ���F1��������Ŀ����ռ�
		vnMatches12 = std::vector<int>(F1->mvKeys.size(), -1);
		
		// Step 1 ������תֱ��ͼ��HISTO_LENGTH = 30
		std::vector<int> rotHist[HISTO_LENGTH];
		// ÿ��bin��Ԥ����30������Ϊʹ�õ���vector�����Ļ������Զ���չ����
		for (int i = 0; i < HISTO_LENGTH; i++)
			rotHist[i].reserve(30);

		//! ԭ���ߴ����� const float factor = 1.0f/HISTO_LENGTH; �Ǵ���ģ�����Ϊ�������   
		const float factor = HISTO_LENGTH / 360.0f;

		// ƥ���Ծ��룬ע���ǰ���F2��������Ŀ����ռ�
		std::vector<int> vMatchedDistance(F2->mvKeys.size(), INT_MAX);
		// ��֡2��֡1�ķ���ƥ�䣬ע���ǰ���F2��������Ŀ����ռ�
		std::vector<int> vnMatches21(F2->mvKeys.size(), -1);

		// ����֡1�е�����������
		for (size_t i1 = 0, iend1 = F1->mvKeys.size(); i1 < iend1; i1++)
		{
			cv::KeyPoint kp1 = F1->mvKeys[i1];
			int level1 = kp1.octave;

			// vbPrevMatched ������ǲο�֡ F1��������
			// windowSize = 100�����������С�������㼶 ��Ϊ0
			std::vector<size_t> vIndices2 = GetFeaturesInArea(F2, F1->mvKeys[i1].pt.x, F1->mvKeys[i1].pt.y, windowSize, level1 - 1, level1 + 1);
			  
			// û�к�ѡ�����㣬����
			if (vIndices2.empty())
				continue;

			// ȡ���ο�֡F1�е�ǰ�����������Ӧ��������
			cv::Mat d1 = F1->mDescriptors.row(i1);

			int bestDist = INT_MAX;     //���������ƥ����룬ԽСԽ��
			int bestDist2 = INT_MAX;    //�μ�������ƥ�����
			int bestIdx2 = -1;          //��Ѻ�ѡ��������F2�е�index

			// Step 3 �����������������е�����Ǳ�ڵ�ƥ���ѡ�㣬�ҵ����ŵĺʹ��ŵ�
			for (auto& vit: vIndices2)
			{
				size_t i2 = vit;
				// ȡ����ѡ�������Ӧ��������
				cv::Mat d2 = F2->mDescriptors.row(i2);
				// �������������������Ӿ���
				int dist = DescriptorDistance(d1, d2);

				// �Ѿ�ƥ����ˣ���һλ
				if (vMatchedDistance[i2] <= dist)
					continue;
				// �����ǰƥ������С��������ѴμѾ���
				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestIdx2 = i2;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}

			// Step 4 �����Ŵ��Ž�����м�飬������ֵ������/���ű�����ɾ���ظ�ƥ��
			// ��ʹ��������������ƥ����룬Ҳ��һ����֤��Գɹ���ҪС���趨��ֵ
			if (bestDist <= TH_LOW)
			{
				// ��Ѿ���ȴμѾ���ҪС���趨�ı����������������ʶ�ȸ���
				if (bestDist < (float)bestDist2*mfNNratio)
				{
					// ����ҵ��ĺ�ѡ�������ӦF1���������Ѿ�ƥ����ˣ�˵���������ظ�ƥ�䣬��ԭ����ƥ��Ҳɾ��
					if (vnMatches21[bestIdx2] >= 0)
					{
						vnMatches12[vnMatches21[bestIdx2]] = -1;
						nmatches--;
					}
					// ���ŵ�ƥ���ϵ��˫����
					// vnMatches12����ο�֡F1��F2ƥ���ϵ��index������F1��Ӧ������������ֵ�������ƥ��õ�F2����������
					vnMatches12[i1] = bestIdx2;
					vnMatches21[bestIdx2] = i1;
					vMatchedDistance[bestIdx2] = bestDist;
					nmatches++;

					// Step 5 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
					if (mbCheckOrientation)
					{
						// ����ƥ��������ĽǶȲ���ﵥλ�ǽǶȡ㣬���ǻ���
						float rot = F1->mvKeys[i1].angle - F2->mvKeys[bestIdx2].angle;
						if (rot < 0.0)
							rot += 360.0f;
						// ǰ��factor = HISTO_LENGTH/360.0f 
						// bin = rot / 360.of * HISTO_LENGTH ��ʾ��ǰrot�������ڵڼ���ֱ��ͼbin  
						int bin = std::floor(rot*factor);
						// ���bin ��������һ���ֻ�
						if (bin == HISTO_LENGTH)
							bin = 0;
						assert(bin >= 0 && bin < HISTO_LENGTH);
						rotHist[bin].push_back(i1);
					}
				}
			}

		}

		// Step 6 ɸ����תֱ��ͼ�С�������������
		if (mbCheckOrientation)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;
			// ɸѡ������ת�ǶȲ�������ֱ��ͼ��������������ǰ����bin������
			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				// �޳�������ǰ����ƥ��ԣ���Ϊ���ǲ����ϡ�������ת����    
				for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
				{
					int idx1 = rotHist[i][j];
					if (vnMatches12[idx1] >= 0)
					{
						vnMatches12[idx1] = -1;
						nmatches--;
					}
				}
			}
		}

		return nmatches;
	}

	/**
	 * @brief �ҵ��� ��x,yΪ����,�뾶Ϊr��Բ�����ҽ������㼶��[minLevel, maxLevel]��������
	 *
	 * @param[in] x                     ����������x
	 * @param[in] y                     ����������y
	 * @param[in] r                     �����뾶
	 * @param[in] minLevel              ��С�������㼶
	 * @param[in] maxLevel              ���������㼶
	 * @return vector<size_t>           �����������ĺ�ѡƥ���id
	 */
	std::vector<size_t> ORBmatcher::GetFeaturesInArea(Frame* F, const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel)
	{
		// �洢���������vector
		std::vector<size_t> vIndices;

		int N = F->mvKeys.size();

		// Step 1 ����뾶ΪrԲ�������±߽����ڵ������к��е�id
		// ���Ұ뾶Ϊr��Բ���߽��������������ꡣ����ط��е��ƣ���������£�
		// (mnMaxX-mnMinX)/FRAME_GRID_COLS����ʾ�з���ÿ���������ƽ���ֵü������أ��϶�����1��
		// mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) �����浹������ʾÿ�����ؿ��Ծ��ּ��������У��϶�С��1��
		// (x-mnMinX-r)�����Կ����Ǵ�ͼ�����߽�mnMinX���뾶r��Բ����߽�����ռ����������
		// ������ˣ���������Ǹ��뾶Ϊr��Բ�����߽����ĸ���������
		// ��֤nMinCellX ������ڵ���0
		const int nMinCellX = std::max(0, (int)floor((x - F->mnMinX - r)*F->mfGridElementWidthInv));

		// ���������õ�Բ����߽����ڵ������г������趨�����ޣ���ô��˵����������Ҳ�������Ҫ��������㣬���ؿ�vector
		if (nMinCellX >= FRAME_GRID_COLS)
			return vIndices;

		// ����Բ���ڵ��ұ߽�����������
		const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - F->mnMinX + r)*F->mfGridElementWidthInv));
		// ����������Բ�ұ߽����ڵ����񲻺Ϸ���˵���������㲻�ã�ֱ�ӷ��ؿ�vector
		if (nMaxCellX < 0)
			return vIndices;

		//����Ĳ���Ҳ�������Ƶģ���������Բ���±߽����ڵ������е�id
		const int nMinCellY = std::max(0, (int)floor((y - F->mnMinY - r)*F->mfGridElementHeightInv));
		if (nMinCellY >= FRAME_GRID_ROWS)
			return vIndices;

		const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - F->mnMinY + r)*F->mfGridElementHeightInv));
		if (nMaxCellY < 0)
			return vIndices;

		// Step 2 ����Բ�������ڵ���������Ѱ�����������ĺ�ѡ�����㣬������index�ŵ������
		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				// ��ȡ��������ڵ������������� Frame::mvKeysUn �е�����
				const std::vector<size_t> vCell = F->mGrid[ix][iy];
				// ������������û�������㣬��ô����������������һ��
				if (vCell.empty())
					continue;

				for (size_t i=0; i<vCell.size(); i++)
				{
					// ���������ȶ�ȡ��������� 
					const cv::KeyPoint &kpUn = F->mvKeys[vCell[i]];

					// ��֤���������ڽ������㼶minLevel��maxLevel֮�䣬���ǵĻ�����
					if (kpUn.octave < minLevel || kpUn.octave > maxLevel)
						continue;

					// ͨ����飬�����ѡ�����㵽Բ���ĵľ��룬�鿴�Ƿ��������Բ������֮��
					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					// ���x�����y����ľ��붼��ָ���İ뾶֮�ڣ��洢��indexΪ��ѡ������
					if (fabs(distx) < r && fabs(disty) < r)
						vIndices.push_back(vCell[i]);
				}
			}
		}
		return vIndices;
	}

	/**
	 * @brief ɸѡ������ת�ǶȲ�������ֱ��ͼ��������������ǰ����bin������
	 *
	 * @param[in] histo         ƥ�����������ת�����ֱ��ͼ
	 * @param[in] L             ֱ��ͼ�ߴ�
	 * @param[in & out] ind1          binֵ��һ���Ӧ������
	 * @param[in & out] ind2          binֵ�ڶ����Ӧ������
	 * @param[in & out] ind3          binֵ�������Ӧ������
	 */
	void ORBmatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
	{
		int max1 = 0;
		int max2 = 0;
		int max3 = 0;

		for (int i = 0; i < L; i++)
		{
			const int s = histo[i].size();
			if (s > max1)
			{
				max3 = max2;
				max2 = max1;
				max1 = s;
				ind3 = ind2;
				ind2 = ind1;
				ind1 = i;
			}
			else if (s > max2)
			{
				max3 = max2;
				max2 = s;
				ind3 = ind2;
				ind2 = i;
			}
			else if (s > max3)
			{
				max3 = s;
				ind3 = i;
			}
		}

		// ������̫����,˵�����ŵķǳ�����,��������Է�����,����Ϊ-1
		if (max2 < 0.1f*(float)max1)
		{
			ind2 = -1;
			ind3 = -1;
		}
		else if (max3 < 0.1f*(float)max1)
		{
			ind3 = -1;
		}
	}

	// Bit set count operation from
	// Hamming distance�����������ƴ�֮��ĺ������룬ָ�����䲻ͬλ���ĸ���
	// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
	{
		const int *pa = a.ptr<int32_t>();
		const int *pb = b.ptr<int32_t>();

		int dist = 0;
		// 8*32=256bit
		for (int i = 0; i < 8; i++, pa++, pb++)
		{
			unsigned  int v = *pa ^ *pb;        // ���Ϊ0,����Ϊ1
			// ����Ĳ������Ǽ�������bitΪ1�ĸ�����,�����������������Ӿͺ�
			// ��ʵ�Ҿ���Ҳ������ֱ��ʹ��8bit�Ĳ��ұ�,Ȼ����32��Ѱַ�����������;����ȱ����û�����ú�CPU���ֳ�
			v = v - ((v >> 1) & 0x55555555);
			v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
			dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
		}
		return dist;
	}
}