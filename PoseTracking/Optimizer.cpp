#include "Optimizer.h"

namespace PoseTracking
{
	/*
	 * @brief Pose Only Optimization
	 *
	 * 3D-2D ��С����ͶӰ��� e = (u,v) - project(Tcw*Pw) \n
	 * ֻ�Ż�Frame��Tcw�����Ż�MapPoints������
	 *
	 * 1. Vertex: g2o::VertexSE3Expmap()������ǰ֡��Tcw
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZOnlyPose()��BaseUnaryEdge
	 *         + Vertex�����Ż���ǰ֡��Tcw
	 *         + measurement��MapPoint�ڵ�ǰ֡�еĶ�άλ��(u,v)
	 *         + InfoMatrix: invSigma2(�����������ڵĳ߶��й�)
	 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()��BaseUnaryEdge
	 *         + Vertex�����Ż���ǰ֡��Tcw
	 *         + measurement��MapPoint�ڵ�ǰ֡�еĶ�άλ��(ul,v,ur)
	 *         + InfoMatrix: invSigma2(�����������ڵĳ߶��й�)
	 *
	 * @param   pFrame Frame
	 * @return  inliers����
	 */
	int Optimizer::PoseOptimization(Frame *pFrame)
	{
		// ���Ż�������Ҫ����Tracking�߳��У��˶����١��ο�֡���١���ͼ���١��ض�λ
		// Step 1������g2o�Ż���, BlockSolver_6_3��ʾ��λ�� _PoseDim Ϊ6ά��·��� _LandmarkDim ��3ά
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		// �����֡��,��Ч��,�����Ż����̵�2D-3D���
		int nInitialCorrespondences = 0;

		// Step 2����Ӷ��㣺���Ż���ǰ֡��Tcw
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));

		// ����id
		vSE3->setId(0);
		// Ҫ�Ż��ı��������Բ��̶ܹ�
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		int N = pFrame->mvpMapPoints.size();
		// for Monocular
		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);

		// ���ɶ�Ϊ2�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ5.991
		const float deltaMono = sqrt(5.991);
		// ���ɶ�Ϊ3�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ7.815   
		const float deltaStereo = sqrt(7.815);

		for (int i=0; i<N; i++)
		{
			MapPoint* pMP = pFrame->mvpMapPoints[i];
			if (pMP)
			{
				nInitialCorrespondences++;
				pFrame->mvbOutlier[i] = false;

				// �������ͼ��Ĺ۲�
				Eigen::Matrix<double, 2, 1> obs;
				const cv::KeyPoint &kpUn = pFrame->mvKeys[i];
				obs << kpUn.pt.x, kpUn.pt.y;

				// �½���Ŀ�ıߣ�һԪ�ߣ����Ϊ�۲������������ȥͶӰ�������
				g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
				// ���ñߵĶ���
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
				e->setMeasurement(obs);

				// �����Ŀ��ų̶Ⱥ����������ڵ�ͼ���й�
				const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				// ������ʹ����³���˺���
				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(deltaMono);    // ǰ���ᵽ���Ŀ�����ֵ

				e->fx = mK.at<float>(0, 0);
				e->fy = mK.at<float>(1, 1);
				e->cx = mK.at<float>(0, 2);
				e->cy = mK.at<float>(1, 2);

				// ��ͼ��Ŀռ�λ��,��Ϊ�����ĳ�ʼֵ
				cv::Mat Xw = pMP->GetWorldPos();
				e->Xw[0] = Xw.at<float>(0);
				e->Xw[1] = Xw.at<float>(1);
				e->Xw[2] = Xw.at<float>(2);

				optimizer.addEdge(e);

				vpEdgesMono.push_back(e);
				vnIndexEdgeMono.push_back(i);
			}
		}

		// ���û���㹻��ƥ���,��ô��ֻ�÷�����
		if (nInitialCorrespondences < 3)
			return 0;

		// Step 4����ʼ�Ż����ܹ��Ż��ĴΣ�ÿ���Ż�����10��,ÿ���Ż��󣬽��۲��Ϊoutlier��inlier��outlier�������´��Ż�
		// ����ÿ���Ż����Ƕ����еĹ۲����outlier��inlier�б����֮ǰ���б�Ϊoutlier�п��ܱ��inlier����֮��Ȼ
		// ���ڿ���������������ֵ�����������һ�����ص�ƫ�
		const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };          // ��Ŀ
		const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };       // ˫Ŀ
		const int its[4] = { 10,10,10,10 };// �Ĵε�����ÿ�ε����Ĵ���

		// bad �ĵ�ͼ�����
		int nBad = 0;
		// һ�������Ĵ��Ż�
		for (size_t it = 0; it < 4; it++)
		{
			// ��ʵ���ǳ�ʼ���Ż���,����Ĳ���0�����ǲ���д,Ĭ��Ҳ��0,Ҳ����ֻ��levelΪ0�ı߽����Ż�
			optimizer.initializeOptimization(0);
			// ��ʼ�Ż����Ż�10��
			optimizer.optimize(its[it]);

			nBad = 0;
			// �Ż�����,��ʼ���������Ż���ÿһ������(��Ŀ)
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

				const size_t idx = vnIndexEdgeMono[i];

				// �������������������outlier
				if (pFrame->mvbOutlier[idx])
				{
					e->computeError();
				}

				// ����error*\Omega*error,����������������С(�������Ŷ��Ժ�)
				const float chi2 = e->chi2();

				if (chi2 > chi2Mono[it])
				{
					pFrame->mvbOutlier[idx] = true;
					e->setLevel(1);                 // ����Ϊoutlier , level 1 ��ӦΪ���,����Ĺ���������������Ϊ���Ż�
					nBad++;
				}
				else
				{
					pFrame->mvbOutlier[idx] = false;
					e->setLevel(0);                 // ����Ϊinlier, level 0 ��ӦΪ�ڵ�,����Ĺ��������Ǿ���Ҫ�Ż���Щ��ϵ
				}

				if (it == 2)
					e->setRobustKernel(0); // ����ǰ�����Ż���ҪRobustKernel����, ������Ż�������Ҫ -- ��Ϊ��ͶӰ������Ѿ������Ե��½���
			}// �Ե�Ŀ���ߵĴ���
		} // һ��Ҫ�����Ĵ��Ż�

		// Step 5 �õ��Ż���ĵ�ǰ֡��λ��
		g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = Converter::toCvMat(SE3quat_recov);
		pFrame->SetPose(pose);

		// ���ҷ����ڵ���Ŀ
		return nInitialCorrespondences - nBad;
	}

	/*
	 * @brief Local Bundle Adjustment
	 *
	 * 1. Vertex:
	 *     - g2o::VertexSE3Expmap()��LocalKeyFrames������ǰ�ؼ�֡��λ�ˡ��뵱ǰ�ؼ�֡�����Ĺؼ�֡��λ��
	 *     - g2o::VertexSE3Expmap()��FixedCameras�����ܹ۲⵽LocalMapPoints�Ĺؼ�֡�����Ҳ�����LocalKeyFrames����λ�ˣ����Ż�����Щ�ؼ�֡��λ�˲���
	 *     - g2o::VertexSBAPointXYZ()��LocalMapPoints����LocalKeyFrames�ܹ۲⵽������MapPoints��λ��
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZ()��BaseBinaryEdge
	 *         + Vertex���ؼ�֡��Tcw��MapPoint��Pw
	 *         + measurement��MapPoint�ڹؼ�֡�еĶ�άλ��(u,v)
	 *         + InfoMatrix: invSigma2(�����������ڵĳ߶��й�)
	 *     - g2o::EdgeStereoSE3ProjectXYZ()��BaseBinaryEdge
	 *         + Vertex���ؼ�֡��Tcw��MapPoint��Pw
	 *         + measurement��MapPoint�ڹؼ�֡�еĶ�άλ��(ul,v,ur)
	 *         + InfoMatrix: invSigma2(�����������ڵĳ߶��й�)
	 *
	 * @param pKF        KeyFrame
	 * @param pbStopFlag �Ƿ�ֹͣ�Ż��ı�־
	 * @param pMap       ���Ż��󣬸���״̬ʱ��Ҫ�õ�Map�Ļ�����mMutexMapUpdate
	 * @note �ɾֲ���ͼ�̵߳���,�Ծֲ���ͼ�����Ż��ĺ���
	 */
	void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
	{
		// ���Ż���������LocalMapping�̵߳ľֲ�BA�Ż�
		// �ֲ��ؼ�֡
		std::list<KeyFrame*> lLocalKeyFrames;

		// Step 1 ����ǰ�ؼ�֡���乲�ӹؼ�֡����ֲ��ؼ�֡
		lLocalKeyFrames.push_back(pKF);
		pKF->mnBALocalForKF = pKF->mnId;

		// �ҵ��ؼ�֡���ӵĹ��ӹؼ�֡��һ��������������ֲ��ؼ�֡��
		const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
		for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
		{
			KeyFrame* pKFi = vNeighKFs[i];

			// �Ѳ���ֲ�BA��ÿһ���ؼ�֡�� mnBALocalForKF����Ϊ��ǰ�ؼ�֡��mnId����ֹ�ظ����
			pKFi->mnBALocalForKF = pKF->mnId;

			// ��֤�ùؼ�֡��Ч���ܼ���
			if (!pKFi->isBad())
				lLocalKeyFrames.push_back(pKFi);
		}

		// Step 2 �����ֲ��ؼ�֡��(һ������)�ؼ�֡�������ǹ۲�ĵ�ͼ����뵽�ֲ���ͼ��
		std::list<MapPoint*> lLocalMapPoints;
		// �����ֲ��ؼ�֡�е�ÿһ���ؼ�֡
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			// ȡ���ùؼ�֡��Ӧ�ĵ�ͼ��
			std::vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
			// ��������ؼ�֡�۲⵽��ÿһ����ͼ�㣬���뵽�ֲ���ͼ��
			for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
			{
				MapPoint* pMP = *vit;
				if (pMP)
				{
					if (!pMP->isBad())   //��֤��ͼ����Ч
						// �Ѳ���ֲ�BA��ÿһ����ͼ��� mnBALocalForKF����Ϊ��ǰ�ؼ�֡��mnId
						// mnBALocalForKF ��Ϊ�˷�ֹ�ظ����
						if (pMP->mnBALocalForKF != pKF->mnId)
						{
							lLocalMapPoints.push_back(pMP);
							pMP->mnBALocalForKF = pKF->mnId;
						}
				}   // �ж������ͼ���Ƿ���
			} // ��������ؼ�֡�۲⵽��ÿһ����ͼ��
		} // ���� lLocalKeyFrames �е�ÿһ���ؼ�֡

			// Step 3 �õ��ܱ��ֲ���ͼ��۲⵽���������ھֲ��ؼ�֡�Ĺؼ�֡(��������)����Щ���������ؼ�֡�ھֲ�BA�Ż�ʱ���Ż�
		std::list<KeyFrame*> lFixedCameras;
		// �����ֲ���ͼ�е�ÿ����ͼ��
		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			// �۲⵽�õ�ͼ���KF�͸õ�ͼ����KF�е�����
			std::map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			// �������й۲⵽�õ�ͼ��Ĺؼ�֡
			for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;

				// pKFi->mnBALocalForKF!=pKF->mnId ��ʾ�����ھֲ��ؼ�֡��
				// pKFi->mnBAFixedForKF!=pKF->mnId ��ʾ��δ���Ϊfixed���̶��ģ��ؼ�֡
				if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
				{
					// ���ֲ���ͼ���ܹ۲⵽�ġ����ǲ����ھֲ�BA��Χ�Ĺؼ�֡��mnBAFixedForKF���ΪpKF�������ֲ�BA�ĵ�ǰ�ؼ�֡����mnId
					pKFi->mnBAFixedForKF = pKF->mnId;
					if (!pKFi->isBad())
						lFixedCameras.push_back(pKFi);
				}
			}
		}

		// Setup optimizer
		// Step 4 ����g2o�Ż���
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
		// LM�󷨺�
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		// ������ Tracking::NeedNewKeyFrame() ����λ
		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		// ��¼����ֲ�BA�����ؼ�֡mnId
		unsigned long maxKFid = 0;

		// Step 5 ��Ӵ��Ż���λ�˶��㣺�ֲ��ؼ�֡��λ��
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			// ���ó�ʼ�Ż�λ��
			vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			// ����ǳ�ʼ�ؼ�֡��Ҫ��סλ�˲��Ż�
			vSE3->setFixed(pKFi->mnId == 0);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Step  6 ��Ӳ��Ż���λ�˶��㣺�̶��ؼ�֡��λ�ˣ�ע�����������vSE3->setFixed(true)
		// ���Ż�ΪɶҲҪ��ӣ��ش�Ϊ������Լ����Ϣ
		for (std::list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			// ���е���Щ�����λ�˶����Ż���ֻ��Ϊ������Լ����
			vSE3->setFixed(true);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Step  7 ��Ӵ��Ż��ľֲ���ͼ�㶥��
		// �ߵ������Ŀ = λ����Ŀ * ��ͼ����Ŀ
		const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

		std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
		vpEdgesMono.reserve(nExpectedSize);

		std::vector<KeyFrame*> vpEdgeKFMono;
		vpEdgeKFMono.reserve(nExpectedSize);

		std::vector<MapPoint*> vpMapPointEdgeMono;
		vpMapPointEdgeMono.reserve(nExpectedSize);

		// ���ɶ�Ϊ2�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ5.991
		const float thHuberMono = sqrt(5.991);
		// ���ɶ�Ϊ3�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ7.815
		const float thHuberStereo = sqrt(7.815);

		// �������еľֲ���ͼ��
		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			// ��Ӷ��㣺MapPoint
			MapPoint* pMP = *lit;
			g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
			vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

			// ǰ���¼maxKFid����������������
			int id = pMP->mnId + maxKFid + 1;
			vPoint->setId(id);

			// ��Ϊʹ����LinearSolverType��������Ҫ�����е���ά���Ե����
			vPoint->setMarginalized(true);
			optimizer.addVertex(vPoint);

			// �۲⵽�õ�ͼ���KF�͸õ�ͼ����KF�е�����
			const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			// Step 8 ���������һ����ͼ��֮��, ��ÿһ�Թ����ĵ�ͼ��͹ؼ�֡������
			// �������й۲⵽��ǰ��ͼ��Ĺؼ�֡
			for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;
				if (!pKFi->isBad())
				{
					const cv::KeyPoint &kpUn = pKFi->mvKeys[mit->second];

					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
					// �ߵĵ�һ�������ǵ�ͼ��
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
					// �ߵĵ�һ�������ǹ۲⵽�õ�ͼ��Ĺؼ�֡
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);

					// Ȩ��Ϊ����������ͼ��������Ĳ����ĵ���
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

					// ʹ��³���˺����������
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					e->fx = mK.at<float>(0, 0);
					e->fy = mK.at<float>(1, 1);
					e->cx = mK.at<float>(0, 2);
					e->cy = mK.at<float>(1, 2);

					// ������ӵ��Ż�������¼�ߡ������ӵĹؼ�֡�������ӵĵ�ͼ����Ϣ
					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);
				}
			}
		}

		// ��ʼBAǰ�ٴ�ȷ���Ƿ����ⲿ����ֹͣ�Ż�����Ϊ������������ô��ݣ������ⲿ�仯
		// ������ Tracking::NeedNewKeyFrame(), mpLocalMapper->InsertKeyFrame ����λ
		if (pbStopFlag)
			if (*pbStopFlag)
				return;

		// Step 9 �ֳ������׶ο�ʼ�Ż���
		// ��һ�׶��Ż�
		optimizer.initializeOptimization();
		// ����5��
		optimizer.optimize(5);

		bool bDoMore = true;
		// ����Ƿ��ⲿ����ֹͣ
		if (pbStopFlag)
			if (*pbStopFlag)
				bDoMore = false;

		// ������ⲿ����ֹͣ,��ô�Ͳ��ڽ��еڶ��׶ε��Ż�
		if (bDoMore)
		{
			// Step 10 ���outlier���������´β��Ż�
			// �������еĵ�Ŀ����
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
				MapPoint* pMP = vpMapPointEdgeMono[i];

				if (pMP->isBad())
					continue;

				// ���ڿ���������������ֵ�����������һ�����ص�ƫ�
				// ���ɶ�Ϊ2�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ5.991
				// ��� ��ǰ��������ֵ�����߱����ӵĵ�ͼ�����ֵΪ����˵������������⣬���Ż��ˡ�
				if (e->chi2() > 5.991 || !e->isDepthPositive())
				{
					// ���Ż�
					e->setLevel(1);
				}
				// �ڶ��׶��Ż���ʱ������ھ������,���ԾͲ�ʹ�ú˺���
				e->setRobustKernel(0);
			}

			// Step 11���ų����ϴ��outlier���ٴ��Ż� -- �ڶ��׶��Ż�
			optimizer.initializeOptimization(0);
			optimizer.optimize(10);
		}

		std::vector<std::pair<KeyFrame*, MapPoint*> > vToErase;
		vToErase.reserve(vpEdgesMono.size());

		// Step 12�����Ż������¼������޳��������Ƚϴ�Ĺؼ�֡�͵�ͼ��
		// ���ڵ�Ŀ����
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		{
			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
			MapPoint* pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad())
				continue;

			// ���ڿ���������������ֵ�����������һ�����ص�ƫ�
			// ���ɶ�Ϊ2�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ5.991
			// ��� ��ǰ��������ֵ�����߱����ӵĵ�ͼ�����ֵΪ����˵������������⣬Ҫɾ����
			if (e->chi2() > 5.991 || !e->isDepthPositive())
			{
				// outlier
				KeyFrame* pKFi = vpEdgeKFMono[i];
				vToErase.push_back(std::make_pair(pKFi, pMP));
			}
		}

		// Get Map Mutex
		std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

		// ɾ����
		// ����ƫ��Ƚϴ��ڹؼ�֡���޳��Ըõ�ͼ��Ĺ۲�
		// ����ƫ��Ƚϴ��ڵ�ͼ�����޳��Ըùؼ�֡�Ĺ۲�
		if (!vToErase.empty())
		{
			for (size_t i = 0; i < vToErase.size(); i++)
			{
				KeyFrame* pKFi = vToErase[i].first;
				MapPoint* pMPi = vToErase[i].second;
				pKFi->EraseMapPointMatch(pMPi);
				pMPi->EraseObservation(pKFi);
			}
		}

		// Step 13���Ż�����¹ؼ�֡λ���Լ���ͼ���λ�á�ƽ���۲ⷽ�������
		//Keyframes
		for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKF = *lit;
			g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
			g2o::SE3Quat SE3quat = vSE3->estimate();
			pKF->SetPose(Converter::toCvMat(SE3quat));
		}

		//Points
		for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			MapPoint* pMP = *lit;
			g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
			pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
			pMP->UpdateNormalAndDepth();
		}
	}






}