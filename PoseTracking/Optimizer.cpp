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
	int Optimizer::PoseOptimization(Tracking* pTrack, Frame *pFrame)
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

				e->fx = pTrack->mK.at<float>(0, 0);
				e->fy = pTrack->mK.at<float>(1, 1);
				e->cx = pTrack->mK.at<float>(0, 2);
				e->cy = pTrack->mK.at<float>(1, 2);

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



}