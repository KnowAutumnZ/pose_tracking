#include "Optimizer.h"

namespace PoseTracking
{
	/*
	 * @brief Pose Only Optimization
	 *
	 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
	 * 只优化Frame的Tcw，不优化MapPoints的坐标
	 *
	 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
	 *         + Vertex：待优化当前帧的Tcw
	 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
	 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
	 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
	 *         + Vertex：待优化当前帧的Tcw
	 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
	 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
	 *
	 * @param   pFrame Frame
	 * @return  inliers数量
	 */
	int Optimizer::PoseOptimization(Tracking* pTrack, Frame *pFrame)
	{
		// 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位
		// Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		// 输入的帧中,有效的,参与优化过程的2D-3D点对
		int nInitialCorrespondences = 0;

		// Step 2：添加顶点：待优化当前帧的Tcw
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));

		// 设置id
		vSE3->setId(0);
		// 要优化的变量，所以不能固定
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		int N = pFrame->mvpMapPoints.size();
		// for Monocular
		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);

		// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
		const float deltaMono = sqrt(5.991);
		// 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815   
		const float deltaStereo = sqrt(7.815);

		for (int i=0; i<N; i++)
		{
			MapPoint* pMP = pFrame->mvpMapPoints[i];
			if (pMP)
			{
				nInitialCorrespondences++;
				pFrame->mvbOutlier[i] = false;

				// 对这个地图点的观测
				Eigen::Matrix<double, 2, 1> obs;
				const cv::KeyPoint &kpUn = pFrame->mvKeys[i];
				obs << kpUn.pt.x, kpUn.pt.y;

				// 新建单目的边，一元边，误差为观测特征点坐标减去投影点的坐标
				g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
				// 设置边的顶点
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
				e->setMeasurement(obs);

				// 这个点的可信程度和特征点所在的图层有关
				const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				// 在这里使用了鲁棒核函数
				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(deltaMono);    // 前面提到过的卡方阈值

				e->fx = pTrack->mK.at<float>(0, 0);
				e->fy = pTrack->mK.at<float>(1, 1);
				e->cx = pTrack->mK.at<float>(0, 2);
				e->cy = pTrack->mK.at<float>(1, 2);

				// 地图点的空间位置,作为迭代的初始值
				cv::Mat Xw = pMP->GetWorldPos();
				e->Xw[0] = Xw.at<float>(0);
				e->Xw[1] = Xw.at<float>(1);
				e->Xw[2] = Xw.at<float>(2);

				optimizer.addEdge(e);

				vpEdgesMono.push_back(e);
				vnIndexEdgeMono.push_back(i);
			}
		}

		// 如果没有足够的匹配点,那么就只好放弃了
		if (nInitialCorrespondences < 3)
			return 0;

		// Step 4：开始优化，总共优化四次，每次优化迭代10次,每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
		// 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
		// 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
		const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };          // 单目
		const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };       // 双目
		const int its[4] = { 10,10,10,10 };// 四次迭代，每次迭代的次数

		// bad 的地图点个数
		int nBad = 0;
		// 一共进行四次优化
		for (size_t it = 0; it < 4; it++)
		{
			// 其实就是初始化优化器,这里的参数0就算是不填写,默认也是0,也就是只对level为0的边进行优化
			optimizer.initializeOptimization(0);
			// 开始优化，优化10次
			optimizer.optimize(its[it]);

			nBad = 0;
			// 优化结束,开始遍历参与优化的每一条误差边(单目)
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

				const size_t idx = vnIndexEdgeMono[i];

				// 如果这条误差边是来自于outlier
				if (pFrame->mvbOutlier[idx])
				{
					e->computeError();
				}

				// 就是error*\Omega*error,表征了这个点的误差大小(考虑置信度以后)
				const float chi2 = e->chi2();

				if (chi2 > chi2Mono[it])
				{
					pFrame->mvbOutlier[idx] = true;
					e->setLevel(1);                 // 设置为outlier , level 1 对应为外点,上面的过程中我们设置其为不优化
					nBad++;
				}
				else
				{
					pFrame->mvbOutlier[idx] = false;
					e->setLevel(0);                 // 设置为inlier, level 0 对应为内点,上面的过程中我们就是要优化这些关系
				}

				if (it == 2)
					e->setRobustKernel(0); // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要 -- 因为重投影的误差已经有明显的下降了
			}// 对单目误差边的处理
		} // 一共要进行四次优化

		// Step 5 得到优化后的当前帧的位姿
		g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = Converter::toCvMat(SE3quat_recov);
		pFrame->SetPose(pose);

		// 并且返回内点数目
		return nInitialCorrespondences - nBad;
	}



}