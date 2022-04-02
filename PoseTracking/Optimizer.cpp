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
	int Optimizer::PoseOptimization(Frame *pFrame)
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

				e->fx = mK.at<float>(0, 0);
				e->fy = mK.at<float>(1, 1);
				e->cx = mK.at<float>(0, 2);
				e->cy = mK.at<float>(1, 2);

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

	/*
	 * @brief Local Bundle Adjustment
	 *
	 * 1. Vertex:
	 *     - g2o::VertexSE3Expmap()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
	 *     - g2o::VertexSE3Expmap()，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalKeyFrames）的位姿，在优化中这些关键帧的位姿不变
	 *     - g2o::VertexSBAPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
	 *         + Vertex：关键帧的Tcw，MapPoint的Pw
	 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
	 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
	 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
	 *         + Vertex：关键帧的Tcw，MapPoint的Pw
	 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
	 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
	 *
	 * @param pKF        KeyFrame
	 * @param pbStopFlag 是否停止优化的标志
	 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
	 * @note 由局部建图线程调用,对局部地图进行优化的函数
	 */
	void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
	{
		// 该优化函数用于LocalMapping线程的局部BA优化
		// 局部关键帧
		std::list<KeyFrame*> lLocalKeyFrames;

		// Step 1 将当前关键帧及其共视关键帧加入局部关键帧
		lLocalKeyFrames.push_back(pKF);
		pKF->mnBALocalForKF = pKF->mnId;

		// 找到关键帧连接的共视关键帧（一级相连），加入局部关键帧中
		const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
		for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
		{
			KeyFrame* pKFi = vNeighKFs[i];

			// 把参与局部BA的每一个关键帧的 mnBALocalForKF设置为当前关键帧的mnId，防止重复添加
			pKFi->mnBALocalForKF = pKF->mnId;

			// 保证该关键帧有效才能加入
			if (!pKFi->isBad())
				lLocalKeyFrames.push_back(pKFi);
		}

		// Step 2 遍历局部关键帧中(一级相连)关键帧，将它们观测的地图点加入到局部地图点
		std::list<MapPoint*> lLocalMapPoints;
		// 遍历局部关键帧中的每一个关键帧
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			// 取出该关键帧对应的地图点
			std::vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
			// 遍历这个关键帧观测到的每一个地图点，加入到局部地图点
			for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
			{
				MapPoint* pMP = *vit;
				if (pMP)
				{
					if (!pMP->isBad())   //保证地图点有效
						// 把参与局部BA的每一个地图点的 mnBALocalForKF设置为当前关键帧的mnId
						// mnBALocalForKF 是为了防止重复添加
						if (pMP->mnBALocalForKF != pKF->mnId)
						{
							lLocalMapPoints.push_back(pMP);
							pMP->mnBALocalForKF = pKF->mnId;
						}
				}   // 判断这个地图点是否靠谱
			} // 遍历这个关键帧观测到的每一个地图点
		} // 遍历 lLocalKeyFrames 中的每一个关键帧

			// Step 3 得到能被局部地图点观测到，但不属于局部关键帧的关键帧(二级相连)，这些二级相连关键帧在局部BA优化时不优化
		std::list<KeyFrame*> lFixedCameras;
		// 遍历局部地图中的每个地图点
		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			// 观测到该地图点的KF和该地图点在KF中的索引
			std::map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			// 遍历所有观测到该地图点的关键帧
			for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;

				// pKFi->mnBALocalForKF!=pKF->mnId 表示不属于局部关键帧，
				// pKFi->mnBAFixedForKF!=pKF->mnId 表示还未标记为fixed（固定的）关键帧
				if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
				{
					// 将局部地图点能观测到的、但是不属于局部BA范围的关键帧的mnBAFixedForKF标记为pKF（触发局部BA的当前关键帧）的mnId
					pKFi->mnBAFixedForKF = pKF->mnId;
					if (!pKFi->isBad())
						lFixedCameras.push_back(pKFi);
				}
			}
		}

		// Setup optimizer
		// Step 4 构造g2o优化器
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
		// LM大法好
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		// 可能在 Tracking::NeedNewKeyFrame() 里置位
		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		// 记录参与局部BA的最大关键帧mnId
		unsigned long maxKFid = 0;

		// Step 5 添加待优化的位姿顶点：局部关键帧的位姿
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			// 设置初始优化位姿
			vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			// 如果是初始关键帧，要锁住位姿不优化
			vSE3->setFixed(pKFi->mnId == 0);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Step  6 添加不优化的位姿顶点：固定关键帧的位姿，注意这里调用了vSE3->setFixed(true)
		// 不优化为啥也要添加？回答：为了增加约束信息
		for (std::list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			// 所有的这些顶点的位姿都不优化，只是为了增加约束项
			vSE3->setFixed(true);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Step  7 添加待优化的局部地图点顶点
		// 边的最大数目 = 位姿数目 * 地图点数目
		const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

		std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
		vpEdgesMono.reserve(nExpectedSize);

		std::vector<KeyFrame*> vpEdgeKFMono;
		vpEdgeKFMono.reserve(nExpectedSize);

		std::vector<MapPoint*> vpMapPointEdgeMono;
		vpMapPointEdgeMono.reserve(nExpectedSize);

		// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
		const float thHuberMono = sqrt(5.991);
		// 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815
		const float thHuberStereo = sqrt(7.815);

		// 遍历所有的局部地图点
		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			// 添加顶点：MapPoint
			MapPoint* pMP = *lit;
			g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
			vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

			// 前面记录maxKFid的作用在这里体现
			int id = pMP->mnId + maxKFid + 1;
			vPoint->setId(id);

			// 因为使用了LinearSolverType，所以需要将所有的三维点边缘化掉
			vPoint->setMarginalized(true);
			optimizer.addVertex(vPoint);

			// 观测到该地图点的KF和该地图点在KF中的索引
			const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			// Step 8 在添加完了一个地图点之后, 对每一对关联的地图点和关键帧构建边
			// 遍历所有观测到当前地图点的关键帧
			for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;
				if (!pKFi->isBad())
				{
					const cv::KeyPoint &kpUn = pKFi->mvKeys[mit->second];

					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
					// 边的第一个顶点是地图点
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
					// 边的第一个顶点是观测到该地图点的关键帧
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);

					// 权重为特征点所在图像金字塔的层数的倒数
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

					// 使用鲁棒核函数抑制外点
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					e->fx = mK.at<float>(0, 0);
					e->fy = mK.at<float>(1, 1);
					e->cx = mK.at<float>(0, 2);
					e->cy = mK.at<float>(1, 2);

					// 将边添加到优化器，记录边、边连接的关键帧、边连接的地图点信息
					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);
				}
			}
		}

		// 开始BA前再次确认是否有外部请求停止优化，因为这个变量是引用传递，会随外部变化
		// 可能在 Tracking::NeedNewKeyFrame(), mpLocalMapper->InsertKeyFrame 里置位
		if (pbStopFlag)
			if (*pbStopFlag)
				return;

		// Step 9 分成两个阶段开始优化。
		// 第一阶段优化
		optimizer.initializeOptimization();
		// 迭代5次
		optimizer.optimize(5);

		bool bDoMore = true;
		// 检查是否外部请求停止
		if (pbStopFlag)
			if (*pbStopFlag)
				bDoMore = false;

		// 如果有外部请求停止,那么就不在进行第二阶段的优化
		if (bDoMore)
		{
			// Step 10 检测outlier，并设置下次不优化
			// 遍历所有的单目误差边
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
				MapPoint* pMP = vpMapPointEdgeMono[i];

				if (pMP->isBad())
					continue;

				// 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
				// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
				// 如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，不优化了。
				if (e->chi2() > 5.991 || !e->isDepthPositive())
				{
					// 不优化
					e->setLevel(1);
				}
				// 第二阶段优化的时候就属于精求解了,所以就不使用核函数
				e->setRobustKernel(0);
			}

			// Step 11：排除误差较大的outlier后再次优化 -- 第二阶段优化
			optimizer.initializeOptimization(0);
			optimizer.optimize(10);
		}

		std::vector<std::pair<KeyFrame*, MapPoint*> > vToErase;
		vToErase.reserve(vpEdgesMono.size());

		// Step 12：在优化后重新计算误差，剔除连接误差比较大的关键帧和地图点
		// 对于单目误差边
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		{
			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
			MapPoint* pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad())
				continue;

			// 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
			// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
			// 如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，要删掉了
			if (e->chi2() > 5.991 || !e->isDepthPositive())
			{
				// outlier
				KeyFrame* pKFi = vpEdgeKFMono[i];
				vToErase.push_back(std::make_pair(pKFi, pMP));
			}
		}

		// Get Map Mutex
		std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

		// 删除点
		// 连接偏差比较大，在关键帧中剔除对该地图点的观测
		// 连接偏差比较大，在地图点中剔除对该关键帧的观测
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

		// Step 13：优化后更新关键帧位姿以及地图点的位置、平均观测方向等属性
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