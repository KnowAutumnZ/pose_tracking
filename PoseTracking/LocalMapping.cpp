#include "LocalMapping.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

namespace PoseTracking
{
	// 构造函数
	LocalMapping::LocalMapping(Map *pMap, const float bMonocular) :
		mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
		mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
	{
		/*
		 * mbStopRequested：    外部线程调用，为true，表示外部线程请求停止 local mapping
		 * mbStopped：          为true表示可以并终止localmapping 线程
		 * mbNotStop：          true，表示不要停止 localmapping 线程，因为要插入关键帧了。需要和 mbStopped 结合使用
		 * mbAcceptKeyFrames：  true，允许接受关键帧。tracking 和local mapping 之间的关键帧调度
		 * mbAbortBA：          是否流产BA优化的标志位
		 * mbFinishRequested：  请求终止当前线程的标志。注意只是请求，不一定终止。终止要看 mbFinished
		 * mbResetRequested：   请求当前线程复位的标志。true，表示一直请求复位，但复位还未完成；表示复位完成为false
		 * mbFinished：         判断最终LocalMapping::Run() 是否完成的标志。
		 */
	}

	// 设置追踪线程句柄
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

	// 查看当前是否允许接受关键帧
	bool LocalMapping::AcceptKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexAccept);
		return mbAcceptKeyFrames;
	}

	// 设置"允许接受关键帧"的状态标志
	void LocalMapping::SetAcceptKeyFrames(bool flag)
	{
		std::unique_lock<std::mutex> lock(mMutexAccept);
		mbAcceptKeyFrames = flag;
	}

	// 终止BA
	void LocalMapping::InterruptBA()
	{
		mbAbortBA = true;
	}

	// 插入关键帧,由外部（Tracking）线程调用;这里只是插入到列表中,等待线程主函数对其进行处理
	void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		// 将关键帧插入到列表中
		mlNewKeyFrames.push_back(pKF);
		mbAbortBA = true;
	}

	// 查看列表中是否有等待被处理的关键帧,
	bool LocalMapping::CheckNewKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		return(!mlNewKeyFrames.empty());
	}

	// 线程主函数
	void LocalMapping::Run()
	{
		// 标记状态，表示当前run函数正在运行，尚未结束
		mbFinished = false;
		// 主循环
		while (1)
		{
			// Step 1 告诉Tracking，LocalMapping正处于繁忙状态，请不要给我发送关键帧打扰我
			// LocalMapping线程处理的关键帧都是Tracking线程发来的
			SetAcceptKeyFrames(false);

			// 等待处理的关键帧列表不为空
			if (CheckNewKeyFrames())
			{
				// Step 2 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
				ProcessNewKeyFrame();

				// Step 3 根据地图点的观测情况剔除质量不好的地图点
				MapPointCulling();

				// Step 4 当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
				CreateNewMapPoints();

				// 已经处理完队列中的最后的一个关键帧
				if (!CheckNewKeyFrames())
				{
					//  Step 5 检查并融合当前关键帧与相邻关键帧帧（两级相邻）中重复的地图点
					SearchInNeighbors();
				}

				// 终止BA的标志
				mbAbortBA = false;
				// 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
				if (!CheckNewKeyFrames())
				{
					// Local BA
					// Step 6 当局部地图中的关键帧大于2个的时候进行局部地图的BA
					if (mpMap->KeyFramesInMap() > 2)
						// 注意这里的第二个参数是按地址传递的,当这里的 mbAbortBA 状态发生变化时，能够及时执行/停止BA
						Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

					// Check redundant local Keyframes
					// Step 7 检测并剔除当前帧相邻的关键帧中冗余的关键帧
					// 冗余的判定：该关键帧的90%的地图点可以被其它关键帧观测到
					KeyFrameCulling();
				}
			}

			// Tracking will see that Local Mapping is not busy
			SetAcceptKeyFrames(true);
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
	}

	/**
	 * @brief 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
	 *
	 */
	void LocalMapping::ProcessNewKeyFrame()
	{
		// Step 1：从缓冲队列中取出一帧关键帧
		// 该关键帧队列是Tracking线程向LocalMapping中插入的关键帧组成
		{
			std::unique_lock<std::mutex> lock(mMutexNewKFs);
			// 取出列表中最前面的关键帧，作为当前要处理的关键帧
			mpCurrentKeyFrame = mlNewKeyFrames.front();
			// 取出最前面的关键帧后，在原来的列表里删掉该关键帧
			mlNewKeyFrames.pop_front();
		}

		// Step 3：当前处理关键帧中有效的地图点，更新normal，描述子等信息
		// TrackLocalMap中和当前帧新匹配上的地图点和当前关键帧进行关联绑定
		const std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
		// 对当前处理的这个关键帧中的所有的地图点展开遍历
		for (size_t i = 0; i < vpMapPointMatches.size(); i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
					{
						// 如果地图点不是来自当前帧的观测（比如来自局部地图点），为当前地图点添加观测
						pMP->AddObservation(mpCurrentKeyFrame, i);
						// 获得该点的平均观测方向和观测距离范围
						pMP->UpdateNormalAndDepth();
						// 更新地图点的最佳描述子
						pMP->ComputeDistinctiveDescriptors();
					}
					else // this can only happen for new stereo points inserted by the Tracking
					{
						// 如果当前帧中已经包含了这个地图点,但是这个地图点中却没有包含这个关键帧的信息
						// 这些地图点可能来自双目或RGBD跟踪过程中新生成的地图点，或者是CreateNewMapPoints 中通过三角化产生
						// 将上述地图点放入mlpRecentAddedMapPoints，等待后续MapPointCulling函数的检验
						mlpRecentAddedMapPoints.push_back(pMP);
					}
				}
			}
		}

		// Step 4：更新关键帧间的连接关系（共视图）
		mpCurrentKeyFrame->UpdateConnections();

		// Step 5：将该关键帧插入到地图中
		mpMap->AddKeyFrame(mpCurrentKeyFrame);
	}

	/**
	 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增的地图点
	 * mlpRecentAddedMapPoints：存储新增的地图点，这里是要删除其中不靠谱的
	 */
	void LocalMapping::MapPointCulling()
	{
		std::list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
		const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

		// Step 1：根据相机类型设置不同的观测阈值
		int nThObs;
		if (mbMonocular)
			nThObs = 2;
		else
			nThObs = 3;
		const int cnThObs = nThObs;

		// Step 2：遍历检查新添加的地图点
		while (lit != mlpRecentAddedMapPoints.end())
		{
			MapPoint* pMP = *lit;
			if (pMP->isBad())
			{
				// Step 2.1：已经是坏点的地图点仅从队列中删除
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if (pMP->GetFoundRatio() < 0.25f)
			{
				// Step 2.2：跟踪到该地图点的帧数相比预计可观测到该地图点的帧数的比例小于25%，从地图中删除
				// (mnFound/mnVisible） < 25%
				// mnFound ：地图点被多少帧（包括普通帧）看到，次数越多越好
				// mnVisible：地图点应该被看到的次数
				// (mnFound/mnVisible）：对于大FOV镜头这个比例会高，对于窄FOV镜头这个比例会低
				pMP->SetBadFlag();
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
			{
				// Step 2.3：从该点建立开始，到现在已经过了不小于2个关键帧
				// 但是观测到该点的相机数却不超过阈值cnThObs，从地图中删除
				pMP->SetBadFlag();
				lit = mlpRecentAddedMapPoints.erase(lit);
			}
			else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
				// Step 2.4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
				// 因此没有SetBadFlag()，仅从队列中删除
				lit = mlpRecentAddedMapPoints.erase(lit);
			else
				lit++;
		}
	}

	/**
	 * @brief 用当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
	 *
	 */
	void LocalMapping::CreateNewMapPoints()
	{
		// nn表示搜索最佳共视关键帧的数目
		// 不同传感器下要求不一样,单目的时候需要有更多的具有较好共视关系的关键帧来建立地图
		int nn = 10;
		if (mbMonocular)
			nn = 20;

		// Step 1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻关键帧
		const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

		// 特征点匹配配置 最佳距离 < 0.6*次佳距离，比较苛刻了。不检查旋转
		ORBmatcher matcher(0.6, false);

		// 取出当前帧从世界坐标系到相机坐标系的变换矩阵
		cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));

		// 得到当前关键帧（左目）光心在世界坐标系中的坐标、内参
		cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

		// mfScaleFactor = 1.2
		const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
		// 记录三角化成功的地图点数目
		int nnew = 0;
		// Step 2：遍历相邻关键帧，搜索匹配并用极线约束剔除误匹配，最终三角化
		for (size_t i = 0; i < vpNeighKFs.size(); i++)
		{
			KeyFrame* pKF2 = vpNeighKFs[i];
			// 相邻的关键帧光心在世界坐标系中的坐标
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			// 基线向量，两个关键帧间的相机位移
			cv::Mat vBaseline = Ow2 - Ow1;
			// 基线长度
			const float baseline = cv::norm(vBaseline);

			// 单目相机情况
			// 相邻关键帧的场景深度中值
			const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
			// 基线与景深的比例
			const float ratioBaselineDepth = baseline / medianDepthKF2;
			// 如果比例特别小，基线太短恢复3D点不准，那么跳过当前邻接的关键帧，不生成3D点
			if (ratioBaselineDepth < 0.01)
				continue;

			// Step 4：根据两个关键帧的位姿计算它们之间的基础矩阵
			cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);
			// Step 5：通过词袋对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对
			std::vector<std::pair<size_t, size_t> > vMatchedIndices;
			matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

			cv::Mat Rcw2 = pKF2->GetRotation();
			cv::Mat Rwc2 = Rcw2.t();
			cv::Mat tcw2 = pKF2->GetTranslation();
			cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			// Step 6：对每对匹配通过三角化生成3D点,和 Triangulate函数差不多
			const int nmatches = vMatchedIndices.size();
			for (int ikp = 0; ikp < nmatches; ikp++)
			{
				// Step 6.1：取出匹配特征点
				// 当前匹配对在当前关键帧中的索引
				const int &idx1 = vMatchedIndices[ikp].first;
				// 当前匹配对在邻接关键帧中的索引
				const int &idx2 = vMatchedIndices[ikp].second;

				// 当前匹配在当前关键帧中的特征点
				const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeys[idx1];
				// 当前匹配在邻接关键帧中的特征点
				const cv::KeyPoint &kp2 = pKF2->mvKeys[idx2];

				// Step 6.2：利用匹配点反投影得到视差角
				// 特征点反投影,其实得到的是在各自相机坐标系下的一个非归一化的方向向量,和这个点的反投影射线重合
				cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx)*invfx, (kp1.pt.y - cy)*invfy, 1.0);
				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx)*invfx, (kp2.pt.y - cy)*invfy, 1.0);

				// 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
				cv::Mat ray1 = Rwc1 * xn1;
				cv::Mat ray2 = Rwc2 * xn2;

				// 匹配点射线夹角余弦值
				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

				// Step 6.4：三角化恢复3D点
				cv::Mat x3D;
				// cosParallaxRays > 0 && cosParallaxRays<0.9998表明视差角正常,0.9998 对应1°
				// 匹配点对夹角大，用三角法恢复3D点
				// 参考：https://github.com/raulmur/ORB_SLAM2/issues/345
				if (cosParallaxRays > 0 && cosParallaxRays < 0.9998)
				{
					// Linear Triangulation Method
					// 见Initializer.cc的 Triangulate 函数,实现是一样的,顶多就是把投影矩阵换成了变换矩阵
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

					cv::Mat w, u, vt;
					cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

					x3D = vt.row(3).t();
					// 归一化之前的检查
					if (x3D.at<float>(3) == 0)
						continue;
					// 归一化成为齐次坐标,然后提取前面三个维度作为欧式坐标
					x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
				}
				else
					continue; //No stereo and very low parallax, 放弃

				// 为方便后续计算，转换成为了行向量
				cv::Mat x3Dt = x3D.t();

				// Step 6.5：检测生成的3D点是否在相机前方,不在的话就放弃这个点
				float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
				if (z1 <= 0)
					continue;

				float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
				if (z2 <= 0)
					continue;

				// Step 6.6：计算3D点在当前关键帧下的重投影误差
				const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
				const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
				const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
				const float invz1 = 1.0 / z1;

				// 单目情况下
				float u1 = fx * x1*invz1 + cx;
				float v1 = fy * y1*invz1 + cy;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				// 假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
				if ((errX1*errX1 + errY1 * errY1) > 5.991*sigmaSquare1)
					continue;

				// 计算3D点在另一个关键帧下的重投影误差，操作同上
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

				// Step 6.7：检查尺度连续性
				// 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = cv::norm(normal2);

				if (dist1 == 0 || dist2 == 0)
					continue;

				// ratioDist是不考虑金字塔尺度下的距离比例
				const float ratioDist = dist2 / dist1;
				// 金字塔尺度因子的比例
				const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

				// 距离的比例和图像金字塔的比例不应该差太多，否则就跳过
				if (ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
					continue;

				// Step 6.8：三角化生成3D点成功，构造成MapPoint
				MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

				// Step 6.9：为该MapPoint添加属性：
				// a.观测到该MapPoint的关键帧
				pMP->AddObservation(mpCurrentKeyFrame, idx1);
				pMP->AddObservation(pKF2, idx2);

				mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
				pKF2->AddMapPoint(pMP, idx2);

				// b.该MapPoint的描述子
				pMP->ComputeDistinctiveDescriptors();

				// c.该MapPoint的平均观测方向和深度范围
				pMP->UpdateNormalAndDepth();

				mpMap->AddMapPoint(pMP);

				// Step 6.10：将新产生的点放入检测队列
				// 这些MapPoints都会经过MapPointCulling函数的检验
				mlpRecentAddedMapPoints.push_back(pMP);
				nnew++;
			}
		}
	}

	// 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
	cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
	{
		// 先构造两帧之间的R12,t12
		cv::Mat R1w = pKF1->GetRotation();
		cv::Mat t1w = pKF1->GetTranslation();
		cv::Mat R2w = pKF2->GetRotation();
		cv::Mat t2w = pKF2->GetTranslation();

		cv::Mat R12 = R1w * R2w.t();

		cv::Mat t12 = -R1w * R2w.t()*t2w + t1w;

		// 得到 t12 的反对称矩阵
		cv::Mat t12x = SkewSymmetricMatrix(t12);

		const cv::Mat &K1 = mK;
		const cv::Mat &K2 = mK;

		// Essential Matrix: t12叉乘R12
		// Fundamental Matrix: inv(K1)*E*inv(K2)
		return K1.t().inv()*t12x*R12*K2.inv();
	}

	// 计算三维向量v的反对称矩阵
	cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
	{
		return (cv::Mat_<float>(3, 3) <<
			0, -v.at<float>(2), v.at<float>(1),
			v.at<float>(2), 0, -v.at<float>(0),
			-v.at<float>(1), v.at<float>(0), 0);
	}

	/**
	 * @brief 检查并融合当前关键帧与相邻帧（两级相邻）重复的地图点
	 *
	 */
	void LocalMapping::SearchInNeighbors()
	{
		// Step 1：获得当前关键帧在共视图中权重排名前nn的邻接关键帧
		// 开始之前先定义几个概念
		// 当前关键帧的邻接关键帧，称为一级相邻关键帧，也就是邻居
		// 与一级相邻关键帧相邻的关键帧，称为二级相邻关键帧，也就是邻居的邻居

		// 单目情况要20个邻接关键帧，双目或者RGBD则要10个
		int nn = 10;
		if (mbMonocular)
			nn = 20;

		// 和当前关键帧相邻的关键帧，也就是一级相邻关键帧
		const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

		// Step 2：存储一级相邻关键帧及其二级相邻关键帧
		std::vector<KeyFrame*> vpTargetKFs;
		// 开始对所有候选的一级关键帧展开遍历：
		for (std::vector<KeyFrame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			// 没有和当前帧进行过融合的操作
			if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
				continue;
			// 加入一级相邻关键帧    
			vpTargetKFs.push_back(pKFi);
			// 标记已经加入
			pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

			// 以一级相邻关键帧的共视关系最好的5个相邻关键帧 作为二级相邻关键帧
			const std::vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
			// 遍历得到的二级相邻关键帧
			for (std::vector<KeyFrame*>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
			{
				KeyFrame* pKFi2 = *vit2;
				// 当然这个二级相邻关键帧要求没有和当前关键帧发生融合,并且这个二级相邻关键帧也不是当前关键帧
				if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
					continue;
				// 存入二级相邻关键帧    
				vpTargetKFs.push_back(pKFi2);
			}
		}

		// 使用默认参数, 最优和次优比例0.6,匹配时检查特征点的旋转
		ORBmatcher matcher;
		// Step 3：将当前帧的地图点分别投影到两级相邻关键帧，寻找匹配点对应的地图点进行融合，称为正向投影融合
		std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
		for (std::vector<KeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;

			// 将地图点投影到关键帧中进行匹配和融合；融合策略如下
			// 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
			// 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
			// 注意这个时候对地图点融合的操作是立即生效的
			matcher.Fuse(pKFi, vpMapPointMatches);
		}

		// Step 4：将两级相邻关键帧地图点分别投影到当前关键帧，寻找匹配点对应的地图点进行融合，称为反向投影融合
		// 用于进行存储要融合的一级邻接和二级邻接关键帧所有MapPoints的集合
		std::vector<MapPoint*> vpFuseCandidates;
		vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

		//  Step 4.1：遍历每一个一级邻接和二级邻接关键帧，收集他们的地图点存储到 vpFuseCandidates
		for (std::vector<KeyFrame*>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
		{
			KeyFrame* pKFi = *vitKF;
			std::vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

			// 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints,找出需要进行融合的并且加入到集合中
			for (std::vector<MapPoint*>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
			{
				MapPoint* pMP = *vitMP;
				if (!pMP)
					continue;

				// 如果地图点是坏点，或者已经加进集合vpFuseCandidates，跳过
				if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
					continue;

				// 加入集合，并标记已经加入
				pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
				vpFuseCandidates.push_back(pMP);
			}
		}

		// Step 4.2：进行地图点投影融合,和正向融合操作是完全相同的
		// 不同的是正向操作是"每个关键帧和当前关键帧的地图点进行融合",而这里的是"当前关键帧和所有邻接关键帧的地图点进行融合"
		matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

		// Step 5：更新当前帧地图点的描述子、深度、平均观测方向等属性
		vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
		for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					// 在所有找到pMP的关键帧中，获得最佳的描述子
					pMP->ComputeDistinctiveDescriptors();

					// 更新平均观测方向和观测距离
					pMP->UpdateNormalAndDepth();
				}
			}
		}

		// Step 6：更新当前帧与其它帧的共视连接关系
		mpCurrentKeyFrame->UpdateConnections();
	}

	/**
	 * @brief 检测当前关键帧在共视图中的关键帧，根据地图点在共视图中的冗余程度剔除该共视关键帧
	 * 冗余关键帧的判定：90%以上的地图点能被其他关键帧（至少3个）观测到
	 */
	void LocalMapping::KeyFrameCulling()
	{
		// 该函数里变量层层深入，这里列一下：
		// mpCurrentKeyFrame：当前关键帧，本程序就是判断它是否需要删除
		// pKF： mpCurrentKeyFrame的某一个共视关键帧
		// vpMapPoints：pKF对应的所有地图点
		// pMP：vpMapPoints中的某个地图点
		// observations：所有能观测到pMP的关键帧
		// pKFi：observations中的某个关键帧
		// scaleLeveli：pKFi的金字塔尺度
		// scaleLevel：pKF的金字塔尺度

		// Step 1：根据共视图提取当前关键帧的所有共视关键帧
		std::vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

		// 对所有的共视关键帧进行遍历
		for (std::vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
		{
			KeyFrame* pKF = *vit;
			// 第1个关键帧不能删除，跳过
			if (pKF->mnId == 0)
				continue;

			// Step 2：提取每个共视关键帧的地图点
			const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

			// 观测次数阈值，默认为3
			const int thObs = 3;

			// 记录冗余观测点的数目
			int nRedundantObservations = 0;

			int nMPs = 0;
			// Step 3：遍历该共视关键帧的所有地图点，其中能被其它至少3个关键帧观测到的地图点为冗余地图点
			for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (!pMP->isBad())
				{
					nMPs++;
					// pMP->Observations() 是观测到该地图点的相机总数目（单目1，双目2）
					if (pMP->Observations() > thObs)
					{
						const int &scaleLevel = pKF->mvKeys[i].octave;
						// Observation存储的是可以看到该地图点的所有关键帧的集合
						const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

						int nObs = 0;
						// 遍历观测到该地图点的关键帧
						for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
						{
							KeyFrame* pKFi = mit->first;
							if (pKFi == pKF)
								continue;
							const int &scaleLeveli = pKFi->mvKeys[mit->second].octave;

							// 尺度约束：为什么pKF 尺度+1 要大于等于 pKFi 尺度？
							// 回答：因为同样或更低金字塔层级的地图点更准确
							if (scaleLeveli <= scaleLevel + 1)
							{
								nObs++;
								// 已经找到3个满足条件的关键帧，就停止不找了
								if (nObs >= thObs)
									break;
							}
						}
						// 地图点至少被3个关键帧观测到，就记录为冗余点，更新冗余点计数数目
						if (nObs >= thObs)
						{
							nRedundantObservations++;
						}
					}
				}

			}
			// Step 4：如果该关键帧90%以上的有效地图点被判断为冗余的，则认为该关键帧是冗余的，需要删除该关键帧
			if (nRedundantObservations > 0.9*nMPs)
				pKF->SetBadFlag();
		}
	}
}