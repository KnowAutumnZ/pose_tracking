#pragma once

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"
#include "Tracking.h"
#include "Converter.h"

#include "g2o/types/types_seven_dof_expmap.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"

namespace PoseTracking
{
	/** @brief 优化器,所有的优化相关的函数都在这个类中; 并且这个类只有成员函数没有成员变量,相对要好分析一点 */
	class Optimizer
	{
	public:

		/**
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
		int static PoseOptimization(Tracking* pTrack, Frame* pFrame);


	};
}