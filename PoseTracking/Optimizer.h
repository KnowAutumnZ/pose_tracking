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
#include "g2o/solvers/linear_solver_eigen.h"

namespace PoseTracking
{
	/** @brief �Ż���,���е��Ż���صĺ��������������; ���������ֻ�г�Ա����û�г�Ա����,���Ҫ�÷���һ�� */
	class Optimizer
	{
	public:

		/**
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
		int static PoseOptimization(Frame* pFrame);

		/**
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
		void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);
	};
}