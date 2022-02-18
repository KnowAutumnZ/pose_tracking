#pragma once

#include "MapPoint.h"
#include "orbDetector.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <opencv2/opencv.hpp>

namespace PoseTracking
{
	class Map;
	class MapPoint;
	class Frame;
	class KeyFrameDatabase;

	class KeyFrame
	{
	public:
		/**
		 * @brief 构造函数
		 * @param[in] F         父类普通帧的对象
		 * @param[in] pMap      所属的地图指针
		 * @param[in] pKFDB     使用的词袋模型的指针
		 */
		KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

		/**
		 * @brief 设置当前关键帧的位姿
		 * @param[in] Tcw 位姿
		 */
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();                  ///< 获取位姿
		cv::Mat GetPoseInverse();           ///< 获取位姿的逆
		cv::Mat GetCameraCenter();          ///< 获取(左目)相机的中心
		cv::Mat GetStereoCenter();          ///< 获取双目相机的中心,这个只有在可视化的时候才会用到
		cv::Mat GetRotation();              ///< 获取姿态
		cv::Mat GetTranslation();           ///< 获取位置

		// ====================== MapPoint observation functions ==================================
		/**
		 * @brief Add MapPoint to KeyFrame
		 * @param pMP MapPoint
		 * @param idx MapPoint在KeyFrame中的索引
		 */
		void AddMapPoint(MapPoint* pMP, const size_t &idx);
		/**
		 * @brief 由于其他的原因,导致当前关键帧观测到的某个地图点被删除(bad==true)了,这里是"通知"当前关键帧这个地图点已经被删除了
		 * @param[in] idx 被删除的地图点索引
		 */
		void EraseMapPointMatch(const size_t &idx);

		/**
		 * @brief Get MapPoint Matches 获取该关键帧的MapPoints
		 */
		std::vector<MapPoint*> GetMapPointMatches();

		/** @brief 返回当前关键帧是否已经完蛋了 */
		bool isBad();

		// Compute Scene Depth (q=2 median). Used in monocular.
		/**
		 * @brief 评估当前关键帧场景深度，q=2表示中值
		 * @param q q=2
		 * @return Median Depth
		 */
		float ComputeSceneMedianDepth(const int q);

	public:
		// nNextID名字改为nLastID更合适，表示上一个KeyFrame的ID号
		static long unsigned int nNextId;
		// 在nNextID的基础上加1就得到了mnID，为当前KeyFrame的ID号
		long unsigned int mnId;
		// 每个KeyFrame基本属性是它是一个Frame，KeyFrame初始化的时候需要Frame，
		// mnFrameId记录了该KeyFrame是由哪个Frame初始化的
		const long unsigned int mnFrameId;

		// 和Frame类中的定义相同
		int mnGridCols;
		int mnGridRows;
		float mfGridElementWidthInv;
		float mfGridElementHeightInv;

		//原始左图像提取出的特征点
		std::vector<cv::KeyPoint> mvKeys;
		//原始右图像提取出的特征点
		std::vector<cv::KeyPoint> mvKeysRight;
		//左目摄像头和右目摄像头特征点对应的描述子
		cv::Mat mDescriptors, mDescriptorsRight;

		//Grid over the image to speed up feature matching ,其实应该说是二维的,第三维的 vector中保存的是这个网格内的特征点的索引
		std::vector< std::vector <std::vector<size_t> > > mGrid;

		/**
		* @name 图像金字塔信息
		* @{
		*/
		// Scale pyramid info.
		int mnScaleLevels;                  ///<图像金字塔的层数
		float mfScaleFactor;                ///<图像金字塔的尺度因子
		float mfLogScaleFactor;             ///<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度

		std::vector<float> mvScaleFactors;		///<图像金字塔每一层的缩放因子
		std::vector<float> mvLevelSigma2;		///@todo 目前在frame.c中没有用到，无法下定论
		std::vector<float> mvInvLevelSigma2;	///<上面变量的倒数

	private:
		// SE3 Pose and camera center
		cv::Mat Tcw;    // 当前相机的位姿，世界坐标系到相机坐标系
		cv::Mat Twc;    // 当前相机位姿的逆
		cv::Mat Ow;     // 相机光心(左目)在世界坐标系下的坐标,这里和普通帧中的定义是一样的

		cv::Mat Cw;     //< Stereo middel point. Only for visualization

		// MapPoints associated to keypoints
		std::vector<MapPoint*> mvpMapPoints;

		float mHalfBaseline = 10; //< 对于双目相机来说,双目相机基线长度的一半. Only for visualization

		// 在对位姿进行操作时相关的互斥锁
		std::mutex mMutexPose;
		// 在操作当前关键帧和其他关键帧的公式关系的时候使用到的互斥锁
		std::mutex mMutexConnections;
		// 在操作和特征点有关的变量的时候的互斥锁
		std::mutex mMutexFeatures;

		// Bad flags
		bool mbNotErase;            ///< 当前关键帧已经和其他的关键帧形成了回环关系，因此在各种优化的过程中不应该被删除
		bool mbToBeErased;          ///<
		bool mbBad;                 ///< 
	};
}