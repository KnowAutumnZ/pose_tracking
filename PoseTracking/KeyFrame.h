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
	class KeyFrameDatabase;
	class Frame;

	class KeyFrame
	{
	public:
		/**
		 * @brief 构造函数
		 * @param[in] F         父类普通帧的对象
		 * @param[in] pMap      所属的地图指针
		 * @param[in] pKFDB     使用的词袋模型的指针
		 */
		KeyFrame(Frame* F, Map* pMap, KeyFrameDatabase* pKFDB);

		/**
		 * @brief 设置当前关键帧的位姿
		 * @param[in] Tcw 位姿
		 */
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();                  //< 获取位姿
		cv::Mat GetPoseInverse();           //< 获取位姿的逆
		cv::Mat GetCameraCenter();          //< 获取(左目)相机的中心
		cv::Mat GetStereoCenter();          //< 获取双目相机的中心,这个只有在可视化的时候才会用到
		cv::Mat GetRotation();              //< 获取姿态
		cv::Mat GetTranslation();           //< 获取位置

		// ====================== Covisibility graph functions ============================
		/**
		 * @brief 为关键帧之间添加连接
		 * @details 更新了mConnectedKeyFrameWeights
		 * @param pKF    关键帧
		 * @param weight 权重，该关键帧与pKF共同观测到的3d点数量
		 */
		void AddConnection(KeyFrame* pKF, const int &weight);

		/**
		 * @brief 删除当前关键帧和指定关键帧之间的共视关系
		 * @param[in] pKF 要删除的共视关系
		 */
		void EraseConnection(KeyFrame* pKF);

		/** @brief 更新图的连接  */
		void UpdateConnections();

		/**
		 * @brief 按照权重对连接的关键帧进行排序
		 * @detials 更新后的变量存储在mvpOrderedConnectedKeyFrames和mvOrderedWeights中
		 */
		void UpdateBestCovisibles();

		/**
		 * @brief 得到与该关键帧连接的关键帧(没有排序的)
		 * @return 连接的关键帧
		 */
		std::set<KeyFrame*> GetConnectedKeyFrames();

		/**
		 * @brief 得到与该关键帧连接的关键帧(已按权值排序)
		 * @return 连接的关键帧
		 */
		std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();

		/**
		 * @brief 得到与该关键帧连接的前N个关键帧(已按权值排序)
		 * NOTICE 如果连接的关键帧少于N，则返回所有连接的关键帧,所以说返回的关键帧的数目其实不一定是N个
		 * @param N 前N个
		 * @return 连接的关键帧
		 */
		std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);

		/**
		 * @brief 得到与该关键帧连接的权重大于等于w的关键帧
		 * @param w 权重
		 * @return 连接的关键帧
		 */
		std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);

		/**
		 * @brief 得到该关键帧与pKF的权重
		 * @param  pKF 关键帧
		 * @return     权重
		 */
		int GetWeight(KeyFrame* pKF);

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
		 * @brief 由于其他的原因,导致当前关键帧观测到的某个地图点被删除(bad==true)了,这里是"通知"当前关键帧这个地图点已经被删除了
		 * @param[in] pMP 被删除的地图点指针
		 */
		void EraseMapPointMatch(MapPoint* pMP);

		/**
		 * @brief 地图点的替换
		 * @param[in] idx 要替换掉的地图点的索引
		 * @param[in] pMP 新地图点的指针
		 */
		void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);

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

		/**
		 * @brief 关键帧中，大于等于minObs的MapPoints的数量
		 * @details minObs就是一个阈值，大于minObs就表示该MapPoint是一个高质量的MapPoint \n
		 * 一个高质量的MapPoint会被多个KeyFrame观测到.
		 * @param  minObs 最小观测
		 */
		int TrackedMapPoints(const int &minObs);

		/**
		 * @brief 获取获取当前关键帧的具体的某个地图点
		 * @param[in] idx id
		 * @return MapPoint* 地图点句柄
		 */
		MapPoint* GetMapPoint(const size_t &idx);

		// ========================= Spanning tree functions =======================
		/**
		 * @brief 添加子关键帧（即和子关键帧具有最大共视关系的关键帧就是当前关键帧）
		 * @param[in] pKF 子关键帧句柄
		 */
		void AddChild(KeyFrame* pKF);

		/**
		 * @brief 获取获取当前关键帧的子关键帧
		 * @return std::set<KeyFrame*>  子关键帧集合
		 */
		std::set<KeyFrame*> GetChilds();
		/**
		 * @brief 获取当前关键帧的父关键帧
		 * @return KeyFrame* 父关键帧句柄
		 */
		KeyFrame* GetParent();

		/**
		 * @brief 改变当前关键帧的父关键帧
		 * @param[in] pKF 父关键帧句柄
		 */
		void ChangeParent(KeyFrame* pKF);

		/**
		 * @brief 删除某个子关键帧
		 * @param[in] pKF 子关键帧句柄
		 */
		void EraseChild(KeyFrame* pKF);

		/**
		 * @brief 判断某个关键帧是否是当前关键帧的子关键帧
		 * @param[in] pKF 关键帧句柄
		 * @return true
		 * @return false
		 */
		bool hasChild(KeyFrame* pKF);

		// Image
		/**
		 * @brief 判断某个点是否在当前关键帧的图像中
		 * @param[in] x 点的坐标
		 * @param[in] y 点的坐标
		 * @return true
		 * @return false
		 */
		bool IsInImage(const float &x, const float &y) const;

		/** @brief 设置当前关键帧不要在优化的过程中被删除  */
		void SetNotErase();

		/** @brief 准备删除当前的这个关键帧,表示不进行回环检测过程;由回环检测线程调用 */
		void SetErase();

		/** @brief 真正地执行删除关键帧的操作 */
		void SetBadFlag();

		// KeyPoint functions
		/**
		 * @brief 获取某个特征点的邻域中的特征点id
		 * @param[in] x 特征点坐标
		 * @param[in] y 特征点坐标
		 * @param[in] r 邻域大小(半径)
		 * @return std::vector<size_t> 在这个邻域内找到的特征点索引的集合
		 */
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
	public:
		// nNextID名字改为nLastID更合适，表示上一个KeyFrame的ID号
		static long unsigned int nNextId;
		// 在nNextID的基础上加1就得到了mnID，为当前KeyFrame的ID号
		long unsigned int mnId;
		// 每个KeyFrame基本属性是它是一个Frame，KeyFrame初始化的时候需要Frame，
		// mnFrameId记录了该KeyFrame是由哪个Frame初始化的
		const long unsigned int mnFrameId;

		// local mapping中记录当前处理的关键帧的mnId，表示当前局部BA的关键帧id。mnBALocalForKF 在map point.h里面也有同名的变量。
		long unsigned int mnBALocalForKF;
		// local mapping中记录当前处理的关键帧的mnId, 只是提供约束信息但是却不会去优化这个关键帧
		long unsigned int mnBAFixedForKF;

		//表示它已经是某帧的局部关键帧了，可以防止重复添加局部关键帧
		long unsigned int mnTrackReferenceForFrame;      // 记录它
		long unsigned int mnFuseTargetForKF;			 //< 标记在局部建图线程中,和哪个关键帧进行融合的操作

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

		// Covisibility Graph
		// 与该关键帧连接（至少15个共视地图点）的关键帧与权重
		std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
		// 共视关键帧中权重从大到小排序后的关键帧          
		std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
		// 共视关键帧中从大到小排序后的权重，和上面对应
		std::vector<int> mvOrderedWeights;

		// ===================== Spanning Tree and Loop Edges ========================
		// std::set是集合，相比vector，进行插入数据这样的操作时会自动排序
		bool mbFirstConnection;                     // 是否是第一次生成树
		KeyFrame* mpParent;                         // 当前关键帧的父关键帧 （共视程度最高的）
		std::set<KeyFrame*> mspChildrens;           // 存储当前关键帧的子关键帧
		std::set<KeyFrame*> mspLoopEdges;           // 和当前关键帧形成回环关系的关键帧

		/**
		* @name 图像金字塔信息
		* @{
		*/
		// Scale pyramid info.
		int mnScaleLevels;                  //<图像金字塔的层数
		float mfScaleFactor;                //<图像金字塔的尺度因子
		float mfLogScaleFactor;             //<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度

		std::vector<float> mvScaleFactors;		//<图像金字塔每一层的缩放因子
		std::vector<float> mvLevelSigma2;		//@todo 目前在frame.c中没有用到，无法下定论
		std::vector<float> mvInvLevelSigma2;	//<上面变量的倒数

		const int mnMinX;
		const int mnMinY;
		const int mnMaxX;
		const int mnMaxY;

	private:
		// SE3 Pose and camera center
		cv::Mat Tcw;    // 当前相机的位姿，世界坐标系到相机坐标系
		cv::Mat Twc;    // 当前相机位姿的逆
		cv::Mat Ow;     // 相机光心(左目)在世界坐标系下的坐标,这里和普通帧中的定义是一样的

		cv::Mat Cw;     //< Stereo middel point. Only for visualization

		cv::Mat mTcp;   // Pose relative to parent (this is computed when bad flag is activated)

		// MapPoints associated to keypoints
		std::vector<MapPoint*> mvpMapPoints;

		float mHalfBaseline = 10; //< 对于双目相机来说,双目相机基线长度的一半. Only for visualization

		// 在对位姿进行操作时相关的互斥锁
		std::mutex mMutexPose;
		// 在操作当前关键帧和其他关键帧的公式关系的时候使用到的互斥锁
		std::mutex mMutexConnections;
		// 在操作和特征点有关的变量的时候的互斥锁
		std::mutex mMutexFeatures;

		Map* mpMap;

		// Bad flags
		bool mbNotErase;            //< 当前关键帧已经和其他的关键帧形成了回环关系，因此在各种优化的过程中不应该被删除
		bool mbToBeErased;          //<
		bool mbBad;                 //< 
	};
}