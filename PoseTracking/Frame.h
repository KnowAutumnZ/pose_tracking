#pragma once

#include <opencv2/opencv.hpp>

#include "orbDetector.h"
#include "MapPoint.h"
#include "KeyFrame.h"

namespace PoseTracking
{
	/**
	* @brief 网格的行数
	*
	*/
	#define FRAME_GRID_ROWS 48
	/**
	* @brief 网格的列数
	*
	*/
	#define FRAME_GRID_COLS 64

	class MapPoint;
	class KeyFrame;

	class Frame
	{
	public:
		Frame() {};
		virtual ~Frame() {};

		// Copy constructor. 拷贝构造函数
		/**
		 * @brief 拷贝构造函数
		 * @details 复制构造函数, mLastFrame = Frame(mCurrentFrame) \n
		 * 如果不是自定以拷贝函数的话，系统自动生成的拷贝函数对于所有涉及分配内存的操作都将是浅拷贝 \n
		 * @param[in] frame 引用
		 * @note 另外注意，调用这个函数的时候，这个函数中隐藏的this指针其实是指向目标帧的
		 */
		Frame(const Frame& frame);

		/**
		 * @brief 为单目相机准备的帧构造函数
		 *
		 * @param[in] imGray                            //灰度图
		 * @param[in] timeStamp                         //时间戳
		 * @param[in & out] extractor                   //ORB特征点提取器的句柄
		 * @param[in] K                                 //相机的内参数矩阵
		 * @param[in] Distort                           //相机的去畸变参数
		 */
		Frame(const cv::Mat &imGray, const double &timeStamp, orbDetector* extractor, const cv::Mat &K, const cv::Mat& Distort);

		// 用Tcw更新mTcw
		/**
		 * @brief 用 Tcw 更新 mTcw 以及类中存储的一系列位姿
		 *
		 * @param[in] Tcw 从世界坐标系到当前帧相机位姿的变换矩阵
		 */
		void SetPose(cv::Mat Tcw);

		/**
		 * @brief 根据相机位姿,计算相机的旋转,平移和相机中心等矩阵.
		 * @details 其实就是根据Tcw计算mRcw、mtcw和mRwc、mOw.
		 */
		void UpdatePoseMatrices();

		/**
		 * @brief 用内参对特征点去畸变，结果报存在mvKeys中
		 *
		 */
		void UndistortKeyPoints(const std::vector<cv::KeyPoint>& vKeys, const cv::Mat &K, const cv::Mat& Distort);

		/**
		 * @brief 计算去畸变图像的边界
		 *
		 * @param[in] imLeft            需要计算边界的图像
		 */
		void ComputeImageBounds(const cv::Mat &imLeft, const cv::Mat &K, const cv::Mat& Distort);

		/**
		 * @brief 将提取到的特征点分配到图像网格中 \n
		 * @details 该函数由构造函数调用
		 *
		 */
		void AssignFeaturesToGrid();

		/**
		 * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
		 *
		 * @param[in] kp                    给定的特征点
		 * @param[in & out] posX            特征点所在网格坐标的横坐标
		 * @param[in & out] posY            特征点所在网格坐标的纵坐标
		 * @return true                     如果找到特征点所在的网格坐标，返回true
		 * @return false                    没找到返回false
		 */
		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

		// Returns the camera center.
		/**
		 * @brief 返回位于当前帧位姿时,相机的中心
		 *
		 * @return cv::Mat 相机中心在世界坐标系下的3D点坐标
		 */
		inline cv::Mat GetCameraCenter()
		{
			return mOw.clone();
		}

		/**
		 * @brief Get the Rotation Inverse object
		 * mRwc存储的是从当前相机坐标系到世界坐标系所进行的旋转，而我们一般用的旋转则说的是从世界坐标系到当前相机坐标系的旋转
		 * @return 返回从当前帧坐标系到世界坐标系的旋转
		 */
		inline cv::Mat GetRotationInverse()
		{
			return mRwc.clone();
		}

	public:
		//帧的时间戳
		double mTimeStamp;
		//原始左图像提取出的特征点
		std::vector<cv::KeyPoint> mvKeys;
		//原始右图像提取出的特征点
		std::vector<cv::KeyPoint> mvKeysRight;
		//左目摄像头和右目摄像头特征点对应的描述子
		cv::Mat mDescriptors, mDescriptorsRight;

		// 每个特征点对应的MapPoint.如果特征点没有对应的地图点,那么将存储一个空指针
		std::vector<MapPoint*> mvpMapPoints;

		// 观测不到Map中的3D点
		// 属于外点的特征点标记,在 Optimizer::PoseOptimization 使用了
		std::vector<bool> mvbOutlier;

		cv::Mat mTcw; //< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵,是我们常规理解中的相机位姿

		// Current and Next Frame id.
		// 类的静态成员变量，这些变量则是在整个系统开始执行的时候被初始化的——它在全局区被初始化
		static long unsigned int nNextId; //< Next Frame id.
		long unsigned int mnId; //< Current Frame id.

		// 普通帧与自己共视程度最高的关键帧作为参考关键帧
		KeyFrame* mpReferenceKF;

		// Rotation, translation and camera center
		cv::Mat mRcw; //< Rotation from world to camera
		cv::Mat mtcw; //< Translation from world to camera
		cv::Mat mRwc; //< Rotation from camera to world
		cv::Mat mOw;  //< mtwc,Translation from camera to world

		/**
		* @name 图像金字塔信息
		* @{
		*/
		// Scale pyramid info.
		int mnScaleLevels;                  ///<图像金字塔的层数
		float mfScaleFactor;                ///<图像金字塔的尺度因子
		float mfLogScaleFactor;             ///<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度

		std::vector<float> mvScaleFactors;		///<图像金字塔每一层的缩放因子
		std::vector<float> mvInvScaleFactors;	///<以及上面的这个变量的倒数
		std::vector<float> mvLevelSigma2;		///@todo 目前在frame.c中没有用到，无法下定论
		std::vector<float> mvInvLevelSigma2;	///<上面变量的倒数

		//畸变校正后的图像边界
		float mnMinX, mnMinY, mnMaxX, mnMaxY;

		//是否要进行初始化操作的标志
		//这里给这个标志置位的操作是在最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧，所以这个标志要置位
		bool mbInitialComputations = true;

		// 表示一个图像像素相当于多少个图像网格列（宽）
		float mfGridElementWidthInv;
		// 表示一个图像像素相当于多少个图像网格行（高）
		float mfGridElementHeightInv;

		// 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
		// FRAME_GRID_ROWS 48
		// FRAME_GRID_COLS 64
		// 这个向量中存储的是每个图像网格内特征点的id（左图）
		std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
	};
}