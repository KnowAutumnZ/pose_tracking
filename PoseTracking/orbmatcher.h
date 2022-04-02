#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Tracking.h"

namespace PoseTracking
{
	class ORBmatcher
	{
	public:
		/**
		 * Constructor
		 * @param nnratio  ratio of the best and the second score   最优和次优评分的比例
		 * @param checkOri check orientation                        是否检查方向
		 */
		ORBmatcher(float nnratio = 0.6, bool checkOri = true);

		/**
		 * @brief 将地图点投影到关键帧中进行匹配和融合;并且地图点的替换可以在这个函数中进行
		 * @param[in] pKF           关键帧
		 * @param[in] vpMapPoints   地图点
		 * @param[in] th            搜索窗口的阈值
		 * @return int
		 */
		int Fuse(KeyFrame* pKF, const std::vector<MapPoint*> &vpMapPoints, const float th = 3.0);

		/**
		 * @brief 通过投影地图点到当前帧，对Local MapPoint进行跟踪
		 * 步骤
		 * Step 1 遍历有效的局部地图点
		 * Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
		 * Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
		 * Step 4 寻找候选匹配点中的最佳和次佳匹配点
		 * Step 5 筛选最佳匹配点
		 * @param[in] F                         当前帧
		 * @param[in] vpMapPoints               局部地图点，来自局部关键帧
		 * @param[in] th                        搜索范围
		 * @return int                          成功匹配的数目
		 */
		int SearchByProjection(Frame* F, const std::vector<MapPoint*> &vpMapPoints, const float th = 3);

		/**
		 * @brief Computes the Hamming distance between two ORB descriptors 计算地图点和候选投影点的描述子距离
		 * @param[in] a     一个描述子
		 * @param[in] b     另外一个描述子
		 * @return int      描述子的汉明距离
		 */
		static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

		/**
		 * @brief 单目初始化中用于参考帧和当前帧的特征点匹配
		 * 步骤
		 * Step 1 构建旋转直方图
		 * Step 2 在半径窗口内搜索当前帧F2中所有的候选匹配特征点
		 * Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
		 * Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
		 * Step 5 计算匹配点旋转角度差所在的直方图
		 * Step 6 筛除旋转直方图中“非主流”部分
		 * Step 7 将最后通过筛选的匹配好的特征点保存
		 * @param[in] F1                        初始化参考帧
		 * @param[in] F2                        当前帧
		 * @param[in & out] vbPrevMatched       本来存储的是参考帧的所有特征点坐标，该函数更新为匹配好的当前帧的特征点坐标
		 * @param[in & out] vnMatches12         保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
		 * @param[in] windowSize                搜索窗口
		 * @return int                          返回成功匹配的特征点数目
		 */
		int SearchForInitialization(Frame* F1, Frame* F2, std::vector<int> &vnMatches12, int windowSize = 10);

		/**
		 * @brief 将上一帧跟踪的地图点投影到当前帧，并且搜索匹配点。用于跟踪前一帧
		 * 步骤
		 * Step 1 建立旋转直方图，用于检测旋转一致性
		 * Step 2 计算当前帧和前一帧的平移向量
		 * Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
		 * Step 4 根据相机的前后前进方向来判断搜索尺度范围
		 * Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点
		 * Step 6 计算匹配点旋转角度差所在的直方图
		 * Step 7 进行旋转一致检测，剔除不一致的匹配
		 * @param[in] CurrentFrame          当前帧
		 * @param[in] LastFrame             上一帧
		 * @param[in] th                    搜索范围阈值，默认单目为7，双目15
		 * @param[in] bMono                 是否为单目
		 * @return int                      成功匹配的数量
		 */
		int SearchByProjection(Frame* CurrentFrame, const Frame* LastFrame, const float th, const bool bMono);

		/**
		 * @brief 利用基本矩阵F12，在两个关键帧之间未匹配的特征点中产生新的3d点
		 * @param pKF1          关键帧1
		 * @param pKF2          关键帧2
		 * @param F12           基础矩阵
		 * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示，下标是关键帧1的特征点id，存储的是关键帧2的特征点id
		 * @param bOnlyStereo   在双目和rgbd情况下，是否要求特征点在右图存在匹配
		 * @return              成功匹配的数量
		 */
		int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12, std::vector<std::pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

		int SearchForRefModel(KeyFrame *pKF, Frame* F, std::vector<MapPoint*> &vpMapPointMatches);

	private:
		/**
		 * @brief 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
		 *
		 * @param[in] x                     特征点坐标x
		 * @param[in] y                     特征点坐标y
		 * @param[in] r                     搜索半径
		 * @param[in] minLevel              最小金字塔层级
		 * @param[in] maxLevel              最大金字塔层级
		 * @return vector<size_t>           返回搜索到的候选匹配点id
		 */
		std::vector<size_t> GetFeaturesInArea(Frame* F, const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1);

		/**
		 * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
		 *
		 * @param[in] histo         匹配特征点对旋转方向差直方图
		 * @param[in] L             直方图尺寸
		 * @param[in & out] ind1          bin值第一大对应的索引
		 * @param[in & out] ind2          bin值第二大对应的索引
		 * @param[in & out] ind3          bin值第三大对应的索引
		 */
		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

		/**
		 * @brief 根据观察的视角来计算匹配的时的搜索窗口大小
		 * @param[in] viewCos   视角的余弦值
		 * @return float        搜索窗口的大小
		 */
		float RadiusByViewingCos(const float &viewCos);

		float mfNNratio;            //< 最优评分和次优评分的比例
		bool mbCheckOrientation;    //< 是否检查特征点的方向
	};
}