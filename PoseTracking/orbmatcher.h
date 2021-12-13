#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"

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
		int SearchForInitialization(Frame &F1, Frame &F2, std::vector<int> &vnMatches12, int windowSize = 10);

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
		std::vector<size_t> GetFeaturesInArea(Frame &F, const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1);

		/**
		 * @brief Computes the Hamming distance between two ORB descriptors 计算地图点和候选投影点的描述子距离
		 * @param[in] a     一个描述子
		 * @param[in] b     另外一个描述子
		 * @return int      描述子的汉明距离
		 */
		int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

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


		float mfNNratio;            //< 最优评分和次优评分的比例
		bool mbCheckOrientation;    //< 是否检查特征点的方向
	};
}