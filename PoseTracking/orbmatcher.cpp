#include "orbmatcher.h"

namespace PoseTracking
{
	// 要用到的一些阈值
	const int TH_HIGH = 100;
	const int TH_LOW = 50;
	const int HISTO_LENGTH = 30;

	// 构造函数,参数默认值为0.6,true
	ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
	{

	}

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
	int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, std::vector<int> &vnMatches12, int windowSize)
	{
		int nmatches = 0;
		// F1中特征点和F2中匹配关系，注意是按照F1特征点数目分配空间
		vnMatches12 = std::vector<int>(F1.mvKeys.size(), -1);
		
		// Step 1 构建旋转直方图，HISTO_LENGTH = 30
		std::vector<int> rotHist[HISTO_LENGTH];
		// 每个bin里预分配30个，因为使用的是vector不够的话可以自动扩展容量
		for (int i = 0; i < HISTO_LENGTH; i++)
			rotHist[i].reserve(30);

		//! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码   
		const float factor = HISTO_LENGTH / 360.0f;

		// 匹配点对距离，注意是按照F2特征点数目分配空间
		std::vector<int> vMatchedDistance(F2.mvKeys.size(), INT_MAX);
		// 从帧2到帧1的反向匹配，注意是按照F2特征点数目分配空间
		std::vector<int> vnMatches21(F2.mvKeys.size(), -1);

		// 遍历帧1中的所有特征点
		for (size_t i1 = 0, iend1 = F1.mvKeys.size(); i1 < iend1; i1++)
		{
			cv::KeyPoint kp1 = F1.mvKeys[i1];
			int level1 = kp1.octave;



		}

	}

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
	std::vector<size_t> ORBmatcher::GetFeaturesInArea(Frame &F, const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel)
	{
		// 存储搜索结果的vector
		std::vector<size_t> vIndices;

		int N = F.mvKeys.size();

		// Step 1 计算半径为r圆左右上下边界所在的网格列和行的id
		// 查找半径为r的圆左侧边界所在网格列坐标。这个地方有点绕，慢慢理解下：
		// (mnMaxX-mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
		// mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
		// (x-mnMinX-r)，可以看做是从图像的左边界mnMinX到半径r的圆的左边界区域占的像素列数
		// 两者相乘，就是求出那个半径为r的圆的左侧边界在哪个网格列中
		// 保证nMinCellX 结果大于等于0
		const int nMinCellX = std::max(0, (int)floor((x - F.mnMinX - r)*F.mfGridElementWidthInv));





	}

}