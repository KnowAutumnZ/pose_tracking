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

			// vbPrevMatched 输入的是参考帧 F1的特征点
			// windowSize = 100，输入最大最小金字塔层级 均为0
			std::vector<size_t> vIndices2 = GetFeaturesInArea(F2, F1.mvKeys[i1].pt.x, F1.mvKeys[i1].pt.y, windowSize, level1 - 1, level1 + 1);

			// 没有候选特征点，跳过
			if (vIndices2.empty())
				continue;

			// 取出参考帧F1中当前遍历特征点对应的描述子
			cv::Mat d1 = F1.mDescriptors.row(i1);

			int bestDist = INT_MAX;     //最佳描述子匹配距离，越小越好
			int bestDist2 = INT_MAX;    //次佳描述子匹配距离
			int bestIdx2 = -1;          //最佳候选特征点在F2中的index











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

		// 如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
		if (nMinCellX >= FRAME_GRID_COLS)
			return vIndices;

		// 计算圆所在的右边界网格列索引
		const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - F.mnMinX + r)*F.mfGridElementWidthInv));
		// 如果计算出的圆右边界所在的网格不合法，说明该特征点不好，直接返回空vector
		if (nMaxCellX < 0)
			return vIndices;

		//后面的操作也都是类似的，计算出这个圆上下边界所在的网格行的id
		const int nMinCellY = std::max(0, (int)floor((y - F.mnMinY - r)*F.mfGridElementHeightInv));
		if (nMinCellY >= FRAME_GRID_ROWS)
			return vIndices;

		const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - F.mnMinY + r)*F.mfGridElementHeightInv));
		if (nMaxCellY < 0)
			return vIndices;

		// Step 2 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				// 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引
				const std::vector<size_t> vCell = F.mGrid[ix][iy];
				// 如果这个网格中没有特征点，那么跳过这个网格继续下一个
				if (vCell.empty())
					continue;

				for (size_t i=0; i<vCell.size(); i++)
				{
					// 根据索引先读取这个特征点 
					const cv::KeyPoint &kpUn = F.mvKeys[vCell[i]];

					// 保证特征点是在金字塔层级minLevel和maxLevel之间，不是的话跳过
					if (kpUn.octave < minLevel || kpUn.octave > maxLevel)
						continue;

					// 通过检查，计算候选特征点到圆中心的距离，查看是否是在这个圆形区域之内
					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					// 如果x方向和y方向的距离都在指定的半径之内，存储其index为候选特征点
					if (sqrt(distx*distx + disty * distx))
						vIndices.push_back(vCell[i]);
				}
			}
		}
		return vIndices;
	}

}