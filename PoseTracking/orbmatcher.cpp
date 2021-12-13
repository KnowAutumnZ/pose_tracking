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

			// Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
			for (auto& vit: vIndices2)
			{
				size_t i2 = vit;
				// 取出候选特征点对应的描述子
				cv::Mat d2 = F2.mDescriptors.row(i2);
				// 计算两个特征点描述子距离
				int dist = DescriptorDistance(d1, d2);

				// 已经匹配过了，下一位
				if (vMatchedDistance[i2] <= dist)
					continue;
				// 如果当前匹配距离更小，更新最佳次佳距离
				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestIdx2 = i2;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}

			// Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
			// 即使算出了最佳描述子匹配距离，也不一定保证配对成功。要小于设定阈值
			if (bestDist <= TH_LOW)
			{
				// 最佳距离比次佳距离要小于设定的比例，这样特征点辨识度更高
				if (bestDist < (float)bestDist2*mfNNratio)
				{
					// 如果找到的候选特征点对应F1中特征点已经匹配过了，说明发生了重复匹配，将原来的匹配也删掉
					if (vnMatches21[bestIdx2] >= 0)
					{
						vnMatches12[vnMatches21[bestIdx2]] = -1;
						nmatches--;
					}
					// 次优的匹配关系，双向建立
					// vnMatches12保存参考帧F1和F2匹配关系，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
					vnMatches12[i1] = bestIdx2;
					vnMatches21[bestIdx2] = i1;
					vMatchedDistance[bestIdx2] = bestDist;
					nmatches++;

					// Step 5 计算匹配点旋转角度差所在的直方图
					if (mbCheckOrientation)
					{
						// 计算匹配特征点的角度差，这里单位是角度°，不是弧度
						float rot = F1.mvKeys[i1].angle - F2.mvKeys[bestIdx2].angle;
						if (rot < 0.0)
							rot += 360.0f;
						// 前面factor = HISTO_LENGTH/360.0f 
						// bin = rot / 360.of * HISTO_LENGTH 表示当前rot被分配在第几个直方图bin  
						int bin = std::floor(rot*factor);
						// 如果bin 满了又是一个轮回
						if (bin == HISTO_LENGTH)
							bin = 0;
						assert(bin >= 0 && bin < HISTO_LENGTH);
						rotHist[bin].push_back(i1);
					}
				}
			}

		}

		// Step 6 筛除旋转直方图中“非主流”部分
		if (mbCheckOrientation)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;
			// 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				// 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”    
				for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
				{
					int idx1 = rotHist[i][j];
					if (vnMatches12[idx1] >= 0)
					{
						vnMatches12[idx1] = -1;
						nmatches--;
					}
				}
			}
		}

		return nmatches;
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

	/**
	 * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
	 *
	 * @param[in] histo         匹配特征点对旋转方向差直方图
	 * @param[in] L             直方图尺寸
	 * @param[in & out] ind1          bin值第一大对应的索引
	 * @param[in & out] ind2          bin值第二大对应的索引
	 * @param[in & out] ind3          bin值第三大对应的索引
	 */
	void ORBmatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
	{
		int max1 = 0;
		int max2 = 0;
		int max3 = 0;

		for (int i = 0; i < L; i++)
		{
			const int s = histo[i].size();
			if (s > max1)
			{
				max3 = max2;
				max2 = max1;
				max1 = s;
				ind3 = ind2;
				ind2 = ind1;
				ind1 = i;
			}
			else if (s > max2)
			{
				max3 = max2;
				max2 = s;
				ind3 = ind2;
				ind2 = i;
			}
			else if (s > max3)
			{
				max3 = s;
				ind3 = i;
			}
		}

		// 如果差距太大了,说明次优的非常不好,这里就索性放弃了,都置为-1
		if (max2 < 0.1f*(float)max1)
		{
			ind2 = -1;
			ind3 = -1;
		}
		else if (max3 < 0.1f*(float)max1)
		{
			ind3 = -1;
		}
	}

	// Bit set count operation from
	// Hamming distance：两个二进制串之间的汉明距离，指的是其不同位数的个数
	// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
	{
		const int *pa = a.ptr<int32_t>();
		const int *pb = b.ptr<int32_t>();

		int dist = 0;
		// 8*32=256bit
		for (int i = 0; i < 8; i++, pa++, pb++)
		{
			unsigned  int v = *pa ^ *pb;        // 相等为0,不等为1
			// 下面的操作就是计算其中bit为1的个数了,这个操作看上面的链接就好
			// 其实我觉得也还阔以直接使用8bit的查找表,然后做32次寻址操作就完成了;不过缺点是没有利用好CPU的字长
			v = v - ((v >> 1) & 0x55555555);
			v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
			dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
		}
		return dist;
	}
}