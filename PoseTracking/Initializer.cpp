#include "Initializer.h"

namespace PoseTracking
{
	/**
	 * @brief 根据参考帧构造初始化器
	 *
	 * @param[in] ReferenceFrame        参考帧
	 * @param[in] sigma                 测量误差
	 * @param[in] iterations            RANSAC迭代次数
	 */
	Initializer::Initializer(const cv::Mat& K, const Frame &ReferenceFrame, float sigma, int iterations)
	{
		//从参考帧中获取相机的内参数矩阵
		mK = K.clone();

		// 从参考帧中获取去畸变后的特征点
		mvKeys1 = ReferenceFrame.mvKeys;

		//获取估计误差
		mSigma = sigma;
		mSigma2 = sigma * sigma;

		//最大迭代次数
		mMaxIterations = iterations;
	}

	/**
	 * @brief 计算基础矩阵和单应性矩阵，选取最佳的来恢复出最开始两帧之间的相对姿态，并进行三角化得到初始地图点
	 * Step 1 重新记录特征点对的匹配关系
	 * Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
	 * Step 3 计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算
	 * Step 4 计算得分比例来判断选取哪个模型来求位姿R,t
	 *
	 * @param[in] CurrentFrame          当前帧，也就是SLAM意义上的第二帧
	 * @param[in] vMatches12            当前帧（2）和参考帧（1）图像中特征点的匹配关系
	 *                                  vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
	 *                                  没有匹配关系的话，vMatches12[i]值为 -1
	 * @param[in & out] R21                   相机从参考帧到当前帧的旋转
	 * @param[in & out] t21                   相机从参考帧到当前帧的平移
	 * @param[in & out] vP3D                  三角化测量之后的三维地图点
	 * @param[in & out] vbTriangulated        标记三角化点是否有效，有效为true
	 * @return true                     该帧可以成功初始化，返回true
	 * @return false                    该帧不满足初始化条件，返回false
	 */
	bool Initializer::Initialize(const Frame &CurrentFrame, const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
		std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated)
	{
		//获取当前帧的去畸变之后的特征点
		mvKeys2 = CurrentFrame.mvKeys;

		// mvMatches12记录匹配上的特征点对，记录的是帧2在帧1的匹配索引
		mvMatches12.clear();
		// 预分配空间，大小和关键点数目一致mvKeys2.size()
		mvMatches12.reserve(mvKeys2.size());

		// Step 1 重新记录特征点对的匹配关系存储在mvMatches12，是否有匹配存储在mvbMatched1
		// 将vMatches12（有冗余） 转化为 mvMatches12（只记录了匹配关系）
		for (size_t i=0; i<vMatches12.size(); i++)
		{
			//没有匹配关系的话，vMatches12[i]值为 -1
			if (vMatches12[i] >= 0)
			{
				//mvMatches12 中只记录有匹配关系的特征点对的索引值
				//i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
				mvMatches12.push_back(std::make_pair(i, vMatches12[i]));
			}
		}

		// 有匹配的特征点的对数
		const int N = mvMatches12.size();
		// Indices for minimum set selection
		// 新建一个容器vAllIndices存储特征点索引，并预分配空间
		std::vector<size_t> vAllIndices;
		//初始化所有特征点对的索引，索引值0到N-1
		for (int i = 0; i < N; i++)
			vAllIndices.push_back(i);

		//在RANSAC的某次迭代中，还可以被抽取来作为数据样本的特征点对的索引，所以这里起的名字叫做可用的索引
		std::vector<size_t> vAvailableIndices;

		// Step 2 在所有匹配特征点对中随机选择8对匹配特征点为一组，用于估计H矩阵和F矩阵
		// 共选择 mMaxIterations (默认200) 组
		//mvSets用于保存每次迭代时所使用的向量
		mvSets = std::vector< std::vector<size_t> >(mMaxIterations, std::vector<size_t>(8, 0));

		//用于进行随机数据样本采样，设置随机数种子
		SeedRandOnce(0);

		//开始每一次的迭代 
		for (int it = 0; it < mMaxIterations; it++)
		{
			//迭代开始的时候，所有的点都是可用的
			vAvailableIndices = vAllIndices;

			//选择最小的数据样本集，使用八点法求，所以这里就循环了八次
			for (size_t j = 0; j < 8; j++)
			{
				// 随机产生一对点的id,范围从0到N-1
				int randi = RandomInt(0, vAvailableIndices.size() - 1);
				// idx表示哪一个索引对应的特征点对被选中
				int idx = vAvailableIndices[randi];

				//将本次迭代这个选中的第j个特征点对的索引添加到mvSets中
				mvSets[it][j] = idx;

				// 由于这对点在本次迭代中已经被使用了,所以我们为了避免再次抽到这个点,就在"点的可选列表"中,
				// 将这个点原来所在的位置用vector最后一个元素的信息覆盖,并且删除尾部的元素
				// 这样就相当于将这个点的信息从"点的可用列表"中直接删除了
				vAvailableIndices[randi] = vAvailableIndices.back();
				vAvailableIndices.pop_back();

				//std::vector<size_t>::iterator it = std::find(vAvailableIndices.begin(), vAvailableIndices.end(), idx);
				//vAvailableIndices.erase(it);
			}//依次提取出8个特征点对
		}//迭代mMaxIterations次，选取各自迭代时需要用到的最小数据集

		// Step 3 计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算 
		//这两个变量用于标记在H和F的计算中哪些特征点对被认为是Inlier
		std::vector<bool> vbMatchesInliersH, vbMatchesInliersF;
		//计算出来的单应矩阵和基础矩阵的RANSAC评分，这里其实是采用重投影误差来计算的
		float SH, SF; //score for H and F
		//这两个是经过RANSAC算法后计算出来的单应矩阵和基础矩阵
		cv::Mat H, F;







	}




}