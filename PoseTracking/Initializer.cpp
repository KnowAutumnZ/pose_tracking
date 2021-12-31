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
		mvSets = std::vector< std::vector<size_t> >(mMaxIterations, std::vector<size_t>(20, 0));

		//用于进行随机数据样本采样，设置随机数种子
		SeedRandOnce(0);

		//开始每一次的迭代 
		for (int it = 0; it < mMaxIterations; it++)
		{
			//迭代开始的时候，所有的点都是可用的
			vAvailableIndices = vAllIndices;

			//选择最小的数据样本集，使用八点法求，所以这里就循环了八次
			for (size_t j = 0; j < 20; j++)
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

	/**
	 * @brief 计算单应矩阵，假设场景为平面情况下通过前两帧求取Homography矩阵，并得到该模型的评分
	 * 原理参考Multiple view geometry in computer vision  P109 算法4.4
	 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
	 * Step 2 选择8个归一化之后的点对进行迭代
	 * Step 3 八点法计算单应矩阵矩阵
	 * Step 4 利用重投影误差为当次RANSAC的结果评分
	 * Step 5 更新具有最优评分的单应矩阵计算结果,并且保存所对应的特征点对的内点标记
	 *
	 * @param[in & out] vbMatchesInliers          标记是否是外点
	 * @param[in & out] score                     计算单应矩阵的得分
	 * @param[in & out] H21                       单应矩阵结果
	 */
	void Initializer::FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
	{
		//匹配的特征点对总数
		const int N = mvMatches12.size();

		// Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
		// 具体来说,就是将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
		// 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值
		// 归一化矩阵就是把上述归一化的操作用矩阵来表示。这样特征点坐标乘归一化矩阵可以得到归一化后的坐标

		//归一化后的参考帧1和当前帧2中的特征点坐标
		std::vector<cv::Point2f> vPn1, vPn2;

		// 记录各自的归一化矩阵
		cv::Mat T1, T2;
		Normalize(mvKeys1, vPn1, T1);
		Normalize(mvKeys2, vPn2, T2);

		//这里求的逆在后面的代码中要用到，辅助进行原始尺度的恢复
		cv::Mat T2inv = T2.inv();

		// 记录最佳评分
		score = 0.0;
		// 取得历史最佳评分时,特征点对的inliers标记
		vbMatchesInliers = std::vector<bool>(N, false);

		//某次迭代中，参考帧的特征点坐标
		std::vector<cv::Point2f> vPn1i(20);
		//某次迭代中，当前帧的特征点坐标
		std::vector<cv::Point2f> vPn2i(20);
		//以及计算出来的单应矩阵、及其逆矩阵
		cv::Mat H21i, H12i;

		// 每次RANSAC记录Inliers与得分
		std::vector<bool> vbCurrentInliers(N, false);
		float currentScore;

		//下面进行每次的RANSAC迭代
		for (int it = 0; it < mMaxIterations; it++)
		{
			// Step 2 选择8个归一化之后的点对进行迭代
			for (size_t j = 0; j < 20; j++)
			{
				//从mvSets中获取当前次迭代的某个特征点对的索引信息
				int idx = mvSets[it][j];

				// vPn1i和vPn2i为匹配的特征点对的归一化后的坐标
				// 首先根据这个特征点对的索引信息分别找到两个特征点在各自图像特征点向量中的索引，然后读取其归一化之后的特征点坐标
				vPn1i[j] = vPn1[mvMatches12[idx].first];    //first存储在参考帧1中的特征点索引
				vPn2i[j] = vPn2[mvMatches12[idx].second];   //second存储在参考帧1中的特征点索引
			}//读取8对特征点的归一化之后的坐标

			// Step 3 八点法计算单应矩阵
			// 利用生成的8个归一化特征点对, 调用函数 Initializer::ComputeH21() 使用八点法计算单应矩阵  
			// 关于为什么计算之前要对特征点进行归一化，后面又恢复这个矩阵的尺度？
			// 可以在《计算机视觉中的多视图几何》这本书中P193页中找到答案
			// 书中这里说,8点算法成功的关键是在构造解的方称之前应对输入的数据认真进行适当的归一化
			cv::Mat Hn = ComputeH21(vPn1i, vPn2i);

			// 单应矩阵原理：X2=H21*X1，其中X1,X2 为归一化后的特征点    
			// 特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  得到:T2 * mvKeys2 =  Hn * T1 * mvKeys1   
			// 进一步得到:mvKeys2  = T2.inv * Hn * T1 * mvKeys1
			H21i = T2inv * Hn*T1;
			//然后计算逆
			H12i = H21i.inv();

			// Step 4 利用重投影误差为当次RANSAC的结果评分
			currentScore = CheckHomography(H21i, H12i, 			//输入，单应矩阵的计算结果
				vbCurrentInliers, 								//输出，特征点对的Inliers标记
				mSigma);										//TODO  测量误差，在Initializer类对象构造的时候，由外部给定的

			// Step 5 更新具有最优评分的单应矩阵计算结果,并且保存所对应的特征点对的内点标记
			if (currentScore > score)
			{
				//如果当前的结果得分更高，那么就更新最优计算结果
				H21 = H21i.clone();
				//保存匹配好的特征点对的Inliers标记
				vbMatchesInliers = vbCurrentInliers;
				//更新历史最优评分
				score = currentScore;
			}
		}
	}

	/**
	 * @brief 计算基础矩阵，假设场景为非平面情况下通过前两帧求取Fundamental矩阵，得到该模型的评分
	 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
	 * Step 2 选择8个归一化之后的点对进行迭代
	 * Step 3 八点法计算基础矩阵矩阵
	 * Step 4 利用重投影误差为当次RANSAC的结果评分
	 * Step 5 更新具有最优评分的基础矩阵计算结果,并且保存所对应的特征点对的内点标记
	 *
	 * @param[in & out] vbMatchesInliers          标记是否是外点
	 * @param[in & out] score                     计算基础矩阵得分
	 * @param[in & out] F21                       从特征点1到2的基础矩阵
	 */
	void Initializer::FindFundamental(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
	{
		// 计算基础矩阵,其过程和上面的计算单应矩阵的过程十分相似.

		// 匹配的特征点对总数
		// const int N = vbMatchesInliers.size();  // !源代码出错！请使用下面代替
		const int N = mvMatches12.size();
		// Normalize coordinates
		// Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
		// 具体来说,就是将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
		// 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值
		// 归一化矩阵就是把上述归一化的操作用矩阵来表示。这样特征点坐标乘归一化矩阵可以得到归一化后的坐标

		std::vector<cv::Point2f> vPn1, vPn2;
		cv::Mat T1, T2;
		Normalize(mvKeys1, vPn1, T1);
		Normalize(mvKeys2, vPn2, T2);

		// ! 注意这里取的是归一化矩阵T2的转置,因为基础矩阵的定义和单应矩阵不同，两者去归一化的计算也不相同
		cv::Mat T2t = T2.t();







	}

	/**
	 * @brief 用DLT方法求解单应矩阵H
	 * 这里最少用4对点就能够求出来，不过这里为了统一还是使用了8对点求最小二乘解
	 *
	 * @param[in] vP1               参考帧中归一化后的特征点
	 * @param[in] vP2               当前帧中归一化后的特征点
	 * @return cv::Mat              计算的单应矩阵H
	 */
	cv::Mat Initializer::ComputeH21(
		const std::vector<cv::Point2f> &vP1, //归一化后的点, in reference frame
		const std::vector<cv::Point2f> &vP2) //归一化后的点, in current frame
	{
		// 基本原理：见附件推导过程：
		// |x'|     | h1 h2 h3 ||x|
		// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
		// |1 |     | h7 h8 h9 ||1|
		// 使用DLT(direct linear tranform)求解该模型
		// x' = a H x 
		// ---> (x') 叉乘 (H x)  = 0  (因为方向相同) (取前两行就可以推导出下面的了)
		// ---> Ah = 0 
		// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
		//     |-x -y -1  0  0  0 xx' yx' x'|
		// 通过SVD求解Ah = 0，A^T*A最小特征值对应的特征向量即为解
		// 其实也就是右奇异值矩阵的最后一列

		//获取参与计算的特征点的数目
		const int N = vP1.size();

		// 构造用于计算的矩阵 A 
		cv::Mat A(2 * N,				//行，注意每一个点的数据对应两行
			9,							//列
			CV_32F);      				//float数据类型

		// 构造矩阵A，将每个特征点添加到矩阵A中的元素
		for (int i = 0; i < N; i++)
		{
			//获取特征点对的像素坐标
			const float u1 = vP1[i].x;
			const float v1 = vP1[i].y;
			const float u2 = vP2[i].x;
			const float v2 = vP2[i].y;

			//生成这个点的第一行
			A.at<float>(2 * i, 0) = 0.0;
			A.at<float>(2 * i, 1) = 0.0;
			A.at<float>(2 * i, 2) = 0.0;
			A.at<float>(2 * i, 3) = -u1;
			A.at<float>(2 * i, 4) = -v1;
			A.at<float>(2 * i, 5) = -1;
			A.at<float>(2 * i, 6) = v2 * u1;
			A.at<float>(2 * i, 7) = v2 * v1;
			A.at<float>(2 * i, 8) = v2;

			//生成这个点的第二行
			A.at<float>(2 * i + 1, 0) = u1;
			A.at<float>(2 * i + 1, 1) = v1;
			A.at<float>(2 * i + 1, 2) = 1;
			A.at<float>(2 * i + 1, 3) = 0.0;
			A.at<float>(2 * i + 1, 4) = 0.0;
			A.at<float>(2 * i + 1, 5) = 0.0;
			A.at<float>(2 * i + 1, 6) = -u2 * u1;
			A.at<float>(2 * i + 1, 7) = -u2 * v1;
			A.at<float>(2 * i + 1, 8) = -u2;
		}

		// 定义输出变量，u是左边的正交矩阵U， w为奇异矩阵，vt中的t表示是右正交矩阵V的转置
		cv::Mat u, w, vt;

		//使用opencv提供的进行奇异值分解的函数
		cv::SVDecomp(A,							//输入，待进行奇异值分解的矩阵
			w,									//输出，奇异值矩阵
			u,									//输出，矩阵U
			vt,									//输出，矩阵V^T
			cv::SVD::MODIFY_A | 				//输入，MODIFY_A是指允许计算函数可以修改待分解的矩阵，官方文档上说这样可以加快计算速度、节省内存
			cv::SVD::FULL_UV);					//FULL_UV=把U和VT补充成单位正交方阵

		// 返回最小奇异值所对应的右奇异向量
		// 注意前面说的是右奇异值矩阵的最后一列，但是在这里因为是vt，转置后了，所以是行；由于A有9列数据，故最后一列的下标为8
		return vt.row(8).reshape(0, 			//转换后的通道数，这里设置为0表示是与前面相同
			3); 								//转换后的行数,对应V的最后一列
	}

	/**
	 * @brief 对给定的homography matrix打分,需要使用到卡方检验的知识
	 *
	 * @param[in] H21                       从参考帧到当前帧的单应矩阵
	 * @param[in] H12                       从当前帧到参考帧的单应矩阵
	 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
	 * @param[in] sigma                     方差，默认为1
	 * @return float                        返回得分
	 */
	float Initializer::CheckHomography(
		const cv::Mat &H21,						 //从参考帧到当前帧的单应矩阵
		const cv::Mat &H12,                      //从当前帧到参考帧的单应矩阵
		std::vector<bool> &vbMatchesInliers,     //匹配好的特征点对的Inliers标记
		float sigma)                             //估计误差
	{
		// 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
		// 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
		// 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
		// 误差加权最小二次结果越小，说明观测数据精度越高
		// 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
		// 算法目标： 检查单应变换矩阵
		// 检查方式：通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差

		// 算法流程
		// input: 单应性矩阵 H21, H12, 匹配点集 mvKeys1
		//    do:
		//        for p1(i), p2(i) in mvKeys:
		//           error_i1 = ||p2(i) - H21 * p1(i)||2
		//           error_i2 = ||p1(i) - H12 * p2(i)||2
		//           
		//           w1 = 1 / sigma / sigma
		//           w2 = 1 / sigma / sigma
		// 
		//           if error1 < th
		//              score +=   th - error_i1 * w1
		//           if error2 < th
		//              score +=   th - error_i2 * w2
		// 
		//           if error_1i > th or error_2i > th
		//              p1(i), p2(i) are inner points
		//              vbMatchesInliers(i) = true
		//           else 
		//              p1(i), p2(i) are outliers
		//              vbMatchesInliers(i) = false
		//           end
		//        end
		//   output: score, inliers

		// 特点匹配个数
		const int N = mvMatches12.size();

		// Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
		const float h11 = H21.at<float>(0, 0);
		const float h12 = H21.at<float>(0, 1);
		const float h13 = H21.at<float>(0, 2);
		const float h21 = H21.at<float>(1, 0);
		const float h22 = H21.at<float>(1, 1);
		const float h23 = H21.at<float>(1, 2);
		const float h31 = H21.at<float>(2, 0);
		const float h32 = H21.at<float>(2, 1);
		const float h33 = H21.at<float>(2, 2);

		// 获取从当前帧到参考帧的单应矩阵的各个元素
		const float h11inv = H12.at<float>(0, 0);
		const float h12inv = H12.at<float>(0, 1);
		const float h13inv = H12.at<float>(0, 2);
		const float h21inv = H12.at<float>(1, 0);
		const float h22inv = H12.at<float>(1, 1);
		const float h23inv = H12.at<float>(1, 2);
		const float h31inv = H12.at<float>(2, 0);
		const float h32inv = H12.at<float>(2, 1);
		const float h33inv = H12.at<float>(2, 2);

		// 给特征点对的Inliers标记预分配空间
		vbMatchesInliers.resize(N);

		// 初始化score值
		float score = 0;

		// 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
		// 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
		const float th = 5.991;

		//信息矩阵，方差平方的倒数
		const float invSigmaSquare = 1.0 / (sigma * sigma);

		// Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权重投影误差
		// H21 表示从img1 到 img2的变换矩阵
		// H12 表示从img2 到 img1的变换矩阵 
		for (int i = 0; i < N; i++)
		{
			// 一开始都默认为Inlier
			bool bIn = true;

			// Step 2.1 提取参考帧和当前帧之间的特征匹配点对
			const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
			const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
			const float u1 = kp1.pt.x;
			const float v1 = kp1.pt.y;
			const float u2 = kp2.pt.x;
			const float v2 = kp2.pt.y;

			// Step 2.2 计算 img2 到 img1 的重投影误差
			// x1 = H12*x2
			// 将图像2中的特征点通过单应变换投影到图像1中
			// |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
			// |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
			// |1 |   |h31inv h32inv h33inv||1 |   |  1  |
			// 计算投影归一化坐标
			const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
			const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
			const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

			// 计算重投影误差 = ||p1(i) - H12 * p2(i)||2
			const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
			const float chiSquare1 = squareDist1 * invSigmaSquare;

			// Step 2.3 用阈值标记离群点，内点的话累加得分
			if (chiSquare1 > th)
				bIn = false;
			else
				// 误差越大，得分越低
				score += th - chiSquare1;

			// 计算从img1 到 img2 的投影变换误差
			// x1in2 = H21*x1
			// 将图像2中的特征点通过单应变换投影到图像1中
			// |u2|   |h11 h12 h13||u1|   |u1in2|
			// |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
			// |1 |   |h31 h32 h33||1 |   |  1  |
			// 计算投影归一化坐标
			const float w1in2inv = 1.0 / (h31*u1 + h32 * v1 + h33);
			const float u1in2 = (h11*u1 + h12 * v1 + h13)*w1in2inv;
			const float v1in2 = (h21*u1 + h22 * v1 + h23)*w1in2inv;

			// 计算重投影误差 
			const float squareDist2 = (u2 - u1in2)*(u2 - u1in2) + (v2 - v1in2)*(v2 - v1in2);
			const float chiSquare2 = squareDist2 * invSigmaSquare;

			// 用阈值标记离群点，内点的话累加得分
			if (chiSquare2 > th)
				bIn = false;
			else
				score += th - chiSquare2;

			// Step 2.4 如果从img2 到 img1 和 从img1 到img2的重投影误差均满足要求，则说明是Inlier point
			if (bIn)
				vbMatchesInliers[i] = true;
			else
				vbMatchesInliers[i] = false;
		}
		return score;
	}

	/**
	 * @brief 归一化特征点到同一尺度，作为后续normalize DLT的输入
	 *  [x' y' 1]' = T * [x y 1]'
	 *  归一化后x', y'的均值为0，sum(abs(x_i'-0))=1，sum(abs((y_i'-0))=1
	 *
	 *  为什么要归一化？
	 *  在相似变换之后(点在不同的坐标系下),他们的单应性矩阵是不相同的
	 *  如果图像存在噪声,使得点的坐标发生了变化,那么它的单应性矩阵也会发生变化
	 *  我们采取的方法是将点的坐标放到同一坐标系下,并将缩放尺度也进行统一
	 *  对同一幅图像的坐标进行相同的变换,不同图像进行不同变换
	 *  缩放尺度是为了让噪声对于图像的影响在一个数量级上
	 *
	 *  Step 1 计算特征点X,Y坐标的均值
	 *  Step 2 计算特征点X,Y坐标离均值的平均偏离程度
	 *  Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1
	 *  Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
	 *
	 * @param[in] vKeys                               待归一化的特征点
	 * @param[in & out] vNormalizedPoints             特征点归一化后的坐标
	 * @param[in & out] T                             归一化特征点的变换矩阵
	 */
	void Initializer::Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) 
	{
		// 归一化的是这些点在x方向和在y方向上的一阶绝对矩（随机变量的期望）。

		// Step 1 计算特征点X,Y坐标的均值 meanX, meanY
		float meanX = 0;
		float meanY = 0;

		//获取特征点的数量
		const int N = vKeys.size();

		//设置用来存储归一后特征点的向量大小，和归一化前保持一致
		vNormalizedPoints.resize(N);

		//开始遍历所有的特征点
		for (int i = 0; i < N; i++)
		{
			//分别累加特征点的X、Y坐标
			meanX += vKeys[i].pt.x;
			meanY += vKeys[i].pt.y;
		}

		//计算X、Y坐标的均值
		meanX = meanX / N;
		meanY = meanY / N;

		// Step 2 计算特征点X,Y坐标离均值的平均偏离程度 meanDevX, meanDevY，注意不是标准差
		float meanDevX = 0;
		float meanDevY = 0;

		// 将原始特征点减去均值坐标，使x坐标和y坐标均值分别为0
		for (int i = 0; i < N; i++)
		{
			vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
			vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

			//累计这些特征点偏离横纵坐标均值的程度
			meanDevX += fabs(vNormalizedPoints[i].x);
			meanDevY += fabs(vNormalizedPoints[i].y);
		}

		// 求出平均到每个点上，其坐标偏离横纵坐标均值的程度；将其倒数作为一个尺度缩放因子
		meanDevX = meanDevX / N;
		meanDevY = meanDevY / N;
		float sX = 1.0 / meanDevX;
		float sY = 1.0 / meanDevY;

		// Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1 
		// 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值（期望）
		for (int i = 0; i < N; i++)
		{
			//对，就是简单地对特征点的坐标进行进一步的缩放
			vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
			vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
		}

		// Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
		// |sX  0  -meanx*sX|
		// |0   sY -meany*sY|
		// |0   0      1    |
		T = cv::Mat::eye(3, 3, CV_32F);
		T.at<float>(0, 0) = sX;
		T.at<float>(1, 1) = sY;
		T.at<float>(0, 2) = -meanX * sX;
		T.at<float>(1, 2) = -meanY * sY;
	}
}