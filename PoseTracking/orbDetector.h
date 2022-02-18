#pragma once

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace PoseTracking
{
	/**
	* @brief 提取器节点
	* @details 用于在特征点的分配过程中。
	*
	*/
	// 分配四叉树时用到的结点类型
	class DetectorNode
	{
	public:
		/** @brief 构造函数 */
		DetectorNode() :bNoMore(false) {}

		/**
		* @brief 在八叉树分配特征点的过程中，实现一个节点分裂为4个节点的操作
		*
		* @param[out] n1   分裂的节点1
		* @param[out] n2   分裂的节点2
		* @param[out] n3   分裂的节点3
		* @param[out] n4   分裂的节点4
		*/
		void DivideNode(DetectorNode &n1, DetectorNode &n2, DetectorNode &n3, DetectorNode &n4);

		//保存有当前节点的特征点
		std::vector<cv::KeyPoint> vKeys;

		//当前节点所对应的图像坐标边界
		cv::Point2i UL, UR, BL, BR;

		//这个迭代器提供了访问总节点列表的方式，需要结合cpp文件进行分析
		std::list<DetectorNode>::iterator iter;

		//如果节点中只有一个特征点的话，说明这个节点不能够再进行分裂了，这个标志置位
		//这个节点中如果没有特征点的话，这个节点就直接被删除了
		bool bNoMore;
	};

	/**
	* @brief ORB特征点提取器
	*
	*/
	class orbDetector
	{
	public:
		/**
		* @brief 构造函数
		* @detials 之所以会有两种响应值的阈值，原因是，程序先使用初始的默认FAST响应值阈值提取图像cell中的特征点；如果提取到的
		* 特征点数目不足，那么就降低要求，使用较小FAST响应值阈值进行再次提取，以获得尽可能多的FAST角点。
		* @param[in] nfeatures         指定要提取出来的特征点数目
		* @param[in] scaleFactor       图像金字塔的缩放系数
		* @param[in] nlevels           指定需要提取特征点的图像金字塔层
		* @param[in] iniThFAST         初始的默认FAST响应值阈值
		* @param[in] minThFAST         较小的FAST响应值阈值
		*/
		orbDetector(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

		/**
		* @brief 使用八叉树的方法将提取到的ORB特征点尽可能均匀地分布在整个图像中
		* @details 这里是重载了这个ORBextractor类的括号运算符;函数中实际上并没有用到MASK这个参数。
		*
		* @param[in] image         要操作的图像
		* @param[in] mask          图像掩膜，辅助进行图片处理，可以参考[https://www.cnblogs.com/skyfsm/p/6894685.html]
		* @param[out] keypoints    保存提取出来的特征点的向量
		* @param[out] descriptors  输出用的保存特征点描述子的cv::Mat
		*/
		void operator()(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

		//下面的这些内联函数都是用来直接获取类的成员变量的

		/**
		* @brief 获取图像金字塔的层数
		* @return int 图像金字塔的层数
		*/
		int inline GetLevels() {
			return mnlevels;
		}

		/**
		* @brief 获取当前提取器所在的图像的缩放因子，这个不带s的因子表示是相临近层之间的
		* @return float 当前提取器所在的图像的缩放因子，相邻层之间
		*/
		float inline GetScaleFactor() {
			return mscaleFactor;
		}

		/**
		* @brief 获取图像金字塔中每个图层相对于底层图像的缩放因子
		* @return std::vector<float> 图像金字塔中每个图层相对于底层图像的缩放因子
		*/
		std::vector<float> inline GetScaleFactors() {
			return mvScaleFactor;
		}

		/**
		* @brief 获取上面的那个缩放因子s的倒数
		* @return std::vector<float> 倒数
		*/
		std::vector<float> inline GetInverseScaleFactors() {
			return mvInvScaleFactor;
		}

		/**
		* @brief 获取sigma^2，就是每层图像相对于初始图像缩放因子的平方，参考cpp文件中类构造函数的操作
		* @return std::vector<float> sigma^2
		*/
		std::vector<float> inline GetScaleSigmaSquares() {
			return mvLevelSigma2;
		}

		/**
		* @brief 获取上面sigma平方的倒数
		* @return std::vector<float>
		*/
		std::vector<float> inline GetInverseScaleSigmaSquares() {
			return mvInvLevelSigma2;
		}

		//这个是用来存储图像金字塔的变量，一个元素存储一层图像
		std::vector<cv::Mat> mvImagePyramid;

	private:
		/**
		* @brief 针对给出的一张图像，计算其图像金字塔
		* @param[in] image 给出的图像
		*/
		void ComputePyramid(cv::Mat& image);

		/**
		* @brief 计算某层金字塔图像上特征点的描述子
		*
		* @param[in] image                 某层金字塔图像
		* @param[in] keypoints             特征点vector容器
		* @param[out] descriptors          描述子
		* @param[in] pattern               计算描述子使用的固定随机点集
		*/
		void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const std::vector<cv::Point>& pattern);

		/**
		* @brief 计算ORB特征点的描述子。注意这个是全局的静态函数，只能是在本文件内被调用
		* @param[in] kpt       特征点对象
		* @param[in] img       提取特征点的图像
		* @param[in] pattern   预定义好的采样模板
		* @param[out] desc     用作输出变量，保存计算好的描述子，维度为32*8 = 256 bit
		*/
		void computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern, uchar* desc);

		/**
		* @brief 以八叉树分配特征点的方式，计算图像金字塔中的特征点
		* @detials 这里两层vector的意思是，第一层存储的是某张图片中的所有特征点，而第二层则是存储图像金字塔中所有图像的vectors of keypoints
		* @param[out] allKeypoints 提取得到的所有特征点
		*/
		void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

		/**
		* @brief 对于某一图层，分配其特征点，通过八叉树的方式
		* @param[in] vToDistributeKeys         等待分配的特征点
		* @param[in] minX                      分发的图像范围
		* @param[in] maxX                      分发的图像范围
		* @param[in] minY                      分发的图像范围
		* @param[in] maxY                      分发的图像范围
		* @param[in] nFeatures                 设定的、本图层中想要提取的特征点数目
		* @param[in] level                     要提取的图像所在的金字塔层
		* @return std::vector<cv::KeyPoint>
		*/
		std::vector<cv::KeyPoint> DistributeOctTree(
			const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX, const int &maxX, const int &minY, const int &maxY,
			const int &nFeatures, const int &level);

		/**
		* @brief 计算特征点的方向
		* @param[in] image                 特征点所在当前金字塔的图像
		* @param[in & out] keypoints       特征点向量
		* @param[in] umax                  每个特征点所在图像区块的每行的边界 u_max 组成的vector
		*/
		void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);

		/**
		* @brief 这个函数用于计算特征点的方向，这里是返回角度作为方向。
		* 计算特征点方向是为了使得提取的特征点具有旋转不变性。
		* 方法是灰度质心法：以几何中心和灰度质心的连线作为该特征点方向
		* @param[in] image     要进行操作的某层金字塔图像
		* @param[in] pt        当前特征点的坐标
		* @param[in] u_max     图像块的每一行的坐标边界 u_max
		* @return float        返回特征点的角度，范围为[0,360)角度，精度为0.3°
		*/
		float IC_Angle(const cv::Mat& image, cv::Point2f pt, const std::vector<int> & u_max);

		std::vector<cv::Point> mvpattern;           //<用于计算描述子的随机采样点集合

		int mnfeatures;			                    //<整个图像金字塔中，要提取的特征点数目
		double mscaleFactor;		                //<图像金字塔层与层之间的缩放因子
		int mnlevels;			                    //<图像金字塔的层数
		int miniThFAST;			                    //<初始的FAST响应值阈值
		int minThFAST;			                    //<最小的FAST响应值阈值

		std::vector<int> mv_nFeaturesPerLevel;		//<分配到每层图像中，要提取的特征点数目

		std::vector<int> mv_umax;	                //<计算特征点方向的时候，有个圆形的图像区域，这个vector中存储了每行u轴的边界（四分之一，其他部分通过对称获得）

		std::vector<float> mvScaleFactor;		    //<每层图像的缩放因子
		std::vector<float> mvInvScaleFactor;        //<以及每层缩放因子的倒数
		std::vector<float> mvLevelSigma2;		    //<存储每层的sigma^2,即上面每层图像相对于底层图像缩放倍数的平方
		std::vector<float> mvInvLevelSigma2;	    //<sigma平方的倒数

		const int PATCH_SIZE = 31;			//<使用灰度质心法计算特征点的方向信息时，图像块的大小,或者说是直径
		const int HALF_PATCH_SIZE = 15;		//<上面这个大小的一半，或者说是半径
		const int EDGE_THRESHOLD = 15;
};

}