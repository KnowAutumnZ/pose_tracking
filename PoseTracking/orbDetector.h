#pragma once

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace PoseTracking
{
	/**
	* @brief ��ȡ���ڵ�
	* @details ������������ķ�������С�
	*
	*/
	// �����Ĳ���ʱ�õ��Ľ������
	class DetectorNode
	{
	public:
		/** @brief ���캯�� */
		DetectorNode() :bNoMore(false) {}

		/**
		* @brief �ڰ˲�������������Ĺ����У�ʵ��һ���ڵ����Ϊ4���ڵ�Ĳ���
		*
		* @param[out] n1   ���ѵĽڵ�1
		* @param[out] n2   ���ѵĽڵ�2
		* @param[out] n3   ���ѵĽڵ�3
		* @param[out] n4   ���ѵĽڵ�4
		*/
		void DivideNode(DetectorNode &n1, DetectorNode &n2, DetectorNode &n3, DetectorNode &n4);

		//�����е�ǰ�ڵ��������
		std::vector<cv::KeyPoint> vKeys;

		//��ǰ�ڵ�����Ӧ��ͼ������߽�
		cv::Point2i UL, UR, BL, BR;

		//����������ṩ�˷����ܽڵ��б�ķ�ʽ����Ҫ���cpp�ļ����з���
		std::list<DetectorNode>::iterator iter;

		//����ڵ���ֻ��һ��������Ļ���˵������ڵ㲻�ܹ��ٽ��з����ˣ������־��λ
		//����ڵ������û��������Ļ�������ڵ��ֱ�ӱ�ɾ����
		bool bNoMore;
	};

	/**
	* @brief ORB��������ȡ��
	*
	*/
	class orbDetector
	{
	public:
		/**
		* @brief ���캯��
		* @detials ֮���Ի���������Ӧֵ����ֵ��ԭ���ǣ�������ʹ�ó�ʼ��Ĭ��FAST��Ӧֵ��ֵ��ȡͼ��cell�е������㣻�����ȡ����
		* ��������Ŀ���㣬��ô�ͽ���Ҫ��ʹ�ý�СFAST��Ӧֵ��ֵ�����ٴ���ȡ���Ի�þ����ܶ��FAST�ǵ㡣
		* @param[in] nfeatures         ָ��Ҫ��ȡ��������������Ŀ
		* @param[in] scaleFactor       ͼ�������������ϵ��
		* @param[in] nlevels           ָ����Ҫ��ȡ�������ͼ���������
		* @param[in] iniThFAST         ��ʼ��Ĭ��FAST��Ӧֵ��ֵ
		* @param[in] minThFAST         ��С��FAST��Ӧֵ��ֵ
		*/
		orbDetector(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

		/**
		* @brief ʹ�ð˲����ķ�������ȡ����ORB�����㾡���ܾ��ȵطֲ�������ͼ����
		* @details ���������������ORBextractor������������;������ʵ���ϲ�û���õ�MASK���������
		*
		* @param[in] image         Ҫ������ͼ��
		* @param[in] mask          ͼ����Ĥ����������ͼƬ�������Բο�[https://www.cnblogs.com/skyfsm/p/6894685.html]
		* @param[out] keypoints    ������ȡ�����������������
		* @param[out] descriptors  ����õı��������������ӵ�cv::Mat
		*/
		void operator()(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

		//�������Щ����������������ֱ�ӻ�ȡ��ĳ�Ա������

		/**
		* @brief ��ȡͼ��������Ĳ���
		* @return int ͼ��������Ĳ���
		*/
		int inline GetLevels() {
			return mnlevels;
		}

		/**
		* @brief ��ȡ��ǰ��ȡ�����ڵ�ͼ����������ӣ��������s�����ӱ�ʾ�����ٽ���֮���
		* @return float ��ǰ��ȡ�����ڵ�ͼ����������ӣ����ڲ�֮��
		*/
		float inline GetScaleFactor() {
			return mscaleFactor;
		}

		/**
		* @brief ��ȡͼ���������ÿ��ͼ������ڵײ�ͼ�����������
		* @return std::vector<float> ͼ���������ÿ��ͼ������ڵײ�ͼ�����������
		*/
		std::vector<float> inline GetScaleFactors() {
			return mvScaleFactor;
		}

		/**
		* @brief ��ȡ������Ǹ���������s�ĵ���
		* @return std::vector<float> ����
		*/
		std::vector<float> inline GetInverseScaleFactors() {
			return mvInvScaleFactor;
		}

		/**
		* @brief ��ȡsigma^2������ÿ��ͼ������ڳ�ʼͼ���������ӵ�ƽ�����ο�cpp�ļ����๹�캯���Ĳ���
		* @return std::vector<float> sigma^2
		*/
		std::vector<float> inline GetScaleSigmaSquares() {
			return mvLevelSigma2;
		}

		/**
		* @brief ��ȡ����sigmaƽ���ĵ���
		* @return std::vector<float>
		*/
		std::vector<float> inline GetInverseScaleSigmaSquares() {
			return mvInvLevelSigma2;
		}

		//����������洢ͼ��������ı�����һ��Ԫ�ش洢һ��ͼ��
		std::vector<cv::Mat> mvImagePyramid;

	private:
		/**
		* @brief ��Ը�����һ��ͼ�񣬼�����ͼ�������
		* @param[in] image ������ͼ��
		*/
		void ComputePyramid(cv::Mat& image);

		/**
		* @brief ����ĳ�������ͼ�����������������
		*
		* @param[in] image                 ĳ�������ͼ��
		* @param[in] keypoints             ������vector����
		* @param[out] descriptors          ������
		* @param[in] pattern               ����������ʹ�õĹ̶�����㼯
		*/
		void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const std::vector<cv::Point>& pattern);

		/**
		* @brief ����ORB������������ӡ�ע�������ȫ�ֵľ�̬������ֻ�����ڱ��ļ��ڱ�����
		* @param[in] kpt       ���������
		* @param[in] img       ��ȡ�������ͼ��
		* @param[in] pattern   Ԥ����õĲ���ģ��
		* @param[out] desc     ��������������������õ������ӣ�ά��Ϊ32*8 = 256 bit
		*/
		void computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern, uchar* desc);

		/**
		* @brief �԰˲�������������ķ�ʽ������ͼ��������е�������
		* @detials ��������vector����˼�ǣ���һ��洢����ĳ��ͼƬ�е����������㣬���ڶ������Ǵ洢ͼ�������������ͼ���vectors of keypoints
		* @param[out] allKeypoints ��ȡ�õ�������������
		*/
		void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

		/**
		* @brief ����ĳһͼ�㣬�����������㣬ͨ���˲����ķ�ʽ
		* @param[in] vToDistributeKeys         �ȴ������������
		* @param[in] minX                      �ַ���ͼ��Χ
		* @param[in] maxX                      �ַ���ͼ��Χ
		* @param[in] minY                      �ַ���ͼ��Χ
		* @param[in] maxY                      �ַ���ͼ��Χ
		* @param[in] nFeatures                 �趨�ġ���ͼ������Ҫ��ȡ����������Ŀ
		* @param[in] level                     Ҫ��ȡ��ͼ�����ڵĽ�������
		* @return std::vector<cv::KeyPoint>
		*/
		std::vector<cv::KeyPoint> DistributeOctTree(
			const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX, const int &maxX, const int &minY, const int &maxY,
			const int &nFeatures, const int &level);

		/**
		* @brief ����������ķ���
		* @param[in] image                 ���������ڵ�ǰ��������ͼ��
		* @param[in & out] keypoints       ����������
		* @param[in] umax                  ÿ������������ͼ�������ÿ�еı߽� u_max ��ɵ�vector
		*/
		void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);

		/**
		* @brief ����������ڼ���������ķ��������Ƿ��ؽǶ���Ϊ����
		* ���������㷽����Ϊ��ʹ����ȡ�������������ת�����ԡ�
		* �����ǻҶ����ķ����Լ������ĺͻҶ����ĵ�������Ϊ�������㷽��
		* @param[in] image     Ҫ���в�����ĳ�������ͼ��
		* @param[in] pt        ��ǰ�����������
		* @param[in] u_max     ͼ����ÿһ�е�����߽� u_max
		* @return float        ����������ĽǶȣ���ΧΪ[0,360)�Ƕȣ�����Ϊ0.3��
		*/
		float IC_Angle(const cv::Mat& image, cv::Point2f pt, const std::vector<int> & u_max);

		std::vector<cv::Point> mvpattern;           //<���ڼ��������ӵ���������㼯��

		int mnfeatures;			                    //<����ͼ��������У�Ҫ��ȡ����������Ŀ
		double mscaleFactor;		                //<ͼ������������֮�����������
		int mnlevels;			                    //<ͼ��������Ĳ���
		int miniThFAST;			                    //<��ʼ��FAST��Ӧֵ��ֵ
		int minThFAST;			                    //<��С��FAST��Ӧֵ��ֵ

		std::vector<int> mv_nFeaturesPerLevel;		//<���䵽ÿ��ͼ���У�Ҫ��ȡ����������Ŀ

		std::vector<int> mv_umax;	                //<���������㷽���ʱ���и�Բ�ε�ͼ���������vector�д洢��ÿ��u��ı߽磨�ķ�֮һ����������ͨ���Գƻ�ã�

		std::vector<float> mvScaleFactor;		    //<ÿ��ͼ�����������
		std::vector<float> mvInvScaleFactor;        //<�Լ�ÿ���������ӵĵ���
		std::vector<float> mvLevelSigma2;		    //<�洢ÿ���sigma^2,������ÿ��ͼ������ڵײ�ͼ�����ű�����ƽ��
		std::vector<float> mvInvLevelSigma2;	    //<sigmaƽ���ĵ���

		const int PATCH_SIZE = 31;			//<ʹ�ûҶ����ķ�����������ķ�����Ϣʱ��ͼ���Ĵ�С,����˵��ֱ��
		const int HALF_PATCH_SIZE = 15;		//<���������С��һ�룬����˵�ǰ뾶
		const int EDGE_THRESHOLD = 15;
};

}