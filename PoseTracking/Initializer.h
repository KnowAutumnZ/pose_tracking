#pragma once

#include <cstdlib>
#include <thread>
#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace PoseTracking
{
	typedef std::pair<int, int> Match;

	class Initializer
	{
	public:
		/**
		 * @brief ���ݲο�֡�����ʼ����
		 *
		 * @param[in] ReferenceFrame        �ο�֡
		 * @param[in] sigma                 �������
		 * @param[in] iterations            RANSAC��������
		 */
		Initializer(const Frame* ReferenceFrame, float sigma = 1.0, int iterations = 10);

		/**
		 * @brief �����������͵�Ӧ�Ծ���ѡȡ��ѵ����ָ����ʼ��֮֡��������̬�����������ǻ��õ���ʼ��ͼ��
		 * Step 1 ���¼�¼������Ե�ƥ���ϵ
		 * Step 2 ������ƥ��������������ѡ��8��ƥ��������Ϊһ�飬���ڹ���H�����F����
		 * Step 3 ����fundamental ���� ��homography ����Ϊ�˼��ٷֱ����̼߳���
		 * Step 4 ����÷ֱ������ж�ѡȡ�ĸ�ģ������λ��R,t
		 *
		 * @param[in] CurrentFrame          ��ǰ֡��Ҳ����SLAM�����ϵĵڶ�֡
		 * @param[in] vMatches12            ��ǰ֡��2���Ͳο�֡��1��ͼ�����������ƥ���ϵ
		 *                                  vMatches12[i]���ͣ�i��ʾ֡1�йؼ��������ֵ��vMatches12[i]��ֵΪ֡2�Ĺؼ�������ֵ
		 *                                  û��ƥ���ϵ�Ļ���vMatches12[i]ֵΪ -1
		 * @param[in & out] R21                   ����Ӳο�֡����ǰ֡����ת
		 * @param[in & out] t21                   ����Ӳο�֡����ǰ֡��ƽ��
		 * @param[in & out] vP3D                  ���ǻ�����֮�����ά��ͼ��
		 * @param[in & out] vbTriangulated        ������ǻ����Ƿ���Ч����ЧΪtrue
		 * @return true                     ��֡���Գɹ���ʼ��������true
		 * @return false                    ��֡�������ʼ������������false
		 */
		bool Initialize(const Frame* CurrentFrame,
			const std::vector<int> &vMatches12,
			cv::Mat &R21, cv::Mat &t21,
			std::vector<cv::Point3f> &vP3D,
			std::vector<bool> &vbTriangulated);

	private:
		/**
		 * @brief ���㵥Ӧ���󣬼��賡��Ϊƽ�������ͨ��ǰ��֡��ȡHomography���󣬲��õ���ģ�͵�����
		 * ԭ��ο�Multiple view geometry in computer vision  P109 �㷨4.4
		 * Step 1 ����ǰ֡�Ͳο�֡�е�������������й�һ��
		 * Step 2 ѡ��8����һ��֮��ĵ�Խ��е���
		 * Step 3 �˵㷨���㵥Ӧ�������
		 * Step 4 ������ͶӰ���Ϊ����RANSAC�Ľ������
		 * Step 5 ���¾����������ֵĵ�Ӧ���������,���ұ�������Ӧ��������Ե��ڵ���
		 *
		 * @param[in & out] vbMatchesInliers          ����Ƿ������
		 * @param[in & out] score                     ���㵥Ӧ����ĵ÷�
		 * @param[in & out] H21                       ��Ӧ������
		 */
		void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);

		/**
		 * @brief ����������󣬼��賡��Ϊ��ƽ�������ͨ��ǰ��֡��ȡFundamental���󣬵õ���ģ�͵�����
		 * Step 1 ����ǰ֡�Ͳο�֡�е�������������й�һ��
		 * Step 2 ѡ��8����һ��֮��ĵ�Խ��е���
		 * Step 3 �˵㷨��������������
		 * Step 4 ������ͶӰ���Ϊ����RANSAC�Ľ������
		 * Step 5 ���¾����������ֵĻ������������,���ұ�������Ӧ��������Ե��ڵ���
		 *
		 * @param[in & out] vbMatchesInliers          ����Ƿ������
		 * @param[in & out] score                     �����������÷�
		 * @param[in & out] F21                       ����������
		 */
		void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &F21);

		/**
		 * @brief ��DLT������ⵥӦ����H
		 * ����������4�Ե���ܹ����������������Ϊ��ͳһ����ʹ����8�Ե�����С���˽�
		 *
		 * @param[in] vP1               �ο�֡�й�һ�����������
		 * @param[in] vP2               ��ǰ֡�й�һ�����������
		 * @return cv::Mat              ����ĵ�Ӧ����
		 */
		cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);

		/**
		 * @brief ����������ƥ����fundamental matrix��normalized 8�㷨��
		 * ע��F��������Ϊ2��Լ����������Ҫ����SVD�ֽ�
		 * �ο��� Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (���İ� p191)
		 * @param[in] vP1           �ο�֡�й�һ�����������
		 * @param[in] vP2           ��ǰ֡�й�һ�����������
		 * @return cv::Mat          ������õ��Ļ�������F
		 */
		cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);

		/**
		 * @brief �Ը�����homography matrix���,��Ҫʹ�õ����������֪ʶ
		 *
		 * @param[in] H21                       �Ӳο�֡����ǰ֡�ĵ�Ӧ����
		 * @param[in] H12                       �ӵ�ǰ֡���ο�֡�ĵ�Ӧ����
		 * @param[in] vbMatchesInliers          ƥ��õ�������Ե�Inliers���
		 * @param[in] sigma                     ���Ĭ��Ϊ1
		 * @return float                        ���ص÷�
		 */
		float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &vbMatchesInliers, float sigma);

		/**
		 * @brief �Ը�����Fundamental matrix���
		 *
		 * @param[in] F21                       ��ǰ֡�Ͳο�֮֡��Ļ�������
		 * @param[in] vbMatchesInliers          ƥ��������������inliers�ı��
		 * @param[in] sigma                     ���Ĭ��Ϊ1
		 * @return float                        ���ص÷�
		 */
		float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);

		/**
		 * @brief �ӻ�������F�����λ��R��t����ά��
		 *
		 * @param[in] vbMatchesInliers          ƥ��õ�������Ե�Inliers���
		 * @param[in] F21                       �Ӳο�֡����ǰ֡�Ļ�������
		 * @param[in] K                         ������ڲ�������
		 * @param[in & out] R21                 ����õ�����Ӳο�֡����ǰ֡����ת
		 * @param[in & out] t21                 ����õ�����Ӳο�֡����ǰ֡��ƽ��
		 * @param[in & out] vP3D                ���ǻ�����֮���������Ŀռ�����
		 * @param[in & out] vbTriangulated      ���������ǻ��ɹ��ı�־
		 * @param[in] minParallax               ��Ϊ���ǻ���Ч����С�Ӳ��
		 * @param[in] minTriangulated           ��С���ǻ�������
		 * @return true                         �ɹ���ʼ��
		 * @return false                        ��ʼ��ʧ��
		 */
		bool ReconstructF(std::vector<bool> &vbMatchesInliers,
			cv::Mat &F21, cv::Mat &K,
			cv::Mat &R21,
			cv::Mat &t21,
			std::vector<cv::Point3f> &vP3D,
			std::vector<bool> &vbTriangulated,
			float minParallax,
			int minTriangulated);

		/**
		 * @brief ��H����ָ�R, t����ά��
		 * H����ֽⳣ�������ַ�����Faugeras SVD-based decomposition �� Zhang SVD-based decomposition
		 * ����ʹ����Faugeras SVD-based decomposition�㷨���ο�����
		 * Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988
		 *
		 * @param[in] vbMatchesInliers          ƥ���Ե��ڵ���
		 * @param[in] H21                       �Ӳο�֡����ǰ֡�ĵ�Ӧ����
		 * @param[in] K                         ������ڲ�������
		 * @param[in & out] R21                 ��������������ת
		 * @param[in & out] t21                 ������������ƽ��
		 * @param[in & out] vP3D                ��������ϵ�£����ǻ������������֮��õ���������Ŀռ�����
		 * @param[in & out] vbTriangulated      �������Ƿ�ɹ����ǻ��ı��
		 * @param[in] minParallax               ������������ǻ������У���Ϊ�������Чʱ��Ҫ�������С�Ӳ�ǣ�����Ӳ�ǹ�С�������ǳ���Ĺ۲���,��λ�ǽǶ�
		 * @param[in] minTriangulated           Ϊ�˽����˶��ָ�������Ҫ�����ٵ����ǻ������ɹ��ĵ����
		 * @return true                         ��Ӧ����ɹ������λ�˺���ά��
		 * @return false                        ��ʼ��ʧ��
		 */
		bool ReconstructH(std::vector<bool> &vbMatchesInliers,
			cv::Mat &H21,
			cv::Mat &K,
			cv::Mat &R21,
			cv::Mat &t21,
			std::vector<cv::Point3f> &vP3D,
			std::vector<bool> &vbTriangulated,
			float minParallax,
			int minTriangulated);

		/** ����ͶӰ����P1,P2��ͼ���ϵ�ƥ���������kp1,kp2���Ӷ�������ά������
		 * @brief
		 *
		 * @param[in] kp1               ������, in reference frame
		 * @param[in] kp2               ������, in current frame
		 * @param[in] P1                ͶӰ����P1
		 * @param[in] P2                ͶӰ����P2
		 * @param[in & out] x3D         �������ά��
		 */
		void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

		/**
		 * @brief ��һ�������㵽ͬһ�߶ȣ���Ϊ����normalize DLT������
		 *  [x' y' 1]' = T * [x y 1]'
		 *  ��һ����x', y'�ľ�ֵΪ0��sum(abs(x_i'-0))=1��sum(abs((y_i'-0))=1
		 *
		 *  ΪʲôҪ��һ����
		 *  �����Ʊ任֮��(���ڲ�ͬ������ϵ��),���ǵĵ�Ӧ�Ծ����ǲ���ͬ��
		 *  ���ͼ���������,ʹ�õ�����귢���˱仯,��ô���ĵ�Ӧ�Ծ���Ҳ�ᷢ���仯
		 *  ���ǲ�ȡ�ķ����ǽ��������ŵ�ͬһ����ϵ��,�������ų߶�Ҳ����ͳһ
		 *  ��ͬһ��ͼ������������ͬ�ı任,��ͬͼ����в�ͬ�任
		 *  ���ų߶���Ϊ������������ͼ���Ӱ����һ����������
		 *
		 *  Step 1 ����������X,Y����ľ�ֵ
		 *  Step 2 ����������X,Y�������ֵ��ƽ��ƫ��̶�
		 *  Step 3 ��x�����y����ֱ���г߶ȹ�һ����ʹ��x�����y�����һ�׾��Ծطֱ�Ϊ1
		 *  Step 4 �����һ��������ʵ����ǰ�����Ĳ����þ���任����ʾ����
		 *
		 * @param[in] vKeys                               ����һ����������
		 * @param[in & out] vNormalizedPoints             �������һ���������
		 * @param[in & out] T                             ��һ��������ı任����
		 */
		void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

		/**
		 * @brief ����cheirality check���Ӷ���һ���ҳ�F�ֽ������ʵĽ�
		 * @detials ReconstructF���øú�������cheirality check���Ӷ���һ���ҳ�F�ֽ������ʵĽ�
		 * @param[in]   R				    �����������ת����R
		 * @param[in]   t				    �����������ת����t
		 * @param[in]   vKeys1			    �ο�֡������
		 * @param[in]   vKeys2			    ��ǰ֡������
		 * @param[in]   vMatches12		    ��֡�������ƥ���ϵ
		 * @param[in]   vbMatchesInliers    ������Ե�Inliers���
		 * @param[in]   K				    ������ڲ�������
		 * @param[out]  vP3D				���ǻ�����֮���������Ŀռ�����
		 * @param[in]   th2				    ��ͶӰ������ֵ
		 * @param[out]  vbGood			    �����㣨�ԣ�����good��ı��
		 * @param[out]  parallax			��������ıȽϴ���Ӳ�ǣ�ע�ⲻ��������Ҫ�������г����ע�ͣ�
		 * @return	int ���ر������good�����Ŀ
		 */
		int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
			const std::vector<Match> &vMatches12, std::vector<bool> &vbInliers,
			const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood);

		/**
		 * @brief �ֽ�Essential����
		 * @detials F����ͨ������ڲο��Եõ�Essential���󣬷ֽ�E���󽫵õ�4��� \n
		 * ��4���ֱ�Ϊ[R1,t],[R1,-t],[R2,t],[R2,-t]
		 * @param[in]   E  Essential Matrix
		 * @param[out]  R1 Rotation Matrix 1
		 * @param[out]  R2 Rotation Matrix 2
		 * @param[out]  t  Translation������һ�����ȡ�����෴������
		 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
		 */
		void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

		inline void SeedRandOnce(int seed)
		{
			srand(seed);
		}

		inline int RandomInt(int min, int max) {
			int d = max - min + 1;
			return int(((double)rand() / ((double)RAND_MAX + 1.0)) * d) + min;
		}
	private:
		/** ������� */
		float mSigma, mSigma2;

		/** ��Fundamental��Homography����ʱRANSAC��������  */
		int mMaxIterations;

		/** ��ά��������������Ĵ�СΪ�����������ڲ�������СΪÿ�ε�����H��F������Ҫ�ĵ�,ʵ�����ǰ˶� */
		std::vector<std::vector<size_t> > mvSets;

		// (Frame 1)
		/** �洢Reference Frame�е������� */
		std::vector<cv::KeyPoint> mvKeys1;

		// (Frame 2)
		/** �洢Current Frame�е������� */
		std::vector<cv::KeyPoint> mvKeys2;

		/** Match�����ݽṹ��pair,mvMatches12ֻ��¼Reference��Currentƥ���ϵ��������  */
		std::vector<Match> mvMatches12;
		/** ��¼Reference Frame��ÿ����������Current Frame�Ƿ���ƥ��������� */
		std::vector<bool> mvbMatched1;
	};
}
