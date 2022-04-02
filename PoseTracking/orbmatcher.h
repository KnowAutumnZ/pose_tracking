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
		 * @param nnratio  ratio of the best and the second score   ���źʹ������ֵı���
		 * @param checkOri check orientation                        �Ƿ��鷽��
		 */
		ORBmatcher(float nnratio = 0.6, bool checkOri = true);

		/**
		 * @brief ����ͼ��ͶӰ���ؼ�֡�н���ƥ����ں�;���ҵ�ͼ����滻��������������н���
		 * @param[in] pKF           �ؼ�֡
		 * @param[in] vpMapPoints   ��ͼ��
		 * @param[in] th            �������ڵ���ֵ
		 * @return int
		 */
		int Fuse(KeyFrame* pKF, const std::vector<MapPoint*> &vpMapPoints, const float th = 3.0);

		/**
		 * @brief ͨ��ͶӰ��ͼ�㵽��ǰ֡����Local MapPoint���и���
		 * ����
		 * Step 1 ������Ч�ľֲ���ͼ��
		 * Step 2 �趨�����������ڵĴ�С��ȡ�����ӽ�, ����ǰ�ӽǺ�ƽ���ӽǼнǽ�Сʱ, rȡһ����С��ֵ
		 * Step 3 ͨ��ͶӰ���Լ��������ں�Ԥ��ĳ߶Ƚ�������, �ҳ������뾶�ڵĺ�ѡƥ�������
		 * Step 4 Ѱ�Һ�ѡƥ����е���Ѻʹμ�ƥ���
		 * Step 5 ɸѡ���ƥ���
		 * @param[in] F                         ��ǰ֡
		 * @param[in] vpMapPoints               �ֲ���ͼ�㣬���Ծֲ��ؼ�֡
		 * @param[in] th                        ������Χ
		 * @return int                          �ɹ�ƥ�����Ŀ
		 */
		int SearchByProjection(Frame* F, const std::vector<MapPoint*> &vpMapPoints, const float th = 3);

		/**
		 * @brief Computes the Hamming distance between two ORB descriptors �����ͼ��ͺ�ѡͶӰ��������Ӿ���
		 * @param[in] a     һ��������
		 * @param[in] b     ����һ��������
		 * @return int      �����ӵĺ�������
		 */
		static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

		/**
		 * @brief ��Ŀ��ʼ�������ڲο�֡�͵�ǰ֡��������ƥ��
		 * ����
		 * Step 1 ������תֱ��ͼ
		 * Step 2 �ڰ뾶������������ǰ֡F2�����еĺ�ѡƥ��������
		 * Step 3 �����������������е�����Ǳ�ڵ�ƥ���ѡ�㣬�ҵ����ŵĺʹ��ŵ�
		 * Step 4 �����Ŵ��Ž�����м�飬������ֵ������/���ű�����ɾ���ظ�ƥ��
		 * Step 5 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
		 * Step 6 ɸ����תֱ��ͼ�С�������������
		 * Step 7 �����ͨ��ɸѡ��ƥ��õ������㱣��
		 * @param[in] F1                        ��ʼ���ο�֡
		 * @param[in] F2                        ��ǰ֡
		 * @param[in & out] vbPrevMatched       �����洢���ǲο�֡���������������꣬�ú�������Ϊƥ��õĵ�ǰ֡������������
		 * @param[in & out] vnMatches12         ����ο�֡F1���������Ƿ�ƥ���ϣ�index������F1��Ӧ������������ֵ�������ƥ��õ�F2����������
		 * @param[in] windowSize                ��������
		 * @return int                          ���سɹ�ƥ�����������Ŀ
		 */
		int SearchForInitialization(Frame* F1, Frame* F2, std::vector<int> &vnMatches12, int windowSize = 10);

		/**
		 * @brief ����һ֡���ٵĵ�ͼ��ͶӰ����ǰ֡����������ƥ��㡣���ڸ���ǰһ֡
		 * ����
		 * Step 1 ������תֱ��ͼ�����ڼ����תһ����
		 * Step 2 ���㵱ǰ֡��ǰһ֡��ƽ������
		 * Step 3 ����ǰһ֡��ÿһ����ͼ�㣬ͨ�����ͶӰģ�ͣ��õ�ͶӰ����ǰ֡����������
		 * Step 4 ���������ǰ��ǰ���������ж������߶ȷ�Χ
		 * Step 5 ������ѡƥ��㣬Ѱ�Ҿ�����С�����ƥ���
		 * Step 6 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
		 * Step 7 ������תһ�¼�⣬�޳���һ�µ�ƥ��
		 * @param[in] CurrentFrame          ��ǰ֡
		 * @param[in] LastFrame             ��һ֡
		 * @param[in] th                    ������Χ��ֵ��Ĭ�ϵ�ĿΪ7��˫Ŀ15
		 * @param[in] bMono                 �Ƿ�Ϊ��Ŀ
		 * @return int                      �ɹ�ƥ�������
		 */
		int SearchByProjection(Frame* CurrentFrame, const Frame* LastFrame, const float th, const bool bMono);

		/**
		 * @brief ���û�������F12���������ؼ�֮֡��δƥ����������в����µ�3d��
		 * @param pKF1          �ؼ�֡1
		 * @param pKF2          �ؼ�֡2
		 * @param F12           ��������
		 * @param vMatchedPairs �洢ƥ��������ԣ������������ڹؼ�֡�е�������ʾ���±��ǹؼ�֡1��������id���洢���ǹؼ�֡2��������id
		 * @param bOnlyStereo   ��˫Ŀ��rgbd����£��Ƿ�Ҫ������������ͼ����ƥ��
		 * @return              �ɹ�ƥ�������
		 */
		int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12, std::vector<std::pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

		int SearchForRefModel(KeyFrame *pKF, Frame* F, std::vector<MapPoint*> &vpMapPointMatches);

	private:
		/**
		 * @brief �ҵ��� ��x,yΪ����,�뾶Ϊr��Բ�����ҽ������㼶��[minLevel, maxLevel]��������
		 *
		 * @param[in] x                     ����������x
		 * @param[in] y                     ����������y
		 * @param[in] r                     �����뾶
		 * @param[in] minLevel              ��С�������㼶
		 * @param[in] maxLevel              ���������㼶
		 * @return vector<size_t>           �����������ĺ�ѡƥ���id
		 */
		std::vector<size_t> GetFeaturesInArea(Frame* F, const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1);

		/**
		 * @brief ɸѡ������ת�ǶȲ�������ֱ��ͼ��������������ǰ����bin������
		 *
		 * @param[in] histo         ƥ�����������ת�����ֱ��ͼ
		 * @param[in] L             ֱ��ͼ�ߴ�
		 * @param[in & out] ind1          binֵ��һ���Ӧ������
		 * @param[in & out] ind2          binֵ�ڶ����Ӧ������
		 * @param[in & out] ind3          binֵ�������Ӧ������
		 */
		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

		/**
		 * @brief ���ݹ۲���ӽ�������ƥ���ʱ���������ڴ�С
		 * @param[in] viewCos   �ӽǵ�����ֵ
		 * @return float        �������ڵĴ�С
		 */
		float RadiusByViewingCos(const float &viewCos);

		float mfNNratio;            //< �������ֺʹ������ֵı���
		bool mbCheckOrientation;    //< �Ƿ���������ķ���
	};
}