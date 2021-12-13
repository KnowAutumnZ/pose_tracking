#include "Initializer.h"

namespace PoseTracking
{
	/**
	 * @brief ���ݲο�֡�����ʼ����
	 *
	 * @param[in] ReferenceFrame        �ο�֡
	 * @param[in] sigma                 �������
	 * @param[in] iterations            RANSAC��������
	 */
	Initializer::Initializer(const cv::Mat& K, const Frame &ReferenceFrame, float sigma, int iterations)
	{
		//�Ӳο�֡�л�ȡ������ڲ�������
		mK = K.clone();

		// �Ӳο�֡�л�ȡȥ������������
		mvKeys1 = ReferenceFrame.mvKeys;

		//��ȡ�������
		mSigma = sigma;
		mSigma2 = sigma * sigma;

		//����������
		mMaxIterations = iterations;
	}

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
	bool Initializer::Initialize(const Frame &CurrentFrame, const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
		std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated)
	{
		//��ȡ��ǰ֡��ȥ����֮���������
		mvKeys2 = CurrentFrame.mvKeys;

		// mvMatches12��¼ƥ���ϵ�������ԣ���¼����֡2��֡1��ƥ������
		mvMatches12.clear();
		// Ԥ����ռ䣬��С�͹ؼ�����Ŀһ��mvKeys2.size()
		mvMatches12.reserve(mvKeys2.size());

		// Step 1 ���¼�¼������Ե�ƥ���ϵ�洢��mvMatches12���Ƿ���ƥ��洢��mvbMatched1
		// ��vMatches12�������ࣩ ת��Ϊ mvMatches12��ֻ��¼��ƥ���ϵ��
		for (size_t i=0; i<vMatches12.size(); i++)
		{
			//û��ƥ���ϵ�Ļ���vMatches12[i]ֵΪ -1
			if (vMatches12[i] >= 0)
			{
				//mvMatches12 ��ֻ��¼��ƥ���ϵ��������Ե�����ֵ
				//i��ʾ֡1�йؼ��������ֵ��vMatches12[i]��ֵΪ֡2�Ĺؼ�������ֵ
				mvMatches12.push_back(std::make_pair(i, vMatches12[i]));
			}
		}

		// ��ƥ���������Ķ���
		const int N = mvMatches12.size();
		// Indices for minimum set selection
		// �½�һ������vAllIndices�洢��������������Ԥ����ռ�
		std::vector<size_t> vAllIndices;
		//��ʼ������������Ե�����������ֵ0��N-1
		for (int i = 0; i < N; i++)
			vAllIndices.push_back(i);

		//��RANSAC��ĳ�ε����У������Ա���ȡ����Ϊ����������������Ե���������������������ֽ������õ�����
		std::vector<size_t> vAvailableIndices;

		// Step 2 ������ƥ��������������ѡ��8��ƥ��������Ϊһ�飬���ڹ���H�����F����
		// ��ѡ�� mMaxIterations (Ĭ��200) ��
		//mvSets���ڱ���ÿ�ε���ʱ��ʹ�õ�����
		mvSets = std::vector< std::vector<size_t> >(mMaxIterations, std::vector<size_t>(8, 0));

		//���ڽ���������������������������������
		SeedRandOnce(0);

		//��ʼÿһ�εĵ��� 
		for (int it = 0; it < mMaxIterations; it++)
		{
			//������ʼ��ʱ�����еĵ㶼�ǿ��õ�
			vAvailableIndices = vAllIndices;

			//ѡ����С��������������ʹ�ð˵㷨�����������ѭ���˰˴�
			for (size_t j = 0; j < 8; j++)
			{
				// �������һ�Ե��id,��Χ��0��N-1
				int randi = RandomInt(0, vAvailableIndices.size() - 1);
				// idx��ʾ��һ��������Ӧ��������Ա�ѡ��
				int idx = vAvailableIndices[randi];

				//�����ε������ѡ�еĵ�j��������Ե�������ӵ�mvSets��
				mvSets[it][j] = idx;

				// ������Ե��ڱ��ε������Ѿ���ʹ����,��������Ϊ�˱����ٴγ鵽�����,����"��Ŀ�ѡ�б�"��,
				// �������ԭ�����ڵ�λ����vector���һ��Ԫ�ص���Ϣ����,����ɾ��β����Ԫ��
				// �������൱�ڽ���������Ϣ��"��Ŀ����б�"��ֱ��ɾ����
				vAvailableIndices[randi] = vAvailableIndices.back();
				vAvailableIndices.pop_back();

				//std::vector<size_t>::iterator it = std::find(vAvailableIndices.begin(), vAvailableIndices.end(), idx);
				//vAvailableIndices.erase(it);
			}//������ȡ��8���������
		}//����mMaxIterations�Σ�ѡȡ���Ե���ʱ��Ҫ�õ�����С���ݼ�

		// Step 3 ����fundamental ���� ��homography ����Ϊ�˼��ٷֱ����̼߳��� 
		//�������������ڱ����H��F�ļ�������Щ������Ա���Ϊ��Inlier
		std::vector<bool> vbMatchesInliersH, vbMatchesInliersF;
		//��������ĵ�Ӧ����ͻ��������RANSAC���֣�������ʵ�ǲ�����ͶӰ����������
		float SH, SF; //score for H and F
		//�������Ǿ���RANSAC�㷨���������ĵ�Ӧ����ͻ�������
		cv::Mat H, F;







	}




}