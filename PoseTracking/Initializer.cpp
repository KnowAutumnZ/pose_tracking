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
		mvSets = std::vector< std::vector<size_t> >(mMaxIterations, std::vector<size_t>(20, 0));

		//���ڽ���������������������������������
		SeedRandOnce(0);

		//��ʼÿһ�εĵ��� 
		for (int it = 0; it < mMaxIterations; it++)
		{
			//������ʼ��ʱ�����еĵ㶼�ǿ��õ�
			vAvailableIndices = vAllIndices;

			//ѡ����С��������������ʹ�ð˵㷨�����������ѭ���˰˴�
			for (size_t j = 0; j < 20; j++)
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
	void Initializer::FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
	{
		//ƥ��������������
		const int N = mvMatches12.size();

		// Step 1 ����ǰ֡�Ͳο�֡�е�������������й�һ������Ҫ��ƽ�ƺͳ߶ȱ任
		// ������˵,���ǽ�mvKeys1��mvKey2��һ������ֵΪ0��һ�׾��Ծ�Ϊ1����һ������ֱ�ΪT1��T2
		// ������ν��һ�׾��Ծ���ʵ�������������ȡֵ�����ĵľ���ֵ��ƽ��ֵ
		// ��һ��������ǰ�������һ���Ĳ����þ�������ʾ����������������˹�һ��������Եõ���һ���������

		//��һ����Ĳο�֡1�͵�ǰ֡2�е�����������
		std::vector<cv::Point2f> vPn1, vPn2;

		// ��¼���ԵĹ�һ������
		cv::Mat T1, T2;
		Normalize(mvKeys1, vPn1, T1);
		Normalize(mvKeys2, vPn2, T2);

		//����������ں���Ĵ�����Ҫ�õ�����������ԭʼ�߶ȵĻָ�
		cv::Mat T2inv = T2.inv();

		// ��¼�������
		score = 0.0;
		// ȡ����ʷ�������ʱ,������Ե�inliers���
		vbMatchesInliers = std::vector<bool>(N, false);

		//ĳ�ε����У��ο�֡������������
		std::vector<cv::Point2f> vPn1i(20);
		//ĳ�ε����У���ǰ֡������������
		std::vector<cv::Point2f> vPn2i(20);
		//�Լ���������ĵ�Ӧ���󡢼��������
		cv::Mat H21i, H12i;

		// ÿ��RANSAC��¼Inliers��÷�
		std::vector<bool> vbCurrentInliers(N, false);
		float currentScore;

		//�������ÿ�ε�RANSAC����
		for (int it = 0; it < mMaxIterations; it++)
		{
			// Step 2 ѡ��8����һ��֮��ĵ�Խ��е���
			for (size_t j = 0; j < 20; j++)
			{
				//��mvSets�л�ȡ��ǰ�ε�����ĳ��������Ե�������Ϣ
				int idx = mvSets[it][j];

				// vPn1i��vPn2iΪƥ���������ԵĹ�һ���������
				// ���ȸ������������Ե�������Ϣ�ֱ��ҵ������������ڸ���ͼ�������������е�������Ȼ���ȡ���һ��֮�������������
				vPn1i[j] = vPn1[mvMatches12[idx].first];    //first�洢�ڲο�֡1�е�����������
				vPn2i[j] = vPn2[mvMatches12[idx].second];   //second�洢�ڲο�֡1�е�����������
			}//��ȡ8��������Ĺ�һ��֮�������

			// Step 3 �˵㷨���㵥Ӧ����
			// �������ɵ�8����һ���������, ���ú��� Initializer::ComputeH21() ʹ�ð˵㷨���㵥Ӧ����  
			// ����Ϊʲô����֮ǰҪ����������й�һ���������ָֻ��������ĳ߶ȣ�
			// �����ڡ�������Ӿ��еĶ���ͼ���Ρ��Ȿ����P193ҳ���ҵ���
			// ��������˵,8���㷨�ɹ��Ĺؼ����ڹ����ķ���֮ǰӦ�������������������ʵ��Ĺ�һ��
			cv::Mat Hn = ComputeH21(vPn1i, vPn2i);

			// ��Ӧ����ԭ��X2=H21*X1������X1,X2 Ϊ��һ�����������    
			// �������һ����vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  �õ�:T2 * mvKeys2 =  Hn * T1 * mvKeys1   
			// ��һ���õ�:mvKeys2  = T2.inv * Hn * T1 * mvKeys1
			H21i = T2inv * Hn*T1;
			//Ȼ�������
			H12i = H21i.inv();

			// Step 4 ������ͶӰ���Ϊ����RANSAC�Ľ������
			currentScore = CheckHomography(H21i, H12i, 			//���룬��Ӧ����ļ�����
				vbCurrentInliers, 								//�����������Ե�Inliers���
				mSigma);										//TODO  ��������Initializer��������ʱ�����ⲿ������

			// Step 5 ���¾����������ֵĵ�Ӧ���������,���ұ�������Ӧ��������Ե��ڵ���
			if (currentScore > score)
			{
				//�����ǰ�Ľ���÷ָ��ߣ���ô�͸������ż�����
				H21 = H21i.clone();
				//����ƥ��õ�������Ե�Inliers���
				vbMatchesInliers = vbCurrentInliers;
				//������ʷ��������
				score = currentScore;
			}
		}
	}

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
	 * @param[in & out] F21                       ��������1��2�Ļ�������
	 */
	void Initializer::FindFundamental(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
	{
		// �����������,����̺�����ļ��㵥Ӧ����Ĺ���ʮ������.

		// ƥ��������������
		// const int N = vbMatchesInliers.size();  // !Դ���������ʹ���������
		const int N = mvMatches12.size();
		// Normalize coordinates
		// Step 1 ����ǰ֡�Ͳο�֡�е�������������й�һ������Ҫ��ƽ�ƺͳ߶ȱ任
		// ������˵,���ǽ�mvKeys1��mvKey2��һ������ֵΪ0��һ�׾��Ծ�Ϊ1����һ������ֱ�ΪT1��T2
		// ������ν��һ�׾��Ծ���ʵ�������������ȡֵ�����ĵľ���ֵ��ƽ��ֵ
		// ��һ��������ǰ�������һ���Ĳ����þ�������ʾ����������������˹�һ��������Եõ���һ���������

		std::vector<cv::Point2f> vPn1, vPn2;
		cv::Mat T1, T2;
		Normalize(mvKeys1, vPn1, T1);
		Normalize(mvKeys2, vPn2, T2);

		// ! ע������ȡ���ǹ�һ������T2��ת��,��Ϊ��������Ķ���͵�Ӧ����ͬ������ȥ��һ���ļ���Ҳ����ͬ
		cv::Mat T2t = T2.t();







	}

	/**
	 * @brief ��DLT������ⵥӦ����H
	 * ����������4�Ե���ܹ����������������Ϊ��ͳһ����ʹ����8�Ե�����С���˽�
	 *
	 * @param[in] vP1               �ο�֡�й�һ�����������
	 * @param[in] vP2               ��ǰ֡�й�һ�����������
	 * @return cv::Mat              ����ĵ�Ӧ����H
	 */
	cv::Mat Initializer::ComputeH21(
		const std::vector<cv::Point2f> &vP1, //��һ����ĵ�, in reference frame
		const std::vector<cv::Point2f> &vP2) //��һ����ĵ�, in current frame
	{
		// ����ԭ���������Ƶ����̣�
		// |x'|     | h1 h2 h3 ||x|
		// |y'| = a | h4 h5 h6 ||y|  ��д: x' = a H x, aΪһ���߶�����
		// |1 |     | h7 h8 h9 ||1|
		// ʹ��DLT(direct linear tranform)����ģ��
		// x' = a H x 
		// ---> (x') ��� (H x)  = 0  (��Ϊ������ͬ) (ȡǰ���оͿ����Ƶ����������)
		// ---> Ah = 0 
		// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
		//     |-x -y -1  0  0  0 xx' yx' x'|
		// ͨ��SVD���Ah = 0��A^T*A��С����ֵ��Ӧ������������Ϊ��
		// ��ʵҲ����������ֵ��������һ��

		//��ȡ�����������������Ŀ
		const int N = vP1.size();

		// �������ڼ���ľ��� A 
		cv::Mat A(2 * N,				//�У�ע��ÿһ��������ݶ�Ӧ����
			9,							//��
			CV_32F);      				//float��������

		// �������A����ÿ����������ӵ�����A�е�Ԫ��
		for (int i = 0; i < N; i++)
		{
			//��ȡ������Ե���������
			const float u1 = vP1[i].x;
			const float v1 = vP1[i].y;
			const float u2 = vP2[i].x;
			const float v2 = vP2[i].y;

			//���������ĵ�һ��
			A.at<float>(2 * i, 0) = 0.0;
			A.at<float>(2 * i, 1) = 0.0;
			A.at<float>(2 * i, 2) = 0.0;
			A.at<float>(2 * i, 3) = -u1;
			A.at<float>(2 * i, 4) = -v1;
			A.at<float>(2 * i, 5) = -1;
			A.at<float>(2 * i, 6) = v2 * u1;
			A.at<float>(2 * i, 7) = v2 * v1;
			A.at<float>(2 * i, 8) = v2;

			//���������ĵڶ���
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

		// �������������u����ߵ���������U�� wΪ�������vt�е�t��ʾ������������V��ת��
		cv::Mat u, w, vt;

		//ʹ��opencv�ṩ�Ľ�������ֵ�ֽ�ĺ���
		cv::SVDecomp(A,							//���룬����������ֵ�ֽ�ľ���
			w,									//���������ֵ����
			u,									//���������U
			vt,									//���������V^T
			cv::SVD::MODIFY_A | 				//���룬MODIFY_A��ָ������㺯�������޸Ĵ��ֽ�ľ��󣬹ٷ��ĵ���˵�������Լӿ�����ٶȡ���ʡ�ڴ�
			cv::SVD::FULL_UV);					//FULL_UV=��U��VT����ɵ�λ��������

		// ������С����ֵ����Ӧ������������
		// ע��ǰ��˵����������ֵ��������һ�У�������������Ϊ��vt��ת�ú��ˣ��������У�����A��9�����ݣ������һ�е��±�Ϊ8
		return vt.row(8).reshape(0, 			//ת�����ͨ��������������Ϊ0��ʾ����ǰ����ͬ
			3); 								//ת���������,��ӦV�����һ��
	}

	/**
	 * @brief �Ը�����homography matrix���,��Ҫʹ�õ����������֪ʶ
	 *
	 * @param[in] H21                       �Ӳο�֡����ǰ֡�ĵ�Ӧ����
	 * @param[in] H12                       �ӵ�ǰ֡���ο�֡�ĵ�Ӧ����
	 * @param[in] vbMatchesInliers          ƥ��õ�������Ե�Inliers���
	 * @param[in] sigma                     ���Ĭ��Ϊ1
	 * @return float                        ���ص÷�
	 */
	float Initializer::CheckHomography(
		const cv::Mat &H21,						 //�Ӳο�֡����ǰ֡�ĵ�Ӧ����
		const cv::Mat &H12,                      //�ӵ�ǰ֡���ο�֡�ĵ�Ӧ����
		std::vector<bool> &vbMatchesInliers,     //ƥ��õ�������Ե�Inliers���
		float sigma)                             //�������
	{
		// ˵��������ֵnά�۲�����������N(0��sigma���ĸ�˹�ֲ�ʱ
		// ������Ȩ��С���˽��Ϊ  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
		// ���У�e(i) = [e_x,e_y,...]^T, Qά�۲�����Э������󣬼�sigma * sigma��ɵ�Э�������
		// ����Ȩ��С���ν��ԽС��˵���۲����ݾ���Խ��
		// ��ô��score = SUM((th - e(i)^T * Q^(-1) * e(i)))�ķ�����Խ��
		// �㷨Ŀ�꣺ ��鵥Ӧ�任����
		// ��鷽ʽ��ͨ��H���󣬽��вο�֡�͵�ǰ֮֡���˫��ͶӰ�����������Ȩ��С����ͶӰ���

		// �㷨����
		// input: ��Ӧ�Ծ��� H21, H12, ƥ��㼯 mvKeys1
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

		// �ص�ƥ�����
		const int N = mvMatches12.size();

		// Step 1 ��ȡ�Ӳο�֡����ǰ֡�ĵ�Ӧ����ĸ���Ԫ��
		const float h11 = H21.at<float>(0, 0);
		const float h12 = H21.at<float>(0, 1);
		const float h13 = H21.at<float>(0, 2);
		const float h21 = H21.at<float>(1, 0);
		const float h22 = H21.at<float>(1, 1);
		const float h23 = H21.at<float>(1, 2);
		const float h31 = H21.at<float>(2, 0);
		const float h32 = H21.at<float>(2, 1);
		const float h33 = H21.at<float>(2, 2);

		// ��ȡ�ӵ�ǰ֡���ο�֡�ĵ�Ӧ����ĸ���Ԫ��
		const float h11inv = H12.at<float>(0, 0);
		const float h12inv = H12.at<float>(0, 1);
		const float h13inv = H12.at<float>(0, 2);
		const float h21inv = H12.at<float>(1, 0);
		const float h22inv = H12.at<float>(1, 1);
		const float h23inv = H12.at<float>(1, 2);
		const float h31inv = H12.at<float>(2, 0);
		const float h32inv = H12.at<float>(2, 1);
		const float h33inv = H12.at<float>(2, 2);

		// ��������Ե�Inliers���Ԥ����ռ�
		vbMatchesInliers.resize(N);

		// ��ʼ��scoreֵ
		float score = 0;

		// ���ڿ���������������ֵ�����������һ�����ص�ƫ�
		// ���ɶ�Ϊ2�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ
		const float th = 5.991;

		//��Ϣ���󣬷���ƽ���ĵ���
		const float invSigmaSquare = 1.0 / (sigma * sigma);

		// Step 2 ͨ��H���󣬽��вο�֡�͵�ǰ֮֡���˫��ͶӰ�����������Ȩ��ͶӰ���
		// H21 ��ʾ��img1 �� img2�ı任����
		// H12 ��ʾ��img2 �� img1�ı任���� 
		for (int i = 0; i < N; i++)
		{
			// һ��ʼ��Ĭ��ΪInlier
			bool bIn = true;

			// Step 2.1 ��ȡ�ο�֡�͵�ǰ֮֡�������ƥ����
			const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
			const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];
			const float u1 = kp1.pt.x;
			const float v1 = kp1.pt.y;
			const float u2 = kp2.pt.x;
			const float v2 = kp2.pt.y;

			// Step 2.2 ���� img2 �� img1 ����ͶӰ���
			// x1 = H12*x2
			// ��ͼ��2�е�������ͨ����Ӧ�任ͶӰ��ͼ��1��
			// |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
			// |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
			// |1 |   |h31inv h32inv h33inv||1 |   |  1  |
			// ����ͶӰ��һ������
			const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
			const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
			const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

			// ������ͶӰ��� = ||p1(i) - H12 * p2(i)||2
			const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
			const float chiSquare1 = squareDist1 * invSigmaSquare;

			// Step 2.3 ����ֵ�����Ⱥ�㣬�ڵ�Ļ��ۼӵ÷�
			if (chiSquare1 > th)
				bIn = false;
			else
				// ���Խ�󣬵÷�Խ��
				score += th - chiSquare1;

			// �����img1 �� img2 ��ͶӰ�任���
			// x1in2 = H21*x1
			// ��ͼ��2�е�������ͨ����Ӧ�任ͶӰ��ͼ��1��
			// |u2|   |h11 h12 h13||u1|   |u1in2|
			// |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
			// |1 |   |h31 h32 h33||1 |   |  1  |
			// ����ͶӰ��һ������
			const float w1in2inv = 1.0 / (h31*u1 + h32 * v1 + h33);
			const float u1in2 = (h11*u1 + h12 * v1 + h13)*w1in2inv;
			const float v1in2 = (h21*u1 + h22 * v1 + h23)*w1in2inv;

			// ������ͶӰ��� 
			const float squareDist2 = (u2 - u1in2)*(u2 - u1in2) + (v2 - v1in2)*(v2 - v1in2);
			const float chiSquare2 = squareDist2 * invSigmaSquare;

			// ����ֵ�����Ⱥ�㣬�ڵ�Ļ��ۼӵ÷�
			if (chiSquare2 > th)
				bIn = false;
			else
				score += th - chiSquare2;

			// Step 2.4 �����img2 �� img1 �� ��img1 ��img2����ͶӰ��������Ҫ����˵����Inlier point
			if (bIn)
				vbMatchesInliers[i] = true;
			else
				vbMatchesInliers[i] = false;
		}
		return score;
	}

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
	void Initializer::Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) 
	{
		// ��һ��������Щ����x�������y�����ϵ�һ�׾��Ծأ������������������

		// Step 1 ����������X,Y����ľ�ֵ meanX, meanY
		float meanX = 0;
		float meanY = 0;

		//��ȡ�����������
		const int N = vKeys.size();

		//���������洢��һ���������������С���͹�һ��ǰ����һ��
		vNormalizedPoints.resize(N);

		//��ʼ�������е�������
		for (int i = 0; i < N; i++)
		{
			//�ֱ��ۼ��������X��Y����
			meanX += vKeys[i].pt.x;
			meanY += vKeys[i].pt.y;
		}

		//����X��Y����ľ�ֵ
		meanX = meanX / N;
		meanY = meanY / N;

		// Step 2 ����������X,Y�������ֵ��ƽ��ƫ��̶� meanDevX, meanDevY��ע�ⲻ�Ǳ�׼��
		float meanDevX = 0;
		float meanDevY = 0;

		// ��ԭʼ�������ȥ��ֵ���꣬ʹx�����y�����ֵ�ֱ�Ϊ0
		for (int i = 0; i < N; i++)
		{
			vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
			vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

			//�ۼ���Щ������ƫ����������ֵ�ĳ̶�
			meanDevX += fabs(vNormalizedPoints[i].x);
			meanDevY += fabs(vNormalizedPoints[i].y);
		}

		// ���ƽ����ÿ�����ϣ�������ƫ����������ֵ�ĳ̶ȣ����䵹����Ϊһ���߶���������
		meanDevX = meanDevX / N;
		meanDevY = meanDevY / N;
		float sX = 1.0 / meanDevX;
		float sY = 1.0 / meanDevY;

		// Step 3 ��x�����y����ֱ���г߶ȹ�һ����ʹ��x�����y�����һ�׾��Ծطֱ�Ϊ1 
		// ������ν��һ�׾��Ծ���ʵ�������������ȡֵ�����ĵľ���ֵ��ƽ��ֵ��������
		for (int i = 0; i < N; i++)
		{
			//�ԣ����Ǽ򵥵ض��������������н�һ��������
			vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
			vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
		}

		// Step 4 �����һ��������ʵ����ǰ�����Ĳ����þ���任����ʾ����
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