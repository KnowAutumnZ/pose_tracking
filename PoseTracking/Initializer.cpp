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
		float SH, SF;
		//�������Ǿ���RANSAC�㷨���������ĵ�Ӧ����ͻ�������
		cv::Mat H, F;

		// �����߳�������H������÷�
		// thread�����Ƚ����⣬�ڴ������õ�ʱ�������Ҫ��ref���������ô��ݣ��������ǳ����
		std::thread threadH(&Initializer::FindHomography,	//���̵߳�������
			this,											//����������Ϊ��ĳ�Ա���������Ե�һ��������Ӧ���ǵ�ǰ�����thisָ��
			std::ref(vbMatchesInliersH), 					//�����������Ե�Inlier���
			std::ref(SH), 									//���������ĵ�Ӧ�����RANSAC����
			std::ref(H));									//���������ĵ�Ӧ������
		// ����fundamental matrix����֣����������H��һ���ģ����ﲻ��׸��
		std::thread threadF(&Initializer::FindFundamental, this, std::ref(vbMatchesInliersF), std::ref(SF), std::ref(F));
		//�ȴ����������߳̽���
		threadH.join();
		threadF.join();

		// Step 4 ����÷ֱ������ж�ѡȡ�ĸ�ģ������λ��R,t
		//ͨ������������ж�˭������ռ�ȸ���һЩ��ע�ⲻ�Ǽ򵥵ıȽϾ������ִ�С�����ǿ����ֵ�ռ��
		float RH = SH / (SH + SF);			//RH=Ratio of Homography

		// ע���������������H����ָ�λ�ˡ������Ӧ���������ռ�ȴﵽ��0.4����,��ӵ�Ӧ����ָ��˶�,����ӻ�������ָ��˶�
		if (RH > 0.4)
			return ReconstructH(vbMatchesInliersH,	//���룬ƥ��ɹ����������Inliers���
				H,					//���룬ǰ��RANSAC�����ĵ�Ӧ����
				mK,					//���룬������ڲ�������
				R21, t21,			//������������������Ӳο�֡1����ǰ֡2����������ת��λ�Ʊ任
				vP3D,				//������Ծ������ǲ���֮��Ŀռ����꣬Ҳ���ǵ�ͼ��
				vbTriangulated,		//��������Ƿ�ɹ����ǻ��ı��
				1.0,				//�����Ӧ���β�ΪminParallax������Ϊĳ������������ǻ������У���Ϊ�������Чʱ
									//��Ҫ�������С�Ӳ�ǣ�����Ӳ�ǹ�С�������ǳ���Ĺ۲���,��λ�ǽǶ�
				20);				//Ϊ�˽����˶��ָ�������Ҫ�����ٵ����ǻ������ɹ��ĵ����
		else
			return ReconstructF(vbMatchesInliersF, F, mK, R21, t21, vP3D, vbTriangulated, 1.0, 20);

		//һ��س���Ӧ��ִ�е�������ִ�е�����˵�������ܷ���
		return false;
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

		//���Ž��
		score = 0.0;
		vbMatchesInliers = std::vector<bool>(N, false);

		// Iteration variables
		// ĳ�ε����У��ο�֡������������
		std::vector<cv::Point2f> vPn1i(20);
		// ĳ�ε����У���ǰ֡������������
		std::vector<cv::Point2f> vPn2i(20);
		// ĳ�ε����У�����Ļ�������
		cv::Mat F21i;

		// ÿ��RANSAC��¼��Inliers��÷�
		std::vector<bool> vbCurrentInliers(N, false);
		float currentScore;

		// �������ÿ�ε�RANSAC����
		for (int it = 0; it < mMaxIterations; it++)
		{
			// Select a minimum set
			// Step 2 ѡ��8����һ��֮��ĵ�Խ��е���
			for (int j = 0; j < 20; j++)
			{
				int idx = mvSets[it][j];

				// vPn1i��vPn2iΪƥ���������ԵĹ�һ���������
				// ���ȸ������������Ե�������Ϣ�ֱ��ҵ������������ڸ���ͼ�������������е�������Ȼ���ȡ���һ��֮�������������
				vPn1i[j] = vPn1[mvMatches12[idx].first];        //first�洢�ڲο�֡1�е�����������
				vPn2i[j] = vPn2[mvMatches12[idx].second];       //second�洢�ڲο�֡1�е�����������
			}

			// Step 3 �˵㷨�����������
			cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

			// ��������Լ����p2^t*F21*p1 = 0������p1,p2 Ϊ��λ�����������    
			// �������һ����vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  
			// ���ݻ�������Լ���õ�:(T2 * mvKeys2)^t* Hn * T1 * mvKeys1 = 0   
			// ��һ���õ�:mvKeys2^t * T2^t * Hn * T1 * mvKeys1 = 0
			F21i = T2t * Fn*T1;

			// Step 4 ������ͶӰ���Ϊ����RANSAC�Ľ������
			currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

			// Step 5 ���¾����������ֵĻ������������,���ұ�������Ӧ��������Ե��ڵ���
			if (currentScore > score)
			{
				//�����ǰ�Ľ���÷ָ��ߣ���ô�͸������ż�����
				F21 = F21i.clone();
				vbMatchesInliers = vbCurrentInliers;
				score = currentScore;
			}
		}

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
	 * @brief ����������ƥ����fundamental matrix��normalized 8�㷨��
	 * ע��F��������Ϊ2��Լ����������Ҫ����SVD�ֽ�
	 *
	 * @param[in] vP1           �ο�֡�й�һ�����������
	 * @param[in] vP2           ��ǰ֡�й�һ�����������
	 * @return cv::Mat          ������õ��Ļ�������F
	 */
	cv::Mat Initializer::ComputeF21(
		const std::vector<cv::Point2f> &vP1, //��һ����ĵ�, in reference frame
		const std::vector<cv::Point2f> &vP2) //��һ����ĵ�, in current frame
	{
		// ԭ����������Ƶ�
		// x'Fx = 0 ����ɵã�Af = 0
		// A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
		// ͨ��SVD���Af = 0��A'A��С����ֵ��Ӧ������������Ϊ��

		//��ȡ�����������������
		const int N = vP1.size();

		//��ʼ��A����
		cv::Mat A(N, 9, CV_32F); // N*9ά

		// �������A����ÿ����������ӵ�����A�е�Ԫ��
		for (int i = 0; i < N; i++)
		{
			const float u1 = vP1[i].x;
			const float v1 = vP1[i].y;
			const float u2 = vP2[i].x;
			const float v2 = vP2[i].y;

			A.at<float>(i, 0) = u2 * u1;
			A.at<float>(i, 1) = u2 * v1;
			A.at<float>(i, 2) = u2;
			A.at<float>(i, 3) = v2 * u1;
			A.at<float>(i, 4) = v2 * v1;
			A.at<float>(i, 5) = v2;
			A.at<float>(i, 6) = u1;
			A.at<float>(i, 7) = v1;
			A.at<float>(i, 8) = 1;
		}

		//�洢����ֵ�ֽ����ı���
		cv::Mat u, w, vt;

		// �������������u����ߵ���������U�� wΪ�������vt�е�t��ʾ������������V��ת��
		cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
		// ת���ɻ����������ʽ
		cv::Mat Fpre = vt.row(8).reshape(0, 3); // v�����һ��

		//�����������Ϊ2,�����ǲ��ұ�֤����õ�������������Ϊ2,������Ҫͨ���ڶ�������ֵ�ֽ�,��ǿ��ʹ����Ϊ2
		// �Գ��������Ļ���������е�2������ֵ�ֽ�
		cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

		// ��2Լ����ǿ�ƽ���3������ֵ����Ϊ0
		w.at<float>(2) = 0;

		// ������Ϻ�������Լ���Ļ���������Ϊ���ռ��������� 
		return  u * cv::Mat::diag(w)*vt;
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
	 * @brief �Ը�����Fundamental matrix���
	 *
	 * @param[in] F21                       ��ǰ֡�Ͳο�֮֡��Ļ�������
	 * @param[in] vbMatchesInliers          ƥ��������������inliers�ı��
	 * @param[in] sigma                     ���Ĭ��Ϊ1
	 * @return float                        ���ص÷�
	 */
	float Initializer::CheckFundamental(
		const cv::Mat &F21,                  //��ǰ֡�Ͳο�֮֡��Ļ�������
		std::vector<bool> &vbMatchesInliers, //ƥ��������������inliers�ı��
		float sigma)                         //����
	{

		// ˵��������ֵnά�۲�����������N(0��sigma���ĸ�˹�ֲ�ʱ
		// ������Ȩ��С���˽��Ϊ  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
		// ���У�e(i) = [e_x,e_y,...]^T, Qά�۲�����Э������󣬼�sigma * sigma��ɵ�Э�������
		// ����Ȩ��С���ν��ԽС��˵���۲����ݾ���Խ��
		// ��ô��score = SUM((th - e(i)^T * Q^(-1) * e(i)))�ķ�����Խ��
		// �㷨Ŀ�꣺����������
		// ��鷽ʽ�����öԼ�����ԭ�� p2^T * F * p1 = 0
		// ���裺��ά�ռ��еĵ� P �� img1 �� img2 ��ͼ���ϵ�ͶӰ�ֱ�Ϊ p1 �� p2������Ϊͬ���㣩
		//   ��p2 һ�������ڼ��� l2 �ϣ��� p2*l2 = 0. ��l2 = F*p1 = (a, b, c)^T
		//      ���ԣ����������� e Ϊ p2 �� ���� l2 �ľ��룬�����ֱ���ϣ��� e = 0
		//      ���ݵ㵽ֱ�ߵľ��빫ʽ��d = (ax + by + c) / sqrt(a * a + b * b)
		//      ���ԣ�e =  (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)

		// �㷨����
		// input: �������� F ������ͼƥ��㼯 mvKeys1
		//    do:
		//        for p1(i), p2(i) in mvKeys:
		//           l2 = F * p1(i)
		//           l1 = p2(i) * F
		//           error_i1 = dist_point_to_line(x2,l2)
		//           error_i2 = dist_point_to_line(x1,l1)
		//           
		//           w1 = 1 / sigma / sigma
		//           w2 = 1 / sigma / sigma
		// 
		//           if error1 < th
		//              score +=   thScore - error_i1 * w1
		//           if error2 < th
		//              score +=   thScore - error_i2 * w2
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

		// ��ȡƥ���������Ե��ܶ���
		const int N = mvMatches12.size();

		// Step 1 ��ȡ���������е�Ԫ������
		const float f11 = F21.at<float>(0, 0);
		const float f12 = F21.at<float>(0, 1);
		const float f13 = F21.at<float>(0, 2);
		const float f21 = F21.at<float>(1, 0);
		const float f22 = F21.at<float>(1, 1);
		const float f23 = F21.at<float>(1, 2);
		const float f31 = F21.at<float>(2, 0);
		const float f32 = F21.at<float>(2, 1);
		const float f33 = F21.at<float>(2, 2);

		// Ԥ����ռ�
		vbMatchesInliers.resize(N);

		// �������ֳ�ʼֵ����Ϊ������Ҫ���������ֵ���ۼƣ�
		float score = 0;

		// ���ڿ���������������ֵ
		// ���ɶ�Ϊ1�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ
		// ?����Ϊ�㵽ֱ�߾�����һ�����ɶ���
		const float th = 3.841;

		// ���ɶ�Ϊ2�Ŀ����ֲ���������ˮƽΪ0.05����Ӧ���ٽ���ֵ
		const float thScore = 5.991;

		// ��Ϣ���󣬻� Э�������������
		const float invSigmaSquare = 1.0 / (sigma*sigma);


		// Step 2 ����img1 �� img2 �ڹ��� F ʱ��scoreֵ
		for (int i = 0; i < N; i++)
		{
			//Ĭ��Ϊ�����������Inliers
			bool bIn = true;

			// Step 2.1 ��ȡ�ο�֡�͵�ǰ֮֡�������ƥ����
			const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
			const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

			// ��ȡ�������������
			const float u1 = kp1.pt.x;
			const float v1 = kp1.pt.y;
			const float u2 = kp2.pt.x;
			const float v2 = kp2.pt.y;

			// Reprojection error in second image
			// Step 2.2 ���� img1 �ϵĵ��� img2 ��ͶӰ�õ��ļ��� l2 = F21 * p1 = (a2,b2,c2)
			const float a2 = f11 * u1 + f12 * v1 + f13;
			const float b2 = f21 * u1 + f22 * v1 + f23;
			const float c2 = f31 * u1 + f32 * v1 + f33;

			// Step 2.3 ������� e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
			const float num2 = a2 * u2 + b2 * v2 + c2;
			const float squareDist1 = num2 * num2 / (a2*a2 + b2 * b2);
			// ��Ȩ�����
			const float chiSquare1 = squareDist1 * invSigmaSquare;

			// Step 2.4 ��������ֵ��˵���������Outlier 
			// ? Ϊʲô�ж���ֵ�õ� th��1���ɶȣ�������÷��õ�thScore��2���ɶȣ�
			// ? ������Ϊ�˺�CheckHomography �÷�ͳһ��
			if (chiSquare1 > th)
				bIn = false;
			else
				// ���Խ�󣬵÷�Խ��
				score += thScore - chiSquare1;

			// ����img2�ϵĵ��� img1 ��ͶӰ�õ��ļ��� l1= p2 * F21 = (a1,b1,c1)
			const float a1 = f11 * u2 + f21 * v2 + f31;
			const float b1 = f12 * u2 + f22 * v2 + f32;
			const float c1 = f13 * u2 + f23 * v2 + f33;

			// ������� e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
			const float num1 = a1 * u1 + b1 * v1 + c1;
			const float squareDist2 = num1 * num1 / (a1*a1 + b1 * b1);

			// ��Ȩ�����
			const float chiSquare2 = squareDist2 * invSigmaSquare;

			// ��������ֵ��˵���������Outlier 
			if (chiSquare2 > th)
				bIn = false;
			else
				score += thScore - chiSquare2;

			// Step 2.5 ������
			if (bIn)
				vbMatchesInliers[i] = true;
			else
				vbMatchesInliers[i] = false;
		}
		//  ��������
		return score;
	}

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
	bool Initializer::ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
		cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
	{
		// Ŀ�� ��ͨ����Ӧ����H�ָ���֡ͼ��֮�����ת����R��ƽ������T
		// �ο� ��Motion and structure from motion in a piecewise plannar environment.
		//        International Journal of Pattern Recognition and Artificial Intelligence, 1988
		// https://www.researchgate.net/publication/243764888_Motion_and_Structure_from_Motion_in_a_Piecewise_Planar_Environment

		// ����:
		//      1. ����H���������ֵd'= d2 ���� d' = -d2 �ֱ���� H ����ֽ�� 8 ���
		//        1.1 ���� d' > 0 ʱ�� 4 ���
		//        1.2 ���� d' < 0 ʱ�� 4 ���
		//      2. �� 8 ��������֤����ѡ��������ǰ�����3D��Ľ�Ϊ���Ž�

		// ͳ��ƥ�����������������ڵ�(Inlier)����Ч�����
		int N = 0;
		for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
			if (vbMatchesInliers[i])
				N++;

		// �ο�SLAMʮ�Ľ��ڶ���p170-p171
		// H = K * (R - t * n / d) * K_inv
		// ����: K��ʾ�ڲ�������
		//       K_inv ��ʾ�ڲ����������
		//       R �� t ��ʾ��ת��ƽ������
		//       n ��ʾƽ�淨����
		// �� H = K * A * K_inv
		// �� A = k_inv * H * k
		cv::Mat invK = K.inv();
		cv::Mat A = invK * H21*K;

		// �Ծ���A����SVD�ֽ�
		// A �ȴ�����������ֵ�ֽ�ľ���
		// w ����ֵ����
		// U ����ֵ�ֽ������
		// Vt ����ֵ�ֽ��Ҿ���ע�⺯�����ص���ת��
		// cv::SVD::FULL_UV ȫ���ֽ�
		// A = U * w * Vt
		cv::Mat U, w, Vt, V;
		cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);

		// ��������eq(8)�������������
		V = Vt.t();

		// �������s = det(U) * det(V)
		// ��Ϊdet(V)==det(Vt), ���� s = det(U) * det(Vt)
		float s = cv::determinant(U)*cv::determinant(Vt);

		// ȡ�þ���ĸ�������ֵ
		float d1 = w.at<float>(0);
		float d2 = w.at<float>(1);
		float d3 = w.at<float>(2);

		// SVD�ֽ��������������ֵdiӦ�������ģ�������d1>=d2>=d3
		if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001) {
			return false;
		}

		// ��ORBSLAM��û�ж�����ֵ d1 d2 d3���������������Ĺ�ϵ���з�������, ����ֱ�ӽ����˼���
		// ����8������µ���ת����ƽ�������Ϳռ�����
		std::vector<cv::Mat> vR, vt, vn;
		vR.reserve(8);
		vt.reserve(8);
		vn.reserve(8);

		// Step 1.1 ���� d' > 0 ʱ�� 4 ���
		// ��������eq.(12)��
		// x1 = e1 * sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
		// x2 = 0
		// x3 = e3 * sqrt((d2 * d2 - d2 * d2) / (d1 * d1 - d3 * d3))
		// �� aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3))
		//    aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3))
		// ��
		// x1 = e1 * aux1
		// x3 = e3 * aux2

		// ��Ϊ e1,e2,e3 = 1 or -1
		// ������x1��x3���������
		// x1 =  {aux1,aux1,-aux1,-aux1}
		// x3 =  {aux3,-aux3,aux3,-aux3}

		float aux1 = sqrt((d1*d1 - d2 * d2) / (d1*d1 - d3 * d3));
		float aux3 = sqrt((d2*d2 - d3 * d3) / (d1*d1 - d3 * d3));
		float x1[] = { aux1,aux1,-aux1,-aux1 };
		float x3[] = { aux3,-aux3,aux3,-aux3 };

		// ��������eq.(13)��
		// sin(theta) = e1 * e3 * sqrt(( d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) /(d1 + d3)/d2
		// cos(theta) = (d2* d2 + d1 * d3) / (d1 + d3) / d2 
		// ��  aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2)
		// ��  sin(theta) = e1 * e3 * aux_stheta
		//     cos(theta) = (d2*d2+d1*d3)/((d1+d3)*d2)
		// ��Ϊ e1 e2 e3 = 1 or -1
		// ���� sin(theta) = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta}
		float aux_stheta = sqrt((d1*d1 - d2 * d2)*(d2*d2 - d3 * d3)) / ((d1 + d3)*d2);
		float ctheta = (d2*d2 + d1 * d3) / ((d1 + d3)*d2);
		float stheta[] = { aux_stheta, -aux_stheta, -aux_stheta, aux_stheta };

		// ������ת���� R'
		//���ݲ�ͬ��e1 e3������ó���������R t�Ľ�
		//      | ctheta      0   -aux_stheta|       | aux1|
		// Rp = |    0        1       0      |  tp = |  0  |
		//      | aux_stheta  0    ctheta    |       |-aux3|

		//      | ctheta      0    aux_stheta|       | aux1|
		// Rp = |    0        1       0      |  tp = |  0  |
		//      |-aux_stheta  0    ctheta    |       | aux3|

		//      | ctheta      0    aux_stheta|       |-aux1|
		// Rp = |    0        1       0      |  tp = |  0  |
		//      |-aux_stheta  0    ctheta    |       |-aux3|

		//      | ctheta      0   -aux_stheta|       |-aux1|
		// Rp = |    0        1       0      |  tp = |  0  |
		//      | aux_stheta  0    ctheta    |       | aux3|
		// ��ʼ��������������е�ÿһ��
		for (int i = 0; i < 4; i++)
		{
			//����Rp������eq.(8) �� R'
			cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
			Rp.at<float>(0, 0) = ctheta;
			Rp.at<float>(0, 2) = -stheta[i];
			Rp.at<float>(2, 0) = stheta[i];
			Rp.at<float>(2, 2) = ctheta;

			// eq.(8) ����R
			cv::Mat R = s * U*Rp*Vt;

			// ����
			vR.push_back(R);

			// eq. (14) ����tp 
			cv::Mat tp(3, 1, CV_32F);
			tp.at<float>(0) = x1[i];
			tp.at<float>(1) = 0;
			tp.at<float>(2) = -x3[i];
			tp *= d1 - d3;

			// ������Ȼ��t�й�һ������û�о�����Ŀ����SLAM���̵ĳ߶�
			// ��ΪCreateInitialMapMonocular������3D����Ȼ����ţ�Ȼ�󷴹����� t �иı�
			// eq.(8)�ָ�ԭʼ��t
			cv::Mat t = U * tp;
			vt.push_back(t / cv::norm(t));

			// ���취����np
			cv::Mat np(3, 1, CV_32F);
			np.at<float>(0) = x1[i];
			np.at<float>(1) = 0;
			np.at<float>(2) = x3[i];

			// eq.(8) �ָ�ԭʼ�ķ�����
			cv::Mat n = V * np;
			//��PPT 16ҳ��ͼ������ƽ�淨��������
			if (n.at<float>(2) < 0)
				n = -n;
			// ��ӵ�vector
			vn.push_back(n);
		}

		// Step 1.2 ���� d' < 0 ʱ�� 4 ���
		float aux_sphi = sqrt((d1*d1 - d2 * d2)*(d2*d2 - d3 * d3)) / ((d1 - d3)*d2);
		// cos_theta��
		float cphi = (d1*d3 - d2 * d2) / ((d1 - d3)*d2);
		// ���ǵ�e1,e2��ȡֵ�������sin_theta�����ֿ��ܵĽ�
		float sphi[] = { aux_sphi, -aux_sphi, -aux_sphi, aux_sphi };

		// ����ÿ����e1 e3ȡֵ����϶��γɵ����ֽ�����
		for (int i = 0; i < 4; i++)
		{
			// ������ת���� R'
			cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
			Rp.at<float>(0, 0) = cphi;
			Rp.at<float>(0, 2) = sphi[i];
			Rp.at<float>(1, 1) = -1;
			Rp.at<float>(2, 0) = sphi[i];
			Rp.at<float>(2, 2) = -cphi;

			// �ָ���ԭ����R
			cv::Mat R = s * U*Rp*Vt;
			// Ȼ����ӵ�vector��
			vR.push_back(R);

			// ����tp
			cv::Mat tp(3, 1, CV_32F);
			tp.at<float>(0) = x1[i];
			tp.at<float>(1) = 0;
			tp.at<float>(2) = x3[i];
			tp *= d1 + d3;

			// �ָ���ԭ����t
			cv::Mat t = U * tp;
			// ��һ��֮����뵽vector��,Ҫ�ṩ�������ƽ�ƾ�����Ҫ���й���һ����
			vt.push_back(t / cv::norm(t));

			// ���취����np
			cv::Mat np(3, 1, CV_32F);
			np.at<float>(0) = x1[i];
			np.at<float>(1) = 0;
			np.at<float>(2) = x3[i];

			// �ָ���ԭ���ķ�����
			cv::Mat n = V * np;
			// ��֤������ָ���Ϸ�
			if (n.at<float>(2) < 0)
				n = -n;
			// ��ӵ�vector��
			vn.push_back(n);
		}

		// ��õ�good��
		int bestGood = 0;
		// �����õ�good��
		int secondBestGood = 0;
		// ��õĽ����������ʼֵΪ-1
		int bestSolutionIdx = -1;
		// �����Ӳ��
		float bestParallax = -1;
		// �洢��ý��Ӧ�ģ���������Խ������ǻ������Ľ��
		std::vector<cv::Point3f> bestP3D;
		// ��ѽ�����Ӧ�ģ���Щ���Ա����ǻ������ĵ�ı��
		std::vector<bool> bestTriangulated;

		// Step 2. �� 8 ��������֤����ѡ��������ǰ�����3D��Ľ�Ϊ���Ž�
		for (size_t i = 0; i < 8; i++)
		{
			// ���ǻ�����֮���������Ŀռ�����
			std::vector<cv::Point3f> vP3Di;
			// ��������Ƿ����ǻ��ı��
			std::vector<bool> vbTriangulatedi;

			// ���� Initializer::CheckRT(), ����good�����Ŀ
			int nGood = CheckRT(vR[i], vt[i],                    //��ǰ������ת�����ƽ������
				mvKeys1, mvKeys2,                //������
				mvMatches12, vbMatchesInliers,   //����ƥ���ϵ�Լ�Inlier���
				K,                               //������ڲ�������
				vP3Di,                           //�洢���ǻ�����֮���������ռ������
				mSigma2,						 //���ǻ�����������������ͶӰ���
				vbTriangulatedi);                //�������Ƿ񱻳ɹ��������ǲ����ı��

			// ������ʷ���źʹ��ŵĽ�
			// �������ŵĺʹ��ŵĽ�.������Ž��Ŀ���ǿ������Ž��Ƿ�ͻ��
			if (nGood > bestGood)
			{
				// �����ǰ����good��������ʷ���ţ���ô֮ǰ����ʷ���žͱ������ʷ����
				secondBestGood = bestGood;
				// ������ʷ���ŵ�
				bestGood = nGood;
				// ���Ž��������Ϊi�����ǵ�ǰ�α�����
				bestSolutionIdx = i;
				// ���±���
				bestP3D = vP3Di;
				bestTriangulated = vbTriangulatedi;
			}
			// �����ǰ���good����С����ʷ���ŵ�ȴ������ʷ����
			else if (nGood > secondBestGood)
			{
				// ˵����ǰ�������ʷ���ŵ㣬����֮
				secondBestGood = nGood;
			}
		}

		// Step 3 ѡ�����Ž⡣Ҫ����������ĸ�����
		// 1. good�������Ž����Դ��ڴ��Ž⣬����ȡ0.75����ֵ
		// 2. �ӽǲ���ڹ涨����ֵ
		// 3. good����Ҫ���ڹ涨����С�ı����ǻ��ĵ�����
		// 4. good��Ҫ�㹻�࣬�ﵽ������90%����
		if (secondBestGood<0.75*bestGood &&
			bestGood>minTriangulated &&
			bestGood > 0.9*N)
		{
			// ����ѵĽ���������ʵ�R��t
			vR[bestSolutionIdx].copyTo(R21);
			vt[bestSolutionIdx].copyTo(t21);
			// �����ѽ�ʱ���ɹ����ǻ�����ά�㣬�Ժ���Ϊ��ʼ��ͼ��ʹ��
			vP3D = bestP3D;
			// ��ȡ������ı��ɹ��������ǻ��ı��
			vbTriangulated = bestTriangulated;

			//�����棬�ҵ�����õĽ�
			return true;
		}
		return false;
	}

	/**
	 * @brief �ӻ�������F�����λ��R��t����ά��
	 * F�ֽ��E��E������⣬ѡ��������Ч��ά�㣨������ͷǰ����ͶӰ���С����ֵ���Ӳ�Ǵ�����ֵ��������Ϊ���ŵĽ�
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
	bool Initializer::ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
		cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
	{
		// Step 1 ͳ����Чƥ������������ N ��ʾ
		// vbMatchesInliers �д洢ƥ�����Ƿ�����Ч
		int N = 0;
		for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
			if (vbMatchesInliers[i]) N++;

		// Step 2 ���ݻ��������������ڲ���������㱾�ʾ���
		cv::Mat E21 = K.t()*F21*K;

		// ���屾�ʾ���ֽ������γ������,�ֱ��ǣ�
		// (R1, t) (R1, -t) (R2, t) (R2, -t)
		cv::Mat R1, R2, t;

		// Step 3 �ӱ��ʾ����������R�������t�⣬�������
		// ������������t�⻥Ϊ�෴�������������ֻ��ȡһ��
		// ��Ȼ���������t�й�һ��������û�о�����Ŀ����SLAM���̵ĳ߶�. 
		// ��Ϊ CreateInitialMapMonocular ������3D����Ȼ����ţ�Ȼ�󷴹����� t �иı�.
		//ע�������еķ��š�'����ʾ�����ת��
		//                          |0 -1  0|
		// E = U Sigma V'   let W = |1  0  0|
		//                          |0  0  1|
		// �õ�4���� E = [R|t]
		// R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3
		DecomposeE(E21, R1, R2, t);
		cv::Mat t1 = t;
		cv::Mat t2 = -t;

		// Step 4 �ֱ���֤����4��R��t����ϣ�ѡ��������
		// ԭ����ĳһ���ʹ�ָ��õ���3D��λ�������ǰ����������࣬��ô����Ͼ���������
		// ʵ�֣����ݼ���Ľ���ϳ�Ϊ�������,�����ε��� Initializer::CheckRT() ���м��,�õ����Խ������ǻ������ĵ����Ŀ
		// ���������ֱ��ڶ�ͬһƥ��㼯�������ǻ�����֮���������ռ�����
		std::vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;

		// ���������ֱ��ͬһƥ��㼯����Ч���ǻ������True or False
		std::vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;

		// Step 4.1 ʹ��ͬ����ƥ���ֱ�������⣬��¼��ǰ�����3D��������ͷǰ����ͶӰ���С����ֵ�ĸ�������Ϊ��Ч3D�����
		int nGood1 = CheckRT(R1, t1,							//��ǰ���
			mvKeys1, mvKeys2,				//�ο�֡�͵�ǰ֡�е�������
			mvMatches12, vbMatchesInliers,	//�������ƥ���ϵ��Inliers���
			K, 								//������ڲ�������
			vP3D1,							//�洢���ǻ��Ժ�������Ŀռ�����
			mSigma2,						//���ǻ���������������������ͶӰ���
			vbTriangulated1);				//�ο�֡�б��ɹ��������ǻ�������������ı��
		int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D2, mSigma2, vbTriangulated2);
		int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D3, mSigma2, vbTriangulated3);
		int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D4, mSigma2, vbTriangulated4);

		// Step 4.2 ѡȡ�������ǻ������ĵ����Ŀ
		int maxGood = std::max(nGood1, std::max(nGood2, std::max(nGood3, nGood4)));

		// ���ñ��������ں��渳ֵΪ���R��T
		R21 = cv::Mat();
		t21 = cv::Mat();

		// ͳ����������ؽ�����Ч3D����� > 0.7 * maxGood �Ľ����Ŀ
		// ����ж����ͬʱ�������������Ϊ���̫�ӽ���nsimilar++��nsimilar>1����Ϊ�������ˣ����淵��false
		int nsimilar = 0;
		if (nGood1 > 0.7*maxGood)
			nsimilar++;
		if (nGood2 > 0.7*maxGood)
			nsimilar++;
		if (nGood3 > 0.7*maxGood)
			nsimilar++;
		if (nGood4 > 0.7*maxGood)
			nsimilar++;

		// Step 4.4 �ĸ���������û�����Ե����Ž��������û���㹻���������ǻ��㣬�򷵻�ʧ��
		// ����1: ���������ܹ��ؽ������3D�����С����Ҫ�������3D�������mMinGood����ʧ��
		// ����2: ����������鼰���ϵĽ������ǻ��� >0.7*maxGood�ĵ㣬˵��û���������Ž����ʧ��
		if (maxGood < minTriangulated || nsimilar > 1)
		{
			return false;
		}

		//  Step 4.5 ѡ����ѽ��¼���
		// ����1: ��Ч�ؽ�����3D�㣬��maxGood == nGoodx��Ҳ����λ�����ǰ����3D��������
		// ����2: ���ǻ��Ӳ�� parallax ���������С�Ӳ�� minParallax���Ƕ�Խ��3D��Խ�ȶ�

		//������õ�good���������ֽ�������·�����
		if (maxGood == nGood1)
		{
			// �洢3D����
			vP3D = vP3D1;

			// ��ȡ���������������ǻ��������
			vbTriangulated = vbTriangulated1;

			// �洢�����̬
			R1.copyTo(R21);
			t1.copyTo(t21);

			// ����
			return true;
		}
		else if (maxGood == nGood2)
		{
			vP3D = vP3D2;
			vbTriangulated = vbTriangulated2;

			R2.copyTo(R21);
			t1.copyTo(t21);
			return true;
		}
		else if (maxGood == nGood3)
		{
			vP3D = vP3D3;
			vbTriangulated = vbTriangulated3;

			R1.copyTo(R21);
			t2.copyTo(t21);
			return true;
		}
		else if (maxGood == nGood4)
		{
			vP3D = vP3D4;
			vbTriangulated = vbTriangulated4;

			R2.copyTo(R21);
			t2.copyTo(t21);
			return true;
		}

		// ��������Ž⵫�ǲ������Ӧ��parallax>minParallax����ô����false��ʾ���ʧ��
		return false;
	}

	/**
	 * @brief ��λ����������ƥ������ǻ�������ɸѡ�кϸ����ά��
	 *
	 * @param[in] R                                     ��ת����R
	 * @param[in] t                                     ƽ�ƾ���t
	 * @param[in] vKeys1                                �ο�֡������
	 * @param[in] vKeys2                                ��ǰ֡������
	 * @param[in] vMatches12                            ��֡�������ƥ���ϵ
	 * @param[in] vbMatchesInliers                      ��������ڵ���
	 * @param[in] K                                     ����ڲξ���
	 * @param[in & out] vP3D                            ���ǻ�����֮���������Ŀռ�����
	 * @param[in] th2                                   ��ͶӰ������ֵ
	 * @param[in & out] vbGood                          ��ǳɹ����ǻ��㣿
	 * @return int
	 */
	int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
		const std::vector<Match> &vMatches12, std::vector<bool> &vbMatchesInliers,
		const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood)
	{
		// �Ը�����������Լ���R t , ͨ�����ǻ��������Ч�ԣ�Ҳ��Ϊ cheirality check

		// Calibration parameters
		//������ڲ��������ȡ�����У������
		const float fx = K.at<float>(0, 0);
		const float fy = K.at<float>(1, 1);
		const float cx = K.at<float>(0, 2);
		const float cy = K.at<float>(1, 2);

		//�������Ƿ���good��ı�ǣ������������ָ���ǲο�֡�е�������
		vbGood = std::vector<bool>(vKeys1.size(), false);
		//����洢�ռ�����ĵ�Ĵ�С
		vP3D.resize(vKeys1.size());

		//�洢���������ÿ����������Ӳ�
		std::vector<float> vCosParallax;
		vCosParallax.reserve(vKeys1.size());

		// Camera 1 Projection Matrix K[I|0]
		// Step 1�����������ͶӰ����  
		// ͶӰ����P��һ�� 3x4 �ľ��󣬿��Խ��ռ��е�һ����ͶӰ��ƽ���ϣ������ƽ�����꣬�����ָ����������ꡣ
		// ���ڵ�һ������� P1=K*[I|0]

		// �Ե�һ������Ĺ�����Ϊ��������ϵ, ���������ͶӰ����
		cv::Mat P1(3, 4,				//����Ĵ�С��3x4
			CV_32F,						//���������Ǹ�����
			cv::Scalar(0));				//��ʼ����ֵ��0
		//������K���󿽱���P1��������3x3������Ϊ K*I = K
		K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
		// ��һ������Ĺ�������Ϊ��������ϵ�µ�ԭ��
		cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

		// Camera 2 Projection Matrix K[R|t]
		// ����ڶ��������ͶӰ���� P2=K*[R|t]
		cv::Mat P2(3, 4, CV_32F);
		R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
		t.copyTo(P2.rowRange(0, 3).col(3));
		//���ս����K*[R|t]
		P2 = K * P2;
		// �ڶ�������Ĺ�������������ϵ�µ�����
		cv::Mat O2 = -R.t()*t;

		//�ڱ�����ʼǰ���Ƚ�good���������Ϊ0
		int nGood = 0;

		// ��ʼ�������е��������
		for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
		{
			// ����outliers
			if (!vbMatchesInliers[i])
				continue;

			// Step 2 ��ȡ������ԣ�����Triangulate() �����������ǻ����õ����ǻ�����֮���3D������
			// kp1��kp2��ƥ��õ���Ч������
			const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
			const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];

			//�洢��ά��ĵ�����
			cv::Mat p3dC1;

			// �������Ƿ��ָ���ά��p3dC1
			Triangulate(kp1, kp2,	//������
				P1, P2,				//ͶӰ����
				p3dC1);				//��������ǻ�����֮��������Ŀռ�����		

			// Step 3 ��һ�أ�������ǻ�����ά�������Ƿ�Ϸ���������ֵ��
			// ֻҪ���ǲ����Ľ������һ���������ľ�˵�����ǻ�ʧ�ܣ������Ե�ǰ��Ĵ���������һ��������ı��� 
			if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
			{
				//��ʵ��������ǲ�����дҲû���⣬��ΪĬ�ϵ�ƥ���ԾͲ���good��
				vbGood[vMatches12[i].first] = false;
				//��������һ��ƥ���Ĵ���
				continue;
			}

			// Step 4 �ڶ��أ�ͨ����ά�����ֵ����������������Ӳ�Ǵ�С������Ƿ�Ϸ� 
			// ���ռ��p3dC1�任����2���������ϵ�±�Ϊp3dC2
			cv::Mat p3dC2 = R * p3dC1 + t;
			if ((p3dC1.at<float>(2) <= 0 || p3dC2.at<float>(2) <= 0))
				continue;

			// Step 5 �����أ�����ռ���ڲο�֡�͵�ǰ֡�ϵ���ͶӰ�����������ֵ������
			// ����3D���ڵ�һ��ͼ���ϵ�ͶӰ���
			//ͶӰ���ο�֡ͼ���ϵĵ������x,y
			float im1x, im1y;
			//���ʹ�ܿռ���z����ĵ���
			float invZ1 = 1.0 / p3dC1.at<float>(2);
			//ͶӰ���ο�֡ͼ���ϡ���Ϊ�ο�֡�µ��������ϵ����������ϵ�غϣ���������ֱ�ӽ���ͶӰ�Ϳ�����
			im1x = fx * p3dC1.at<float>(0)*invZ1 + cx;
			im1y = fy * p3dC1.at<float>(1)*invZ1 + cy;

			//�ο�֡�ϵ���ͶӰ�������ȷ���ǰ��ն�������
			float squareError1 = (im1x - kp1.pt.x)*(im1x - kp1.pt.x) + (im1y - kp1.pt.y)*(im1y - kp1.pt.y);

			// ��ͶӰ���̫��������̭
			if (squareError1 > th2)
				continue;

			// ����3D���ڵڶ���ͼ���ϵ�ͶӰ��������̺͵�һ��ͼ������
			float im2x, im2y;
			// ע�������p3dC2�Ѿ��ǵڶ����������ϵ�µ���ά����
			float invZ2 = 1.0 / p3dC2.at<float>(2);
			im2x = fx * p3dC2.at<float>(0)*invZ2 + cx;
			im2y = fy * p3dC2.at<float>(1)*invZ2 + cy;

			// ������ͶӰ���
			float squareError2 = (im2x - kp2.pt.x)*(im2x - kp2.pt.x) + (im2y - kp2.pt.y)*(im2y - kp2.pt.y);

			// ��ͶӰ���̫��������̭
			if (squareError2 > th2)
				continue;
			
			//���ǻ��ɹ�
			vbGood[vMatches12[i].first] = true;

			//�洢������ǻ��������3D������������ϵ�µ�����
			vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
			//good�����++
			nGood++;
		}

		//����good�����
		return nGood;
	}

	/** ����ͶӰ����P1,P2��ͼ���ϵ�ƥ���������kp1,kp2���Ӷ�������ά������
	 * @brief
	 *
	 * @param[in] kp1               ������, in reference frame
	 * @param[in] kp2               ������, in current frame
	 * @param[in] P1                ͶӰ����P1
	 * @param[in] P2                ͶӰ����P2
	 * @param[in & out] x3D         �������ά��
	 */
	void Initializer::Triangulate(
		const cv::KeyPoint &kp1,    //������, in reference frame
		const cv::KeyPoint &kp2,    //������, in current frame
		const cv::Mat &P1,          //ͶӰ����P1
		const cv::Mat &P2,          //ͶӰ����P2
		cv::Mat &x3D)               //��ά��
	{
		// ԭ��
		// Trianularization: ��֪ƥ���������{x x'} �� �����������{P P'}, ������ά�� X
		// x' = P'X  x = PX
		// ���Ƕ����� x = aPXģ��
		//                         |X|
		// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
		// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
		// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
		// ����DLT�ķ�����x���PX = 0
		// |yp2 -  p1|     |0|
		// |p0 -  xp2| X = |0|
		// |xp1 - yp0|     |0|
		// ������:
		// |yp2   -  p1  |     |0|
		// |p0    -  xp2 | X = |0| ===> AX = 0
		// |y'p2' -  p1' |     |0|
		// |p0'   - x'p2'|     |0|
		// ��ɳ����е���ʽ��
		// |xp2  - p0 |     |0|
		// |yp2  - p1 | X = |0| ===> AX = 0
		// |x'p2'- p0'|     |0|
		// |y'p2'- p1'|     |0|
		// Ȼ��������һ����Ԫһ�����������飬SVD��⣬�������������һ�о������յĽ�.

		//�����������ע���еľ���A
		cv::Mat A(4, 4, CV_32F);

		//�����������A
		A.row(0) = kp1.pt.x*P1.row(2) - P1.row(0);
		A.row(1) = kp1.pt.y*P1.row(2) - P1.row(1);
		A.row(2) = kp2.pt.x*P2.row(2) - P2.row(0);
		A.row(3) = kp2.pt.y*P2.row(2) - P2.row(1);

		//����ֵ�ֽ�Ľ��
		cv::Mat u, w, vt;
		//��ϵ������A��������ֵ�ֽ�
		cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
		//����ǰ��Ľ��ۣ�����ֵ�ֽ��Ҿ�������һ����ʵ���ǽ⣬ԭ��������ǰ�������С���˽⣬�ĸ�δ֪���ĸ�������������
		//���������Ǹ�ϰ��������������ʾһ����Ŀռ�����
		x3D = vt.row(3).t();
		//Ϊ�˷�������������ʽ��ʹ���һάΪ1
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
	}

	/**
	 * @brief �ֽ�Essential����õ�R,t
	 * �ֽ�E���󽫵õ�4��⣬��4���ֱ�Ϊ[R1,t],[R1,-t],[R2,t],[R2,-t]
	 * �ο���Multiple View Geometry in Computer Vision - Result 9.19 p259
	 * @param[in] E                 ���ʾ���
	 * @param[in & out] R1          ��ת����1
	 * @param[in & out] R2          ��ת����2
	 * @param[in & out] t           ƽ������������һ��ȡ�෴��
	 */
	void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
	{

		// �Ա��ʾ����������ֵ�ֽ�
		//׼���洢�Ա��ʾ����������ֵ�ֽ�Ľ��
		cv::Mat u, w, vt;
		//�Ա��ʾ����������ֵ�ֽ�
		cv::SVD::compute(E, w, u, vt);

		// ������ֵ����U�����һ�о���t��������й�һ��
		u.col(2).copyTo(t);
		t = t / cv::norm(t);

		// ����һ����Z����תpi/2����ת����W��������ʽ��ϵõ���ת���� R1 = u*W*vt
		//������ɺ�Ҫ���һ����ת��������ʽ����ֵ��ʹ����������ʽΪ1��Լ��
		cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
		W.at<float>(0, 1) = -1;
		W.at<float>(1, 0) = 1;
		W.at<float>(2, 2) = 1;

		//����
		R1 = u * W*vt;
		//��ת����������ʽΪ+1��Լ����������������Ϊ��ֵ����Ҫȡ��
		if (cv::determinant(R1) < 0)
			R1 = -R1;

		// ͬ������Wȡת����������ͬ�Ĺ�ʽ������ת����R2 = u*W.t()*vt

		R2 = u * W.t()*vt;
		//��ת����������ʽΪ1��Լ��
		if (cv::determinant(R2) < 0)
			R2 = -R2;
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