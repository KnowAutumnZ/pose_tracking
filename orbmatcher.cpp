#include "orbmatcher.h"

namespace PoseTracking
{
	// Ҫ�õ���һЩ��ֵ
	const int TH_HIGH = 100;
	const int TH_LOW = 50;
	const int HISTO_LENGTH = 30;

	// ���캯��,����Ĭ��ֵΪ0.6,true
	ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
	{

	}

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
	int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, std::vector<int> &vnMatches12, int windowSize)
	{
		int nmatches = 0;
		// F1���������F2��ƥ���ϵ��ע���ǰ���F1��������Ŀ����ռ�
		vnMatches12 = std::vector<int>(F1.mvKeys.size(), -1);
		
		// Step 1 ������תֱ��ͼ��HISTO_LENGTH = 30
		std::vector<int> rotHist[HISTO_LENGTH];
		// ÿ��bin��Ԥ����30������Ϊʹ�õ���vector�����Ļ������Զ���չ����
		for (int i = 0; i < HISTO_LENGTH; i++)
			rotHist[i].reserve(30);

		//! ԭ���ߴ����� const float factor = 1.0f/HISTO_LENGTH; �Ǵ���ģ�����Ϊ�������   
		const float factor = HISTO_LENGTH / 360.0f;

		// ƥ���Ծ��룬ע���ǰ���F2��������Ŀ����ռ�
		std::vector<int> vMatchedDistance(F2.mvKeys.size(), INT_MAX);
		// ��֡2��֡1�ķ���ƥ�䣬ע���ǰ���F2��������Ŀ����ռ�
		std::vector<int> vnMatches21(F2.mvKeys.size(), -1);

		// ����֡1�е�����������
		for (size_t i1 = 0, iend1 = F1.mvKeys.size(); i1 < iend1; i1++)
		{
			cv::KeyPoint kp1 = F1.mvKeys[i1];
			int level1 = kp1.octave;

			// vbPrevMatched ������ǲο�֡ F1��������
			// windowSize = 100�����������С�������㼶 ��Ϊ0
			std::vector<size_t> vIndices2 = GetFeaturesInArea(F2, F1.mvKeys[i1].pt.x, F1.mvKeys[i1].pt.y, windowSize, level1 - 1, level1 + 1);

			// û�к�ѡ�����㣬����
			if (vIndices2.empty())
				continue;

			// ȡ���ο�֡F1�е�ǰ�����������Ӧ��������
			cv::Mat d1 = F1.mDescriptors.row(i1);

			int bestDist = INT_MAX;     //���������ƥ����룬ԽСԽ��
			int bestDist2 = INT_MAX;    //�μ�������ƥ�����
			int bestIdx2 = -1;          //��Ѻ�ѡ��������F2�е�index

			// Step 3 �����������������е�����Ǳ�ڵ�ƥ���ѡ�㣬�ҵ����ŵĺʹ��ŵ�
			for (auto& vit: vIndices2)
			{
				size_t i2 = vit;
				// ȡ����ѡ�������Ӧ��������
				cv::Mat d2 = F2.mDescriptors.row(i2);
				// �������������������Ӿ���
				int dist = DescriptorDistance(d1, d2);

				// �Ѿ�ƥ����ˣ���һλ
				if (vMatchedDistance[i2] <= dist)
					continue;
				// �����ǰƥ������С��������ѴμѾ���
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

			// Step 4 �����Ŵ��Ž�����м�飬������ֵ������/���ű�����ɾ���ظ�ƥ��
			// ��ʹ��������������ƥ����룬Ҳ��һ����֤��Գɹ���ҪС���趨��ֵ
			if (bestDist <= TH_LOW)
			{
				// ��Ѿ���ȴμѾ���ҪС���趨�ı����������������ʶ�ȸ���
				if (bestDist < (float)bestDist2*mfNNratio)
				{
					// ����ҵ��ĺ�ѡ�������ӦF1���������Ѿ�ƥ����ˣ�˵���������ظ�ƥ�䣬��ԭ����ƥ��Ҳɾ��
					if (vnMatches21[bestIdx2] >= 0)
					{
						vnMatches12[vnMatches21[bestIdx2]] = -1;
						nmatches--;
					}
					// ���ŵ�ƥ���ϵ��˫����
					// vnMatches12����ο�֡F1��F2ƥ���ϵ��index������F1��Ӧ������������ֵ�������ƥ��õ�F2����������
					vnMatches12[i1] = bestIdx2;
					vnMatches21[bestIdx2] = i1;
					vMatchedDistance[bestIdx2] = bestDist;
					nmatches++;

					// Step 5 ����ƥ�����ת�ǶȲ����ڵ�ֱ��ͼ
					if (mbCheckOrientation)
					{
						// ����ƥ��������ĽǶȲ���ﵥλ�ǽǶȡ㣬���ǻ���
						float rot = F1.mvKeys[i1].angle - F2.mvKeys[bestIdx2].angle;
						if (rot < 0.0)
							rot += 360.0f;
						// ǰ��factor = HISTO_LENGTH/360.0f 
						// bin = rot / 360.of * HISTO_LENGTH ��ʾ��ǰrot�������ڵڼ���ֱ��ͼbin  
						int bin = std::floor(rot*factor);
						// ���bin ��������һ���ֻ�
						if (bin == HISTO_LENGTH)
							bin = 0;
						assert(bin >= 0 && bin < HISTO_LENGTH);
						rotHist[bin].push_back(i1);
					}
				}
			}

		}

		// Step 6 ɸ����תֱ��ͼ�С�������������
		if (mbCheckOrientation)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;
			// ɸѡ������ת�ǶȲ�������ֱ��ͼ��������������ǰ����bin������
			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				// �޳�������ǰ����ƥ��ԣ���Ϊ���ǲ����ϡ�������ת����    
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
	 * @brief �ҵ��� ��x,yΪ����,�뾶Ϊr��Բ�����ҽ������㼶��[minLevel, maxLevel]��������
	 *
	 * @param[in] x                     ����������x
	 * @param[in] y                     ����������y
	 * @param[in] r                     �����뾶
	 * @param[in] minLevel              ��С�������㼶
	 * @param[in] maxLevel              ���������㼶
	 * @return vector<size_t>           �����������ĺ�ѡƥ���id
	 */
	std::vector<size_t> ORBmatcher::GetFeaturesInArea(Frame &F, const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel)
	{
		// �洢���������vector
		std::vector<size_t> vIndices;

		int N = F.mvKeys.size();

		// Step 1 ����뾶ΪrԲ�������±߽����ڵ������к��е�id
		// ���Ұ뾶Ϊr��Բ���߽��������������ꡣ����ط��е��ƣ���������£�
		// (mnMaxX-mnMinX)/FRAME_GRID_COLS����ʾ�з���ÿ���������ƽ���ֵü������أ��϶�����1��
		// mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) �����浹������ʾÿ�����ؿ��Ծ��ּ��������У��϶�С��1��
		// (x-mnMinX-r)�����Կ����Ǵ�ͼ�����߽�mnMinX���뾶r��Բ����߽�����ռ����������
		// ������ˣ���������Ǹ��뾶Ϊr��Բ�����߽����ĸ���������
		// ��֤nMinCellX ������ڵ���0
		const int nMinCellX = std::max(0, (int)floor((x - F.mnMinX - r)*F.mfGridElementWidthInv));

		// ���������õ�Բ����߽����ڵ������г������趨�����ޣ���ô��˵����������Ҳ�������Ҫ��������㣬���ؿ�vector
		if (nMinCellX >= FRAME_GRID_COLS)
			return vIndices;

		// ����Բ���ڵ��ұ߽�����������
		const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - F.mnMinX + r)*F.mfGridElementWidthInv));
		// ����������Բ�ұ߽����ڵ����񲻺Ϸ���˵���������㲻�ã�ֱ�ӷ��ؿ�vector
		if (nMaxCellX < 0)
			return vIndices;

		//����Ĳ���Ҳ�������Ƶģ���������Բ���±߽����ڵ������е�id
		const int nMinCellY = std::max(0, (int)floor((y - F.mnMinY - r)*F.mfGridElementHeightInv));
		if (nMinCellY >= FRAME_GRID_ROWS)
			return vIndices;

		const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - F.mnMinY + r)*F.mfGridElementHeightInv));
		if (nMaxCellY < 0)
			return vIndices;

		// Step 2 ����Բ�������ڵ���������Ѱ�����������ĺ�ѡ�����㣬������index�ŵ������
		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				// ��ȡ��������ڵ������������� Frame::mvKeysUn �е�����
				const std::vector<size_t> vCell = F.mGrid[ix][iy];
				// ������������û�������㣬��ô����������������һ��
				if (vCell.empty())
					continue;

				for (size_t i=0; i<vCell.size(); i++)
				{
					// ���������ȶ�ȡ��������� 
					const cv::KeyPoint &kpUn = F.mvKeys[vCell[i]];

					// ��֤���������ڽ������㼶minLevel��maxLevel֮�䣬���ǵĻ�����
					if (kpUn.octave < minLevel || kpUn.octave > maxLevel)
						continue;

					// ͨ����飬�����ѡ�����㵽Բ���ĵľ��룬�鿴�Ƿ��������Բ������֮��
					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					// ���x�����y����ľ��붼��ָ���İ뾶֮�ڣ��洢��indexΪ��ѡ������
					if (sqrt(distx*distx + disty * distx))
						vIndices.push_back(vCell[i]);
				}
			}
		}
		return vIndices;
	}

	/**
	 * @brief ɸѡ������ת�ǶȲ�������ֱ��ͼ��������������ǰ����bin������
	 *
	 * @param[in] histo         ƥ�����������ת�����ֱ��ͼ
	 * @param[in] L             ֱ��ͼ�ߴ�
	 * @param[in & out] ind1          binֵ��һ���Ӧ������
	 * @param[in & out] ind2          binֵ�ڶ����Ӧ������
	 * @param[in & out] ind3          binֵ�������Ӧ������
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

		// ������̫����,˵�����ŵķǳ�����,��������Է�����,����Ϊ-1
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
	// Hamming distance�����������ƴ�֮��ĺ������룬ָ�����䲻ͬλ���ĸ���
	// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
	{
		const int *pa = a.ptr<int32_t>();
		const int *pb = b.ptr<int32_t>();

		int dist = 0;
		// 8*32=256bit
		for (int i = 0; i < 8; i++, pa++, pb++)
		{
			unsigned  int v = *pa ^ *pb;        // ���Ϊ0,����Ϊ1
			// ����Ĳ������Ǽ�������bitΪ1�ĸ�����,�����������������Ӿͺ�
			// ��ʵ�Ҿ���Ҳ������ֱ��ʹ��8bit�Ĳ��ұ�,Ȼ����32��Ѱַ�����������;����ȱ����û�����ú�CPU���ֳ�
			v = v - ((v >> 1) & 0x55555555);
			v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
			dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
		}
		return dist;
	}
}