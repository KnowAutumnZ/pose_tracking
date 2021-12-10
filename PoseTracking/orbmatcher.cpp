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



		}

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





	}

}