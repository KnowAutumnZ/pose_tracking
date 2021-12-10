#include "Frame.h"

namespace PoseTracking
{
	Frame::Frame(const Frame& frame):mbInitialComputations(frame.mbInitialComputations), 
		mnMinX(frame.mnMinX), mnMinY(frame.mnMinY), mnMaxX(frame.mnMaxX), mnMaxY(frame.mnMaxY),
		mfGridElementWidthInv(frame.mfGridElementWidthInv), mfGridElementHeightInv(frame.mfGridElementHeightInv)
	{
		mTimeStamp = frame.mTimeStamp;
		mvKeys = frame.mvKeys;
		mvKeysRight = frame.mvKeysRight;
		mDescriptors = frame.mDescriptors.clone();
		mDescriptorsRight = frame.mDescriptorsRight.clone();

		//�µ�֡����Pose�ĳ�ʼֵ�����ں����Ż�
		if (!frame.mTcw.empty())
			SetPose(frame.mTcw);

		//������ƣ���ʵ����Ҳ�����
		//����û��ʹ��ǰ��������ʽ��ԭ�������mGrid��������vector���Ͷ�����ɵ�vector��
		//�����Լ���֪��vector�ڲ���Դ�벻����丳ֵ��ʽ���ڵ�һά����ֱ��ʹ������ķ������ܻᵼ��
		//����ʹ�ò����ʵĸ��ƺ��������µ�һά�ȵ�vector���ܹ�����ȷ�ء�������
		for (int i = 0; i < FRAME_GRID_COLS; i++)
			for (int j = 0; j < FRAME_GRID_ROWS; j++)
				mGrid[i][j] = frame.mGrid[i][j];
	}

	Frame::Frame(const cv::Mat &imGray, const double &timeStamp, orbDetector* extractor, const cv::Mat &K, const cv::Mat &Distort):mTimeStamp(timeStamp)
	{
		std::vector<cv::KeyPoint> vKeys;

		//����ʹ���˷º�������ɣ���������������� ORBextractor::operator() 
		(*extractor)(imGray,   //����ȡ�������ͼ��
			vKeys,             //������������ڱ�����ȡ���������     
			mDescriptors);     //������������ڱ����������������

		if (vKeys.size() == 0) return;

		//��OpenCV�Ľ����������ڲζ���ȡ������������н��� 
		UndistortKeyPoints(vKeys, K, Distort);

		//  ����ȥ�����ͼ��߽磬����������䵽�����С��������һ�����ڵ�һ֡����������궨���������仯֮�����
		if (mbInitialComputations)
		{
			// ����ȥ�����ͼ��ı߽�
			ComputeImageBounds(imGray, K, Distort);

			// ��ʾһ��ͼ�������൱�ڶ��ٸ�ͼ�������У���
			mfGridElementWidthInv = (float)(FRAME_GRID_COLS) / (float)(mnMaxX - mnMinX);
			// ��ʾһ��ͼ�������൱�ڶ��ٸ�ͼ�������У��ߣ�
			mfGridElementHeightInv = (float)(FRAME_GRID_ROWS) / (float)(mnMaxY - mnMinY);

			//����ĳ�ʼ��������ɣ���־��λ
			mbInitialComputations = false;
		}

		AssignFeaturesToGrid();
	}

	/**
	 * @brief ����ȡ��ORB��������䵽ͼ��������
	 *
	 */
	void Frame::AssignFeaturesToGrid()
	{
		int N = mvKeys.size();
		// Step 1  ���洢��������������� Frame::mGrid Ԥ����ռ�
		// FRAME_GRID_COLS = 64��FRAME_GRID_ROWS=48
		int nReserve = std::ceil(1.0f*N / (FRAME_GRID_COLS*FRAME_GRID_ROWS));
		//��ʼ��mGrid�����ά�����е�ÿһ��vectorԪ�ر�����Ԥ����ռ�
		for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
			for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
				mGrid[i][j].reserve(nReserve);

		// Step 2 ����ÿ�������㣬��ÿ����������mvKeysUn�е�����ֵ�ŵ���Ӧ������mGrid��
		for (int i = 0; i < N; i++)
		{
			//����ĳ�Ա�����л�ȡ�Ѿ�ȥ������������
			const cv::KeyPoint &kp = mvKeys[i];

			//�洢ĳ������������������������꣬nGridPosX��Χ��[0,FRAME_GRID_COLS], nGridPosY��Χ��[0,FRAME_GRID_ROWS]
			int nGridPosX, nGridPosY;
			// ����ĳ������������������������꣬����ҵ����������ڵ��������꣬��¼��nGridPosX,nGridPosY�����true��û�ҵ�����false
			if (PosInGrid(kp, nGridPosX, nGridPosY))
				//����ҵ������������������꣬������������������ӵ���Ӧ���������mGrid��
				mGrid[nGridPosX][nGridPosY].push_back(i);
		}
	}

	// ���������̬
	void Frame::SetPose(cv::Mat Tcw)
	{
		mTcw = Tcw.clone();
		UpdatePoseMatrices();
	}

	//����Tcw����mRcw��mtcw��mRwc��mOw
	void Frame::UpdatePoseMatrices()
	{
		// mOw��    ��ǰ�����������������ϵ������
		// mTcw��   ��������ϵ���������ϵ�ı任����
		// mRcw��   ��������ϵ���������ϵ����ת����
		// mtcw��   ��������ϵ���������ϵ��ƽ������
		// mRwc��   �������ϵ����������ϵ����ת����

		//�ӱ任��������ȡ����ת����
		//ע�⣬rowRange���ֻȡ����Χ����߽磬����ȡ�ұ߽�
		mRcw = mTcw.rowRange(0, 3).colRange(0, 3);

		// mRcw���漴��
		mRwc = mRcw.t();

		// �ӱ任��������ȡ����ת����
		mtcw = mTcw.rowRange(0, 3).col(3);

		// mTcw ������ǵ�ǰ�������ϵ�任����������ϵ�£���Ӧ�Ĺ��ı任����������ϵ�¾��� mTcw���� �ж�Ӧ��ƽ������
		mOw = -mRcw.t()*mtcw;
	}

	/**
	 * @brief ���ڲζ�������ȥ���䣬���������mvKeysUn��
	 *
	 */
	void Frame::UndistortKeyPoints(const std::vector<cv::KeyPoint>& vKeys, const cv::Mat &K, const cv::Mat& Distort)
	{
		// Step 1 �����һ���������Ϊ0������Ҫ��������һ���������k1������Ҫ�ģ�һ�㲻Ϊ0��Ϊ0�Ļ���˵�������������0
		//����mDistCoef�д洢��opencvָ����ʽ��ȥ�����������ʽΪ��(k1,k2,p1,p2,k3)
		if (Distort.at<float>(0) == 0.0)
		{
			mvKeys = vKeys;
			return;
		}

		// Step 2 ������������Ϊ0����OpenCV�������л������
		// NΪ��ȡ��������������Ϊ����OpenCV��������Ҫ�󣬽�N�������㱣����N*2�ľ�����
		int N = vKeys.size();
		cv::Mat mat(N, 2, CV_32F);
		//����ÿ�������㣬�������ǵ����걣�浽������
		for (int i = 0; i < N; i++)
		{
			//Ȼ�����������ĺ�������ֱ𱣴�
			mat.at<float>(i, 0) = mvKeys[i].pt.x;
			mat.at<float>(i, 1) = mvKeys[i].pt.y;
		}

		cv::undistortPoints(
			mat,				//���������������
			mat,				//�����У���������������
			K,					//������ڲ�������
			Distort,			//��������������
			cv::Mat(),			//һ���վ��󣬶�ӦΪ����ԭ���е�R
			K); 				//���ڲ������󣬶�ӦΪ����ԭ���е�P

		//������ֻ��һ��ͨ�����ع����������Ĵ���ʽ
		mat = mat.reshape(1);

		// Step3 �洢У�����������
		mvKeys.resize(N);
		//����ÿһ��������
		for (int i = 0; i < N; i++)
		{
			//����������ȡ���������
			//ע��֮����������������ֱ����������һ������������Ŀ���ǣ��ܹ��õ�Դ������������������
			cv::KeyPoint kp = mvKeys[i];
			//��ȡУ��������겢����������
			kp.pt.x = mat.at<float>(i, 0);
			kp.pt.y = mat.at<float>(i, 1);
			mvKeys[i] = kp;
		}
	}

	/**
	 * @brief ����ȥ����ͼ��ı߽�
	 *
	 * @param[in] imLeft            ��Ҫ����߽��ͼ��
	 */
	void Frame::ComputeImageBounds(const cv::Mat &imLeft, const cv::Mat &K, const cv::Mat& Distort)
	{
		// ������������Ϊ0����OpenCV�������л������
		if (Distort.at<float>(0) != 0.0)
		{
			// �������ǰ��ͼ���ĸ��߽�����꣺ (0,0) (cols,0) (0,rows) (cols,rows)
			cv::Mat mat(4, 2, CV_32F);
			mat.at<float>(0, 0) = 0.0;         //����
			mat.at<float>(0, 1) = 0.0;
			mat.at<float>(1, 0) = imLeft.cols; //����
			mat.at<float>(1, 1) = 0.0;
			mat.at<float>(2, 0) = 0.0;         //����
			mat.at<float>(2, 1) = imLeft.rows;
			mat.at<float>(3, 0) = imLeft.cols; //����
			mat.at<float>(3, 1) = imLeft.rows;

			// ��ǰ��У��������һ���Ĳ��������⼸���߽����Ϊ�������У��
			mat = mat.reshape(2);
			cv::undistortPoints(mat, mat, K, Distort, cv::Mat(), K);
			mat = mat.reshape(1);

			//У������ĸ��߽���Ѿ����ܹ�Χ��һ���ϸ�ľ��Σ����������ı��ε����ӱ߿���Ϊ����ı߽�
			mnMinX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));//���Ϻ����º�������С��
			mnMaxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));//���Ϻ����º���������
			mnMinY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));//���Ϻ�������������С��
			mnMaxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));//���º�������������С��
		}
		else
		{
			// ����������Ϊ0����ֱ�ӻ��ͼ��߽�
			mnMinX = 0.0f;
			mnMaxX = imLeft.cols;
			mnMinY = 0.0f;
			mnMaxY = imLeft.rows;
		}
	}

	/**
	 * @brief ����ĳ������������������������꣬����ҵ����������ڵ��������꣬��¼��nGridPosX,nGridPosY�����true��û�ҵ�����false
	 *
	 * @param[in] kp                    ������������
	 * @param[in & out] posX            ������������������ĺ�����
	 * @param[in & out] posY            �������������������������
	 * @return true                     ����ҵ����������ڵ��������꣬����true
	 * @return false                    û�ҵ�����false
	 */
	bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
	{
		// ����������x,y���������ĸ������ڣ���������ΪposX��posY
		// mfGridElementWidthInv=(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
		// mfGridElementHeightInv=(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
		posX = std::floor((kp.pt.x - mnMinX)*mfGridElementWidthInv);
		posY = std::floor((kp.pt.y - mnMinY)*mfGridElementHeightInv);

		// ��Ϊ�����������ȥ���䣬����ǰ�������roundȡ���������п��ܵõ��ĵ�����ͼ��������������
		// �����������posX��posY������[0,FRAME_GRID_COLS] ��[0,FRAME_GRID_ROWS]����ʾ��������û�ж�Ӧ�������꣬����false
		if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
			return false;

		// ����ɹ�����true
		return true;
	}
}