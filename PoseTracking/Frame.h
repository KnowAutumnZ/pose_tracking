#pragma once

#include <opencv2/opencv.hpp>
#include "orbDetector.h"

namespace PoseTracking
{
	/**
	* @brief ���������
	*
	*/
	#define FRAME_GRID_ROWS 48
	/**
	* @brief ���������
	*
	*/
	#define FRAME_GRID_COLS 64

	class Frame
	{
	public:
		Frame() {};
		virtual ~Frame() {};

		// Copy constructor. �������캯��
		/**
		 * @brief �������캯��
		 * @details ���ƹ��캯��, mLastFrame = Frame(mCurrentFrame) \n
		 * ��������Զ��Կ��������Ļ���ϵͳ�Զ����ɵĿ����������������漰�����ڴ�Ĳ���������ǳ���� \n
		 * @param[in] frame ����
		 * @note ����ע�⣬�������������ʱ��������������ص�thisָ����ʵ��ָ��Ŀ��֡��
		 */
		Frame(const Frame& frame);

		/**
		 * @brief Ϊ��Ŀ���׼����֡���캯��
		 *
		 * @param[in] imGray                            //�Ҷ�ͼ
		 * @param[in] timeStamp                         //ʱ���
		 * @param[in & out] extractor                   //ORB��������ȡ���ľ��
		 * @param[in] K                                 //������ڲ�������
		 * @param[in] Distort                           //�����ȥ�������
		 */
		Frame(const cv::Mat &imGray, const double &timeStamp, orbDetector* extractor, const cv::Mat &K, const cv::Mat& Distort);

		// ��Tcw����mTcw
		/**
		 * @brief �� Tcw ���� mTcw �Լ����д洢��һϵ��λ��
		 *
		 * @param[in] Tcw ����������ϵ����ǰ֡���λ�˵ı任����
		 */
		void SetPose(cv::Mat Tcw);

		/**
		 * @brief �������λ��,�����������ת,ƽ�ƺ�������ĵȾ���.
		 * @details ��ʵ���Ǹ���Tcw����mRcw��mtcw��mRwc��mOw.
		 */
		void UpdatePoseMatrices();

		/**
		 * @brief ���ڲζ�������ȥ���䣬���������mvKeys��
		 *
		 */
		void UndistortKeyPoints(const std::vector<cv::KeyPoint>& vKeys, const cv::Mat &K, const cv::Mat& Distort);

		/**
		 * @brief ����ȥ����ͼ��ı߽�
		 *
		 * @param[in] imLeft            ��Ҫ����߽��ͼ��
		 */
		void ComputeImageBounds(const cv::Mat &imLeft, const cv::Mat &K, const cv::Mat& Distort);

		/**
		 * @brief ����ȡ������������䵽ͼ�������� \n
		 * @details �ú����ɹ��캯������
		 *
		 */
		void AssignFeaturesToGrid();

		/**
		 * @brief ����ĳ������������������������꣬����ҵ����������ڵ��������꣬��¼��nGridPosX,nGridPosY�����true��û�ҵ�����false
		 *
		 * @param[in] kp                    ������������
		 * @param[in & out] posX            ������������������ĺ�����
		 * @param[in & out] posY            �������������������������
		 * @return true                     ����ҵ����������ڵ��������꣬����true
		 * @return false                    û�ҵ�����false
		 */
		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

	public:
		//֡��ʱ���
		double mTimeStamp;
		//ԭʼ��ͼ����ȡ����������
		std::vector<cv::KeyPoint> mvKeys;
		//ԭʼ��ͼ����ȡ����������
		std::vector<cv::KeyPoint> mvKeysRight;
		//��Ŀ����ͷ����Ŀ����ͷ�������Ӧ��������
		cv::Mat mDescriptors, mDescriptorsRight;

		cv::Mat mTcw; //< �����̬ ��������ϵ�������������ϵ�ı任����,�����ǳ�������е����λ��

		// Rotation, translation and camera center
		cv::Mat mRcw; //< Rotation from world to camera
		cv::Mat mtcw; //< Translation from world to camera
		cv::Mat mRwc; //< Rotation from camera to world
		cv::Mat mOw;  //< mtwc,Translation from camera to world

		//����У�����ͼ��߽�
		float mnMinX, mnMinY, mnMaxX, mnMaxY;

		//�Ƿ�Ҫ���г�ʼ�������ı�־
		//����������־��λ�Ĳ����������ϵͳ��ʼ���ص��ڴ��ʱ����еģ���һ֡��������ϵͳ�ĵ�һ֡�����������־Ҫ��λ
		bool mbInitialComputations = true;

		// ��ʾһ��ͼ�������൱�ڶ��ٸ�ͼ�������У���
		float mfGridElementWidthInv;
		// ��ʾһ��ͼ�������൱�ڶ��ٸ�ͼ�������У��ߣ�
		float mfGridElementHeightInv;

		// ÿ�����ӷ����������������ͼ��ֳɸ��ӣ���֤��ȡ��������ȽϾ���
		// FRAME_GRID_ROWS 48
		// FRAME_GRID_COLS 64
		// ��������д洢����ÿ��ͼ���������������id����ͼ��
		std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
	};
}