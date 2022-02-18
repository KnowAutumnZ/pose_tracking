#pragma once

#include <opencv2/opencv.hpp>

#include "orbDetector.h"
#include "MapPoint.h"
#include "KeyFrame.h"

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

	class MapPoint;
	class KeyFrame;

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

		// Returns the camera center.
		/**
		 * @brief ����λ�ڵ�ǰ֡λ��ʱ,���������
		 *
		 * @return cv::Mat �����������������ϵ�µ�3D������
		 */
		inline cv::Mat GetCameraCenter()
		{
			return mOw.clone();
		}

		/**
		 * @brief Get the Rotation Inverse object
		 * mRwc�洢���Ǵӵ�ǰ�������ϵ����������ϵ�����е���ת��������һ���õ���ת��˵���Ǵ���������ϵ����ǰ�������ϵ����ת
		 * @return ���شӵ�ǰ֡����ϵ����������ϵ����ת
		 */
		inline cv::Mat GetRotationInverse()
		{
			return mRwc.clone();
		}

	public:
		//֡��ʱ���
		double mTimeStamp;
		//ԭʼ��ͼ����ȡ����������
		std::vector<cv::KeyPoint> mvKeys;
		//ԭʼ��ͼ����ȡ����������
		std::vector<cv::KeyPoint> mvKeysRight;
		//��Ŀ����ͷ����Ŀ����ͷ�������Ӧ��������
		cv::Mat mDescriptors, mDescriptorsRight;

		// ÿ���������Ӧ��MapPoint.���������û�ж�Ӧ�ĵ�ͼ��,��ô���洢һ����ָ��
		std::vector<MapPoint*> mvpMapPoints;

		// �۲ⲻ��Map�е�3D��
		// ����������������,�� Optimizer::PoseOptimization ʹ����
		std::vector<bool> mvbOutlier;

		cv::Mat mTcw; //< �����̬ ��������ϵ�������������ϵ�ı任����,�����ǳ�������е����λ��

		// Current and Next Frame id.
		// ��ľ�̬��Ա��������Щ��������������ϵͳ��ʼִ�е�ʱ�򱻳�ʼ���ġ�������ȫ��������ʼ��
		static long unsigned int nNextId; //< Next Frame id.
		long unsigned int mnId; //< Current Frame id.

		// ��ͨ֡���Լ����ӳ̶���ߵĹؼ�֡��Ϊ�ο��ؼ�֡
		KeyFrame* mpReferenceKF;

		// Rotation, translation and camera center
		cv::Mat mRcw; //< Rotation from world to camera
		cv::Mat mtcw; //< Translation from world to camera
		cv::Mat mRwc; //< Rotation from camera to world
		cv::Mat mOw;  //< mtwc,Translation from camera to world

		/**
		* @name ͼ���������Ϣ
		* @{
		*/
		// Scale pyramid info.
		int mnScaleLevels;                  ///<ͼ��������Ĳ���
		float mfScaleFactor;                ///<ͼ��������ĳ߶�����
		float mfLogScaleFactor;             ///<ͼ��������ĳ߶����ӵĶ���ֵ�����ڷ���������߶�Ԥ���ͼ��ĳ߶�

		std::vector<float> mvScaleFactors;		///<ͼ�������ÿһ�����������
		std::vector<float> mvInvScaleFactors;	///<�Լ��������������ĵ���
		std::vector<float> mvLevelSigma2;		///@todo Ŀǰ��frame.c��û���õ����޷��¶���
		std::vector<float> mvInvLevelSigma2;	///<��������ĵ���

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