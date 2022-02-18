#pragma once

#include "MapPoint.h"
#include "orbDetector.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <opencv2/opencv.hpp>

namespace PoseTracking
{
	class Map;
	class MapPoint;
	class Frame;
	class KeyFrameDatabase;

	class KeyFrame
	{
	public:
		/**
		 * @brief ���캯��
		 * @param[in] F         ������ͨ֡�Ķ���
		 * @param[in] pMap      �����ĵ�ͼָ��
		 * @param[in] pKFDB     ʹ�õĴʴ�ģ�͵�ָ��
		 */
		KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

		/**
		 * @brief ���õ�ǰ�ؼ�֡��λ��
		 * @param[in] Tcw λ��
		 */
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();                  ///< ��ȡλ��
		cv::Mat GetPoseInverse();           ///< ��ȡλ�˵���
		cv::Mat GetCameraCenter();          ///< ��ȡ(��Ŀ)���������
		cv::Mat GetStereoCenter();          ///< ��ȡ˫Ŀ���������,���ֻ���ڿ��ӻ���ʱ��Ż��õ�
		cv::Mat GetRotation();              ///< ��ȡ��̬
		cv::Mat GetTranslation();           ///< ��ȡλ��

		// ====================== MapPoint observation functions ==================================
		/**
		 * @brief Add MapPoint to KeyFrame
		 * @param pMP MapPoint
		 * @param idx MapPoint��KeyFrame�е�����
		 */
		void AddMapPoint(MapPoint* pMP, const size_t &idx);
		/**
		 * @brief ����������ԭ��,���µ�ǰ�ؼ�֡�۲⵽��ĳ����ͼ�㱻ɾ��(bad==true)��,������"֪ͨ"��ǰ�ؼ�֡�����ͼ���Ѿ���ɾ����
		 * @param[in] idx ��ɾ���ĵ�ͼ������
		 */
		void EraseMapPointMatch(const size_t &idx);

		/**
		 * @brief Get MapPoint Matches ��ȡ�ùؼ�֡��MapPoints
		 */
		std::vector<MapPoint*> GetMapPointMatches();

		/** @brief ���ص�ǰ�ؼ�֡�Ƿ��Ѿ��군�� */
		bool isBad();

		// Compute Scene Depth (q=2 median). Used in monocular.
		/**
		 * @brief ������ǰ�ؼ�֡������ȣ�q=2��ʾ��ֵ
		 * @param q q=2
		 * @return Median Depth
		 */
		float ComputeSceneMedianDepth(const int q);

	public:
		// nNextID���ָ�ΪnLastID�����ʣ���ʾ��һ��KeyFrame��ID��
		static long unsigned int nNextId;
		// ��nNextID�Ļ����ϼ�1�͵õ���mnID��Ϊ��ǰKeyFrame��ID��
		long unsigned int mnId;
		// ÿ��KeyFrame��������������һ��Frame��KeyFrame��ʼ����ʱ����ҪFrame��
		// mnFrameId��¼�˸�KeyFrame�����ĸ�Frame��ʼ����
		const long unsigned int mnFrameId;

		// ��Frame���еĶ�����ͬ
		int mnGridCols;
		int mnGridRows;
		float mfGridElementWidthInv;
		float mfGridElementHeightInv;

		//ԭʼ��ͼ����ȡ����������
		std::vector<cv::KeyPoint> mvKeys;
		//ԭʼ��ͼ����ȡ����������
		std::vector<cv::KeyPoint> mvKeysRight;
		//��Ŀ����ͷ����Ŀ����ͷ�������Ӧ��������
		cv::Mat mDescriptors, mDescriptorsRight;

		//Grid over the image to speed up feature matching ,��ʵӦ��˵�Ƕ�ά��,����ά�� vector�б��������������ڵ������������
		std::vector< std::vector <std::vector<size_t> > > mGrid;

		/**
		* @name ͼ���������Ϣ
		* @{
		*/
		// Scale pyramid info.
		int mnScaleLevels;                  ///<ͼ��������Ĳ���
		float mfScaleFactor;                ///<ͼ��������ĳ߶�����
		float mfLogScaleFactor;             ///<ͼ��������ĳ߶����ӵĶ���ֵ�����ڷ���������߶�Ԥ���ͼ��ĳ߶�

		std::vector<float> mvScaleFactors;		///<ͼ�������ÿһ�����������
		std::vector<float> mvLevelSigma2;		///@todo Ŀǰ��frame.c��û���õ����޷��¶���
		std::vector<float> mvInvLevelSigma2;	///<��������ĵ���

	private:
		// SE3 Pose and camera center
		cv::Mat Tcw;    // ��ǰ�����λ�ˣ���������ϵ���������ϵ
		cv::Mat Twc;    // ��ǰ���λ�˵���
		cv::Mat Ow;     // �������(��Ŀ)����������ϵ�µ�����,�������ͨ֡�еĶ�����һ����

		cv::Mat Cw;     //< Stereo middel point. Only for visualization

		// MapPoints associated to keypoints
		std::vector<MapPoint*> mvpMapPoints;

		float mHalfBaseline = 10; //< ����˫Ŀ�����˵,˫Ŀ������߳��ȵ�һ��. Only for visualization

		// �ڶ�λ�˽��в���ʱ��صĻ�����
		std::mutex mMutexPose;
		// �ڲ�����ǰ�ؼ�֡�������ؼ�֡�Ĺ�ʽ��ϵ��ʱ��ʹ�õ��Ļ�����
		std::mutex mMutexConnections;
		// �ڲ������������йصı�����ʱ��Ļ�����
		std::mutex mMutexFeatures;

		// Bad flags
		bool mbNotErase;            ///< ��ǰ�ؼ�֡�Ѿ��������Ĺؼ�֡�γ��˻ػ���ϵ������ڸ����Ż��Ĺ����в�Ӧ�ñ�ɾ��
		bool mbToBeErased;          ///<
		bool mbBad;                 ///< 
	};
}