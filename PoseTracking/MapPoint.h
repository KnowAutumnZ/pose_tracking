#pragma once

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "orbmatcher.h"

#include <opencv2/opencv.hpp>

namespace PoseTracking
{
	class KeyFrame;
	class Map;
	class Frame;

	class MapPoint
	{
	public:
		/**
		 * @brief ����������keyframe����MapPoint
		 * @details ������: ˫Ŀ��StereoInitialization()��CreateNewKeyFrame()��LocalMapping::CreateNewMapPoints() \n
		 * ��Ŀ��CreateInitialMapMonocular()��LocalMapping::CreateNewMapPoints()
		 * @param[in] Pos       MapPoint�����꣨wrt��������ϵ��
		 * @param[in] pRefKF    KeyFrame
		 * @param[in] pMap      Map
		 */
		MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
		/**
		 * @brief ����������frame����MapPoint
		 * @detials ��˫Ŀ��UpdateLastFrame()����
		 * @param[in] Pos       MapPoint�����꣨��������ϵ��
		 * @param[in] pMap      Map
		 * @param[in] pFrame    Frame
		 * @param[in] idxF      MapPoint��Frame�е�����������Ӧ��������ı��
		 */
		MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF);

		/**
		 * @brief ������������ϵ�µ�ͼ���λ��
		 *
		 * @param[in] Pos ��������ϵ�µ�ͼ���λ��
		 */
		void SetWorldPos(const cv::Mat &Pos);
		/**
		 * @brief ��ȡ��ǰ��ͼ������������ϵ�µ�λ��
		 * @return cv::Mat λ��
		 */
		cv::Mat GetWorldPos();

		/**
		 * @brief ��ȡ��ǰ��ͼ���ƽ���۲ⷽ��
		 * @return cv::Mat һ������
		 */
		cv::Mat GetNormal();
		/**
		 * @brief ��ȡ���ɵ�ǰ��ͼ��Ĳο��ؼ�֡
		 * @return KeyFrame*
		 */
		KeyFrame* GetReferenceKeyFrame();

		/**
		 * @brief ��ȡ�۲⵽��ǰ��ͼ��Ĺؼ�֡
		 * @return std::map<KeyFrame*,size_t> �۲⵽��ǰ��ͼ��Ĺؼ�֡���У�
		 *                                    size_t ��������ӦΪ�õ�ͼ���ڸùؼ�֡��������ķ���id
		 */
		std::map<KeyFrame*, size_t> GetObservations();

		// ��ȡ��ǰ��ͼ��ı��۲����
		int Observations();

		/**
		 * @brief ��ӹ۲�
		 *
		 * ��¼��ЩKeyFrame���Ǹ��������ܹ۲⵽��MapPoint \n
		 * �����ӹ۲�������ĿnObs����Ŀ+1��˫Ŀ����grbd+2
		 * ��������ǽ����ؼ�֡���ӹ�ϵ�ĺ��ĺ������ܹ�ͬ�۲⵽ĳЩMapPoints�Ĺؼ�֡�ǹ��ӹؼ�֡
		 * @param[in] pKF KeyFrame,�۲⵽��ǰ��ͼ��Ĺؼ�֡
		 * @param[in] idx MapPoint��KeyFrame�е�����
		 */
		void AddObservation(KeyFrame* pKF, size_t idx);
		/**
		 * @brief ȡ��ĳ���ؼ�֡�Ե�ǰ��ͼ��Ĺ۲�
		 * @detials ���ĳ���ؼ�֡Ҫ��ɾ������ô�ᷢ���������
		 * @param[in] pKF
		 */
		void EraseObservation(KeyFrame* pKF);

		/**
		 * @brief ��ȡ�۲⵽��ǰ��ͼ��Ĺؼ�֡,�ڹ۲������е�����
		 *
		 * @param[in] pKF   �ؼ�֡
		 * @return int      ����
		 */
		int GetIndexInKeyFrame(KeyFrame* pKF);
		/**
		 * @brief �鿴ĳ���ؼ�֡�Ƿ񿴵��˵�ǰ�ĵ�ͼ��
		 *
		 * @param[in] pKF   �ؼ�֡
		 * @return true
		 * @return false
		 */
		bool IsInKeyFrame(KeyFrame* pKF);

		/**
		 * @brief ��֪���Թ۲⵽��MapPoint��Frame����MapPoint�ѱ�ɾ��
		 *
		 */
		void SetBadFlag();
		/**
		 * @brief �жϵ�ǰ��ͼ���Ƿ���bad
		 *
		 * @return true
		 * @return false
		 */
		bool isBad();

		/**
		 * @brief ���γɱջ���ʱ�򣬻���� KeyFrame �� MapPoint ֮��Ĺ�ϵ
		 * ��ʵҲ�����໥�滻?
		 *
		 * @param[in] pMP ��ͼ��
		 */
		void Replace(MapPoint* pMP);
		/**
		 * @brief ��ȡȡ����ǰ��ͼ��ĵ�? //?
		 *
		 * @return MapPoint* //?
		 */
		MapPoint* GetReplaced();

		/**
		 * @brief ���ӿ��Ӵ���
		 * @detials Visible��ʾ��
		 * \n 1. ��MapPoint��ĳЩ֡����Ұ��Χ�ڣ�ͨ��Frame::isInFrustum()�����ж�
		 * \n 2. ��MapPoint����Щ֡�۲⵽��������һ���ܺ���Щ֡��������ƥ����
		 * \n   ���磺��һ��MapPoint����ΪM������ĳһ֡F����Ұ��Χ�ڣ�
		 *    �����������õ�M���Ժ�F��һ֡��ĳ����������ƥ����
		 * @param[in] n Ҫ���ӵĴ���
		 */
		void IncreaseVisible(int n = 1);
		/**
		 * @brief Increase Found
		 * @detials ���ҵ��õ��֡��+n��nĬ��Ϊ1
		 * @param[in] n ���ӵĸ���
		 * @see Tracking::TrackLocalMap()
		 */
		void IncreaseFound(int n = 1);
		//? ���������?
		float GetFoundRatio();
		/**
		 * @brief ��ȡ���ҵ��Ĵ���
		 *
		 * @return int ���ҵ��Ĵ���
		 */
		inline int GetFound() {
			return mnFound;
		}

		/**
		 * @brief ������д����������
		 * @detials ����һ��MapPoint�ᱻ�������۲⵽������ڲ���ؼ�֡����Ҫ�ж��Ƿ���µ�ǰ������ʺϵ������� \n
		 * �Ȼ�õ�ǰ������������ӣ�Ȼ�����������֮����������룬��õ�������������������Ӧ�þ�����С�ľ�����ֵ
		 * @see III - C3.3
		 */
		void ComputeDistinctiveDescriptors();

		/**
		 * @brief ��ȡ��ǰ��ͼ���������
		 *
		 * @return cv::Mat
		 */
		cv::Mat GetDescriptor();

		/**
		 * @brief ����ƽ���۲ⷽ���Լ��۲���뷶Χ
		 *
		 * ����һ��MapPoint�ᱻ�������۲⵽������ڲ���ؼ�֡����Ҫ������Ӧ����
		 * @see III - C2.2 c2.4
		 */
		void UpdateNormalAndDepth();

		//?
		float GetMinDistanceInvariance();
		//?
		float GetMaxDistanceInvariance();
		//? �߶�Ԥ��?
		int PredictScale(const float &currentDist, KeyFrame*pKF);
		//? 
		int PredictScale(const float &currentDist, Frame* pF);

	public:
		long unsigned int mnId; ///< Global ID for MapPoint
		static long unsigned int nNextId;
		const long int mnFirstKFid; ///< ������MapPoint�Ĺؼ�֡ID
		//��,����Ǵ�֡�д����Ļ�,�Ὣ��ͨ֡��id���������
		const long int mnFirstFrame; ///< ������MapPoint��֡ID����ÿһ�ؼ�֡��һ��֡ID��

		// ���۲⵽�������Ŀ����Ŀ+1��˫Ŀ��RGB-D��+2
		int nObs;

		// Variables used by the tracking
		float mTrackProjX;             ///< ��ǰ��ͼ��ͶӰ��ĳ֡�Ϻ������
		float mTrackProjY;             ///< ��ǰ��ͼ��ͶӰ��ĳ֡�Ϻ������
		float mTrackProjXR;            ///< ��ǰ��ͼ��ͶӰ��ĳ֡�Ϻ������(��Ŀ)
		int mnTrackScaleLevel;         ///< �����ĳ߶�, ������������в��� //?
		float mTrackViewCos;           ///< ��׷�ٵ�ʱ,��֡���������ǰ��ͼ����ӽ�
		// TrackLocalMap - SearchByProjection �о����Ƿ�Ըõ����ͶӰ�ı���
		// NOTICE mbTrackInView==false�ĵ��м��֣�
		// a �Ѿ��͵�ǰ֡����ƥ�䣨TrackReferenceKeyFrame��TrackWithMotionModel�������Ż���������Ϊ�����
		// b �Ѿ��͵�ǰ֡����ƥ����Ϊ�ڵ㣬�����Ҳ����Ҫ�ٽ���ͶӰ   //? Ϊʲô�Ѿ����ڵ���֮��Ͳ���Ҫ�ٽ���ͶӰ����? 
		// c ���ڵ�ǰ�����Ұ�еĵ㣨��δͨ��isInFrustum�жϣ�     //? 
		bool mbTrackInView;
		// TrackLocalMap - UpdateLocalPoints �з�ֹ��MapPoints�ظ������mvpLocalMapPoints�ı��
		long unsigned int mnTrackReferenceForFrame;

		// TrackLocalMap - SearchLocalPoints �о����Ƿ����isInFrustum�жϵı���
		// NOTICE mnLastFrameSeen==mCurrentFrame.mnId�ĵ��м��֣�
		// a �Ѿ��͵�ǰ֡����ƥ�䣨TrackReferenceKeyFrame��TrackWithMotionModel�������Ż���������Ϊ�����
		// b �Ѿ��͵�ǰ֡����ƥ����Ϊ�ڵ㣬�����Ҳ����Ҫ�ٽ���ͶӰ
		long unsigned int mnLastFrameSeen;

		//REVIEW �����....��û������
		// Variables used by local mapping
		// local mapping�м�¼��ͼ���Ӧ��ǰ�ֲ�BA�Ĺؼ�֡��mnId��mnBALocalForKF ��map point.h����Ҳ��ͬ���ı�����
		long unsigned int mnBALocalForKF;
		long unsigned int mnFuseCandidateForKF;     ///< �ھֲ���ͼ�߳���ʹ��,��ʾ���������е�ͼ���ںϵĹؼ�֡(�洢��������ؼ�֡��id)

		// Variables used by loop closing -- һ�㶼��Ϊ�˱����ظ�����
		/// ��ǵ�ǰ��ͼ������Ϊ�ĸ�"��ǰ�ؼ�֡"�Ļػ���ͼ��(���ػ��ؼ�֡�ϵĵ�ͼ��),�ڻػ�����߳��б�����
		long unsigned int mnLoopPointForKF;
		// ��������ͼ���Ӧ�Ĺؼ�֡���뵽�˻ػ����Ĺ�����,��ô�ڻػ����������Ѿ�ʹ��������ؼ�֡����ֻ�е�λ���������������ͼ��,��ô�����־λ��λ
		long unsigned int mnCorrectedByKF;
		long unsigned int mnCorrectedReference;
		// ȫ��BA�Ż���(�����ǰ��ͼ��μ��˵Ļ�),�����¼�Ż����λ��
		cv::Mat mPosGBA;
		// �����ǰ���λ�˲��뵽��ȫ��BA�Ż�,��ô���������¼���Ǹ�����ȫ��BA��"��ǰ�ؼ�֡"��id
		long unsigned int mnBAGlobalForKF;

		///ȫ��BA�жԵ�ǰ����в�����ʱ��ʹ�õĻ�����
		static std::mutex mGlobalMutex;

	protected:

		cv::Mat mWorldPos; ///< MapPoint����������ϵ�µ�����

		// �۲⵽��MapPoint��KF�͸�MapPoint��KF�е�����
		std::map<KeyFrame*, size_t> mObservations;

		// ��MapPointƽ���۲ⷽ��
		// �����жϵ��Ƿ��ڿ��ӷ�Χ��
		cv::Mat mNormalVector;

		// ÿ��3D��Ҳ��һ�������ӣ��������3D����Թ۲�����ά�����㣬����ѡ��һ�����д����Ե�
		 //ͨ�� ComputeDistinctiveDescriptors() �õ������д�����������,�������������ӵ�ƽ��������С
		cv::Mat mDescriptor;

		// ͨ�������MapPoint�Ĳο��ؼ�֡���Ǵ�����MapPoint���Ǹ��ؼ�֡
		KeyFrame* mpRefKF;

		/// Tracking counters
		int mnVisible;
		int mnFound;

		/// Bad flag (we do not currently erase MapPoint from memory)
		bool mbBad;
		//? �滻����ͼ��ĵ�? 
		MapPoint* mpReplaced;

		/// Scale invariance distances
		//? 
		float mfMinDistance;
		float mfMaxDistance;

		///�����ĵ�ͼ
		Map* mpMap;

		///�Ե�ǰ��ͼ��λ�˽��в�����ʱ��Ļ�����
		std::mutex mMutexPos;
		///�Ե�ǰ��ͼ���������Ϣ���в�����ʱ��Ļ�����
		std::mutex mMutexFeatures;
	};

}