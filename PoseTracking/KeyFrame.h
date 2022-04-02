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
	class KeyFrameDatabase;
	class Frame;

	class KeyFrame
	{
	public:
		/**
		 * @brief ���캯��
		 * @param[in] F         ������ͨ֡�Ķ���
		 * @param[in] pMap      �����ĵ�ͼָ��
		 * @param[in] pKFDB     ʹ�õĴʴ�ģ�͵�ָ��
		 */
		KeyFrame(Frame* F, Map* pMap, KeyFrameDatabase* pKFDB);

		/**
		 * @brief ���õ�ǰ�ؼ�֡��λ��
		 * @param[in] Tcw λ��
		 */
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();                  //< ��ȡλ��
		cv::Mat GetPoseInverse();           //< ��ȡλ�˵���
		cv::Mat GetCameraCenter();          //< ��ȡ(��Ŀ)���������
		cv::Mat GetStereoCenter();          //< ��ȡ˫Ŀ���������,���ֻ���ڿ��ӻ���ʱ��Ż��õ�
		cv::Mat GetRotation();              //< ��ȡ��̬
		cv::Mat GetTranslation();           //< ��ȡλ��

		// ====================== Covisibility graph functions ============================
		/**
		 * @brief Ϊ�ؼ�֮֡���������
		 * @details ������mConnectedKeyFrameWeights
		 * @param pKF    �ؼ�֡
		 * @param weight Ȩ�أ��ùؼ�֡��pKF��ͬ�۲⵽��3d������
		 */
		void AddConnection(KeyFrame* pKF, const int &weight);

		/**
		 * @brief ɾ����ǰ�ؼ�֡��ָ���ؼ�֮֡��Ĺ��ӹ�ϵ
		 * @param[in] pKF Ҫɾ���Ĺ��ӹ�ϵ
		 */
		void EraseConnection(KeyFrame* pKF);

		/** @brief ����ͼ������  */
		void UpdateConnections();

		/**
		 * @brief ����Ȩ�ض����ӵĹؼ�֡��������
		 * @detials ���º�ı����洢��mvpOrderedConnectedKeyFrames��mvOrderedWeights��
		 */
		void UpdateBestCovisibles();

		/**
		 * @brief �õ���ùؼ�֡���ӵĹؼ�֡(û�������)
		 * @return ���ӵĹؼ�֡
		 */
		std::set<KeyFrame*> GetConnectedKeyFrames();

		/**
		 * @brief �õ���ùؼ�֡���ӵĹؼ�֡(�Ѱ�Ȩֵ����)
		 * @return ���ӵĹؼ�֡
		 */
		std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();

		/**
		 * @brief �õ���ùؼ�֡���ӵ�ǰN���ؼ�֡(�Ѱ�Ȩֵ����)
		 * NOTICE ������ӵĹؼ�֡����N���򷵻��������ӵĹؼ�֡,����˵���صĹؼ�֡����Ŀ��ʵ��һ����N��
		 * @param N ǰN��
		 * @return ���ӵĹؼ�֡
		 */
		std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);

		/**
		 * @brief �õ���ùؼ�֡���ӵ�Ȩ�ش��ڵ���w�Ĺؼ�֡
		 * @param w Ȩ��
		 * @return ���ӵĹؼ�֡
		 */
		std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);

		/**
		 * @brief �õ��ùؼ�֡��pKF��Ȩ��
		 * @param  pKF �ؼ�֡
		 * @return     Ȩ��
		 */
		int GetWeight(KeyFrame* pKF);

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
		 * @brief ����������ԭ��,���µ�ǰ�ؼ�֡�۲⵽��ĳ����ͼ�㱻ɾ��(bad==true)��,������"֪ͨ"��ǰ�ؼ�֡�����ͼ���Ѿ���ɾ����
		 * @param[in] pMP ��ɾ���ĵ�ͼ��ָ��
		 */
		void EraseMapPointMatch(MapPoint* pMP);

		/**
		 * @brief ��ͼ����滻
		 * @param[in] idx Ҫ�滻���ĵ�ͼ�������
		 * @param[in] pMP �µ�ͼ���ָ��
		 */
		void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);

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

		/**
		 * @brief �ؼ�֡�У����ڵ���minObs��MapPoints������
		 * @details minObs����һ����ֵ������minObs�ͱ�ʾ��MapPoint��һ����������MapPoint \n
		 * һ����������MapPoint�ᱻ���KeyFrame�۲⵽.
		 * @param  minObs ��С�۲�
		 */
		int TrackedMapPoints(const int &minObs);

		/**
		 * @brief ��ȡ��ȡ��ǰ�ؼ�֡�ľ����ĳ����ͼ��
		 * @param[in] idx id
		 * @return MapPoint* ��ͼ����
		 */
		MapPoint* GetMapPoint(const size_t &idx);

		// ========================= Spanning tree functions =======================
		/**
		 * @brief ����ӹؼ�֡�������ӹؼ�֡��������ӹ�ϵ�Ĺؼ�֡���ǵ�ǰ�ؼ�֡��
		 * @param[in] pKF �ӹؼ�֡���
		 */
		void AddChild(KeyFrame* pKF);

		/**
		 * @brief ��ȡ��ȡ��ǰ�ؼ�֡���ӹؼ�֡
		 * @return std::set<KeyFrame*>  �ӹؼ�֡����
		 */
		std::set<KeyFrame*> GetChilds();
		/**
		 * @brief ��ȡ��ǰ�ؼ�֡�ĸ��ؼ�֡
		 * @return KeyFrame* ���ؼ�֡���
		 */
		KeyFrame* GetParent();

		/**
		 * @brief �ı䵱ǰ�ؼ�֡�ĸ��ؼ�֡
		 * @param[in] pKF ���ؼ�֡���
		 */
		void ChangeParent(KeyFrame* pKF);

		/**
		 * @brief ɾ��ĳ���ӹؼ�֡
		 * @param[in] pKF �ӹؼ�֡���
		 */
		void EraseChild(KeyFrame* pKF);

		/**
		 * @brief �ж�ĳ���ؼ�֡�Ƿ��ǵ�ǰ�ؼ�֡���ӹؼ�֡
		 * @param[in] pKF �ؼ�֡���
		 * @return true
		 * @return false
		 */
		bool hasChild(KeyFrame* pKF);

		// Image
		/**
		 * @brief �ж�ĳ�����Ƿ��ڵ�ǰ�ؼ�֡��ͼ����
		 * @param[in] x �������
		 * @param[in] y �������
		 * @return true
		 * @return false
		 */
		bool IsInImage(const float &x, const float &y) const;

		/** @brief ���õ�ǰ�ؼ�֡��Ҫ���Ż��Ĺ����б�ɾ��  */
		void SetNotErase();

		/** @brief ׼��ɾ����ǰ������ؼ�֡,��ʾ�����лػ�������;�ɻػ�����̵߳��� */
		void SetErase();

		/** @brief ������ִ��ɾ���ؼ�֡�Ĳ��� */
		void SetBadFlag();

		// KeyPoint functions
		/**
		 * @brief ��ȡĳ��������������е�������id
		 * @param[in] x ����������
		 * @param[in] y ����������
		 * @param[in] r �����С(�뾶)
		 * @return std::vector<size_t> ������������ҵ��������������ļ���
		 */
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
	public:
		// nNextID���ָ�ΪnLastID�����ʣ���ʾ��һ��KeyFrame��ID��
		static long unsigned int nNextId;
		// ��nNextID�Ļ����ϼ�1�͵õ���mnID��Ϊ��ǰKeyFrame��ID��
		long unsigned int mnId;
		// ÿ��KeyFrame��������������һ��Frame��KeyFrame��ʼ����ʱ����ҪFrame��
		// mnFrameId��¼�˸�KeyFrame�����ĸ�Frame��ʼ����
		const long unsigned int mnFrameId;

		// local mapping�м�¼��ǰ����Ĺؼ�֡��mnId����ʾ��ǰ�ֲ�BA�Ĺؼ�֡id��mnBALocalForKF ��map point.h����Ҳ��ͬ���ı�����
		long unsigned int mnBALocalForKF;
		// local mapping�м�¼��ǰ����Ĺؼ�֡��mnId, ֻ���ṩԼ����Ϣ����ȴ����ȥ�Ż�����ؼ�֡
		long unsigned int mnBAFixedForKF;

		//��ʾ���Ѿ���ĳ֡�ľֲ��ؼ�֡�ˣ����Է�ֹ�ظ���Ӿֲ��ؼ�֡
		long unsigned int mnTrackReferenceForFrame;      // ��¼��
		long unsigned int mnFuseTargetForKF;			 //< ����ھֲ���ͼ�߳���,���ĸ��ؼ�֡�����ںϵĲ���

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

		// Covisibility Graph
		// ��ùؼ�֡���ӣ�����15�����ӵ�ͼ�㣩�Ĺؼ�֡��Ȩ��
		std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
		// ���ӹؼ�֡��Ȩ�شӴ�С�����Ĺؼ�֡          
		std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
		// ���ӹؼ�֡�дӴ�С������Ȩ�أ��������Ӧ
		std::vector<int> mvOrderedWeights;

		// ===================== Spanning Tree and Loop Edges ========================
		// std::set�Ǽ��ϣ����vector�����в������������Ĳ���ʱ���Զ�����
		bool mbFirstConnection;                     // �Ƿ��ǵ�һ��������
		KeyFrame* mpParent;                         // ��ǰ�ؼ�֡�ĸ��ؼ�֡ �����ӳ̶���ߵģ�
		std::set<KeyFrame*> mspChildrens;           // �洢��ǰ�ؼ�֡���ӹؼ�֡
		std::set<KeyFrame*> mspLoopEdges;           // �͵�ǰ�ؼ�֡�γɻػ���ϵ�Ĺؼ�֡

		/**
		* @name ͼ���������Ϣ
		* @{
		*/
		// Scale pyramid info.
		int mnScaleLevels;                  //<ͼ��������Ĳ���
		float mfScaleFactor;                //<ͼ��������ĳ߶�����
		float mfLogScaleFactor;             //<ͼ��������ĳ߶����ӵĶ���ֵ�����ڷ���������߶�Ԥ���ͼ��ĳ߶�

		std::vector<float> mvScaleFactors;		//<ͼ�������ÿһ�����������
		std::vector<float> mvLevelSigma2;		//@todo Ŀǰ��frame.c��û���õ����޷��¶���
		std::vector<float> mvInvLevelSigma2;	//<��������ĵ���

		const int mnMinX;
		const int mnMinY;
		const int mnMaxX;
		const int mnMaxY;

	private:
		// SE3 Pose and camera center
		cv::Mat Tcw;    // ��ǰ�����λ�ˣ���������ϵ���������ϵ
		cv::Mat Twc;    // ��ǰ���λ�˵���
		cv::Mat Ow;     // �������(��Ŀ)����������ϵ�µ�����,�������ͨ֡�еĶ�����һ����

		cv::Mat Cw;     //< Stereo middel point. Only for visualization

		cv::Mat mTcp;   // Pose relative to parent (this is computed when bad flag is activated)

		// MapPoints associated to keypoints
		std::vector<MapPoint*> mvpMapPoints;

		float mHalfBaseline = 10; //< ����˫Ŀ�����˵,˫Ŀ������߳��ȵ�һ��. Only for visualization

		// �ڶ�λ�˽��в���ʱ��صĻ�����
		std::mutex mMutexPose;
		// �ڲ�����ǰ�ؼ�֡�������ؼ�֡�Ĺ�ʽ��ϵ��ʱ��ʹ�õ��Ļ�����
		std::mutex mMutexConnections;
		// �ڲ������������йصı�����ʱ��Ļ�����
		std::mutex mMutexFeatures;

		Map* mpMap;

		// Bad flags
		bool mbNotErase;            //< ��ǰ�ؼ�֡�Ѿ��������Ĺؼ�֡�γ��˻ػ���ϵ������ڸ����Ż��Ĺ����в�Ӧ�ñ�ɾ��
		bool mbToBeErased;          //<
		bool mbBad;                 //< 
	};
}