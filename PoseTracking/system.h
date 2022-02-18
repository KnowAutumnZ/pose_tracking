#pragma once

#include <string>
#include <iostream>

#include "Tracking.h"

namespace PoseTracking
{
	//���ö���������� ��ʾ��ϵͳ��ʹ�õĴ���������
	enum eSensor {
		MONOCULAR = 0,
		STEREO = 1,
		RGBD = 2
	};

	//Ҫ�õ����������ǰ������
	class Viewer;
	class FrameDrawer;
	class Map;
	class Tracking;

	class System
	{
	public:
		System(const std::string &strSettingsFile,   //ָ�������ļ���·��
			   const eSensor sensor);                //ָ����ʹ�õĴ���������

		//ͼ�� ʱ���
		cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);
		
	private:
		// ����������
		eSensor mSensor;

		//ָ���ͼ�����ݿ⣩��ָ��
		Map* mpMap;

		// �鿴�������ӻ� ����
		Viewer* mpViewer;

		//֡������
		FrameDrawer* mpFrameDrawer;
		//��ͼ������
		MapDrawer* mpMapDrawer;

		// ׷���������˽����˶�׷���⻹Ҫ���𴴽��ؼ�֡�������µ�ͼ��ͽ����ض�λ�Ĺ�������ϸ��Ϣ���ÿ�����ļ�
		Tracking* mpTracker;

		std::thread* mptViewer;
	};
}