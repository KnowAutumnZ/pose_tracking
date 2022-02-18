#include "system.h"

namespace PoseTracking
{
	System::System(const std::string &strSettingsFile, const eSensor sensor):mSensor(sensor)
	{
		// �����ǰ����������
		std::cout << "Input sensor was set to: ";

		if (mSensor == MONOCULAR)
			std::cout << "Monocular" << std::endl;
		else if (mSensor == STEREO)
			std::cout << "Stereo" << std::endl;
		else if (mSensor == RGBD)
			std::cout << "RGB-D" << std::endl;

		//Create the Map
		mpMap = new Map();

		//�����֡�������͵�ͼ���������ᱻ���ӻ���Viewer��ʹ��
		mpFrameDrawer = new FrameDrawer(mpMap);
		mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

		//�ڱ��������г�ʼ��׷���߳�
		mpTracker = new Tracking(strSettingsFile, mpFrameDrawer, mpMap, mpMapDrawer, mSensor);

		//���ָ���ˣ���������й�������Ҫ���п��ӻ�����
		//�½�viewer
		mpViewer = new Viewer(this, 			//�������
					mpFrameDrawer,				//֡������
					mpMapDrawer,				//��ͼ������
					mpTracker,					//׷����
					strSettingsFile);			//�����ļ��ķ���·��
		//�½�viewer�߳�
		mptViewer = new thread(&Viewer::Run, mpViewer);
	}

	//ͬ������Ϊ��Ŀͼ��ʱ��׷�����ӿ�
	cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
	{
		//��ȡ���λ�˵Ĺ��ƽ��
		cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

		return Tcw;
	}

}
