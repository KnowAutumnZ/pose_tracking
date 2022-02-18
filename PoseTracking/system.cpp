#include "system.h"

namespace PoseTracking
{
	System::System(const std::string &strSettingsFile, const eSensor sensor):mSensor(sensor)
	{
		// 输出当前传感器类型
		std::cout << "Input sensor was set to: ";

		if (mSensor == MONOCULAR)
			std::cout << "Monocular" << std::endl;
		else if (mSensor == STEREO)
			std::cout << "Stereo" << std::endl;
		else if (mSensor == RGBD)
			std::cout << "RGB-D" << std::endl;

		//Create the Map
		mpMap = new Map();

		//这里的帧绘制器和地图绘制器将会被可视化的Viewer所使用
		mpFrameDrawer = new FrameDrawer(mpMap);
		mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

		//在本主进程中初始化追踪线程
		mpTracker = new Tracking(strSettingsFile, mpFrameDrawer, mpMap, mpMapDrawer, mSensor);

		//如果指定了，程序的运行过程中需要运行可视化部分
		//新建viewer
		mpViewer = new Viewer(this, 			//又是这个
					mpFrameDrawer,				//帧绘制器
					mpMapDrawer,				//地图绘制器
					mpTracker,					//追踪器
					strSettingsFile);			//配置文件的访问路径
		//新建viewer线程
		mptViewer = new thread(&Viewer::Run, mpViewer);
	}

	//同理，输入为单目图像时的追踪器接口
	cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
	{
		//获取相机位姿的估计结果
		cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

		return Tcw;
	}

}
