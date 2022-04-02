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

		//初始化局部建图线程并运行
		mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);

		//运行这个局部建图线程
		mptLocalMapping = new thread(&LocalMapping::Run,	//这个线程会调用的函数
			mpLocalMapper);									//这个调用函数的参数

		//如果指定了，程序的运行过程中需要运行可视化部分
		//新建viewer
		mpViewer = new Viewer(this, 			//又是这个
					mpFrameDrawer,				//帧绘制器
					mpMapDrawer,				//地图绘制器
					mpTracker,					//追踪器
					strSettingsFile);			//配置文件的访问路径
		//新建viewer线程
		mptViewer = new thread(&Viewer::Run, mpViewer);

		//设置线程间的指针
		mpTracker->SetLocalMapper(mpLocalMapper);
		mpLocalMapper->SetTracker(mpTracker);
	}

	//同理，输入为单目图像时的追踪器接口
	cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
	{
		//获取相机位姿的估计结果
		cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

		return Tcw;
	}

}
