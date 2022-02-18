#pragma once

#include <string>
#include <iostream>

#include "Tracking.h"

namespace PoseTracking
{
	//这个枚举类型用于 表示本系统所使用的传感器类型
	enum eSensor {
		MONOCULAR = 0,
		STEREO = 1,
		RGBD = 2
	};

	//要用到的其他类的前视声明
	class Viewer;
	class FrameDrawer;
	class Map;
	class Tracking;

	class System
	{
	public:
		System(const std::string &strSettingsFile,   //指定配置文件的路径
			   const eSensor sensor);                //指定所使用的传感器类型

		//图像 时间戳
		cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);
		
	private:
		// 传感器类型
		eSensor mSensor;

		//指向地图（数据库）的指针
		Map* mpMap;

		// 查看器，可视化 界面
		Viewer* mpViewer;

		//帧绘制器
		FrameDrawer* mpFrameDrawer;
		//地图绘制器
		MapDrawer* mpMapDrawer;

		// 追踪器，除了进行运动追踪外还要负责创建关键帧、创建新地图点和进行重定位的工作。详细信息还得看相关文件
		Tracking* mpTracker;

		std::thread* mptViewer;
	};
}