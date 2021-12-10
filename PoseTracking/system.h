#pragma once

#include <string>
#include <iostream>

#include "Tracking.h"

namespace PoseTracking
{
	class System
	{
	public:
		System(const std::string &strSettingsFile,   //指定配置文件的路径
			   const eSensor sensor);                //指定所使用的传感器类型
		
	private:
		// 传感器类型
		eSensor mSensor;

		// 追踪器，除了进行运动追踪外还要负责创建关键帧、创建新地图点和进行重定位的工作。详细信息还得看相关文件
		Tracking* mpTracker;
	};
}