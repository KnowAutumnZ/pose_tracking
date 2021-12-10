#pragma once

#include <string>
#include <iostream>

#include "Tracking.h"

namespace PoseTracking
{
	class System
	{
	public:
		System(const std::string &strSettingsFile,   //ָ�������ļ���·��
			   const eSensor sensor);                //ָ����ʹ�õĴ���������
		
	private:
		// ����������
		eSensor mSensor;

		// ׷���������˽����˶�׷���⻹Ҫ���𴴽��ؼ�֡�������µ�ͼ��ͽ����ض�λ�Ĺ�������ϸ��Ϣ���ÿ�����ļ�
		Tracking* mpTracker;
	};
}