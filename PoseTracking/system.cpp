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

		mpTracker = new Tracking(strSettingsFile, mSensor);

	}



}
