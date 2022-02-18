#include <vector>

#include "system.h"

using namespace PoseTracking;

int main()
{
	std::vector<std::string> vimpath;
	cv::glob("./data/rgbd_dataset_freiburg2_desk/rgb/*.png", vimpath);

	std::string settingpath= "./cfg/";
	System Pose(settingpath, MONOCULAR);

	for (size_t i=0; i<vimpath.size(); i++)
	{
		cv::Mat im = cv::imread(vimpath[i]);
		Pose.TrackMonocular(im, 0.01);
	}

	return 0;
}