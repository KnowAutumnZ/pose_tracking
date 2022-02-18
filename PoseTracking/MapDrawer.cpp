#include "MapDrawer.h"

#include <mutex>

namespace PoseTracking
{
	//构造函数
	MapDrawer::MapDrawer(Map* pMap, const std::string &strSettingPath) :mpMap(pMap)
	{
		std::string TrackingCFG = strSettingPath + "TrackingCFG.ini";

		rr::RrConfig config;
		config.ReadConfig(TrackingCFG);

		//从配置文件中读取设置的
		mKeyFrameSize = config.ReadFloat("PoseTracking", "KeyFrameSize", 0.05);
		mKeyFrameLineWidth = config.ReadFloat("PoseTracking", "KeyFrameLineWidth", 1.0);
		mGraphLineWidth = config.ReadFloat("PoseTracking", "GraphLineWidth", 0.9);
		mPointSize = config.ReadFloat("PoseTracking", "PointSize", 2.0);

		mCameraSize = config.ReadFloat("PoseTracking", "CameraSize", 0.08);
		mCameraLineWidth = config.ReadFloat("PoseTracking", "CameraLineWidth", 3.0);
	}

	void MapDrawer::DrawMapPoints()
	{
		//取出所有的地图点
		const std::vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
		//取出mvpReferenceMapPoints，也即局部地图d点
		const std::vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

		//将vpRefMPs从vector容器类型转化为set容器类型，便于使用set::count快速统计 - 我觉得称之为"重新构造"可能更加合适一些
		//补充, set::count用于返回集合中为某个值的元素的个数
		std::set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

		if (vpMPs.empty())
			return;

		// for AllMapPoints
		//显示所有的地图点（不包括局部地图点），大小为2个像素，黑色
		glPointSize(mPointSize);
		glBegin(GL_POINTS);
		glColor3f(0.0, 0.0, 0.0);         //黑色

		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			// 不包括ReferenceMapPoints（局部地图点）
			if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
				continue;
			cv::Mat pos = vpMPs[i]->GetWorldPos();
			glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
		}
		glEnd();

		// for ReferenceMapPoints
		//显示局部地图点，大小为2个像素，红色
		glPointSize(mPointSize);
		glBegin(GL_POINTS);
		glColor3f(1.0, 0.0, 0.0);

		for (std::set<MapPoint*>::iterator sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
		{
			if ((*sit)->isBad())
				continue;
			cv::Mat pos = (*sit)->GetWorldPos();
			glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));

		}
		glEnd();
	}

	//关于gl相关的函数，可直接google, 并加上msdn关键词
	void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
	{
		//历史关键帧图标：宽度占总宽度比例为0.05
		const float &w = mKeyFrameSize;
		const float h = w * 0.75;
		const float z = w * 0.6;

		// step 1：取出所有的关键帧
		const std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

		// step 2：显示所有关键帧图标
		//通过显示界面选择是否显示历史关键帧图标
		if (bDrawKF)
		{
			for (size_t i = 0; i < vpKFs.size(); i++)
			{
				KeyFrame* pKF = vpKFs[i];
				//NOTICE 转置, OpenGL中的矩阵为列优先存储
				cv::Mat Twc = pKF->GetPoseInverse().t();

				glPushMatrix();

				//（由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
				//因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
				//NOTICE 竟然还可以这样写,牛逼牛逼
				glMultMatrixf(Twc.ptr<GLfloat>(0));

				//设置绘制图形时线的宽度
				glLineWidth(mKeyFrameLineWidth);
				//设置当前颜色为蓝色(关键帧图标显示为蓝色)
				glColor3f(0.0f, 0.0f, 1.0f);
				//用线将下面的顶点两两相连
				glBegin(GL_LINES);
				glVertex3f(0, 0, 0);
				glVertex3f(w, h, z);
				glVertex3f(0, 0, 0);
				glVertex3f(w, -h, z);
				glVertex3f(0, 0, 0);
				glVertex3f(-w, -h, z);
				glVertex3f(0, 0, 0);
				glVertex3f(-w, h, z);

				glVertex3f(w, h, z);
				glVertex3f(w, -h, z);

				glVertex3f(-w, h, z);
				glVertex3f(-w, -h, z);

				glVertex3f(-w, h, z);
				glVertex3f(w, h, z);

				glVertex3f(-w, -h, z);
				glVertex3f(w, -h, z);
				glEnd();

				glPopMatrix();
			}
		}
	}

	//关于gl相关的函数，可直接google, 并加上msdn关键词
	void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
	{
		//相机模型大小：宽度占总宽度比例为0.08
		const float &w = mCameraSize;
		const float h = w * 0.75;
		const float z = w * 0.6;

		//百度搜索：glPushMatrix 百度百科
		glPushMatrix();

		//将4*4的矩阵Twc.m右乘一个当前矩阵
		//（由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
		//因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
		//一个是整型,一个是浮点数类型
#ifdef HAVE_GLES
		glMultMatrixf(Twc.m);
#else
		glMultMatrixd(Twc.m);
#endif

		//设置绘制图形时线的宽度
		glLineWidth(mCameraLineWidth);
		//设置当前颜色为绿色(相机图标显示为绿色)
		glColor3f(0.0f, 1.0f, 0.0f);
		//用线将下面的顶点两两相连
		glBegin(GL_LINES);
		glVertex3f(0, 0, 0);
		glVertex3f(w, h, z);
		glVertex3f(0, 0, 0);
		glVertex3f(w, -h, z);
		glVertex3f(0, 0, 0);
		glVertex3f(-w, -h, z);
		glVertex3f(0, 0, 0);
		glVertex3f(-w, h, z);

		glVertex3f(w, h, z);
		glVertex3f(w, -h, z);

		glVertex3f(-w, h, z);
		glVertex3f(-w, -h, z);

		glVertex3f(-w, h, z);
		glVertex3f(w, h, z);

		glVertex3f(-w, -h, z);
		glVertex3f(w, -h, z);
		glEnd();

		glPopMatrix();
	}

	//设置当前帧相机的位姿, 设置这个函数是因为要处理多线程的操作
	void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
	{
		std::unique_lock<std::mutex> lock(mMutexCamera);
		mCameraPose = Tcw.clone();
	}

	// 将相机位姿mCameraPose由Mat类型转化为OpenGlMatrix类型
	void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
	{
		if(!mCameraPose.empty())
		{
			cv::Mat Rwc(3,3,CV_32F);
			cv::Mat twc(3,1,CV_32F);
			{
				std::unique_lock<std::mutex> lock(mMutexCamera);
				Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
				twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
			}

			M.m[0] = Rwc.at<float>(0,0);
			M.m[1] = Rwc.at<float>(1,0);
			M.m[2] = Rwc.at<float>(2,0);
			M.m[3]  = 0.0;

			M.m[4] = Rwc.at<float>(0,1);
			M.m[5] = Rwc.at<float>(1,1);
			M.m[6] = Rwc.at<float>(2,1);
			M.m[7]  = 0.0;

			M.m[8] = Rwc.at<float>(0,2);
			M.m[9] = Rwc.at<float>(1,2);
			M.m[10] = Rwc.at<float>(2,2);
			M.m[11]  = 0.0;

			M.m[12] = twc.at<float>(0);
			M.m[13] = twc.at<float>(1);
			M.m[14] = twc.at<float>(2);
			M.m[15]  = 1.0;
		}
		else
			M.SetIdentity();
	}

} //namespace ORB_SLAM
