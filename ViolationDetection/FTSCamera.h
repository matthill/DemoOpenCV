#ifndef _FTS_LV_CAMERA_
#define _FTS_LV_CAMERA_

#ifdef _MSC_VER
#define snprintf _snprintf_s
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


struct FTSCamera {
	FTSCamera() {};

	std::string strVideoSrc;
	std::string strCameraId;
	std::string strAlgorithm;
	std::string strParamFile;
	std::string strOutputFolder;
	std::string strOutputMapFolder;

	void write(cv::FileStorage& fs) const //Write serialization for this class
	{
		fs << "{"
			<< "Video" << this->strVideoSrc
			<< "CameraID" << this->strCameraId
			<< "Algorithm" << this->strAlgorithm
			<< "ParamFile" << this->strParamFile
			<< "OutputFolder" << this->strOutputFolder
			<< "NetMapFolder" << this->strOutputMapFolder
			<< "}";
	}

	void read(const cv::FileNode& node)  //Read serialization for this class
	{
		this->strVideoSrc = (std::string)node["Video"];
		this->strCameraId = (std::string)node["CameraID"];
		this->strAlgorithm = (std::string) node["Algorithm"];
		this->strParamFile = (std::string)node["ParamFile"];
		this->strOutputFolder = (std::string)node["OutputFolder"];
		this->strOutputMapFolder = (std::string)node["NetMapFolder"];
	}
};

//These write and read functions must exist as per the inline functions in operations.hpp
static void write(cv::FileStorage& fs, const std::string&, const FTSCamera& cam) {
	cam.write(fs);
}
static void read(const cv::FileNode& node, FTSCamera& cam, const FTSCamera& default_value = FTSCamera()) {
	if (node.empty())
		cam = default_value;
	else
		cam.read(node);
}


static void readCameraFile(std::vector<FTSCamera>& listCams, const std::string& strParamFile) {
	cv::FileStorage fs(strParamFile, cv::FileStorage::READ);

	if (fs.isOpened()) {
		int count;
		char buff[100];

		fs["count"] >> count;
		for (int i = 0; i < count; i++) {
			FTSCamera cam;
			//_itoa_s(i, buff, 10);
			snprintf(buff, 100, "camera-%d", i);
			fs[buff] >> cam;
			listCams.push_back(cam);
		}
		fs.release();
	}
}

static void writeCameraFile(std::vector<FTSCamera> listCams, const std::string &strParamFile) {
	cv::FileStorage fs(strParamFile, cv::FileStorage::WRITE);
	std::string data("camera-");

	char buff[100];

	int count = (int) listCams.size();

	fs << "count" << count;
	for (int i = 0; i < count; i++) {

		snprintf(buff, 100, "camera-%d", i);

		fs << buff << listCams[i];
	}

	fs.release();
}

#endif