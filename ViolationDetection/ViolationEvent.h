#ifndef _FTS_LV_VIOLATIONEVENT_
#define _FTS_LV_VIOLATIONEVENT_

#ifdef _MSC_VER
#define snprintf _snprintf_s
#endif

#include <ctime>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.h"
#include "LogInit.h"

enum VehicleType {
	VEHICLE_UNDECIDED,
	VEHICLE_BIKE,
	VEHICLE_CAR
};

enum ViolationType {
	LANE_VIOLATION,
	REDSIGNAL_VIOLATION
};

static std::string getVehicleTypeString(VehicleType t) {
	switch (t) {
	case VEHICLE_UNDECIDED:
		return "";
	case VEHICLE_BIKE:
		return "XMA";
	case VEHICLE_CAR:
		return "OTO";
	}

	return "";
}

static std::string getViolationTypeString(ViolationType t) {
	switch (t) {
	case LANE_VIOLATION:
		return "SLD";
	case REDSIGNAL_VIOLATION:
		return "VDD";
	}

	return "";
}


struct FTSPlate {
	cv::Mat imgPlate;
	std::string strPlate;
	float fConfident;
};

struct ViolationEventXml {
	std::string strDeviceID;
	std::string strVTime;
	std::string strVType;
	std::string strVFullImage;
	std::string strPlateNumber;
	std::string strPImage;
	std::string strPConfident;
	std::string strVideo;
	std::string strVehicleType;
	std::string strSpeedDetected;
	cv::Rect rectBoundingbox;
	//Write serialization for this class
	void write(cv::FileStorage& fs) const {
		std::stringstream bbRect;
		bbRect << this->rectBoundingbox.x 
				<< "," << this->rectBoundingbox.y
				<< "," << this->rectBoundingbox.width
				<< "," << this->rectBoundingbox.height;
		fs << "{"
			<< "DeviceID" << this->strDeviceID
			<< "VTime" << this->strVTime
			<< "VType" << this->strVType
			<< "VFullImage" << this->strVFullImage
			<< "PlateNumber" << this->strPlateNumber
			<< "PImage" << this->strPImage
			<< "PConfident" << this->strPConfident
			<< "Video" << this->strVideo
			<< "VehicleType" << this->strVehicleType
			<< "SpeedDetected" << this->strSpeedDetected
			<< "Rectangle" << bbRect.str()
				/*<< "{" 
				<< "x" << this->rectBoundingbox.x 
				<< "y" << this->rectBoundingbox.y
				<< "width" << rectBoundingbox.width
				<< "height" << rectBoundingbox.height
				<< "}"*/
			<< "}";


	}

	void read(const cv::FileNode& node) {
		this->strDeviceID = (std::string)node["DeviceID"];
	}
};

static std::string writeXmlString(ViolationEventXml e) {
	cv::FileStorage fs(".xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
	fs << "ViolationEvent" << e;

	return fs.releaseAndGetString();
}

static std::string createDirectory(std::string strParent, std::string strSub, std::string strDeviceID, std::string strDate, std::string strType) {
	std::string sFolder = strParent;
	if (sFolder[sFolder.size() - 1] != '\\')
		sFolder.append("\\");

	if (strSub.compare("") != 0) {
		sFolder.append(strSub);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strDeviceID.compare("") != 0) {
		sFolder.append(strDeviceID);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strDate.compare("") != 0) {
		sFolder.append(strDate);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strType.compare("") != 0) {
		sFolder.append(strType);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	return sFolder;
}

static std::string replaceString(std::string subject, const std::string& search, const std::string& replace) {
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
	return subject;
}

struct ViolationEvent {
	static std::string const className() { return "ViolationEvent"; }

	ViolationEvent(): fSpeed(0.f) {
		time(&timer);		
		lEventTime = clock();
	};

	FTSPlate plate;
	std::time_t timer;
	clock_t lEventTime;		//03.07 trungnt1 add to distinct frame index

	cv::Mat imgOrg;

	std::string strDeviceID;
	ViolationType violationType;
	VehicleType vehicleType;

	float fSpeed;

	cv::Rect rectBoundingBox;
	std::string strViolationTime;

	std::string strOutputFolder;
	std::string strOutputMapFolder;
	std::string strVideoUrl;
	
	void process() {
		ViolationEventXml xml;
		char buff[100];
		cv::FileStorage fs;

		//handle xml
		xml.strDeviceID = this->strDeviceID;
		xml.strVTime = getDateTimeString(this->timer, "%Y-%m-%d %H:%M:%S");
		xml.strVType = getViolationTypeString(this->violationType);
		xml.strPlateNumber = this->plate.strPlate;
		xml.strVehicleType = getVehicleTypeString(vehicleType);

		snprintf(buff, 100, "%f", this->plate.fConfident);
		xml.strPConfident = buff;
		snprintf(buff, 100, "%f", this->fSpeed);
		xml.strSpeedDetected = buff;

		//get & create image folder
		std::string strTime = getDateTimeString(this->timer, "%Y%m%d_%H%M%S");
		std::string sImgFolder = createDirectory(this->strOutputFolder, "MEDIA", this->strDeviceID, getDateTimeString(this->timer, "%Y%m%d"), "IMG");
		//std::string sImgFilePath = getDateTimeString(this->timer, sImgFolder + this->strDeviceID + "_%Y%m%d_%H%M%S.jpg");
		std::string sImgFilePath = formatString("%s%s_%s_%d.jpg", sImgFolder.c_str(), this->strDeviceID.c_str(), strTime.c_str(), this->lEventTime);
		cv::imwrite(sImgFilePath, this->imgOrg);
		xml.strVFullImage = this->strOutputMapFolder.compare("") != 0 ? replaceString(sImgFilePath, this->strOutputFolder, this->strOutputMapFolder) : sImgFilePath;

		//get & create plate folder
		std::string sPlateFolder = createDirectory(this->strOutputFolder, "MEDIA", this->strDeviceID, getDateTimeString(this->timer, "%Y%m%d"), "PLATE");
		//std::string sPlateFilePath = getDateTimeString(this->timer, sPlateFolder + this->strDeviceID + "_%Y%m%d_%H%M%S.jpg");
		std::string sPlateFilePath = formatString("%s%s_%s_%d.jpg", sPlateFolder.c_str(), this->strDeviceID.c_str(), strTime.c_str(), this->lEventTime);
		cv::imwrite(sPlateFilePath, this->plate.imgPlate);
		xml.strPImage = this->strOutputMapFolder.compare("") != 0 ? replaceString(sPlateFilePath, this->strOutputFolder, this->strOutputMapFolder) : sPlateFilePath;

		//get & create video folder
		//std::string sVideoFolder = createDirectory (this->strOutputFolder, "MEDIA", this->strDeviceID, getDateTimeString (this->timer, "%Y%m%d"), "VIDEO");
		//xml.strVideo = this->strViolationTime + ";" + getDateTimeString(this->timer, sVideoFolder + this->strDeviceID + "_%Y%m%d_%H%M%S.mp4");
		xml.strVideo = this->strViolationTime + ";" + this->strVideoUrl;

		//create bounding box
		xml.rectBoundingbox = this->rectBoundingBox;

		//get & create xml folder
		std::string sXmlFolder = createDirectory(this->strOutputFolder, "", "", "", "Unprocessed");
		fs.open(getDateTimeString(this->timer, sXmlFolder + this->strDeviceID + "_%Y%m%d_%H%M%S.xml"), cv::FileStorage::WRITE);

		//write xml to output file
		fs << "ViolationEvent" << xml;
		fs.release();

		boost::log::sources::severity_channel_logger< severity_level > lg;
		BOOST_LOG_CHANNEL_SEV(lg, ViolationEvent::className(), LOG_INFO) << std::endl << writeXmlString(xml);
		std::cout << writeXmlString(xml);
	};
};


//These write and read functions must exist as per the inline functions in operations.hpp
static void write(cv::FileStorage& fs, const std::string& name, const ViolationEventXml& e) {
	e.write(fs);
}

//static void read(const cv::FileNode& node, ViolationEvent& event, const ViolationEvent& default_value = ViolationEvent()) {
//	if (node.empty())
//		event = default_value;
//	else
//		event.read(node);
//}




#endif