#ifndef _FTS_LV_VIDEO_ALG_
#define _FTS_LV_VIDEO_ALG_

#include <queue>
#include <thread>
#include <mutex>

#include "Ctracker.h"
#include "FTSCamera.h"
#include "FTSAlgorithm.h"
#include "FTSANPR.h"

//setup task queue
static const unsigned int MAX_WORKER = std::max(std::thread::hardware_concurrency(), 4U);
static const unsigned int MAX_QUEUE = 100;
static const unsigned int MAX_RETRY = 10;
static const unsigned int RETRY_TIME = 1;
static const std::string WORKER_NAME = "Worker";

extern int stop_count;


//extern std::vector<std::thread> workers;
//extern std::mutex mutexTaskQueue;
extern std::queue<ViolationEvent> taskQueue;

static void doTask(FTSANPR& anpr) {
	boost::log::sources::severity_channel_logger< severity_level, std::string > lg;
	for (; stop_count < anpr.camera_count;) {
		if (!taskQueue.empty()) {
			//if (workers.size() < MAX_WORKER) {
			//	mutexTaskQueue.lock();
			//	if (!taskQueue.empty()) {

			//		workers.push_back(std::thread(anpr, taskQueue.front()));

					anpr(taskQueue.front());
					taskQueue.pop();
					BOOST_LOG_CHANNEL_SEV(lg, WORKER_NAME, LOG_TRACE) << "Start new worker! Number of task remaining: " << taskQueue.size();
					//std::cout << "Start new worker! Number of task remaining: " << taskQueue.size() << std::endl << std::endl;
				//}
			//	mutexTaskQueue.unlock();
			//} else {
			//	for (size_t i = 0; i < workers.size(); i++) {
			//		if (workers[i].joinable()) {
			//			workers[i].join();
			//			workers.erase(workers.begin() + i);
			//			BOOST_LOG_CHANNEL_SEV(lg, WORKER_NAME, LOG_TRACE) << "Done";
			//			//std::cout << "Work done!" << std::endl << std::endl;
			//		}
			//	}
			//}
		} else {
			//std::cout << "Free task. Waiting for 10s." << std::endl << std::endl;
			BOOST_LOG_CHANNEL_SEV(lg, WORKER_NAME, LOG_TRACE) << "Free task. Waiting for " << RETRY_TIME << " seconds.";
			std::this_thread::sleep_for(std::chrono::seconds(RETRY_TIME));
		}
	}
}

class FTSVideoAlgorithm: public FTSAlgorithm {
public:
	static std::string const className() { return "FTSVideoAlgorithm"; }
	virtual void operator() (FTSCamera camInfo) {};
	virtual cv::AlgorithmInfo* info() const;
	FTSVideoAlgorithm() {};
	~FTSVideoAlgorithm() {};

	std::string strAnprParamFile;

protected:
	std::vector<cv::Rect> trackVehicle(CTracker& tracker, std::vector<cv::Rect> vehicleBbs, std::vector<cv::Point2d> vehicleCenters) {
		std::vector<cv::Rect> trackedVechileBbs;
		if (vehicleCenters.empty()){
			tracker.updateSkipedSkippedFrames();
		}
		else{
			tracker.Update(vehicleCenters);
		}
		for (size_t i = 0; i < tracker.tracks.size(); i++) {
			//track for only 2 frame
			if (tracker.tracks[i]->trace.size() == 2 && !tracker.tracks[i]->isCaught) {
				// Compare it to predefined direction
				size_t iTraceLength = tracker.tracks[i]->trace.size();

				int ind = tracker.assignment[i];
				if (ind > -1) {
					tracker.tracks[i]->isCaught = true;
					trackedVechileBbs.push_back(vehicleBbs[ind]);
				}
			}
		}

		return trackedVechileBbs;
	}
	void handleViolation(const cv::Mat& img, const cv::Rect& rectRoi, float fScaleRatio, std::vector<cv::Rect>& vehicleViolationBbs, const string& vehicleViolationTime, FTSCamera camInfo, VehicleType type, ViolationType vType){
		std::vector<cv::Mat> fakeVehicleViolationPlates;
		std::vector<cv::Rect> fakePlateViolationBbs;
		handleViolation(img, rectRoi, fScaleRatio, vehicleViolationBbs, fakePlateViolationBbs, fakeVehicleViolationPlates, vehicleViolationTime, camInfo, type, vType);
	}

	void handleViolation(const cv::Mat& img, const cv::Rect& rectRoi, float fScaleRatio, std::vector<cv::Rect>& vehicleViolationBbs, std::vector<cv::Rect>& plateViolationBbs, std::vector<cv::Mat>& vehicleViolationPlates, const string& vehicleViolationTime, FTSCamera camInfo, VehicleType type, ViolationType vType) {
		if (vehicleViolationPlates.size() > 0 && vehicleViolationPlates.size() != vehicleViolationBbs.size()){
			return;
		}
		//if (bDebug)
		//	anpr.setOutDebugFolder(camInfo.strOutputFolder, camInfo.strCameraId);
		bool isHaveRecognizePlate =false;
		if (plateViolationBbs.size() > 0 && plateViolationBbs.size() == vehicleViolationBbs.size())
		{
			isHaveRecognizePlate = true;
		}
		bool isHavePlateToView = false;
		if (plateViolationBbs.size() > 0 && plateViolationBbs.size() == vehicleViolationBbs.size())
		{
			isHavePlateToView = true;
		}
		cv::Rect wholeImgRect(cv::Point(0, 0), cv::Point(img.cols, img.rows));
		for (int i = 0; i < vehicleViolationBbs.size(); i++) {
			
			if (vehicleViolationBbs[i].width == 0 || vehicleViolationBbs[i].height == 0 || !wholeImgRect.contains(vehicleViolationBbs[i].tl()) || !wholeImgRect.contains(vehicleViolationBbs[i].br()))
			{
				//std::cout << "Invalid bb " << vehicleViolationBbs[i] << std::endl;
				continue;
			}
			BOOST_LOG_CHANNEL_SEV(lg, FTSVideoAlgorithm::className(), LOG_INFO) << "Violation Type: " << getViolationTypeString(vType) << " Vehicle Type : " << getVehicleTypeString(type);

			cv::Rect rectOrgin;
			rectOrgin.x = int((vehicleViolationBbs[i].x + rectRoi.x) / fScaleRatio);
			rectOrgin.y = int((vehicleViolationBbs[i].y + rectRoi.y) / fScaleRatio);
			rectOrgin.height = int(vehicleViolationBbs[i].height / fScaleRatio);
			rectOrgin.width = int(vehicleViolationBbs[i].width / fScaleRatio);

			//2014.06.22 Trung add check rectBoundingBox
			if(rectOrgin.area() == 0)
			{
				BOOST_LOG_CHANNEL_SEV(lg, FTSVideoAlgorithm::className(), LOG_INFO) << "ERROR! OBJECT BOUNDING BOX IS NULL!";
				continue;
			}

			//Create violation event
			ViolationEvent e;
			if (isHaveRecognizePlate){
				e.imgPlate = vehicleViolationPlates[i].clone();
			}
			if (isHavePlateToView){
				cv::Rect rectOrginPlate;
				rectOrginPlate.x = int ( (plateViolationBbs[i].x +rectRoi.x) / fScaleRatio);
				rectOrginPlate.y = int ( (plateViolationBbs[i].y +rectRoi.y) / fScaleRatio);
				rectOrginPlate.width = int (plateViolationBbs[i].width / fScaleRatio);
				rectOrginPlate.x = int (plateViolationBbs[i].height / fScaleRatio);
				if (rectOrginPlate.area() == 0 && !wholeImgRect.contains(rectOrginPlate.tl()) || !wholeImgRect.contains(rectOrginPlate.br()))
					continue;
				e.rectViewPlateBB = rectOrginPlate;
			}
			e.imgOrg = img.clone();
			e.strDeviceID = camInfo.strCameraId;
			e.strOutputFolder = camInfo.strOutputFolder;
			e.strOutputMapFolder = camInfo.strOutputMapFolder;
			e.vehicleType = type;
			e.violationType = vType;
			e.rectBoundingBox = rectOrgin;
			e.strViolationTime = vehicleViolationTime;
			e.strVideoUrl = camInfo.strVideoSrc;

			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_TRACE) << "Push new task to event queue";
			taskQueue.push(e);
		}
	}

	void reconnectVideoSource(FTSCamera camInfo) {
		BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << "Cannot connect to video Source";

		if (this->iRetryCount < MAX_RETRY) {
			iRetryCount++;

			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << camInfo.strVideoSrc << " Retry count " << this->iRetryCount;
			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << "Sleep for " << (RETRY_TIME << this->iRetryCount) << " seconds.";
			std::this_thread::sleep_for(std::chrono::seconds(RETRY_TIME << iRetryCount));
			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << camInfo.strVideoSrc << " Retry count " << this->iRetryCount;
			this->operator()(camInfo);
		} else {
			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_ERROR) << "Exceed maxium retry - Stop process";
			return;
		}

	}

	int iRetryCount;
};

#endif