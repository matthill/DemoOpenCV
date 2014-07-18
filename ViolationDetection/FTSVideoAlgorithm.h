#ifndef _FTS_LV_VIDEO_ALG_
#define _FTS_LV_VIDEO_ALG_

#include <queue>
#include <vector>
#include <thread>
#include <mutex>

#include "Ctracker.h"
#include "FTSCamera.h"
#include "FTSAlgorithm.h"
#include "FTSANPR.h"
#include "FTSTaskQueue.h"
#include "util.h"


class FTSVideoAlgorithm: public FTSAlgorithm {
public:
	static std::string const className() { return "FTSVideoAlgorithm"; }

	virtual void operator() (FTSCamera camInfo, std::queue<ViolationEvent>& taskQueue) {};
	virtual cv::AlgorithmInfo* info() const;
	FTSVideoAlgorithm() {};
	~FTSVideoAlgorithm() {};

	std::string strAnprParamFile;
	FTSANPR anpr;

	boost::mutex engineMutex;



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

	void handleViolation(const cv::Mat& img, const cv::Rect& rectRoi, float fScaleRatio, std::vector<cv::Rect> vehicleViolationBbs, const string& vehicleViolationTime, FTSCamera camInfo, VehicleType type, ViolationType vType, std::queue<ViolationEvent>& taskQueue) {
		if (bDebug)
			anpr.setOutDebugFolder(camInfo.strOutputFolder, camInfo.strCameraId);

		for (int i = 0; i < vehicleViolationBbs.size(); i++) {
			cv::Rect wholeImgRect(cv::Point(0,0), cv::Point(img.cols, img.rows));
			if (vehicleViolationBbs[i].width == 0 || vehicleViolationBbs[i].height == 0 || !wholeImgRect.contains(vehicleViolationBbs[i].tl()) || !wholeImgRect.contains(vehicleViolationBbs[i].br()))
			{
				std::cout << "Invalid bb " << vehicleViolationBbs[i] << std::endl;
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
			e.imgOrg = img.clone();
			e.strDeviceID = camInfo.strCameraId;
			e.strOutputFolder = camInfo.strOutputFolder;
			e.strOutputMapFolder = camInfo.strOutputMapFolder;
			e.vehicleType = type;
			e.violationType = vType;
			e.rectBoundingBox = rectOrgin;
			e.strViolationTime = vehicleViolationTime;
			e.strVideoUrl = camInfo.strVideoSrc;

			if (taskQueue.size() > 100) {
				std::cout << "Task queue is full. Violation will be dropped!!" << std::endl << std::endl;
			} else {
				std::cout << "New Violation. Number of task: " << taskQueue.size() << std::endl << std::endl;
				taskQueue.push(e);
			}
			//FTSTaskQueue queue;
			//queue.pushTask(e);
		}
	}

	void reconnectVideoSource(FTSCamera camInfo, std::queue<ViolationEvent>& taskQueue) {
		BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << "Cannot connect to video Source";

		if (this->iRetryCount < 5) {
			iRetryCount++;

			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << camInfo.strVideoSrc << " Retry count " << this->iRetryCount;
			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << "Sleep for " << (5 << this->iRetryCount) << " seconds";
#if _MSC_VER > 1600
			std::this_thread::sleep_for(std::chrono::seconds(5 << iRetryCount));
#else
			boost::this_thread::sleep_for(boost::chrono::seconds(5 << iRetryCount));
#endif
			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_WARN) << camInfo.strVideoSrc << " Retry count " << this->iRetryCount;
			this->operator()(camInfo, taskQueue);
		} else {
			BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_ERROR) << "Exceed maxium retry - Stop process";
			return;
		}

	}

	int iRetryCount;
};

#endif