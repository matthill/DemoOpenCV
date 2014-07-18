#include "CVUtil.h"
#include "FTSLaneViolationRegion.h"
#include "OpticalFlowTracker.h"

void FTSLaneViolationRegion::operator()(FTSCamera camInfo) {
	cv::Mat img, imgFrame, imgBack, imgFore, imgRoiWhole, imgOverlay;
	cv::Mat imgRoiCar, imgRoiBike, bikeLaneMap, carLaneMap;
	std::vector< std::vector<cv::Point> > vContours;
	cv::Rect rectRange;


	//open video stream
	cv::VideoCapture video(camInfo.strVideoSrc);
	if (!video.isOpened()) {
		reconnectVideoSource(camInfo);
		return;
	}

	BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_INFO) << "Start Camera process";
	this->iRetryCount = 0;


	//get video info
	int iFrameWidth = (int) (video.get(CV_CAP_PROP_FRAME_WIDTH) * this->fScaleRatio);
	int iFrameHeight = (int) (video.get(CV_CAP_PROP_FRAME_HEIGHT) * this->fScaleRatio);
	float fFrameRate = (float) video.get(CV_CAP_PROP_FPS);

	ListConnectComponent ccObjs, ccBikeLaneViolation, ccCarLaneViolation;

	cv::Size subPixWinSize(30, 30);

	//init Mat
	imgOverlay = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8UC3);

	//Read ROI img
	img = cv::imread(this->strRoiImg, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty()) {
		if (bDebug)
			cout << "ROI Image's missing" << endl;
		return;
	}
	cv::resize(img, img, cv::Size(iFrameWidth, iFrameHeight));

	rectRange = getMaxContoursFromROIImage(img);
	imgRoiWhole = img(rectRange);
	unsigned int iRoiArea = cv::countNonZero(imgRoiWhole);

	//Read Car ROI img
	img = cv::imread(this->strRoiCarImg, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty()) {
		if (bDebug)
			cout << "ROI Car Image's missing" << endl;
		return;
	}
	cv::resize(img, img, cv::Size(iFrameWidth, iFrameHeight));
	imgRoiCar = img(rectRange);

	//Read Bike ROI img
	img = cv::imread(this->strRoiBikeImg, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty()) {
		if (bDebug)
			cout << "ROI Bike Image's missing" << endl;
		return;
	}
	cv::resize(img, img, cv::Size(iFrameWidth, iFrameHeight));
	imgRoiBike = img(rectRange);
	OpticalFlowTracker oft_carTracker, oft_bikeTracker;
	cv::Mat imgGrayFrame;
#ifdef TUNING_LANE_VIOLATION
	ofstream out_carLane, out_bikeLane;
	out_carLane.open("Tunning_CarLane.csv");
	out_bikeLane.open("Tunning_bikeLane.csv");
#endif //TUNING_LANE_VIOLATION
	//setup background subtraction
	cv::BackgroundSubtractorMOG2 bg(this->iHistory, this->fVarThreshold, true);
	bg.set("nmixtures", 3);

	std::deque<float> density_queue;
	unsigned int num_group_frame = (int) (fFrameRate * this->iInterval);

	//Re-calculate min, max size after scaled
	int calMinSize, calMaxSize, calMinBikeSize, calMaxBikeSize, calMinCarSize, calMaxCarSize;
	if (!this->bHardThreshold) {
		calMinSize = int(this->iContourMinArea * this->fScaleRatio * this->fScaleRatio);
		calMaxSize = int(this->iContourMaxArea * this->fScaleRatio * this->fScaleRatio);
		calMinBikeSize = int(this->iBikeMinSize * this->fScaleRatio * this->fScaleRatio);
		calMaxBikeSize = int(this->iBikeMaxSize * this->fScaleRatio * this->fScaleRatio);
		calMinCarSize = int(this->iCarMinSize * this->fScaleRatio * this->fScaleRatio);
		calMaxCarSize = int(this->iCarMaxSize * this->fScaleRatio * this->fScaleRatio);
	} else {
		calMinSize = this->iContourMinArea;
		calMaxSize = this->iContourMaxArea;
		calMinBikeSize = this->iBikeMinSize;
		calMaxBikeSize = this->iBikeMaxSize;
		calMinCarSize = this->iCarMinSize;
		calMaxCarSize = this->iCarMaxSize;
	}

	if (bDebug) {
		stringstream ss;
		ss << "calMinSize=" << calMinSize 
			<< ",calMaxSize=" << calMaxSize 
			<< ",calMinBikeSize=" << calMinBikeSize
			<< ",calMaxBikeSize=" << calMaxBikeSize
			<< ",calMinCarSize=" << calMinCarSize
			<< ",calMaxCarSize=" << calMaxCarSize << endl;
		std::cout << ss.str();
		BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_INFO) << ss.str();
	}

	float fLearningRate = this->fMaxLearningRate;
	float list_values[5] = {0, 1, 2, 3, 4};

	//init tracker for bike and car
	//CTracker bikeTracker(_dt, _Accel_noise_mag, _dist_thres, _cos_thres, _maximum_allowed_skipped_frames, _max_trace_length, _very_large_cost);
	//CTracker carTracker(_dt, _Accel_noise_mag, _dist_thres, _cos_thres, _maximum_allowed_skipped_frames, _max_trace_length, _very_large_cost);
	int numNonData = 0;
	int index = 0;
	//bikeLaneMap = imgFore.mul(imgRoiBike);
	//carLaneMap = imgFore.mul(imgRoiCar);
#ifdef MEASURE_TIME_LANEVIOLATIONREGION
	ofstream out_log_stream;
	out_log_stream.open("MEASURE_TIME_LANEVIOLATIONREGION.log");
	std::string strMeasureTimeBuffer;
#endif //MEASURE_TIME_LANEVIOLATIONREGION
	for (; video.read(img);) {
		BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_TRACE) << "Frame " << index;
		if (!img.data) {
			std::cout << "No data!" << std::endl;
			numNonData++;
			if (numNonData > 4) {
				break;
			}
			continue;
		}

		cv::resize(img, imgFrame, cv::Size(iFrameWidth, iFrameHeight));
		imgFrame = imgFrame(rectRange);
		cv::cvtColor(imgFrame, imgGrayFrame, CV_BGR2GRAY);
#ifdef MEASURE_TIME_LANEVIOLATIONREGION
		clock_t t;
		t = clock();
#endif//MEASURE_TIME_LANEVIOLATIONREGION
		bg(imgFrame, imgFore, fLearningRate);
		cv::threshold(imgFore, imgFore, 200, 255, CV_THRESH_BINARY);
#ifdef MEASURE_TIME_LANEVIOLATIONREGION
		t = clock() - t;
		strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process background substraction: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
		out_log_stream << strMeasureTimeBuffer;
		//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_LANEVIOLATIONREGION
		//Display image on screen for debug
		if (bDebug) {
			imgFrame.copyTo(imgOverlay);
		}

		//process frame after training
		if (index > this->iTrainingFrame) {
			//START - Using bg subtraction to dectect violation and put to violatingVehicleCenters object
			dilation(imgFore, cv::MORPH_RECT, 1);
			erosion(imgFore, cv::MORPH_RECT, 1);

			imgFore = imgFore.mul(imgRoiWhole);

#ifdef MEASURE_TIME_LANEVIOLATIONREGION
			t = clock();
#endif//MEASURE_TIME_LANEVIOLATIONREGION
			cv::findContours(imgFore, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			examineContours(vContours, calMinSize, calMaxSize);
			ccObjs.extractConnectedComponentsFormContours(vContours);

#ifdef MEASURE_TIME_LANEVIOLATIONREGION
			t = clock() - t;
			strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process detect vehicle: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
			out_log_stream << strMeasureTimeBuffer;
			//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_LANEVIOLATIONREGION

#ifdef MEASURE_TIME_LANEVIOLATIONREGION
			t = clock();
#endif//MEASURE_TIME_LANEVIOLATIONREGION
			ccObjs.detectLaneViolationVehicles(imgRoiBike, calMinBikeSize, calMaxBikeSize, ccBikeLaneViolation);
			oft_bikeTracker.Update(imgGrayFrame, ccBikeLaneViolation);
#ifdef TUNING_LANE_VIOLATION
			for (size_t i = 0; i < ccBikeLaneViolation.getSize(); i++)
			{
				out_bikeLane << ccBikeLaneViolation.getListArea()[i] << "," << std::endl;
			}
#endif //TUNING_LANE_VIOLATION
			std::vector<cv::Rect> trackedBike = oft_bikeTracker.getNewBoudingBox();
			if (ccBikeLaneViolation.getSize() > 0) {
				if (trackedBike.size() > 0) {
					BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_INFO) << "Found Lane Violation - Bike Type";
					std::string strTimeViolation = "";
					if (camInfo.strVideoSrc.length() > 0)
						strTimeViolation = getCurrentTimeInVideoAsString(fFrameRate, index);
					handleViolation(img, rectRange, this->fScaleRatio, trackedBike, strTimeViolation, camInfo, VEHICLE_BIKE, LANE_VIOLATION);
				}

				//mark bike violation
				if (bDebug) {
					for (int i = 0; i < trackedBike.size(); i++) {
						//cout << trackedBike[i] << endl;
						cv::rectangle(imgOverlay, trackedBike[i], cv::Scalar(255, 255, 0), 2);
					}
				}
			}

#ifdef MEASURE_TIME_LANEVIOLATIONREGION
			t = clock() - t;
			strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process detect violation on bike lane: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
			out_log_stream << strMeasureTimeBuffer;
			//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_LANEVIOLATIONREGION


#ifdef MEASURE_TIME_LANEVIOLATIONREGION
			t = clock();
#endif//MEASURE_TIME_LANEVIOLATIONREGION
			//detect car violation
			ccObjs.detectLaneViolationVehicles(imgRoiCar, calMinCarSize, calMaxCarSize, ccCarLaneViolation, true);
#ifdef TUNING_LANE_VIOLATION
			for (size_t i = 0; i < ccCarLaneViolation.getSize(); i++)
			{
				out_carLane << ccCarLaneViolation.getListArea()[i] << "," << std::endl;
			}
#endif //TUNING_LANE_VIOLATION
			oft_carTracker.Update(imgGrayFrame, ccCarLaneViolation);
			std::vector<cv::Rect> trackedCar = oft_carTracker.getNewBoudingBox();
			if (ccCarLaneViolation.getSize() > 0) {

				if (trackedCar.size() > 0) {
					BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_INFO) << "Found Lane Violation - Car Type";
					std::string strTimeViolation = "";
					if (camInfo.strVideoSrc != "")
						strTimeViolation = getCurrentTimeInVideoAsString(fFrameRate, index);
					handleViolation(img, rectRange, this->fScaleRatio, trackedCar, strTimeViolation, camInfo, VEHICLE_CAR, LANE_VIOLATION);

				}

				//mark car violation
				if (bDebug) {
					for (int i = 0; i < trackedCar.size(); i++) {
						//cout << trackedCar[i] << endl;
						cv::rectangle(imgOverlay, trackedCar[i], cv::Scalar(255, 255, 0), 2);
					}
				}
			}

#ifdef MEASURE_TIME_LANEVIOLATIONREGION
			t = clock() - t;
			strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process detect violation on car lane: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
			out_log_stream << strMeasureTimeBuffer;
			//std::cout << strMeasureTimeBuffer;
#endif //MEASURE_TIME_LANEVIOLATIONREGION

			// START - Recalculate learning rate base on occupied ratio
			float fOccupiedSpace = (float) (cv::countNonZero(imgFore) * 100) / iRoiArea;
			density_queue.push_back(fOccupiedSpace);
			if (density_queue.size() > num_group_frame) {
				density_queue.pop_front();
			}
			float mean_density = meanQueue(density_queue);
			int density_rate;
			std::string density = getDensityStatus(fOccupiedSpace, density_rate);

			fLearningRate = calculateLearningRate(list_values, mean_density, density_rate, this->fMaxLearningRate);
			// END - Recalculate learning rate base on occupied ratio

			//debug case - display image
			if (bDebug) {
				if (ccObjs.getSize() > 0) {
					overlayContourAreas(imgOverlay, imgRoiBike, ccObjs.getListCenter(), ccObjs.getListArea());
					overlayContourAreas(imgOverlay, imgRoiCar, ccObjs.getListCenter(), ccObjs.getListArea());
				}
				char status[100];
				sprintf(status, "%d\nStatus: %s (%.1f %%)", index, density.c_str(), mean_density);
				std::string density_status(status);

				putText(imgOverlay, density_status, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2, 8, false);

				cv::findContours(imgRoiWhole, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);
				cv::findContours(imgRoiCar, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);
				cv::findContours(imgRoiBike, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);
				//
				resize(imgOverlay, imgOverlay, Size(imgOverlay.cols / 2, imgOverlay.rows / 2));
				//
				cv::imshow("Overlay" + camInfo.strCameraId, imgOverlay);
			}

			//clear
			ccObjs.clear();
			ccBikeLaneViolation.clear();
			ccCarLaneViolation.clear();
		} else {
			if (bDebug) {
				std::string status = "Modeling background...";
				putText(imgOverlay, status, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2, 8, false);
				//
				resize(imgOverlay, imgOverlay, Size(imgOverlay.cols / 2, imgOverlay.rows / 2));
				//
				cv::imshow("Overlay" + camInfo.strCameraId, imgOverlay);
			}
		}

		if (bDebug && cv::waitKey(10) == 27)
			break;
		
		index++;
		
		if(index == INT_MAX) //trungnt1 add to avoid overflow
			index = 0;
#ifdef MEASURE_TIME_LANEVIOLATIONREGION
		out_log_stream << "\n";
#endif //MEASURE_TIME_LANEVIOLATIONREGION
	}
#ifdef MEASURE_TIME_LANEVIOLATIONREGION
	
	out_log_stream.close();
#endif //MEASURE_TIME_LANEVIOLATIONREGION
#ifdef TUNING_LANE_VIOLATION
	out_carLane.close();
	out_bikeLane.close();
#endif//TUNING_LANE_VIOLATION
}

void FTSLaneViolationRegion::read(const cv::FileNode& fn) {
	this->info()->read(this, fn);

	fn["strRoiCarImg"] >> this->strRoiCarImg;
	fn["strRoiBikeImg"] >> this->strRoiBikeImg;

	//read anpr param file
	cv::FileStorage fs(this->strAnprParamFile, cv::FileStorage::READ);
	if (fs.isOpened()) {
		this->anpr.read(fs.root());
		fs.release();
	}
}

void FTSLaneViolationRegion::write(cv::FileStorage& fs) const {
	this->info()->write(this, fs);
	fs << "strRoiCarImg" << this->strRoiCarImg;
	fs << "strRoiBikeImg" << this->strRoiBikeImg;
}


FTSLaneViolationRegion::FTSLaneViolationRegion() {
	BOOST_LOG_CHANNEL_SEV(lg, FTSLaneViolationRegion::className(), LOG_INFO) << "Init";

	iRetryCount = 0;

	// Following init value will be override in config file - only for development
	this->strRoiImg = "";

	this->fVarThreshold = 50;
	this->fMaxLearningRate = .005f;
	this->fScaleRatio = .1f;

	this->iHistory = 10;
	this->iTrainingFrame = 100;
	this->iInterval = 60;

	this->iBikeMinSize = 25000;
	this->iBikeMaxSize = 30000000;
	this->iCarMinSize = 5000;
	this->iCarMaxSize = 20000;

	this->iContourMinArea = 1500;
	this->iContourMaxArea = 1000000;

	this->bHardThreshold = true;

	//tracker params 0.2f, 0.5f, 40.f, 0, 5, 10, 1000000
	this->_dt = 0.2f;
	this->_Accel_noise_mag = 0.5f;
	this->_dist_thres = 40.0f;
	this->_cos_thres = 0;
	this->_maximum_allowed_skipped_frames = 5;
	this->_max_trace_length = 10;
	this->_very_large_cost = 1000000;
}

FTSLaneViolationRegion::~FTSLaneViolationRegion() {}
