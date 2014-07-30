#include "FTSRedSignalViolationBS.h"
#include "CVUtil.h"

bool FTSRedSignalViolationBS::checkRedSignal(bool isRed, const cv::Mat& redTL, Light& red_light, const cv::Mat& yelTL, Light& yel_light, const cv::Mat& grnTL, Light& grn_light) {
	// Red
	if (!isRed && red_light.isOn(redTL) && !yel_light.isOn(yelTL) && !grn_light.isOn(grnTL)) {
		return true;
	}
	// No Red
	else if (isRed && (!red_light.isOn(redTL) && !yel_light.isOn(yelTL) && grn_light.isOn(grnTL))) {
		return false;
	}
	return isRed;
}

void FTSRedSignalViolationBS::operator() (FTSCamera camInfo) {
	cv::Mat img, imgFrame, imgRoiWhole, rsvdImg, imgFore, imgBack, imgOverlay, imgTrafficLight, imgStopRegionImage;
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

	//Read ROI img
	imgRoiWhole = cv::imread(this->strRoiImg, CV_LOAD_IMAGE_GRAYSCALE);
	if (imgRoiWhole.empty()) {
		if (bDebug)
			cout << "ROI Image's missing" << endl;
		return;
	}
	cv::resize(imgRoiWhole, imgRoiWhole, cv::Size(iFrameWidth, iFrameHeight));
	rectRange = getMaxContoursFromROIImage(imgRoiWhole);
	imgRoiWhole = imgRoiWhole(rectRange);

	//read STOP REGION ROI IMAGE
	imgStopRegionImage = cv::imread(this->strStopRoiImage, CV_LOAD_IMAGE_GRAYSCALE);
	if (imgStopRegionImage.empty()){
		if (bDebug)
			std::cout << "STOP REGION IMAGE's missing" << std::endl;
		return;
	}
	cv::resize(imgStopRegionImage, imgStopRegionImage, cv::Size(iFrameWidth, iFrameHeight));

	// Background subtraction
	cv::BackgroundSubtractorMOG2 bg(this->iHistory, this->fVarThreshold, true);
	bg.set("nmixtures", 3);
	float fLearningRate = this->fMaxLearningRate;
	std::deque<float> density_queue;
	unsigned int num_group_frame = (int) (fFrameRate * this->iInterval);
	float list_values[5] = {0, 1, 2, 3, 4};

	//re-calculate top-left of all traffic light
	if (bDebug) {
		std::cout << "Before -- TL red: " << this->lightRed.rect.tl() << " TL yellow: " << this->lightYellow.rect.tl() << " TL green: " << this->lightGreen.rect.tl() << std::endl;
	}
	this->lightRed.rect.x -= this->rectTrafficLight.x;
	this->lightRed.rect.y -= this->rectTrafficLight.y;
	this->lightGreen.rect.x -= this->rectTrafficLight.x;
	this->lightGreen.rect.y -= this->rectTrafficLight.y;
	this->lightYellow.rect.x -= this->rectTrafficLight.x;
	this->lightYellow.rect.y -= this->rectTrafficLight.y;
	if (bDebug) {
		std::cout << "After -- TL red: " << this->lightRed.rect.tl() << " TL yellow: " << this->lightYellow.rect.tl() << " TL green: " << this->lightGreen.rect.tl() << std::endl;
	}



	std::vector<cv::Mat> listTLChannels;

	unsigned int roi_area = countPixels(imgStopRegionImage, 0);

	// Tracking
	std::vector<cv::Rect>  violatingVehicleBbs;
	//std::vector<cv::Point2d> potentialViolatingCenters;

	ListConnectComponent ccObjs;

	cv::Size subPixWinSize(30, 30);
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.3);

	//Re-calculate min, max size after scaled
	int calMinSize, calMaxSize;
	if (bHardThreshold){
		calMinSize = this->iContourMinArea ;
		calMaxSize = this->iContourMaxArea ;
	}
	else{
		calMinSize = int(this->iContourMinArea * this->fScaleRatio * this->fScaleRatio);
		calMaxSize = int(this->iContourMaxArea * this->fScaleRatio * this->fScaleRatio);
	}
	// Tracking initialize
	//CTracker tracker(0.2f, 0.5f, 40.0f, 0, 10, 10, 1000000);
	CTracker suspectsTracker(_dt, _Accel_noise_mag, _dist_thres, _cos_thres, _maximum_allowed_skipped_frames, _max_trace_length, _very_large_cost);
	std::vector<int> trackVehMap;
	std::vector<int> trackIndices;
	// Counting lines

	// Transform violation lines
	Line_<double> inLine, outLine;
	inLine.start.x = this->inLine.start.x * this->fScaleRatio - rectRange.x;
	inLine.start.y = this->inLine.start.y * this->fScaleRatio - rectRange.y;
	inLine.end.x = this->inLine.end.x * this->fScaleRatio - rectRange.x;
	inLine.end.y = this->inLine.end.y * this->fScaleRatio - rectRange.y;
	outLine.start.x = this->outLine.start.x * this->fScaleRatio - rectRange.x;
	outLine.end.x = this->outLine.end.x * this->fScaleRatio - rectRange.x;
	outLine.start.y = this->outLine.start.y * this->fScaleRatio - rectRange.y;
	outLine.end.y = this->outLine.end.y * this->fScaleRatio - rectRange.y;

	int index = 0;

	bool bRed = false;
#ifdef MEASURE_TIME_REDLIGHT
	ofstream out_log_stream;
	out_log_stream.open("MEASURE_TIME_REDLIGHT.log");
	std::string strMeasureTimeBuffer;
	clock_t t;
#endif //MEASURE_TIME_REDLIGHT
#ifdef TUNNING_RED_LIGHT
	ofstream out_vehicleSize;
	out_vehicleSize.open("Tunning_vehicleSizeRedLight.csv");
#endif // TUNNING_RED_LIGHT
	for (; video.read(img);) {
		BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_TRACE) << "Frame " << index;

		cv::resize(img, imgFrame, cv::Size(iFrameWidth, iFrameHeight));
		imgFrame = imgFrame(rectRange);
#ifdef MEASURE_TIME_REDLIGHT
		t = clock();
#endif//MEASURE_TIME_REDLIGHT
		bg(imgFrame, imgFore, fLearningRate);
		//bg.getBackgroundImage(imgBack);
		cv::threshold(imgFore, imgFore, 200, 255, CV_THRESH_BINARY);
#ifdef MEASURE_TIME_REDLIGHT
		t = clock() - t;
		strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process background: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
		out_log_stream << strMeasureTimeBuffer;
		//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_REDLIGHT

		// Test Red Signal
		imgTrafficLight = img(this->rectTrafficLight);

		//split image to 3 channel for check light threshold
		cv::split(imgTrafficLight, listTLChannels);
		cv::Mat redTL = listTLChannels[2];
		cv::Mat grnTL = listTLChannels[1];
		cv::Mat yelTL = (listTLChannels[1] + listTLChannels[2]) / 2;

		bRed = checkRedSignal(bRed, redTL, lightRed, yelTL, lightYellow, grnTL, lightGreen);

		//finish training
		if (index > this->iTrainingFrame) {

#ifdef MEASURE_TIME_REDLIGHT
			t = clock();
#endif//MEASURE_TIME_REDLIGHT
			intersectROI(imgFore, imgRoiWhole, imgFore);

			dilation(imgFore, cv::MORPH_RECT, 1);
			erosion(imgFore, cv::MORPH_RECT, 1);
#ifdef TUNNING_RED_LIGHT
			if (true) {
#else
			if (!bRed) {
				suspectsTracker.~CTracker();
			}
			else{
#endif //TUNNING_RED_LIGHT
			findContours(imgFore, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			examineContours(vContours, calMinSize, calMaxSize);

			ccObjs.extractConnectedComponentsFormContours(vContours);
#ifdef MEASURE_TIME_REDLIGHT
			t = clock() - t;
			strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process detect component: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
			out_log_stream << strMeasureTimeBuffer;
			//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_REDLIGHT

			//violatingVehicleBbs = std::vector<cv::Rect>();
			violatingVehicleBbs.clear();
			
			
#ifdef MEASURE_TIME_REDLIGHT
			t = clock();
#endif//MEASURE_TIME_REDLIGHT
			suspectsTracker.Update(ccObjs.getListCenter());
#ifdef MEASURE_TIME_REDLIGHT
			t = clock() - t;
			strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process tracking: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
			out_log_stream << strMeasureTimeBuffer;
			//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_REDLIGHT
			if (bDebug) {
				//std::cout << suspectsTracker.tracks.size() << endl;
				suspectsTracker.drawTrackToImage(imgFrame);
			}

			if (bDebug) {
				for (int i = 0; i < ccObjs.getSize(); i++) {
					circle(imgFrame, ccObjs.getListCenter()[i], 3, Scalar(0, 255, 0), 2, CV_AA);
				}
				std::cout << "Num of tracks: " << suspectsTracker.tracks.size() << std::endl;
			}


				//detectObjsInROI(mRoiRSVImg, tracker, objBbs, potentialViolatingBbs, potentialViolatingCenters);
				//detectObjsInROI(mRoiRSVImg, objBbs, objCenters, potentialViolatingBbs, potentialViolatingCenters);

				//std::cout << "Num of potentials: " << potentialViolatingBbs.size() << std::endl;
				
#ifdef MEASURE_TIME_REDLIGHT
				t = clock();
#endif//MEASURE_TIME_REDLIGHT
				//tracking violation

				//if (potentialViolatingBbs.size() > 0) {

				// Map from indices of tracks to indices of moving objects

				// Double-line mode
				if (detectionMode == 2) {
					detectCrossingDoubleLines(suspectsTracker, trackIndices, inLine, outLine);
				} else if (detectionMode == 1) {
					detectBasedMovingDirection(suspectsTracker, trackIndices, outLine);
				}
				if (ccObjs.getSize() > 0){
					for (size_t i = 0; i < trackIndices.size(); i++) {
						int bbInd = suspectsTracker.assignment[trackIndices[i]];
						if (bbInd > -1 && !suspectsTracker.tracks[trackIndices[i]]->isCaught) {
							violatingVehicleBbs.push_back(ccObjs.getListBoundingBox()[bbInd]);
							suspectsTracker.tracks[trackIndices[i]]->isCaught = true;
#ifdef TUNNING_RED_LIGHT
							out_vehicleSize << ccObjs.getListArea()[bbInd] << "," << std::endl;
#endif//TUNNING_RED_LIGHT
						}
					}
				}
				/*if (bDebug) {
					for (size_t i = 0; i < trackIndices.size(); i++)
					{
					int bbInd = suspectsTracker.assignment[trackIndices[i]];
					if (bbInd > -1 && suspectsTracker.tracks[trackIndices[i]]->isCaught) {
					cv::circle(imgFrame, objCenters[bbInd], 4, Scalar(0, 0, 255), 2, CV_AA);
					}
					}
					}
					*/
				//handle violation
				if (violatingVehicleBbs.size() > 0) {
					BOOST_LOG_CHANNEL_SEV(lg, camInfo.strCameraId, LOG_INFO) << "Found Red Light Violation";
					std::string strTimeViolation = "";
					if (camInfo.strVideoSrc.length() > 0)
						strTimeViolation = getCurrentTimeInVideoAsString(fFrameRate, index);
					handleViolation(img, rectRange, this->fScaleRatio, violatingVehicleBbs, strTimeViolation, camInfo, VEHICLE_UNDECIDED, REDSIGNAL_VIOLATION);
				}
#ifdef MEASURE_TIME_REDLIGHT
				t = clock() - t;
				strMeasureTimeBuffer = "Frame " + std::to_string(index) + " Time process red light violation: " + std::to_string(((float)t) / CLOCKS_PER_SEC) + "\n";
				out_log_stream << strMeasureTimeBuffer;
				//std::cout << strMeasureTimeBuffer ;
#endif //MEASURE_TIME_REDLIGHT
			}


			// Recalculate learning rate base on occupied ratio
			float occupied_space = calOccupiedSpace(imgFore, roi_area, 1);
			density_queue.push_back(occupied_space);
			if (density_queue.size() > num_group_frame) {
				density_queue.pop_front();
			}
			float mean_density = meanQueue(density_queue);
			int density_rate;
			std::string density = getDensityStatus(occupied_space, density_rate);
			if (bRed){
				fLearningRate = 0;
			}
			else{
				fLearningRate = calculateLearningRate(list_values, mean_density, density_rate, this->fMaxLearningRate);
			}

			if (bDebug) {
				rectangle(imgTrafficLight, lightRed.rect, cv::Scalar(0, 0, 255), 2);
				rectangle(imgTrafficLight, lightGreen.rect, cv::Scalar(0, 255, 0), 2);
				rectangle(imgTrafficLight, lightYellow.rect, cv::Scalar(0, 255, 255), 2);
				if (bRed){
					putText(imgTrafficLight, "RED", cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2, 8, false);
				}
				cv::imshow("Traffic light" + camInfo.strCameraId, imgTrafficLight);
				if (detectionMode == 2) {
					cv::line(imgFrame, inLine.start, inLine.end, cv::Scalar(0, 255, 255), 2);
				}
				cv::line(imgFrame, outLine.start, outLine.end, cv::Scalar(0, 0, 255), 2);

			}

			//debug case - display image
			if (bDebug) {
				//overlayMap(imgFrame, imgOverlay, imgFore);
				imgFrame.copyTo(imgOverlay);

				if (violatingVehicleBbs.size() > 0) {
					markViolations(imgOverlay, violatingVehicleBbs, cv::Scalar(255, 255, 0));
				}

				char status[100];
				sprintf(status, "%d\nStatus: %s (%.1f %%)", index, density.c_str(), mean_density);
				std::string density_status(status);
				//std::cout << camInfo.strCameraId << ": " << status << std::endl;

				putText(imgOverlay, density_status, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2, 8, false);

				if (false) {
					cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(255, 0, 0), 1);
					cv::findContours(imgRoiWhole, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
					cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);
				}
				//cv::findContours(mRoiRSVImg, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
				//cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(255, 0, 0), 1);

				cv::imshow("Overlay" + camInfo.strCameraId, imgOverlay);

			}

			//clear
			trackIndices.clear();
			vContours.clear();
			ccObjs.clear();
			violatingVehicleBbs.clear();
			listTLChannels.clear();

		} else {
			if (bRed){
				fLearningRate = 0;
			}
			else{
				fLearningRate = this->fMaxLearningRate;
			}
			if (bDebug) {
				//display image
				imgFrame.copyTo(imgOverlay);

				std::string status = "Preprocessing...";
				putText(imgOverlay, status, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 2, 8, false);
				cv::imshow("Overlay" + camInfo.strCameraId, imgOverlay);
			}
		}

		if (bDebug && cv::waitKey(1) == 27)
			break;

		index++;
		if(index == INT_MAX) //trungnt1 add to avoid overflow
			index = 0;
#ifdef MEASURE_TIME_REDLIGHT
		out_log_stream << "\n";
#endif //MEASURE_TIME_REDLIGHT
	}
}

void FTSRedSignalViolationBS::read(const cv::FileNode& fn) {
	this->info()->read(this, fn);

	fn["RectLight"] >> this->rectTrafficLight;

	fn["InLine"] >> this->inLine;
	fn["OutLine"] >> this->outLine;

	this->lightRed.read(fn["red"]);
	this->lightGreen.read(fn["green"]);
	this->lightYellow.read(fn["yellow"]);

	//read anpr param file
	cv::FileStorage fs(this->strAnprParamFile, cv::FileStorage::READ);
	if (fs.isOpened()) {
		//this->anpr.read(fs.root());
		fs.release();
	}
}

void FTSRedSignalViolationBS::write(cv::FileStorage& fs) const {
	this->info()->write(this, fs);

	fs << "RectLight" << this->rectTrafficLight;

	fs << "InLine" << this->inLine;
	fs << "OutLine" << this->outLine;

	fs << "red" << this->lightRed;
	fs << "green" << this->lightGreen;
	fs << "yellow" << this->lightYellow;
}

FTSRedSignalViolationBS::FTSRedSignalViolationBS() {
	BOOST_LOG_CHANNEL_SEV(lg, FTSRedSignalViolationBS::className(), LOG_INFO) << "Init";

	iRetryCount = 0;

	// Following init value will be override in config file - only for development
	this->strRoiImg = "";

	this->fScaleRatio = .2f;

	// Background subtraction
	this->fVarThreshold = 50;
	this->fMaxLearningRate = .005f;
	this->fScaleRatio = .3f;

	this->iHistory = 10;
	this->iTrainingFrame = 100;
	this->iInterval = 60;
	this->iContourMinArea = 3000;
	this->iContourMaxArea = 1000000;

	// Traffic light
	this->lightRed = Light(516, 522, 1625, 1630, 20, 1);
	this->lightYellow = Light(541, 546, 1625, 1630, 20, 1);
	this->lightGreen = Light(566, 572, 1625, 1630, 20, 1);

	this->rectTrafficLight = cv::Rect(1585, 450, 100, 150);
	//this->inLine = Line_<double> (461, 975, 1647, 1017);
	this->inLine = Line_<double>(570, 800, 1475, 800);
	this->outLine = Line_<double>(570, 750, 1475, 700);

	this->bDebug = true;
	this->detectionMode = 2;

	this->bHardThreshold = true;
	//tracker params 0.2f, 0.5f, 40.f, 0, 10, 10, 1000000
	this->_dt = 0.2f;
	this->_Accel_noise_mag = 0.5f;
	this->_dist_thres = 40.0f;
	this->_cos_thres = 0;
	this->_maximum_allowed_skipped_frames = 10;
	this->_max_trace_length = 10;
	this->_very_large_cost = 1000000;
}


FTSRedSignalViolationBS::~FTSRedSignalViolationBS() {}
