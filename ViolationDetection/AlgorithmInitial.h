#ifndef _FTS_LV_ALGORITHMINITIAL_
#define _FTS_LV_ALGORITHMINITIAL_

#include <opencv2/opencv.hpp>

#include "FTSAlgorithm.h"
#include "FTSVideoAlgorithm.h"
#include "FTSLaneViolation.h"
#include "FTSLaneViolationRegion.h"
#include "FTSPlateScanner.h"
#include "FTSRedSignalViolation.h"
#include "FTSRedSignalViolationBS.h"
#include "FTSVehicleCounting.h"
#include "FTSANPR.h"


static cv::Algorithm* createFTSAlgorithm() {
	return new FTSAlgorithm;
}
static cv::Algorithm* createFTSVideoAlgorithm() {
	return new FTSVideoAlgorithm;
}
static cv::Algorithm* createFTS_ANPR() {
	return new FTSANPR;
}
static cv::Algorithm* createFTS_LaneViolation() {
	return new FTSLaneViolation;
}
static cv::Algorithm* createFTS_LaneViolationRegion() {
	return new FTSLaneViolationRegion;
}
static cv::Algorithm* createFTS_PlateScanner() {
	return new FTSPlateScanner;
}
static cv::Algorithm* createFTS_RedSignalViolationBS() {
	return new FTSRedSignalViolation;
}
static cv::Algorithm* createFTS_RedSignalViolation() {
	return new FTSRedSignalViolation;
}
static cv::Algorithm* createFTS_VehicleCounting() {
	return new FTSVehicleCounting;
}


static cv::AlgorithmInfo al_info("FTSAlgorithm", createFTSAlgorithm);
static cv::AlgorithmInfo alv_info("FTSAlgorithm.Video", createFTSVideoAlgorithm);
static cv::AlgorithmInfo lv_info("FTSAlgorithm.LaneViolation", createFTS_LaneViolation);
static cv::AlgorithmInfo lvr_info("FTSAlgorithm.LaneViolationRegion", createFTS_LaneViolationRegion);
static cv::AlgorithmInfo ps_info("FTSAlgorithm.PlateScanner", createFTS_PlateScanner);
static cv::AlgorithmInfo rvbs_info("FTSAlgorithm.RedSignalViolationBS", createFTS_RedSignalViolationBS);
static cv::AlgorithmInfo rv_info("FTSAlgorithm.RedSignalViolation", createFTS_RedSignalViolation);
static cv::AlgorithmInfo vc_info("FTSAlgorithm.VehicleCounting", createFTS_VehicleCounting);
static cv::AlgorithmInfo anpr_info("FTSAlgorithm.ANPR", createFTS_ANPR);


cv::AlgorithmInfo* FTSAlgorithm::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSAlgorithm obj;
		al_info.addParam(obj, "bDebug", obj.bDebug);

		initialized = true;
	}
	return &al_info;
}

cv::AlgorithmInfo* FTSVideoAlgorithm::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSVideoAlgorithm obj;
		alv_info.addParam(obj, "bDebug", obj.bDebug);
		alv_info.addParam(obj, "AnprParamFile", obj.strAnprParamFile);

		initialized = true;
	}
	return &alv_info;
}

cv::AlgorithmInfo* FTSLaneViolation::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSLaneViolation obj;
		lv_info.addParam(obj, "bDebug", obj.bDebug);
		lv_info.addParam(obj, "AnprParamFile", obj.strAnprParamFile);

		lv_info.addParam(obj, "strRoiImg", obj.strRoiImg);

		lv_info.addParam(obj, "fMaxLearningRate", obj.fMaxLearningRate);
		lv_info.addParam(obj, "fScaleRatio", obj.fScaleRatio);
		lv_info.addParam(obj, "fVarThreshold", obj.fVarThreshold);

		lv_info.addParam(obj, "iBikeMinSize", obj.iBikeMinSize);
		lv_info.addParam(obj, "iBikeMaxSize", obj.iBikeMaxSize);
		lv_info.addParam(obj, "iCarMinSize", obj.iCarMinSize);
		lv_info.addParam(obj, "iCarMaxSize", obj.iCarMaxSize);
		lv_info.addParam(obj, "iContourMinArea", obj.iContourMinArea);
		lv_info.addParam(obj, "iContourMaxArea", obj.iContourMaxArea);
		lv_info.addParam(obj, "iHistory", obj.iHistory);
		lv_info.addParam(obj, "iInterval", obj.iInterval);
		lv_info.addParam(obj, "iTrainingFrame", obj.iTrainingFrame);

		lv_info.addParam(obj, "_dt", obj._dt);
		lv_info.addParam(obj, "_Accel_noise_mag", obj._Accel_noise_mag);
		lv_info.addParam(obj, "_dist_thres", obj._dist_thres);
		lv_info.addParam(obj, "_cos_thres", obj._cos_thres);
		lv_info.addParam(obj, "_maximum_allowed_skipped_frames", obj._maximum_allowed_skipped_frames);
		lv_info.addParam(obj, "_max_trace_length", obj._max_trace_length);
		lv_info.addParam(obj, "_very_large_cost", obj._very_large_cost);

		initialized = true;
	}
	return &lv_info;
}

cv::AlgorithmInfo* FTSLaneViolationRegion::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSLaneViolationRegion obj;
		lvr_info.addParam(obj, "bDebug", obj.bDebug);
		lvr_info.addParam(obj, "AnprParamFile", obj.strAnprParamFile);

		lvr_info.addParam(obj, "strRoiImg", obj.strRoiImg);
		lvr_info.addParam(obj, "strRoiBikeImg", obj.strRoiBikeImg);
		lvr_info.addParam(obj, "strRoiCarImg", obj.strRoiCarImg);

		lvr_info.addParam(obj, "fMaxLearningRate", obj.fMaxLearningRate);
		lvr_info.addParam(obj, "fScaleRatio", obj.fScaleRatio);
		lvr_info.addParam(obj, "fVarThreshold", obj.fVarThreshold);

		lvr_info.addParam(obj, "iBikeMinSize", obj.iBikeMinSize);
		lvr_info.addParam(obj, "iBikeMaxSize", obj.iBikeMaxSize);
		lvr_info.addParam(obj, "iCarMinSize", obj.iCarMinSize);
		lvr_info.addParam(obj, "iCarMaxSize", obj.iCarMaxSize);
		lvr_info.addParam(obj, "iContourMinArea", obj.iContourMinArea);
		lvr_info.addParam(obj, "iContourMaxArea", obj.iContourMaxArea);
		lvr_info.addParam(obj, "iHistory", obj.iHistory);
		lvr_info.addParam(obj, "iInterval", obj.iInterval);
		lvr_info.addParam(obj, "iTrainingFrame", obj.iTrainingFrame);

		lvr_info.addParam(obj, "_dt", obj._dt);
		lvr_info.addParam(obj, "_Accel_noise_mag", obj._Accel_noise_mag);
		lvr_info.addParam(obj, "_dist_thres", obj._dist_thres);
		lvr_info.addParam(obj, "_cos_thres", obj._cos_thres);
		lvr_info.addParam(obj, "_maximum_allowed_skipped_frames", obj._maximum_allowed_skipped_frames);
		lvr_info.addParam(obj, "_max_trace_length", obj._max_trace_length);
		lvr_info.addParam(obj, "_very_large_cost", obj._very_large_cost);

		initialized = true;
	}
	return &lvr_info;
}

cv::AlgorithmInfo* FTSPlateScanner::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSPlateScanner obj;
		ps_info.addParam(obj, "bDebug", obj.bDebug);
		ps_info.addParam(obj, "AnprParamFile", obj.strAnprParamFile);

		ps_info.addParam(obj, "strRoiImg", obj.strRoiImg);
		ps_info.addParam(obj, "strRoiVehicleImg", obj.strRoiVehicleImg);

		ps_info.addParam(obj, "fMaxLearningRate", obj.fMaxLearningRate);
		ps_info.addParam(obj, "fScaleRatio", obj.fScaleRatio);
		ps_info.addParam(obj, "fVarThreshold", obj.fVarThreshold);

		ps_info.addParam(obj, "iContourMinArea", obj.iContourMinArea);
		ps_info.addParam(obj, "iContourMaxArea", obj.iContourMaxArea);
		ps_info.addParam(obj, "iHistory", obj.iHistory);
		ps_info.addParam(obj, "iInterval", obj.iInterval);
		ps_info.addParam(obj, "iTrainingFrame", obj.iTrainingFrame);

		ps_info.addParam(obj, "_dt", obj._dt);
		ps_info.addParam(obj, "_Accel_noise_mag", obj._Accel_noise_mag);
		ps_info.addParam(obj, "_dist_thres", obj._dist_thres);
		ps_info.addParam(obj, "_cos_thres", obj._cos_thres);
		ps_info.addParam(obj, "_maximum_allowed_skipped_frames", obj._maximum_allowed_skipped_frames);
		ps_info.addParam(obj, "_max_trace_length", obj._max_trace_length);
		ps_info.addParam(obj, "_very_large_cost", obj._very_large_cost);

		initialized = true;
	}
	return &ps_info;
}

cv::AlgorithmInfo* FTSRedSignalViolation::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSRedSignalViolation obj;
		rv_info.addParam(obj, "bDebug", obj.bDebug);
		rv_info.addParam(obj, "AnprParamFile", obj.strAnprParamFile);

		rv_info.addParam(obj, "strRoiImg", obj.strRoiImg);
		rv_info.addParam(obj, "strStopRoiImage", obj.strStopRoiImage);

		rv_info.addParam(obj, "fMaxLearningRate", obj.fMaxLearningRate);
		rv_info.addParam(obj, "fScaleRatio", obj.fScaleRatio);
		rv_info.addParam(obj, "fVarThreshold", obj.fVarThreshold);

		rv_info.addParam(obj, "iContourMinArea", obj.iContourMinArea);
		rv_info.addParam(obj, "iContourMaxArea", obj.iContourMaxArea);
		rv_info.addParam(obj, "iHistory", obj.iHistory);
		rv_info.addParam(obj, "iInterval", obj.iInterval);
		rv_info.addParam(obj, "iTrainingFrame", obj.iTrainingFrame);

		rv_info.addParam(obj, "_dt", obj._dt);
		rv_info.addParam(obj, "_Accel_noise_mag", obj._Accel_noise_mag);
		rv_info.addParam(obj, "_dist_thres", obj._dist_thres);
		rv_info.addParam(obj, "_cos_thres", obj._cos_thres);
		rv_info.addParam(obj, "_maximum_allowed_skipped_frames", obj._maximum_allowed_skipped_frames);
		rv_info.addParam(obj, "_max_trace_length", obj._max_trace_length);
		rv_info.addParam(obj, "_very_large_cost", obj._very_large_cost);

		initialized = true;
	}
	return &rv_info;
}

cv::AlgorithmInfo* FTSRedSignalViolationBS::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSRedSignalViolationBS obj;
		rvbs_info.addParam(obj, "bDebug", obj.bDebug);
		rvbs_info.addParam(obj, "AnprParamFile", obj.strAnprParamFile);

		rvbs_info.addParam(obj, "strRoiImg", obj.strRoiImg);
		rvbs_info.addParam(obj, "strStopRoiImage", obj.strStopRoiImage);

		rvbs_info.addParam(obj, "fMaxLearningRate", obj.fMaxLearningRate);
		rvbs_info.addParam(obj, "fScaleRatio", obj.fScaleRatio);
		rvbs_info.addParam(obj, "fVarThreshold", obj.fVarThreshold);

		rvbs_info.addParam(obj, "iContourMinArea", obj.iContourMinArea);
		rvbs_info.addParam(obj, "iContourMaxArea", obj.iContourMaxArea);
		rvbs_info.addParam(obj, "iHistory", obj.iHistory);
		rvbs_info.addParam(obj, "iInterval", obj.iInterval);
		rvbs_info.addParam(obj, "iTrainingFrame", obj.iTrainingFrame);

		rvbs_info.addParam(obj, "_dt", obj._dt);
		rvbs_info.addParam(obj, "_Accel_noise_mag", obj._Accel_noise_mag);
		rvbs_info.addParam(obj, "_dist_thres", obj._dist_thres);
		rvbs_info.addParam(obj, "_cos_thres", obj._cos_thres);
		rvbs_info.addParam(obj, "_maximum_allowed_skipped_frames", obj._maximum_allowed_skipped_frames);
		rvbs_info.addParam(obj, "_max_trace_length", obj._max_trace_length);
		rvbs_info.addParam(obj, "_very_large_cost", obj._very_large_cost);

		initialized = true;
	}
	return &rvbs_info;
}
cv::AlgorithmInfo* FTSVehicleCounting::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSVehicleCounting obj;
		vc_info.addParam(obj, "bDebug", obj.bDebug);

		initialized = true;
	}
	return &vc_info;
}

cv::AlgorithmInfo* FTSANPR::info() const {
	static volatile bool initialized = false;
	if (!initialized) {
		FTSANPR obj;
		anpr_info.addParam(obj, "bDebug", obj.bDebug);
		anpr_info.addParam(obj, "bDelayOnFrame", obj.m_bDelayOnFrame);
		anpr_info.addParam(obj, "bDisplayDbgImg", obj.m_bDisplayDbgImg);
		anpr_info.addParam(obj, "LogLevel", obj.m_iLogLevel);
		anpr_info.addParam(obj, "bOutCharSegments", obj.m_bOutCharSegments);
#ifndef USING_SVM_OCR
		anpr_info.addParam(obj, "RuntimeOcrDir", obj.m_strRuntimeOcrDir);
		anpr_info.addParam(obj, "OcrLanguage", obj.m_strOcrLanguage);
#else
		anpr_info.addParam(obj, "DigitModelsDir", obj.m_strDigitModelsDir);
		anpr_info.addParam(obj, "LetterModelsDir", obj.m_strLetterModelsDir);
		anpr_info.addParam(obj, "ModelPairsDir", obj.m_strModelPairsDir);
		anpr_info.addParam(obj, "ModelPairPrefix", obj.m_strModelPairPrefix);
#endif//USING_SVM_OCR
		anpr_info.addParam(obj, "PostProcessFile", obj.m_strPostProcessFile);
		anpr_info.addParam(obj, "CascadeFile", obj.m_strCascadeFile);

		anpr_info.addParam(obj, "MinCharToProcess", obj.m_iMinCharToProcess);
		anpr_info.addParam(obj, "MaxCharToProcess", obj.m_iMaxCharToProcess);
		anpr_info.addParam(obj, "MinOcrFont", obj.m_iMinOcrFont);
		anpr_info.addParam(obj, "EnableOcrDebug", obj.m_bOcrDebug);

		anpr_info.addParam(obj, "MinCharConf", obj.m_iMinCharConf);
		anpr_info.addParam(obj, "ConfidenceSkipCharLevel", obj.m_iConfSkipCharLevel);
		anpr_info.addParam(obj, "MaxSubstitutions", obj.m_iMaxSubstitutions);
		anpr_info.addParam(obj, "EnablePostProcessDebug", obj.m_bPostProcessDebug);

		anpr_info.addParam(obj, "LPDScaleFactor", obj.m_fLPDScaleFactor);
		anpr_info.addParam(obj, "LPDMinNeighbors", obj.m_iLPDMinNeighbors);
		anpr_info.addParam(obj, "LPDMinPlateWidth", obj.m_iLPDMinPlateWidth);
		anpr_info.addParam(obj, "LPDMinPlateHeight", obj.m_iLPDMinPlateHeight);
		anpr_info.addParam(obj, "LPDMaxPlateWidth", obj.m_iLPDMaxPlateWidth);
		anpr_info.addParam(obj, "LPDMaxPlateHeight", obj.m_iLPDMaxPlateHeight);
		anpr_info.addParam(obj, "ExpandRectX", obj.m_fExpandRectX);
		anpr_info.addParam(obj, "ExpandRectY", obj.m_fExpandRectY);
		anpr_info.addParam(obj, "TemplatePlateWidth", obj.m_iTemplatePlateWidth);

		anpr_info.addParam(obj, "filterByArea", obj.filterByArea);
		anpr_info.addParam(obj, "filterByBBArea", obj.filterByBBArea);
		anpr_info.addParam(obj, "minBBArea", obj.minBBArea);
		anpr_info.addParam(obj, "maxBBArea", obj.maxBBArea);
		anpr_info.addParam(obj, "minBBHoW", obj.minBBHoW);
		anpr_info.addParam(obj, "maxBBHoW", obj.maxBBHoW);
		anpr_info.addParam(obj, "minBBHRatio", obj.minBBHRatio);
		anpr_info.addParam(obj, "minDistBetweenBlobs", obj.minDistBetweenBlobs);
		anpr_info.addParam(obj, "useXDist", obj.useXDist);
		anpr_info.addParam(obj, "useAdaptiveThreshold", obj.useAdaptiveThreshold);
		anpr_info.addParam(obj, "nbrOfthresholds", obj.nbrOfthresholds);
		// newly add expand top, bottom, left, right
		anpr_info.addParam(obj, "nExpandTop", obj.nExpandTop);
		anpr_info.addParam(obj, "nExpandBottom", obj.nExpandBottom);
		anpr_info.addParam(obj, "nExpandLeft", obj.nExpandLeft);
		anpr_info.addParam(obj, "nExpandRight", obj.nExpandRight);

		initialized = true;
	}
	return &anpr_info;
}

bool initFTSAlgorithm() {
	cv::Ptr<cv::Algorithm> al = createFTSAlgorithm(),
		alv = createFTSVideoAlgorithm(),
		lv = createFTS_LaneViolation(),
		lvr = createFTS_LaneViolationRegion(),
		ps  = createFTS_PlateScanner(),
		rv = createFTS_RedSignalViolation(),
		rvbs = createFTS_RedSignalViolationBS(),
		vc = createFTS_VehicleCounting(),
		anpr = createFTS_ANPR();
	return al->info() != 0 && lv->info() != 0 && lvr->info() != 0 && rv->info() != 0 && vc->info() != 0 && anpr->info() != 0 && rvbs->info() !=0 && ps->info() != 0;
}

#endif