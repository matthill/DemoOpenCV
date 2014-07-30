#include <ctime>
#include "FTSANPR.h"
#include "util.h"

void FTSANPR::InitANPREngine()
{
	assert(pEngine != NULL);
#ifndef USING_SVM_OCR
	pEngine->setParams(this->m_strRuntimeOcrDir, this->m_strOcrLanguage, this->m_strPostProcessFile, this->m_strCascadeFile,
						   this->m_iMinCharToProcess,
						   this->m_iMaxCharToProcess,
						   this->m_iMinOcrFont,
						   this->m_bOcrDebug,
						   this->m_iMinCharConf,
						   this->m_iConfSkipCharLevel,
						   this->m_iMaxSubstitutions,
						   this->m_bPostProcessDebug);
#else
	pEngine->setParams(this->m_strDigitModelsDir, this->m_strLetterModelsDir, this->m_strModelPairsDir,
		this->m_strModelPairPrefix,
		this->m_strPostProcessFile, 
		this->m_strCascadeFile,
		this->m_iMinCharToProcess,
		this->m_iMaxCharToProcess,
		this->m_iMinOcrFont,
		this->m_bOcrDebug,
		this->m_iMinCharConf,
		this->m_iConfSkipCharLevel,
		this->m_iMaxSubstitutions,
		this->m_bPostProcessDebug);
#endif
	//trungnt1 add expert params
	AnprParams params;
	params.m_fLPDScaleFactor = this->m_fLPDScaleFactor;				//scale factor used in cascade.detectmultiscale
	params.m_iLPDMinNeighbors = this->m_iLPDMinNeighbors;			//min neighbor used in cascade.detectmultiscale
	params.m_iLPDMinPlateWidth = this->m_iLPDMinPlateWidth;			//min plate width
	params.m_iLPDMinPlateHeight = this->m_iLPDMinPlateHeight;		//min plate height
	params.m_iLPDMaxPlateWidth = this->m_iLPDMaxPlateWidth;			//max plate width
	params.m_iLPDMaxPlateHeight = this->m_iLPDMaxPlateHeight;		//max plate height
	params.m_fExpandRectX = this->m_fExpandRectX;					//expandRect in X-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	params.m_fExpandRectY = this->m_fExpandRectY;					//expandRect in Y-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	params.m_iTemplatePlateWidth = this->m_iTemplatePlateWidth;		//template plate after resize to recognize
	params.filterByArea = this->filterByArea;
	params.filterByBBArea = this->filterByBBArea;
	params.minBBArea = this->minBBArea;
	params.maxBBArea = this->maxBBArea;
	params.minBBHoW = this->minBBHoW;
	params.maxBBHoW = this->maxBBHoW;
	params.minBBHRatio = this->minBBHRatio;
	params.minDistBetweenBlobs = this->minDistBetweenBlobs;
	params.useXDist = this->useXDist;
	params.useAdaptiveThreshold = this->useAdaptiveThreshold;
	params.nbrOfthresholds = this->nbrOfthresholds;
	params.nExpandTop = this->nExpandTop;
	params.nExpandBottom = this->nExpandBottom;
	params.nExpandLeft = this->nExpandLeft;
	params.nExpandRight = this->nExpandRight;
	pEngine->setParamsExpert(params);
	//
	pEngine->setDebugMode(this->bDebug, this->m_bDelayOnFrame, this->m_bDisplayDbgImg);
	pEngine->setLogMode(this->m_iLogLevel, this->m_strLogFile);
	pEngine->initEngine();
}

bool FTSANPR::operator() (ViolationEvent& e) {
	BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "Start ANPR Process";
#if _MSC_VER > 1600
	BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_TRACE) << "Current ANPR Thread Id: " << std::this_thread::get_id();
#else
	BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_TRACE) << "Current ANPR Thread Id: " << boost::this_thread::get_id();
#endif
	clock_t begin = clock();

	if (!pEngine->isInitialized()) {
		BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "Init ANPR Engine";
		InitANPREngine();
	}

	Mat imgPlate, imgVehicle;
	if (e.imgPlate.empty()){
		imgPlate = e.imgOrg(e.rectBoundingBox);
	}
	else{
		imgPlate = e.imgPlate;
		pEngine->setRunPlateDetect(false);
	}
	imgVehicle = e.imgOrg(e.rectBoundingBox);
	if (imgPlate.channels() == 3) {
		cvtColor(imgPlate, imgPlate, CV_BGR2GRAY);
	}

	bool bRet = false;
	vector<AlprResult> result;
	int nLPRRes = pEngine->recognize(imgPlate, result);
	
	BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "ANPR process time: " << double(clock() - begin) / CLOCKS_PER_SEC << "s";
	if(nLPRRes == ANPR_ERR_ENGINE_NOT_INIT)
	{
		BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "ANPR ERROR ENGINE NOT INIT!!!";
		e.plate.imgPlate = imgPlate;
	}
	else if(nLPRRes == ANPR_ERR_SOURCE_NULL)
	{
		BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "ANPR ERROR NULL INPUT SOURCE!!!";
		e.plate.imgPlate = imgPlate;
	} 
	else  if(nLPRRes == ANPR_ERR_NOPLATEDETECT)
	{
		BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "WARNING: No Plate Regconized!!!";
		e.plate.imgPlate = imgPlate;
	}
	else if (nLPRRes == ANPR_ERR_OCR_NULL)
	{
		BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "WARNING: PLATE DETECTED BUT FAIL TO SEGMENT/OCR!!!";
		e.plate.imgPlate = imgPlate;
	}
	else
	{
		for(int i = 0; i < result.size(); i++)
		{
			if(result[i].bestPlate.characters.size() > 0 && result[i].bestPlate.matches_template)
			{
				BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "Plate Detected - Plate Info";
				e.plate.strPlate = result[i].bestPlate.characters;
				e.plate.fConfident = result[i].bestPlate.overall_confidence;
				if (e.rectViewPlateBB.area() > 0){
					e.plate.imgPlate = e.imgOrg(e.rectViewPlateBB);
				}
				else
					e.plate.imgPlate = imgPlate(result[i].plateRect);
				BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << e.plate.strPlate << " " << e.plate.fConfident;
				bRet = true;
				break;
			}
			else
			{
				BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "Plate Regconized But Not Match VN Plate Template";
				e.plate.strPlate = result[i].bestPlate.characters;
				e.plate.fConfident = result[i].bestPlate.overall_confidence;
				if (e.rectViewPlateBB.area() > 0){
					e.plate.imgPlate = e.imgOrg(e.rectViewPlateBB);
				}
				else
					e.plate.imgPlate = imgPlate(result[i].plateRect);
				BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << e.plate.strPlate << " " << e.plate.fConfident;
			}
		}
	}

	//out debug plate image to folder
	if (bDebug) {
		COUNTER++;
		if (COUNTER == ULONG_MAX)
			COUNTER = 0;
		time_t timer;
		//++ 03.07 use timer of ViolationEvent to sync time
		//time(&timer);
		timer = e.timer;
		//--
		std::string sSrcFolder = createDirectory(this->m_strOutDebugFolder, "RAW", this->m_strDeviceID, getDateTimeString(timer, "%Y%m%d"), 
			nLPRRes == ANPR_ERR_NOPLATEDETECT ? "SOURCE_NODETECT" : "SOURCE");
		std::string sPlateFolder = createDirectory(this->m_strOutDebugFolder, "RAW", this->m_strDeviceID, getDateTimeString(timer, "%Y%m%d"), "PLATE");
		std::string sCharSegmentsFolder = createDirectory(this->m_strOutDebugFolder, "RAW", "", "", "CHAR_SEGMENTS");
		std::string strTime = getDateTimeString(timer, "%Y%m%d_%H%M%S");
		
		std::string sImgFilePath = formatString("%sSrc_%s_%d_%ld.jpg", sSrcFolder.c_str(), strTime.c_str(), e.lEventTime, COUNTER);
		
		imwrite(sImgFilePath, imgVehicle);

		//write overview debug image and log
		if (nLPRRes > ANPR_ERR_NOPLATEDETECT && result.size() > 0) 
		{
			for(int i = 0; i < result.size(); i++)
			{
				std::string sPlateID = formatString("%d_%ld_%d.jpg", e.lEventTime, COUNTER, i);
				result[i].outputDebugInfo(this->m_strOutDebugFolder, this->m_strDeviceID, timer, sPlateID);
				//++25.06 trung add out plate image
				std::string sPlateFilePath = formatString("%sPlate_%s_%s.jpg", sPlateFolder.c_str(), strTime.c_str(), sPlateID.c_str());
				imwrite(sPlateFilePath, imgPlate(result[i].plateRect));
				//--25.06
				//++27.06 trung add code to out char segment if enabled
				if(m_bOutCharSegments)
				{
					result[i].outputCharSegmentResult(sCharSegmentsFolder);
				}
				//--
			}
		}
	}

	//output xml event
	if (bRet)
		e.process();

	return bRet;
}

FTSANPR::FTSANPR() {
	BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "Init";

	// Following init value will be override in config file - only for development
#ifdef _DEBUG
	this->m_bDelayOnFrame = false;
	this->m_bDisplayDbgImg = true;
#else
	this->m_bDelayOnFrame = false;
	this->m_bDisplayDbgImg = false;
#endif
	this->m_iLogLevel = 2;
	this->m_strLogFile = "";

	this->m_bOutCharSegments = false;

	this->m_strOutDebugFolder = ".";
#ifndef USING_SVM_OCR
	this->m_strRuntimeOcrDir = ".\\runtime_data\\ocr";
	this->m_strOcrLanguage = "lus";
#else
	this->m_strDigitModelsDir = ".\\models\\models_digits";
	this->m_strLetterModelsDir = ".\\models\\models_letters";
	this->m_strModelPairsDir = ".\\models\\models_pairs";
	this->m_strModelPairPrefix = "model";
#endif//USING_SVM_OCR
	this->m_strPostProcessFile = ".\\runtime_data\\postprocess\\vn.patterns";
	this->m_strCascadeFile = ".\\cascade\\cascade_lbp_21x40_15000_22196_unfiltered.xml";

	this->m_iMinCharToProcess = DEFAULT_MINCHAR;
	this->m_iMaxCharToProcess = DEFAULT_MAXCHAR;
	this->m_iMinOcrFont = DEFAULT_MINOCRFONT;
	this->m_bOcrDebug = DEFAULT_OCRDEBUG;

	this->m_iMinCharConf = DEFAULT_MINCHARCONF;
	this->m_iConfSkipCharLevel = DEFAULT_CONFSKIPLEVEL;
	this->m_iMaxSubstitutions = DEFAULT_MAX_SUBSTITUTIONS;
	this->m_bPostProcessDebug = DEFAULT_POSTPROCESSDEBUG;

	this->m_fLPDScaleFactor = DEFAULT_LPD_SCALEFACTOR;					//scale factor used in cascade.detectmultiscale
	this->m_iLPDMinNeighbors = DEFAULT_LPD_MINNEIGHBOR;					//min neighbor used in cascade.detectmultiscale
	this->m_iLPDMinPlateWidth = DEFAULT_LPD_MINPLATEWIDTH;				//min plate width
	this->m_iLPDMinPlateHeight = DEFAULT_LPD_MINPLATEHEIGHT;			//min plate height
	this->m_iLPDMaxPlateWidth = DEFAULT_LPD_MAXPLATEWIDTH;				//max plate width
	this->m_iLPDMaxPlateHeight = DEFAULT_LPD_MAXPLATEHEIGHT;			//max plate height
	this->m_fExpandRectX = DEFAULT_LPD_EXPAND_RECT_X;					//expandRect in X-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	this->m_fExpandRectY = DEFAULT_LPD_EXPAND_RECT_Y;					//expandRect in Y-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	this->m_iTemplatePlateWidth = DEFAULT_LPD_TEMPLATE_PLATE_WIDTH;		//template plate after resize to recognize

	this->filterByArea = false;
	this->filterByBBArea = true;
	this->minBBArea = 35;
	this->maxBBArea = 2500;
	this->minBBHoW = 0.4;
	this->maxBBHoW = 10.0;
	this->minBBHRatio = 0.125;
	this->minDistBetweenBlobs = 4.0f;
	this->useXDist = false;
	this->useAdaptiveThreshold = false;
	this->nbrOfthresholds = 5;

	this->nExpandTop = 20;
	this->nExpandBottom = 0;
	this->nExpandLeft = 10;
	this->nExpandRight = 10;
#ifndef USING_SVM_OCR //use tesseract
	pEngine = new Fts_Anpr_Engine(m_strRuntimeOcrDir, m_strOcrLanguage, m_strPostProcessFile, m_strCascadeFile);
#else
	pEngine = new Fts_Anpr_Engine(m_strDigitModelsDir, m_strLetterModelsDir, m_strModelPairsDir, m_strModelPairPrefix, m_strPostProcessFile, m_strCascadeFile);
#endif//USING_SVM_OCR
}

FTSANPR::~FTSANPR() 
{
	/*if(pEngine)
	{
		delete pEngine;
		pEngine = NULL;
	}*/
}

void FTSANPR::read(const cv::FileNode& fn) {
	this->info()->read(this, fn);

	if (!pEngine->isInitialized()) {
		BOOST_LOG_CHANNEL_SEV(lg, FTSANPR::className(), LOG_INFO) << "Init ANPR Engine";
		InitANPREngine();
	}
}

void FTSANPR::write(cv::FileStorage& fs) const {
	this->info()->write(this, fs);
}
