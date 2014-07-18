
#include "fts_anpr_engine.h"
#include "fts_anpr_rotate.h"
#include "fts_anpr_cropper.h"
#include "fts_ip_util.h"
#include "fts_anpr_util.h"

#ifndef USING_SVM_OCR
#include "fts_ip_ocr.h"
#else
#include "fts_anpr_svmocr.h"
#endif

#include "filesystem.h"

#include "fts_anpr_object.h"

#ifdef _DEBUG
#define GUI
#endif

AnprParams::AnprParams()
{
	this->reset();
}

void AnprParams::reset()
{
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

	nExpandTop = 20;
	nExpandBottom = 0;
	nExpandLeft = 10;
	nExpandRight = 10;
}

void AnprParams::operator=(const AnprParams& params)
{
	this->m_fLPDScaleFactor = (params.m_fLPDScaleFactor > 1.0f ) ? params.m_fLPDScaleFactor : DEFAULT_LPD_SCALEFACTOR;
	this->m_iLPDMinNeighbors = (params.m_iLPDMinNeighbors >=2 && params.m_iLPDMinNeighbors <= 4)? params.m_iLPDMinNeighbors : DEFAULT_LPD_MINNEIGHBOR;
	this->m_iLPDMinPlateWidth = (params.m_iLPDMinPlateWidth > 0 ) ? params.m_iLPDMinPlateWidth : DEFAULT_LPD_MINPLATEWIDTH;				//min plate width
	this->m_iLPDMinPlateHeight = (params.m_iLPDMinPlateHeight > 0 ) ? params.m_iLPDMinPlateHeight : DEFAULT_LPD_MINPLATEHEIGHT;			//min plate height
	this->m_iLPDMaxPlateWidth = (params.m_iLPDMaxPlateWidth > 0 ) ? params.m_iLPDMaxPlateWidth : DEFAULT_LPD_MAXPLATEWIDTH;				//max plate width
	this->m_iLPDMaxPlateHeight = (params.m_iLPDMaxPlateHeight > 0 ) ? params.m_iLPDMaxPlateHeight : DEFAULT_LPD_MAXPLATEHEIGHT;			//max plate height
	this->m_fExpandRectX = (params.m_fExpandRectX > 0.0f ) ? params.m_fExpandRectX : DEFAULT_LPD_EXPAND_RECT_X;
	this->m_fExpandRectY = (params.m_fExpandRectY > 0.0f ) ? params.m_fExpandRectY : DEFAULT_LPD_EXPAND_RECT_Y;
	this->m_iTemplatePlateWidth = params.m_iTemplatePlateWidth > 0 ? params.m_iTemplatePlateWidth : DEFAULT_LPD_TEMPLATE_PLATE_WIDTH;	//template plate after resize to recognize

	this->filterByArea = params.filterByArea;
	this->filterByBBArea = params.filterByBBArea;
	this->minBBArea = params.minBBArea;
	this->maxBBArea = params.maxBBArea;
	this->minBBHoW = params.minBBHoW;
	this->maxBBHoW = params.maxBBHoW;
	this->minBBHRatio = params.minBBHRatio;
	this->minDistBetweenBlobs = params.minDistBetweenBlobs;
	this->useXDist = params.useXDist;
	this->useAdaptiveThreshold = params.useAdaptiveThreshold;
	this->nbrOfthresholds = params.nbrOfthresholds;

	this->nExpandTop = params.nExpandTop;
	this->nExpandBottom = params.nExpandBottom;
	this->nExpandLeft = params.nExpandLeft;
	this->nExpandRight = params.nExpandRight;
}

//AlprResult
AlprResult::AlprResult()
{
	nErrorCode = ANPR_ERR_NONE;
	requested_topn = 0;
    result_count = 0;

	bestPlateIndex = -1;
	topNPlates.clear();

    processing_time_ms = 0.0f;
    regionConfidence = 0;
    region = "";
}

AlprResult::~AlprResult()
{
}

void AlprResult::outputDebugInfo(std::string strOutDebugFolder, std::string strDeviceID, time_t timer, std::string frameID)
{
	this->oAnprObject.write(strOutDebugFolder, strDeviceID, timer, frameID);
}

void AlprResult::outputCharSegmentResult(std::string strOutputFolder)
{
	this->oAnprObject.outputCharSegmentResult(strOutputFolder);
}

// ALPR code
#ifndef USING_SVM_OCR
Fts_Anpr_Engine::Fts_Anpr_Engine(	const std::string runtimeOcrDir, const std::string ocrLanguage, 
									const std::string patternFile, const std::string cascadeFile)
#else
Fts_Anpr_Engine::Fts_Anpr_Engine(	const std::string strDigitModelsDir,		//"./models/models_digits"
									const std::string strLetterModelsDir,		//"./models/models_letters"
									const std::string strModelPairsDir,
									const std::string strModelPairPrefix,
									const std::string patternFile, const std::string cascadeFile)
#endif
{
	this->detectRegion = DEFAULT_DETECT_REGION;
	this->m_bRunPlateDetect = DEFAULT_PLATE_DETECT;
	this->topN = DEFAULT_TOPN;
	this->defaultRegion = "";

#ifndef USING_SVM_OCR
	this->strRuntimeOcrDir = runtimeOcrDir;
	this->strOcrLanguage = ocrLanguage;
#else
	this->strDigitModelsDir = strDigitModelsDir;
	this->strLetterModelsDir = strLetterModelsDir;
	this->strPairModelsDir = strModelPairsDir;
	this->strPairModelPrefix = strModelPairPrefix;
#endif
	this->strPostProcessFile = patternFile;	
	//this->strCascadeFile = ".\\cascade\\cascade_lbp_21x40_15000_22196_unfiltered.xml";
	this->strCascadeFile = cascadeFile;

	this->m_iMinCharToProcess = DEFAULT_MINCHAR;
	this->m_iMaxCharToProcess = DEFAULT_MAXCHAR;
	this->m_iMinOcrFont = DEFAULT_MINOCRFONT;
	this->m_bOcrDebug = DEFAULT_OCRDEBUG;

	this->m_iMinCharConf = DEFAULT_MINCHARCONF;
	this->m_iConfSkipCharLevel = DEFAULT_CONFSKIPLEVEL;
	this->m_iSvmMinCharConf = 0.1;
	this->m_iSvmConfSkipCharLevel = 0.2;
	this->m_iMaxSubstitutions = DEFAULT_MAX_SUBSTITUTIONS;
	this->m_bPostProcessDebug = DEFAULT_POSTPROCESSDEBUG;

	this->m_bInit = false;
	this->bSingleLine = false;
	this->bBlackChar = true;
	this->m_bDebug = false;
	this->m_bDelayOnFrame = false;

	this->m_logLevel = ANPR_LOG_ERROR;
	this->m_strLogFileName = "";
		
	m_oParams.reset();

	oRotate = NULL;
	oCropper = NULL;
#ifndef USING_SVM_OCR
	oTessOcr = NULL;
#else
	oSvmOcr = NULL;
#endif
	plate_cascade = NULL;

	//DV: 16/06/2014
	MAX_PLATE_WIDTH  = INT_MAX;
	plateExpand.nT = 20;
	plateExpand.nB = 0;
	plateExpand.nL = 10;
	plateExpand.nR = 10;
}

Fts_Anpr_Engine::~Fts_Anpr_Engine()
{
	//delete impl;
	if(oRotate) delete oRotate;
	if(oCropper) delete oCropper;
#ifndef USING_SVM_OCR
	if(oTessOcr) 
	{
		oTessOcr->clean();
		delete oTessOcr; 
	}
#else
	if(oSvmOcr) 
	{
		oSvmOcr->clean();
		delete oSvmOcr; 
	}
#endif
	if(plate_cascade) delete plate_cascade;
}

void Fts_Anpr_Engine::resetParamsToDefault()
{
	this->detectRegion = DEFAULT_DETECT_REGION;
	this->m_bRunPlateDetect = DEFAULT_PLATE_DETECT;
	this->topN = DEFAULT_TOPN;
	this->defaultRegion = "";

	/*this->strRuntimeOcrDir = runtimeOcrDir;
	this->strOcrLanguage = ocrLanguage;
	this->strPostProcessFile = patternFile;	
	this->strCascadeFile = cascadeFile;*/

	this->m_iMinCharToProcess = DEFAULT_MINCHAR;
	this->m_iMaxCharToProcess = DEFAULT_MAXCHAR;
	this->m_iMinOcrFont = DEFAULT_MINOCRFONT;
	this->m_bOcrDebug = DEFAULT_OCRDEBUG;

	this->m_iMinCharConf = DEFAULT_MINCHARCONF;
	this->m_iConfSkipCharLevel = DEFAULT_CONFSKIPLEVEL;
	this->m_iSvmMinCharConf = 0.1;
	this->m_iSvmConfSkipCharLevel = 0.2;
	this->m_iMaxSubstitutions = DEFAULT_MAX_SUBSTITUTIONS;
	this->m_bPostProcessDebug = DEFAULT_POSTPROCESSDEBUG;	

	this->m_bInit = false;
	this->bSingleLine = false;
	this->bBlackChar = true;
	this->m_bDebug = false;
	this->m_bDelayOnFrame = false;

	this->m_logLevel = ANPR_LOG_ERROR;
	this->m_strLogFileName = "";

	m_oParams.reset();

	//DV: 16/06/2014
	MAX_PLATE_WIDTH  = INT_MAX;
	plateExpand.nT = 20;
	plateExpand.nB = 0;
	plateExpand.nL = 10;
	plateExpand.nR = 10;
}

//void Fts_Anpr_Engine::setDetectRegion(bool detectRegion)
//{
//  impl->setDetectRegion(detectRegion);
//}

void Fts_Anpr_Engine::setRunPlateDetect(bool detectPlate)
{
	this->m_bRunPlateDetect = detectPlate;
}

void Fts_Anpr_Engine::setTopN(int topN)
{
	this->topN = topN;
}

void Fts_Anpr_Engine::setSingleLine(bool singleLine)
{
	this->bSingleLine = singleLine;
}

void Fts_Anpr_Engine::setBlackChar(bool blackChar)
{
	this->bBlackChar = blackChar;
}

void Fts_Anpr_Engine::setDebugMode(bool bDebug, bool bDelayOnFrame, bool bDisplayDbgImg)
{
	this->m_bDebug = bDebug;
	this->m_bDelayOnFrame = bDelayOnFrame;
	this->m_bDisplayDbgImg = bDisplayDbgImg;
}

void Fts_Anpr_Engine::setLogMode(int level, std::string destName)
{
	this->m_logLevel = level;
	this->m_strLogFileName = destName;
}

bool Fts_Anpr_Engine::setParams(
#ifndef USING_SVM_OCR
								const std::string runtimeOcrDir, 
								const std::string ocrLanguage, 
#else
								const std::string digitModelsDir, 
								const std::string letterModelsDir, 
								const std::string pairModelsDir,
								const std::string pairModelPrefix,
#endif
								const std::string patternFile, 
								const std::string cascadeFile,
								int iMinCharToProcess,
								int iMaxCharToProcess,
								int iMinOcrFont,
								bool bOcrDebug,
								int iMinCharConf,
								int iConfSkipCharLevel,
								int iMaxSubstitutions,
								bool bPostProcessDebug)
{	
#ifndef USING_SVM_OCR
	if(!DirectoryExists(runtimeOcrDir.c_str()))
	{
		cout << "[setParams] Could not find Runtime Ocr Directory!!! " << runtimeOcrDir << endl;
		return false;
	}
	this->strRuntimeOcrDir = runtimeOcrDir;

	string ocrTrainFilePath = this->strRuntimeOcrDir;
	if(this->strRuntimeOcrDir[this->strRuntimeOcrDir.size()-1] != '\\')
		ocrTrainFilePath.append("\\");
	ocrTrainFilePath.append("tessdata\\" + ocrLanguage + ".traineddata");
	if(!fileExists(ocrTrainFilePath.c_str()))
	{
		cout << "[setParams] Could not find Tesseract Trained File!!! " << ocrTrainFilePath << endl;
		return false;
	}
	this->strOcrLanguage = ocrLanguage;
#else
	if(!DirectoryExists(digitModelsDir.c_str()))
	{
		cout << "[setParams] Could not find Digit SVM Model Directory!!! " << digitModelsDir << endl;
		return false;
	}
	this->strDigitModelsDir = digitModelsDir;

	if(!DirectoryExists(letterModelsDir.c_str()))
	{
		cout << "[setParams] Could not find Letter SVM Model Directory!!! " << letterModelsDir << endl;
		return false;
	}
	this->strLetterModelsDir = letterModelsDir;

	if (!DirectoryExists(pairModelsDir.c_str()))
	{
		cout << "[setParams] Could not find model-pair SVM Model Directory!!! " << pairModelsDir << endl;
		return false;
	}
	this->strPairModelsDir = pairModelsDir;

	this->strPairModelPrefix = pairModelPrefix;
	
#endif
	
	if(!fileExists(patternFile.c_str()))
	{
		cout << "[setParams] Could not find Post Process Patterns File!!! " << patternFile << endl;
		return false;
	}
	this->strPostProcessFile = patternFile;	
	
	if(!fileExists(cascadeFile.c_str()))
	{
		cout << "[setParams] Could not find Cascade File!!! " << strCascadeFile << endl;
		return false;
	}
	this->strCascadeFile = cascadeFile;

	this->m_iMinCharToProcess = iMinCharToProcess;
	this->m_iMaxCharToProcess = iMaxCharToProcess;
	this->m_iMinOcrFont = iMinOcrFont;
	this->m_bOcrDebug = bOcrDebug;

	this->m_iMinCharConf = iMinCharConf;
	this->m_iConfSkipCharLevel = iConfSkipCharLevel;
	this->m_iMaxSubstitutions = iMaxSubstitutions;
	this->m_bPostProcessDebug = bPostProcessDebug;

	return true;
}

bool Fts_Anpr_Engine::setParamsExpert(const AnprParams& params)
{
	m_oParams = params;
	return true;
}

bool Fts_Anpr_Engine::isLoaded()
{
  return m_bInit;
}

bool Fts_Anpr_Engine::initEngine()
{
	oRotate = new FTS_ANPR_Rotate();
	oCropper = new FTS_ANPR_Cropper();
#ifdef USING_SVM_OCR	
	oSvmOcr = new FTS_ANPR_SvmOcr(  this->strDigitModelsDir,	//model folder paths
									this->strLetterModelsDir,	//model folder paths
									this->strPairModelsDir,
									this->strPairModelPrefix,
									this->m_iMinCharToProcess,	//min char to process									
									true	//enable console debug
									);
	oSvmOcr->initPostProcess(this->strPostProcessFile, //file patterns	
							 this->m_iSvmMinCharConf,			//min confidence level
							 this->m_iSvmConfSkipCharLevel,	//confidence skip level
							 this->m_iMaxSubstitutions,		//max substistutions
							 this->m_bPostProcessDebug		//enable console debug
							 );
#else	
	oTessOcr = new FTS_ANPR_TessOcr(this->strRuntimeOcrDir,		//parent dir to load tessdata	
									this->strOcrLanguage,		//language
									this->m_iMinCharToProcess,	//min char to process
									this->m_iMinOcrFont,		//min font point
									this->m_bOcrDebug			//enable console debug
									);
	oTessOcr->initPostProcess(this->strPostProcessFile,		//file patterns	
							 this->m_iMinCharConf,			//min confidence level
							 this->m_iConfSkipCharLevel,	//confidence skip level
							 this->m_iMaxSubstitutions,		//max substistutions
							 this->m_bPostProcessDebug		//enable console debug
							 );
#endif

	plate_cascade = new CascadeClassifier();
#ifndef WIN32
	if( !plate_cascade->load( "/home/sensen/data/cascade/cascade_haar_21x40_15000_22196_unfiltered.xml" ) )
#else
	if( !plate_cascade->load( this->strCascadeFile ) )
#endif
	{
		printf("--(!)Error loading cascade\n");
		m_bInit = false;
		return false;
	};

	m_bInit = true;

	return true;
}

//void Test_LSD(IplImage* img)
void Fts_Anpr_Engine::Test_LSD( const Mat& img, Mat& oLinesImg )
{
    //IplImage* grey = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    //cvCvtColor(img, grey, CV_BGR2GRAY);
	Mat grey(img.rows,img.cols, CV_8UC1);
	if(img.type() == CV_8UC1)
		grey = img.clone();
	else
		cvtColor(img, grey, CV_BGR2GRAY);
    //image_double image;
    //ntuple_list out;
	double* image;
	double* out;
    int x,y,i,n;
    //image = new_image_double(img->width,img->height);
	image = (double *) malloc( img.cols * img.rows * sizeof(double) );
    for(x=0;x<grey.cols;x++)
    for(y=0;y<grey.rows;y++)
    {
      //CvScalar s= cvGet2D(grey,y,x); 
	  Scalar s = grey.at<uchar>(y,x);
      double pix = s.val[0];
      image[ x + y * img.cols ]= pix; /* image(x,y) */
    }

    /* call LSD */
    out = lsd(&n, image, img.cols, img.rows);

    /* print output */
    if(m_bDebug) printf("%u line segments found:\n",n);
    vector<Line> vec_lines;
    for(i=0;i<n;i++)
    {
      //for(j=0;j<out->dim;j++)
      {
          Line line;
          line.x1= out[ i * 7 + 0];
          line.y1= out[ i * 7 + 1];
          line.x2= out[ i * 7 + 2];
          line.y2= out[ i * 7 + 3];
          vec_lines.push_back(line);
      }
    }

	oLinesImg = Mat::zeros( grey.size(), grey.type() );
    for( size_t i = 0; i < vec_lines.size(); ++i )
    {
		line( oLinesImg, Point(vec_lines[i].x1,vec_lines[i].y1),
						 Point(vec_lines[i].x2,vec_lines[i].y2),
						 Scalar(255,255,255), 1, 0 );
    }

	free((void *)image);
	free((void *)out);
}

int Fts_Anpr_Engine::recognize(std::string filepath, std::vector<AlprResult>& result)
{
	if(!m_bInit)
	{
		cout << "FTS_Anpr_Engine is not initialized! Please setParams and call initEngine!!!" << endl;
		return ANPR_ERR_ENGINE_NOT_INIT;
	}

	cv::Mat oSrc = cv::imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
	if( oSrc.empty() )
	{
		return ANPR_ERR_SOURCE_NULL;
	}
	result.clear();
	return this->recognize(oSrc, result);
}

int Fts_Anpr_Engine::recognize(const cv::Mat& oSrc, std::vector<AlprResult>& result)
{
	if(!m_bInit)
	{
		cout << "FTS_Anpr_Engine is not initialized! Please setParams and call initEngine!!!" << endl;
		return ANPR_ERR_ENGINE_NOT_INIT;
	}

	//-- Detect plates
	std::vector<Rect> plates;
	long long lPlateDetectTime=0;
	this->plateDetect(oSrc, plates, lPlateDetectTime);
	if(plates.size() == 0)
	{
		return ANPR_ERR_NOPLATEDETECT;
	}

	// DV: 14/07/2014 - use this to do segmentation on the original image
//	plates.clear();
//	plates.push_back( Rect( 0, 0, oSrc.cols, oSrc.rows ) );
	
	for (size_t i = 0; i < plates.size(); i++)
	{
		//FTS_ANPR_OBJECT oAnprObject; //debug engine object		
		AlprResult plateResult;
		plateResult.oAnprObject.setLogLevel(this->m_logLevel);
		plateResult.oAnprObject.lPlateDetectTime = lPlateDetectTime;
		plateResult.nErrorCode = ANPR_ERR_SEGMENT_FAIL;
		
		//SEGMENT
		Rect expandRect;
		bool bSegmentSuccess = this->plateSegment(oSrc, plates[i], expandRect, plateResult.oAnprObject);

		//OCR
		if(bSegmentSuccess)
		{
			this->plateOcr( oSrc,
					        expandRect,
					        plateResult.oAnprObject.oBestCharBoxes,
					        plateResult.oAnprObject.oBestBinImages,
					        plateResult );
			result.push_back( plateResult );
			plateResult.nErrorCode = plateResult.bestPlateIndex>=0 ? ANPR_ERR_OCR_SUCCESS : ANPR_ERR_OCR_NOMATCH; 
		}
		else
		{
			plateResult.nErrorCode = ANPR_ERR_SEGMENT_FAIL; 
		}
		
		//print to console info
		plateResult.oAnprObject.printf(this->m_logLevel);
	}

	//return result.size() > 0;
	if(result.size() == 0)
	{
		return ANPR_ERR_OCR_NULL;
	}
	else
	{
		int ret = ANPR_ERR_OCR_NULL;
		for(size_t i = 0; i < result.size(); i++)
		{
			if(result[i].nErrorCode == ANPR_ERR_OCR_SUCCESS)
				return ANPR_ERR_OCR_SUCCESS;
			else
				ret = std::max(ret, result[i].nErrorCode);
		}
		return ret;
	}
}

void Fts_Anpr_Engine::plateDetect(const cv::Mat& oSrc, std::vector<Rect>& plates, long long& lPlateDetectTime)
{
	lPlateDetectTime = 0;
	if(m_bRunPlateDetect)
	{
		long long start = getCurrentTimeInMS();
		plateDetectMutex.lock();
		plate_cascade->detectMultiScale( oSrc,
										plates,
										this->m_oParams.m_fLPDScaleFactor,
										this->m_oParams.m_iLPDMinNeighbors,
										0 | CV_HAAR_SCALE_IMAGE,
										Size(m_oParams.m_iLPDMinPlateWidth, m_oParams.m_iLPDMinPlateHeight),
										Size(m_oParams.m_iLPDMaxPlateWidth, m_oParams.m_iLPDMaxPlateHeight));
		plateDetectMutex.unlock();

		//++trungnt1 add merge intersect plate
		if(plates.size() > 1)
		{
			plates = FTS_BASE_Util::CheckAndMergeOverlapRects(plates);
		}
		//--

		long long end = getCurrentTimeInMS();
		lPlateDetectTime = end - start;

		//printf("Time to detect plate regions = %lld ms\n", lPlateDetectTime);

		cv::Mat oSrcColor;
		cv::cvtColor(oSrc, oSrcColor, CV_GRAY2BGR);
		for (size_t i = 0; i < plates.size(); i++)
		{
			// Expand rect
			Rect oExpandedRect;

#ifdef FIRST_EXPAND_BY_FIXED_PIXELS
			oExpandedRect = FTS_IP_Util::expandRectTBLR( plates[i], plateExpand, oSrc.cols - 1, oSrc.rows - 1);
#elif FIRST_EXPAND_BY_RATIO
			int expandX = round(plates[i].width * 0.5);
			int expandY = round(plates[i].height * 0.1);
			oExpandedRect = FTS_IP_Util::expandRectXY(plates[i], expandX, expandY, oSrc.cols - 1, oSrc.rows - 1);
#else
			oExpandedRect = plates[i];
#endif
			cv::rectangle(oSrcColor, oExpandedRect.tl(), oExpandedRect.br(), CV_RGB(255, 0, 0));
		}
		if(m_bDisplayDbgImg) FTS_GUI_DisplayImage::ShowAndScaleBy2( "Detect Plate Region", oSrcColor, 1.0, 1.0, 0, 0 );
	}
	else
	{
		plates.push_back(Rect(0, 0, oSrc.cols, oSrc.rows));
	}

	printf( "There are %d plates detected in the image\n", plates.size() );
}

bool Fts_Anpr_Engine::plateSegment( const cv::Mat& oSrc, 
									const Rect& plateRect,
									Rect& oExpandedRect,
									FTS_ANPR_OBJECT& oAnprObject)
{
	int nPadded = 1;
	long long start = getCurrentTimeInMS();

	// Default crop
#ifdef FIRST_EXPAND_BY_FIXED_PIXELS
	oExpandedRect = FTS_IP_Util::expandRectTBLR( plateRect, plateExpand, oSrc.cols - 1, oSrc.rows - 1);
#elif FIRST_EXPAND_BY_RATIO
	int expandX = round(plates[i].width * 0.5);
	int expandY = round(plates[i].height * 0.1);
	oExpandedRect = FTS_IP_Util::expandRectXY(plates[i], expandX, expandY, oSrc.cols - 1, oSrc.rows - 1);
#else
	oExpandedRect = plates[i];
#endif
	oAnprObject.oPlate = oSrc( oExpandedRect );

	// Crop then expand if enabled
#ifdef CROP_THEN_EXPAND
	CvRect oCroppedRect;
	if( oCropper->processDetection( oAnprObject.oPlate , oCroppedRect ) )
	{
		Rect oCroppedRectOriginal( oExpandedRect );
			 oCroppedRectOriginal.x += oCroppedRect.x;
			 oCroppedRectOriginal.y += oCroppedRect.y;
			 oCroppedRectOriginal.width  = oCroppedRect.width;
			 oCroppedRectOriginal.height = oCroppedRect.height;

		Rect oExpandedRectFromCrop( oCroppedRectOriginal);

		// DV: 16/06/2014 - do not expand if the width is already
		// bigger than max plate width
		// By default, disable this feature by setting MAX_PLATE_WIDTH = INT_MAX
		if( oCroppedRect.width < MAX_PLATE_WIDTH )
		{
			// TODO: DV: 16/06/2014 - Settings if enabled
			FTS_IP_Util::ExpandByPixels expLocal;
			expLocal.nT = 5;
			expLocal.nB = 5;
			expLocal.nL = 5;
			expLocal.nR = 5;

			oExpandedRectFromCrop = FTS_IP_Util::expandRectTBLR( oCroppedRectOriginal,
																 expLocal,
																 oSrc.cols - 1,
																 oSrc.rows - 1 );
		}

//		if(m_bDebug && m_bDisplayDbgImg)
//		{
//			FTS_GUI_DisplayImage::ShowAndScaleBy2( "Cropped", oAnprObject.oPlate( oCroppedRect ), 1.0, 1.0, 0, 0 );
//			FTS_GUI_DisplayImage::ShowAndScaleBy2( "ExpandFromCrop", oSrc( oExpandedRectFromCrop ), 1.0, 1.0, 0, 0 );
//		}

		oAnprObject.oPlate = oSrc( oExpandedRectFromCrop );
	}
#endif

	// Scale to fixed width & height
	// Also adjust cropped X coordinates
	int FINAL_PLATE_WIDTH = this->m_oParams.m_iTemplatePlateWidth;
	resize( oAnprObject.oPlate, oAnprObject.oPlate, Size( FINAL_PLATE_WIDTH, FINAL_PLATE_WIDTH*oAnprObject.oPlate.rows/oAnprObject.oPlate.cols ) );

	// Crop
	cv::Mat oCropped = oAnprObject.oPlate.clone();

	// Detect blobs
	FTS_IP_SimpleBlobDetector::Params params;
	FTS_IP_SimpleBlobDetector myBlobDetector(params);
	params.filterByArea = false;
	params.filterByBBArea = true;
	params.minBBArea = 35;
	params.maxBBArea = 2500;
	params.minBBHoW = 0.4;
	params.maxBBHoW = 10.0;
	params.minBBHRatio = 0.125;
	params.minDistBetweenBlobs = 4.0f;
	params.useXDist = false;
	params.useAdaptiveThreshold = false;
	params.bDebug = m_bDebug;
	params.bDisplayDbgImg = m_bDisplayDbgImg;
	myBlobDetector.updateParams( params );

	// DV 23/06/2014: refer to ANPR object
	myBlobDetector.m_poANPRObject = &oAnprObject;

	// Rotate image
	myBlobDetector.rotate( oCropped, bBlackChar, Mat(), -nPadded, -nPadded, oAnprObject.oSrcRotated );

	// SEGMENTATION
	//==============================================================================
	bool bSegmentSuccess;
	bool bUseLocalOtsu = true;
	if( bSingleLine )
	{
		cv::Mat oMask;
		oMask.create( oAnprObject.oSrcRotated.size(), oAnprObject.oSrcRotated.type() );
		oMask = cv::Scalar(255);
		bSegmentSuccess = doSegment( oAnprObject.oSrcRotated,
					  oMask,
					  bSingleLine,
					  bBlackChar,
					  INT_MAX,
					  INT_MIN,
					  vector<FTS_IP_SimpleBlobDetector::SimpleBlob>(),
					  myBlobDetector.m_voBinarizedImages,
					  MAX_NUM_OF_SINGLE_LINE,
					  oAnprObject,
					  bUseLocalOtsu );
	}
	else
	{
		// DV: this function not just find middle cut, it finds:
		// 1. all candidate blobs
		// 2. the max number of blobs on a single line
		// 3. min, max x of all blobs
		vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oTopBlobs, oBottomBlobs;
		Mat oTopMask, oBottomMask;
		int nMinX = INT_MAX, nMaxX = INT_MIN;
		int nMaxBlobsPerLine;
		myBlobDetector.m_poANPRObject = &oAnprObject;
		int nCut = myBlobDetector.findMiddleCut( oAnprObject.oSrcRotated,
												 true,
												 Mat(),
												 -nPadded,
												 -nPadded,
												 oTopMask,
												 oBottomMask,
												 oTopBlobs,
												 oBottomBlobs,
												 nMinX, nMaxX, nMaxBlobsPerLine );

		if(m_bDebug)
		{
			oAnprObject.oDebugLogs.info( "Found nCut = %d", nCut );
			oAnprObject.oDebugLogs.info( "Found %d top blobs, %d bottom blobs", oTopBlobs.size(), oBottomBlobs.size() );
			oAnprObject.oDebugLogs.info( "Found min x = %d, max x = %d", nMinX, nMaxX );
		}

		//try read upper part
		bUseLocalOtsu = false;	// DV: 01/07/2014 - local otsu is not good for top line
		bSegmentSuccess = doSegment( oAnprObject.oSrcRotated,
					  oTopMask,
					  bSingleLine,
					  bBlackChar,
					  nMinX,
					  nMaxX,
					  oTopBlobs,
					  myBlobDetector.m_voBinarizedImages,
					  MAX_NUM_OF_DUAL_LINE_TOP,
					  oAnprObject,
					  bUseLocalOtsu );

		//try read lower part
		bUseLocalOtsu = true;	// DV: 01/07/2014 - local otsu is ok for bottom line
		bSegmentSuccess |= doSegment( oAnprObject.oSrcRotated,
					  oBottomMask,
					  bSingleLine,
					  bBlackChar,
					  nMinX,
					  nMaxX,
					  oBottomBlobs,
					  myBlobDetector.m_voBinarizedImages,
					  MAX_NUM_OF_DUAL_LINE_BOTTOM,
					  oAnprObject,
					  bUseLocalOtsu );
	}

	long long end = getCurrentTimeInMS();
	oAnprObject.lPlateSegmentTime = end - start;

	return bSegmentSuccess;
}

void Fts_Anpr_Engine::plateOcr(const cv::Mat& oSrc, 
							   const Rect& plateRect,
							   const vector<Rect>& oBestCharBoxes,
							   const vector<Mat>& oBestBinImages,
							   AlprResult& plateResult)
{
	long long startOcr = getCurrentTimeInMS();
	//AlprResult plateResult;
	//plateResult.region = dispatcher->defaultRegion;
	plateResult.regionConfidence = 0;
	plateResult.plateRect = plateRect;
	plateResult.oAnprObject.plateRect = plateRect;

	//DV add
	// Get exact boundaries
	vector< vector<Rect> > charRegions2D = getExactCharBB(oBestBinImages, oBestCharBoxes );

	// DV: 16/06/2014 - Find short, mostly full, mostly empty blobs by going through all binary images
	vector< vector<Rect> > charRegionsFinal2D = getCorrectSizedCharRegions( oBestBinImages,
																			oBestCharBoxes,
																			charRegions2D,
																			plateResult.oAnprObject );

	// 21.06 Trung
	// DV: 23/06/2014 - copy exact bounding boxes
	plateResult.oAnprObject.charRegions2D = charRegionsFinal2D;

	//Tesseract API
	//ocrMutex.lock();
	{
		boost::mutex::scoped_lock lock(ocrMutex);
		//vector<OcrResult> ocrResults;
		plateResult.oAnprObject.ocrResults.clear();
#ifndef USING_SVM_OCR
		oTessOcr->performOCR(oBestBinImages, plateResult.oAnprObject.charRegions2D, plateResult.oAnprObject);
		oTessOcr->clean();
#else
		oSvmOcr->performOCR(plateResult.oAnprObject.oSrcRotated, plateResult.oAnprObject.oBestCharBoxes, plateResult.oAnprObject);
		oSvmOcr->clean();
#endif
	}
	//ocrMutex.unlock();
	
	//int topN = 10;
	//oTessOcr->postProcessor->analyze("base", topN);
	plateResult.oAnprObject.ppResults.clear();
#ifndef USING_SVM_OCR
	oTessOcr->postProcessor->analyze("base", topN, plateResult.oAnprObject);
#else
	oSvmOcr->postProcessor->analyze("base", topN, plateResult.oAnprObject);
#endif
	long long endOcr = getCurrentTimeInMS();	
	plateResult.oAnprObject.lPlateOcrTime = endOcr - startOcr;

	const vector<FTS_ANPR_PPResult> ppResults = plateResult.oAnprObject.getResults();
	plateResult.bestPlateIndex = -1;
	for (unsigned int pp = 0; pp < ppResults.size(); pp++)
	{
		if( (int)pp >= topN )
		{
			break;
		}		
		if ((int)ppResults[pp].letters.size() >= this->m_iMinCharToProcess &&
			(int)ppResults[pp].letters.size() <= this->m_iMaxCharToProcess)
		{
			AlprPlate aplate;
			aplate.characters = ppResults[pp].letters;
			aplate.overall_confidence = ppResults[pp].totalscore;
			aplate.matches_template = ppResults[pp].matchesTemplate;
			plateResult.topNPlates.push_back(aplate);
			if (plateResult.bestPlateIndex == -1 && ppResults[pp].matchesTemplate)
			{
				plateResult.bestPlateIndex = plateResult.topNPlates.size() - 1;
				plateResult.bestPlate = plateResult.topNPlates[plateResult.bestPlateIndex];
			}
		}
	}
	plateResult.result_count = plateResult.topNPlates.size();
    
	if(plateResult.topNPlates.size() > 0 && plateResult.bestPlateIndex >= 0)
	{
		plateResult.nErrorCode = ANPR_ERR_OCR_SUCCESS;
		plateResult.oAnprObject.oDebugLogs.none( "\nBest OCR = %s, confidence=%f, MATCHED",
			plateResult.bestPlate.characters.c_str(), plateResult.bestPlate.overall_confidence);		
	}
	else if(plateResult.topNPlates.size() > 0)
	{
		plateResult.bestPlate = plateResult.topNPlates[0];
		plateResult.nErrorCode = ANPR_ERR_OCR_NOMATCH;
		plateResult.oAnprObject.oDebugLogs.none( "\nBest OCR = %s, confidence=%f, UNMATCHED",
			plateResult.topNPlates[0].characters.c_str(), plateResult.topNPlates[0].overall_confidence);		
	}
	else
	{
		plateResult.nErrorCode = ANPR_ERR_OCR_NULL;
		plateResult.oAnprObject.oDebugLogs.none( "\nBest OCR = """);
	}

	if(m_bDebug)
	{
		//plateResult.oAnprObject.ppResults = ppResults;
		plateResult.oAnprObject.createOverviewDebugImg();
		if(m_bDisplayDbgImg) FTS_GUI_DisplayImage::ShowAndScaleBy2( "Overview Debug ANPR Image", plateResult.oAnprObject.oOverviewDebugImg_Resized, 1.0, 1.0, 0, 0 );
	}
}

bool Fts_Anpr_Engine::doSegment( const cv::Mat& oSrc,
							     const cv::Mat& oMask,
							     const bool bSingleLine,
							     const bool bBlackChar,
							     const int nInputMinX,		// soft bound by cropping
							     const int nInputMaxX,		// soft bound by cropping
							     const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oInputBlobs,
							     vector<Mat>& oBinaryImages,
							     const int nMaxNbrOfChar,
							     FTS_ANPR_OBJECT& oAnprObject,
								 bool bUseLocalOtsu )
{
#ifdef TRY_LSD	
	Test_LSD( oSrc, oAnprObject.oLinesImg );
#endif

	int nPaddedBorder = 1;

	cv::Mat oSrcRotated = oSrc.clone();

	// Contours work better on padded images
	cv::Mat srcPadded;
	cv::copyMakeBorder( oSrcRotated,
						srcPadded,
						nPaddedBorder,
						nPaddedBorder,
						nPaddedBorder,
						nPaddedBorder,
						IPL_BORDER_CONSTANT, bBlackChar?Scalar(255):Scalar(0)  );

	// Threshold by OTSU
	const int nThresholdType = bBlackChar ? CV_THRESH_BINARY_INV : CV_THRESH_BINARY;
#ifdef TEST_MINMAX_OTSU
	cv::Mat oBinByOtsu;
	cv::Mat minOtsuImg;
	cv::Mat maxOtsuImg;	
	double rOtsu = cv::threshold( oSrcRotated, oBinByOtsu, 0, 255, nThresholdType | CV_THRESH_OTSU );

	double rMinOtsu = rOtsu - 40;
	double rMaxOtsu = rOtsu + 20;
	cv::threshold( oSrcRotated, minOtsuImg, rMinOtsu, 255, nThresholdType );
	cv::threshold( oSrcRotated, maxOtsuImg, rMaxOtsu, 255, nThresholdType );
#endif

	// Detect blobs
	FTS_IP_SimpleBlobDetector::Params params;
	params.bDebug = m_bDebug;
	params.bDisplayDbgImg = m_bDisplayDbgImg;
	FTS_IP_SimpleBlobDetector myBlobDetector(params);
	// DV 23/06/2014: refer to ANPR object
	myBlobDetector.m_poANPRObject = &oAnprObject;
	vector<FTS_IP_SimpleBlobDetector::SimpleBlob> blobs = oInputBlobs;
	if(m_bDebug) oAnprObject.oDebugLogs.info( "INPUT BLOBS = %d", blobs.size() );
	if( oInputBlobs.size() == 0 )
	{
		// Detect blobs using bounding boxes
		params.filterByArea = false;
		params.filterByBBArea = true;
		params.minBBArea = 35;
		params.maxBBArea = 2500;
		params.minBBHoW = 0.35;
		params.maxBBHoW = 10.0;
		params.minBBHRatio = bSingleLine ? 0.15 : 0.125;
		params.minDistBetweenBlobs = 4.0f;

		// Use adaptive
		params.useAdaptiveThreshold = false;
		params.nbrOfthresholds = 5;
		params.useXDist = false;

		// DV: 16/06/2014
		// remove long lines with hope to find blobs
		if( !bSingleLine )
		{
			params.removeLongLine = true;
			params.longLineLengthRatio = (nMaxNbrOfChar != 0 ) ? 1.0 / nMaxNbrOfChar : 0.5;	// TODO: can be better???
		}

		myBlobDetector.updateParams( params );

		cv::Mat oFirstMaskPadded;
		oFirstMaskPadded.create( oMask.rows + 2*nPaddedBorder, oMask.cols + 2*nPaddedBorder, oMask.type() );
		cv::copyMakeBorder( oMask,
							oFirstMaskPadded,
							nPaddedBorder,
							nPaddedBorder,
							nPaddedBorder,
							nPaddedBorder,
							IPL_BORDER_CONSTANT, cv::Scalar(0) );
		myBlobDetector.detectFTS( srcPadded,
								  blobs,
								  bBlackChar,
								  oFirstMaskPadded,
								 -nPaddedBorder,
								 -nPaddedBorder );

		// DV: 21/06/2014
		// In case of single line, binary images passed in this function
		// are before rotation, so we have to update them
		// DV: 04/07/2014 - copy only the top or bottom half
		// this is to fix the occasionally top or bottom half is fully black

		for( size_t i = 0; i < myBlobDetector.m_voBinarizedImages.size(); i++ )
		{
			// DV: 12/07/2014: fixed copyTo() error of mis-matching size, replaced oMask by oFirstMaskPadded
			myBlobDetector.m_voBinarizedImages[i].copyTo( oBinaryImages[i], oFirstMaskPadded );
		}
		oAnprObject.rawBins = oBinaryImages;
	}

	if( blobs.size() == 0 )
	{
		oAnprObject.oDebugLogs.error( "NO BLOB IS DETECTED AT FIRST ATTEMPT, SKIP" );
		/*FTS_GUI_DisplayImage::ShowAndScaleBy2( "Source", oSrc,
					FTS_GUI_DisplayImage::SCALE_X,
					FTS_GUI_DisplayImage::SCALE_Y,0, 0 );*/			//no need to display here
		if(m_bDelayOnFrame) waitKey(0);
		return false;
	}

	// DV: 01/07/2014 - if not using local otsu, replace it by the adaptive block = 19
	Mat oLocalOtsuSubstitue = oBinaryImages[4];	// TODO: only good if adaptive is good

	// SORT BY X COORDINATE
	// =====================================================================
	sort( blobs.begin(), blobs.end(), FTS_IP_SimpleBlobDetector::less_than_x_coord() );

	// MEDIAN WIDTH, HEIGHT, OTSU
	// =====================================================================
	// TODO DV: ONLY consider blobs within the cropped X range
	if(m_bDebug)
	{
		oAnprObject.oDebugLogs.info( "nInputMinX = %d, nInputMaxX = %d", nInputMinX, nInputMaxX );
	}
	int nValidMinX = ( nInputMinX == INT_MAX ) ? 0 : nInputMinX;
	int nValidMaxX = ( nInputMaxX == INT_MIN ) ? oSrcRotated.cols - 1 : nInputMaxX;
	vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobsWithinCroppedXRange = getBlobsInXRange( blobs,
																								nValidMinX,
																								nValidMaxX );

	if( oBlobsWithinCroppedXRange.size() == 0 )
	{
		oAnprObject.oDebugLogs.warn( "NO BLOB IS DETECTED WITHIN MINX = %d, MAXX = %d", nValidMinX, nValidMaxX );
		return false;
	}

	// DV: 14/07/2014 - 1,I, or thin characters should not contribute to the median width
	float rMinWoH = 0.25;	// TODO: i think these values are safe.
	float rMaxWoH = 0.8;
	int nMinMedianWidth = 0;

//	FTS_BASE_STACK_ARRAY( int, oBlobsWithinCroppedXRange.size(), oWidths );
//	fillBlobWidthArray( oWidths, oBlobsWithinCroppedXRange );
//	oAnprObject.nMedianBlobWidth = FTS_BASE_MedianBiasHigh( oWidths );
	oAnprObject.nMedianBlobWidth = FTS_ANPR_Util::findMedianBlobWidthOfWoHInRange( oBlobsWithinCroppedXRange, rMinWoH, rMaxWoH, nMinMedianWidth );

//	FTS_BASE_STACK_ARRAY( int, oBlobsWithinCroppedXRange.size(), oHeights );
//	fillBlobHeightArray( oHeights, oBlobsWithinCroppedXRange );
//	oAnprObject.nMedianBlobHeight = FTS_BASE_MedianBiasHigh( oHeights );
	oAnprObject.nMedianBlobHeight = FTS_ANPR_Util::findMedianBlobHeight( oBlobsWithinCroppedXRange, nMinMedianWidth );

	if(m_bDebug)
	{
		oAnprObject.oDebugLogs.info( "Median blob width  = %d, height = %d", oAnprObject.nMedianBlobWidth, oAnprObject.nMedianBlobHeight );		
	}

	// DV: 01/07/2014
	// The idea is to again binarize the image with the optimal median otsu value
	// However in case the plate has shadow( mostly motorbike ), the otsu does
	// not give a good binary image, so replace it with an adaptive alternative
	if( bUseLocalOtsu )
	{
		FTS_BASE_STACK_ARRAY( double, oBlobsWithinCroppedXRange.size(), oOtsus );
		fillBlobOtsuArray( oOtsus, oBlobsWithinCroppedXRange, oSrcRotated );
		oAnprObject.rMediaBlobOtsu = FTS_BASE_Median( oOtsus );
		if(m_bDebug)
		{
			oAnprObject.oDebugLogs.info( "Median otsu        = %f", oAnprObject.rMediaBlobOtsu  );
	#ifdef TEST_MINMAX_OTSU
			oAnprObject.oDebugLogs.info( "rOtsu = %f", rOtsu );
	#endif
		}

		// Binarize with the new otsu
		cv::threshold( oSrcRotated, oAnprObject.oMedOtsuThreshBinImg, oAnprObject.rMediaBlobOtsu, 255, nThresholdType );
	}
	else
	{
		//++04.07 trungnt1 modify to apply on original scale, not padded border
		cv::Rect rSrcRect(nPaddedBorder, nPaddedBorder, oSrcRotated.cols, oSrcRotated.rows);
		oAnprObject.oMedOtsuThreshBinImg = oLocalOtsuSubstitue(rSrcRect);
		//--
	}

	// TOP & BOTTOM LINES
	// =====================================================================
	if(m_bDebug) oAnprObject.oDebugLogs.info( "TOP & BOTTOM LINES" );

	// DV: ONLY blobs within the horizontal cropped regions will be considered
	// TODO: remove if not good for single-line plate
	vector<bool> goodIndices;
	for( unsigned int i = 0; i < oBlobsWithinCroppedXRange.size(); i++ )
	{
		goodIndices.push_back( true );
	}
	vector<Point> oPlateBoundingPolygon = myBlobDetector.getBoundingPolygonFromBlobs(
			oSrcRotated.cols, oSrcRotated.rows, oBlobsWithinCroppedXRange, goodIndices );
//	vector<bool> goodIndices;
//	for( unsigned int i = 0; i < blobs.size(); i++ )
//	{
//		goodIndices.push_back( true );
//	}
//	vector<Point> oPlateBoundingPolygon = myBlobDetector.getBoundingPolygonFromBlobs(
//			oSrcRotated.cols, oSrcRotated.rows, blobs, goodIndices );

	FTS_BASE_LineSegment oTopLine( oPlateBoundingPolygon[0].x,
								   oPlateBoundingPolygon[0].y,
								   oPlateBoundingPolygon[1].x,
								   oPlateBoundingPolygon[1].y );

	FTS_BASE_LineSegment oBottomLine( oPlateBoundingPolygon[2].x,
									  oPlateBoundingPolygon[2].y,
									  oPlateBoundingPolygon[3].x,
									  oPlateBoundingPolygon[3].y );

	// Mask the region between top and bottom lines
	Mat oTBLinesMask = Mat::zeros( oSrcRotated.size(), CV_8U );
	fillConvexPoly( oTBLinesMask, oPlateBoundingPolygon.data(), oPlateBoundingPolygon.size(), Scalar(255,255,255));

	//cv::Mat oLines;
	cv::cvtColor( oSrcRotated, oAnprObject.oLines, CV_GRAY2BGR );
	cv::line( oAnprObject.oLines, oPlateBoundingPolygon[0], oPlateBoundingPolygon[1], cv::Scalar( 255, 0, 0), 1, CV_AA );
	cv::line( oAnprObject.oLines, oPlateBoundingPolygon[1], oPlateBoundingPolygon[2], cv::Scalar( 255, 0, 0), 1, CV_AA );
	cv::line( oAnprObject.oLines, oPlateBoundingPolygon[2], oPlateBoundingPolygon[3], cv::Scalar( 255, 0, 0), 1, CV_AA );
	cv::line( oAnprObject.oLines, oPlateBoundingPolygon[3], oPlateBoundingPolygon[0], cv::Scalar( 255, 0, 0), 1, CV_AA );

#ifdef TRY_LSD
	bitwise_and( oAnprObject.oLinesImg, oTBLinesMask, oAnprObject.oLinesImg );
//	Mat oLinesColor;
//	cv::cvtColor( oLinesImg, oLinesColor, CV_GRAY2BGR );
//	cv::line( oLinesColor, oPlateBoundingPolygon[0], oPlateBoundingPolygon[1], cv::Scalar( 255, 0, 0), 1, CV_AA );
//	cv::line( oLinesColor, oPlateBoundingPolygon[2], oPlateBoundingPolygon[3], cv::Scalar( 255, 0, 0), 1, CV_AA );
#endif

	// Find min, max X to filter edge characters
	if(m_bDebug) oAnprObject.oDebugLogs.info( "Valid min x = %d, valid max x = %d", nValidMinX, nValidMaxX );
	int nMinX, nMaxX;
	int nAdjustedMinX, nAdjustedMaxX;
	int nXLimit = oSrc.cols - 1;
	if( nValidMinX > 0 )
	{
		nMinX = nValidMinX;
		nMaxX = nValidMaxX;

		// Adjust min, max X based on cropping and the above find min, max X results
		int nSafePadded = 2;	// TODO DV: setting
		nMinX = nMinX - nSafePadded;
		nMaxX = nMaxX + nSafePadded;
		if( nMinX < 0 ) nMinX = 0;
		if( nMaxX > nXLimit ) nMaxX = nXLimit;

		nAdjustedMinX = nMinX;
		nAdjustedMaxX = nMaxX;
	}
	else
	{
		FTS_IP_Util::findMinMaxX( myBlobDetector.m_ovvAllBlobs,
								  nXLimit,
								  params.minRepeatability,
								  oTopLine,
								  oBottomLine,
								  nMinX,
								  nMaxX,
								  nAdjustedMinX,
								  nAdjustedMaxX );
	}

	if(m_bDebug) oAnprObject.oDebugLogs.info( "After adjust, MinX = %d, MaxX = %d, width = %d", nAdjustedMinX, nAdjustedMaxX, (nXLimit+1) );
	oAnprObject.nAdjustedMinX = nAdjustedMinX;
	oAnprObject.nAdjustedMaxX = nAdjustedMaxX;

	// HISTOGRAM
	// =====================================================================
	//vector<Mat> allHistograms;

	// Each binary images will give a list of boxes
	vector<Rect> ovAllBoxes;
	Rect oInnerRect = Rect( nPaddedBorder, nPaddedBorder, srcPadded.cols - nPaddedBorder, srcPadded.rows - nPaddedBorder );
	for( size_t i = 0; i < oBinaryImages.size(); i++)
	{
		FTS_IP_VerticalHistogram oVertHist( oBinaryImages[i]( oInnerRect ), oTBLinesMask );

		// Debug ???
		Mat histoCopy(oVertHist.histoImg.size(), oVertHist.histoImg.type());
		if(m_bDebug) 
		{
			cvtColor(oVertHist.histoImg, histoCopy, CV_GRAY2RGB);
			oAnprObject.allHistograms.push_back(histoCopy);
		}

		float rScore = 0;
		vector<Rect> charBoxes = myBlobDetector.getBlobsByHist( oVertHist,
																oTopLine,
																oBottomLine,
																(float)oAnprObject.nMedianBlobWidth,
																(float)oAnprObject.nMedianBlobHeight,
																rScore );

		for( size_t z = 0; z < charBoxes.size(); z++)
		{
			ovAllBoxes.push_back(charBoxes[z]);
			rectangle( histoCopy, charBoxes[z], Scalar(255,0,0) );
		}
	}

	//if(m_bDebug) FTS_GUI_DisplayImage::ShowGroupScaleBy2( "Char histograms", 1.0, oAnprObject.allHistograms, 1 );

	if( ovAllBoxes.size() == 0 )
	{
		oAnprObject.oDebugLogs.warn( "NO BLOB IS DETECTED USING VERTICAL PROJECTION, SKIP" );
		return false;
	}

	// Only use boxes within x range
	vector<Rect> oBoxesWithinCroppedXRange = getBoxesInXRange( ovAllBoxes, nAdjustedMinX, nAdjustedMaxX );
	if( oBoxesWithinCroppedXRange.size() == 0 )
	{
		oAnprObject.oDebugLogs.warn( "ovAllBoxes has %d boxes, yet no box is within [%d, %d]", ovAllBoxes.size(), nAdjustedMinX, nAdjustedMaxX );
		return false;
	}

	// Now let's re-calculate the median width
//	FTS_BASE_STACK_ARRAY( int, oBoxesWithinCroppedXRange.size(), oArrWidths );
//	fillBlobWidthArray2( oArrWidths, oBoxesWithinCroppedXRange );
//	oAnprObject.nMedianBlobWidth = FTS_BASE_MedianBiasHigh( oArrWidths );
	oAnprObject.nMedianBlobWidth = FTS_ANPR_Util::findMedianBBWidthOfWoHInRange( oBoxesWithinCroppedXRange, rMinWoH, rMaxWoH, nMinMedianWidth );

	// Select best boxes
	vector<Rect> candidateBoxes = myBlobDetector.getBestBoxes( oSrcRotated,
															   oBoxesWithinCroppedXRange,
															   oAnprObject.nMedianBlobWidth,
															   oTopLine,
															   oBottomLine );

	oAnprObject.oDebugLogs.info( "After getBestBoxes() - number of blobs = %d", candidateBoxes.size() );

//	// Chop left and right edge characters to the median width
//	FTS_BASE_STACK_ARRAY( int, candidateBoxes.size(), oArrBestWidths );
//	fillBlobWidthArray2( oArrBestWidths, candidateBoxes );
//	oAnprObject.nMedianBlobWidth = FTS_BASE_MedianBiasHigh( oArrBestWidths );
//	bool bLeftEdgeCharFixed = false;
//	bool bRightEdgeCharFixed = false;
//	if( candidateBoxes.size() > 1 )
//	{
//		// Left edge char - only chop if the edge is out of [minX, maxX]
//		// DV: 23/06/2014 - only remove if more than 50% blob is out of red lines
//		if(    candidateBoxes[0].width > oAnprObject.nMedianBlobWidth
//			&& candidateBoxes[0].x < nAdjustedMinX )
//		{
//			oAnprObject.oDebugLogs.info( "Left edge char is chopped, w = %d > %d = median width",
//					candidateBoxes[0].width, oAnprObject.nMedianBlobWidth );
//			candidateBoxes[0].x    += candidateBoxes[0].width - oAnprObject.nMedianBlobWidth;
//			candidateBoxes[0].width = oAnprObject.nMedianBlobWidth;
//
//			bLeftEdgeCharFixed = true;
//		}
//
//		// Right edge char - only chop if the edge is out of [minX, maxX]
//		if(    candidateBoxes[candidateBoxes.size() - 1].width > oAnprObject.nMedianBlobWidth
//			&& candidateBoxes[candidateBoxes.size() - 1].x + candidateBoxes[candidateBoxes.size() - 1].width > nAdjustedMaxX )
//		{
//			oAnprObject.oDebugLogs.info( "Right edge char is chopped, w = %d > %d = median width",
//								candidateBoxes[candidateBoxes.size() - 1].width, oAnprObject.nMedianBlobWidth );
//			candidateBoxes[candidateBoxes.size() - 1].width = oAnprObject.nMedianBlobWidth;
//
//			bRightEdgeCharFixed = true;
//		}
//	}
//
//	// DEBUG: Draw detected blobs
//	if( bLeftEdgeCharFixed ) blobs[0].sStatus 				= FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED;
//	if( bRightEdgeCharFixed ) blobs[blobs.size()-1].sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED;
//	if(m_bDebug)
//	{
//		oAnprObject.oFirstBlobImg = drawBlobs( oAnprObject.oFirstBlobImg.data?oAnprObject.oFirstBlobImg:oSrcRotated, blobs );
//	}

	// Assign new blob candidates, remove blobs outside [minX, maxX]
	oAnprObject.oDebugLogs.info( "Remove blobs outside [minX, maxX] - number of blobs = %d", candidateBoxes.size() );
	blobs.clear();
	for( size_t i = 0; i < candidateBoxes.size(); i++ )
	{
		FTS_IP_SimpleBlobDetector::SimpleBlob sb( candidateBoxes[i] );

		// DV: 23/06/2014 - only remove if more than 50% blob is out of red lines
		if( isOutsideXRange( candidateBoxes[i], nAdjustedMinX, nAdjustedMaxX ) )
		{
			if(m_bDebug) oAnprObject.oDebugLogs.info( "WARNING BLOB REMOVAL - EDGE BLOB: x = %d, y = %d", candidateBoxes[i].x, candidateBoxes[i].y );
			sb.sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE;
		}
		blobs.push_back( sb );
	}

	// REMOVE NOISY CHARS
	removeNoisyBlobs( blobs );
	if( blobs.size() == 0 )
	{
		oAnprObject.oDebugLogs.warn( "NO BLOB IS DETECTED BY HISTOGRAM. RETURN" );
		return false;
	}

	// Vertical projection
	if(m_bDebug) oAnprObject.oDebugLogs.info( "HISTOGRAM - number of blobs = %d", blobs.size() );
	FTS_IP_VerticalHistogram oVertHist( oAnprObject.oMedOtsuThreshBinImg, oTBLinesMask );

	// MERGE & SPLIT
	// =====================================================================
//	FTS_BASE_STACK_ARRAY( int, blobs.size(), oWidthsBeforeMerge );
//	fillBlobWidthArray( oWidthsBeforeMerge, blobs );
//	oAnprObject.nMedianBlobWidth = FTS_BASE_MedianBiasHigh( oWidthsBeforeMerge );
	oAnprObject.nMedianBlobWidth = FTS_ANPR_Util::findMedianBlobWidthOfWoHInRange( blobs, rMinWoH, rMaxWoH, nMinMedianWidth );

	if(m_bDebug)
	{
		oAnprObject.oDebugLogs.info( "MERGE & SPLIT" );
		oAnprObject.oDebugLogs.info( "Before merge Median width = %d; num of blobs = %d", oAnprObject.nMedianBlobWidth, blobs.size() );		
	}
	myBlobDetector.mergeBlobs( blobs, oAnprObject.nMedianBlobWidth );

	// Re-calculate median width
//	FTS_BASE_STACK_ARRAY( int, blobs.size(), oWidthsBeforeSplit );
//	fillBlobWidthArray( oWidthsBeforeSplit, blobs );
//	oAnprObject.nMedianBlobWidth = FTS_BASE_MedianBiasHigh( oWidthsBeforeSplit );
	oAnprObject.nMedianBlobWidth = FTS_ANPR_Util::findMedianBlobWidthOfWoHInRange( blobs, rMinWoH, rMaxWoH, nMinMedianWidth );

	if(m_bDebug)
	{
		oAnprObject.oDebugLogs.info( "Median width before split = %d", oAnprObject.nMedianBlobWidth );
		oAnprObject.oDebugLogs.info( "After merge: num of blobs = %d", blobs.size() );
	}
	blobs = myBlobDetector.splitBlobs( oVertHist, blobs, (float)oAnprObject.nMedianBlobWidth, (float)oAnprObject.nMedianBlobHeight );
	if(m_bDebug) oAnprObject.oDebugLogs.info( "After split: num of blobs = %d", blobs.size() );
	cv::Mat oFirstBlobOutliersRemovedBlobMergedSplitImg = drawBlobs( oSrcRotated, blobs );

	// ADJUST X,Y ( 2 overlapped blobs ) and HEIGHT( blob is too short )
	// =====================================================================
	if(m_bDebug) oAnprObject.oDebugLogs.info( "ADJUST X,Y ( 2 overlapped blobs ) and HEIGHT( blob is too short )" );
	myBlobDetector.adjustBlobHeight( blobs, (float)oAnprObject.nMedianBlobHeight, oTopLine, oBottomLine );

	// Draw detected blobs
	if(m_bDebug)
	{
		if(!oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg.data)
			oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg = drawBlobs( oSrcRotated, blobs );
		else
			oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg = drawBlobs( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, blobs );

		// Draw lines
		line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg,
			  Point( nMinX, 0 ), Point( nMinX, oSrcRotated.rows-1 ), Scalar(0, 255,255) );
		line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg,
			  Point( nMaxX, 0 ), Point( nMaxX, oSrcRotated.rows-1 ), Scalar(0, 255,255) );

		line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg,
			  Point( nAdjustedMinX, 0 ), Point( nAdjustedMinX, oSrcRotated.rows-1 ), Scalar(0, 0, 255) );
		line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg,
			  Point( nAdjustedMaxX, 0 ), Point( nAdjustedMaxX, oSrcRotated.rows-1 ), Scalar(0, 0, 255) );
	}

	// OCR PREPARATION
	// =====================================================================
	if(m_bDebug) oAnprObject.oDebugLogs.info( "Start OCR Preparation " );
	stringstream ss;

	if(m_bDebug) oAnprObject.oDebugLogs.info( "Before using OTSU filter, there are %d characters", blobs.size() );

	cv::Rect oSrcRotatedRect(0, 0, oSrcRotated.cols, oSrcRotated.rows);
	for( unsigned int i = 0; i < blobs.size(); i++ )
	{
		if(m_bDebug) oAnprObject.oDebugLogs.info( "Character %d: [%d, %d, %d, %d]", i+1, blobs[i].oBB.x, blobs[i].oBB.y, blobs[i].oBB.width, blobs[i].oBB.height );
		if(blobs[i].oBB.area() == 0)
		{
			oAnprObject.oDebugLogs.warn("FOUND NULL BLOBS at %d!!!", i+1);
			continue;
		}
		
		// Character bounding box
		cv::Mat oCharBin;
		cv::Rect oExpandedBox( blobs[i].oBB );
		if (!FTS_BASE_Util::IsRectIntersect(oSrcRotatedRect, oExpandedBox)){
			blobs.erase(blobs.begin() + i);
			i--;
			continue;
		}

		if (oExpandedBox.x < 0)
		{
			oExpandedBox.x = 0;
		}
		if (oExpandedBox.y < 0)
		{
			oExpandedBox.y = 0;
		}
		if (oExpandedBox.x + oExpandedBox.width > oSrcRotated.cols)
		{
			oExpandedBox.width = oSrcRotated.cols - oExpandedBox.x;
		}
		if (oExpandedBox.y + oExpandedBox.height > oSrcRotated.rows)
		{
			oExpandedBox.height = oSrcRotated.rows - oExpandedBox.y;
		}

#ifdef EXPAND_CHAR_BOX
		oExpandedBox = FTS_IP_Util::expandRectXY( oCharBox, 1, 1,	oSrcRotated.cols, oSrcRotated.rows );
#endif
		// Binarize the expanded region using local otsu if enabled, else use alternative
		if( bUseLocalOtsu )
		{
			cv::threshold( oSrcRotated(oExpandedBox), oCharBin, 0, 255, nThresholdType | CV_THRESH_OTSU );
		}
		else
		{
			oCharBin = oLocalOtsuSubstitue(oExpandedBox).clone();
		}

		// mask character by mask found earlier
		cv::bitwise_and( oCharBin, oMask(oExpandedBox), oCharBin );

		if(m_bDebug) oAnprObject.oDebugLogs.info( "oCharBin.width = %d, oCharBin.height = %d", oCharBin.cols, oCharBin.rows );
		// Crop black pixels
		cv::Rect oSubBox = FTS_IP_Util::MinAreaRect( oCharBin );
		if(m_bDebug) oAnprObject.oDebugLogs.info( "oCharBin.width = %d, oCharBin.height = %d", oCharBin.cols, oCharBin.rows );

		cv::Rect oFinalBox( oExpandedBox );
				 oFinalBox.x 	 += oSubBox.x;
				 oFinalBox.y 	 += oSubBox.y;
				 oFinalBox.width  = oSubBox.width;
				 oFinalBox.height = oSubBox.height;

		if(m_bDebug)
		{
			oAnprObject.oDebugLogs.info( "Original char box: x = %d, y = %d, w = %d, h = %d",
				blobs[i].oBB.x, blobs[i].oBB.y, blobs[i].oBB.width, blobs[i].oBB.height );
			oAnprObject.oDebugLogs.info( "Expanded char box: x = %d, y = %d, w = %d, h = %d",
					oExpandedBox.x, oExpandedBox.y, oExpandedBox.width, oExpandedBox.height );
			oAnprObject.oDebugLogs.info( "Sub box: x = %d, y = %d, w = %d, h = %d",
					oSubBox.x, oSubBox.y, oSubBox.width, oSubBox.height );
			oAnprObject.oDebugLogs.info( "Final box: x = %d, y = %d, w = %d, h = %d",
					oFinalBox.x, oFinalBox.y, oFinalBox.width, oFinalBox.height );
			oAnprObject.oDebugLogs.info( "oCharBin.width = %d, oCharBin.height = %d", oCharBin.cols, oCharBin.rows );
		}

//		// DV: 01/07/2014
//		// After using local otsu to crop the character, use the alternative binary image if enabled
//		if( !bUseLocalOtsu )
//		{
//			oCharBin = oLocalOtsuSubstitue(oExpandedBox).clone();
//			cv::bitwise_and( oCharBin, oMask(oExpandedBox), oCharBin );
//		}
		cv::Mat oROISrc = oCharBin(oSubBox);

		if(m_bDebug)
		{
			oAnprObject.oDebugLogs.info( "oCharBin.width = %d, oCharBin.height = %d", oCharBin.cols, oCharBin.rows );
			oAnprObject.oDebugLogs.info( "oROISrc.width = %d, oROISrc.height = %d", oROISrc.cols, oROISrc.rows ) ;
		}

		bool b = ( i == 0 || i == blobs.size() - 1 );
		float rImgMean = mean( oROISrc )[0];
		if(m_bDebug) oAnprObject.oDebugLogs.info( "Intensity mean = %f", rImgMean );
		if(    ( rImgMean < FTS_IP_SimpleBlobDetector::CHAR_MIN_FILLED  )
			|| ( rImgMean > FTS_IP_SimpleBlobDetector::MIDDLE_CHAR_MAX_FILLED  )
			|| ( b && rImgMean > FTS_IP_SimpleBlobDetector::EDGE_CHAR_MAX_FILLED  ) )
		{
			if(m_bDebug) 
			{
				oAnprObject.oDebugLogs.warn( "WARNING BLOB REMOVAL - EDGE MOSTLY FULL OR EMPTY CHAR( LOCAL OTSU BIN )" );
				line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, 
					blobs[i].oBB.tl(), blobs[i].oBB.br(), Scalar( 0, 0, 255 ) );
				line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg,
					  Point( blobs[i].oBB.x + blobs[i].oBB.width, blobs[i].oBB.y ),
					  Point( blobs[i].oBB.x, blobs[i].oBB.y + blobs[i].oBB.height ), Scalar( 0, 0, 255 ) );
			}

			blobs[i].sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_REMOVED;
		}
		//++04.07 trungnt1 Not need here, use later
		else if( (float)oFinalBox.height < 0.7* blobs[i].oBB.height )
		{
			if(m_bDebug) 
			{
				oAnprObject.oDebugLogs.warn( "WARNING BLOB REMOVAL - SHORT CHAR( LOCAL OTSU BIN )" );
				line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, 
					blobs[i].oBB.tl(), blobs[i].oBB.br(), Scalar( 0, 0, 255 ) );
				line( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg,
					  Point( blobs[i].oBB.x + blobs[i].oBB.width, blobs[i].oBB.y ),
					  Point( blobs[i].oBB.x, blobs[i].oBB.y + blobs[i].oBB.height ), Scalar( 0, 0, 255 ) );
			}

			blobs[i].sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_REMOVED;
		}
		//--
		else
		{
			blobs[i].oBB = oFinalBox;	// TODO this ROI is on the original image, NOT padded
			if(m_bDebug) oAnprObject.oDebugLogs.info( "GOOD CHAR HAS BEEN ADDED" );
		}

		//if(m_bDebug) oAnprObject.oDebugLogs.info( "");
	}

	// REMOVE NOISY CHARS
	if(m_bDebug) oAnprObject.oDebugLogs.info( "REMOVE NOISY CHARS" );
	removeNoisyBlobs( blobs );
	if( blobs.size() == 0 )
	{
		oAnprObject.oDebugLogs.warn( "THERE ARE NONE OF GOOD CHARACTERS TO DO OCR. RETURN NOW." );
		return false;
	}

	// DV: 01/07/2014 - Merge again before otsu filter
	// This is to not only fix H, N cut in half but help better otsu filter too
	// NB: This is only for top line, because for Vietnam plates, bottom line
	// does not have "letter"
//	if(    nMaxNbrOfChar == MAX_NUM_OF_DUAL_LINE_TOP
//		&& blobs.size() > 2 )
//	{
//		// Merge blobs again to fix H,N cut in half
//		vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::const_iterator first = blobs.begin();
//		vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::const_iterator last  = blobs.begin() + 2;
//		vector<FTS_IP_SimpleBlobDetector::SimpleBlob> newVec(first, last);
//
//		FTS_BASE_STACK_ARRAY( int, newVec.size(), oWidthsBefore2ndMerge );
//		fillBlobWidthArray( oWidthsBefore2ndMerge, newVec );
//		int nMedianToplineFirst2BlobWidth = FTS_BASE_MedianBiasHigh( oWidthsBefore2ndMerge );
//		if(m_bDebug) oAnprObject.oDebugLogs.info( "MERGE THE 2ND TIME - median width = %d", nMedianToplineFirst2BlobWidth );
//		myBlobDetector.mergeBlobs( blobs, nMedianToplineFirst2BlobWidth );
//	}
	int nMedianToplineFirst2BlobWidth = FTS_ANPR_Util::findMedianBlobWidthOfWoHInRange( blobs, rMinWoH, rMaxWoH, nMinMedianWidth );
	if(m_bDebug) oAnprObject.oDebugLogs.info( "MERGE THE 2ND TIME - median width = %d", nMedianToplineFirst2BlobWidth );
	myBlobDetector.mergeBlobs( blobs, nMedianToplineFirst2BlobWidth );

	// OTSU FILTER TO REMOVE EDGE CHARS
	// =====================================================================
	vector<double> rvOtsuVals;
	for( size_t i = 0; i < blobs.size(); i++ )
	{
		// Binarize the expanded region using otsu
		Mat oTmpBin;
		double rCharOtsuThresh = cv::threshold( oSrcRotated(blobs[i].oBB), oTmpBin, 0, 255, nThresholdType | CV_THRESH_OTSU );
		rvOtsuVals.push_back( rCharOtsuThresh );
	}

	int otsuHistByMeanStartX = 0;
	if(!oAnprObject.otsuHistByMean.data)
		oAnprObject.otsuHistByMean = Mat::zeros( Size(300,255), CV_8UC3 );
	else
		otsuHistByMeanStartX = 150;
	vector<cv::Point2f> points;
	if(m_bDebug) oAnprObject.oDebugLogs.info( "OTSU: rvOtsuVals.size() =%d", rvOtsuVals.size() );
	for( size_t i = 0; i < rvOtsuVals.size(); i++ )
	{
		if(m_bDebug) oAnprObject.oDebugLogs.info( "%d, ", (int)rvOtsuVals[i] );
		points.push_back( Point2f(otsuHistByMeanStartX+20*(i+1), 255 - rvOtsuVals[i]) );
	}

	// Median otsu
	double SAFE_OTSU_EPSILON = 15;	// TODO DV: setting
	vector<double> rTmpOtsuVals = rvOtsuVals;
	double rMedOtsu = FTS_BASE_Median( rTmpOtsuVals );

	// Refine the values for mean calculation
	vector<double> rRefinedOtsuVals;
	for( size_t i = 0; i < rvOtsuVals.size(); i++ )
	{
		double rDistMedian = abs( rvOtsuVals[i] - rMedOtsu );
		if( rDistMedian < SAFE_OTSU_EPSILON )
		{
			rRefinedOtsuVals.push_back( rvOtsuVals[i] );
		}
	}

	// Mean otsu
	// DV: 03/07/2014 - Otsu filter now has soft and hard threshold
	// TODO: tuning?
	double MAX_OTSU_DIST_SOFT = 30.0;
	double MAX_OTSU_DIST_HARD = 70.0;
	double sum = accumulate( rRefinedOtsuVals.begin(), rRefinedOtsuVals.end(), 0.0 );
	double mean = sum / rRefinedOtsuVals.size();
	vector<double> diff(rRefinedOtsuVals.size());
	transform(rRefinedOtsuVals.begin(), rRefinedOtsuVals.end(), diff.begin(),
				   bind2nd(minus<double>(), mean));
	double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stdev = sqrt(sq_sum / rRefinedOtsuVals.size());
	if(m_bDebug) oAnprObject.oDebugLogs.info("    MEAN = %f, MEDIAN  = %f,  SD = %f", mean, rMedOtsu, stdev );

	assert( blobs.size() == rvOtsuVals.size() );
	for( size_t i = 0; i < rvOtsuVals.size(); i++ )
	{
		Scalar color;
		double rDistMean   = abs( rvOtsuVals[i] - mean );
		double rDistMedian = abs( rvOtsuVals[i] - rMedOtsu );
		if(m_bDebug) oAnprObject.oDebugLogs.info( "rDistMean = %f, rDistMedian = %f", rDistMean/stdev, rDistMedian/stdev);

		bool bIsEdgeBlob 	= ( i == 0 || i == blobs.size() - 1 );	// most left or most right char
		bool bIsOtsuOutlierSoft = rDistMean > MAX_OTSU_DIST_SOFT;	// otsu outlier
		bool bIsOtsuOutlierHard = rDistMean > MAX_OTSU_DIST_HARD;

		if( rDistMean < 3*stdev ) color = Scalar( 0, 255, 0 );			// green
		else
		{
			color = Scalar( 0, 0, 255 );

			Rect oBox( blobs[i].oBB );
			Point2f oCenter( oBox.x + (float)oBox.width/2, oBox.y + (float)oBox.height/2 );
			// DV: 23/06/2014 - only remove if more than 50% blob is out of red lines
			if( isOutsideXRange( oBox, nMinX, nMaxX ) )
			{
				circle( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, oCenter, oBox.width/2, Scalar( 0, 0, 255 ), CV_FILLED );

				// EDGE char
				oAnprObject.oDebugLogs.warn( "WARNING BLOB REMOVAL - OTSU OUTLIER + OUT OF [MINX, MAXX]" );
				blobs[i].sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE;
			}
			// DV: 03/07/2014 - this is to safe to remove
			else if( bIsOtsuOutlierHard )
			{
				circle( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, oCenter, oBox.width/2, Scalar( 0, 0, 255 ), CV_FILLED );

				// EDGE char
				oAnprObject.oDebugLogs.warn( "WARNING BLOB REMOVAL - OTSU OUTLIER OF HARD THRESHOLD" );
				blobs[i].sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE;
			}
			// DV: 16/06/2014 - make sure median Otsu filter
			// 					does not remove good characters
			else if( bIsOtsuOutlierSoft  )
			{
				bool bIsFiltered = false;

				// 1. If exceed max num of chars, no regret remove it
				int nNumOfValidBlob = countValidBlobs( blobs );
				if( nNumOfValidBlob > nMaxNbrOfChar )
				{
					bIsFiltered = true;
				}
				// 2. If number of blob = max num of chars, then remove it if
				//    a. Blob is edge
				//	  b. Blob is far from the next blob
				else if( nNumOfValidBlob == nMaxNbrOfChar )
				{
					if( bIsEdgeBlob && blobs.size() > 1 )
					{
						int nXDist = 0;
						int nMinBlobWidth = 0;
						if( i == 0 )
						{
							nXDist = FTS_IP_SimpleBlobDetector::calcXDistBetween2Blobs( blobs[i], blobs[i+1] );
							nMinBlobWidth = min( blobs[i].oBB.width, blobs[i+1].oBB.width );
						}
						else if( i == blobs.size() - 1 )
						{
							nXDist = FTS_IP_SimpleBlobDetector::calcXDistBetween2Blobs( blobs[i-1], blobs[i] );
							nMinBlobWidth = min( blobs[i-1].oBB.width, blobs[i].oBB.width );
						}

						// Test if two blobs are far from each other
						if( nXDist > 2 * nMinBlobWidth )	// TODO: setting?
						{
							bIsFiltered = true;
						}
					}
				}

				if( bIsFiltered )
				{
					circle( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, oCenter, oBox.width/2, Scalar( 0, 255, 255 ), CV_FILLED );

					// EDGE char
					oAnprObject.oDebugLogs.warn( "WARNING BLOB REMOVAL - OTSU OUTLIER" );
					blobs[i].sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE;
				}
				else
				{
					circle( oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, oCenter, oBox.width/2, Scalar( 255, 0, 0 ), CV_FILLED );
				}
			}
		}
		circle( oAnprObject.otsuHistByMean, points[i], 3, color, CV_FILLED );
	}

	// REMOVE EDGE CHARS
	removeNoisyBlobs( blobs );

	//OCRInput input;
	vector<Rect> newBestCharBoxes (blobs.size());
	blobs2Rects( blobs, newBestCharBoxes, nPaddedBorder, nPaddedBorder, srcPadded.cols-1, srcPadded.rows-1 );

	// Remove mostly empty blobs
	vector<Rect> oCandidatesAfterEmptyBoxesRemoved = myBlobDetector.removeEmptyBoxes( oBinaryImages, newBestCharBoxes);
	if(m_bDebug)
	{
		if(!oAnprObject.oTestEmptyBlob.data)
			cvtColor( oSrcRotated, oAnprObject.oTestEmptyBlob, CV_GRAY2BGR );
		for( size_t i = 0; i < newBestCharBoxes.size(); i++ )
		{
			rectangle( oAnprObject.oTestEmptyBlob, newBestCharBoxes[i], Scalar(0,0,255) );
		}
		for( size_t i = 0; i < oCandidatesAfterEmptyBoxesRemoved.size(); i++ )
		{
			rectangle( oAnprObject.oTestEmptyBlob, oCandidatesAfterEmptyBoxesRemoved[i], Scalar(0,255,0) );
		}
	}

	// DV: 01/07/2014 - Remove noises of top line
	// NB: TODO: This is ONLY for Vietnam plates
//	if(    nMaxNbrOfChar == MAX_NUM_OF_DUAL_LINE_TOP
//		&& oCandidatesAfterEmptyBoxesRemoved.size() > 2 )
//	{
//		// Merge blobs again to fix H,N cut in half
//		vector<Rect>::const_iterator first = oCandidatesAfterEmptyBoxesRemoved.begin();
//		vector<Rect>::const_iterator last  = oCandidatesAfterEmptyBoxesRemoved.begin() + 2;
//		vector<Rect> newVec(first, last);
//
//		FTS_BASE_STACK_ARRAY( int, newVec.size(), oWidthsBefore2ndMerge );
//		fillBlobWidthArray2( oWidthsBefore2ndMerge, newVec );
//		int nMedianToplineFirst2BlobWidth = FTS_BASE_MedianBiasHigh( oWidthsBefore2ndMerge );
//		if(m_bDebug) oAnprObject.oDebugLogs.info( "MERGE THE 2ND TIME - median width = %d", oAnprObject.nMedianBlobWidth );
//		myBlobDetector.mergeBlobsVector( oCandidatesAfterEmptyBoxesRemoved, nMedianToplineFirst2BlobWidth );
//	}
	nMedianToplineFirst2BlobWidth = FTS_ANPR_Util::findMedianBBWidthOfWoHInRange( oCandidatesAfterEmptyBoxesRemoved, rMinWoH, rMaxWoH, nMinMedianWidth );
	if(m_bDebug) oAnprObject.oDebugLogs.info( "MERGE THE 3RD TIME - median width = %d", nMedianToplineFirst2BlobWidth );
	myBlobDetector.mergeBlobsVector( oCandidatesAfterEmptyBoxesRemoved, nMedianToplineFirst2BlobWidth );

	// DV: 03/07/2014 - Chop edge chars to median width
	chopEdgeCharToMedianWidth( oCandidatesAfterEmptyBoxesRemoved );

	newBestCharBoxes = oCandidatesAfterEmptyBoxesRemoved;

	// Get final clean chars
	if(!oAnprObject.oCleanCharBin.data)
	{
		oAnprObject.oCleanCharBin = cv::Mat::zeros( srcPadded.size(), srcPadded.type() );
		
	}
	//++29.06 trungnt1 add to specify the line of char boxes
	int nLinePos = (oAnprObject.oBestCharBoxes.size() > 0) ? 1 : 0; //0 => top line, 1: bottom line;
	//--
	for( size_t i = 0; i < newBestCharBoxes.size(); i++ )
	{
		// Run local otsu
		// TODO: local otsu is not good for the top line as per observation
		// For temporary, use adaptive threshold with window = 19
		cv::Mat oROISrc;
		if( bUseLocalOtsu )
		{
			threshold( srcPadded(newBestCharBoxes[i]), oROISrc, 0, 255, nThresholdType | CV_THRESH_OTSU );
		}
		else
		{
			oROISrc = oLocalOtsuSubstitue(newBestCharBoxes[i]);
		}

		cv::Mat oROIDst = oAnprObject.oCleanCharBin( newBestCharBoxes[i] );
		oROISrc.copyTo( oROIDst );
		oAnprObject.oBestCharBoxes.push_back(newBestCharBoxes[i]);
		oAnprObject.oCharPosLine.push_back(nLinePos);	//29.06 define char box line position
	}	

	if(m_bDebug) oAnprObject.oDebugLogs.info( "CHARACTER PREPARATION for OCR ..." );
	vector<Mat> oMaskBinaries(oBinaryImages.size());
	for( size_t i = 0; i < oBinaryImages.size(); i++ )
	{
		// Mask char regions
		oMaskBinaries[i] = Mat::zeros( oBinaryImages[i].size(), oBinaryImages[i].type() );
		for( size_t j = 0; j < newBestCharBoxes.size(); j++ )
		{
			Mat oSrcROI = oBinaryImages[i]( newBestCharBoxes[j] );
			Mat oDstROI = oMaskBinaries[i]( newBestCharBoxes[j] );
			oSrcROI.copyTo( oDstROI );
		}
	}

	// Do OCR on 4 best binary images
	// 1. Wolfzilion 1
	// 2. Adaptive 15
	// 3. Adaptive 19
	// 4. Local OTSU or Adaptive 11
	assert( oBinaryImages.size() >= 5 );
	if(oAnprObject.oBestBinImages.size() == 0)
	{
		oAnprObject.oBestBinImages.push_back( oMaskBinaries[0] );
		oAnprObject.oBestBinImages.push_back( oMaskBinaries[3] );
		oAnprObject.oBestBinImages.push_back( oMaskBinaries[4] );
		if( bUseLocalOtsu )
		{
			oAnprObject.oBestBinImages.push_back( oAnprObject.oCleanCharBin );
		}
		else
		{
			oAnprObject.oBestBinImages.push_back( oMaskBinaries[2] );
		}
	}
	else
	{
		bitwise_or(oAnprObject.oBestBinImages[0], oMaskBinaries[0], oAnprObject.oBestBinImages[0]);
		bitwise_or(oAnprObject.oBestBinImages[1], oMaskBinaries[3], oAnprObject.oBestBinImages[1]);
		bitwise_or(oAnprObject.oBestBinImages[2], oMaskBinaries[4], oAnprObject.oBestBinImages[2]);
		if( bUseLocalOtsu )
		{
			bitwise_or(oAnprObject.oBestBinImages[3], oAnprObject.oCleanCharBin, oAnprObject.oBestBinImages[3]);
		}
		else
		{
			bitwise_or(oAnprObject.oBestBinImages[3], oMaskBinaries[2], oAnprObject.oBestBinImages[3]);
		}
	}
	

#ifdef GUI
	if(m_bDebug)
	{		
		//Column 1
		//FTS_GUI_DisplayImage::ShowAndScaleBy2( "Rotated", oSrc,
		//			FTS_GUI_DisplayImage::SCALE_X,
		//			FTS_GUI_DisplayImage::SCALE_Y,0, 0 );
		//FTS_GUI_DisplayImage::ShowAndScaleBy2( "Hist", histoCopy, FTS_GUI_DisplayImage::SCALE_X,
		//			FTS_GUI_DisplayImage::SCALE_Y, 0, 400 );

		/*FTS_GUI_DisplayImage::ShowAndScaleBy2( "LSD images", oAnprObject.oLinesImg, FTS_GUI_DisplayImage::SCALE_X,
				FTS_GUI_DisplayImage::SCALE_Y, 0, 400 );

		FTS_GUI_DisplayImage::ShowAndScaleBy2( "First attempt", oAnprObject.oFirstBlobImg, FTS_GUI_DisplayImage::SCALE_X,
				FTS_GUI_DisplayImage::SCALE_Y, 860, 110 );

		FTS_GUI_DisplayImage::ShowAndScaleBy2( "First attempt, outliers, merged, split, height", 
			oAnprObject.oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, FTS_GUI_DisplayImage::SCALE_X,
				FTS_GUI_DisplayImage::SCALE_Y, 860, 500 );

		FTS_GUI_DisplayImage::ShowAndScaleBy2( "Blob Bounding Polygon", oAnprObject.oLines, FTS_GUI_DisplayImage::SCALE_X,
				FTS_GUI_DisplayImage::SCALE_Y, 860, 600 );

		FTS_GUI_DisplayImage::ShowAndScaleBy2( "OTSU HIST MEAN", oAnprObject.otsuHistByMean, 1.0, 1.0, 0, 0 );
		FTS_GUI_DisplayImage::ShowAndScaleBy2( "Clean Char", oAnprObject.oCleanCharBin, 1.0, 1.0, 500, 0 );*/

		oAnprObject.oDebugLogs.info( "After adjust, MinX = %d, MaxX = %d, width = %d", nMinX, nMaxX, (nXLimit+1) );
	}
#endif

	return true;
}

bool Fts_Anpr_Engine::chopEdgeCharToMedianWidth( vector<Rect>& candidateBoxes )
{
	if( candidateBoxes.size() < 2 )
	{
		return false;
	}

	// Chop left and right edge characters to the median width
	FTS_BASE_STACK_ARRAY( int, candidateBoxes.size(), oArrBestWidths );
	fillBlobWidthArray2( oArrBestWidths, candidateBoxes );
	int nMedianBlobWidth = FTS_BASE_MedianBiasHigh( oArrBestWidths );

	bool bLeftEdgeCharFixed = false;
	bool bRightEdgeCharFixed = false;

	// Left edge char
	if(  candidateBoxes[0].width > nMedianBlobWidth )
	{
		printf( "Left edge char is chopped, w = %d > %d = median width\n",
				candidateBoxes[0].width, nMedianBlobWidth );
		candidateBoxes[0].x    += candidateBoxes[0].width - nMedianBlobWidth;
		candidateBoxes[0].width = nMedianBlobWidth;

		bLeftEdgeCharFixed = true;
	}

	// Right edge char
	if( candidateBoxes[candidateBoxes.size() - 1].width > nMedianBlobWidth )
	{
		printf( "Right edge char is chopped, w = %d > %d = median width\n",
							candidateBoxes[candidateBoxes.size() - 1].width, nMedianBlobWidth );
		candidateBoxes[candidateBoxes.size() - 1].width = nMedianBlobWidth;

		bRightEdgeCharFixed = true;
	}

	return ( bLeftEdgeCharFixed || bRightEdgeCharFixed );
}

bool Fts_Anpr_Engine::isOutsideXRange( const Rect& oBox,
					  const int nMinX,
					  const int nMaxX )
{
	// DV: 23/06/2014 - only remove if more than 50% blob is out of red lines
	int nOutsideLeft  = nMinX - oBox.x;
	int nOutsideRight = oBox.x + oBox.width - nMaxX;

	// Percent of outside portion
	float rOutsideRatioLeft  = (float)nOutsideLeft  / (float)oBox.width;
	float rOutsideRatioRight = (float)nOutsideRight / (float)oBox.width;

	if(    rOutsideRatioLeft > 0.5		// DV: 23/06/2014 - Change 0.5 to 0.0 if there is any problem
		|| rOutsideRatioRight > 0.5 )
	{
		return true;
	}

	return false;
}

void Fts_Anpr_Engine::blobs2Rects( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
						vector<Rect> & rects,
						const int nPaddedX,
						const int nPaddedY,
						const int maxW,
						const int maxH )
{
	rects.reserve( blobs.size() );
	for( size_t i = 0; i < blobs.size(); i++ )
	{
		Rect oTmpRect = blobs[i].oBB;
		oTmpRect = blobs[i].oBB;

		oTmpRect.x += nPaddedX;
		oTmpRect.y += nPaddedY;

		rects[i] = FTS_IP_Util::expandRectXY( oTmpRect, 2, 2, maxW, maxH );
	}
}

void Fts_Anpr_Engine::removeNoisyBlobs( vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
	vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::iterator it = blobs.begin();
	for ( ; it != blobs.end(); )
	{
		if(    strcmp( it->sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE.c_str() ) != 0
			&& strcmp( it->sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED.c_str() ) != 0 )
		{
			it = blobs.erase(it);
		}
		else
		{
			++it;
		}
	}
}

int Fts_Anpr_Engine::countValidBlobs( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
	int nCount = blobs.size();

	for( unsigned int i = 0; i < blobs.size(); i++ )
	{
		if(    strcmp( blobs[i].sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE.c_str() )  != 0
			&& strcmp( blobs[i].sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED.c_str() ) != 0 )
		{
			nCount--;
		}
	}

	return nCount;
}

cv::Mat Fts_Anpr_Engine::drawBlobs( const cv::Mat& oSrc,
									const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
	// Draw detected blobs after removing outliers
	cv::Mat oDst;
	if(oSrc.channels() == 1)
		cv::cvtColor( oSrc, oDst, CV_GRAY2BGR );
	else
		oDst = oSrc.clone();
	for( unsigned int i = 0; i < blobs.size(); i++ )
	{
//		if(    blobs[i].sStatus != FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE
//			&& blobs[i].sStatus != FTS_IP_SimpleBlobDetector::s_sSTATUS_OUTLIER_RECONSIDERED )
//		{
//			continue;
//		}

		cv::Rect oBox = blobs[i].toRect();
		if (oBox.x < 0)
		{
			oBox.x = 0;
		}
		if (oBox.y < 0)
		{
			oBox.y = 0;
		}
		if (oBox.x + oBox.width > oSrc.cols)
		{
			oBox.width = oSrc.cols - oBox.x;
		}
		if (oBox.y + oBox.height > oSrc.rows)
		{
			oBox.height = oSrc.rows - oBox.y;
		}

		if( strcmp( blobs[i].sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE.c_str() ) == 0 )
		{
			cv::rectangle( oDst, oBox, CV_RGB( 255, 0, 0) );
		}
		else if( strcmp( blobs[i].sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED.c_str() ) == 0 )
		{
			cv::rectangle( oDst, oBox, CV_RGB( 255, 182, 193) );
		}
		else if( strcmp( blobs[i].sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE.c_str() ) == 0 )
		{
			cv::rectangle( oDst, oBox, CV_RGB( 0, 255, 0) );
		}
	}

	return oDst;
}

void Fts_Anpr_Engine::fillBlobWidthArray( FTS_BASE_StackArray<int>& oArray,
						 const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].oBB.width;
	}
}

void Fts_Anpr_Engine::fillBlobWidthArray2( FTS_BASE_StackArray<int>& oArray,
						  const vector<Rect>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].width;
	}
}

void Fts_Anpr_Engine::fillBlobHeightArray( FTS_BASE_StackArray<int>& oArray,
						  const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].oBB.height;
	}
}

void Fts_Anpr_Engine::fillBlobOtsuArray( FTS_BASE_StackArray<double>& oArray,
						const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
						const cv::Mat& oSrc )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
		cv::Mat oTmp;
		oArray.at(i) = cv::threshold( oSrc( blobs[i].oBB ), oTmp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
	}
}

void Fts_Anpr_Engine::findBlobsInXRange( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobs,
						const int nMinX,
						const int nMaxX,
						vector<int>& nvGoodIndices )
{
	nvGoodIndices.resize( oBlobs.size() );
	for( unsigned int i = 0; i < oBlobs.size(); i++ )
	{
		if( oBlobs[i].oBB.x < nMinX || oBlobs[i].oBB.x + oBlobs[i].oBB.width > nMaxX )
		{
			nvGoodIndices[i] = 0;
		}
		else
		{
			nvGoodIndices[i] = 0;
		}
	}
}

vector<FTS_IP_SimpleBlobDetector::SimpleBlob> Fts_Anpr_Engine::getBlobsInXRange(
								const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobs,
								const int nMinX,
								const int nMaxX )
{
	vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oNewBlobs;
	for( unsigned int i = 0; i < oBlobs.size(); i++ )
	{
		if( oBlobs[i].oBB.x < nMinX || oBlobs[i].oBB.x + oBlobs[i].oBB.width > nMaxX )
		{
			continue;
		}

		oNewBlobs.push_back( oBlobs[i] );
	}

	return oNewBlobs;
}

void Fts_Anpr_Engine::findBoxesInXRange( const vector<Rect>& oBoxes,
						const int nMinX,
						const int nMaxX,
						vector<int>& nvGoodIndices )
{
	nvGoodIndices.resize( oBoxes.size() );
	for( unsigned int i = 0; i < oBoxes.size(); i++ )
	{
		if( oBoxes[i].x < nMinX || oBoxes[i].x + oBoxes[i].width > nMaxX )
		{
			nvGoodIndices[i] = 0;
		}
		else
		{
			nvGoodIndices[i] = 0;
		}
	}
}

vector<Rect> Fts_Anpr_Engine::getBoxesInXRange( const vector<Rect>& oBoxes,
								const int nMinX,
								const int nMaxX )
{
	vector<Rect> oNewBoxes;
	for( unsigned int i = 0; i < oBoxes.size(); i++ )
	{
		// DV: 23/06/2014 - change outside condition check
		//if( oBoxes[i].x < nMinX || oBoxes[i].x + oBoxes[i].width > nMaxX )
		if( isOutsideXRange( oBoxes[i], nMinX, nMaxX ) )
		{
			continue;
		}

		oNewBoxes.push_back( oBoxes[i] );
	}

	return oNewBoxes;
}

std::string Fts_Anpr_Engine::getVersion()
{
  std::stringstream ss;  
  ss << FTSANPR_MAJOR_VERSION << "." << FTSANPR_MINOR_VERSION << "." << FTSANPR_PATCH_VERSION;
  return ss.str();
}

vector< vector<Rect> > Fts_Anpr_Engine::getExactCharBB( const vector<Mat>& thresholds,
								 	   const vector< Rect >& oBestCharBoxes )
{
	vector< vector<Rect> > charRegions;

	for (unsigned int i = 0; i < thresholds.size(); i++)
	{
		vector<Rect> oExactBestCharBoxes = oBestCharBoxes;
		for (unsigned int j = 0; j < oBestCharBoxes.size(); j++)
		{
			oExactBestCharBoxes[j] = FTS_IP_Util::getExactBB( thresholds[i], oBestCharBoxes[j], 0, 0 );
		}

		charRegions.push_back( oExactBestCharBoxes );
	}

	return charRegions;
}

vector< vector<Rect> > Fts_Anpr_Engine::getCorrectSizedCharRegions( const vector<Mat>& oBestBinImages,
										    	 const vector< Rect >& oBestCharBoxes,
										    	 const vector< vector<Rect> >& charRegions,
												 FTS_ANPR_OBJECT& oAnprObject )
{
	oAnprObject.oDebugLogs.info( "Test blob height, emptiness, fullness - median height = %d", oAnprObject.nMedianBlobHeight );
	Mat vnFlag2D = Mat::ones( Size( oBestCharBoxes.size(), oBestBinImages.size() ), CV_8U );
	for( size_t i = 0; i < oBestBinImages.size(); i++ )
	{
		const vector<Rect>& ovCurrCharRegions = charRegions[i];
		for( size_t j = 0; j < oBestCharBoxes.size(); j++ )
		{
			// DV: 03/07/2014 - Find max contour
			// If max contour is < 0.3 * whole box area, likely not a char
			std::vector<cv::Point> oLargestCC = FTS_IP_Util::FindLargestCC( oBestBinImages[i]( ovCurrCharRegions[j] ) );

			bool bIsMostLeftorRight = ( j == 0 || j == oBestCharBoxes.size() - 1 );

			// DV: 23/06/2014 - mean is calculated on both orginal & the exact char bounding box
			float rExactBBOverOrginalHeight    = ovCurrCharRegions[j].height / (float)oAnprObject.nMedianBlobHeight;
			float rBinCharMeanOnExactBB 	   = mean( oBestBinImages[i]( ovCurrCharRegions[j] ) )[0];
			float rBinCharMeanOnOriginalHeight = rBinCharMeanOnExactBB * rExactBBOverOrginalHeight;

			// DV: 03/07/2014 - this is a very safe condition
			// If needed we can add 1 extra condition to make this very robust
			if( !oLargestCC.empty() )
			{
				Rect oBoundingbox = boundingRect( oLargestCC );
				if(   (float)( oBoundingbox.width * oBoundingbox.height )
					< 0.3 * (float)ovCurrCharRegions[j].width * (float)oAnprObject.nMedianBlobHeight )	// DV: 04/07/2014 use median height
				{
					oAnprObject.oDebugLogs.info( "WARNING: BLOB %d IS MOSTLY NOISY, LARGEST CC AREA RATIO = %f",
												 j+1,
												 (float)( oBoundingbox.width * oBoundingbox.height )
												 / (float)(ovCurrCharRegions[j].width * (float)oAnprObject.nMedianBlobHeight ) );
					vnFlag2D.at<uchar>( i, j ) = 0;
				}
			}
			else if( rBinCharMeanOnOriginalHeight < FTS_IP_SimpleBlobDetector::CHAR_MIN_FILLED  )	// empty
			{
				oAnprObject.oDebugLogs.info( "WARNING: BLOB %d IS MOSTLY EMPTY, MEAN INTENSITY = %f",
											 j+1,
											 rBinCharMeanOnOriginalHeight );
				vnFlag2D.at<uchar>( i, j ) = 0;
			}
			else if(    ( rBinCharMeanOnExactBB > FTS_IP_SimpleBlobDetector::MIDDLE_CHAR_MAX_FILLED  )		// full
					 || ( bIsMostLeftorRight && rBinCharMeanOnExactBB > FTS_IP_SimpleBlobDetector::EDGE_CHAR_MAX_FILLED ) )
			{
				oAnprObject.oDebugLogs.info( "WARNING: BLOB %d IS MOSTLY FULL, MEAN INTENSITY = %f",
											 j+1,
											 rBinCharMeanOnExactBB );
				vnFlag2D.at<uchar>( i, j ) = 0;
			}
//			else if( (float)ovCurrCharRegions[j].height < 0.7* oBestCharBoxes[j].height )	// short
			// DV: 03/07/2014 - use median height
			else if( (float)ovCurrCharRegions[j].height < 0.7* oAnprObject.nMedianBlobHeight )	// short
			{
				oAnprObject.oDebugLogs.info( "WARNING: BLOB %d IS SHORT, HEIGHT = %d", ovCurrCharRegions[j].height );
				vnFlag2D.at<uchar>( i, j ) = 0;
			}
		}
	}

	for( size_t j = 0; j < oBestCharBoxes.size(); j++ )
	{
		oAnprObject.oDebugLogs.info( "BLOB %d: MEAN = %f", j, mean( vnFlag2D.col(j) )[0] );
	}

	vector< vector<Rect> > charRegionsFinal2D;
	vector<Mat> oBinsColor(oBestBinImages.size());
	for( size_t i = 0; i < oBestBinImages.size(); i++ )
	{
		if(this->m_bDebug) cvtColor( oBestBinImages[i], oBinsColor[i], CV_GRAY2BGR );

		vector<Rect> charRegionsFinal;
		for( size_t j = 0; j < charRegions[i].size(); j++ )
		{
			if( mean( vnFlag2D.col(j) )[0] <= 0.5 )
			{
				if(this->m_bDebug) rectangle( oBinsColor[i], charRegions[i][j], Scalar(0,0,255) );
			}
			else
			{
				charRegionsFinal.push_back( charRegions[i][j] );
				if(this->m_bDebug) rectangle( oBinsColor[i], charRegions[i][j], Scalar(0,255,0) );
			}
		}

		charRegionsFinal2D.push_back( charRegionsFinal );
	}

//	if(this->m_bDebug && this->m_bDisplayDbgImg) FTS_GUI_DisplayImage::ShowGroupScaleBy2( "bins", 1.0, oBinsColor, 1 );

	return charRegionsFinal2D;
}
