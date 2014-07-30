
//#include "fts_anpr_cropper.h"
//#include "fts_anpr_nlcut.h"
#include "fts_anpr_util.h"
//#include "fts_anpr_seg.h"
//#include "fts_anpr_rotate.h"
//#include "fts_ip_simpleblobdetector.h"
//#include "fts_anpr_pcaocr.h"

//#include "fts_base_util.h"
//#include "fts_ip_util.h"

//#include "fts_gui_displayimage.h"

//#include "fts_anpr_object.h"
//#include "direct.h"
#ifdef WIN32
#include "dirent.h"
#else
#include <dirent.h>
#include <inttypes.h>
#endif

#include "fts_anpr_engine.h"

//#define SECOND_ATTEMPT_BLOB_DETECTION
//#define FILL_CONVEX_HULL
//#define DO_EDGE
//#define DETECT_BLOBS_AS_KEYPOINTS
//#define EXPAND_CHAR_BOX
//#define DISPLAY_ONE_WINDOW

#define MAX_NUM_OF_SINGLE_LINE 9
#define MAX_NUM_OF_DUAL_LINE_TOP 4
#define MAX_NUM_OF_DUAL_LINE_BOTTOM 5

//#define CROP_THEN_EXPAND

#define FIRST_EXPAND_BY_FIXED_PIXELS
//#define FIRST_EXPAND_BY_RATIO

#define GUI

#define EXPORT_CHAR_SEGMENTS

#ifndef WIN32
#define _DEBUG
#endif

//
//#ifdef WIN32
//static inline double round(double val)
//{   
//    return floor(val + 0.5);
//}
//#endif
//int index = 0;


bool bBlackChar = true;
bool bSingleLine = false;
#ifdef _DEBUG
bool m_bDebug = true;
bool m_bDelayOnFrame = true;
bool m_bDisplayDbgImg = true;
int m_iLogLevel = ANPR_LOG_INFO;
#else
bool m_bDebug = true;
bool m_bDelayOnFrame = false;
bool m_bDisplayDbgImg = false;
int m_iLogLevel = ANPR_LOG_WARN;
#endif
int MAX_PLATE_WIDTH  = INT_MAX;	// TODO DV: 16/06/2014 - change this to force not to
								// expand if the width is already big enough
// TODO: DV: setting
FTS_IP_Util::ExpandByPixels plateExpand;
static int frameIndex=0;

int64 getTimeMs64();

void testThresholding();
void testThresholding()
{
	Mat image = cv::imread("/home/sensen/data/20140626/PLATE/0.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	imshow( "Original", image );

	// Divide the image by its morphologically closed counterpart
	Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size(19,19) );
	Mat closed;
	morphologyEx( image, closed, MORPH_CLOSE, kernel );

	image.convertTo(image, CV_32F); // divide requires floating-point
	divide( image, closed, image, 1, CV_32F );
	normalize( image, image, 0, 255, NORM_MINMAX );
	image.convertTo( image, CV_8UC1 ); // convert back to unsigned int

	// Threshold each block (3x3 grid) of the image separately to
	// correct for minor differences in contrast across the image.
	int nHLength = image.cols/3;
	int nVLength = image.rows/3;
	for (int i = 0; i < 3; i++) {
	    for (int j = 0; j < 3; j++) {
	        Mat block = image.rowRange(nVLength*i, nVLength*(i+1)).colRange(nHLength*j, nHLength*(j+1));
	        threshold(block, block, -1, 255, THRESH_BINARY_INV | THRESH_OTSU);
	    }
	}

	imshow( "result", image );
	waitKey();
}

void testBlobDetector();
void testBlobDetector()
{
#ifndef WIN32
	std::string strPostProcessFile = "/home/sensen/ANPR/runtime_data/postprocess/vn.patterns";
	std::string strCascadeFile = "/home/sensen/ANPR/cascade/cascade_lbp_24x20_3k_5k.xml";
#else	
	std::string strPostProcessFile = "./runtime_data/postprocess/vn.patterns";
	std::string strCascadeFile = "./cascade/cascade_lbp_24x20_3k_5k.xml";
#endif

#ifdef USING_SVM_OCR
	Fts_Anpr_Engine* pEngine = new Fts_Anpr_Engine("./models/models_digits", 
													"./models/models_letters", 
													"./models/models_pairs", 
													"model", 
													strPostProcessFile, strCascadeFile);
#else
	Fts_Anpr_Engine* pEngine = new Fts_Anpr_Engine("./runtime_data/ocr/", "lus", strPostProcessFile, strCascadeFile);
#endif

	assert(pEngine != NULL);
	pEngine->setParams(	
#ifndef USING_SVM_OCR
						"./runtime_data/ocr/",		//tesseract runtime data dir to load trained data file
						"lus",						//language to train
#else
						"./models/models_digits",	//digit SVM models
						"./models/models_letters",	//letter SVM models
						"./models/models_pairs", 
						"model",					//modelPrefix
#endif
						strPostProcessFile,			//pattern file to validate ocr results
						strCascadeFile,				//cascade file to run plate detect
						4,							//minimum char to process ocr
						9,							//maximum char to process ocr
						4,							//min font size => use for TESSERACT only
						true,						//enable ocr debug
#ifndef USING_SVM_OCR
						60,							//TESSERACT MIN OCR CONFIDENT TO ACCEPT
						75,							//TESSERACT CONF LEVEL TO ADD SKIP CHAR
#else
						0.1,						//SVM+HOG MIN OCR CONFIDENT TO ACCEPT
						0.2,						//SVM+HOG CONF LEVEL TO ADD SKIP CHAR
#endif
						3,							//max substitutions
						true);						//enable post process debug
	//trungnt1 add expert params
	AnprParams params;
	params.m_fLPDScaleFactor = 1.05;		//scale factor used in cascade.detectmultiscale
	params.m_iLPDMinNeighbors = 2;			//min neighbor used in cascade.detectmultiscale
	params.m_iLPDMinPlateWidth = 40;		//min plate width
	params.m_iLPDMinPlateHeight = 21;		//min plate height
	params.m_iLPDMaxPlateWidth = 250;		//max plate width
	params.m_iLPDMaxPlateHeight = 100;		//max plate height
	params.m_fExpandRectX = 70;				//expandRect in X-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	params.m_fExpandRectY = 70;				//expandRect in Y-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	params.m_iTemplatePlateWidth = 256;		//template plate after resize to recognize
	
	params.filterByArea = false;
	params.filterByBBArea = true;
	params.minBBArea = 35;
	params.maxBBArea = 2500;
	params.minBBHoW = 0.4;
	params.maxBBHoW = 10.0;
	params.minBBHRatio = 0.15;
	params.minDistBetweenBlobs = 4.0f;
	params.useXDist = false;
	params.useAdaptiveThreshold = false;
	params.nbrOfthresholds = 5;

	params.nExpandTop = 10;
	params.nExpandBottom = 10;
	params.nExpandLeft = 10;
	params.nExpandRight = 10;
	
	pEngine->setParamsExpert(params);
	//
	pEngine->setRunPlateDetect(false);	//Run plate detect or ot depends on input source
	pEngine->setDebugMode(m_bDebug, m_bDelayOnFrame, m_bDisplayDbgImg);
	pEngine->setLogMode(m_iLogLevel, "");
	pEngine->initEngine();

#ifndef WIN32
	string sRootPath = "/home/sensen/data/20140626/PLATE/";	//"/home/sensen/data/India/cars/";	//2RowPlateImages/";
#else
	//string sRootPath = "c:\\temp\\testcrop\\MTData\\";
	//string sRootPath = "c:\\temp\\testcrop\\ShortLPs_Cropped2\\";
	//string sRootPath = "C:\\temp\\testcrop\\2RowPlateImages\\";
	//string sRootPath = "C:\\temp\\testcrop\\MTDataCropPlate\\";
	string sRootPath = "c:\\temp\\testcrop\\PLATE_CANTHO\\111_SEGMENTFAILED\\";
#endif
	std::vector<std::string> validExtensions;
	validExtensions.clear();
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");

	// Scan and sort file names
	std::vector<std::string> files;
	FTS_ANPR_Util::getFilesInDirectory(sRootPath, files, validExtensions);
	std::sort( files.begin(), files.end() );

	// Loop each file
	for( size_t i = 0; i < files.size(); i++ )
	{			
		// Load image file
		printf( "Loading %s\n", files[i].c_str() );
		cv::Mat oSrc = cv::imread( files[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
		if( oSrc.empty() )
		{
			continue;
		}

		cout << "Width: " << oSrc.cols << " Height: " << oSrc.rows << endl;
			
		if (!pEngine->isInitialized()) 
		{
			cout << "ERROR! NOT Init ANPR Engine!!!" << endl;
			break;
		}

		frameIndex++;
		time_t timer;
		time(&timer);

		vector<AlprResult> result;
		//
		//equalizeHist( oSrc, oSrc ); //17.07 Trungnt1 try using equalize histograms
		//
		int nLPRRes = pEngine->recognize(oSrc, result);
		printf( "Done processing %s - Error code = %d\n", files[i].c_str(), nLPRRes );
		for( size_t j = 0; j < result.size(); j++)
		{
			char buff[255];
			snprintf(buff, sizeof(buff), "%d_%d", frameIndex, j);
#ifndef WIN32
			result[j].outputDebugInfo("/home/sensen/data/debug/","CAM-001",timer,buff);
#else
			result[j].outputDebugInfo("c:\\temp\\output\\","IMG_CAM-111_SEGMENTFAILED_EQUALHIST",timer,buff);
#endif

			//++27.06 trung add code to out char segment if enabled
#ifdef EXPORT_CHAR_SEGMENTS
			result[j].outputCharSegmentResult("./Results_longlp");
#endif

			printf("Time to detect plate regions = %lld ms\n", result[j].oAnprObject.lPlateDetectTime);
			printf("Time to segment plate region = %lld ms\n", result[j].oAnprObject.lPlateSegmentTime);
			printf("Time to ocr plate region = %lld ms\n", result[j].oAnprObject.lPlateOcrTime);

			if(result.size() > 1 && m_bDelayOnFrame) cv::waitKey(0);
		}

		if(m_bDelayOnFrame) cv::waitKey(0);
	}
}

int main(int argc, char *argv[])
{
	// Test blob detector
	testBlobDetector();

//	testThresholding();

	return 1;
}



#ifdef WIN32
#include <time.h>
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

int64 getTimeMs64()
{
#ifndef WIN32
	struct timeval tv;

	gettimeofday(&tv, NULL);

	uint64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	ret += (tv.tv_sec * 1000);
#else
	time_t now;
	time(&now);

	uint64 ret = now*1000;
#endif

	return ret;
}
