
#ifndef FTS_ANPR_ENGINE_H
#define FTS_ANPR_ENGINE_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "fts_ip_simpleblobdetector.h"
#include "fts_ip_util.h"

#include "fts_anpr_object.h"

#include <boost/thread/mutex.hpp>

#include "lsd.h"

struct Line
{
	int x1;
	int y1;
	int x2;
	int y2;
};

//#ifdef _WIN32
//#define USING_SVM_OCR
//#endif

#define DEFAULT_TOPN 25
#define DEFAULT_MINCHAR 4
#define DEFAULT_MAXCHAR 9
#define DEFAULT_MINOCRFONT 4
#define DEFAULT_OCRDEBUG true

#define DEFAULT_MINCHARCONF 40
#define DEFAULT_CONFSKIPLEVEL 60
#define DEFAULT_MAX_SUBSTITUTIONS 2
#define DEFAULT_POSTPROCESSDEBUG true

#define DEFAULT_LPD_SCALEFACTOR 1.2f
#define DEFAULT_LPD_MINNEIGHBOR 3
#define DEFAULT_LPD_MINPLATEWIDTH 40
#define DEFAULT_LPD_MINPLATEHEIGHT 21
#define DEFAULT_LPD_MAXPLATEWIDTH 250
#define DEFAULT_LPD_MAXPLATEHEIGHT 100
#define DEFAULT_LPD_EXPAND_RECT_X 0.6f
#define DEFAULT_LPD_EXPAND_RECT_Y 0.4f
#define DEFAULT_LPD_TEMPLATE_PLATE_WIDTH 256

#define DEFAULT_DETECT_REGION false
#define DEFAULT_PLATE_DETECT true

#define MAX_NUM_OF_SINGLE_LINE 9
#define MAX_NUM_OF_DUAL_LINE_TOP 4
#define MAX_NUM_OF_DUAL_LINE_BOTTOM 5

#define CROP_THEN_EXPAND

#define FIRST_EXPAND_BY_FIXED_PIXELS
//#define FIRST_EXPAND_BY_RATIO

#define FTSANPR_MAJOR_VERSION 1
#define FTSANPR_MINOR_VERSION 0
#define FTSANPR_PATCH_VERSION 0

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(FTS_ANPR_EXPORTS)
       #define FTS_ANPR_API __declspec(dllexport)
    #else
       #define FTS_ANPR_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
      #if defined(FTS_ANPR_EXPORTS) || defined(TESS_IMPORTS)
          #define FTS_ANPR_API  __attribute__ ((visibility ("default")))
      #else
          #define FTS_ANPR_API
      #endif
    #else
      #define FTS_ANPR_API
    #endif
#endif

enum FTS_ANPR_ERRCODE
{
    ANPR_ERR_NONE = 0,
	ANPR_ERR_ENGINE_NOT_INIT,
	ANPR_ERR_SOURCE_NULL,
    ANPR_ERR_NOPLATEDETECT,
    ANPR_ERR_SEGMENT_FAIL,
	ANPR_ERR_OCR_NULL,    
	ANPR_ERR_OCR_NOMATCH,
	ANPR_ERR_OCR_SUCCESS
}; 

enum FTS_ANPR_LOCALE
{
    ANPR_LOCALE_VN = 0,
    ANPR_LOCALE_AU
};

struct FTS_ANPR_API AnprParams
{
	float m_fLPDScaleFactor;			//scale factor used in cascade.detectmultiscale
	int	  m_iLPDMinNeighbors;			//min neighbor used in cascade.detectmultiscale
	int   m_iLPDMinPlateWidth;			//min plate width
	int   m_iLPDMinPlateHeight;			//min plate height
	int   m_iLPDMaxPlateWidth;			//max plate width
	int   m_iLPDMaxPlateHeight;			//max plate height
	float m_fExpandRectX;				//expandRect in X-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	float m_fExpandRectY;				//expandRect in Y-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	int   m_iTemplatePlateWidth;		//template plate after resize to recognize

	//SimpleBlobDetector params
	bool filterByArea;
	bool filterByBBArea;
	int  minBBArea;
	int  maxBBArea;
	float minBBHoW;
	float maxBBHoW;
	float minBBHRatio;
	float minDistBetweenBlobs;
	bool useXDist;
	bool useAdaptiveThreshold;
	int nbrOfthresholds;

	int nExpandTop;
	int nExpandBottom;
	int nExpandLeft;
	int nExpandRight;

	// DV: 21/07/2014 - Region
	int nLocale;

	AnprParams();
	void operator=(const AnprParams& src);
	void reset();
};

struct FTS_ANPR_API AlprPlate
{
  std::string characters;
  float overall_confidence;

  bool matches_template;
  //int char_confidence[];
};

struct FTS_ANPR_API AlprCoordinate
{
  int x;
  int y;
};

class FTS_ANPR_API AlprResult
{
  public:
    AlprResult();
    virtual ~AlprResult();

	int nErrorCode;

    int requested_topn;
    int result_count;

	int bestPlateIndex;
    AlprPlate bestPlate;
    std::vector<AlprPlate> topNPlates;

    float processing_time_ms;
    //AlprCoordinate plate_points[4];
	cv::Rect plateRect;

    int regionConfidence;
    std::string region;

	FTS_ANPR_OBJECT oAnprObject;
	void outputDebugInfo(std::string strOutputFolder, std::string strDeviceID, time_t timer, std::string frameID);
	void outputCharSegmentResult(std::string strOutputFolder);
};

class AlprImpl;
class FTS_ANPR_Rotate;
class FTS_ANPR_Cropper;

#ifndef USING_SVM_OCR
class FTS_ANPR_TessOcr;
#else
class FTS_ANPR_SvmOcr;
#endif

class FTS_ANPR_API Fts_Anpr_Engine
{
public:
#ifndef USING_SVM_OCR
    Fts_Anpr_Engine(const std::string runtimeOcrDir, const std::string ocrLanguage, 		
					const std::string patternFile, const std::string cascadeFile);
#else
	Fts_Anpr_Engine::Fts_Anpr_Engine(const std::string strDigitModelsDir,		//"./models/models_digits"
		const std::string strLetterModelsDir,		//"./models/models_letters"
		const std::string strModelPairsDir,
		const std::string strModelPairPrefix,
		const std::string patternFile, const std::string cascadeFile);
#endif
    ~Fts_Anpr_Engine();

    void setDetectRegion(bool detectRegion=false);
	void setRunPlateDetect(bool detectPlate=true);
    void setTopN(int topN=10);
    void setDefaultRegion(std::string region="");

	void setSingleLine(bool singleLine=false);
	void setBlackChar(bool blackChar=true);
	void setDebugMode(bool bDebug=false, bool bDelayOnFrame=false, bool bDisplayDbgImg=false);
	void setLogMode(int level=ANPR_LOG_ERROR, std::string destName="");
	
	bool isInitialized() { return m_bInit; }

	bool setParams(	
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
					int iMinCharToProcess = 4,
					int iMaxCharToProcess = 9,
					int iMinOcrFont = 4,
					bool bOcrDebug = true,
					int iMinCharConf = 40,
					int iConfSkipCharLevel = 60,
					int iMaxSubstitutions = 3,
					bool bPostProcessDebug = true,
					bool bEnableDetection = true);

	bool setParamsExpert(const AnprParams& params);

	void resetParamsToDefault();
	
	bool initEngine();

	int recognize(std::string filepath, std::vector<AlprResult>& result);
	int recognize(const cv::Mat& oSrc, std::vector<AlprResult>& result);

    std::string toJson(const std::vector<AlprResult> results);

    bool isLoaded();
    
    static std::string getVersion();

private:
    //AlprImpl* impl;
	bool m_bInit;
	bool m_bDebug;
	bool m_bDelayOnFrame;
	bool m_bDisplayDbgImg;
	int topN;
    bool detectRegion;
	bool m_bRunPlateDetect;
    std::string defaultRegion;

#ifndef USING_SVM_OCR
	//tesseract ocr
	std::string strRuntimeOcrDir;		//"./runtime_data/ocr/"
	std::string strOcrLanguage;			//"lvn"
#else
	//SVM ocr
	std::string strDigitModelsDir;		//"./models/models_digits"
	std::string strLetterModelsDir;		//"./models/models_letters"
	std::string strPairModelsDir;		//"./models/models_pairs
	std::string strPairModelPrefix;
#endif
	//post process
	std::string strPostProcessFile;		//"./runtime_data/postprocess/vn.patterns"
	std::string strCascadeFile;			//"./cascade/cascade_lbp_21x40_15000_22196_unfiltered.xml"

	int m_iMinCharToProcess;			//4
	int m_iMaxCharToProcess;			//9
	int m_iMinOcrFont;					//4
	bool m_bOcrDebug;					//true

	//Tesseract
	int m_iMinCharConf;					//40
	int m_iConfSkipCharLevel;			//60
	//svm
	int m_iSvmMinCharConf;				//0.1
	int m_iSvmConfSkipCharLevel;		//0.2

	int m_iMaxSubstitutions;			//2
	bool m_bPostProcessDebug;			//true

	bool bSingleLine;					//false
	bool bBlackChar;					//false
	
	AnprParams m_oParams;

	FTS_ANPR_Rotate* oRotate;
	FTS_ANPR_Cropper* oCropper;
#ifndef USING_SVM_OCR
	FTS_ANPR_TessOcr* oTessOcr;
#else
	FTS_ANPR_SvmOcr* oSvmOcr;
#endif
	cv::CascadeClassifier* plate_cascade;

	boost::mutex plateDetectMutex;
	boost::mutex segmentMutex;
	boost::mutex ocrMutex;

	int			m_logLevel;
	std::string m_strLogFileName;	

	int MAX_PLATE_WIDTH;	// TODO DV: 16/06/2014 - change this to force not to
							// expand if the width is already big enough
	FTS_IP_Util::ExpandByPixels plateExpand;

private:

	bool findAllBlobsIfNotDone( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oInputBlobs,
							    const cv::Mat& oSrc,
							    const cv::Mat& oMask,
							    const bool bSingleLine,
							    const int nPaddedBorder,
							    const int nMaxNbrOfChar,
							    FTS_ANPR_OBJECT& oAnprObject,
							    vector<Mat>& oBinaryImages,
							    FTS_IP_SimpleBlobDetector& myBlobDetector,
							    vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );
	bool refineBlobsInRange( const cv::Mat& oSrc,
						     const int nInputMinX,		// soft bound by cropping
						     const int nInputMaxX,		// soft bound by cropping
						     const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
						     const bool bUseLocalOtsu,
						     const int nThresholdType,
						     const int nPaddedBorder,
						     const Mat oLocalOtsuSubstitue,
						     const int nMinMedianWidth,
							 const int nMinMedianHeight,
						     const float rMinWoH,
						     const float rMaxWoH,
						     FTS_ANPR_OBJECT& oAnprObject,
						     vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobsWithinCroppedXRange,
						     int& nValidMinX,
						     int& nValidMaxX);

	void findPlateBoundaries( const cv::Mat& oSrc,
						      const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobs,
						      const int nMinArrSize,
						      const int nValidMinX,
						      const int nValidMaxX,
						      FTS_IP_SimpleBlobDetector& myBlobDetector,
						      FTS_ANPR_OBJECT& oAnprObject,
						      FTS_BASE_LineSegment& oTopLine,
						      FTS_BASE_LineSegment& oBottomLine,
						      Mat& oTBLinesMask );

	bool findBlobsByVerticalProjection( const cv::Mat& oSrc,
									    const int nPaddedBorder,
									    const int nMinMedianWidth,
									    const float rMinWoH,
									    const float rMaxWoH,
									    const Mat& oTBLinesMask,
									    const FTS_BASE_LineSegment& oTopLine,
									    const FTS_BASE_LineSegment& oBottomLine,
									    FTS_ANPR_OBJECT& oAnprObject,
									    vector<Mat>& oBinaryImages,
									    FTS_IP_SimpleBlobDetector& myBlobDetector,
									    vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
									    FTS_IP_VerticalHistogram& oVertHist );

	void mergeSplitBlobs( const cv::Mat& oSrc,
					      const int nMinMedianWidth,
					      const float rMinWoH,
					      const float rMaxWoH,
					      const FTS_BASE_LineSegment& oTopLine,
					      const FTS_BASE_LineSegment& oBottomLine,
					      const FTS_IP_VerticalHistogram& oVertHist,
					      FTS_ANPR_OBJECT& oAnprObject,
					      FTS_IP_SimpleBlobDetector& myBlobDetector,
					      vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

	bool refineBlobsBySize( const cv::Mat& oSrc,
							const int nMinMedianWidth,
							const float rMinWoH,
							const float rMaxWoH,
							const bool bUseLocalOtsu,
							const int nThresholdType,
							const Mat& oMask,
							const Mat& oLocalOtsuSubstitue,
							FTS_ANPR_OBJECT& oAnprObject,
							FTS_IP_SimpleBlobDetector& myBlobDetector,
							vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

	void removeNoisesByOtsu( const cv::Mat& oSrc,
							 const bool bUseLocalOtsu,
							 const int nThresholdType,
							 const int nMaxNbrOfChar,
							 FTS_ANPR_OBJECT& oAnprObject,
							 vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

	bool refineBlobsByHeuristic( const cv::Mat& oSrc,
								 const int nPaddedBorder,
								 const bool bUseLocalOtsu,
								 const int nThresholdType,
								 const int nMinMedianWidth,
								 const float rMinWoH,
								 const float rMaxWoH,
								 const Mat& oLocalOtsuSubstitue,
								 const vector<Mat>& oBinaryImages,
								 const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
								 FTS_ANPR_OBJECT& oAnprObject,
								 FTS_IP_SimpleBlobDetector& myBlobDetector,
								 vector<Rect>& newBestCharBoxes,
								 vector<Mat>& oMaskBinaries );

	void finalizeBinImagesForOCR( const vector<Mat>& oMaskBinaries,
								  const vector<Mat>& oBinaryImages,
								  const bool bUseLocalOtsu,
								  FTS_ANPR_OBJECT& oAnprObject );

	void fixVietnamTopLineBlobs( const int nMaxNbrOfChar,
								 FTS_ANPR_OBJECT& oAnprObject,
								 vector< vector<Rect> >& charRegionsFinal2D );

	vector< vector<Rect> >  removeNoisesAcrossBinaryImages( const vector<Mat> & oBestBinImages,
															const vector<Rect>& oBestCharBoxes,
															FTS_ANPR_OBJECT& oAnprObject );

	void storeFinalBlobs( const vector< vector<Rect> >& charRegionsFinal2D,
						  FTS_ANPR_OBJECT& oAnprObject );

	bool doSegment( const cv::Mat& oSrc,
				    const cv::Mat& oMask,
				    const bool bSingleLine,
				    const bool bBlackChar,
				    const int nInputMinX,		// soft bound by cropping
				    const int nInputMaxX,		// soft bound by cropping
				    const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oInputBlobs,
				    vector<Mat>& oBinaryImages,
				    const int nMaxNbrOfChar,
				    FTS_ANPR_OBJECT& oAnprObject,
				    bool bUseLocalOtsu = false );

	void plateDetect(const cv::Mat& oSrc, 
					 std::vector<Rect>& plates, long long& lPlateDetectTime);

	bool plateSegment(const cv::Mat& oSrc, 
					  const Rect& plateRect,
					  Rect& plateExpandRect,
					  FTS_ANPR_OBJECT& oAnprObject);

	void plateOcr( const cv::Mat& oSrc,
				   const Rect& plateRect,
				   const vector<Rect>& oBestCharBoxes,
				   const vector<Mat>& oBestBinImages,
				   AlprResult& result);

	void fillBlobWidthArray( FTS_BASE_StackArray<int>& oArray,
						 const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

	void fillBlobWidthArray2( FTS_BASE_StackArray<int>& oArray,
							  const std::vector<Rect>& blobs );

	void fillBlobHeightArray( FTS_BASE_StackArray<int>& oArray,
							  const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

	void fillBlobOtsuArray( FTS_BASE_StackArray<double>& oArray,
							const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
							const cv::Mat& oSrc );

	void findBlobsInXRange( const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobs,
							const int nMinX,
							const int nMaxX,
							vector<int>& nvGoodIndices );

	vector<FTS_IP_SimpleBlobDetector::SimpleBlob> getBlobsInXRange(
									const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobs,
									const int nMinX,
									const int nMaxX );

	void findBoxesInXRange( const vector<Rect>& oBoxes,
							const int nMinX,
							const int nMaxX,
							vector<int>& nvGoodIndices );

	vector<Rect> getBoxesInXRange( const vector<Rect>& oBoxes,
									const int nMinX,
									const int nMaxX );	
	
	bool isOutsideXRange( const Rect& oBox,
					  const int nMinX,
					  const int nMaxX );

	bool chopEdgeCharToMedianWidth( vector<Rect>& candidateBoxes );

	cv::Mat drawBlobs( const cv::Mat& oSrc,
					   const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

	void removeNoisyBlobs( std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );
	void moveNoisyBlobs( vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
						 vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& reservedNoisyBlobs );
	int countValidBlobs( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );
	void blobs2Rects( const std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs,
			std::vector<Rect> & rects,
			const int nPaddedX,
			const int nPaddedY,
			const int maxW,
			const int maxH );

	vector< vector<Rect> > getExactCharBB( const vector<Mat>& thresholds,
								 	   const vector< Rect >& oBestCharBoxes );
	
	vector< vector<Rect> > getCorrectSizedCharRegions( const vector<Mat>& oBestBinImages,
										    	 const vector< Rect >& oBestCharBoxes,
										    	 const vector< vector<Rect> >& charRegions,
												 FTS_ANPR_OBJECT& oAnprObject  );

	void Test_LSD( const Mat& img, Mat& oLinesImg );
};

#endif // FTS_ANPR_ENGINE_H
