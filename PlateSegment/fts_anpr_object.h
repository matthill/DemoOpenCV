#ifndef _FTS_ANPR_OBJECT_H
#define _FTS_ANPR_OBJECT_H

#include "fts_base_externals.h"

#define SKIP_CHAR '~'

struct Letter
{
  char letter;
  int charposition;
  float totalscore;
  int occurences;

  explicit Letter(char c, int pos, float conf)
  {
	  this->letter = c;
	  this->charposition = pos;
	  this->totalscore = conf;
	  this->occurences = 1;
  }
};

//++ 21.06 trung
class OcrResult
{
public:
	OcrResult();
	OcrResult(int minConf, int confSkipLevel);
	~OcrResult();

	void init(int minConf, int confSkipLevel);

	void addLetter(char letter, int charposition, float score);
	void clear();

	int m_iPostProcessMinConfidence;
	int m_iPostProcessConfidenceSkipLevel;
	vector<vector<Letter> > letters;

private:
	void insertLetter(char letter, int charPosition, float score);
};
//--

struct FTS_ANPR_PPResult
{
  string letters;
  float totalscore;
  bool matchesTemplate;
};

enum fts_anpr_log_level_t
{
    ANPR_LOG_NONE = 0,
    ANPR_LOG_FATAL = 1,
    ANPR_LOG_ERROR = 2,
    ANPR_LOG_WARN = 3,
    ANPR_LOG_INFO = 4
};

struct FTS_DEBUG_LOG
{
	int			Level;
	std::string Content;
};

class FTS_LOG_ITEMS
{
public:
	FTS_LOG_ITEMS();
	~FTS_LOG_ITEMS();

	void printf(int level);

	vector<FTS_DEBUG_LOG> vDebugLines;

	//OWN LOGGER STUFFs
	void setLogLevel(int level) { this->logLevel = level; }

	int log(int level, const char* fmt, ...)
    {
        va_list arglist;
        va_start(arglist, fmt);
        int ret = this->_log(level,fmt,arglist);
        va_end(arglist);
        return ret;
    }

	int log(int level, const char* fmt, va_list arglist)
    {
        int ret = this->_log(level,fmt,arglist);
        return ret;
    }

#define ANPR_LOG_METHOD(NAME,LEVEL) \
    int NAME(const char* fmt, ...) \
    { \
        va_list ap; \
        va_start(ap, fmt); \
        int ret = this->_log(LEVEL, fmt, ap); \
        va_end(ap); \
        return ret; \
    }
	
    ANPR_LOG_METHOD(none, ANPR_LOG_NONE)
    ANPR_LOG_METHOD(fatal, ANPR_LOG_FATAL)
    ANPR_LOG_METHOD(error, ANPR_LOG_ERROR)
    ANPR_LOG_METHOD(warn, ANPR_LOG_WARN)
    ANPR_LOG_METHOD(info, ANPR_LOG_INFO)

private:
	int logLevel;

	int _log(int level, const char* fmt, va_list arglist)
    {
        if (level > logLevel ) return -1;
		FTS_DEBUG_LOG newDbgLog;
		newDbgLog.Level = level;
		char buff[1024];
        int ret = vsprintf(buff, fmt, arglist);
		newDbgLog.Content = buff;
		vDebugLines.push_back(newDbgLog);
        return ret;
    }
};


class FTS_ANPR_OBJECT
{
public:
    FTS_ANPR_OBJECT();
    virtual ~FTS_ANPR_OBJECT();
	
	Rect plateRect;								//candidate plate region

	Mat oPlate;									//plate image original source => gray
	Mat oSrcRotated;							//plate image after rotated	: gray

	Mat oLinesImg;								//Line segment detection result => srcRotated no padded

	int nMedianBlobWidth;
	int nMedianBlobHeight;
	double rMediaBlobOtsu;
	Mat oMedOtsuThreshBinImg;					//srcRotated no padded

	Mat oLines;									//color, srcRotated, no padded top & bottom lines & mask

	int nAdjustedMinX, nAdjustedMaxX;

	vector<Mat> allHistograms;					//srcRotated no padded
	
	Mat oFirstBlobImg;											//srcRotated no padded
	Mat oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg;		//srcRotated no padded
	Mat otsuHistByMean;							//300,255
	Mat oTestEmptyBlob;							//srcRotated COLOR
	Mat oCleanCharBin;							//srcPadded	GRAY
	Mat oFindMiddleCut;							//srcPadded	GRAY

	vector<Rect> oBestCharBoxes;		//cal srcPadded
	vector<int>	oCharPosLine;			//29.06 trungnt1 add to specify line property of char segment boxes
	vector<Mat> oBestBinImages;			//srcPadded

	OcrResult ocrResults;					//single char ocr results from tesseract
	vector<FTS_ANPR_PPResult> ppResults;	//plate ocr results after post processing
	const vector<FTS_ANPR_PPResult>& getResults();

	long long lPlateDetectTime;
	long long lPlateSegmentTime;
	long long lPlateOcrTime;

	// DV: 23/06/2014 - more debug info
	vector<Mat> rawBins;
	vector< vector<Rect> > charRegions2D;

	Mat oOverviewDebugImg;
	Mat oOverviewDebugImg_Resized;	// DV: 28/06/2014 - resized
	void createOverviewDebugImg();

	void write(std::string strOutputFolder, std::string strDeviceID, time_t timer, std::string frameID);
	void outputCharSegmentResult(std::string strOutputFolder);	//25.06 trung add to export char segment grayscale to folders

	FTS_LOG_ITEMS oDebugLogs;
	void setLogLevel(int level);
	void printf(int level);



	static void extendImage(const cv::Mat &src, Rect &rect, Mat &dist);
	static void extendImage(const cv::Mat &src, Mat &dist);

	//vector<FTS_DEBUG_LOG> vDebugLines;
	//void printf(int level);

//	//OWN LOGGER STUFFs
//	void setLogLevel(int level) { this->logLevel = level; }
//
//	int log(int level, const char* fmt, ...)
//    {
//        va_list arglist;
//        va_start(arglist, fmt);
//        int ret = this->_log(level,fmt,arglist);
//        va_end(arglist);
//        return ret;
//    }
//
//	#define ANPR_LOG_METHOD(NAME,LEVEL) \
//    int NAME(const char* fmt, ...) \
//    { \
//        va_list ap; \
//        va_start(ap, fmt); \
//        int ret = this->_log(LEVEL, fmt, ap); \
//        va_end(ap); \
//        return ret; \
//    }
//
//    ANPR_LOG_METHOD(fatal, ANPR_LOG_FATAL)
//    ANPR_LOG_METHOD(error, ANPR_LOG_ERROR)
//    ANPR_LOG_METHOD(warn, ANPR_LOG_WARN)
//    ANPR_LOG_METHOD(info, ANPR_LOG_INFO)
//
//private:
//	int logLevel;
//
//	int _log(int level, const char* fmt, va_list arglist)
//    {
//        if (level > logLevel ) return -1;
//		FTS_DEBUG_LOG newDbgLog;
//		newDbgLog.Level = level;
//		char buff[1024];
//        int ret = sprintf(buff, fmt, arglist);
//		newDbgLog.Content = buff;
//		vDebugLines.push_back(newDbgLog);
//        return ret;
//    }
};

#endif // _FTS_ANPR_OBJECT_H
