#ifndef _FTS_LV_ANPR_
#define _FTS_LV_ANPR_

#include "FTSAlgorithm.h"
#include "fts_anpr_engine.h"

#include <vector>

using namespace std;

static unsigned long COUNTER = 0;

class FTSANPR:
	public FTSAlgorithm {

protected:
	Fts_Anpr_Engine* pEngine;

	bool m_bDelayOnFrame;
	bool m_bDisplayDbgImg;
	int		m_iLogLevel;
	string  m_strLogFile;
	bool m_bOutCharSegments;

	std::string m_strDeviceID;
	std::string m_strOutDebugFolder;
#ifndef USING_SVM_OCR
	std::string m_strRuntimeOcrDir;
	std::string m_strOcrLanguage;
#else
	std::string m_strDigitModelsDir;
	std::string m_strLetterModelsDir;
	std::string m_strModelPairsDir;
	std::string m_strModelPairPrefix;
#endif // USING_SVM_OCR
	std::string m_strPostProcessFile;
	std::string m_strCascadeFile;

	int m_iMinCharToProcess;
	int m_iMaxCharToProcess;
	int m_iMinOcrFont;
	bool m_bOcrDebug;

	int m_iMinCharConf;
	int m_iConfSkipCharLevel;
	int m_iMaxSubstitutions;
	bool m_bPostProcessDebug;

	float m_fLPDScaleFactor;			//scale factor used in cascade.detectmultiscale
	int	  m_iLPDMinNeighbors;			//min neighbor used in cascade.detectmultiscale
	int   m_iLPDMinPlateWidth;			//min plate width
	int   m_iLPDMinPlateHeight;			//min plate height
	int   m_iLPDMaxPlateWidth;			//max plate width
	int   m_iLPDMaxPlateHeight;			//max plate height
	float m_fExpandRectX;				//expandRect in X-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	float m_fExpandRectY;				//expandRect in Y-axis: < 1.0 => percent-based, > 1 => increase by pixel size
	int   m_iTemplatePlateWidth;

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

	void InitANPREngine();

public:
	static std::string const className() { return "FTS_Anpr_Engine"; }
	virtual cv::AlgorithmInfo* info() const;
	virtual void read(const cv::FileNode& fn);
	virtual void write(cv::FileStorage& fs) const;
	virtual bool  operator() (ViolationEvent& e);

	void setOutDebugFolder(string dir, string deviceID) { m_strOutDebugFolder = dir; m_strDeviceID = deviceID; }

	FTSANPR();
	~FTSANPR();

	vector<string> files;
};

#endif