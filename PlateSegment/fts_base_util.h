#ifndef _FTS_BASE_UTILITY_H_
#define _FTS_BASE_UTILITY_H_

#include "fts_base_common.h"
#include "fts_base_debug.h"

#include <vector>

long long getCurrentTimeInMS();

class FTS_BASE_Util
{
private:

	explicit FTS_BASE_Util();
	virtual ~FTS_BASE_Util();


public:

	enum
	{
		BINARYMAXVALUE = 255
	};

	static void Rotate(
	            const IplImage* piiSrc,
	                  IplImage* piiDst,
	            float rAngleDegrees,
	            bool bFillOutlierModeReplicateBorder = false,
	            CvScalar oFillOutlierWith = cvScalar(0)  );

	static bool parseTycoFilenameByRegex( const std::string& sFilename,
			const std::string& sRegex,
			std::string& sDate,
			std::string& sTime,
			std::string& sCamId,
			std::string& sZoneId,
			std::string& sFpm,
			std::string& sFileExt );

    static int OtsuAlgorithm( FTS_BASE_StackArray<double>& oHist );
    static void Tokenize( const std::string& sInput, const char* pcDelim, std::vector< std::string >& svTokens );

    static void CreateDirectory( const std::string& sDir );
    static void CreateDirectory( const std::string& sDir, bool& bExists );

    static cv::Rect enlargeRect(
                const cv::Rect oRect,
                float rEnlargeFactor,
                unsigned int nClipWidth,
                unsigned int nClipHeight );

    static void findReplace(
    		std::string& str,
    		const std::string& oldStr,
    		const std::string& newStr );

    static bool Exists (const std::string& name);

    static void CropRect( cv::Rect& oRect, unsigned int nWidth, unsigned int nHeight );

    static void removeNoises( const cv::Mat& oBin,
    								cv::Mat& oDst,
							  const unsigned int nMinDensity,
							  const unsigned int nMaxDensity,
							  const unsigned int nMinArea,
							  const unsigned int nMaxArea,
							  const float rMinHoWRatio,
							  const float rMaxHoWRatio,
							  const float rMinDensityRatio,
							  const float rMaxDensityRatio );

    static const unsigned int MIN_DENSITY_DEFAULT;
	static const unsigned int MAX_DENSITY_DEFAULT;
	static const unsigned int MIN_AREA_DEFAULT;
    static const unsigned int MAX_AREA_DEFAULT;
    static const float MIN_HOW_RATIO_DEFAULT;
    static const float MAX_HOW_RATIO_DEFAULT;
    static const float MIN_DENSITY_RATIO_DEFAULT;
    static const float MAX_DENSITY_RATIO_DEFAULT;

    static double distanceBetweenPoints( cv::Point p1, cv::Point p2);
    static float  angleBetweenPoints( cv::Point p1, cv::Point p2);

    static bool IsRectIntersect( const int X1, const int Y1, const int W1, const int H1,
    							 const int X2, const int Y2, const int W2, const int H2 );
	static bool IsRectIntersect(cv::Rect rect1, cv::Rect rect2);
	static void IntersectRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& intersectRect);
	static void UnionRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& unionRect);
	static std::vector<cv::Rect> CheckAndMergeOverlapRects(std::vector<cv::Rect>& src);
	static void extendRect(cv::Rect& rect, int padding);
};

#endif // _FTS_BASE_UTILITY_H_
