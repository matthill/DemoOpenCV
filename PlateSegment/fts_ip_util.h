#ifndef _FTS_IP_UTILITY_H_
#define _FTS_IP_UTILITY_H_

#include "fts_base_common.h"
#include "fts_base_debug.h"
#include "fts_base_linesegment.h"
#include "fts_ip_simpleblobdetector.h"

#if defined( WIN32 )
static inline double round(double val)
{   
    return floor(val + 0.5);
}
#endif

class FTS_IP_Util
{
private:

	explicit FTS_IP_Util();
	virtual ~FTS_IP_Util();


public:

	struct ExpandByPixels
	{
		ExpandByPixels()
		{
			nT = 0;
			nB = 0;
			nL = 0;
			nR = 0;
		};

		int nT;
		int nB;
		int nL;
		int nR;
	};


public:

	enum
	{
		BINARYMAXVALUE = 255
	};


    static const float SAMPLE_CONST;

    static bool MaskByLargestCC( const cv::Mat& oGray,
    								   cv::Mat& oMask,
    							 const float rMinWidthRatio,
    							 bool bBlackChar = true );
    static std::vector<cv::Point> FindLargestCC( const cv::Mat& oBin );

    static void FindConvexHull( const cv::Size& oSize,
    					 const std::vector<cv::Point> points,
    					 cv::Mat& oConvexHull );

    static vector<Point> getBoundingPolygonFromContours( const int cols,
    													 const int rows,
														 vector<vector<Point> > contours,
														 vector<bool> goodIndices );

    static vector<Point> getBoundingPolygonFromBoxes( const int cols,
													  const int rows,
													  const vector<Rect >& oBoxes,
													  const vector<bool>& goodIndices );


    static vector<FTS_BASE_LineSegment> FindBoundingFromEdges( Mat edges, float sensitivityMultiplier, bool vertical);
    static cv::Rect MinAreaRect( const cv::Mat& oBin );
    static cv::Rect expandRectXY( const cv::Rect& original,
    						  const int& expandXPixels,
    						  const int& expandYPixels,
    						  const int& maxX,
    						  const int& maxY);
    static cv::Rect expandRectTBLR( const cv::Rect& original,
						        const int& top,
						        const int& bottom,
						        const int& left,
						        const int& right,
						        const int& maxX,
						        const int& maxY );
    static cv::Rect expandRectTBLR( const cv::Rect& original,
								const ExpandByPixels& exp,
								const int& maxX,
								const int& maxY );

    static void findMinMaxX( const vector < vector<FTS_IP_SimpleBlobDetector::SimpleBlob> >& ovvAllBlobs,
			  const int nXLimit,
			  const int nMinArrSize,
			  const FTS_BASE_LineSegment& oTopLine,
			  const FTS_BASE_LineSegment& oBottomLine,
			  int& nMinX,
			  int& nMaxX,
			  int& nADjustedMinX,
			  int& nADjustedMaxX );

    static void FindBlobsMinMaxX( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& ovAllBlobs,
			  int& nMinX,
			  int& nMaxX );

    static void RemoveNoisyBlobs( std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

    static std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> GetNRemoveNoisyBlobs( std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );

    static void setLabel( Mat& im, const string& label, vector<Point>& contour );

    static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);

    static cv::Rect getExactBB( const Mat& oSrcBin,
			  const cv::Rect& oBB,
			  const int nExpandX,
			  const int nExpandY );

    static void MorphBinary( Mat& oSrc, Mat& oDst, int nThresholdType );
};

#endif // _FTS_IP_UTILITY_H_
