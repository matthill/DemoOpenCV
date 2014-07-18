/*
 * fts_ip_simpleblobdetector.h
 *
 *  Created on: May 7, 2014
 *      Author: sensen
 */

#ifndef FTS_IP_SIMPLEBLOBDETECTOR_H_
#define FTS_IP_SIMPLEBLOBDETECTOR_H_

#include "fts_base_externals.h"
#include "fts_ip_verticalhistogram.h"
#include "fts_base_linesegment.h"
#include "fts_gui_displayimage.h"
#include "fts_ip_verticalhistogram.h"
#include "fts_anpr_object.h"

class FTS_IP_SimpleBlobDetector
{

public:
	struct Params
	{
		Params();
		float thresholdStep;
		float minThreshold;
		float maxThreshold;
		size_t minRepeatability;
		bool useXDist;
		float minDistBetweenBlobs;

		bool filterByBBArea;
		bool useAdaptiveThreshold;
		int nbrOfthresholds;
		float minBBArea, maxBBArea;
		float minBBHoW, maxBBHoW;
		float minBBHRatio;

		bool filterByColor;
		uchar blobColor;

		bool filterByArea;
		float minArea, maxArea;

		bool filterByCircularity;
		float minCircularity, maxCircularity;

		bool filterByInertia;
		float minInertiaRatio, maxInertiaRatio;

		bool filterByConvexity;
		float minConvexity, maxConvexity;

		// DV: 16/06/2014 - enable/disable long line removal
		bool removeLongLine;
		float longLineLengthRatio;
				
		bool bDebug;
		bool bDisplayDbgImg;

		void read( const cv::FileNode& fn );
		void write( cv::FileStorage& fs ) const;
	};

	// Dont use moments, use normal x,y
	struct SimpleBlob
	{
		std::vector<cv::Point> oContour;
		cv::Rect oBB;

		std::string sStatus;

		SimpleBlob( const Rect& oRect = Rect() )
		{
			oBB = oRect;
			sStatus = s_sSTATUS_CANDIDATE;
		}

		cv::Rect toRect() const
		{
			return oBB;
		}
	};

	struct SimpleBlob_Old
	{
		std::vector<cv::Point> oContour;
		cv::Point2d location;
		double horzradius;
		double vertradius;
		double threshold;
		double confidence;

		std::string sStatus;

		SimpleBlob_Old( const Rect& oRect = Rect() )
		{
			location = Point2d( oRect.x + oRect.width  / 2, oRect.y + oRect.height / 2 );
			horzradius = (double)oRect.width / 2;
			vertradius = (double)oRect.height / 2;
		}

		cv::Rect toRect() const
		{
			return cv::Rect( floor( location.x - horzradius ),
						 floor( location.y - vertradius ),
						 ceil( horzradius * 2 ),
						 ceil( vertradius * 2 ) );
		}
	};

	struct less_than_x_coord
	{
	    inline bool operator()( const SimpleBlob& struct1, const SimpleBlob& struct2 )
	    {
	        return(   ( struct1.oBB.x < struct2.oBB.x ));
	    }
	};

	struct less_than_height
	{
		inline bool operator()( const SimpleBlob& struct1, const SimpleBlob& struct2 )
		{
			return(   struct1.oBB.height < struct2.oBB.height);
		}
	};

	static const std::string s_sSTATUS_CANDIDATE;
	static const std::string s_sSTATUS_OUTLIER;
	static const std::string s_sSTATUS_OUTLIER_RECONSIDERED;
	static const std::string s_sSTATUS_REMOVED;
	static const std::string s_sSTATUS_EDGE;
	static const std::string s_sSTATUS_EDGE_SAVED;

	// TODO: DV - settings
	static const float MIDDLE_CHAR_MAX_FILLED;
	static const float EDGE_CHAR_MAX_FILLED;
	static const float CHAR_MIN_FILLED;

	static int isSameSize( const void* poSegChar1, const void* poSegChar2, void* poSeg );
	static int isSameHeight( const void* poSegChar1, const void* poSegChar2, void* poSeg );
	static int isSameHeightAndClosed( const void* poSegChar1, const void* poSegChar2, void* poSeg );
	static int isClosed( const void* poSegChar1, const void* poSegChar2, void* poSeg );
	static int isSameY( const void* poSegChar1, const void* poSegChar2, void* poSeg );
	static int isSameYAndClosed( const void* poSegChar1, const void* poSegChar2, void* poSeg );

	static int calcXDistBetween2Blobs( const SimpleBlob& o1,
									   const SimpleBlob& o2 );

    vector<Point> getBoundingPolygonFromBlobs( const int cols,
		    const int rows,
		    const vector<SimpleBlob>& oBlobs,
		    const vector<bool>& goodIndices );

	explicit FTS_IP_SimpleBlobDetector(const FTS_IP_SimpleBlobDetector::Params &parameters = FTS_IP_SimpleBlobDetector::Params());
	virtual ~FTS_IP_SimpleBlobDetector();

	//  virtual void read( const cv::FileNode& fn );
	//  virtual void write( cv::FileStorage& fs ) const;

	void detect( const cv::Mat& oSrc, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
	void detectFTS( const cv::Mat& oSrc,
				  std::vector<SimpleBlob>& blobs,
				  bool bBlackChar = true,
				  const cv::Mat& mask=cv::Mat(),
				  int nOffsetX = 0,
				  int nOffsetY = 0 );

	void rotate( const Mat& oSrc,
				 const bool bBlackchar,
				 const Mat& mask,
				 const int nOffsetX,
				 const int nOffsetY,
				   	   Mat& oRotated );

	// DV: this function not just find middle cut, it finds:
	// 1. all candidate blobs
	// 2. the max number of blobs on a single line
	// 3. min, max x of all blobs
	int findMiddleCut( const Mat& oSrc,
	   	   	   	       const bool bBlackchar,
					   const Mat& mask,
					   int nOffsetX,
					   int nOffsetY,
					   Mat& oTopMask,
					   Mat& oBottomMask,
					   vector<SimpleBlob>& oTopBlobs,
					   vector<SimpleBlob>& oBottomBlobs,
					   int& nMinX,
					   int& nMaxX,
					   int& nMaxNbrOfBlobsPerLine   );

	void updateParams( const FTS_IP_SimpleBlobDetector::Params &parameters );

	void mergeBlobs( vector<SimpleBlob>& oBlobs, const int nMedianCharWidth  );
	void mergeBlobsVector( vector<Rect>& oBlobs, const int nMedianCharWidth  );

	vector<SimpleBlob> splitBlobs(
			const FTS_IP_VerticalHistogram& oVertHist,
			const vector<SimpleBlob>& oBlobs,
			const float& rMedianCharWidth,
			const float& rMedianCharHeight );

	vector<Rect> getBlobsByHist(
			const FTS_IP_VerticalHistogram& histogram,
			const FTS_BASE_LineSegment& top,
		    const FTS_BASE_LineSegment& bottom,
			const float rMedianCharWidth,
			const float rMedianCharHeight,
			float& score);

	vector<Rect> getBestBoxes( const Mat& img,
											 const vector<Rect>& oBlobs,
											 const float rMedianCharWidth,
											 const FTS_BASE_LineSegment& top,
											 const FTS_BASE_LineSegment& bottom );

//	vector<SimpleBlob> getBestBlobs( const Mat& img,
//										 const vector<SimpleBlob>& oBlobs,
//										 const float rMedianCharWidth,
//										 const FTS_BASE_LineSegment& top,
//										 const FTS_BASE_LineSegment& bottom );
	vector<Rect> get1DHits( const Mat& img,
						    const int yOffset,
						    const FTS_BASE_LineSegment& top,
						    const FTS_BASE_LineSegment& bottom );

	void adjustBlobHeight( vector<SimpleBlob>& oBlobs,
			  const float rMedianCharHeight,
			  const FTS_BASE_LineSegment& oTopLine,
			  const FTS_BASE_LineSegment& oBottomLine );

	void adjustBlobXY( vector<SimpleBlob>& oBlobs,
					   const FTS_IP_VerticalHistogram& oVertHist );

	void removeZombieBlobs( vector<SimpleBlob>& oBlobs,
						    const FTS_IP_VerticalHistogram& oVertHist,
						    const float& rMedianCharHeight );

	void filterEdgeBoxes( 	   vector<Mat>& thresholds,
						 const vector<Rect>& charRegions,
						 const float rMedianBoxWidth,
						 const float rMedianBoxHeight,
						 FTS_BASE_LineSegment& top,
						 FTS_BASE_LineSegment& bottom );

	vector<Rect> removeEmptyBoxes( const vector<Mat>&  thresholds,
										 const vector<Rect>& charRegions);

	int clusterBlobs( vector<SimpleBlob>& oBlobs,
					   CvCmpFunc is_equal );

//	cv::Mat getHistogram( const cv::Mat& oBin, const cv::Mat& oMask );


	vector< Mat > m_voBinarizedImages;
	vector < vector<SimpleBlob> > m_ovvAllBlobs;

	// Create memory oStorage
	CvMemStorage* m_poStorage;
	CvSeq* m_poSegCharSeq;
	std::vector<int> m_oIntVector;
	unsigned int m_nIsSameSizeAndNoOverlapMaxWidthDiff;
	unsigned int m_nIsSameSizeAndNoOverlapMaxHeightDiff;

	// DV: 23/06/2014
	// ANPR object
	FTS_ANPR_OBJECT* m_poANPRObject;

//	int m_nMinX;	// DV: move this calculation outside of this class
//	int m_nMaxX;

private:

	struct Center
	{
		cv::Point2d location;
		double radius;
		double confidence;
	};
	void detectImpl( const cv::Mat& oSrc, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
	void findBlobs(const cv::Mat &oSrc, const cv::Mat &binaryImage, std::vector<Center> &centers) const;

	void detectImplFTS( const cv::Mat& oSrc,
					    std::vector<SimpleBlob>& blobs,
					    bool bBlackChar,
					    const cv::Mat& mask=Mat() );
	void findBlobsFTS( const cv::Mat &oSrc,
					   cv::Mat &binaryImage,
					   const double thresholdVal,
					   std::vector<SimpleBlob>& blobs,
					   Rect& oMaxBB );

	void findAllBlobs( const Mat& oSrc,
					   const bool bBlackchar,
					   const Mat& mask,
					   const int nOffsetX,
					   const int nOffsetY,
							 vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& oBlobs );

	void removeOutliers( vector<SimpleBlob>& oBlobs);

	void  maskCleanBlobs( Mat& oSrc,
						  const vector<SimpleBlob>& oBlobs,
						  const bool bBlackchar,
						  const int nOffsetX,
						  int& nCut );
	void findTBLines( const Mat& oSrc,
					  const vector<SimpleBlob>& oBlobs,
					  vector<Point>& oPlateBoundingPolygon,
					  FTS_BASE_LineSegment& oTopLine,
					  FTS_BASE_LineSegment& oBottomLine );

	bool isTopHalf( const vector<SimpleBlob>& oBlobs,
						  FTS_BASE_LineSegment& oTopLine,
						  FTS_BASE_LineSegment& oBottomLine,
						  Mat& oColor );	// DEBUG

	void fillTBMasks( const Mat& oSrc,
					  const bool bIsTopHalf,
					  const vector<Point>& oPlateBoundingPolygon,
					  Mat& oTopMask,
					  Mat& oBottomMask,
					  Mat& oColor );	//DEBUG

	Mat getCharBoxMask( const Mat& oBin,
				   	    const vector<Rect>& charBoxes );

	int getLongestBlobLengthBetweenLines( const Mat& img,
										  const int col,
										  FTS_BASE_LineSegment& top,
										  FTS_BASE_LineSegment& bottom );

	//23.06 trungnt1 add to put debug logs to ANPR object
	int printLog(int level, const char* fmt, va_list ap);
	int printLogWarn(const char* fmt, ...);
	int printLogInfo(const char* fmt, ...);

	Params params;

};

#endif /* FTS_IP_SIMPLEBLOBDETECTOR_H_ */
