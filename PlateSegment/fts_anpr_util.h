/*
 *
 *
 * sn_nlpr_util.h
 *
 *
 */
#ifndef _FTS_ANPR_UTIL_H_
#define _FTS_ANPR_UTIL_H_

#include "fts_base_common.h"
#include "fts_base_stackarray.h"
#include "fts_ip_simpleblobdetector.h"



typedef struct
{
    int iStart;
    int iEnd;
    int iLabel;
} line_segment_struct;

/*!
 *
 *
 *
 *
 *
 */
class FTS_ANPR_Util
{
public:
    //! Returns the total area removed
//    static void RemoveSmallConnectedComponents( cv::Mat& oBinary,
//                                                cv::Mat& oTemp,
//                                                cv::Mat* poDst,  // pass in 0 to use oBinary as dst
//                                                CvMemStorage* poStorage,
//                                                double rAreaThresh,
//                                                CvScalar oFillColor = CV_RGB(0,0,0),
//                                                bool bFillContourBoundingBox = true //!< if true, fill the bounding box of the contour
//                                                                                     //!< with oColor, otherwise, only fill the contour with oColor.
//                                                );
    static bool RobustFitLinePDF( const std::vector<CvPoint2D32f>& oPoints,
                                  float rInlierPDFThresh,
                                  std::vector<int>& oInlierFlags
                                  );

    static void RemoveHorizontalLongLines( cv::Mat& oImage,
                                           cv::Rect oROI,
                                           unsigned int nLongLineLengthThreshold );

    static void RemoveVerticalLongLines( cv::Mat& oImage,
                                         cv::Rect oROI,
                                         unsigned int nLongLineLengthThreshold );

    static int RemoveLeftRightEdge(
            IplImage* poBinSrcImg,
            IplImage* poBinDstImg,
            int iEdgeLengthThreshold,
            float fEdgeHeightRatio );

    static int RemoveHorizontalLongLineOld(
                                  IplImage* poBinSrcImg,
                                  IplImage* poBinDstImg,
                                  int iEdgeLengthThreshold,
                                  int iLongLineThreshold );

    static int HorzLinearCut(
            IplImage* poGreyImg,
            bool bBlackChar,
            int nStartRow,
            int nEndRow );

	static double ComputeSkew(const cv::Mat& src, bool bDisplayRes);
	static void Deskew(cv::Mat& src, double angle, cv::Mat& cropped);

//	static double CalcOtsuThreshold(const cv::Mat& src);
	static bool LPMiddleCut(const cv::Mat& imgGrayFullLp, cv::Mat &imgUpperLp, cv::Mat &imgLowerLp, bool bPreferCenter = true);
	static double SumMatRows(cv::Mat mat, std::vector<int>& hist);

	static std::string toLowerCase(const std::string& in);
	static void getFilesInDirectory(const std::string& dirName, std::vector<std::string>& fileNames, const std::vector<std::string>& validExtensions);

	static int findMedianBlobWidthOfWoHInRange( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobs,
												const float rMinWoH,
												const float rMaxWoH,
												const int nMinMedian = 0 );

	static int findMedianBBWidthOfWoHInRange( const vector<Rect> oBlobs,
											  const float rMinWoH,
											  const float rMaxWoH,
											  const int nMinMedian = 0 );

	static int findMedianBlobWidth( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobs, const int nMinMedian = 0 );
	static int findMedianBBWidth( const vector<Rect> oBlobs, const int nMinMedian = 0  );

	static int findMedianBlobHeight( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobs, const int nMinMedian = 0  );
	static int findMedianBBHeight( const vector<Rect> oBlobs , const int nMinMedian = 0 );

	static void fillBlobWidthArray( FTS_BASE_StackArray<int>& oArray, const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );
	static void fillBBWidthArray  ( FTS_BASE_StackArray<int>& oArray, const vector<Rect>& blobs );

	static void fillBlobHeightArray( FTS_BASE_StackArray<int>& oArray, const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs );
	static void fillBBHeightArray  ( FTS_BASE_StackArray<int>& oArray, const vector<Rect>& blobs );

private:
	explicit FTS_ANPR_Util();

};



#endif // _FTS_ANPR_UTIL_H_
