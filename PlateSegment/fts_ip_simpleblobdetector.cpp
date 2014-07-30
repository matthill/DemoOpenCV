/*
 * fts_ip_simpleblobdetector.cpp
 *
 *  Created on: May 7, 2014
 *      Author: sensen
 */

#include "fts_ip_simpleblobdetector.h"
#include "fts_anpr_util.h"
#include "fts_base_binarizewolf.h"
#include "fts_ip_util.h"
#include <iterator>

#include "fts_anpr_object.h"

//#define DEBUG_HIST

FTS_IP_SimpleBlobDetector::Params::Params()
{
	thresholdStep = 10;
	minThreshold = 50;
	maxThreshold = 220;
	minRepeatability = 2;
	useXDist = true;
	minDistBetweenBlobs = 10;

	filterByBBArea = true;
	useAdaptiveThreshold = true;
	nbrOfthresholds = 5;
	minBBArea = 50;
	maxBBArea = 1000;
	minBBHoW = 0.4;
	maxBBHoW = 4.0;
	minBBHRatio = 0.4;

	filterByColor = false;
	blobColor = 0;

	filterByArea = false;
	minArea = 25;
	maxArea = 5000;

	filterByCircularity = false;
	minCircularity = 0.8f;
	maxCircularity = std::numeric_limits<float>::max();

	filterByInertia = false;
	minInertiaRatio = 0.1f;
	maxInertiaRatio = std::numeric_limits<float>::max();

	filterByConvexity = false;
	//minConvexity = 0.8;
	minConvexity = 0.95f;
	maxConvexity = std::numeric_limits<float>::max();

	removeLongLine = false;
	longLineLengthRatio = 1.0;	// safe

#ifdef _DEBUG
	bDebug = true;
	bDisplayDbgImg = true;
#else
	bDebug = false;
	bDisplayDbgImg = false;
#endif
}


const std::string FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE			( "blob_is_a_candidate" );
const std::string FTS_IP_SimpleBlobDetector::s_sSTATUS_OUTLIER				( "blob_is_an_outlier" );
const std::string FTS_IP_SimpleBlobDetector::s_sSTATUS_OUTLIER_RECONSIDERED	( "blob_is_an_outlier_but_reconsidered" );
const std::string FTS_IP_SimpleBlobDetector::s_sSTATUS_REMOVED				( "blob_will_be_removed" );
const std::string FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE					( "blob_is_edge" );
const std::string FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED			( "blob_is_edge_but_saved" );

// TODO: DV - settings
const float FTS_IP_SimpleBlobDetector::MIDDLE_CHAR_MAX_FILLED( 0.95 * 255 );
const float FTS_IP_SimpleBlobDetector::EDGE_CHAR_MAX_FILLED  ( 0.90 * 255 );
const float FTS_IP_SimpleBlobDetector::CHAR_MIN_FILLED		 ( 0.30 * 255 );

FTS_IP_SimpleBlobDetector::FTS_IP_SimpleBlobDetector(const FTS_IP_SimpleBlobDetector::Params &parameters)
	: m_voBinarizedImages()
	, m_ovvAllBlobs()
	, m_oIntVector()
	, m_nIsSameSizeAndNoOverlapMaxWidthDiff( 2 )
	, m_nIsSameSizeAndNoOverlapMaxHeightDiff( 4 )
	, m_poANPRObject( 0 )
	, params(parameters)
{
	m_poStorage = cvCreateMemStorage( 0 );

	// We need this to use opencv's partitioning algorithms.
	// Should be OK using the same storage, we don't really have any persistent content
	// in there. So m_poStorage is cleared pretty often, meaning there won't be much fragmentation.
	m_poSegCharSeq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(SimpleBlob*), m_poStorage );
}

FTS_IP_SimpleBlobDetector::~FTS_IP_SimpleBlobDetector()
{
	// Nothing
}

void FTS_IP_SimpleBlobDetector::updateParams( const FTS_IP_SimpleBlobDetector::Params &parameters )
{
	params = parameters;
}

void FTS_IP_SimpleBlobDetector::findBlobs( const cv::Mat &image,
										   const cv::Mat &binaryImage,
										   vector<Center> &centers) const
{
	(void)image;
	centers.clear();

	vector < vector<Point> > contours;
	Mat tmpBinaryImage = binaryImage.clone();
	findContours(tmpBinaryImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
	//  Mat keypointsImage;
	//  cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
	//
	//  Mat contoursImage;
	//  cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
	//  drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
	//  imshow("contours", contoursImage );
#endif

	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		Center center;
		center.confidence = 1;
		Moments moms = moments(Mat(contours[contourIdx]));
		if (params.filterByArea)
		{
			double area = moms.m00;
			if (area < params.minArea || area >= params.maxArea)
				continue;
		}

		if (params.filterByCircularity)
		{
			double area = moms.m00;
			double perimeter = arcLength(Mat(contours[contourIdx]), true);
			double ratio = 4 * CV_PI * area / (perimeter * perimeter);
			if (ratio < params.minCircularity || ratio >= params.maxCircularity)
				continue;
		}

		if (params.filterByInertia)
		{
			double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;
			double ratio;
			if (denominator > eps)
			{
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				double cosmax = -cosmin;
				double sinmax = -sinmin;

				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;
			}
			else
			{
				ratio = 1;
			}

			if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
				continue;

			center.confidence = ratio * ratio;
		}

		if (params.filterByConvexity)
		{
			vector < Point > hull;
			convexHull(Mat(contours[contourIdx]), hull);
			double area = contourArea(Mat(contours[contourIdx]));
			double hullArea = contourArea(Mat(hull));
			double ratio = area / hullArea;
			if (ratio < params.minConvexity || ratio >= params.maxConvexity)
				continue;
		}

		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		if (params.filterByColor)
		{
			if (binaryImage.at<uchar> (cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
				continue;
		}

		//compute blob radius
		{
			vector<double> dists;
			for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
			{
				Point2d pt = contours[contourIdx][pointIdx];
				dists.push_back(norm(center.location - pt));
			}
			std::sort(dists.begin(), dists.end());
			center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
		}

		centers.push_back(center);

#ifdef DEBUG_BLOB_DETECTOR
		//    circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
	}
#ifdef DEBUG_BLOB_DETECTOR
	//  imshow("bk", keypointsImage );
	//  waitKey();
#endif
}

void FTS_IP_SimpleBlobDetector::detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat&) const
{
	//TODO: support mask
	keypoints.clear();
	Mat grayscaleImage;
	if (image.channels() == 3)
		cvtColor(image, grayscaleImage, CV_BGR2GRAY);
	else
		grayscaleImage = image;

	vector < vector<Center> > centers;
	for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
	{
		Mat binarizedImage;
		threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

#ifdef DEBUG_BLOB_DETECTOR
		//    Mat keypointsImage;
		//    cvtColor( binarizedImage, keypointsImage, CV_GRAY2RGB );
#endif

		vector < Center > curCenters;
		findBlobs(grayscaleImage, binarizedImage, curCenters);
		vector < vector<Center> > newCenters;
		for (size_t i = 0; i < curCenters.size(); i++)
		{
#ifdef DEBUG_BLOB_DETECTOR
			//      circle(keypointsImage, curCenters[i].location, curCenters[i].radius, Scalar(0,0,255),-1);
#endif

			bool isNew = true;
			for (size_t j = 0; j < centers.size(); j++)
			{
				double dist = norm(centers[j][ centers[j].size() / 2 ].location - curCenters[i].location);
				isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][ centers[j].size() / 2 ].radius && dist >= curCenters[i].radius;
				if (!isNew)
				{
					centers[j].push_back(curCenters[i]);

					size_t k = centers[j].size() - 1;
					while( k > 0 && centers[j][k].radius < centers[j][k-1].radius )
					{
						centers[j][k] = centers[j][k-1];
						k--;
					}
					centers[j][k] = curCenters[i];

					break;
				}
			}
			if (isNew)
			{
				newCenters.push_back(vector<Center> (1, curCenters[i]));
				//centers.push_back(vector<Center> (1, curCenters[i]));
			}
		}
		std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));

#ifdef DEBUG_BLOB_DETECTOR
		//    imshow("binarized", keypointsImage );
		//waitKey();
#endif
	}

	for (size_t i = 0; i < centers.size(); i++)
	{
		if (centers[i].size() < params.minRepeatability)
			continue;
		Point2d sumPoint(0, 0);
		double normalizer = 0;
		for (size_t j = 0; j < centers[i].size(); j++)
		{
			sumPoint += centers[i][j].confidence * centers[i][j].location;
			normalizer += centers[i][j].confidence;
		}
		sumPoint *= (1. / normalizer);
		KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius));
		keypoints.push_back(kpt);
	}

#ifdef DEBUG_BLOB_DETECTOR
	if(params.bDebug)
	{
		namedWindow("keypoints", CV_WINDOW_NORMAL);
		Mat outImg = image.clone();
		for(size_t i=0; i<keypoints.size(); i++)
		{
			circle(outImg, keypoints[i].pt, keypoints[i].size, Scalar(255, 0, 255), -1);
		}
		//drawKeypoints(image, keypoints, outImg);
		if(params.bDisplayDbgImg) imshow("keypoints", outImg);
		waitKey();
	}
#endif
}

void FTS_IP_SimpleBlobDetector::detect( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask ) const
{
    keypoints.clear();

    if( image.empty() )
    {
        return;
    }
    assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );

    detectImpl( image, keypoints, mask );
}

void FTS_IP_SimpleBlobDetector::findBlobsFTS( const cv::Mat &oSrc,
											  cv::Mat &binaryImage,
											  const double thresholdVal,
											  std::vector<SimpleBlob>& blobs,
											  Rect& oMaxBB )
{
	//(void)oSrc;
	blobs.clear();

	vector < vector<Point> > contours;
	vector<Vec4i> hierarchy;

	std::vector<Point> approx;

	Mat tmpBinaryImage = binaryImage.clone();
	findContours(tmpBinaryImage, contours, hierarchy, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		// DV: In addition to moments, use the actual bounding box
		cv::Rect oBoundingbox = cv::boundingRect( contours[contourIdx] );

		if(    hierarchy[contourIdx][2] == -1
			&& hierarchy[contourIdx][3] != -1 )
		{
			if( oMaxBB.width < oBoundingbox.width )
			{
				oMaxBB = oBoundingbox;
			}
		}

		// DV: Only accept hierarchy 1
		if( hierarchy[contourIdx][3] != -1 )
		{
			continue;
		}

		if (params.filterByBBArea)
		{
			double area = oBoundingbox.width * oBoundingbox.height;
			double rHoW = (double) oBoundingbox.height / (double) oBoundingbox.width;
			double rHR  = (double) oBoundingbox.height / (double) binaryImage.rows;
//			printf( "Contour - x = %d, y = %d,  area = %f, rHoW = %f, rHR = %f",
//					oBoundingbox.x, oBoundingbox.y, area, rHoW, rHR );
			if(area < params.minBBArea || area >= params.maxBBArea)
			{
				// DV: remove small components as they should
				// not contribute to the histogram later
				if( area < params.minBBArea )
				{
					cv::drawContours( binaryImage,
									  contours,
									  contourIdx,
									  Scalar(0),
									  CV_FILLED );
				}
//				printf(" - NO by area\n");
				continue;
			}

			if( rHoW < params.minBBHoW || rHoW >= params.maxBBHoW)
			{
//				printf(" - NO by HoW, minBBHoW = %f\n", params.minBBHoW);
				continue;
			}

			if( rHR < params.minBBHRatio )
			{
				// DV: remove short components as they should
				// not contribute to the histogram later
//				cv::drawContours( binaryImage,
//								  contours,
//								  contourIdx,
//								  Scalar(0),
//								  CV_FILLED );
//				printf(" - NO by HR\n");
				continue;
			}
		}

//		printf(" - YES\n");

		// Store blob
		SimpleBlob blob( oBoundingbox );

		// DV: save blob contour
		blob.oContour = contours[contourIdx];

		// DV: save status
		blob.sStatus = s_sSTATUS_CANDIDATE;

		blobs.push_back(blob);
	}
}


void FTS_IP_SimpleBlobDetector::detectImplFTS( const cv::Mat& oSrc,
											   std::vector<SimpleBlob>& blobs,
											   bool bBlackChar,
											   const cv::Mat& mask )
{
	float rWhiteIntensityLim = 0.8 * 255;	// TODO DV: setting?

	int nThreshType = bBlackChar ? THRESH_BINARY_INV : THRESH_BINARY;
	blobs.clear();
	Mat grayscaleImage;
	if (oSrc.channels() == 3)
	{
		cvtColor(oSrc, grayscaleImage, CV_BGR2GRAY);
	}
	else
	{
		grayscaleImage = oSrc;
	}

	// TODO DV: from openalpr for better binary images?
	medianBlur( grayscaleImage, grayscaleImage, 3 );

	// Get a bunch of binary images either using otsu or adaptive depend on the params
	m_ovvAllBlobs.clear();
	m_voBinarizedImages.clear();
	vector< int > threshVals;

	// Wolf
	int k = 0, win=18;
	Mat binarizedImage1( grayscaleImage.size(), grayscaleImage.type() );
	FTS_BASE_BinarizeWolf (grayscaleImage, binarizedImage1, WOLFJOLION, win, win, 0.05 + (k * 0.35));
	bitwise_not(binarizedImage1, binarizedImage1);
	m_voBinarizedImages.push_back( binarizedImage1 );
	threshVals.push_back( 0 );

	k = 1;
	win = 22;
	Mat binarizedImage2( grayscaleImage.size(), grayscaleImage.type() );
	FTS_BASE_BinarizeWolf (grayscaleImage, binarizedImage2, WOLFJOLION, win, win, 0.05 + (k * 0.35));
	bitwise_not(binarizedImage2, binarizedImage2);
	m_voBinarizedImages.push_back( binarizedImage2 );
	threshVals.push_back( 0 );

//	// Sauvola
//	k = 1;
//	Mat binarizedImage3( grayscaleImage.size(), grayscaleImage.type() );
//	FTS_BASE_BinarizeWolf (grayscaleImage, binarizedImage3, SAUVOLA, 12, 12, 0.18 * k);
//	bitwise_not(binarizedImage3, binarizedImage3);
//	binarizedImages.push_back( binarizedImage3 );
//	threshVals.push_back( 0 );
//
//	k=2;
//	Mat binarizedImage4( grayscaleImage.size(), grayscaleImage.type() );
//	FTS_BASE_BinarizeWolf (grayscaleImage, binarizedImage4, SAUVOLA, 12, 12, 0.18 * k);
//	bitwise_not(binarizedImage4, binarizedImage4);
//	m_voBinarizedImages.push_back( binarizedImage4 );
//	threshVals.push_back( 0 );

	if( params.useAdaptiveThreshold )
	{
		for( int i = 0; i < params.nbrOfthresholds; i++ )
		{
			Mat binarizedImage;
			adaptiveThreshold( grayscaleImage,
							   binarizedImage,
							   255,
							   CV_ADAPTIVE_THRESH_MEAN_C,
							   nThreshType,
							   11 + 2 * i,
							   5 );

			m_voBinarizedImages.push_back( binarizedImage );
			threshVals.push_back( 3 + 2 * i );

			if( i == 0 )
			{
				if(params.bDebug && params.bDisplayDbgImg) 
					FTS_GUI_DisplayImage::ShowAndScaleBy2(
						"Adaptive Min", binarizedImage,
						FTS_GUI_DisplayImage::SCALE_X,
						FTS_GUI_DisplayImage::SCALE_Y,
						430, 0 );
			}
			if( i == params.nbrOfthresholds - 1 )
			{
				if(params.bDebug && params.bDisplayDbgImg) 
					FTS_GUI_DisplayImage::ShowAndScaleBy2(
						"Adaptive Max", binarizedImage,
						FTS_GUI_DisplayImage::SCALE_X,
						FTS_GUI_DisplayImage::SCALE_Y,
						430, 110 );
			}
		}
	}
	else
	{
		// 3 adaptive thresholds
		for( int i = 0; i < 3; i++ )
		{
			Mat binarizedImage;
			adaptiveThreshold( grayscaleImage,
							   binarizedImage,
							   255,
							   CV_ADAPTIVE_THRESH_MEAN_C,
							   nThreshType,
							   11 + 4 * i,
							   5 );
			m_voBinarizedImages.push_back( binarizedImage );
			threshVals.push_back( 11 + 4 * i );

//			printf( "MEAN = %f\n", mean(binarizedImage)[0] );
		}

		// 1 OTSU
		Mat binarizedImage;
//		double rOtsuThresh = threshold( grayscaleImage, binarizedImage, 0, 255, nThreshType | CV_THRESH_OTSU );
//		if( mean(binarizedImage)[0] > rWhiteIntensityLim )
//		{
////			printf( "OTSU THRESHOLD = %f, which is not good. Replace it by adaptive\n" );
//		}
//		else
//		{
//			m_voBinarizedImages.push_back( binarizedImage );
//			threshVals.push_back( rOtsuThresh );
//			if(params.bDebug) printLogInfo( "THE OTSU THRESHOLD = %f????????? MEAN = %f\n", rOtsuThresh, mean(binarizedImage)[0] );
//		}

		// DV: 30/06/2014 - MORPHOLOGICAL THRESHOLD
//		FTS_IP_Util::MorphBinary( grayscaleImage, binarizedImage, nThreshType );
//		m_voBinarizedImages.push_back( binarizedImage );
//		threshVals.push_back( 0 );
//
//		FTS_GUI_DisplayImage::ShowAndScaleBy2(
//								"MORPHHHHHHHHHH", binarizedImage,
//								FTS_GUI_DisplayImage::SCALE_X,
//								FTS_GUI_DisplayImage::SCALE_Y,
//								430, 110 );

//		// NIBLACK
//		k=0;
//		Mat binarizedImage4( grayscaleImage.size(), grayscaleImage.type() );
//		FTS_BASE_BinarizeWolf (grayscaleImage, binarizedImage4, NIBLACK, 20, 20, 0.18 * k);
//		bitwise_not(binarizedImage4, binarizedImage4);
//		m_voBinarizedImages.push_back( binarizedImage4 );
//		threshVals.push_back( 0 );

		// Sauvola
		k = 1;
		Mat binarizedImage3( grayscaleImage.size(), grayscaleImage.type() );
		FTS_BASE_BinarizeWolf (grayscaleImage, binarizedImage3, SAUVOLA, 12, 12, 0.18 * k);
		bitwise_not(binarizedImage3, binarizedImage3);
		m_voBinarizedImages.push_back( binarizedImage3 );
		threshVals.push_back( 0 );
	}

//	int m_nMinX = oSrc.cols;
//	int m_nMaxX = 0;
	vector<Mat> debugBinImages( m_voBinarizedImages.size() );
	for( size_t i = 0; i < m_voBinarizedImages.size(); i++ )
	{
		cv::Mat binarizedImage = m_voBinarizedImages[i];

		if( !mask.empty() )
		{
			cv::bitwise_and( binarizedImage, mask, binarizedImage );
		}

		// DV: 16/06/2014 - Remove long line if enabled
		if( params.removeLongLine )
		{
			Rect oROI( 0, 0, binarizedImage.cols, binarizedImage.rows );
			unsigned int nLongLineLengthThreshold = (int)( binarizedImage.cols * params.longLineLengthRatio );
			FTS_ANPR_Util::RemoveHorizontalLongLines( binarizedImage, oROI, nLongLineLengthThreshold );
		}

		vector < SimpleBlob > curSimpleBlobs;
		if(params.bDebug) printLogInfo( "Analysing binary image %d", i );
		if(params.bDebug) cvtColor( binarizedImage, debugBinImages[i], CV_GRAY2BGR );
		Rect oMaxBB;
		findBlobsFTS( grayscaleImage,
					  binarizedImage,
					  threshVals[i],
					  curSimpleBlobs,
					  oMaxBB );

		// DEBUG draw the max bounding box
		// DV: 16/06/2014 - remove drawing biggest BB lines
		//if( params.bDebug && oMaxBB.width > 0 )
		//{
		//	line( debugBinImages[i], Point( oMaxBB.x, 0 ),
		//			            	 Point( oMaxBB.x, debugBinImages[i].rows - 1 ), Scalar(0, 0,255 ), 2 );
		//	line( debugBinImages[i], Point( oMaxBB.x + oMaxBB.width, 0 ),
		//							 Point( oMaxBB.x + oMaxBB.width, debugBinImages[i].rows - 1 ), Scalar(0, 0,255 ), 2 );
		//}

		vector < vector<SimpleBlob> > newSimpleBlobs;
		if( curSimpleBlobs.size() == 0 )
		{
			if(params.bDebug) printLogInfo( "WEIRD, CAN'T FIND ANY BLOB. WRONG SIZE? TOO BIG, TOO SMALL?" );
		}
		for (size_t ii = 0; ii < curSimpleBlobs.size(); ii++)
		{
			if(params.bDebug) rectangle( debugBinImages[i], curSimpleBlobs[ii].oBB, Scalar( 0, 0, 255 ) );
			bool isNew = true;
			for (size_t j = 0; j < m_ovvAllBlobs.size(); j++)
			{
				for( size_t m = 0; m < m_ovvAllBlobs[j].size(); m++ )
				{
					// Distance
					float rCenterX1 = (float)( m_ovvAllBlobs[j][ m ].oBB.x ) + (float)( m_ovvAllBlobs[j][ m ].oBB.width ) / 2;
					float rCenterY1 = (float)( m_ovvAllBlobs[j][ m ].oBB.y ) + (float)( m_ovvAllBlobs[j][ m ].oBB.height ) / 2;
					float rCenterX2 = (float)( curSimpleBlobs[ii].oBB.x     ) + (float)( curSimpleBlobs[ii].oBB.width ) / 2;
					float rCenterY2 = (float)( curSimpleBlobs[ii].oBB.y     ) + (float)( curSimpleBlobs[ii].oBB.height ) / 2;
					double dist;
					if( params.useXDist )
					{
						dist = fabs( rCenterX1 - rCenterX2 );
						exit(-1);
					}
					else
					{
						dist = norm( Point2f(rCenterX1, rCenterY1 ) - Point2f(rCenterX2, rCenterY2 ) );
					}

					// Check if this is a new char of a new group
					isNew =    dist >= params.minDistBetweenBlobs;
					if (!isNew)
					{
//						printf( "Appending blob2: x =%d, y = %d, w = %d, h = %d to blob1: x = %d, y =%d, w = %d, h = %d | rCenterX1 = %f, rCenterX2 = %f, dist = %f\n",
//								curSimpleBlobs[ii].oBB.x, curSimpleBlobs[ii].oBB.y, curSimpleBlobs[ii].oBB.width, curSimpleBlobs[ii].oBB.height,
//								m_ovvAllBlobs[j][ m ].oBB.x, m_ovvAllBlobs[j][ m ].oBB.y, m_ovvAllBlobs[j][ m ].oBB.width, m_ovvAllBlobs[j][ m ].oBB.height,
//								rCenterX1, rCenterX2, dist );
						m_ovvAllBlobs[j].push_back(curSimpleBlobs[ii]);

						// Order blobs by width( ascending )
						size_t k = m_ovvAllBlobs[j].size() - 1;
						while(    k > 0
							   && m_ovvAllBlobs[j][k].oBB.width < m_ovvAllBlobs[j][k-1].oBB.width )
						{
							m_ovvAllBlobs[j][k] = m_ovvAllBlobs[j][k-1];
							k--;
						}
						m_ovvAllBlobs[j][k] = curSimpleBlobs[ii];

						break;
					}
				}
				if (!isNew)
				{
					break;
				}
			}
			if (isNew)
			{
				newSimpleBlobs.push_back(vector<SimpleBlob> (1, curSimpleBlobs[ii]));
			}
		}
		std::copy(newSimpleBlobs.begin(), newSimpleBlobs.end(), std::back_inserter(m_ovvAllBlobs));
	}

//	printf( "Detected %d blobsssssssssssssssssssssssssssss\n", m_ovvAllBlobs.size() );
	for (size_t i = 0; i < m_ovvAllBlobs.size(); i++)
	{
		if (m_ovvAllBlobs[i].size() < params.minRepeatability)
		{
			if(params.bDebug)	printLogInfo( "Group size = %d, ignore", m_ovvAllBlobs[i].size());
			continue;
		}

		// DV: sort by height( ascending )
//		std::sort( m_ovvAllBlobs[i].begin(), m_ovvAllBlobs[i].end(), less_than_height() );
		blobs.push_back( m_ovvAllBlobs[i][m_ovvAllBlobs[i].size() / 2] );
//		printf( "Blob %d:\n", i );
//		for( size_t j = 0; j < m_ovvAllBlobs[i].size(); j++ )
//		{
//			printf( "    Candidate %d: x = %f, y = %f, height = %f\n", j,
//					m_ovvAllBlobs[i][j].location.x, m_ovvAllBlobs[i][j].location.y, m_ovvAllBlobs[i][j].vertradius * 2 );
//		}
		if(params.bDebug) 
			printLogInfo( "Blob %d - %d: x = %d, y = %d, width = %d, height = %d", i, m_ovvAllBlobs[i].size() / 2,
				m_ovvAllBlobs[i][m_ovvAllBlobs[i].size() / 2].oBB.x, m_ovvAllBlobs[i][m_ovvAllBlobs[i].size() / 2].oBB.y,
				m_ovvAllBlobs[i][m_ovvAllBlobs[i].size() / 2].oBB.width,
				m_ovvAllBlobs[i][m_ovvAllBlobs[i].size() / 2].oBB.height );
//		blobs.push_back( m_ovvAllBlobs[i][m_ovvAllBlobs[i].size() / 2] );
	}

//	// Only accept min, max X coords of safe width
//	printf( "MinX = %d, MaxX = %d, width = %d\n", m_nMinX, m_nMaxX, oSrc.cols );
//	if( (m_nMaxX - m_nMinX ) < ceil( 0.65 * oSrc.cols ) )
//	{
//		m_nMinX = 0;
//		m_nMaxX = oSrc.cols - 1;
//	}
//	else
//	{
//		int nSafePadded = 2;	// TODO DV: setting
//		m_nMinX = m_nMinX - nSafePadded;
//		m_nMaxX = m_nMaxX + nSafePadded;
//		if( m_nMinX < 0 ) m_nMinX = 0;
//		if( m_nMaxX > oSrc.cols - 1 ) m_nMaxX = oSrc.cols - 1;
//	}
//	printf( "MinX = %d, MaxX = %d, width = %d\n", m_nMinX, m_nMaxX, oSrc.cols );

//	if(params.bDebug && params.bDisplayDbgImg)
//	{
//		FTS_GUI_DisplayImage::ShowGroupScaleBy2( "raw bins", 1.0, debugBinImages, 1 );
//	}

	// DV: 23/06/2014 : store the binary images
	if( m_poANPRObject )
	{
		m_poANPRObject->rawBins = debugBinImages;
	}
}

void FTS_IP_SimpleBlobDetector::detectFTS( const Mat& oSrc,
										   std::vector<SimpleBlob>& blobs,
										   bool bBlackChar,
										   const Mat& mask,
										   int nOffsetX,
										   int nOffsetY)
{
	blobs.clear();

    if( oSrc.empty() ) return;

//    assert( mask.type() == CV_8UC1 && mask.size() == oSrc.size() );

    detectImplFTS( oSrc, blobs, bBlackChar, mask );

    // Fix offset
//    printf( "Detected %d BLOBS\n", blobs.size() );
    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
		// Remember to offset as the binary has been padded
		blobs[i].oBB.x += nOffsetX;
		blobs[i].oBB.y += nOffsetY;

		// Fix width and height if needed
		if( blobs[i].oBB.x + blobs[i].oBB.width > ( oSrc.cols + nOffsetX ) )
		{
			blobs[i].oBB.width = ( oSrc.cols + nOffsetX ) - blobs[i].oBB.x;
		}

		if( blobs[i].oBB.y + blobs[i].oBB.height > ( oSrc.rows + nOffsetX ) )
		{
			blobs[i].oBB.height = ( oSrc.rows + nOffsetX ) - blobs[i].oBB.y;
		}

		// Fix contours coordinates too
		for( unsigned int j = 0; j < blobs[i].oContour.size(); j++ )
		{
			blobs[i].oContour[j].x += nOffsetX;
			blobs[i].oContour[j].y += nOffsetY;
		}
	}
}

void FTS_IP_SimpleBlobDetector::findAllBlobs( const Mat& oSrc,
											  const bool bBlackchar,
											  const Mat& mask,
											  const int nOffsetX,
											  const int nOffsetY,
						 	 	 	 	 	 	    vector<SimpleBlob>& oBlobs )
{
	oBlobs.clear();

	// Padding
	cv::Mat oSrcPadded;
	cv::copyMakeBorder( oSrc,
						oSrcPadded,
						abs(nOffsetY),
						abs(nOffsetY),
						abs(nOffsetX),
						abs(nOffsetX),
						IPL_BORDER_CONSTANT, bBlackchar?Scalar(255):Scalar(0)  );

	detectFTS( oSrcPadded,		// blobs are offset inside this function
			   oBlobs,
			   bBlackchar,
			   mask,
			   nOffsetX,
			   nOffsetY );
}

void FTS_IP_SimpleBlobDetector::removeOutliers( vector<SimpleBlob>& oBlobs)
{
	float rFilterCharCandOutlierThresh = 0.15;	// TODO: maybe just hard-coded???
	std::vector<CvPoint2D32f> oPoints2D32f;
	for( unsigned int i = 0; i < oBlobs.size(); i++ )
	{
		CvPoint2D32f p;
					 p.x = oBlobs[i].oBB.x + oBlobs[i].oBB.width/2;
					 p.y = oBlobs[i].oBB.y + oBlobs[i].oBB.height/2;
		oPoints2D32f.push_back( p );
	}

	std::vector<int>  oInlierFlags;
	cv::Mat oFirstBlobOutliersRemovedImg;
	if( FTS_ANPR_Util::RobustFitLinePDF( oPoints2D32f, rFilterCharCandOutlierThresh, oInlierFlags ) )
	{
		std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::iterator blobIter = oBlobs.begin();
		unsigned int iCount = 0;
		for ( ; blobIter != oBlobs.end(); iCount++ )
		{
			if( !oInlierFlags[iCount] )
			{
				blobIter->sStatus = s_sSTATUS_OUTLIER;
			}
			++blobIter;
		}
	}
}

void  FTS_IP_SimpleBlobDetector::maskCleanBlobs( Mat& oSrc,
												 const vector<SimpleBlob>& oBlobs,
												 const bool bBlackchar,
												 const int nOffsetX,
												 int& nCut )
{
	for( size_t i = 0; i < oBlobs.size(); i++ )
	{
		const Rect& oBox = oBlobs[i].oBB;
		Rect oRect;
		if ( ( oBox.x < ( oSrc.cols - 2 * abs( nOffsetX ) ) - 1 ) &&
			 ( oBox.height > 2 ) )
		{
			oRect = Rect( oBox.x, oBox.y, max( 0, oBox.width ), oBox.height - 2 );
		}
		else
		{
			oRect = oBox;
		}

		if ( oBox.height > oSrc.rows*0.6 )
		{
			nCut = oBox.y - 2 * abs( nOffsetX ) - 1 + oBox.height/2;
		}

		oSrc( oRect ) = bBlackchar ? Scalar( 0 ) : Scalar( 255 );
	}
}

void FTS_IP_SimpleBlobDetector::findTBLines( const Mat& oSrc,
				  	  	  	  	  	  	  	 const vector<SimpleBlob>& oBlobs,
				  	  	  	  	  	  	  	 vector<Point>& oPlateBoundingPolygon,
				  	  	  	  	  	  	  	 FTS_BASE_LineSegment& oTopLine,
				  	  	  	  	  	  	  	 FTS_BASE_LineSegment& oBottomLine )
{
	vector<bool> goodIndices;
	for( unsigned int i = 0; i < oBlobs.size(); i++ )
	{
		if( strcmp( oBlobs[i].sStatus.c_str(), s_sSTATUS_OUTLIER.c_str() ) == 0 )
		{
			goodIndices.push_back( false ); //trungnt1 fixed
			//continue;
		}

		goodIndices.push_back( true );
	}

	// TOP & BOTTOM LINES
	oPlateBoundingPolygon = getBoundingPolygonFromBlobs( oSrc.cols, oSrc.rows, oBlobs, goodIndices );

	oTopLine.init( oPlateBoundingPolygon[0].x,
				   oPlateBoundingPolygon[0].y,
				   oPlateBoundingPolygon[1].x,
				   oPlateBoundingPolygon[1].y );

	oBottomLine.init( oPlateBoundingPolygon[3].x,
					  oPlateBoundingPolygon[3].y,
					  oPlateBoundingPolygon[2].x,
					  oPlateBoundingPolygon[2].y );
}

bool FTS_IP_SimpleBlobDetector::isTopHalf( const vector<SimpleBlob>& oBlobs,
	  	  	  	 	 	 	 	 	 	         FTS_BASE_LineSegment& oTopLine,
	  	  	  	 	 	 	 	 	 	         FTS_BASE_LineSegment& oBottomLine,
	  	  	  	 	 	 	 	 	 	         Mat& oColor )
{
	bool bIsTopHalf = true;
	int nCountTop    = 0;
	int nCountBottom = 0;
	for( unsigned int i = 0; i < oBlobs.size(); i++ )
	{
		float rCenterX = (float)oBlobs[i].oBB.x + (float)oBlobs[i].oBB.width / 2;
		float rCenterY = (float)oBlobs[i].oBB.y + (float)oBlobs[i].oBB.height / 2;
		if( oTopLine.getPointAt( rCenterX ) > rCenterY )
		{
			nCountBottom++;

			if(params.bDebug) 
				printLogInfo( "Blob center X = %d, Y = %d, Top line Y =%f",
								oBlobs[i].oBB.x,
								oBlobs[i].oBB.y,
								oTopLine.getPointAt( rCenterX ) );

			// DEBUG
			if( !oColor.empty() )
			{
				rectangle( oColor, oBlobs[i].oBB, Scalar(128,0,128) );
			}
		}
		if( oBottomLine.getPointAt( rCenterX ) < rCenterY )
		{
			nCountTop++;

			if(params.bDebug) 
				printLogInfo( "Blob center X = %d, Y = %d, Top line Y =%f",
					oBlobs[i].oBB.x,
					oBlobs[i].oBB.y,
					oTopLine.getPointAt( rCenterX ) );

			// DEBUG
			if( !oColor.empty() )
			{
				rectangle( oColor, oBlobs[i].oBB, Scalar(255,0,0) );
			}
		}
	}

	if( nCountBottom >= nCountTop )
	{
		bIsTopHalf = false;
	}

	return bIsTopHalf;
}

void FTS_IP_SimpleBlobDetector::fillTBMasks( const Mat& oSrc,
											 const bool bIsTopHalf,
											 const vector<Point>& oPlateBoundingPolygon,
											 Mat& oTopMask,
											 Mat& oBottomMask,
											 Mat& oColor )
{
	vector<Point> oTopPoly, oBottomPoly;
	oTopPoly.push_back( Point(0,0) );
	oTopPoly.push_back( Point(oSrc.cols-1, 0) );

	oBottomPoly.push_back( Point(0, oSrc.rows-1) );
	oBottomPoly.push_back( Point(oSrc.cols-1, oSrc.rows-1) );
	if( bIsTopHalf )
	{
		if(params.bDebug) 
			printLogInfo( "Green blobs are of the TOP half of the plate" );
		Point oBR( oPlateBoundingPolygon[2] );
		Point oBL( oPlateBoundingPolygon[3] );

		if( oBR.y < oSrc.rows-1 ) ++oBR.y;	// 1 pixel for conservative
		if( oBL.y < oSrc.rows-1 ) ++oBL.y;

		oTopPoly.push_back( oBR );
		oTopPoly.push_back( oBL );
		oBottomPoly.push_back( oPlateBoundingPolygon[2] );
		oBottomPoly.push_back( oPlateBoundingPolygon[3] );

		if( !oColor.empty() )
		{
			cv::line( oColor, oBL, oBR, cv::Scalar( 0, 0, 255) );
		}
	}
	else
	{
		if(params.bDebug) 
			printLogInfo( "Green blobs are of the BOTTOM half of the plate" );

		Point oBR( oPlateBoundingPolygon[1] );
		Point oBL( oPlateBoundingPolygon[0] );

		if( oBR.y < oSrc.rows-1 ) --oBR.y;	// 1 pixel for conservative
		if( oBL.y < oSrc.rows-1 ) --oBL.y;

		oTopPoly.push_back( oPlateBoundingPolygon[1] );
		oTopPoly.push_back( oPlateBoundingPolygon[0] );
		oBottomPoly.push_back( oBR );
		oBottomPoly.push_back( oBL );

		if( !oColor.empty() )
		{
			cv::line( oColor, oBL, oBR, cv::Scalar( 0, 0, 255) );
		}
	}

	fillConvexPoly( oTopMask,
					oTopPoly.data(),
					oTopPoly.size(),
					Scalar(255,255,255));

	fillConvexPoly( oBottomMask,
					oBottomPoly.data(),
					oBottomPoly.size(),
					Scalar(255,255,255));
}

void FTS_IP_SimpleBlobDetector::rotate( const Mat& oSrc,
										const bool bBlackchar,
									    const Mat& mask,
									    const int nOffsetX,
									    const int nOffsetY,
									          Mat& oRotated )
{
	// Find all possible blobs
	std::vector<SimpleBlob> oAllBlobs;
	findAllBlobs( oSrc, bBlackchar, mask, nOffsetX, nOffsetY, oAllBlobs );

	// Return the results
	if( oAllBlobs.size() < 2 )
	{
		oRotated = oSrc.clone();
		return;
	}

	// Find sub blobs of similar Y and closed to each other, either top or bottom group
	std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oSameLineBlobs = oAllBlobs;
	int nNbrOfBlobsInTheLargestGroup = clusterBlobs( oSameLineBlobs, isSameYAndClosed );
	std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oRemainingBlobs = FTS_IP_Util::GetNRemoveNoisyBlobs( oSameLineBlobs );

	// Remove outliers
	removeOutliers( oSameLineBlobs );
	FTS_IP_Util::RemoveNoisyBlobs( oSameLineBlobs );

	// Return if cant find enough blobs on the same line
	if( oSameLineBlobs.size() < 2 )
	{
		oRotated = oSrc.clone();
		return;
	}

	// Find top & bottom lines
	vector<Point> oPlateBoundingPolygon;
	FTS_BASE_LineSegment oTopLine, oBottomLine;
	findTBLines( oSrc, oSameLineBlobs, oPlateBoundingPolygon, oTopLine, oBottomLine );

	// Find angle
	double rAngle = atan2( (double)(oTopLine.p1.y - oTopLine.p2.y), (double)(oTopLine.p2.x - oTopLine.p1.x) );
	rAngle = rAngle * 180/ CV_PI;

	if(params.bDebug) 
		printLogInfo( "ANGLE = %f", rAngle );

	 // Source image
	IplImage oSrcIpl = oSrc;

	// Destination image
	oRotated.create( oSrc.size(), oSrc.type() );
	IplImage oDstIpl = oRotated;

	if( rAngle >= 45 )
	{
		FTS_BASE_Util::Rotate( &oSrcIpl, &oDstIpl, (90 - rAngle), true );
	}
	else
	{
		FTS_BASE_Util::Rotate( &oSrcIpl, &oDstIpl, (360 - rAngle), true );
	}

//	Mat oColor;
//	cvtColor( oSrc, oColor, CV_GRAY2BGR );
//	line( oColor, oTopLine.p1, oTopLine.p2, Scalar(0,0,255), 1 );
//	FTS_GUI_DisplayImage::ShowAndScaleBy2( "Chop", oColor, 2.0, 2.0, 200, 200 );
//	FTS_GUI_DisplayImage::ShowAndScaleBy2( "Rotate", oRotated, 2.0, 2.0, 400, 400 );
}

int FTS_IP_SimpleBlobDetector::findMiddleCut( const Mat& oSrc,
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
										      int& nMaxNbrOfBlobsPerLine )
{
	int nCut = 0;   // return value

	// Clone the source image
	Mat oCopy = oSrc.clone();
	oTopMask = Mat::zeros( oSrc.size(), oSrc.type() );
	oBottomMask = Mat::zeros( oSrc.size(), oSrc.type() );

	// Find all possible blobs
	std::vector<SimpleBlob> oAllBlobs;
	findAllBlobs( oCopy, bBlackchar, mask, nOffsetX, nOffsetY, oAllBlobs );

	// Find blobs closed to each other
	int nNbrOfBlobsInTheLargestGroup = clusterBlobs( oAllBlobs, isSameHeightAndClosed );
	if(params.bDebug)  
		printLogInfo( "No of blobs in the largest group by position = %d", nNbrOfBlobsInTheLargestGroup );
	FTS_IP_Util::RemoveNoisyBlobs( oAllBlobs );

	// color debug
	Mat oColor;
	cvtColor( oCopy, oColor, CV_GRAY2BGR );

	// Return the results
	if( oAllBlobs.size() < 2 )
	{
		if(params.bDebug)  
			printLogInfo( "RANDOM : nCut = %d", nCut );
		nCut = (int)( (double)oSrc.rows * 0.5f ) + nOffsetY;

		oTopMask   ( Rect( 0, 0, oTopMask.cols, nCut + 1 ) ) = Scalar(255);
		oBottomMask( Rect( 0, nCut, oBottomMask.cols, oBottomMask.rows - nCut - 1 ) ) = Scalar(255);
	}
	else
	{
		// Find sub blobs of similar Y and closed to each other, either top or bottom group
		std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oSameLineBlobs = oAllBlobs;
		nNbrOfBlobsInTheLargestGroup = clusterBlobs( oSameLineBlobs, isSameYAndClosed );
		std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oRemainingBlobs = FTS_IP_Util::GetNRemoveNoisyBlobs( oSameLineBlobs );
		if(params.bDebug)  
			printLogInfo( "Remaining blobs = %d", oRemainingBlobs.size() );

		// Remove outliers
		removeOutliers( oSameLineBlobs );

		if(params.bDebug)  
			printLogInfo( "No of blobs in the largest group by Y & position = %d", nNbrOfBlobsInTheLargestGroup );
		if(    oSameLineBlobs.size() < 2 )
//			|| oSameLineBlobs.size() == oAllBlobs.size() )	// all blobs are on the same line
		{
			// Mask clean blobs
			maskCleanBlobs( oCopy, oAllBlobs, bBlackchar, nOffsetX, nCut );

			// Cut by finding the local minimum of horizontal projection
			if ( nCut == 0 )
			{
				IplImage oII = oCopy;
				nCut = FTS_ANPR_Util::HorzLinearCut(
						&oII,
						bBlackchar,
						oCopy.rows/3,
						oCopy.rows*2/3 );

				oTopMask   ( Rect( 0, 0, oTopMask.cols, nCut + 1 ) ) = Scalar(255);
				oBottomMask( Rect( 0, nCut, oBottomMask.cols, oBottomMask.rows - nCut - 1 ) ) = Scalar(255);
			}

			line( oColor, Point(0, oColor.rows/3), Point(oColor.cols-1, oColor.rows/3), Scalar(0,255,255), 1 );
			line( oColor, Point(0, oColor.rows*2/3), Point(oColor.cols-1, oColor.rows*2/3), Scalar(0,255,255), 1 );
			line( oColor, Point(0, nCut), Point(oColor.cols-1, nCut), Scalar(0,0,255), 1 );
		}
		else	// found either top or bottom half
		{
			// Find top & bottom lines
			vector<Point> oPlateBoundingPolygon;
			FTS_BASE_LineSegment oTopLine, oBottomLine;
			findTBLines( oCopy, oSameLineBlobs, oPlateBoundingPolygon, oTopLine, oBottomLine );

			// Check if this is the top or bottom half of the plate
			bool bIsTopHalf = isTopHalf( oAllBlobs, oTopLine, oBottomLine, oColor );

			// Find blobs on the remaining line
			if( oRemainingBlobs.size() > 1 )
			{
				nNbrOfBlobsInTheLargestGroup = clusterBlobs( oRemainingBlobs, isSameYAndClosed );

				// DV: 16/06/2014 - if the largest group has only 1 blob
				// 					then pick blob that are not cut through by the middle line
				if( nNbrOfBlobsInTheLargestGroup == 1 )
				{
					  if(m_poANPRObject) 
							m_poANPRObject->oDebugLogs.info( "In remaining blobs, the largest group has only 1 blob. Pick the best one" );
					for( size_t nIdx = 0; nIdx < oRemainingBlobs.size(); ++nIdx )
					{
						bool bIsNoise = false;

						if( bIsTopHalf )
						{
							if( !oBottomLine.isPointBelowLine( oRemainingBlobs[nIdx].oBB.tl() ) )
							{
								bIsNoise = true;
							}
						}
						else
						{
							if( oTopLine.isPointBelowLine( oRemainingBlobs[nIdx].oBB.br() ) )
							{
								bIsNoise = true;
							}
						}

						if( bIsNoise )
						{
							oRemainingBlobs[nIdx].sStatus = s_sSTATUS_REMOVED;
						}
						else
						{
							oRemainingBlobs[nIdx].sStatus = s_sSTATUS_CANDIDATE;
						}
					}
				}
			}

			// Fill top & bottom masked polygons
			fillTBMasks( oCopy, bIsTopHalf, oPlateBoundingPolygon, oTopMask, oBottomMask, oColor );

			// DEBUG
			for( size_t i = 0; i < oSameLineBlobs.size(); i++ )
			{
				Rect& oBox = oSameLineBlobs[i].oBB;
				if( strcmp( oSameLineBlobs[i].sStatus.c_str(), s_sSTATUS_OUTLIER.c_str() ) == 0 )
				{
					rectangle( oColor, oBox, Scalar(0,0,255) );
				}
				else
				{
					rectangle( oColor, oBox, Scalar(0,255,0) );
				}
			}

			// Remove non-blobs
			FTS_IP_Util::RemoveNoisyBlobs( oSameLineBlobs );
			FTS_IP_Util::RemoveNoisyBlobs( oRemainingBlobs );

			if(params.bDebug)
			{
				printLogInfo( "No of blobs in the largest group after removing outliers = %d\n", oSameLineBlobs.size() );
			}

			// output top and bottom blobs
			if( bIsTopHalf )
			{
				oTopBlobs    = oSameLineBlobs;
				oBottomBlobs = oRemainingBlobs;

				// DV:24/07/2014 - store the middle line
				m_poANPRObject->middleLine.init( oBottomLine.p1.x, oBottomLine.p1.y, oBottomLine.p2.x, oBottomLine.p2.y );
			}
			else
			{
				oTopBlobs    = oRemainingBlobs;
				oBottomBlobs = oSameLineBlobs;

				// DV:24/07/2014 - store the middle line
				m_poANPRObject->middleLine.init( oTopLine.p1.x, oTopLine.p1.y, oTopLine.p2.x, oTopLine.p2.y );
			}


		}

		// Merge blobs from 2 lines
		vector<SimpleBlob> oFinalBlobs;
		oFinalBlobs.reserve( oTopBlobs.size() + oBottomBlobs.size() ); // preallocate memory
		oFinalBlobs.insert( oFinalBlobs.end(), oTopBlobs.begin(), oTopBlobs.end() );
		oFinalBlobs.insert( oFinalBlobs.end(), oBottomBlobs.begin(), oBottomBlobs.end() );

		// DEBUG
		// Draw min, max X vertical lines
		FTS_IP_Util::FindBlobsMinMaxX( oFinalBlobs, nMinX, nMaxX );
		line( oColor, Point(nMinX, 0), Point(nMinX, oColor.rows-1), Scalar(0,255,255), 1 );
		line( oColor, Point(nMaxX, 0), Point(nMaxX, oColor.rows-1), Scalar(0,255,255), 1 );
	}

	m_poANPRObject->oFindMiddleCut = oColor;

	return nCut;
}

vector<FTS_IP_SimpleBlobDetector::SimpleBlob> FTS_IP_SimpleBlobDetector::splitBlobs(
														const FTS_IP_VerticalHistogram& oVertHist,
														const vector<SimpleBlob>& oBlobs,
														const float& rMedianCharWidth,
														const float& rMedianCharHeight )
{
	vector<SimpleBlob> oNewBlobs;
	float rMaxBlobWidth  = rMedianCharWidth * 1.35;	// TODO: DV: from setting?
	float rMinHistHeight = rMedianCharHeight * 0.3;	// TODO: DV: from setting?

	for( size_t i = 0; i < oBlobs.size(); i++)
	{
		cv::Rect oBlobRect = oBlobs[i].toRect();
		if(    (float)oBlobRect.width > rMedianCharWidth * 2.5	// DV: 16/06/2014 - 1.9 --> 2.5
			&& (float)oBlobRect.width < rMaxBlobWidth * 2 )	// TODO: 1.9 is tuned?
		{
			int nBeginCol = oBlobRect.x + (int) ( oBlobRect.width * 0.4f );
			int nEndCol   = oBlobRect.x + (int) ( oBlobRect.width * 0.6f );

			int minX = oVertHist.getLocalMinimum( nBeginCol, nEndCol );
			int maxXChar1 = oVertHist.getLocalMaximum( oBlobRect.x, minX );
			int maxXChar2 = oVertHist.getLocalMaximum( minX, oBlobRect.x + oBlobRect.width );
			int minHeight = oVertHist.getHeightAt( minX );

			int maxHeightChar1 = oVertHist.getHeightAt( maxXChar1);
			int maxHeightChar2 = oVertHist.getHeightAt( maxXChar2);

			if(params.bDebug) 
			{
				printLogInfo( "Split blob %d : width = %d, rMedianCharWidth = %f", i, oBlobRect.width, rMedianCharWidth );
				printLogInfo( "maxHeightChar1 = %d, rMinHistHeight = %f, minHeight = %d, maxHeightChar1 = %d",
								maxHeightChar1, rMinHistHeight, minHeight, maxHeightChar1 );
			}

			bool bChar1 = false;
			bool bChar2 = false;
			if( maxHeightChar1 > rMinHistHeight && minHeight < (0.5 * ((float) maxHeightChar1) ) )
			{
				bChar1 = true;

				// Add a box for Char1
				Point botRight = Point( minX - 1, oBlobRect.y + oBlobRect.height );
				cv::Rect oNewRect = Rect( oBlobRect.tl(), botRight);
				SimpleBlob sb( oNewRect );
				sb.sStatus    = oBlobs[i].sStatus;
				oNewBlobs.push_back( sb );

				if(params.bDebug)   
					printLogInfo( "Char 1: x = %d, y = %d, w = %d, h = %d",
								oNewRect.x, oNewRect.y, oNewRect.width, oNewRect.height );
			}
			if (maxHeightChar2 > rMinHistHeight && minHeight < (0.5 * ((float) maxHeightChar2)))
			{
				bChar2 = true;

				// Add a box for Char2
				Point topLeft = Point( minX + 1, oBlobRect.y );
				cv::Rect oNewRect = Rect(topLeft, oBlobRect.br());
				SimpleBlob sb( oNewRect );
				sb.sStatus    = oBlobs[i].sStatus;
				oNewBlobs.push_back( sb );

				if(params.bDebug)   
					printLogInfo( "Char 2: x = %d, y = %d, w = %d, h = %d",
								oNewRect.x, oNewRect.y, oNewRect.width, oNewRect.height );
			}

			// No split, keep the original blob
			if( !bChar1 && !bChar2 )
			{
				oNewBlobs.push_back(oBlobs[i]);
			}
		}
		else
		{
			oNewBlobs.push_back(oBlobs[i]);
		}
	}

	return oNewBlobs;
}

void FTS_IP_SimpleBlobDetector::mergeBlobs( vector<SimpleBlob>& oBlobs,
											const int nMedianCharWidth  )
{
	float rOverlapWidthThresh  = 0.5;	// TODO DV: setting?
	float rOverlapHeightThresh = 0.5;	// TODO DV: setting?

	float rNonOverlapWidthThresh  = 0.5;	// TODO DV: setting?
	float rNonOverlapHeightThresh = 0.5;	// TODO DV: setting?

	vector<SimpleBlob>::iterator it = oBlobs.begin();
	int iCount = 0;
	for ( ; it != oBlobs.end(); )
	{
		// If this is the last element
		vector<SimpleBlob>::iterator itNext = it+1;
		if ( itNext == oBlobs.end() )
		{
			break;
		}

		// Get the 2 bounding rectangles
		cv::Rect oBlob1 = it->toRect();
		cv::Rect oBlob2 = itNext->toRect();

		// Find intersect and merged blobs
		cv::Rect oIntersectBox = oBlob1 & oBlob2;
		cv::Rect oBigBox       = oBlob1 | oBlob2;

		// Find intersect width, height ratio
		int nMinWidth = min( oBlob1.width, oBlob2.width );
		int nMaxWidth = max( oBlob1.width, oBlob2.width );
		int nMinHeight = min( oBlob1.height, oBlob2.height );
		int nMaxHeight = max( oBlob1.height, oBlob2.height );

		float rOverlapWRatio    = (float)oIntersectBox.width  / (float)nMinWidth;
		float rNonOverlapWRatio = (float)( nMinWidth - oIntersectBox.width ) / (float)nMaxWidth;
		float rOverlapHRatio    = (float)oIntersectBox.height / (float)nMinHeight;
		float rNonOverlapHRatio = (float)( nMinHeight - oIntersectBox.height ) / (float)nMaxHeight;

		float rBigDiff   = fabs( (double)( oBigBox.width - nMedianCharWidth ) );
		rBigDiff *= 1.3;	// TODO DV: hard-coded?

		float rW1Diff = fabs( (double)( oBlob1.width - nMedianCharWidth ) );
		float rW2Diff = fabs( (double)( oBlob2.width - nMedianCharWidth ) );

		if(params.bDebug) 
		{
			printLogInfo( "Current blob: x = %d, y = %d, width = %d, height = %d",
				 oBlob1.x, oBlob1.y, oBlob1.width, oBlob1.height );
			printLogInfo( "Next blob: x = %d, y = %d, width = %d, height = %d",
				 oBlob2.x, oBlob2.y, oBlob2.width, oBlob2.height );
			printLogInfo( "oBigBox.width = %d, nMedianCharWidth = %d, rWRatio = %f, rHRatio = %f, rBigDiff = %f, rW1Diff = %f, rW2Diff = %f",
				oBigBox.width, nMedianCharWidth, rOverlapWRatio, rOverlapHRatio, rBigDiff, rW1Diff, rW2Diff );
		}

		bool bCond1 = ( rBigDiff < rW1Diff && rBigDiff < rW2Diff );
		bool bCond2 = ( rOverlapWRatio > rOverlapWidthThresh && rOverlapHRatio > rOverlapHeightThresh );
		bool bCond3 = ( rNonOverlapWRatio < rNonOverlapWidthThresh && rNonOverlapHRatio < rNonOverlapHeightThresh );
		if( bCond1 || bCond2 || bCond3 )
		{
			if( bCond1 )
			{
				if(params.bDebug) printLogInfo("MERGING BLOBS due to size");
			}
			if( bCond2 || bCond3 )
			{
				if(params.bDebug) printLogInfo("MERGING BLOBS due to overlap");
			}

			SimpleBlob oBigBlob( oBigBox );
			oBigBlob.oContour.reserve( it->oContour.size() + itNext->oContour.size() ); // preallocate memory
			oBigBlob.oContour.insert( oBigBlob.oContour.end(), it->oContour.begin(), it->oContour.end() );
			oBigBlob.oContour.insert( oBigBlob.oContour.end(), itNext->oContour.begin(), itNext->oContour.end() );
			oBigBlob.sStatus = s_sSTATUS_CANDIDATE;

			// Update the first blob ,remove the second blob,
			(*it) = oBigBlob;
			it = oBlobs.erase( itNext ) - 1;
		}
		else
		{
			++it;
		}

		iCount++;
	}
}

void FTS_IP_SimpleBlobDetector::mergeBlobsVector( vector<Rect>& oBlobs,
												  const int nMedianCharWidth  )
{
	float rOverlapWidthThresh  = 0.5;	// TODO DV: setting?
	float rOverlapHeightThresh = 0.5;	// TODO DV: setting?

	float rNonOverlapWidthThresh  = 0.5;	// TODO DV: setting?
	float rNonOverlapHeightThresh = 0.5;	// TODO DV: setting?

	vector<Rect>::iterator it = oBlobs.begin();
	int iCount = 0;
	for ( ; it != oBlobs.end(); )
	{
		// If this is the last element
		vector<Rect>::iterator itNext = it+1;
		if ( itNext == oBlobs.end() )
		{
			break;
		}

		// Get the 2 bounding rectangles
		cv::Rect oBlob1 = (*it);
		cv::Rect oBlob2 = (*itNext);

		// Find intersect and merged blobs
		cv::Rect oIntersectBox = oBlob1 & oBlob2;
		cv::Rect oBigBox       = oBlob1 | oBlob2;

		// Find intersect width, height ratio
		int nMinWidth = min( oBlob1.width, oBlob2.width );
		int nMaxWidth = max( oBlob1.width, oBlob2.width );
		int nMinHeight = min( oBlob1.height, oBlob2.height );
		int nMaxHeight = max( oBlob1.height, oBlob2.height );

		float rOverlapWRatio    = (float)oIntersectBox.width  / (float)nMinWidth;
		float rNonOverlapWRatio = (float)( nMinWidth - oIntersectBox.width ) / (float)nMaxWidth;
		float rOverlapHRatio    = (float)oIntersectBox.height / (float)nMinHeight;
		float rNonOverlapHRatio = (float)( nMinHeight - oIntersectBox.height ) / (float)nMaxHeight;

		float rBigDiff   = fabs( (double)( oBigBox.width - nMedianCharWidth ) );
		rBigDiff *= 1.3;	// TODO DV: hard-coded?

		float rW1Diff = fabs( (double)( oBlob1.width - nMedianCharWidth ) );
		float rW2Diff = fabs( (double)( oBlob2.width - nMedianCharWidth ) );

		if(params.bDebug)
		{
			printLogInfo( "Current blob: x = %d, y = %d, width = %d, height = %d\n",
				 oBlob1.x, oBlob1.y, oBlob1.width, oBlob1.height );
			printLogInfo( "Next blob: x = %d, y = %d, width = %d, height = %d\n",
				 oBlob2.x, oBlob2.y, oBlob2.width, oBlob2.height );
			printLogInfo(	"oBigBox.width = %d, nMedianCharWidth = %d, rWRatio = %f, rHRatio = %f, rBigDiff = %f, rW1Diff = %f, rW2Diff = %f\n",
				oBigBox.width, nMedianCharWidth, rOverlapWRatio, rOverlapHRatio, rBigDiff, rW1Diff, rW2Diff );
		}

		bool bCond1 = ( rBigDiff < rW1Diff && rBigDiff < rW2Diff );
		bool bCond2 = ( rOverlapWRatio > rOverlapWidthThresh && rOverlapHRatio > rOverlapHeightThresh );
		bool bCond3 = ( rNonOverlapWRatio < rNonOverlapWidthThresh && rNonOverlapHRatio < rNonOverlapHeightThresh );
		if( bCond1 || bCond2 || bCond3 )
		{
			if( bCond1 )
			{
				if(params.bDebug) printLogInfo("MERGING BLOBS due to size\n");
			}
			if( bCond2 || bCond3 )
			{
				if(params.bDebug) printLogInfo("MERGING BLOBS due to overlap\n");
			}

			// Update the first blob ,remove the second blob,
			(*it) = oBigBox;
			it = oBlobs.erase( itNext ) - 1;
		}
		else
		{
			++it;
		}

		iCount++;
	}
}

vector<Rect> FTS_IP_SimpleBlobDetector::getBlobsByHist( const FTS_IP_VerticalHistogram& histogram,
														const FTS_BASE_LineSegment& top,
														const FTS_BASE_LineSegment& bottom,
														const float rMedianCharWidth,
														const float rMedianCharHeight,
														float& score)
{
	float rMinHistHeight = rMedianCharHeight * 0.35;	// TODO: DV: from setting?
	float rMaxBlobWidth  = rMedianCharWidth  * 1.55;	// TODO: DV: from setting?

	// DV: 23/07/2014 - this is a very important number because it decide how to segment
	// the vertical projection of the binary image. If we set too low, it might miss characters
	// if we set too high, it might split a normal character into halfs.
	int pxLeniency = 3;	// default is 2

	vector<Rect> charBoxes;
	vector<Rect> allBoxes = get1DHits( histogram.histoImg, pxLeniency, top, bottom );

	for( size_t i = 0; i < allBoxes.size(); i++ )
	{
		cv::Rect& oBox = allBoxes[i];

		// DV: filter our thos boxes that are too short
		// This is a bug from openalpr because box height is fixed to the top & bottom lines
		int nMaxX = histogram.getLocalMaximum( oBox.x, oBox.x + oBox.width );
		int nMaxBoxHeight = histogram.getHeightAt(nMaxX);
		if( (float)nMaxBoxHeight < rMinHistHeight )
		{
			printLogWarn( "REMOVE BOX: reason is too short, height = %d - min height = %f", nMaxBoxHeight, rMinHistHeight );
			continue;
		}

		// Continue to filter by width and height
		if (	oBox.width >= 3		// TODO DV: setting
			 && oBox.width <= rMaxBlobWidth )
		{
			charBoxes.push_back(oBox);
		}
		else if (oBox.width > rMaxBlobWidth && oBox.width < rMaxBlobWidth * 2 )
		{
			// Split blobs if too big
			int leftEdge = oBox.x + (int) (((float) oBox.width) * 0.4f);
			int rightEdge = oBox.x + (int) (((float) oBox.width) * 0.6f);

			int minX = histogram.getLocalMinimum(leftEdge, rightEdge);
			int maxXChar1 = histogram.getLocalMaximum(oBox.x, minX);
			int maxXChar2 = histogram.getLocalMaximum(minX, oBox.x + oBox.width);
			int minHeight = histogram.getHeightAt(minX);

			int maxHeightChar1 = histogram.getHeightAt(maxXChar1);
			int maxHeightChar2 = histogram.getHeightAt(maxXChar2);

			if(params.bDebug) printLogInfo( "Detect a big blog, width = %d, minX = %d, maxHeightChar1 = %d, maxHeightChar2 =%d",
					oBox.width, minX, maxHeightChar1, maxHeightChar2 );
			if (maxHeightChar1 > rMinHistHeight && minHeight < (0.5 * ((float) maxHeightChar1)))
			{
				// Add a box for Char1
				if(params.bDebug) printLogInfo( "Add box fo char 1" );
				Point botRight = Point(minX - 1, oBox.y + oBox.height);
				charBoxes.push_back(Rect(oBox.tl(), botRight) );
			}
			if (maxHeightChar2 > rMinHistHeight && minHeight < (0.5 * ((float) maxHeightChar2)))
			{
				// Add a box for Char2
				if(params.bDebug) printLogInfo( "Add box fo char 2" );
				Point topLeft = Point(minX + 1, oBox.y);
				charBoxes.push_back(Rect(topLeft, oBox.br()) );
			}
		}
	}

	return charBoxes;
}


//vector<FTS_IP_SimpleBlobDetector::SimpleBlob> FTS_IP_SimpleBlobDetector::getBestBlobs(
//										 const Mat& oSrc,
//										 const vector<SimpleBlob>& oBlobs,
//										 const float rMedianCharWidth,
//										 const FTS_BASE_LineSegment& oTopLine,
//									     const FTS_BASE_LineSegment& oBottomLine )
//{
//	vector<SimpleBlob> oBestBlobs;	// output
//
//
//	float rMaxCharWidth = rMedianCharWidth * 1.55;	// TODO DV: setting?
//
//	// This histogram is based on how many char boxes (from ALL of the many thresholded images)
//	// are covering each column.
//	// Makes a sort of histogram from all the previous char boxes.  Figures out the best fit from that.
//	Mat oHistImg = Mat::zeros(Size(oSrc.cols, oSrc.rows), CV_8U);
//	int nColumnCount;
//	for (int col = 0; col < oSrc.cols; col++)
//	{
//		nColumnCount = 0;
//
//		for( size_t i = 0; i < oBlobs.size(); i++)
//		{
//			if( col >= oBlobs[i].oBB.x && col < oBlobs[i].oBB.x + oBlobs[i].oBB.width )
//			{
//				nColumnCount++;
//			}
//		}
//
//		// Fill the line of the histogram
//		for (; nColumnCount > 0; nColumnCount--)
//		{
//			oHistImg.at<uchar>(oHistImg.rows -  nColumnCount, col) = 255;
//		}
//	}
//
//	// Get the histogram
//	FTS_IP_VerticalHistogram histogram(oHistImg, Mat::ones(oHistImg.size(), CV_8U));
//
//	// Go through each row in the oHistImg and score it.
//	// Try to find the single line that gives me the most
//	// right-sized character regions (based on rMedianCharWidth)
//	int   nBestRowIndex = 0;
//	float rBestRowScore = 0;
//	for (int row = 0; row < oHistImg.rows; row++)
//	{
//		vector<SimpleBlob> validBoxes;
//		vector<Rect> allBoxes = get1DHits( oHistImg, row, oTopLine, oBottomLine );
//
//		if (allBoxes.size() == 0)
//		{
//			break;
//		}
//
//		float rowScore = 0;
//		unsigned int MIN_SEG_WIDTH_PX = 4;	// TODO: DV: move to setting
//		for( size_t boxidx = 0; boxidx < allBoxes.size(); boxidx++)
//		{
//			unsigned int w = allBoxes[boxidx].width;
//			if( w >= MIN_SEG_WIDTH_PX && w <= rMaxCharWidth )
//			{
//				float widthDiffPixels = abs(w - rMedianCharWidth);
//				float widthDiffPercent = widthDiffPixels / rMedianCharWidth;
//				rowScore += 10 * (1 - widthDiffPercent);
//
//				if (widthDiffPercent < 0.25)	// Bonus points when it's close to the average character width
//					rowScore += 8;
//
//				// Add a good blob
//				SimpleBlob sb( allBoxes[boxidx] );
//				sb.sStatus = s_sSTATUS_CANDIDATE;
//
//				validBoxes.push_back( sb );
//			}
//			else if (w > rMedianCharWidth * 2  && w <= rMaxCharWidth * 2 )
//			{
//				// Try to split up doubles into two good char regions, check for a break between 40% and 60%
//				int leftEdge = allBoxes[boxidx].x + (int) (((float) allBoxes[boxidx].width) * 0.4f);
//				int rightEdge = allBoxes[boxidx].x + (int) (((float) allBoxes[boxidx].width) * 0.6f);
//
//				int minX = histogram.getLocalMinimum(leftEdge, rightEdge);
//				int maxXChar1 = histogram.getLocalMaximum(allBoxes[boxidx].x, minX);
//				int maxXChar2 = histogram.getLocalMaximum(minX, allBoxes[boxidx].x + allBoxes[boxidx].width);
//				int minHeight = histogram.getHeightAt(minX);
//
//				int maxHeightChar1 = histogram.getHeightAt(maxXChar1);
//				int maxHeightChar2 = histogram.getHeightAt(maxXChar2);
//
//				if (  minHeight < (0.25 * ((float) maxHeightChar1)))
//				{
//					// Add a box for Char1
//					Point botRight = Point(minX - 1, allBoxes[boxidx].y + allBoxes[boxidx].height);
//
//					// Add a good blob
//					SimpleBlob sb( Rect(allBoxes[boxidx].tl(), botRight) );
//					sb.sStatus = s_sSTATUS_CANDIDATE;
//
//					validBoxes.push_back( sb );
//				}
//				if (  minHeight < (0.25 * ((float) maxHeightChar2)))
//				{
//					// Add a box for Char2
//					Point topLeft = Point(minX + 1, allBoxes[boxidx].y);
//
//					// Add a good blob
//					SimpleBlob sb( Rect(topLeft, allBoxes[boxidx].br()) );
//					sb.sStatus = s_sSTATUS_CANDIDATE;
//
//					validBoxes.push_back( sb );
//				}
//			}
//		}
//
//		if (rowScore > rBestRowScore)
//		{
//			rBestRowScore = rowScore;
//			nBestRowIndex = row;
//			oBestBlobs = validBoxes;
//		}
//	}
//
//	printf( "Best row index = %d\n", nBestRowIndex );
//
//#ifdef DEBUG_HIST
//
//	cvtColor(oHistImg, oHistImg, CV_GRAY2BGR);
//	line( oHistImg,
//		  Point(0, oHistImg.rows - 1 - nBestRowIndex),
//		  Point(oHistImg.cols, oHistImg.rows - 1 - nBestRowIndex),
//		  Scalar(0, 255, 0) );
//
//	Mat oBestCandidatesImg( oSrc.size(), oSrc.type() );
//	oSrc.copyTo( oBestCandidatesImg );
//	cvtColor( oBestCandidatesImg, oBestCandidatesImg, CV_GRAY2BGR );
//	for( unsigned int i = 0; i < oBestBlobs.size(); i++)
//	{
//		rectangle( oBestCandidatesImg, oBestBlobs[i].toRect(), Scalar(0, 255, 0));
//	}
//
////	FTS_GUI_DisplayImage::ShowAndScaleBy2( "All Histograms", oHistImg,
////			FTS_GUI_DisplayImage::SCALE_X,
////			FTS_GUI_DisplayImage::SCALE_Y, 0, 600 );
////	FTS_GUI_DisplayImage::ShowAndScaleBy2( "Best Boxes", oBestCandidatesImg,
////			FTS_GUI_DisplayImage::SCALE_X,
////			FTS_GUI_DisplayImage::SCALE_Y, 120, 600 );
//
//	std::vector< cv::Mat > oGrayImages;
//	oGrayImages.push_back( oHistImg );
//	oGrayImages.push_back( oBestCandidatesImg );
//	FTS_GUI_DisplayImage::ShowGroupScaleBy2( "Gray", oGrayImages, 1 );
//
//#endif
//
//	return oBestBlobs;
//}


vector<Rect> FTS_IP_SimpleBlobDetector::getBestBoxes(
										 const Mat& oSrc,
										 const vector<Rect>& oBlobs,
										 const float rMedianCharWidth,
										 const FTS_BASE_LineSegment& oTopLine,
									     const FTS_BASE_LineSegment& oBottomLine )
{
	vector<Rect> oBestBlobs;	// output
	if(params.bDebug) printLogInfo( "Median width inside getBestBoxes = %f, oBlobs.size() = %d", rMedianCharWidth, oBlobs.size() );
	float rMaxCharWidth = rMedianCharWidth * 1.55;	// TODO DV: setting?

	Mat oHistImg = Mat::zeros(Size(oSrc.cols, oSrc.rows), CV_8U);
	int nColumnCount;
	for (int col = 0; col < oSrc.cols; col++)
	{
		nColumnCount = 0;

		for( size_t i = 0; i < oBlobs.size(); i++)
		{
			if( col >= oBlobs[i].x && col < oBlobs[i].x + oBlobs[i].width )
			{
				nColumnCount++;
			}
		}

		// Fill the line of the histogram
		for (; nColumnCount > 0; nColumnCount--)
		{
			oHistImg.at<uchar>(oHistImg.rows -  nColumnCount, col) = 255;
		}
	}

	// Get the histogram
	FTS_IP_VerticalHistogram histogram(oHistImg, Mat::ones(oHistImg.size(), CV_8U));

	// Go through each row in the oHistImg and score it.
	// Try to find the single line that gives the most
	// right-sized character regions (based on rMedianCharWidth)
	int   nBestRowIndex = 0;
	float rBestRowScore = 0;
	for (int row = 0; row < oHistImg.rows; row++)
	{
		vector<Rect> validBoxes;
		vector<Rect> allBoxes = get1DHits( oHistImg, row, oTopLine, oBottomLine );

		if (allBoxes.size() == 0)
		{
			break;
		}

		float rowScore = 0;
		unsigned int MIN_SEG_WIDTH_PX = 3;	// TODO: DV: move to setting
		for( size_t boxidx = 0; boxidx < allBoxes.size(); boxidx++)
		{
			unsigned int w = allBoxes[boxidx].width;
			if( w >= MIN_SEG_WIDTH_PX && w <= rMaxCharWidth )
			{
				float widthDiffPixels = abs(w - rMedianCharWidth);
				float widthDiffPercent = widthDiffPixels / rMedianCharWidth;
				rowScore += 10 * (1 - widthDiffPercent);

				// DV: 16/06/2014: Lower the bonus
				if (widthDiffPercent < 0.25)	// Bonus points when it's close to the average character width
					rowScore += 8 * ( 1.0 - widthDiffPercent );

				// Add a good box
				validBoxes.push_back( allBoxes[boxidx] );
//				printf( "validBoxes.size() = %d, rowScore = %f - %d\n",
//						validBoxes.size(),
//						rowScore,
//						(widthDiffPercent < 0.25)?1:0 );
			}
			else if (w > rMaxCharWidth  && w <= rMaxCharWidth * 2 )
			{
				// Try to split up doubles into two good char regions, check for a break between 40% and 60%
				int leftEdge  = allBoxes[boxidx].x + (int) (((float) allBoxes[boxidx].width) * 0.4f);
				int rightEdge = allBoxes[boxidx].x + (int) (((float) allBoxes[boxidx].width) * 0.6f);

				int minX = histogram.getLocalMinimum(leftEdge, rightEdge);
				int maxXChar1 = histogram.getLocalMaximum(allBoxes[boxidx].x, minX);
				int maxXChar2 = histogram.getLocalMaximum(minX, allBoxes[boxidx].x + allBoxes[boxidx].width);
				int minHeight = histogram.getHeightAt(minX);

				int maxHeightChar1 = histogram.getHeightAt(maxXChar1);
				int maxHeightChar2 = histogram.getHeightAt(maxXChar2);

				if(params.bDebug) printLogInfo( "Detect a big blog, x = %d, w = %d, minHeight = %d, maxHeightChar1 = %d, maxHeightChar2 =%d",
						allBoxes[boxidx].x,	allBoxes[boxidx].width, minHeight, maxHeightChar1, maxHeightChar2 );
				if (  minHeight < (0.25 * ((float) maxHeightChar1)))
				{
					// Add a box for Char1
					if(params.bDebug) printLogInfo( "Add box fo char 1" );
					Point botRight = Point(minX - 1, allBoxes[boxidx].y + allBoxes[boxidx].height);

					// Add a good box
					validBoxes.push_back( Rect(allBoxes[boxidx].tl(), botRight) );
				}
				else if (  minHeight < (0.25 * ((float) maxHeightChar2)))
				{
					// Add a box for Char2
					if(params.bDebug) printLogInfo( "Add box fo char 2" );
					Point topLeft = Point(minX + 1, allBoxes[boxidx].y);

					// Add a good box
					validBoxes.push_back( Rect(topLeft, allBoxes[boxidx].br()) );
				}
				else
				{
					if(params.bDebug) printLogInfo( "Can't split, so simply accept the big blob" );
					validBoxes.push_back( allBoxes[boxidx] );
				}
			}
		}

		if (rowScore > rBestRowScore)
		{
			rBestRowScore = rowScore;
			nBestRowIndex = row;
			oBestBlobs = validBoxes;
		}
	}

	if(params.bDebug) printLogInfo( "Best row index = %d, num of best blobs = %d", nBestRowIndex, oBestBlobs.size() );

#ifdef DEBUG_HIST
	cvtColor(oHistImg, oHistImg, CV_GRAY2BGR);
//	line( oHistImg,
//		  Point(0, oHistImg.rows - 1 - nBestRowIndex),
//		  Point(oHistImg.cols, oHistImg.rows - 1 - nBestRowIndex),
//		  Scalar(0, 255, 0) );

	Mat oBestCandidatesImg( oSrc.size(), oSrc.type() );
	oSrc.copyTo( oBestCandidatesImg );
	cvtColor( oBestCandidatesImg, oBestCandidatesImg, CV_GRAY2BGR );
	for( unsigned int i = 0; i < oBestBlobs.size(); i++)
	{
		rectangle( oBestCandidatesImg, oBestBlobs[i], Scalar(0, 255, 0));
	}

//	line( oBestCandidatesImg, Point( m_nMinX, 0 ), Point( m_nMinX, oBestCandidatesImg.rows-1 ), Scalar(0,0,255) );
//	line( oBestCandidatesImg, Point( m_nMaxX, 0 ), Point( m_nMaxX, oBestCandidatesImg.rows-1 ), Scalar(0,0,255) );

	if(params.bDebug && params.bDisplayDbgImg) 
		 FTS_GUI_DisplayImage::ShowAndScaleBy2( "1D Histograms Sum", oHistImg,
			FTS_GUI_DisplayImage::SCALE_X,
			FTS_GUI_DisplayImage::SCALE_Y, 0, 600 );
//	FTS_GUI_DisplayImage::ShowAndScaleBy2( "Good Candidates", oBestCandidatesImg,
//			FTS_GUI_DisplayImage::SCALE_X,
//			FTS_GUI_DisplayImage::SCALE_Y, 300, 700 );

	std::vector< cv::Mat > oGrayImages;
	oGrayImages.push_back( oHistImg );
	oGrayImages.push_back( oBestCandidatesImg );
	if(params.bDebug && params.bDisplayDbgImg) FTS_GUI_DisplayImage::ShowGroupScaleBy2( "Vertical Projection", 1.0, oGrayImages, 2 );
#endif

	return oBestBlobs;
}


// DV: this original function accept very small( short ) blobs to be a candidate
// After this function, remember to check height to filter out those
vector<Rect> FTS_IP_SimpleBlobDetector::get1DHits( const Mat& img,
												   const int yOffset,
												   const FTS_BASE_LineSegment& top,
												   const FTS_BASE_LineSegment& bottom )
{
	vector<Rect> hits;

	bool onSegment = false;
	int curSegmentLength = 0;
	for (int col = 0; col < img.cols; col++)
	{
		bool isOn = img.at<uchar>(img.rows - 1 - yOffset, col);
		if (isOn)
		{
			// We're on a segment.  Increment the length
			onSegment = true;
			curSegmentLength++;
		}

		if (onSegment && (isOn == false || (col == img.cols - 1)))
		{
			// A segment just ended or we're at the very end of the row and we're on a segment
			int nY = (int)top.getPointAt( (float)(col - curSegmentLength) );
			Point topLeft  = Point( col - curSegmentLength, nY - 1);
			Point botRight = Point( col, (int)bottom.getPointAt( (float)(col + 1) ) );

			hits.push_back(Rect(topLeft, botRight));

			onSegment = false;
			curSegmentLength = 0;
		}
	}

	return hits;
}

void FTS_IP_SimpleBlobDetector::removeZombieBlobs( vector<SimpleBlob>& oBlobs,
											  	   const FTS_IP_VerticalHistogram& oVertHist,
											  	   const float& rMedianCharHeight )
{
	float rMinHistHeight = rMedianCharHeight * 0.35;	// TODO: DV: from setting?
	std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::iterator blobIter = oBlobs.begin();
	if(params.bDebug) printLogInfo( "Before removing zombies, no of blobs = %d", oBlobs.size() );
	for ( ; blobIter != oBlobs.end();  )
	{
		cv::Rect oBox = blobIter->toRect();
		int maxX = oVertHist.getLocalMaximum( oBox.x, oBox.x + oBox.width );
		int maxHeight = oVertHist.getHeightAt( maxX);

		if(params.bDebug) printLogInfo( "Blob local max histogram = %d - min hist height = %f", maxHeight, rMinHistHeight );
		if( (float)maxHeight < rMinHistHeight )
		{
			blobIter = oBlobs.erase( blobIter );
		}
		else
		{
			++blobIter;
		}
	}
	if(params.bDebug) printLogInfo( "Done removing zombies" );
}

// blobs that are overlapped horizontally will be adjusted
// must be ordered in x before passing into this function
void FTS_IP_SimpleBlobDetector::adjustBlobXY( vector<SimpleBlob>& oBlobs,
											  const FTS_IP_VerticalHistogram& oVertHist )
{
	for( unsigned int i = 0; i < oBlobs.size() - 1; i++ )
	{
		cv::Rect oBox1 = oBlobs[i].toRect();
		cv::Rect oBox2 = oBlobs[i+1].toRect();
		cv::Rect oBigBox = oBox1 | oBox2;

		if( ( oBox1.x + oBox1.width ) > oBox2.x )	// TODO DV: strict enough? Setting?
		{
			if(params.bDebug) printLogInfo( "Adjust x, y - Blob %d: x = %d, y = %d, height = %d  ;  Blob %d: x = %d, y = %d, height = %d",
					i  , oBlobs[i].oBB.x, oBlobs[i].oBB.y, oBox1.height,
					i+1, oBlobs[i+1].oBB.x, oBlobs[i+1].oBB.y, oBox2.height );

			int nBeginCol = oBigBox.x + (int) ( oBigBox.width * 0.0f );
			int nEndCol   = oBigBox.x + (int) ( oBigBox.width * 1.0f );

			int minX = oVertHist.getLocalMinimum( nBeginCol, nEndCol );

			// Adjust
			int nNewRightX1 = minX - 1;
			int nNewLeftX2  = minX + 1;

			// Blob 1
			oBlobs[i].oBB.width = nNewRightX1 - oBox1.x;

			// Blob 2
			oBlobs[i+1].oBB.x = nNewLeftX2;
		}
	}
}

void FTS_IP_SimpleBlobDetector::adjustBlobHeight( vector<SimpleBlob>& oBlobs,
												  const float rMedianCharHeight,
												  const FTS_BASE_LineSegment& oTopLine,
												  const FTS_BASE_LineSegment& oBottomLine )
{
	for( unsigned int i = 0; i < oBlobs.size(); i++ )
	{
		cv::Rect oRect = oBlobs[i].toRect();

//		float rHeightDiffRatio = ceil(oBlobs[i].vertradius * 2 ) / rMedianCharHeight;

		float rTopY    = oTopLine   .getPointAt( (float)oRect.x );
		float rBottomY = oBottomLine.getPointAt( (float)oRect.x );
//		if( rHeightDiffRatio < 0.9 )	// TODO DV: strict enough? Setting?

//		printf( "Blob %d - TopY = %d, BottomY = %d, Y = %f", i, oRect.y, oRect.y + oRect.height, oBlobs[i].location.y );

		int nNewTopY = oRect.y;
		int nNewBotY = oRect.y + oRect.height;
		if( (float)nNewTopY > rTopY )
		{
			nNewTopY = floor( rTopY );
		}
		if( (float)nNewBotY < rBottomY )
		{
			nNewBotY = ceil( rBottomY );
		}

		oBlobs[i].oBB.y = nNewTopY;
		oBlobs[i].oBB.height = nNewBotY - nNewTopY;
		// TODO: what if the character is higher than the median width
	}
}

vector<Point> FTS_IP_SimpleBlobDetector::getBoundingPolygonFromBlobs( const int cols,
													    const int rows,
													    const vector<SimpleBlob>& oBlobs,
													    const vector<bool>& goodIndices )
{
	vector<Point> bestStripe;

	vector<Rect> charRegions;

	for( unsigned int i = 0; i < oBlobs.size(); i++)
	{
		if (goodIndices[i])
		{
			charRegions.push_back( oBlobs[i].oBB );
		}
	}

	// Find the best fit line segment that is parallel with the most char segments
	if (charRegions.size() == 0)
	{
		bestStripe.push_back( Point( 0, 0 ) );
		bestStripe.push_back( Point( cols - 1, 0 ) );
		bestStripe.push_back( Point( cols - 1, rows - 1 ) );
		bestStripe.push_back( Point( 0, rows - 1 ) );

		// DV 06/06/2014
		// TODO: Enhance later because sometimes the characters are totally connected
		//       to the top pr bottom, binary images can't segment those characters
		//       Suggestion: use long line removal
	}
	// DV: 06/06/2014: if there is only 1 blob, still return its top and bottom lines correctly
	else if (charRegions.size() == 1)
	{
		bestStripe.push_back( Point( 0, charRegions[0].y ) );
		bestStripe.push_back( Point( cols - 1, charRegions[0].y ) );
		bestStripe.push_back( Point( cols - 1, charRegions[0].y + charRegions[0].height ) );
		bestStripe.push_back( Point( 0, charRegions[0].y + charRegions[0].height ) );
	}
	else
	{
		vector<FTS_BASE_LineSegment> topLines;
		vector<FTS_BASE_LineSegment> bottomLines;
		// Iterate through each possible char and find all possible lines for the top and bottom of each char segment
		for( unsigned int i = 0; i < charRegions.size() - 1; i++ )
		{
			for( unsigned int k = i+1; k < charRegions.size(); k++ )
			{
				Rect* leftRect;
				Rect* rightRect;
				if (charRegions[i].x < charRegions[k].x)
				{
					leftRect  = &charRegions[i];
					rightRect = &charRegions[k];
				}
				else
				{
					leftRect  = &charRegions[k];
					rightRect = &charRegions[i];
				}

				int x1, y1, x2, y2;

				if (leftRect->y > rightRect->y)	// Rising line, use the top left corner of the rect
				{
					x1 = leftRect->x;
					x2 = rightRect->x;
				}
				else					// falling line, use the top right corner of the rect
				{
					x1 = leftRect->x + leftRect->width;
					x2 = rightRect->x + rightRect->width;
				}
				y1 = leftRect->y;
				y2 = rightRect->y;

				//cv::line(tempImg, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255));
				topLines.push_back( FTS_BASE_LineSegment(x1, y1, x2, y2) );

				if (leftRect->y > rightRect->y)	// Rising line, use the bottom right corner of the rect
				{
				  x1 = leftRect->x + leftRect->width;
				  x2 = rightRect->x + rightRect->width;
				}
				else					// falling line, use the bottom left corner of the rect
				{
				  x1 = leftRect->x;
				  x2 = rightRect->x;
				}
				y1 = leftRect->y + leftRect->height;
				y2 = rightRect->y + leftRect->height;

				//cv::line(tempImg, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255));
				bottomLines.push_back(FTS_BASE_LineSegment(x1, y1, x2, y2));

				//drawAndWait(&tempImg);
			}
		}

		int bestScoreIndex = 0;
		int bestScore = -1;
		int bestScoreDistance = -1; // Line segment distance is used as a tie breaker

		// Now, among all possible lines, find the one that is the best fit
		for( unsigned int i = 0; i < topLines.size(); i++ )
		{
			float SCORING_MIN_THRESHOLD = 0.97;
			float SCORING_MAX_THRESHOLD = 1.03;

			int curScore = 0;
			for( unsigned int charidx = 0; charidx < charRegions.size(); charidx++ )
			{
				float topYPos = topLines[i].getPointAt(charRegions[charidx].x);
				float botYPos = bottomLines[i].getPointAt(charRegions[charidx].x);

				float minTop = charRegions[charidx].y * SCORING_MIN_THRESHOLD;
				float maxTop = charRegions[charidx].y * SCORING_MAX_THRESHOLD;
				float minBot = (charRegions[charidx].y + charRegions[charidx].height) * SCORING_MIN_THRESHOLD;
				float maxBot = (charRegions[charidx].y + charRegions[charidx].height) * SCORING_MAX_THRESHOLD;
				if ( (topYPos >= minTop && topYPos <= maxTop) &&
					 (botYPos >= minBot && botYPos <= maxBot))
				{
					curScore++;
				}
			}

			// Tie goes to the one with longer line segments
			if( (curScore > bestScore) ||
			    (curScore == bestScore && topLines[i].length > bestScoreDistance) )
			{
				bestScore = curScore;
				bestScoreIndex = i;
				// Just use x distance for now
				bestScoreDistance = topLines[i].length;
			}
		}

		Point topLeft 		= Point(0, topLines[bestScoreIndex].getPointAt(0) );
		Point topRight 		= Point(cols, topLines[bestScoreIndex].getPointAt(cols));
		Point bottomRight 	= Point(cols, bottomLines[bestScoreIndex].getPointAt(cols));
		Point bottomLeft 	= Point(0, bottomLines[bestScoreIndex].getPointAt(0));

		// DV 08/05/2014: be conservative, add some epsilon
//		if( topLeft.y  > 0 ) topLeft.y--;
//		if( topRight.y > 0 ) topRight.y--;
//		if( bottomRight.y < rows - 1 ) bottomRight.y++;
//		if( bottomLeft.y < rows - 1 ) bottomLeft.y++;

		bestStripe.push_back(topLeft);
		bestStripe.push_back(topRight);
		bestStripe.push_back(bottomRight);
		bestStripe.push_back(bottomLeft);
	}

	return bestStripe;
}

int FTS_IP_SimpleBlobDetector::clusterBlobs( vector<SimpleBlob>& oBlobs,
											 CvCmpFunc is_equal )
{
    //cvClearSeq( m_poSegCharSeq );
	
	m_poSegCharSeq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(SimpleBlob*), m_poStorage );

    // Fill an opencv sequence with chars
    // --------------------------------------------------------------------
    vector<SimpleBlob>::iterator i  = oBlobs.begin();
    vector<SimpleBlob>::iterator iE = oBlobs.end();
    for( ; i != iE; ++i )
    {
    	SimpleBlob* poBlob = &(*i);

        cvSeqPush( m_poSegCharSeq, &poBlob );
    }

    // Partition into equivalent classes such that all chars in the same class
    // are the same size and non-overlapping.
    // --------------------------------------------------------------------
    CvSeq* poLabels = 0;
    int nClassCount = cvSeqPartition( m_poSegCharSeq,
                                      0,
                                      &poLabels,
                                      is_equal,// FTS_IP_SimpleBlobDetector::isSameSizeAndNoOverlap,
                                      this );

    int nCleanCharCount = 0;

	if(params.bDebug)	printLogInfo( "nClassCount = %d", nClassCount );
    if( nClassCount < (int) oBlobs.size() ) // means there's at least one class with at least 2 chars in it
    {
        // Find the biggest partition. This partition contains all the chars that are
        // the same size and DO NOT overlap with each other.
        // --------------------------------------------------------------------

        std::vector<int>& oCount = m_oIntVector; // Num chars in each class

        oCount.clear();
        oCount.assign( nClassCount, 0 );

        for( int n = 0; n < poLabels->total; ++n )
        {
            int nClassIdx = *(int*) cvGetSeqElem( poLabels, n );
            ++oCount.at( nClassIdx );
        }

        int nBiggestClassIdx = std::distance(                   oCount.begin(),
                                              std::max_element( oCount.begin(), oCount.end() )
                                              );

        // Mark all chars in the biggest class as clean chars.
        // --------------------------------------------------------------------
        i  = oBlobs.begin();
        iE = oBlobs.end();
        for( unsigned int nIdx = 0; i != iE; ++i, ++nIdx )
        {
            int nClassIdx = *(int*) cvGetSeqElem(poLabels, nIdx);
            if (nClassIdx == nBiggestClassIdx)
            {
            	if(params.bDebug)	printLogInfo( "Clean char x = %d, y = %d, w = %d",
            							i->oBB.x, i->oBB.y, i->oBB.width );
                i->sStatus = s_sSTATUS_CANDIDATE;
                ++nCleanCharCount;
            }
            else
            {
//            	printf( "Remove this blob, idx = %d", nIdx );
            	i->sStatus = s_sSTATUS_REMOVED;
            }
        }

    } // if at least one class with more than one char
    else	// DV: 16/06/2014 - exception case
    {
    	nCleanCharCount = 1;
    }


    // Release the memory storage of label sequence
    // --------------------------------------------------------------------
    cvClearSeq( poLabels );
    cvClearSeq( m_poSegCharSeq );

    cvClearMemStorage( m_poStorage ); // clear, don't deallocate

    return nCleanCharCount;
}

int FTS_IP_SimpleBlobDetector::isSameSize( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const SimpleBlob& o1 = **(const SimpleBlob**) poSegChar1;
    const SimpleBlob& o2 = **(const SimpleBlob**) poSegChar2;

    const FTS_IP_SimpleBlobDetector& oSeg = *(const FTS_IP_SimpleBlobDetector*) poSeg;

//    int nX1 = o1.oBB.x;
//    int nX2 = o2.oBB.x;

    int nW1 = o1.oBB.width;
    int nW2 = o2.oBB.width;

    int w = abs( nW1 - nW2 );
    int h = abs(   (int) o1.oBB.height
                 - (int) o2.oBB.height  );

    // If chars not the same size, return false
    if(    w > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxWidthDiff
        || h > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxHeightDiff )
    {
        nRet = 0;
    }

//    // Gets here if chars are same size
//    // If chars overlap, then return false
//    if( nX1 < nX2 )
//    {
//        if( nX1 + nW1 > nX2 )
//        {
//            nRet = 0;
//        }
//    }
//    else
//    {
//        if( nX2 + nW2 > nX1 )
//        {
//            nRet = 0;
//        }
//    }

    // Chars are same size AND do not overlap.
    return nRet;
}

int FTS_IP_SimpleBlobDetector::isSameHeight( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const SimpleBlob& o1 = **(const SimpleBlob**) poSegChar1;
    const SimpleBlob& o2 = **(const SimpleBlob**) poSegChar2;

    const FTS_IP_SimpleBlobDetector& oSeg = *(const FTS_IP_SimpleBlobDetector*) poSeg;

    int h = abs(   (int) o1.oBB.height
                 - (int) o2.oBB.height  );

    // If chars not the same size, return false
    if(   h > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxHeightDiff )
    {
        nRet = 0;
    }

    // Chars are same size AND do not overlap.
    return nRet;
}

int FTS_IP_SimpleBlobDetector::isSameHeightAndClosed( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const SimpleBlob& o1 = **(const SimpleBlob**) poSegChar1;
    const SimpleBlob& o2 = **(const SimpleBlob**) poSegChar2;

    const FTS_IP_SimpleBlobDetector& oSeg = *(const FTS_IP_SimpleBlobDetector*) poSeg;

    int h 	   = abs( o1.oBB.height - o2.oBB.height  );

    // DV: 16/06/2014 - use distance rather than x diff
    int nXDiff = calcXDistBetween2Blobs( o1, o2 );

    // If chars not the same size, return false
    if(    h > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxHeightDiff
    	|| nXDiff > 3 * min( o1.oBB.width, o2.oBB.width ) )
    {
        nRet = 0;
    }

    // Chars are same size AND do not overlap.
    return nRet;
}

int FTS_IP_SimpleBlobDetector::calcXDistBetween2Blobs( const SimpleBlob& o1,
													   const SimpleBlob& o2 )
{
	int nXDiff = abs( o1.oBB.x - o2.oBB.x  );

	int nXDist = ( o2.oBB.x > o1.oBB.x ) ?
				 ( o2.oBB.x - ( o1.oBB.x + o1.oBB.width ) ) :
				 ( o1.oBB.x - ( o2.oBB.x + o2.oBB.width ) );
	if( nXDist > 0 )
	{
		nXDiff = nXDist;
	}

	return nXDiff;
}

int FTS_IP_SimpleBlobDetector::isClosed( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const SimpleBlob& o1 = **(const SimpleBlob**) poSegChar1;
    const SimpleBlob& o2 = **(const SimpleBlob**) poSegChar2;

    int nXDiff = abs( o1.oBB.x - o2.oBB.x  );

    // If chars not the same size, return false
    if( nXDiff > 3 * min( o1.oBB.width, o2.oBB.width ) )
    {
        nRet = 0;
    }

    // Chars are same size AND do not overlap.
    return nRet;
}

int FTS_IP_SimpleBlobDetector::isSameY( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const SimpleBlob& o1 = **(const SimpleBlob**) poSegChar1;
    const SimpleBlob& o2 = **(const SimpleBlob**) poSegChar2;

    int nYDiff = abs( o1.oBB.y - o2.oBB.y );

    // If chars not the same size, return false
    if( nYDiff > floor( (float)o1.oBB.height / 4  ) )
    {
        nRet = 0;
    }

    // Chars are same Y.
    return nRet;
}

int FTS_IP_SimpleBlobDetector::isSameYAndClosed( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const SimpleBlob& o1 = **(const SimpleBlob**) poSegChar1;
    const SimpleBlob& o2 = **(const SimpleBlob**) poSegChar2;

    int nYDiff = abs( o1.oBB.y - o2.oBB.y );

    // DV: 16/06/2014 - use distance rather than x diff
	int nXDiff = calcXDistBetween2Blobs( o1, o2 );

    int nMaxHeight = max( o1.oBB.height, o2.oBB.height );

    // If chars not the same size, return false
    if(    nYDiff > floor( (float)nMaxHeight / 4  )
		|| nXDiff > 3 * min( o1.oBB.width, o2.oBB.width )
    	|| max( o1.oBB.width, o2.oBB.width ) > 3 * min( o1.oBB.width, o2.oBB.width ) )
    {
        nRet = 0;
    }

    // Chars are same Y.
    return nRet;
}

vector<Rect> FTS_IP_SimpleBlobDetector::removeEmptyBoxes( const vector<Mat>&  oBinaryImages,
														  const vector<Rect>& oBoxes)
{
	// Of the n thresholded images, if box 3 (for example) is empty in half (for example)
	// of the thresholded images, clear all data for every box #3.
	const float MIN_CONTOUR_HEIGHT_PERCENT = 0.65;

	Mat mask = getCharBoxMask(oBinaryImages[0], oBoxes );

	vector<int> boxScores(oBoxes.size());

	for (size_t i = 0; i < oBoxes.size(); i++)
	boxScores[i] = 0;

	for (size_t i = 0; i < oBinaryImages.size(); i++)
	{
		for (size_t j = 0; j < oBoxes.size(); j++)
		{
			//float minArea = oBoxes[j].area() * MIN_AREA_PERCENT;

			Mat tempImg = Mat::zeros(oBinaryImages[i].size(), oBinaryImages[i].type());
			rectangle(tempImg, oBoxes[j], Scalar(255,255,255), CV_FILLED);
			bitwise_and(oBinaryImages[i], tempImg, tempImg);

			vector<vector<Point> > contours;
			findContours(tempImg, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

//			float biggestContourHeight = 0;

			vector<Point> allPointsInBox;
			for (size_t c = 0; c < contours.size(); c++)
			{
				if (contours[c].size() == 0)
				{
					continue;
				}

				for (size_t z = 0; z < contours[c].size(); z++)
				{
					allPointsInBox.push_back(contours[c][z]);
				}
			}

			float height = 0;
			if (allPointsInBox.size() > 0)
			{
				height = boundingRect(allPointsInBox).height;
			}

			if (height >= ((float) oBoxes[j].height * MIN_CONTOUR_HEIGHT_PERCENT))
			{
				boxScores[j] = boxScores[j] + 1;
			}
			//		  else if (1)
			//		  {
			//			drawX(imgDbgCleanStages[i], oBoxes[j], COLOR_DEBUG_EMPTYFILTER, 3);
			//		  }
		}
	}

	vector<Rect> newBoxes;

	int maxBoxScore = 0;
	for( size_t i = 0; i < oBoxes.size(); i++)
	{
		if (boxScores[i] > maxBoxScore)
		{
			maxBoxScore = boxScores[i];
		}
	}

	// Need a good char sample in at least 50% of the boxes for it to be valid.
	int MIN_FULL_BOXES = maxBoxScore * 0.49;

	// Now check each score.  If it's below the minimum, remove the charRegion
	for( size_t i = 0; i < oBoxes.size(); i++)
	{
	if (boxScores[i] > MIN_FULL_BOXES)
	{
		newBoxes.push_back(oBoxes[i]);
	}
//	else
//	{
//		// Erase the box from the Mat... mainly for debug purposes
//		if (1)
//		{
//			cout << "Mostly Empty Filter: box index: " << i;
//			cout << " this box had a score of : " << boxScores[i];;
//			cout << " MIN_FULL_BOXES: " << MIN_FULL_BOXES << endl;;
//
//			for (int z = 0; z < oBinaryImages.size(); z++)
//			{
//			  rectangle(oBinaryImages[z], oBoxes[i], Scalar(0,0,0), -1);
//
//			  drawX(imgDbgCleanStages[z], oBoxes[i], COLOR_DEBUG_EMPTYFILTER, 1);
//			}
//		}
//	}
//
//	if (1)
//	  cout << " Box Score: " << boxScores[i] << endl;
	}

	return newBoxes;
}

Mat FTS_IP_SimpleBlobDetector::getCharBoxMask( const Mat& oBin,
											   const vector<Rect>& charBoxes )
{
	Mat mask = Mat::zeros(oBin.size(), CV_8U);
	for (size_t i = 0; i < charBoxes.size(); i++)
	rectangle(mask, charBoxes[i], Scalar(255, 255, 255), -1);

	return mask;
}

void FTS_IP_SimpleBlobDetector::filterEdgeBoxes( 	   vector<Mat>& thresholds,
												 const vector<Rect>& charRegions,
												 const float rMedianBoxWidth,
												 const float rMedianBoxHeight,
												 FTS_BASE_LineSegment& top,
												 FTS_BASE_LineSegment& bottom )
{
	const float MIN_ANGLE_FOR_ROTATION = 0.4;
	int MIN_CONNECTED_EDGE_PIXELS = (rMedianBoxHeight * 1.5);
	int MIN_SEG_WIDTH_PX = 4;

	// Sometimes the rectangle won't be very tall, making it impossible to detect an edge
	// Adjust for this here.
	int alternate = thresholds[0].rows * 0.92;
	if (alternate < MIN_CONNECTED_EDGE_PIXELS && alternate > rMedianBoxHeight)
	MIN_CONNECTED_EDGE_PIXELS = alternate;

	//
	// Pay special attention to the edge boxes.  If it's a skinny box, and the vertical height extends above our bounds... remove it.
	//while (charBoxes.size() > 0 && charBoxes[charBoxes.size() - 1].width < MIN_SEGMENT_WIDTH_EDGES)
	//  charBoxes.erase(charBoxes.begin() + charBoxes.size() - 1);
	// Now filter the "edge" boxes.  We don't want to include skinny boxes on the edges, since these could be plate boundaries
	//while (charBoxes.size() > 0 && charBoxes[0].width < MIN_SEGMENT_WIDTH_EDGES)
	//  charBoxes.erase(charBoxes.begin() + 0);

	// TECHNIQUE #1
	// Check for long vertical lines.  Once the line is too long, mask the whole region

	if (charRegions.size() <= 1)
	return;

	// Check both sides to see where the edges are
	// The first starts at the right edge of the leftmost char region and works its way left
	// The second starts at the left edge of the rightmost char region and works its way right.
	// We start by rotating the threshold image to the correct angle
	// then check each column 1 by 1.

	vector<int> leftEdges;
	vector<int> rightEdges;

	for( size_t i = 0; i < thresholds.size(); i++)
	{
		Mat rotated;

		if (top.angle > MIN_ANGLE_FOR_ROTATION)
		{
			// Rotate image:
			rotated = Mat(thresholds[i].size(), thresholds[i].type());
			Mat rot_mat( 2, 3, CV_32FC1 );
			Point center = Point( thresholds[i].cols/2, thresholds[i].rows/2 );

			rot_mat = getRotationMatrix2D( center, top.angle, 1.0 );
			warpAffine( thresholds[i], rotated, rot_mat, thresholds[i].size() );
		}
		else
		{
			rotated = thresholds[i];
		}

		int leftEdgeX = 0;
		int rightEdgeX = rotated.cols;
		// Do the left side
		int col = charRegions[0].x + charRegions[0].width;
		while (col >= 0)
		{
			int rowLength = getLongestBlobLengthBetweenLines(rotated, col, top, bottom);

			if (rowLength > MIN_CONNECTED_EDGE_PIXELS)
			{
				leftEdgeX = col;
				break;
			}

			col--;
		}

		col = charRegions[charRegions.size() - 1].x;
		while (col < rotated.cols)
		{
			int rowLength = getLongestBlobLengthBetweenLines(rotated, col, top, bottom);

			if (rowLength > MIN_CONNECTED_EDGE_PIXELS)
			{
				rightEdgeX = col;
				break;
			}
			col++;
		}

		if (leftEdgeX != 0)
		{
			leftEdges.push_back(leftEdgeX);
		}
		if (rightEdgeX != thresholds[i].cols)
		{
			rightEdges.push_back(rightEdgeX);
		}

		int leftEdge = 0;
		int rightEdge = thresholds[0].cols;

		// Assign the edge values to the SECOND closest value
		if (leftEdges.size() > 1)
		{
			sort (leftEdges.begin(), leftEdges.begin()+leftEdges.size());
			leftEdge = leftEdges[leftEdges.size() - 2] + 1;
		}
		if (rightEdges.size() > 1)
		{
			sort (rightEdges.begin(), rightEdges.begin()+rightEdges.size());
			rightEdge = rightEdges[1] - 1;
		}

		if (leftEdge != 0 || rightEdge != thresholds[0].cols)
		{
			Mat mask = Mat::zeros(thresholds[0].size(), CV_8U);
			rectangle(mask, Point(leftEdge, 0), Point(rightEdge, thresholds[0].rows), Scalar(255,255,255), -1);

			if (top.angle > MIN_ANGLE_FOR_ROTATION)
			{
			  // Rotate mask:
			  Mat rot_mat( 2, 3, CV_32FC1 );
			  Point center = Point( mask.cols/2, mask.rows/2 );

			  rot_mat = getRotationMatrix2D( center, top.angle * -1, 1.0 );
			  warpAffine( mask, mask, rot_mat, mask.size() );
			}

			// If our edge mask covers more than x% of the char region, mask the whole thing...
			const float MAX_COVERAGE_PERCENT = 0.6;
			int leftCoveragePx = leftEdge - charRegions[0].x;
			float leftCoveragePercent = ((float) leftCoveragePx) / ((float) charRegions[0].width);
			float rightCoveragePx = (charRegions[charRegions.size() -1].x + charRegions[charRegions.size() -1].width) - rightEdge;
			float rightCoveragePercent = ((float) rightCoveragePx) / ((float) charRegions[charRegions.size() -1].width);
			if ((leftCoveragePercent > MAX_COVERAGE_PERCENT) ||
				(charRegions[0].width - leftCoveragePx < MIN_SEG_WIDTH_PX ))
			{
				rectangle(mask, charRegions[0], Scalar(0,0,0), -1);	// Mask the whole region
				if (params.bDebug)
				{
					printLogInfo("Edge Filter: Entire left region is erased");
				}
			}
			if ((rightCoveragePercent > MAX_COVERAGE_PERCENT) ||
				(charRegions[charRegions.size() -1].width - rightCoveragePx < MIN_SEG_WIDTH_PX))
			{
				rectangle(mask, charRegions[charRegions.size() -1], Scalar(0,0,0), -1);
				if (params.bDebug)
				{
					printLogInfo("Edge Filter: Entire right region is erased");
				}
			}

			for (size_t i = 0; i < thresholds.size(); i++)
			{
				bitwise_and(thresholds[i], mask, thresholds[i]);
			}

			if (params.bDebug)
			{
				printLogInfo("Edge Filter: left=%d, right=%d", leftEdge, rightEdge);
			}
		}
	}
}

int FTS_IP_SimpleBlobDetector::getLongestBlobLengthBetweenLines( const Mat& img,
															     const int col,
															     FTS_BASE_LineSegment& top,
															     FTS_BASE_LineSegment& bottom )
{
  int longestBlobLength = 0;

  bool onSegment = false;
  bool wasbetweenLines = false;
  float curSegmentLength = 0;
  for (int row = 0; row < img.rows; row++)
  {
	bool isbetweenLines = false;

	bool isOn = img.at<uchar>(row, col);
	// check two rows at a time.
	if (!isOn && col < img.cols)
	  isOn = img.at<uchar>(row, col);

	if (isOn)
	{
	  // We're on a segment.  Increment the length
	  isbetweenLines =     top   .isPointBelowLine( Point(col, row) )
					   && !bottom.isPointBelowLine( Point(col, row) );
	  float incrementBy = 1;

	  // Add a little extra to the score if this is outside of the lines
	  if (!isbetweenLines)
		incrementBy = 1.1;

	  onSegment = true;
	  curSegmentLength += incrementBy;
	}
	if (isOn && isbetweenLines)
	{
	  wasbetweenLines = true;
	}

	if (onSegment && (isOn == false || (row == img.rows - 1)))
	{
	  if (wasbetweenLines && curSegmentLength > longestBlobLength)
		longestBlobLength = curSegmentLength;

	  onSegment = false;
	  isbetweenLines = false;
	  curSegmentLength = 0;
	}
  }

  return longestBlobLength;
}

int FTS_IP_SimpleBlobDetector::printLog(int level, const char* fmt, va_list ap)
{
	int ret = 0;
	if(this->m_poANPRObject)
	{
		if(level < ANPR_LOG_NONE) level = ANPR_LOG_NONE;
		if(level > ANPR_LOG_INFO) level = ANPR_LOG_INFO;
		this->m_poANPRObject->oDebugLogs.log(level, fmt, ap);
	}
	else
	{
		char buff[1024];
		vsprintf(buff, fmt, ap);
		printf(buff);
	}
	return ret;
}

//23.06 trungnt1 add to put debug logs to ANPR object
int FTS_IP_SimpleBlobDetector::printLogWarn(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int ret = this->printLog(ANPR_LOG_WARN, fmt, ap);
	va_end(ap);
	return ret;
}

//23.06 trungnt1 add to put debug logs to ANPR object
int FTS_IP_SimpleBlobDetector::printLogInfo(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int ret = this->printLog(ANPR_LOG_WARN, fmt, ap);
    va_end(ap);
	return ret;
}
