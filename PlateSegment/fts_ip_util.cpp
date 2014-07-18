#include "fts_ip_util.h"
#include "fts_base_linesegment.h"
#include <boost/regex.hpp>

using namespace std;

const float FTS_IP_Util::SAMPLE_CONST 		= 0;

FTS_IP_Util::FTS_IP_Util()
{
	// Nothing
}

FTS_IP_Util::~FTS_IP_Util()
{
	// Nothing
}

bool FTS_IP_Util::MaskByLargestCC( const cv::Mat& oGray,
		   	   	   	   	   	   	   	   	 cv::Mat& oMask,
								   const float rMinWidthRatio,
								   bool bBlackChar )
{
	// Binarize
	int nThreshType = bBlackChar?CV_THRESH_BINARY:CV_THRESH_BINARY_INV;
	cv::Mat oBin;
	cv::threshold( oGray, oBin, 0, 255, nThreshType | CV_THRESH_OTSU );

	// Find contour
	std::vector < std::vector<cv::Point> > contours;
	findContours( oBin, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// Iterate contours to find the largest CC
	unsigned int nImgArea = oBin.cols * oBin.rows;
	int nMaxWidth = 0;
	int nMaxWidthIdx = 0;
	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		double rCCSize = fabs( cv::contourArea( contours[contourIdx] ) );
		if ( rCCSize/(double)nImgArea > 0.40 ) //! if contour size is larger than 40% of the image then it's likely to be the plate
		{
			cv::Rect oBoundingbox = cv::boundingRect( contours[contourIdx] );
			if( oBoundingbox.width > nMaxWidth )
			{
				nMaxWidth = oBoundingbox.width;
				nMaxWidthIdx = contourIdx;
			}
		}
	}

	// If the largest CC is not wide enough, do not proceed
	if( (float)nMaxWidth / (float)oBin.cols < rMinWidthRatio )
	{
		oMask.create( oBin.size(), oBin.type() );
		oMask = cv::Scalar(255);
		return false;
	}

	// Convexhull
	FindConvexHull( oBin.size(), contours[nMaxWidthIdx], oMask );
//	oMask = cv::Mat::zeros( oBin.size(), oBin.type() );
//	cv::drawContours( oMask, contours, nMaxWidthIdx, cv::Scalar( 255 ), CV_FILLED );
//	std::vector<std::vector<cv::Point> >hull(1);
//	cv::convexHull( cv::Mat(contours[nMaxWidthIdx]), hull[0], false );
//	cv::fillConvexPoly( oMask, cv::Mat(hull[0]), cv::Scalar( 255 ) );

	return true;
}

std::vector<cv::Point> FTS_IP_Util::FindLargestCC( const cv::Mat& oBin )
{
	Mat oCopy = oBin.clone();

	// Find contour
	std::vector < std::vector<cv::Point> > contours;
	findContours( oCopy, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	if( contours.empty() )
	{
		std::vector<cv::Point> emptyVec;
		return emptyVec;
	}

	// Iterate contours to find the largest CC
	int nMaxArea = 0;
	int nMaxAreaIdx = 0;
	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		cv::Rect oBoundingbox = cv::boundingRect( contours[contourIdx] );
		if( oBoundingbox.width * oBoundingbox.height > nMaxArea )
		{
			nMaxArea = oBoundingbox.width * oBoundingbox.height;
			nMaxAreaIdx = contourIdx;
		}
	}

	return contours[nMaxAreaIdx];
}

void FTS_IP_Util::FindConvexHull( const cv::Size& oSize,
								  const std::vector<cv::Point> points,
								  cv::Mat& oConvexHull )
{
	if( points.size() < 1 )
	{
		oConvexHull = cv::Mat::ones( oSize, CV_8UC1 );
		return;
	}

	oConvexHull = cv::Mat::zeros( oSize, CV_8UC1 );
	std::vector<std::vector<cv::Point> >hull(1);
	cv::convexHull( cv::Mat( points ), hull[0], false );
	cv::fillConvexPoly( oConvexHull, cv::Mat(hull[0]), cv::Scalar( 255 ) );
}


vector<Point> FTS_IP_Util::getBoundingPolygonFromContours( const int cols,
														   const int rows,
														   vector<vector<Point> > contours,
														   vector<bool> goodIndices )
{
	vector<Point> bestStripe;

	vector<Rect> charRegions;

	for( unsigned int i = 0; i < contours.size(); i++)
	{
		if (goodIndices[i])
		{
			charRegions.push_back(boundingRect(contours[i]));
		}
	}

	// Find the best fit line segment that is parallel with the most char segments
	if (charRegions.size() <= 1)
	{
		bestStripe.push_back( Point( 0, 0 ) );
		bestStripe.push_back( Point( cols - 1, 0 ) );
		bestStripe.push_back( Point( cols - 1, rows - 1 ) );
		bestStripe.push_back( Point( 0, rows - 1 ) );
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
				  leftRect = &charRegions[i];
				  rightRect = &charRegions[k];
				}
				else
				{
				  leftRect = &charRegions[k];
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
				topLines.push_back(FTS_BASE_LineSegment(x1, y1, x2, y2));

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
		if( topLeft.y  > 0 ) topLeft.y--;
		if( topRight.y > 0 ) topRight.y--;
		if( bottomRight.y < rows - 1 ) bottomRight.y++;
		if( bottomLeft.y < rows - 1 ) bottomLeft.y++;

		bestStripe.push_back(topLeft);
		bestStripe.push_back(topRight);
		bestStripe.push_back(bottomRight);
		bestStripe.push_back(bottomLeft);
	}

	return bestStripe;
}

vector<Point> FTS_IP_Util::getBoundingPolygonFromBoxes( const int cols,
													    const int rows,
													    const vector<Rect>& oBoxes,
													    const vector<bool>& goodIndices )
{
	vector<Point> bestStripe;

	vector<Rect> charRegions;

	for( unsigned int i = 0; i < oBoxes.size(); i++)
	{
		if (goodIndices[i])
		{
			charRegions.push_back( oBoxes[i] );
		}
	}

	// Find the best fit line segment that is parallel with the most char segments
	if (charRegions.size() <= 1)
	{
		bestStripe.push_back( Point( 0, 0 ) );
		bestStripe.push_back( Point( cols - 1, 0 ) );
		bestStripe.push_back( Point( cols - 1, rows - 1 ) );
		bestStripe.push_back( Point( 0, rows - 1 ) );
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
		if( topLeft.y  > 0 ) topLeft.y--;
		if( topRight.y > 0 ) topRight.y--;
		if( bottomRight.y < rows - 1 ) bottomRight.y++;
		if( bottomLeft.y < rows - 1 ) bottomLeft.y++;

		bestStripe.push_back(topLeft);
		bestStripe.push_back(topRight);
		bestStripe.push_back(bottomRight);
		bestStripe.push_back(bottomLeft);
	}

	return bestStripe;
}



vector<FTS_BASE_LineSegment> FTS_IP_Util::FindBoundingFromEdges( Mat edges, float sensitivityMultiplier, bool vertical)
{
	static int HORIZONTAL_SENSITIVITY = 45;
	static int VERTICAL_SENSITIVITY   = 25;

	vector<Vec2f> allLines;
	vector<FTS_BASE_LineSegment> filteredLines;

	int sensitivity;
	if (vertical)
	{
		sensitivity = VERTICAL_SENSITIVITY * (1.0 / sensitivityMultiplier);
	}
	else
	{
		sensitivity = HORIZONTAL_SENSITIVITY * (1.0 / sensitivityMultiplier);
	}

	HoughLines( edges, allLines, 1, CV_PI/180, sensitivity, 0, 0 );

	for( size_t i = 0; i < allLines.size(); i++ )
	{
	float rho = allLines[i][0], theta = allLines[i][1];
	Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;

	double angle = theta * (180 / CV_PI);
	pt1.x = cvRound(x0 + 1000*(-b));
	pt1.y = cvRound(y0 + 1000*(a));
	pt2.x = cvRound(x0 - 1000*(-b));
	pt2.y = cvRound(y0 - 1000*(a));

	if (vertical)
	{
		if (angle < 20 || angle > 340 || (angle > 160 && angle < 210))
		{
			FTS_BASE_LineSegment line;
			if (pt1.y <= pt2.y)
			{
				line = FTS_BASE_LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);
			}
			else
			{
				line = FTS_BASE_LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);
			}

			FTS_BASE_LineSegment top(0, 0, edges.cols, 0);
			FTS_BASE_LineSegment bottom(0, edges.rows, edges.cols, edges.rows);
			Point p1 = line.intersection(bottom);
			Point p2 = line.intersection(top);
			filteredLines.push_back(FTS_BASE_LineSegment(p1.x, p1.y, p2.x, p2.y));
		}
	}
	else
	{
		if ( (angle > 70 && angle < 110) || (angle > 250 && angle < 290))
		{
			FTS_BASE_LineSegment line;
			if (pt1.x <= pt2.x)
			{
				line = FTS_BASE_LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);
			}
			else
			{
				line =FTS_BASE_LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);
			}

			int newY1 = line.getPointAt(0);
			int newY2 = line.getPointAt(edges.cols);

			filteredLines.push_back(FTS_BASE_LineSegment(0, newY1, edges.cols, newY2));
		}
	}
	}

	return filteredLines;
}

cv::Rect FTS_IP_Util::MinAreaRect( const cv::Mat& oBin )
{
	// Store the set of points in the image before assembling the bounding box
	std::vector<cv::Point> points;

	for( int y = 0; y < oBin.rows; y++ )
	{
		for( int x = 0; x < oBin.cols; x++ )
		{
			if( oBin.at<uchar>( y,x ) != 0 )
			{
				points.push_back( Point(x,y) );
			}
		}
	}

	if( points.size() < 2 )
	{
		return Rect( 0, 0, oBin.cols, oBin.rows );
	}

	return cv::boundingRect( points );
//	// Compute minimal bounding box
//	cv::RotatedRect box = cv::minAreaRect( cv::Mat( points ) );
//
//	// Set Region of Interest to the area defined by the box
//	printf( "Center x = %f, y = %f, w = %f, h = %f\n",
//			box.center.x, box.center.y,
//			box.size.width, box.size.height );
//	cv::Rect roi;
//	roi.x = (int)( box.center.x - ((float)box.size.width / 2) );
//	roi.y = (int)( box.center.y - ((float)box.size.height / 2) );
//	roi.width  = (int)box.size.width;
//	roi.height = (int)box.size.height;
//
//	if (roi.x < 0)
//	{
//		roi.x = 0;
//	}
//	if (roi.y < 0)
//	{
//		roi.y = 0;
//	}
//	if (roi.x + roi.width > oBin.cols )
//	{
//		roi.width = oBin.cols - roi.x;
//	}
//	if (roi.y + roi.height > oBin.rows )
//	{
//		roi.height = oBin.rows - roi.y;
//	}
//
//	// Crop the original image to the defined ROI
//	return roi;
}

Rect FTS_IP_Util::expandRectXY( const Rect& original,
							    const int& expandXPixels,
							    const int& expandYPixels,
							    const int& maxX,
							    const int& maxY)
{
  Rect expandedRegion = Rect(original);

  float halfX = round((float) expandXPixels / 2.0);
  float halfY = round((float) expandYPixels / 2.0);
  expandedRegion.x = expandedRegion.x - halfX;
  expandedRegion.width =  expandedRegion.width + expandXPixels;
  expandedRegion.y = expandedRegion.y - halfY;
  expandedRegion.height =  expandedRegion.height + expandYPixels;

  if (expandedRegion.x < 0)
    expandedRegion.x = 0;
  if (expandedRegion.y < 0)
    expandedRegion.y = 0;
  if (expandedRegion.x + expandedRegion.width > maxX)
    expandedRegion.width = maxX - expandedRegion.x;
  if (expandedRegion.y + expandedRegion.height > maxY)
    expandedRegion.height = maxY - expandedRegion.y;

  return expandedRegion;
}

Rect FTS_IP_Util::expandRectTBLR( const Rect& original,
								  const int& top,
								  const int& bottom,
								  const int& left,
								  const int& right,
								  const int& maxX,
								  const int& maxY )
{
	Rect expandedRegion = Rect(original);

	expandedRegion.x = expandedRegion.x - left;
	expandedRegion.width =  expandedRegion.width + left + right;
	expandedRegion.y = expandedRegion.y - top;
	expandedRegion.height =  expandedRegion.height + top + bottom;

	if (expandedRegion.x < 0) expandedRegion.x = 0;
	if (expandedRegion.y < 0) expandedRegion.y = 0;
	if (expandedRegion.x + expandedRegion.width > maxX)
		expandedRegion.width = maxX - expandedRegion.x;
	if (expandedRegion.y + expandedRegion.height > maxY)
		expandedRegion.height = maxY - expandedRegion.y;

	return expandedRegion;
}

Rect FTS_IP_Util::expandRectTBLR( const Rect& original,
								  const ExpandByPixels& exp,
								  const int& maxX,
								  const int& maxY )
{
	return expandRectTBLR( original, exp.nT, exp.nB, exp.nL, exp.nR, maxX, maxY );
}

void FTS_IP_Util::findMinMaxX( const vector < vector<FTS_IP_SimpleBlobDetector::SimpleBlob> >& ovvAllBlobs,
				  const int nXLimit,
				  const int nMinArrSize,
				  const FTS_BASE_LineSegment& oTopLine,
				  const FTS_BASE_LineSegment& oBottomLine,
				  int& nMinX,
				  int& nMaxX,
				  int& nADjustedMinX,
				  int& nADjustedMaxX )
{
	nMinX = nXLimit;
	nMaxX = 0;
	for (size_t j = 0; j < ovvAllBlobs.size(); j++)
	{
		if ( (int)ovvAllBlobs[j].size() < nMinArrSize ) continue;

		printf( "Group size = %d\n", ovvAllBlobs[j].size() );
		for( size_t m = 0; m < ovvAllBlobs[j].size(); m++ )
		{
			cv::Rect oBox = ovvAllBlobs[j][m].oBB;
			printf( "Candidate %d, x = %d, y = %d, w = %d, h = %d - ", m+1, oBox.x, oBox.y, oBox.width, oBox.height );

			float rCenterX = (float)oBox.x + (float)oBox.width / 2;
			float rCenterY = (float)oBox.y + (float)oBox.height / 2;

			float rTopY    = oTopLine.getPointAt( rCenterX );
			float rBottomY = oBottomLine.getPointAt( rCenterX );

			float rHalfHeight = ( rBottomY - rTopY ) / 2;
			float rHalfY = rTopY + rHalfHeight;

//			printf( "x = %d, y = %d, rCenterX = %f, rCenterY = %f, top Y = %f, bottom Y = %f, rHalfY = %f, rHalfHeight = %f\n",
//					oBox.x, oBox.y,rCenterX,
//									rCenterY,
//									oTopLine.getPointAt( rCenterX ),
//									oBottomLine.getPointAt( rCenterX ),
//									rHalfY,
//									rHalfHeight );

			// Make sure top and bottom line not intersect
			if( rHalfHeight == 0 )
			{
				continue;
			}

			// Make sure the center is not far from the center of top and bottom lines
			float rDiffYRatio = abs( rCenterY - rHalfY ) / rHalfHeight;
			if( rDiffYRatio > 0.5  )	// TODO DV: setting?
			{
				printf( "Blob center is way too disaligned. rCenterY = %f, top Y = %f, bottom Y = %f\n",
						rCenterY, oTopLine.getPointAt( rCenterX ), oBottomLine.getPointAt( rCenterX ) );
				continue;
			}

			// Make sure the blob is not too tall (edge blob)
			if( (float)oBox.height > 1.5 * ( rBottomY - rTopY ) )
			{
				printf( "Blob is way too tall, height = %d\n", oBox.height );
				continue;
			}

			// Find max min X
			bool bNewMinX = nMinX > oBox.x ;
			bool bNewMaxX = nMaxX < oBox.x + oBox.width;
			if( bNewMinX )
			{
				nMinX = oBox.x;
				printf( "New Min X = %d. ", nMinX  );
			}
			if( bNewMaxX )
			{
				nMaxX = oBox.x + oBox.width;
				printf( "New Max X = %d.", nMaxX  );
			}

			printf( "\n" );
		}
	}

	// Only accept min, max X coords of safe width
	printf( "MinX = %d, MaxX = %d, width = %d\n", nMinX, nMaxX, (nXLimit+1) );
	if( (nMaxX - nMinX ) < ceil( 0.65 * (nXLimit+1) ) )
	{
		nADjustedMinX = 0;
		nADjustedMaxX = nXLimit;
	}
	else
	{
		int nSafePadded = 2;	// TODO DV: setting
		nMinX = nMinX - nSafePadded;
		nMaxX = nMaxX + nSafePadded;
		if( nMinX < 0 ) nMinX = 0;
		if( nMaxX > nXLimit ) nMaxX = nXLimit;

		nADjustedMinX = nMinX;
		nADjustedMaxX = nMaxX;
	}
	printf( "After adjust, MinX = %d, MaxX = %d, width = %d\n", nADjustedMinX, nADjustedMaxX, (nXLimit+1) );
}

void FTS_IP_Util::FindBlobsMinMaxX( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& ovAllBlobs,
										  int& nMinX,
										  int& nMaxX )
{
	nMinX = INT_MAX;
	nMaxX = INT_MIN;
	for( size_t m = 0; m < ovAllBlobs.size(); m++ )
	{
		cv::Rect oBox = ovAllBlobs[m].oBB;

		bool bNewMinX = nMinX > oBox.x ;
		bool bNewMaxX = nMaxX < oBox.x + oBox.width;
		if( bNewMinX )
		{
			nMinX = oBox.x;
		}
		if( bNewMaxX )
		{
			nMaxX = oBox.x + oBox.width;
		}
	}
}

void FTS_IP_Util::RemoveNoisyBlobs( std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
	std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::iterator it = blobs.begin();
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

std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> FTS_IP_Util::GetNRemoveNoisyBlobs( std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
	std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oRetBlobs;

	std::vector<FTS_IP_SimpleBlobDetector::SimpleBlob>::iterator it = blobs.begin();
	for ( ; it != blobs.end(); )
	{
		if(    strcmp( it->sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE.c_str() ) != 0
			&& strcmp( it->sStatus.c_str(), FTS_IP_SimpleBlobDetector::s_sSTATUS_EDGE_SAVED.c_str() ) != 0 )
		{
			// DV: change status back to normal then move it to the archive
			it->sStatus = FTS_IP_SimpleBlobDetector::s_sSTATUS_CANDIDATE;
			oRetBlobs.push_back( (*it) );
			it = blobs.erase(it);
		}
		else
		{
			++it;
		}
	}

	return oRetBlobs;
}


void FTS_IP_Util::setLabel( Mat& im, const string& label, vector<Point>& contour )
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::Rect r = cv::boundingRect(contour);

    cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(255,0,0), thickness, 8);
}

double FTS_IP_Util::angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

Rect FTS_IP_Util::getExactBB( const Mat& oSrcBin,
							  const Rect& oBB,
							  const int nExpandX,
							  const int nExpandY )
{
	// Character bounding box, expand it if requested
	cv::Rect oExpandedBox( oBB );
	if( nExpandX != 0 || nExpandY != 0 )
	{
		oExpandedBox = FTS_IP_Util::expandRectXY( oBB, 1, 1, oSrcBin.cols, oSrcBin.rows );
	}

	// Crop the expanded box
	cv::Mat oCharBin = oSrcBin(oExpandedBox);

	// Crop black pixels
	cv::Rect oSubBox = FTS_IP_Util::MinAreaRect( oCharBin );

	cv::Rect oFinalBox( oExpandedBox );
			 oFinalBox.x 	 += oSubBox.x;
			 oFinalBox.y 	 += oSubBox.y;
			 oFinalBox.width  = oSubBox.width;
			 oFinalBox.height = oSubBox.height;

	return oFinalBox;
}

void FTS_IP_Util::MorphBinary( Mat& oSrc, Mat& oDst, int nThresholdType )
{
	if( oSrc.rows < 19 || oSrc.cols < 19 )
	{
		threshold( oSrc, oDst, -1, 255, nThresholdType | THRESH_OTSU );
		return;
	}

	oDst = oSrc.clone();

	// Divide the oSrc by its morphologically closed counterpart
	Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size(9,9) );
	Mat closed;
	morphologyEx( oDst, closed, MORPH_CLOSE, kernel );

	oDst.convertTo(oDst, CV_32F); // divide requires floating-point
	divide( oDst, closed, oDst, 1, CV_32F );
	normalize( oDst, oDst, 0, 255, NORM_MINMAX );
	oSrc.convertTo( oDst, CV_8UC1 ); // convert back to unsigned int

	// Threshold each block (3x3 grid) of the oSrc separately to
	// correct for minor differences in contrast across the oSrc.
	int nHLength = oSrc.cols/3;
	int nVLength = oSrc.rows/3;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Mat block = oDst.rowRange(nVLength*i, nVLength*(i+1)).colRange(nHLength*j, nHLength*(j+1));
			threshold( block, block, -1, 255, nThresholdType | THRESH_OTSU );
		}
	}
}
