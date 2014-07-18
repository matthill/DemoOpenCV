/*
 * fts_base_linesegment.h
 *
 *  Created on: May 7, 2014
 *      Author: sensen
 */

#ifndef FTS_BASE_LINESEGMENT_H_
#define FTS_BASE_LINESEGMENT_H_

#include "fts_base_externals.h"
#include "fts_base_util.h"

//using namespace cv;

class FTS_BASE_LineSegment
{

  public:
	Point p1, p2;
	float slope;
	float length;
	float angle;

	// FTS_BASE_LineSegment(Point point1, Point point2);
	FTS_BASE_LineSegment();
	FTS_BASE_LineSegment(int x1, int y1, int x2, int y2);
	FTS_BASE_LineSegment(Point p1, Point p2);

	void init(int x1, int y1, int x2, int y2);

	bool isPointBelowLine(Point tp);

	float getPointAt(float x) const;

	Point closestPointOnSegmentTo(Point p);

	Point intersection(FTS_BASE_LineSegment line);

	FTS_BASE_LineSegment getParallelLine(float distance);

	Point midpoint();

	inline std::string str()
	{
	  std::stringstream ss;
	  ss << "(" << p1.x << ", " << p1.y << ") : (" << p2.x << ", " << p2.y << ")";
	  return ss.str() ;
	}

};

#endif /* FTS_BASE_LINESEGMENT_H_ */
