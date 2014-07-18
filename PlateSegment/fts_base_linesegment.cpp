/*
 * fts_base_linesegment.cpp
 *
 *  Created on: May 7, 2014
 *      Author: sensen
 */

#include "fts_base_linesegment.h"

FTS_BASE_LineSegment::FTS_BASE_LineSegment()
{
  init(0, 0, 0, 0);
}

FTS_BASE_LineSegment::FTS_BASE_LineSegment(Point p1, Point p2)
{
  init(p1.x, p1.y, p2.x, p2.y);
}

FTS_BASE_LineSegment::FTS_BASE_LineSegment(int x1, int y1, int x2, int y2)
{
  init(x1, y1, x2, y2);
}

void FTS_BASE_LineSegment::init(int x1, int y1, int x2, int y2)
{
  this->p1 = Point(x1, y1);
  this->p2 = Point(x2, y2);

  if (p2.x - p1.x == 0)
	this->slope = 0.00000000001;
  else
	this->slope = (float) (p2.y - p1.y) / (float) (p2.x - p1.x);

  this->length = FTS_BASE_Util::distanceBetweenPoints(p1, p2);

  this->angle = FTS_BASE_Util::angleBetweenPoints(p1, p2);
}

bool FTS_BASE_LineSegment::isPointBelowLine( Point tp )
{
  return ((p2.x - p1.x)*(tp.y - p1.y) - (p2.y - p1.y)*(tp.x - p1.x)) > 0;
}

float FTS_BASE_LineSegment::getPointAt(float x) const
{
  return slope * (x - p2.x) + p2.y;
}

Point FTS_BASE_LineSegment::closestPointOnSegmentTo(Point p)
{
  float top = (p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y)*(p2.y - p1.y);

  float bottom = FTS_BASE_Util::distanceBetweenPoints(p2, p1);
  bottom = bottom * bottom;

  float u = top / bottom;

  float x = p1.x + u * (p2.x - p1.x);
  float y = p1.y + u * (p2.y - p1.y);

  return Point(x, y);
}

Point FTS_BASE_LineSegment::intersection(FTS_BASE_LineSegment line)
{
  float c1, c2;
  float intersection_X = -1, intersection_Y= -1;

  c1 = p1.y - slope * p1.x; // which is same as y2 - slope * x2

  c2 = line.p2.y - line.slope * line.p2.x; // which is same as y2 - slope * x2

  if( (slope - line.slope) == 0)
  {
	//std::cout << "No Intersection between the lines" << endl;
  }
  else if (p1.x == p2.x)
  {
	// Line1 is vertical
	return Point(p1.x, line.getPointAt(p1.x));
  }
  else if (line.p1.x == line.p2.x)
  {
	// Line2 is vertical
	return Point(line.p1.x, getPointAt(line.p1.x));
  }
  else
  {
	intersection_X = (c2 - c1) / (slope - line.slope);
	intersection_Y = slope * intersection_X + c1;
  }

  return Point(intersection_X, intersection_Y);
}

Point FTS_BASE_LineSegment::midpoint()
{
  // Handle the case where the line is vertical
  if (p1.x == p2.x)
  {
	float ydiff = p2.y-p1.y;
	float y = p1.y + (ydiff/2);
	return Point(p1.x, y);
  }
  float diff = p2.x - p1.x;
  float midX = ((float) p1.x) + (diff / 2);
  int midY = getPointAt(midX);

  return Point(midX, midY);
}

#ifdef WIN32
static inline double round(double val)
{   
    return floor(val + 0.5);
}
#endif

FTS_BASE_LineSegment FTS_BASE_LineSegment::getParallelLine(float distance)
{
  float diff_x = p2.x - p1.x;
  float diff_y = p2.y - p1.y;
  float angle = atan2( diff_x, diff_y);
  float dist_x = distance * cos(angle);
  float dist_y = -distance * sin(angle);

  int offsetX = (int)round(dist_x);
  int offsetY = (int)round(dist_y);

  FTS_BASE_LineSegment result(p1.x + offsetX, p1.y + offsetY,
					 p2.x + offsetX, p2.y + offsetY);

  return result;
}

