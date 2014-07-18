
#include "fts_base_common.h"

#ifndef _FTS_BASE_GEOMETRY_H_
#define _FTS_BASE_GEOMETRY_H_

class FTS_BASE_Geometry
{

public:

	         FTS_BASE_Geometry();
	virtual ~FTS_BASE_Geometry();


public:

	// constants
    enum
    {
    	   INSIDE  = 1
        , OUTSIDE  = 0
        , BINARYMAXVALUE    = 255
    };

    static int isInsideRectangle( CvRect oRect, CvPoint2D32f oPoint );

    static void MajorMinorAxes(
            CvPoint2D32f* poPoints,
            unsigned int nNumOfPoints,
            float& rMajorAxisLength,
            float& rMinorAxisLength );

};


class FTS_BASE_RotationSkewAngles
{

public:

			 FTS_BASE_RotationSkewAngles();
	virtual ~FTS_BASE_RotationSkewAngles();


public:

	float m_rRotationAngleDegrees;
	float m_rSkewAngleDegrees;


private: // Not copyable

    // Copy Constructor & Assignment Operator
	FTS_BASE_RotationSkewAngles( const FTS_BASE_RotationSkewAngles& rs );
	FTS_BASE_RotationSkewAngles& operator=( const FTS_BASE_RotationSkewAngles& rs );
};

#endif // _FTS_BASE_GEOMETRY_H_












