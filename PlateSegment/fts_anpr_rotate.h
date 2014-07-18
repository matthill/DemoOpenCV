#ifndef _FTS_ANPR_ROTATE_H_
#define _FTS_ANPR_ROTATE_H_

#include "fts_base_externals.h"
#include "fts_base_debug.h"

/*!
 *
 *
 *
 */
class FTS_ANPR_Rotate
{
public:
    explicit FTS_ANPR_Rotate();
    virtual ~FTS_ANPR_Rotate();

public:

    // Public interface
    virtual void rotate( const cv::Mat& oInput, cv::Mat& oOutput, bool bBlackChar=true );

    virtual bool findPlateAngle( const cv::Mat& oInput, double& rAngle, bool bBlackChar=true );

protected:

    cv::Mat m_oExpandedCrop;

    cv::Mat m_oTempBuffer1;
    cv::Mat m_oTempBuffer2;

    CvMemStorage* m_poStorage;


private:

    FTS_ANPR_Rotate( const FTS_ANPR_Rotate& r );
    FTS_ANPR_Rotate& operator=( const FTS_ANPR_Rotate& r );

};





#endif // _FTS_ANPR_ROTATE_H_
