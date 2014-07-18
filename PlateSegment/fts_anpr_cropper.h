/*
 *
 * FTS_ANPR_Cropper.h
 *
 */
#ifndef _FTS_ANPR_CROPPER_H_
#define _FTS_ANPR_CROPPER_H_

#include "fts_base_externals.h"

/*!
 *
 *
 *
 */
class FTS_ANPR_Cropper
{
public:
    explicit FTS_ANPR_Cropper();
    virtual ~FTS_ANPR_Cropper();

public:

    // Public interface
    virtual bool processDetection( IplImage* poSrc, CvRect& oCropRect );
    virtual bool processDetection( const cv::Mat& oSrc, CvRect& oCropRect );
    void uninitParameters();
    void computeDefaultParamters();


public:

    // Parameters
    // TODO: many of these should be xml settings

    float m_rSumVerThreshFactor;
    float m_rHorSmallSmoothThreshFrac;
    float m_rHorBoundaryWindowLoThresh;
    float m_rHorBoundaryWindowHiThresh;
    float m_rVerBoundaryWindowThresh;
    float m_rNonPlateTestThresh1;
    float m_rNonPlateTestThresh2;

    unsigned int m_nMaxPlateHalfWidth;
    unsigned int m_nHorLargeSmoothHalfWindow;
    unsigned int m_nHorSmallSmoothHalfWindow;
    unsigned int m_nHorStepHalfWindow;
    unsigned int m_nBoundaryWindowMaxWidth;
    unsigned int m_nVerSmallSmoothHalfWindow;
    unsigned int m_nMaxHalfPlateSize;
    unsigned int m_nEdgeTestPlateWidthMin;
    unsigned int m_nEdgeTestPlateHeightMin;
    unsigned int m_nEdgeTestVerticalSlideWindowProjectionMin;
    unsigned int m_nEdgeTestVerticalSlideWindowWidth;

    unsigned int m_nSumVerArrSize;
	std::vector< double > m_oSumVer;
	std::vector< double > m_oSumVerRunSum;
	std::vector< double > m_oSumVerSmoothLarge;
	std::vector< double > m_oSumVerSmoothSmall;
	std::vector< double > m_oSumVerStepFiltered;

	unsigned int m_nSumHorArrSize;
	std::vector< double > m_oSumHor;
	std::vector< double > m_oSumHorRunSum;
	std::vector< double > m_oSumHorSmooth;


protected:

    //! The main cropping algorithm
	bool crop( const cv::Mat& oSrc, CvRect& oCropRect );

    bool crop( IplImage* poSrc, CvRect& oCropRect );

    void leftRightCrop( IplImage& oEdge16SC1,
                        unsigned int& nLeftOut,
                        unsigned int& nRightOut,
                        double&       rSumVerMeanOut );
    void topBottomCrop( IplImage& oEdge16SC1,
                        unsigned int& nTopOut,
                        unsigned int& nBottomOut );

    int searchLeftTillBelowThreshold ( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold );
    int searchLeftTillAboveThreshold ( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold );
    int searchRightTillBelowThreshold( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold );
    int searchRightTillAboveThreshold( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold );

    int searchTillBelowThreshold( double* prFirst, double* prLast, double rThreshold, bool bSearchRight );
    void stepFilter( double* prRunSum, double* prDst, int nLen, int nHalfStep );
    void smooth( double* prRunSum, double* prDst, int nLen, int nHalfWind );
    void runSum( double* prSrc, double* prDst, int nLen );
    void clipHi( double* prSrc, double* prDst, int nLen, double rThresh );
    double mean( double* prSrc, int nLen );
    void sumVer16SC1( IplImage* poSrc, double* rSum );
    void sumHor16SC1( IplImage* poSrc, double* rSum );

    void computeDefaultParamter( unsigned int nParameter, float rScale );


protected:

    static const unsigned int UNINITIALISED;

    static const float s_rScaleHorLargeSmoothHalfWindow;
    static const float s_rScaleHorSmallSmoothHalfWindow;
    static const float s_rScaleHorStepHalfWindow;
    static const float s_rScaleBoundaryWindowMaxWidth;
    static const float s_rScaleVerSmallSmoothHalfWindow;
    static const float s_rScaleEdgeTestPlateWidthMin;
    static const float s_rScaleEdgeTestPlateHeightMin;



private:

    FTS_ANPR_Cropper( const FTS_ANPR_Cropper& r );
    FTS_ANPR_Cropper& operator=( const FTS_ANPR_Cropper& r );

};





#endif // _FTS_ANPR_CROPPER_H_
