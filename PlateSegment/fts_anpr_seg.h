/*
 *
 * Seg.h
 *
 */

#ifndef _FTS_ANPR_SEG_H_
#define _FTS_ANPR_SEG_H_

#include "fts_base_common.h"
#include "fts_base_geometry.h"

/*!
 *
 * Individual segmented characters
 *
 */
class FTS_ANPR_SegChar
{
public:
    explicit FTS_ANPR_SegChar();
    virtual ~FTS_ANPR_SegChar();

public:
    // Public interfaces
    virtual bool clone( const FTS_ANPR_SegChar& r );
    virtual void crop( const cv::Rect oROI );

    inline int   getWidth () { return   m_oCharRect.width ; }
    inline int   getHeight() { return   m_oCharRect.height; }
    inline int   getArea  () { return   m_oCharRect.width
                                      * m_oCharRect.height; }

    inline float getHoW   () { return   (float) m_oCharRect.height
                                      / (float) m_oCharRect.width; }

    inline CvPoint2D32f getCentroid() { return cvPoint2D32f( (float)m_oCharRect.x + (float)m_oCharRect.width  / 2.0f,
                                                             (float)m_oCharRect.y + (float)m_oCharRect.height / 2.0f ); };

    // Comparison
    static inline bool LessInX( const FTS_ANPR_SegChar* a, const FTS_ANPR_SegChar* b )
    {
        return (   a->m_oCharRect.x
                 < b->m_oCharRect.x  );
    }

    // Duc 14th Aug 2012: Try to go straight to the characters
    // For the dual line plate
	static inline bool LessInYThenX( const FTS_ANPR_SegChar* a, const FTS_ANPR_SegChar* b )
	{
		// Consider y
		float y1 = a->m_oCharRect.y  +  a->m_oCharRect.height / 2.0f;
		if( y1 < b->m_oCharRect.y )
		{
			return true;
		}

		// Then x
		return (   a->m_oCharRect.x
				 < b->m_oCharRect.x  );
	}


public:
	bool m_bClean;
    unsigned int m_nTag;

    cv::Rect m_oCharRect;
    cv::Mat  m_oCharBin;
    cv::Mat  m_oCharGray;

protected:
    FTS_ANPR_SegChar( const FTS_ANPR_SegChar& );
    FTS_ANPR_SegChar& operator=( const FTS_ANPR_SegChar& );

};


class FTS_ANPR_SegResult
{
public:
    explicit FTS_ANPR_SegResult();
    explicit FTS_ANPR_SegResult( cv::Size oPlateSize );
    virtual ~FTS_ANPR_SegResult();

public:
    // Public interfaces
    virtual void clone( const FTS_ANPR_SegResult& oSrc );
    virtual void clear();

    virtual int   medianWidth ();
    virtual int   medianHeight();
    virtual int   medianArea  ();
    virtual float medianHoW   ();

    virtual bool overlapsWith( FTS_ANPR_SegResult* poSegResult );
    virtual float minorMajorAxesRatio();

    //TODO: maybe we should make a single template out of these
	virtual void getCharTopArray   ( FTS_BASE_StackArray<int>&   oArray ) const;
	virtual void getCharBottomArray( FTS_BASE_StackArray<int>&   oArray ) const;
	virtual void getWidthArray     ( FTS_BASE_StackArray<int>&   oArray ) const;
	virtual void getHeightArray    ( FTS_BASE_StackArray<int>&   oArray ) const;
	virtual void getAreaArray      ( FTS_BASE_StackArray<int>&   oArray ) const;
	virtual void getHoWArray       ( FTS_BASE_StackArray<float>& oArray ) const;
	virtual void getCharCentroidArray( FTS_BASE_StackArray<CvPoint2D32f>& oArray ) const;

public:
    cv::Size m_oPlateSize;

    // We need to sort these chars, so use a list to avoid copying.
    // You can easily create a vector of pointers to the content when
    // you need random access.
    std::list< FTS_ANPR_SegChar* >  m_oChars;

    // Duc 20/09/2012: Due to the go straight to characters,
    // we might temporarily want to hold the OCR result
//    std::string m_sTempOcr;

    // Duc 07/10/2012: Due to the go straight to characters,
	// Let say single-line hypothesis: SGJ8087L
    //         dual-line hypothesis  : GJ8087L
    // We should pick single-line as the best hypothesis
    bool m_bIsTheBestHypo;

private:
    FTS_ANPR_SegResult( const FTS_ANPR_SegResult& );
    FTS_ANPR_SegResult& operator=( const FTS_ANPR_SegResult& );
};


class FTS_ANPR_Seg
{
public:
    explicit FTS_ANPR_Seg();
    virtual ~FTS_ANPR_Seg();

    float m_nMinCharHeightToPlateHeightRatio;

	unsigned int m_nSmallCompAreaThresh;
	unsigned int m_nLargeCompAreaThresh;

	unsigned int m_nMaxNumCharCandidates;

	float  m_rMaxCharHeightOverWidthRatio;
	float  m_rMinCharHeightOverWidthRatio;

	unsigned int m_nIsSameSizeAndNoOverlapMaxWidthDiff;
	unsigned int m_nIsSameSizeAndNoOverlapMaxHeightDiff;

	cv::Mat m_oPadded;

	// Create memory oStorage
	CvMemStorage* m_poStorage;
	CvSeq* m_poSegCharSeq;

public:
	virtual int findCleanChar( FTS_ANPR_SegResult& oSegResult, CvCmpFunc is_equal );
	void maskCleanChar( FTS_ANPR_SegResult& oSegResult,
					    cv::Mat& oDst,
					    const cv::Scalar& oMaskValue );

    virtual void extractCharByCCAnalysis( const cv::Mat& oInput, FTS_ANPR_SegResult& oSegResult );

    virtual bool testArea( const cv::Rect& oBox );
    virtual bool testHeightOverWidth( const cv::Rect& oBox );
    virtual bool testHeight( int nCharHeight, int nPlateHeight );

    static int isSameSizeAndNoOverlap( const void* poSegChar1, const void* poSegChar2, void* poSeg );
    static int isSameHeight			 ( const void* poSegChar1, const void* poSegChar2, void* poSeg );

protected:
    // Temp variables
	std::vector<int> m_oIntVector;

private:
    FTS_ANPR_Seg( const FTS_ANPR_Seg& r );
    FTS_ANPR_Seg& operator=( const FTS_ANPR_Seg& r );
};

#endif // _FTS_ANPR_SEG_H_









