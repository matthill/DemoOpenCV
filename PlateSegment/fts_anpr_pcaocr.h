/*
 * fts_anpr_pcaocr.h
 *
 *  Created on: May 11, 2014
 *      Author: sensen
 */

#ifndef FTS_ANPR_PCAOCR_H_
#define FTS_ANPR_PCAOCR_H_

#include "fts_base_externals.h"

class FTS_ANPR_PcaOcr
{

public:

	FTS_ANPR_PcaOcr();
	virtual ~FTS_ANPR_PcaOcr();

	enum
	{
		NUM_OF_COMPONENT = 10,
		STANDARD_PCA_CHAR_WIDTH  = 16,
		STANDARD_PCA_CHAR_HEIGHT = 32
	};

	bool load( const string& sTrainPath );
	string ocr( const cv::Mat& img ) const;

	// vectors to hold pca
	vector<PCA> m_voPca;

	// standard image size
	Size m_oStandardCharSize;

	// characters array
	vector<string> m_oCharClasses;


private:

	Mat formatImagesForPCA( const vector<Mat>& data );

};

#endif /* FTS_ANPR_PCAOCR_H_ */
