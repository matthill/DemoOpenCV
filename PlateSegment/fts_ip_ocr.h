
#ifndef _FTS_ANPR_OCR_H
#define _FTS_ANPR_OCR_H

#include "fts_base_externals.h"
#include "fts_ip_postprocess.h"
#include "fts_anpr_object.h"

#include "baseapi.h"
using namespace tesseract;


#define POST_PROCESS

class FTS_ANPR_TessOcr
{
  public:
    FTS_ANPR_TessOcr(string tessDataDir, string ocrLanguage, int minCharToProcess, int m_iOcrMinFontSize, bool bDebugOCR);
    virtual ~FTS_ANPR_TessOcr();

#ifdef POST_PROCESS
	void initPostProcess(string patternsFile, int minConfidence, int confidenceSkipLevel, int maxSubs, bool debugPP);
#endif
    void performOCR( const vector<Mat>& thresholds,
    				 const vector< vector<Rect> >& charRegions,
    				 //OcrResult& ocrResults ); //21.06 trungnt1 add this params to separate ocr & post process
					 FTS_ANPR_OBJECT& oAnprObject ); //27.06 trungnt1 change to FTS_ANPR_OBJECT to put result & debug infos

	void clean();

#ifdef POST_PROCESS
    FTS_ANPR_PostProcess* postProcessor;
#endif
    //string recognizedText;
    //float confidence;
    //float overallConfidence;

	int m_iPostProcessMinCharacters;
	bool m_bDebugOcr;
	int m_iOcrMinFontSize;

  private:
    //Config* config;

    TessBaseAPI *tesseract;

};

#endif // _FTS_ANPR_OCR_H
