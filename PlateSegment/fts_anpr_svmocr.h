
#ifndef _FTS_ANPR_SVM_OCR_H
#define _FTS_ANPR_SVM_OCR_H

#include "fts_base_externals.h"
#include "fts_ip_postprocess.h"
#include "fts_anpr_object.h"

#ifdef USING_SVM_OCR
#include <Hog_svm_classifiers.h>
#ifdef _DEBUG
#pragma comment(lib, "HOG_SVM_LIBd.lib")
#else
#pragma comment(lib, "HOG_SVM_LIB.lib")
#endif
#endif

#define POST_PROCESS

class FTS_ANPR_SvmOcr
{
  public:
	  FTS_ANPR_SvmOcr(string _digitModelFolderPath, string _letterModelsPath, string pairModelsPath = "", string _pairModelPrefix = "model", int minCharToProcess = 6, bool bDebugOCR = false);
    virtual ~FTS_ANPR_SvmOcr();

	bool isInitialized();

#ifdef POST_PROCESS
	void initPostProcess(string patternsFile, int minConfidence, int confidenceSkipLevel, int maxSubs, bool debugPP);
#endif
    void performOCR( const Mat& oSrcImg,
    				 const vector<Rect>& charRegions,
    				 //OcrResult& ocrResults ); //21.06 trungnt1 add this params to separate ocr & post process
					 FTS_ANPR_OBJECT& oAnprObject ); //27.06 trungnt1 change to FTS_ANPR_OBJECT to put result & debug infos

	void clean();

#ifdef POST_PROCESS
    FTS_ANPR_PostProcess* postProcessor;
#endif
    
	int m_iPostProcessMinCharacters;
	bool m_bDebugOcr;
	bool isDigit(const std::string& s);

#ifdef USING_SVM_OCR
	bool loadDigitModels(string _digitModelsPath);
	bool loadLetterModels(string _letterModelsPath);
	bool loadAllCharModels(string _digitModelsPath, string _letterModelsPath, string _digitLetterModelsPath = "");
	void getStandardCharImgs(Mat src, const vector<Rect>& bbs, vector<Mat>& charImgs, int padding = 0);
	bool findBestChar(const cv::Mat& charImg, const std::vector<character>& listChars, character& bestChar, CLASSIFIER_SET mode = DIGIT_LETTER);
	bool findBestChar(const std::vector<float>& featureVector, const std::vector<character>& listChars, character& bestChar, CLASSIFIER_SET mode = DIGIT_LETTER);
	bool checkCharType(character ch, CLASSIFIER_SET charType);
	void findBestCharBasedConfidence(const std::vector<character>& listChars, character& bestChar);
#endif

private:
    bool m_bInitialized;
	float aspectRatio;
	/*std::string digitModelsPath;
	std::string letterModelsPath;
	std::string digitLetterModelsPath;*/

	std::map<std::string, std::string> pairModelPathMap;
	std::string pairModelPrefix;
#ifdef USING_SVM_OCR 
    HogSvmClassifiers digitClassifier;
	HogSvmClassifiers letterClassifier;
	HogSvmClassifiers digitLetterClassifier;
#endif
};

#endif // _FTS_ANPR_SVM_OCR_H
