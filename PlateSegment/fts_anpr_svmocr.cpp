#include "fts_anpr_svmocr.h"
#include "fts_ip_util.h"
#include <regex>
#include <ctime>

FTS_ANPR_SvmOcr::FTS_ANPR_SvmOcr(string _digitModelFolderPath, string _letterModelsPath, string pairModelsPath, string _pairModelPrefix, int minCharToProcess, bool bDebugOCR)
{
	/*this->digitModelsPath = _digitModelFolderPath;
	this->letterModelsPath = _letterModelsPath;
	this->digitLetterModelsPath = _digitLetterModelsPath;*/
	//this->config = config;
	this->m_iPostProcessMinCharacters = minCharToProcess;
	//this->m_iOcrMinFontSize = ocrMinFontSize;
	this->m_bDebugOcr = bDebugOCR;
	if (!_pairModelPrefix.empty()) {
		this->pairModelPrefix = _pairModelPrefix;
	}
	
	DIR * pairModelDir;
	dirent* pdir;
	if ((pairModelDir = opendir(pairModelsPath.c_str())))
	{
		while (pdir = readdir(pairModelDir)) {
			if (pdir->d_type == DT_DIR) {
				std::string dirName(pdir->d_name, pdir->d_namlen);
				std::cmatch modelNameRes;
				std::string patStr = this->pairModelPrefix + "_([\\d\\w]+)_([\\d\\w]+)";
				//std::regex modelNamesPat("model_([\d\w]+)_([\d\w]+)");
				std::regex modelNamesPat(patStr);
				if (std::regex_search(dirName.c_str(), modelNameRes, modelNamesPat))
				{
					std::string first = modelNameRes[1];
					std::string second = modelNameRes[2];
					std::string key1 = first + "_" + second;
					//std::string key2 = second + "_" + first;
					pairModelPathMap[key1] = pairModelsPath + "\\" + dirName;
					//pairModelPathMap[key2] = pairModelsPath + "\\" + dirName;
				}
			}
		}
		
	}
	

#ifdef POST_PROCESS
	this->postProcessor = NULL;
#endif
	this->aspectRatio = 2.0f;
	m_bInitialized = true;
#ifdef USING_SVM_OCR
	if (!this->loadDigitModels(_digitModelFolderPath) || !this->loadLetterModels(_letterModelsPath))	//"models_letters"
	{
		std::cout << "Model error" << std::endl;
		m_bInitialized = false;
	}
#endif
}

bool FTS_ANPR_SvmOcr::isInitialized()
{
	return this->m_bInitialized;
}

#ifdef POST_PROCESS
void FTS_ANPR_SvmOcr::initPostProcess(string patternsFile, int minConfidence, int confidenceSkipLevel, int maxSubs, bool debugPP)
{	
	this->postProcessor = new FTS_ANPR_PostProcess(patternsFile, minConfidence, confidenceSkipLevel, maxSubs, debugPP);
}
#endif

FTS_ANPR_SvmOcr::~FTS_ANPR_SvmOcr()
{
#ifdef POST_PROCESS
	if(this->postProcessor)
	{
		delete postProcessor;
	}
#endif
}

void FTS_ANPR_SvmOcr::clean()
{
#ifdef POST_PROCESS
	if(this->postProcessor)
	{
		this->postProcessor->clear();
	}
#endif
}

#ifdef USING_SVM_OCR
bool FTS_ANPR_SvmOcr::loadDigitModels(string _digitModelsPath) {
	return this->digitClassifier.loadModelFormPath(_digitModelsPath);
}

bool FTS_ANPR_SvmOcr::loadLetterModels(string _letterModelsPath) {
	return this->letterClassifier.loadModelFormPath(_letterModelsPath);
}

bool FTS_ANPR_SvmOcr::loadAllCharModels(string _digitModelsPath, string _letterModelsPath, string _digitLetterModelsPath) {
	return this->digitClassifier.loadModelFormPath(_digitModelsPath) && this->letterClassifier.loadModelFormPath(_letterModelsPath) /*&& this->digitLetterClassifier.loadModelFormPath(_digitLetterModelsPath)*/;
}

bool FTS_ANPR_SvmOcr::checkCharType(character ch, CLASSIFIER_SET charType) {
	bool bCorrect = false;
	if ((isDigit(ch.strLabel) && charType == CLASSIFIER_SET::DIGIT_ONLY) || charType == CLASSIFIER_SET::DIGIT_LETTER)
	{
		bCorrect = true;
	}
	else if ((!isDigit(ch.strLabel) && charType == CLASSIFIER_SET::LETTER_ONLY) || charType == CLASSIFIER_SET::DIGIT_LETTER)
	{
		bCorrect = true;
	}
	return bCorrect;
}

bool FTS_ANPR_SvmOcr::findBestChar(const cv::Mat& charImg, const std::vector<character>& listChars, character& bestChar, CLASSIFIER_SET mode) {
	if (listChars.size() == 0)
	{
		return false;
	}

	int ind = 0;
	std::cout << "SIZE " << listChars.size() << std::endl;
	/*for (; ind < listChars.size(); ind++)
	{
		if (!checkCharType(listChars[ind], mode)) {
			break;
		}
	}*/
	while (ind < listChars.size() && !checkCharType(listChars[ind], mode) )
	{
		ind++;
	}
	if (ind < listChars.size())
	{
		bestChar = listChars[ind];
	}
	for (size_t i = ind + 1; i < listChars.size(); i++)
	{
		if (!checkCharType(listChars[i], mode)) {
			continue;
		}
		std::string primClass = bestChar.strLabel;
		std::string secClass = listChars[i].strLabel;
		std::string pairModelStr = primClass + "_" + secClass;

		if (this->pairModelPathMap[pairModelStr].empty())
		{
			primClass = listChars[i].strLabel;
			secClass = bestChar.strLabel;
			pairModelStr = primClass + "_" + secClass;
			if (this->pairModelPathMap[pairModelStr].empty())
			{
				if (bestChar.confidence < listChars[i].confidence)
				{
					bestChar = listChars[i];
					continue;
				}
			}
		}

		HogSvmClassifiers classifier;
		classifier.loadModelFormPath(this->pairModelPathMap[pairModelStr]);
		classifier.singleClassify(charImg, primClass, secClass, bestChar);
	}
}

bool FTS_ANPR_SvmOcr::findBestChar(const std::vector<float>& featureVector, const std::vector<character>& listChars, character& bestChar, CLASSIFIER_SET mode) {
	if (listChars.size() == 0)
	{
		return false;
	}

	int ind = 0;
	std::cout << "SIZE " << listChars.size() << std::endl;
	/*for (; ind < listChars.size(); ind++)
	{
	if (!checkCharType(listChars[ind], mode)) {
	break;
	}
	}*/
	while (ind < listChars.size() && !checkCharType(listChars[ind], mode))
	{
		ind++;
	}
	if (ind < listChars.size())
	{
		bestChar = listChars[ind];
	}
	for (size_t i = ind + 1; i < listChars.size(); i++)
	{
		if (!checkCharType(listChars[i], mode)) {
			continue;
		}
		std::string primClass = bestChar.strLabel;
		std::string secClass = listChars[i].strLabel;
		std::string pairModelStr = primClass + "_" + secClass;

		if (this->pairModelPathMap[pairModelStr].empty())
		{
			primClass = listChars[i].strLabel;
			secClass = bestChar.strLabel;
			pairModelStr = primClass + "_" + secClass;
			if (this->pairModelPathMap[pairModelStr].empty())
			{
				if (bestChar.confidence < listChars[i].confidence)
				{
					bestChar = listChars[i];
					
				} 
				continue;
			}
		}

		HogSvmClassifiers classifier;
		classifier.loadModelFormPath(this->pairModelPathMap[pairModelStr]);
		classifier.singleClassify(featureVector, primClass, secClass, bestChar);
	}
}



void FTS_ANPR_SvmOcr::findBestCharBasedConfidence(const std::vector<character>& listChars, character& bestChar) {
	if (listChars.size() == 0)
	{
		return;
	}
	bestChar = listChars[0];
	for (size_t i = 1; i < listChars.size(); i++)
	{
		if (bestChar.confidence < listChars[i].confidence)
		{
			bestChar = listChars[i];
		}
	}
}

void FTS_ANPR_SvmOcr::getStandardCharImgs(Mat src, const vector<Rect>& bbs, vector<Mat>& charImgs, int padding)
//void extendImage(const cv::Mat &src, Rect &rect, Mat &dist)
{
	for (size_t i = 0; i < bbs.size(); i++)
	{
		Rect rect = bbs[i];
		int w = rect.width;
		int h = rect.height;
		
		if (h > this->aspectRatio * w)
		{
			//stdMat = Mat::zeros(cv::Size(this->aspectRatio * w, h), CV_8UC1);
			rect.width = int(h / this->aspectRatio);
			rect.x = rect.x + w / 2 - rect.width / 2;
		}
		FTS_BASE_Util::extendRect(rect, padding);
		Mat stdMat = Mat::zeros(rect.size(), CV_8UC1);

		Rect rectSrc(0, 0, src.cols, src.rows);
		if (rectSrc.contains(cv::Point(rect.x, rect.y)) &&
			rectSrc.contains(cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1)))
		{
			Mat it = src(rect);
			it.copyTo(stdMat);
			charImgs.push_back(stdMat);
		}
	}
	
}
#endif
bool FTS_ANPR_SvmOcr::isDigit(const std::string& s)
{
	return !s.empty() && std::find_if(s.begin(),
		s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

void FTS_ANPR_SvmOcr::performOCR(const Mat& oSrcImg,
	const vector<Rect>& charRegions,
	//OcrResult& ocrResults )		//21.06
	FTS_ANPR_OBJECT& oAnprObject)	//27.06
{
#ifdef POST_PROCESS
	if (this->postProcessor) postProcessor->clear();
#endif

	oAnprObject.ocrResults.clear();
#ifdef POST_PROCESS
	oAnprObject.ocrResults.init(this->postProcessor->m_iPostProcessMinConfidence,			//min confidence level
		this->postProcessor->m_iPostProcessConfidenceSkipLevel		//confidence skip level
		);
#endif

	// Don't waste time on OCR processing if it is impossible to get sufficient characters
	if ((int)charRegions.size() < m_iPostProcessMinCharacters)
	{
		oAnprObject.oDebugLogs.warn("NUMBER OF CHARACTERS = %d, IGNORE\n", (int)charRegions.size());
		return;
	}

	if (!this->isInitialized())
	{
		cout << "SVM OCR engine is not initialized!!!" << endl;
		return;
	}

	//TODO: Hoang add code here
#ifdef USING_SVM_OCR

	std::vector<cv::Mat> stdCharImgs;
	this->getStandardCharImgs(oSrcImg, charRegions, stdCharImgs, 0);
	//std::vector<std::string> listedChars;
	//listedChars.push_back("L");
	//listedChars.push_back("X");
	//listedChars.push_back("S"); 
	//classify.setClassifierIdx(CLASSIFIER_SET::CUSTOM, listedChars);
	std::vector<character> lowerChars, upperChars;
	std::vector<std::vector<character> > upperPotentialChars;
	
	std::vector<character> listSortChars;
	//std::vector<int> upCharIdx;
	std::vector<std::vector<float> > listFeatureVector;
	// Get potential lower and upper chars
	//int upperInd = 0;
	clock_t start = clock();
	for (int i = 0; i < stdCharImgs.size(); i++)
	{
		character resChar;
		cv::HOGDescriptor hog;
		cv::Size winStride, padding;
		std::vector<float> featureVector;
		this->digitClassifier.getHogParameters(hog, winStride, padding);
		HogSvmClassifiers hsclassifier;
		hsclassifier.setHogParameters(hog, winStride, padding);
		hsclassifier.calculateFeaturesFromInput(stdCharImgs[i], featureVector);
		
		//classifier.calculateFeaturesFromInput(stdCharImgs[i], testFeatureVector);
		if (oAnprObject.oCharPosLine[i] == 1) { // Lower row
			//this->digitClassifier.multiClassify(stdCharImgs[i], listSortChars, false);
			this->digitClassifier.multiClassify(featureVector, listSortChars, false);
			if (listSortChars.size() > 0)
			{
				findBestChar(featureVector, listSortChars, resChar, CLASSIFIER_SET::DIGIT_ONLY);
				lowerChars.push_back(resChar);
			}
			else
			{
				character c;
				c.strLabel = "~";
				c.confidence = 1.0;
				lowerChars.push_back(c);
			}

		}
		else if (oAnprObject.oCharPosLine[i] == 0) { // Upper row
			/*this->digitClassifier.multiClassify(stdCharImgs[i], listSortChars, false);
			this->letterClassifier.multiClassify(stdCharImgs[i], listSortChars, true);*/
			this->digitClassifier.multiClassify(featureVector, listSortChars, false);
			this->letterClassifier.multiClassify(featureVector, listSortChars, true);
			if (listSortChars.size() > 0)
			{
				//findBestChar(stdCharImgs[i], listSortChars, resChar);
				upperPotentialChars.push_back(listSortChars);
				listFeatureVector.push_back(featureVector);
				//upCharIdx.push_back(upperInd);
				//upperInd++;
			}
			else
			{
				std::vector<character> vc;
				character c;
				c.strLabel = "~";
				c.confidence = 1.0;
				vc.push_back(c);
				upperPotentialChars.push_back(vc);
				listFeatureVector.push_back(featureVector);
			}
		}

		listSortChars.clear();
	}

	std::vector<CLASSIFIER_SET> modes;
	// Consider upper part
	if (upperPotentialChars.size() == 3 /* && checkPositionBalance()*/)
	{
		modes.push_back(CLASSIFIER_SET::DIGIT_ONLY);
		modes.push_back(CLASSIFIER_SET::DIGIT_ONLY);
		modes.push_back(CLASSIFIER_SET::LETTER_ONLY);
	}
	else if (upperPotentialChars.size() == 4)
	{
		modes.push_back(CLASSIFIER_SET::DIGIT_ONLY);
		modes.push_back(CLASSIFIER_SET::DIGIT_ONLY);
		modes.push_back(CLASSIFIER_SET::LETTER_ONLY);
		modes.push_back(CLASSIFIER_SET::DIGIT_LETTER);
	}
	else
	{
		for (size_t k = 0; k < upperPotentialChars.size(); k++)
		{
			modes.push_back(CLASSIFIER_SET::DIGIT_LETTER);
		}
	}

	for (size_t j = 0; j < upperPotentialChars.size(); j++)
	{
		character ch;
		findBestChar(listFeatureVector[j], upperPotentialChars[j], ch, modes[j]);
		upperChars.push_back(ch);
	}
	
	listSortChars.clear();
	for (size_t i = 0; i < upperChars.size(); i++)
	{
		listSortChars.push_back(upperChars[i]);
	}
	for (size_t i = 0; i < lowerChars.size(); i++)
	{
		listSortChars.push_back(lowerChars[i]);
	}
	clock_t end = clock();
	std::cout << "OCR running time = " << double(end - start) / CLOCKS_PER_SEC << std::endl;
	for (size_t i = 0; i < listSortChars.size(); i++)
	{
		character resChar = listSortChars[i];
		if (this->m_bDebugOcr)
		{
			std::cout << "charpos " << i + 1 << ": " << resChar.strLabel << " confidence: " << resChar.confidence << std::endl;
			oAnprObject.oDebugLogs.info("char %d: %s confidence: %f\n", i + 1, resChar.strLabel.c_str(), resChar.confidence);
		}
		if (resChar.strLabel.size() > 0)
		{
			oAnprObject.ocrResults.addLetter(resChar.strLabel[0], i, resChar.confidence);
		}
	}
	if(this->m_bDebugOcr)
	{
		/*Mat oInputColor;
		cv::cvtColor(oSrcImg, oInputColor, CV_GRAY2BGR);
		for(int i = 0; i < charRegions.size(); i++)
		{
			cv::rectangle(oInputColor, charRegions[i], Scalar(0,255,0));
		}
		imshow("imgTest", oInputColor);*/		
		//FTS_GUI_DisplayImage::ShowGroupScaleBy2("SVM OCR Source", 1.0, stdCharImgs, 1);
		//cv::waitKey(0);
	}
#endif
}