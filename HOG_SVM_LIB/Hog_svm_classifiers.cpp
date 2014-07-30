#ifdef DETECT_MEM_LEAK
#include <vld.h>
#endif
#include "Hog_svm_classifiers.h"
#include "svm_util.h"
#include <assert.h>

#ifdef USE_SVM_CLASSIFY
class HogSvmClassifiersInvoker : public cv::ParallelLoopBody{
public:
	HogSvmClassifiersInvoker(HogSvmClassifiers& _cc, cv::Mat& _img, std::vector<DetectionObject>& _objects, cv::Size _sz1, double _scale, int  _yStep, int _stripSize, cv::Mutex* _mtx){
		this->classifer = &_cc;
		this->objects = &_objects;
		//this->scores = &_scores;
		//this->classLabels = &_classLabels;
		this->mtx = _mtx;
		yStep = _yStep;
		processingRectSize = _sz1;
		img = _img;
		winSize = classifer->getHogWinSize();
		scale = _scale;
		realWinSize.width = winSize.width * scale;
		realWinSize.height = winSize.height * scale;
		stripSize = _stripSize;
	}
	void operator()(const cv::Range& range) const
	{
		//std::cout << "hog size x " << range.start << " " << range.end << std::endl;
		int y1 = range.start * stripSize;
		int y2 = std::min(range.end * stripSize, processingRectSize.height);

		for (int y = y1; y < y2; y += yStep)
		{
			for (int x = 0; x < processingRectSize.width; x += yStep)
			{
				cv::Rect rectBuffer(x, y, winSize.width, winSize.height);
				cv::Rect realWinSizeBuffer(x * scale, y * scale, realWinSize.width, realWinSize.height);
				std::vector<character> listSortChars;
				cv::Mat imgBuffer = img(rectBuffer);
				classifer->multiClassify(imgBuffer, listSortChars);
				if (listSortChars.size() >0){
					int i = 0;
					//for (size_t i = 0; i < listSortChars.size(); i++)
					{
						if (listSortChars[i].strLabel.at(0) != '~'){
							mtx->lock();
							DetectionObject tmp;
							tmp.boundingBox = realWinSizeBuffer;
							tmp.score = listSortChars[i].confidence;
							tmp.label = listSortChars[i].strLabel;
							objects->push_back(tmp);
						
							mtx->unlock();
							x += yStep;
						}
					}
				}

			}
		}
	}
	cv::Mat img;
	cv::Size winSize;
	cv::Size realWinSize;
	HogSvmClassifiers* classifer;
	cv::Mutex* mtx;
	std::vector<DetectionObject>* objects;
	//std::vector<double>* scores;
	//std::vector<std::string>* classLabels;
	cv::Size processingRectSize;
	int yStep;
	double scale;
	int stripSize;
};

HogSvmClassifiers::HogSvmClassifiers(const HogSvmClassifiers& hsc, cv::Size _winStride, cv::Size _padding) {
	this->hog.winSize = hsc.hog.winSize;
	this->hog.blockSize = hsc.hog.blockSize;
	this->hog.blockStride = hsc.hog.blockStride;
	this->hog.cellSize = hsc.hog.cellSize;
	this->winStride = _winStride;
	this->padding = _padding;
}

HogSvmClassifiers::~HogSvmClassifiers(){
	for (size_t i = 0; i < listModelSvms.size(); i++){
		//listModelSvms[i]->~SvmLightClassify();
		delete listModelSvms[i];
	}
	listModelSvms.clear();
}
bool HogSvmClassifiers::loadModelFormPath(const std::string& strPath){
	if (!isFileExist(strPath + "\\config.xml"))
		return false;
	cv::FileStorage fs(strPath + "\\config.xml", cv::FileStorage::READ);
	int numOfClass = fs["Num_Of_Class"];
	if (numOfClass <= 0){
		return false;
	}
	for (int i = 0; i < numOfClass; i++){
		std::string strBuff = "class_" + std::to_string(i);
		std::string file_name = fs[strBuff]["file_name"];
		if (!isFileExist(strPath + "\\" + file_name))
			return false;
	}
	clearListOfString(this->listStrLabel);
	for (int i = 0; i < numOfClass; i++){
		std::string strBuff = "class_" + std::to_string(i);
		std::string file_name = fs[strBuff]["file_name"];
		std::string cBuff = fs[strBuff]["label"];
		this->listStrLabel.push_back(cBuff);
		// Create map from class labels to model indices
		this->classIdxMap[cBuff] = i;
		// Set default model indices which includes all
		this->listModelIdx.push_back(i);
		loadModelFromFile(strPath + "\\" + file_name);
	}
	int w, h;
	w = fs["Hog_WinSize"]["width"];
	h = fs["Hog_WinSize"]["height"];
	if (w > 0 && h > 0){
		this->hog.winSize = cv::Size(w, h);
	}

	w = fs["Hog_BlockSize"]["width"];
	h = fs["Hog_BlockSize"]["height"];
	if (w > 0 && h > 0){
		this->hog.blockSize = cv::Size(w, h);
	}

	w = fs["Hog_blockStride"]["width"];
	h = fs["Hog_blockStride"]["height"];
	if (w > 0 && h > 0){
		this->hog.blockStride = cv::Size(w, h);
	}

	w = fs["Hog_cellSize"]["width"];
	h = fs["Hog_cellSize"]["height"];
	if (w > 0 && h > 0){
		this->hog.cellSize = cv::Size(w, h);
	}

	w = fs["WinStride"]["width"];
	h = fs["WinStride"]["height"];
	if (w >= 0 && h >= 0){
		this->winStride = cv::Size(w, h);
	}
	
	w = fs["padding"]["width"];
	h = fs["padding"]["height"];
	if (w >= 0 && h >= 0){
		this->padding = cv::Size(w, h);
	}
	if (listModelSvms.size() == 1 && listModelSvms[0]->getSVMKernelType() == LINEAR){
		std::vector<float> svmDecriptor;
		listModelSvms[0]->getSVMDecriptor(svmDecriptor);
		hog.setSVMDetector(svmDecriptor);
	}
}
void HogSvmClassifiers::setHogParameters(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride, cv::Size _cellSize, cv::Size _winStride, cv::Size _padding){
	this->hog.winSize = _winSize;
	this->hog.blockSize = _blockSize;
	this->hog.blockStride = _blockStride;
	this->hog.cellSize = _cellSize;
	this->winStride = _winStride;
	this->padding = _padding;
}

void HogSvmClassifiers::setHogParameters(const cv::HOGDescriptor& hog, cv::Size _winStride, cv::Size _padding){
	this->hog.winSize = hog.winSize;
	this->hog.blockSize = hog.blockSize;
	this->hog.blockStride = hog.blockStride;
	this->hog.cellSize = hog.cellSize;
	this->winStride = _winStride;
	this->padding = _padding;
}

void HogSvmClassifiers::getHogParameters(cv::HOGDescriptor& _hog, cv::Size& _winStride, cv::Size& _padding) {
	_hog.winSize = this->hog.winSize;
	_hog.blockSize = this->hog.blockSize;
	_hog.blockStride = this->hog.blockStride;
	_hog.cellSize = this->hog.cellSize;
	_winStride = this->winStride;
	_padding = this->padding;
}

void HogSvmClassifiers::loadModelFromFile(const std::string& _modelFileName){
	SvmLightClassify* svm_tmp = new SvmLightClassify;
	svm_tmp->loadModelFromFile(_modelFileName);
	this->listModelSvms.push_back(svm_tmp);
}
void HogSvmClassifiers::calculateFeaturesFromInput(const cv::Mat& imgTest, std::vector<float>& featureVector) {
	/** for imread flags from openCV documentation,
	* @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
	* @note If you get a compile-time error complaining about following line (esp. imread),
	* you either do not have a current openCV version (>2.0)
	* or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
	*/
	cv::Mat imageData;
	cv::resize(imgTest, imageData, this->hog.winSize);
	if (imageData.empty()) {
		featureVector.clear();
		printf("Error: HOG image  is empty, features calculation skipped!\n");
		return;
	}
	// Check for mismatching dimensions
	if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
		featureVector.clear();
		printf("Error: Image dimensions (%u x %u) do not match HOG window size (%u x %u)!\n",  imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
		return;
	}
	std::vector<cv::Point> locations;
	//this->hog.compute(imageData, featureVector, cv::Size(0, 0), cv::Size(0, 0), locations);
	this->hog.compute(imageData, featureVector, winStride, padding, locations);
	imageData.release(); // Release the image again after features are extracted
}
double HogSvmClassifiers::classify(int indexModel, const std::vector<float>& featureVectorSample){
	if (indexModel < 0 || indexModel >= listModelSvms.size())
		return 0;
	return this->listModelSvms[indexModel]->classify(featureVectorSample);
}

void HogSvmClassifiers::singleClassify(const cv::Mat& imgTest, const std::string& posClassName, const std::string& negClassName, character& resChar){
	std::vector<float> featureVector;
	calculateFeaturesFromInput(imgTest, featureVector);
	this->singleClassify(featureVector, posClassName, negClassName, resChar);
}

void HogSvmClassifiers::singleClassify(const std::vector<float>& featureVector, const std::string& posClassName, const std::string& negClassName, character& resChar){

	double dist = this->classify(0, featureVector);
	if (dist > 0){
		resChar.strLabel = posClassName;
		resChar.confidence = dist;
	}
	else
	{
		resChar.strLabel = negClassName;
		resChar.confidence = -dist;
	}
}


void HogSvmClassifiers::multiClassify(const cv::Mat& imgTest, std::vector<character>& listSortChars, bool bAppend){
	std::vector<float> featureVector;
	calculateFeaturesFromInput(imgTest, featureVector);
	this->multiClassify(featureVector, listSortChars, bAppend);
}

void HogSvmClassifiers::multiClassify(const std::vector<float>& featureVector, std::vector<character>& listSortChars, bool bAppend){
	
	std::vector<character> listChar;

	for (size_t i = 0; i < this->listModelSvms.size(); i++)
	{
		double dist = this->classify(i, featureVector);
		if (dist >= 0){
			character c;
			c.strLabel = this->listStrLabel[i];
			c.confidence = dist;
			listChar.push_back(c);
		}
	}
	if (!bAppend)
	{
		listSortChars.clear();
	}

	while (!listChar.empty()){

		size_t indexMax = 0;
		for (size_t i = 0; i < listChar.size(); i++)
		{
			if (listChar[indexMax].confidence < listChar[i].confidence)
				indexMax = i;
		}
		listSortChars.push_back(listChar[indexMax]);
		listChar.erase(listChar.begin() + indexMax);
	}
	/*for (std::vector<character>::iterator it = listSortChars.begin(); it != listSortChars.end(); it++){
	it->confidence /= listSortChars[0].confidence;
	std::cout << "char: " << it->strLabel << " confidence: " << it->confidence << std::endl;
	}*/
}


void HogSvmClassifiers::setClassifierIdx(CLASSIFIER_SET classifierSetType, const std::vector<std::string>& customClassLabels) {
	this->listModelIdx.clear();
	std::vector<std::string> _classLabels;
	if (classifierSetType == CLASSIFIER_SET::CUSTOM)
	{
		_classLabels = customClassLabels;
	}
	else if (classifierSetType == CLASSIFIER_SET::DIGIT_ONLY)
	{
		for (size_t i = 0; i < 10; i++)
		{
			_classLabels.push_back(std::to_string(i));
		}
		_classLabels.push_back("~");
	}
	else if (classifierSetType == CLASSIFIER_SET::LETTER_ONLY)
	{
		_classLabels.push_back("A");
		_classLabels.push_back("B");
		_classLabels.push_back("C");
		_classLabels.push_back("D");
		_classLabels.push_back("E");
		_classLabels.push_back("F");
		_classLabels.push_back("G");
		_classLabels.push_back("H");
		_classLabels.push_back("K");
		_classLabels.push_back("L");
		_classLabels.push_back("M");
		_classLabels.push_back("N");
		_classLabels.push_back("P");
		_classLabels.push_back("R");
		_classLabels.push_back("S");
		_classLabels.push_back("T");
		_classLabels.push_back("U");
		_classLabels.push_back("V");
		_classLabels.push_back("X");
		_classLabels.push_back("Y");
		_classLabels.push_back("Z");
		_classLabels.push_back("~");
	}
	else if (classifierSetType == CLASSIFIER_SET::DIGIT_LETTER)
	{
		for (size_t i = 0; i < 10; i++)
		{
			_classLabels.push_back(std::to_string(i));
		}
		_classLabels.push_back("A");
		_classLabels.push_back("B");
		_classLabels.push_back("C");
		_classLabels.push_back("D");
		_classLabels.push_back("E");
		_classLabels.push_back("F");
		_classLabels.push_back("G");
		_classLabels.push_back("H");
		_classLabels.push_back("K");
		_classLabels.push_back("L");
		_classLabels.push_back("M");
		_classLabels.push_back("N");
		_classLabels.push_back("P");
		_classLabels.push_back("R");
		_classLabels.push_back("S");
		_classLabels.push_back("T");
		_classLabels.push_back("U");
		_classLabels.push_back("V");
		_classLabels.push_back("X");
		_classLabels.push_back("Y");
		_classLabels.push_back("Z");
		_classLabels.push_back("~");
	}

	for (size_t i = 0; i < _classLabels.size(); i++)
	{
		if (this->classIdxMap[_classLabels[i]]) {
			this->listModelIdx.push_back(this->classIdxMap[_classLabels[i]]);
		}
		/*for (size_t j = 0; j < this->listStrLabel.size(); j++)
		{
			if (std::strcmp(_classLabels[i].c_str(), this->listStrLabel[j].c_str()) == 0) {
				this->listModelIdx.push_back(j);
				break;
			}
		}*/
	}
}

void HogSvmClassifiers::customMultiClassify(const cv::Mat& imgTest, std::vector<character>& listSortChars){
	
	std::vector<float> featureVector;
	calculateFeaturesFromInput(imgTest, featureVector);
	std::vector<character> listChar;

	for (size_t i = 0; i < this->listModelIdx.size(); i++)
	{
		int k = this->listModelIdx[i];
		double dist = this->classify(k, featureVector);
		if (dist > 0){
			character c;
			c.strLabel = this->listStrLabel[k];
			c.confidence = dist;
			listChar.push_back(c);
		}
	}

	listSortChars.clear();
	while (!listChar.empty()){

		size_t indexMax = 0;
		for (size_t i = 0; i < listChar.size(); i++)
		{
			if (listChar[indexMax].confidence < listChar[i].confidence)
				indexMax = i;
		}
		listSortChars.push_back(listChar[indexMax]);
		listChar.erase(listChar.begin() + indexMax);
	}
}

void HogSvmClassifiers::detectSingleScale(cv::Mat& image, double scale, std::vector<DetectionObject>& objects){
	int step = 5;
	cv::Size processingRectSize;
	
	cv::Size winSize = hog.winSize;
	cv::Mat imageReSize;
	cv::Size sz;
	sz.width = image.size().width * scale;
	sz.height= image.size().height * scale;
	cv::resize(image, imageReSize, sz);
	processingRectSize.width = imageReSize.cols - hog.winSize.width;
	processingRectSize.height = imageReSize.rows - hog.winSize.height;
	objects.clear();
	std::vector<double> scores;
	cv::Mutex mtx;
	int stripSize = processingRectSize.height / step;
	cv::parallel_for_(cv::Range(0, stripSize),
		HogSvmClassifiersInvoker(*this, imageReSize, objects, processingRectSize, scale, step, step, &mtx));
	int minNeighbors = 1;
	const double GROUP_EPS = 0.5;
	//cv::groupRectangles(objects, minNeighbors, GROUP_EPS);
	groupRectangles(objects, minNeighbors, GROUP_EPS, GROUP_MAX);
	/*for (int y = 0; y < processingRectSize.height; y+= step)
	{
		for (int x = 0; x < processingRectSize.width; x += step)
		{
			cv::Rect rectBuffer(x, y, winSize.width, winSize.height);
			std::vector<character> listSortChars;
			cv::Mat imgBuffer = image(rectBuffer);
			multiclassClassify(imgBuffer, listSortChars);
			if (listSortChars.size() >0){
				for (size_t i = 0; i < listSortChars.size(); i++)
				{
					if (listSortChars[i].strLabel.at(0) != '~'){
						objects.push_back(rectBuffer);
						x += step;
					}
				}
			}
			
		}
	}*/
}
void HogSvmClassifiers::detectSingleScale(cv::Mat& image, int stripCount, cv::Size processingRectSize,
	int stripSize, int yStep, double factor, std::vector<DetectionObject>& candidates){
	
	cv::Mutex mtx;
	
	cv::parallel_for_(cv::Range(0, stripCount),
		HogSvmClassifiersInvoker(*this, image, candidates, processingRectSize, factor, yStep, stripSize, &mtx));
}
void HogSvmClassifiers::detectMultiScale(const cv::Mat& image, std::vector<DetectionObject>& objects,
	double scaleFactor, int minNeighbors, double hitThreshold,
	cv::Size minObjectSize, cv::Size maxObjectSize){
	
	std::vector<cv::Rect> found;
	/*cv::Size padding(cv::Size(0, 0));
	cv::Size winStride(cv::Size(8, 8));*/
	//double hitThreshold = -0.0; // tolerance
	objects.clear();
	//cv::resize(imageData, imageData, sz );
	hog.detectMultiScale(image, found, minObjectSize, maxObjectSize, hitThreshold, winStride, padding, scaleFactor, minNeighbors);
	for (size_t i = 0; i < found.size(); i++)
	{
		DetectionObject buff;
		buff.boundingBox = found[i];
		buff.label = listStrLabel[0];
		buff.score = 1.0;
		objects.push_back(buff);
	}
}
void HogSvmClassifiers::detectMultiScaleBasedBoost(const cv::Mat& image, std::vector<DetectionObject>& objects,
	double scaleFactor, int minNeighbors,
	int flags, cv::Size minObjectSize, cv::Size maxObjectSize)
{
	const double GROUP_EPS = 0.2;
	assert(scaleFactor > 1 && image.depth() == CV_8U );
	if (listModelSvms.size() == 0)
		return;
	objects.clear();
	//mark generator

	if (maxObjectSize.height == 0 || maxObjectSize.width == 0)
		maxObjectSize = image.size();
	cv::Mat grayImage = image;
	if (grayImage.channels() > 1)
	{
		cv::Mat temp;
		cvtColor(grayImage, temp, CV_BGR2GRAY);
		grayImage = temp;
	}
	cv::Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
	std::vector<DetectionObject> candidates;
	for (double factor = 1;; factor *= scaleFactor)
	{
		cv::Size originalWindowSize = this->getHogWinSize();

		cv::Size windowSize(cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor));
		cv::Size scaledImageSize(cvRound(grayImage.cols / factor), cvRound(grayImage.rows / factor));
		cv::Size processingRectSize(scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height);

		if (processingRectSize.width <= 0 || processingRectSize.height <= 0)
			break;
		if (windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height)
			break;
		if (windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height)
			continue;

		cv::Mat scaledImage(scaledImageSize, CV_8U, imageBuffer.data);
		resize(grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR);

		int yStep;
		
		yStep = 4;
		

		int stripCount, stripSize;

		const int PTS_PER_THREAD = 1000;
		stripCount = ((processingRectSize.width / yStep)*(processingRectSize.height + yStep - 1) / yStep + PTS_PER_THREAD / 2) / PTS_PER_THREAD;
		stripCount = std::min(std::max(stripCount, 1), 100);
		stripSize = (((processingRectSize.height + stripCount - 1) / stripCount + yStep - 1) / yStep)*yStep;

		//detect single scale in here
		detectSingleScale(scaledImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates);
	}
	objects.resize(candidates.size());
	std::copy(candidates.begin(), candidates.end(), objects.begin());
	groupRectangles(objects, minNeighbors, GROUP_EPS, GROUP_AVG);
}
#else
void HogSvmClassifiers::str2Size(const std::string& str, cv::Size& pt){
	std::string s = str;
	std::string delimiter = " ";

	size_t pos = 0;
	std::string token;
	pt.width = 0;
	pt.height = 0;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		pt.width = atoi(token.c_str());
		s.erase(0, pos + delimiter.length());
	}
	if (s.length() > 0)
		pt.height = atoi(s.c_str());
}
void HogSvmClassifiers::loadVectorDescriptorsFromFile(const std::string &file_name){
	cv::FileStorage fs(file_name, cv::FileStorage::READ);
	char buffer[100];
	char bufferVector[50];
	int numOfClass = (int)fs["Num_Of_Class"];
	std::string strBuffer;
	cv::Size winSize, blockSize, blockStride, cellSize;
	strBuffer = fs["Hog_WinSize"];
	str2Size(strBuffer, winSize);
	strBuffer = fs["Hog_BlockSize"];
	str2Size(strBuffer, blockSize);
	strBuffer = fs["Hog_blockStride"];
	str2Size(strBuffer, blockStride);
	strBuffer = fs["Hog_cellSize"];
	str2Size(strBuffer, cellSize);

	
	std::vector<float> descriptorVector;
	for (int i = 0; i < numOfClass; ++i){
		cv::HOGDescriptor hog;
		hog.winSize = winSize;
		hog.blockSize = blockSize;
		hog.blockStride = blockStride;
		hog.cellSize = cellSize;

		sprintf_s(buffer, "Class_%d", i);
		int sizeOfVector = fs[buffer]["Size"];
		for (int j = 0; j < sizeOfVector; ++j){
			sprintf_s(bufferVector, "v_%d", j);
			float fTmp = fs[buffer][bufferVector];
			descriptorVector.push_back(fTmp);
		}
		hog.setSVMDetector(descriptorVector);
		listHog.push_back(hog);
		descriptorVector.clear();
	}
}

void HogSvmClassifiers::detectBinary(const cv::Mat& imageData, size_t index, double& weight) {
	if (index < 0 || index >= this->listHog.size())
		return;

	/*cv::Mat imgBuffer;
	cv::resize(imageData, imgBuffer, this->listHog[i].winSize);
	
	std::vector<cv::Point> found_locations;
	std::vector<cv::Point> locations;
	std::vector <double> confidences;
	locations.push_back(cv::Point(0, 0));
	listHog[i].detectROI(imgBuffer, locations, found_locations, confidences, 0.0, listHog[i].blockSize, cv::Size(0, 0));
	if (confidences.size() > 0)
	{
		printf("detect, confidence: %f\n", confidences[0]);
	}*/
	std::vector<cv::Rect> found;
	std::vector<cv::Point> foundLocations, searchLocations;
	std::vector<double> weights;
	int groupThreshold = 2;
	cv::Size padding(cv::Size(3, 3));
	cv::Size winStride(cv::Size(1, 1));
	double hitThreshold = -0.0; // tolerance
	cv::Mat resizedImg;
	cv::resize(imageData, resizedImg, this->listHog[index].winSize);

	this->listHog[index].detect(resizedImg, foundLocations, weights, hitThreshold, winStride, padding, searchLocations);

	if (foundLocations.size() > 0){
		weight = 0.0;
		for (size_t i = 0; i < weights.size(); i++){
			weight += weights[i];
		}
	}
	else{
		weight = 0.0;
	}

}

void HogSvmClassifiers::classifiers(const cv::Mat& img, std::vector<character>& listSortChars){
	cv::Mat imgBuffer;
	cv::resize(img, imgBuffer, listHog[0].winSize);

	std::vector<cv::Rect> found;
	std::vector<cv::Point> foundLocations, searchLocations;
	std::vector<double> weights;
	int groupThreshold = 2;
	cv::Size padding(cv::Size(2, 2));
	cv::Size winStride(cv::Size(1, 1));
	double hitThreshold = -0.0; // tolerance
	double weight = 0.0;
	
	double max_confidence = 0.0;
	std::vector<character> listChar;
	for (size_t i = 0; i < listHog.size(); ++i){
		this->listHog[i].detect(imgBuffer, foundLocations, weights, hitThreshold, winStride, padding, searchLocations);
		if (foundLocations.size() > 0){
			weight = 0.0;
			for (size_t i = 0; i < weights.size(); i++){
				weight += weights[i];
			}
		}
		else{
			weight = 0.0;
		}
		character c;
		c.number = char(i);
		c.confidence = weight;
		listChar.push_back(c);
		foundLocations.clear();
		weights.clear();
		
	}
	listSortChars.clear();
	while (!listChar.empty()){

		size_t indexMax = 0;
		for (size_t i = 1; i < listChar.size(); i++)
		{
			if (listChar[indexMax].confidence < listChar[i].confidence)
				indexMax = i;
		}
		listSortChars.push_back(listChar[indexMax]);
		listChar.erase(listChar.begin() + indexMax);
	}
	
}
#endif //USE_SVM_CLASSIFY