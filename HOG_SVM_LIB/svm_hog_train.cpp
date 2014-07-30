#ifdef DETECT_MEM_LEAK
#include <vld.h>
#endif

#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "svm_hog_train.h"
#include "dirent.h"
#include "svm_util.h"
#include "cvplot.h"
#include "direct.h"
//#include "svmlight.h"
static std::string KERNER_TYPE[] = { "LINEAR", "POLY", "RBF", "SIGMOD" };
SvmHogTrain::SvmHogTrain() : numOfSeparation(10) {}

SvmHogTrain::~SvmHogTrain(){
	clearListOfString(listDir);
	clearListOfString(listFileToTrain);
	listAcc.clear();
	for (size_t i = 0; i < listDescriptorVector.size(); i++){
		listDescriptorVector[i].clear();
	}
	listDescriptorVector.clear();
	for (size_t i = 0; i < listOfFeatureOfClass.size(); i++)
	{
		for (size_t j = 0; j < listOfFeatureOfClass[i].size(); j++)
		{
			listOfFeatureOfClass[i][j].featureVector.clear();
			listOfFeatureOfClass[i][j].file_name.clear();
		}
		listOfFeatureOfClass[i].clear();
	}
	listOfFeatureOfClass.clear();
	this->padding = cv::Size(0, 0);
	this->winStride = cv::Size(0, 0);
}
void SvmHogTrain::setHogParameters(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride, cv::Size _cellSize, cv::Size _padding, cv::Size _winStride){
	this->hog.winSize		= _winSize;
	this->hog.blockSize		= _blockSize;
	this->hog.blockStride	= _blockStride;
	this->hog.cellSize		= _cellSize;
	this->padding			= _padding;
	this->winStride = _winStride;
}

void SvmHogTrain::setSvmParameters(std::string _alpha_file, long _type, double _svmC, long _kernel_type, 
	long _remove_inconsistent, long _verbosity, double  _rbf_gamma){
	SVMlight::getInstance()->setParameters(_alpha_file, _type, _svmC, _kernel_type, 
		_remove_inconsistent, _verbosity, _rbf_gamma);
}
void SvmHogTrain::setSvmParameters(long _kernel_type, double _svmC, double  _rbf_gamma){
	SVMlight::getInstance()->setParameters(_kernel_type, _svmC, _rbf_gamma);
}
void SvmHogTrain::setListLabels(std::vector<std::string>& _listLabels){
	this->listLabels.clear();
	this->listLabels = _listLabels;
}
void SvmHogTrain::clearListDir(){
	this->listDir.clear();
}
void SvmHogTrain::addDirToList(const std::string& dir){
	this->listDir.push_back(dir);
}
void SvmHogTrain::addListFileToTrain(const std::string& file_name){
	this->listFileToTrain.push_back(file_name);
}

/**
* This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
* @param imageFilename file path of the image file to read and calculate feature vector from
* @param descriptorVector the returned calculated feature vector<float> ,
*      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
* @param hog HOGDescriptor containin HOG settings
*/
void SvmHogTrain::calculateFeaturesFromInput(const std::string& imageFilename, std::vector<float>& featureVector) {
	/** for imread flags from openCV documentation,
	* @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
	* @note If you get a compile-time error complaining about following line (esp. imread),
	* you either do not have a current openCV version (>2.0)
	* or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
	*/
	cv::Mat imageData = cv::imread(imageFilename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(imageData, imageData, hog.winSize);
	if (imageData.empty()) {
		featureVector.clear();
		printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
		return;
	}
	// Check for mismatching dimensions
	if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
		featureVector.clear();
		printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
		return;
	}
	std::vector<cv::Point> locations;

	//this->hog.compute(imageData, featureVector, cv::Size(1, 1), cv::Size(0, 0), locations);
	this->hog.compute(imageData, featureVector, this->winStride, this->padding, locations);

	imageData.release(); // Release the image again after features are extracted
}

/**
* Saves the given descriptor vector to a file
* @param descriptorVector the descriptor vector to save
* @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
* @param fileName
* @TODO Use _vectorIndices to write correct indices
*/
void SvmHogTrain::saveDescriptorVectorToFile(std::vector<float>& descriptorVector, std::vector<unsigned int>& _vectorIndices, std::string fileName) {
	printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
	std::string separator = " "; // Use blank as default separator between single features
	std::fstream File;
	double percent;
	File.open(fileName.c_str(), std::ios::out);
	if (File.good() && File.is_open()) {
		printf("Saving descriptor vector features:\t");
		storeCursor();
		for (size_t feature = 0; feature < descriptorVector.size(); ++feature) {
			if ((feature % 10 == 0) || (feature == (descriptorVector.size() - 1))) {
				percent = ((1 + feature) * 100 / descriptorVector.size());
				printf("%4u (%3.0f%%)", feature, percent);
				fflush(stdout);
				resetCursor();
			}
			File << descriptorVector.at(feature) << separator;
		}
		printf("\n");
		File << std::endl;
		File.flush();
		File.close();
	}
}
bool SvmHogTrain::extractFeatureToTrain()
{
	
	// Get the files to train from somewhere
	std::vector<std::string> positiveTrainingImages;
	std::vector<std::string> negativeTrainingImages;
	std::vector<std::string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");
	// </editor-fold>
	std::string feature_fileOuput;
	for (size_t i = 0; i < listDir.size(); ++i)
	{
		// <editor-fold defaultstate="collapsed" desc="Read image files">
		
		getFilesInDirectory(listDir[i], positiveTrainingImages, validExtensions);
		for (size_t j = 0; j < 10; j++)
		{
			if (i != j)
			{
				getFilesInDirectory(listDir[j], negativeTrainingImages, validExtensions);
			}
		}
		/// Retrieve the descriptor vectors from the samples
		unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
		// </editor-fold>

		// <editor-fold defaultstate="collapsed" desc="Calculate HOG features and save to file">
		// Make sure there are actually samples to train
		if (overallSamples == 0) {
			printf("No training sample files found, nothing to do!\n");
			return EXIT_SUCCESS;
		}

		/// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
		setlocale(LC_ALL, "C"); // Do not use the system locale
		setlocale(LC_NUMERIC, "C");
		setlocale(LC_ALL, "POSIX");


		float percent;
		/**
		* Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
		* @NOTE: If you split these steps into separate steps:
		* 1. calculating features into memory (e.g. into a cv::Mat or vector< vector<float> >),
		* 2. saving features to file / directly inject from memory to machine learning algorithm,
		* the program may consume a considerable amount of main memory
		*/
		char c = i + 48;
		feature_fileOuput = "";
		feature_fileOuput = feature_fileOuput + "svm_input\\" + c + "_hog_svm_input.txt";
		printf("Reading files, generating HOG features and save them to file '%s':\n", feature_fileOuput.c_str());
		std::fstream File;
		File.open(feature_fileOuput.c_str(), std::ios::out);
		if (File.good() && File.is_open()) {
			// Remove following line for libsvm which does not support comments
			// File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl;
			// Iterate over sample images
			for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
				//storeCursor();
				std::vector<float> featureVector;
				// Get positive or negative sample image file path
				const std::string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
				// Output progress
				if ((currentFile + 1) % 10 == 0 || (currentFile + 1) == overallSamples) {
					percent = ((currentFile + 1) * 100 / overallSamples);
					printf("%5lu (%3.0f%%):\tFile '%s' \n", (currentFile + 1), percent, currentImageFile.c_str());
					fflush(stdout);
					//resetCursor();
				}
				// Calculate feature vector from current image file
				calculateFeaturesFromInput(currentImageFile, featureVector);
				if (!featureVector.empty()) {
					/* Put positive or negative sample class to file,
					* true=positive, false=negative,
					* and convert positive class to +1 and negative class to -1 for SVMlight
					*/
					File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
					// Save feature vector components
					for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
						File << " " << (feature + 1) << ":" << featureVector.at(feature);
					}
					File << std::endl;
				}
			}
			printf("\n");
			File.flush();
			File.close();
		}
		else {
			printf("Error opening file '%s'!\n", feature_fileOuput.c_str());
			return EXIT_FAILURE;
		}
		positiveTrainingImages.clear();
		negativeTrainingImages.clear();
	}
	// </editor-fold>
	return EXIT_SUCCESS;
}

void SvmHogTrain::trainBinary(const std::string &sInHOHFeatureFile) {
	
	std::vector<unsigned int> descriptorVectorIndices;
	SVMlight::getInstance()->read_problem(const_cast<char*> (sInHOHFeatureFile.c_str()));
	SVMlight::getInstance()->train();
	//SVMlight::getInstance()->saveModelToFile(output);
	// Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
	
	//SVMlight::getInstance()->release();//duong.tb release after get model & test
}

void SvmHogTrain::trainBinaryBaseMap(const std::string &sInHOHFeatureFile, const int *map, long lengthMap, double classLabel) {

	std::vector<unsigned int> descriptorVectorIndices;
	SVMlight::getInstance()->read_problem(const_cast<char*> (sInHOHFeatureFile.c_str()), map, lengthMap, classLabel);
	SVMlight::getInstance()->train();
	//SVMlight::getInstance()->saveModelToFile("model_0_vs_all_1.txt", "svmlight");
	// Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
	//SVMlight::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
	//descriptorVectorIndices.clear();
	//SVMlight::getInstance()->release();

}


void SvmHogTrain::trainMultiClass(const std::string &file_input, const std::vector<long> &kernel_type, const std::vector<double> & vSvmC, const std::vector<double> &vGamma) {
	size_t mapLength = 0;
	char output[100];
	int numClass = int(kernel_type.size());
	for (size_t i = 0; i < this->listOfFeatureOfClass.size(); i++)
	{
		mapLength += this->listOfFeatureOfClass[i].size();
	}

	int *map = new int[mapLength];
	memset(map, 1, mapLength * sizeof(int));
	for (size_t i = 0; i < numClass; i++){
		std::cout << "Training class " << i << std::endl;
		//for (size_t i = 0; i < 1; i++){
		//SVMlight::getInstance()->setParameters(kernel_type[i], vSvmC[i], vGamma[i]);
		setSvmParameters(kernel_type[i], vSvmC[i], vGamma[i]);
		//std::vector<float> descriptorVector;
		trainBinaryBaseMap(file_input, map, mapLength, double(i));
		sprintf(output, "model\\model_%d_vsAll.dat", i);
		SVMlight::getInstance()->saveModelToFile(output);
		SVMlight::getInstance()->release();
	}
	delete[] map;
}



void SvmHogTrain::trainAllData(const std::string& file_input, const std::string& strPathOutput, const std::vector<std::string>& _listLabel, std::vector<long> &kernel_type, const std::vector<double> & vSvmC, const std::vector<double> &vGamma)
{
	this->listLabels.clear();
	this->listLabels = _listLabel;
	if (this->listOfFeatureOfClass.size() != this->listLabels.size()){
		std::cout << "Num Of class must be equal size of listLabels" << std::endl;
		return;
	}
	//char buffer[100];
	std::string output;
	std::vector<float> descriptorVector;
	/*std::vector<double> listSVM_C;
	listSVM_C.push_back(0.07);
	listSVM_C.push_back(0.05);
	listSVM_C.push_back(0.13);
	listSVM_C.push_back(0.5);
	listSVM_C.push_back(0.07);
	listSVM_C.push_back(0.03);
	listSVM_C.push_back(0.05);
	listSVM_C.push_back(0.05);
	listSVM_C.push_back(0.09);
	listSVM_C.push_back(0.11);*/
	//double  xa_error, xa_recall, xa_precision;
	size_t mapLength = 0;
	int numClass;
	if (this->listOfFeatureOfClass.size() > 2)
	{
		numClass = int(this->listOfFeatureOfClass.size());
	}
	else if (this->listOfFeatureOfClass.size() == 2)
	{
		numClass = 1;
	}
	else
	{
		std::cout << "Do NOT suppose 1-class SVM yet!" << std::endl;
		return;
	}
	for (size_t i = 0; i < this->listOfFeatureOfClass.size(); i++)
	{
		mapLength += this->listOfFeatureOfClass[i].size();
	}
	int *map = new int[mapLength];
	memset(map, 1, mapLength * sizeof(int));

	DIR * outDir;
	if ((outDir = opendir(strPathOutput.c_str())) == NULL)
	{
		_mkdir(strPathOutput.c_str());
	}
	else
	{
		closedir(outDir);
	}

	cv::FileStorage fs(strPathOutput + "\\config.xml", cv::FileStorage::WRITE);
	fs << "Num_Of_Class" << numClass;
	
	fs << "Hog_WinSize" << "{" << "width" << this->hog.winSize.width
		<< "height" << this->hog.winSize.height << "}";
	fs << "Hog_BlockSize" << "{" << "width" << this->hog.blockSize.width
		<< "height" << this->hog.blockSize.height << "}";

	fs << "Hog_blockStride" << "{" << "width" << this->hog.blockStride.width
		<< "height" << this->hog.blockStride.height << "}";
	fs << "Hog_cellSize" << "{" << "width" << this->hog.cellSize.width
		<< "height" << this->hog.cellSize.height << "}";
	fs << "WinStride" << "{" << "width" << this->winStride.width
		<< "height" << this->winStride.height << "}";
	fs << "padding" << "{" << "width" << this->padding.width
		<< "height" << this->padding.height << "}";
	for (size_t i = numClass-1; i < numClass; ++i)
	{
		//printf("Calling SVMlight\n");
		std::cout << "Training class " << _listLabel[i] << "..." << std::endl;
		setSvmParameters(kernel_type[i], vSvmC[i], vGamma[i]);
		trainBinaryBaseMap(file_input, map, mapLength, double(i));
		std::string strBuff = "class_" + std::to_string(i);
		
		if (this->listOfFeatureOfClass.size() == 2)
		{
			std::string label = this->listLabels[0] + "_" + this->listLabels[1];
			output = strPathOutput + "\\model_" + this->listLabels[0] + "_" + this->listLabels[1] + ".dat";
			fs << strBuff << "{" << "file_name" << "model_" + label + ".dat"
				<< "label" << label << "}";
		}
		else
		{
			output = strPathOutput + "\\model_" + this->listLabels[i] + "_vsAll.dat";
			fs << strBuff << "{" << "file_name" << "model_" + this->listLabels[i] + "_vsAll.dat"
				<< "label" << this->listLabels[i] << "}";
		}
		
		SVMlight::getInstance()->saveModelToFile(output);
		SVMlight::getInstance()->release();
		
		listDescriptorVector.push_back(descriptorVector);
		descriptorVector.clear();

		/*if (SVMlight::getInstance()->getAccuracy(xa_error, xa_recall, xa_precision)){
			_Accuracy acc;
			acc.xa_error = xa_error;
			acc.xa_precision = xa_precision;
			acc.xa_recall = xa_recall;
			listAcc.push_back(acc);
		}*/
		// </editor-fold>
		
	}
	fs.release();
	delete[] map;
}

void SvmHogTrain::saveListDescriptorVectorToFile(const std::string &file_name){
	cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
	char buffer[100];
	//sprintf_s(buffer, "%d", listDescriptorVector.size());
	int iBuffer = (int)listDescriptorVector.size();
	fs << "Num_Of_Class" << iBuffer;
	sprintf_s(buffer, "%d %d", this->hog.winSize.width, this->hog.winSize.height);
	fs << "Hog_WinSize" << buffer;
	sprintf_s(buffer, "%d %d", this->hog.blockSize.width, this->hog.blockSize.height);
	fs << "Hog_BlockSize" << buffer;
	sprintf_s(buffer, "%d %d", this->hog.blockStride.width, this->hog.blockStride.height);
	fs << "Hog_blockStride" << buffer;
	sprintf_s(buffer, "%d %d", this->hog.cellSize.width, this->hog.cellSize.height);
	fs << "Hog_cellSize" << buffer;
	for (size_t i = 0; i < listDescriptorVector.size(); ++i){
		sprintf_s(buffer, "Class_%d", i);
		int sizeOfVector = (int)listDescriptorVector[i].size();
		fs << buffer;
		//sprintf_s(buffer, "%d", sizeOfVector);
		fs << "{" << "Size" << sizeOfVector;
		for (size_t j = 0; j < sizeOfVector; ++j)
		{
			/*std::stringstream streamFeature;
			streamFeature << listDescriptorVector[i][j];*/
			sprintf_s(buffer, "v_%d", j);
			fs << buffer << listDescriptorVector[i][j];
		}
			
		fs	<< "}";
	}

	
	fs.release();
}

//void SvmHogTrain::saveAccuracy(const std::string & file_name){
//	std::fstream File;
//	File.open(file_name.c_str(), std::ios::out);
//	if (File.good() && File.is_open()) {
//		File << "Class, Error, Precision, Recall" << std::endl;
//		for (size_t i = 0; i < listAcc.size(); ++i){
//			File << i << "," << listAcc[i].xa_error << ","
//				<< listAcc[i].xa_precision << ","
//				<< listAcc[i].xa_recall << std::endl;
//		}
//	}
//
//	File.close();
//}

//eval
void SvmHogTrain::swapRandomListImage(std::vector<std::string>& listImage){
	size_t length = listImage.size();
	size_t lengthRandom = length / 2;
	int n1, n2;
	srand(time(NULL));
	for (size_t i = 0; i < lengthRandom; i++)
	{
		n1 = rand() % int(length);
		n2 = rand() % int(length);
		if (n1 >= 0 && n1 < length && n2 >= 0 && n2 < length){
			std::swap(listImage[n1], listImage[n2]);
		}
	}
}
bool SvmHogTrain::extractAllFeature(bool isRandom){
	std::vector <std::string> listFileTraningImages;
	std::vector<std::string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");

	FeatureHog fthog;
	std::vector<FeatureHog> listFtHog;
	for (size_t i = 0; i < listDir.size(); ++i){
		
		getFilesInDirectory(listDir[i], listFileTraningImages, validExtensions);
		if (isRandom){
			swapRandomListImage(listFileTraningImages);
		}

		for (size_t currentFile = 0; currentFile < listFileTraningImages.size(); ++currentFile){
			const std::string currentImageFile = listFileTraningImages[currentFile];
			std::vector<float> featureVector;
			calculateFeaturesFromInput(currentImageFile, featureVector);
			if (!featureVector.empty()){
				fthog.featureVector = featureVector;
				fthog.file_name = currentImageFile;
				fthog.label = int(i);
				listFtHog.push_back(fthog);
			}
		}
		this->listOfFeatureOfClass.push_back(listFtHog);

		for (size_t i = 0; i < listFtHog.size(); i++)
		{
			listFtHog[i].featureVector.clear();
		}
		fthog.featureVector.clear();
		listFtHog.clear();
		listFileTraningImages.clear();
	}
	validExtensions.clear();
	return EXIT_SUCCESS;
}

void SvmHogTrain::saveMultiClassFeatureToFile(std::string sMulClassFeatureFile) {
	
	std::fstream File;
	std::cout << "Saving multi-class features into 1 file..." << std::endl;
	File.open(sMulClassFeatureFile.c_str(), std::ios::out);

	for (size_t i = 0; i < this->listOfFeatureOfClass.size(); i++)
	{
		std::string label = std::to_string(i);
		std::cout << "Class " << label << std::endl;
		File << "# Class " << label << std::endl;
		for (size_t j = 0; j < this->listOfFeatureOfClass[i].size(); j++)
		{
			std::cout << ".";
			std::string sFeatures = label;
			std::vector<float> featureVector = this->listOfFeatureOfClass[i][j].featureVector;
			for (size_t feature = 0; feature < featureVector.size(); feature++)
			{
				sFeatures = sFeatures + " " + std::to_string(feature + 1) + ":" + std::to_string(featureVector.at(feature));
			}
			File << sFeatures << std::endl;
		}
		std::cout << std::endl;
	}
	File.flush();
	File.close();
}

void SvmHogTrain::saveListFeatureToFile(const std::vector< std::vector<float> >& listFeatureVectorPositive, 
	const std::vector< std::vector<float> >& listFeatureVectorNegative, const std::string& file_name){
	std::fstream File;
	std::cout << "Save Feature" << std::endl;
	File.open(file_name.c_str(), std::ios::out);
	if (File.good() && File.is_open()) {
		for (size_t i = 0; i < listFeatureVectorPositive.size(); i++)
		{
			std::vector<float> featureVector = listFeatureVectorPositive[i];
			if (!featureVector.empty()) {
				/* Put positive or negative sample class to file,
				* true=positive, false=negative,
				* and convert positive class to +1 and negative class to -1 for SVMlight
				*/
				File << "+1";
				// Save feature vector components
				for (size_t feature = 0; feature < featureVector.size(); ++feature) {
					File << " " << (feature + 1) << ":" << featureVector.at(feature);
				}
				File << std::endl;
			}
		}

		for (size_t i = 0; i < listFeatureVectorNegative.size(); i++)
		{
			std::vector<float> featureVector = listFeatureVectorNegative[i];
			if (!featureVector.empty()) {
				/* Put positive or negative sample class to file,
				* true=positive, false=negative,
				* and convert positive class to +1 and negative class to -1 for SVMlight
				*/
				File << "-1";
				// Save feature vector components
				for (size_t feature = 0; feature < featureVector.size(); ++feature) {
					File << " " << (feature + 1) << ":" << featureVector.at(feature);
				}
				File << std::endl;
			}
		}

		printf("\n");
		File.flush();
		File.close();
	}
}

void SvmHogTrain::crossValidation(const std::string& file_input, std::fstream& File){
	int numClassifier;
	if (this->listOfFeatureOfClass.size() != this->listLabels.size()){
		std::cout << "Num Of class must be equal size of listLabels" << std::endl;
		return;
	}	
	else if (this->listOfFeatureOfClass.size() == 2)
	{
		numClassifier = 1;
	}
	else if (this->listOfFeatureOfClass.size() > 2)
	{
		numClassifier = this->listOfFeatureOfClass.size();
	}
	else
	{
		std::cout << "Do NOT suppose 1-class SVM yet!" << std::endl;
		return;
	}

	std::vector<std::vector<int>> listIndexOfseparation;
	int *map;
	long mapLength = 0;
	// List of starting indices of classes in the concated list
	std::vector<int> listStartIndexOfClass;	
	int startInd = 0;
	for (size_t i = 0; i < this->listOfFeatureOfClass.size(); i++)
	{
		listStartIndexOfClass.push_back(startInd);
		startInd += this->listOfFeatureOfClass[i].size();
	}


	// Split samples of classes into k folds, contain them in indexOfEachClass[i] (corresponding to class i)
	for (size_t i = 0; i < this->listOfFeatureOfClass.size(); ++i){
		std::vector<int> indexOfEachClass;
		if (this->listOfFeatureOfClass[i].size() > 0){
			//indexOfEachClass.push_back(0);
			int step = this->listOfFeatureOfClass[i].size() / numOfSeparation;
			int numOfSection = this->listOfFeatureOfClass[i].size() / step;

			for (size_t j = 0; j <= numOfSection; ++j){
				indexOfEachClass.push_back(j * step);
			}
			listIndexOfseparation.push_back(indexOfEachClass);
			mapLength += listOfFeatureOfClass[i].size();
		}
	}
	map = new int[mapLength];
	std::vector< std::vector<float> > listOfPositiveVectorTrain;
	std::vector< std::vector<float> > listOfNegativeVectorTrain;

	std::vector<std::string> listOfFilePositiveTrain;
	std::vector<std::string> listOfFileNegativeTrain;

	std::vector<std::string> listOfFilePositiveTest;
	std::vector<std::string> listOfFileNegativeTest;

	std::vector< std::vector<float> > listOfPositiveVectorTest;
	std::vector< std::vector<float> > listOfNegativeVectorTest;

	std::vector<_Accuracy> tmpAcc;

	
	// Iteration through all classes
	// 'i' is the index of current class
	//for (size_t i = 0; i < listIndexOfseparation.size(); i++){ // num of class
	for (size_t i = 0; i < numClassifier; i++){ // num of class
		
		//for (size_t i = 0; i < 1; i++){ // num of class
		// Consider fold of each class
		//for (size_t j = 0; j < 2; j++){
		for (size_t j = 0; j < listIndexOfseparation[i].size() - 1; j++){
			if (this->listLabels.size() == 2) {
				std::cout << std::endl << "=== Class " << this->listLabels[0] << " vs " << this->listLabels[1] << "===" << std::endl;
			}
			else
			{
				std::cout << std::endl << "=== Class " << this->listLabels[i] << "-vs-All ===" << std::endl;
			}
			std::cout << ">> C = " << SVMlight::getInstance()->learn_parm->svm_c << std::endl;
			std::cout << "Fold " << j << std::endl;
			//testing data

			//memset(map, 1, sizeof(int)* mapLength);
			for (size_t k = 0; k < mapLength; k++)
			{
				map[k] = 1;
			}
			// Label all elements in map correspondint to current fold 0.
			// They are test samples.
			// Store test images in listOfFilePositiveTest
			for (size_t h = listIndexOfseparation[i][j]; h < listIndexOfseparation[i][j + 1]; h++)
			{
				//listOfFilePositiveTest.push_back(listOfFeatureOfClass[i][h].file_name);
				listOfPositiveVectorTest.push_back(listOfFeatureOfClass[i][h].featureVector);
				if (listStartIndexOfClass[i] + h >= mapLength) {
					std::cout << "[ERR] Out of range of 'map'" << std::endl;
				}
				else
				{
					map[listStartIndexOfClass[i] + h] = 0;
				}
			}
			for (size_t k = 0; k < listIndexOfseparation.size(); k++)
			{
				if (k != i)
				{
					for (size_t h = listIndexOfseparation[k][j]; h < listIndexOfseparation[k][j + 1]; h++)
					{
						//listOfFileNegativeTest.push_back(listOfFeatureOfClass[k][h].file_name);
						listOfNegativeVectorTest.push_back(listOfFeatureOfClass[k][h].featureVector);
						if (listStartIndexOfClass[k] + h >= mapLength) {
							std::cout << "[ERR] Out of range of 'map'" << std::endl;
						}
						else
						{
							map[listStartIndexOfClass[k] + h] = 0;
						}					
					}
				}
			}
			//testing data end

			//train feature
			for (size_t h = 0; h < listOfFeatureOfClass[i].size(); h++)//positive data
			{
				if (!(h >= listIndexOfseparation[i][j] && h < listIndexOfseparation[i][j + 1])){
					listOfPositiveVectorTrain.push_back(listOfFeatureOfClass[i][h].featureVector);
					listOfFilePositiveTrain.push_back(listOfFeatureOfClass[i][h].file_name);
				}
			}
			for (size_t k = 0; k < listIndexOfseparation.size(); k++)
			{
				if (k != i)
				{
					for (size_t h = 0; h < listOfFeatureOfClass[k].size(); h++)
					{
						if (!(h >= listIndexOfseparation[k][j] && h < listIndexOfseparation[k][j + 1])){
							listOfNegativeVectorTrain.push_back(listOfFeatureOfClass[k][h].featureVector);
							listOfFileNegativeTrain.push_back(listOfFeatureOfClass[k][h].file_name);
						}
					}
				}
			}
			//train feature end
			//saveListFeatureToFile(listOfPositiveVectorTrain, listOfNegativeVectorTrain, "vs_all.dat");

			trainBinaryBaseMap(file_input, map, mapLength, i);

			//trainBinary("vs_all.dat", descriptorVector);

			//testing 
			/*cv::HOGDescriptor hog_test;
			hog_test.winSize = this->hog.winSize;
			hog_test.blockSize = this->hog.blockSize;
			hog_test.blockStride = this->hog.blockStride;
			hog_test.cellSize = this->hog.cellSize;
			hog_test.setSVMDetector(descriptorVector);*/

			int tp, tn, fp, fn;
			_Accuracy acc;
			evaluate(listOfPositiveVectorTest, listOfNegativeVectorTest, tp, tn, fp, fn);
			acc.test_error = double(fp + fn) / (tp + tn + fp + fn);
			acc.test_precision = double(tp) / (tp + fp);
			acc.test_recall = double(tp) / (tp + fn);

			std::cout << "TP : " << tp << " TN : " << tn << " FP : " << fp << " FN : " << fn << std::endl;
			SVMlight::getInstance()->release();
			/*evaluate(hog_test, listOfFilePositiveTrain, listOfFileNegativeTrain, tp, tn, fp, fn);
			acc.train_error = double(fp + fn) / (tp + tn + fp + fn);
			acc.train_precision = double(tp) / (tp + fp);
			acc.train_recall = double(tp) / (tp + fn);*/

			tmpAcc.push_back(acc);
			//end
			//write to file
			//if (File)
			
			// end
			
			//clear
			//clearListOfString(listOfFilePositiveTest);
			//clearListOfString(listOfFileNegativeTest);
			clearListOfVectorFloat(listOfPositiveVectorTest);
			clearListOfVectorFloat(listOfNegativeVectorTest);
			clearListOfString(listOfFileNegativeTrain);
			clearListOfString(listOfFilePositiveTrain);

			/*listOfFilePositiveTest.clear();
			listOfFileNegativeTest.clear();			
			listOfFileNegativeTrain.clear();
			listOfFilePositiveTrain.clear();*/
			
			for (size_t k = 0; k < listOfPositiveVectorTrain.size(); k++){
				listOfPositiveVectorTrain[k].clear();
			}
			listOfPositiveVectorTrain.clear();
			for (size_t k = 0; k < listOfNegativeVectorTrain.size(); k++){
				listOfNegativeVectorTrain[k].clear();
			}
			listOfNegativeVectorTrain.clear();
			//clear end;
		}

		_Accuracy _avgAcc;
		_avgAcc.test_error = 0.0;
		_avgAcc.test_precision = 0.0;
		_avgAcc.test_recall = 0.0;

		_avgAcc.train_error = 0.0;
		_avgAcc.train_precision = 0.0;
		_avgAcc.train_recall = 0.0;
		for (size_t k = 0; k < tmpAcc.size(); k++){
			_avgAcc.test_error += tmpAcc[k].test_error ;
			_avgAcc.test_precision += tmpAcc[k].test_precision;
			_avgAcc.test_recall += tmpAcc[k].test_recall;

			_avgAcc.train_error += tmpAcc[k].train_error;
			_avgAcc.train_precision += tmpAcc[k].train_precision;
			_avgAcc.train_recall += tmpAcc[k].train_recall;
		}
		_avgAcc.test_error /= tmpAcc.size();
		_avgAcc.test_precision /= tmpAcc.size();
		_avgAcc.test_recall /= tmpAcc.size();

		_avgAcc.train_error /= tmpAcc.size();
		_avgAcc.train_precision /= tmpAcc.size();
		_avgAcc.train_recall /= tmpAcc.size();
		listAcc.push_back(_avgAcc);
		if (File.good() && File.is_open()){
			File << listLabels[i] << "," << KERNER_TYPE[SVMlight::getInstance()->getKernelType()] << "," << SVMlight::getInstance()->getSvmC() << "," << _avgAcc.test_error << "," << _avgAcc.test_precision << "," << _avgAcc.test_recall << std::endl;
		}
	}


	// Free 'map' allocation	
	if (map) {
		if (mapLength == 1) {
			delete  map;
		}
		else if (mapLength > 1)
		{
			delete[] map;
		}
	}
}
void SvmHogTrain::crossValidation(const std::string& file_input){
	this->crossValidation(file_input, std::fstream());
}

void SvmHogTrain::validation(const std::string& file_input, const std::vector<FeatureHog>& listOfFeatureTrain, const std::vector<FeatureHog>& listOfFeatureTest, std::fstream& File){
	int numClassifier = 0;
	if (this->listOfFeatureOfClass.size() != this->listLabels.size()){
		std::cout << "Num Of class must be equal size of listLabels" << std::endl;
		return;
	}
	else if (this->listOfFeatureOfClass.size() == 2)
	{
		numClassifier = 1;
	}
	else if (this->listOfFeatureOfClass.size() > 2)
	{
		numClassifier = this->listOfFeatureOfClass.size();
	}
	else
	{
		std::cout << "Do NOT suppose 1-class SVM yet!" << std::endl;
		return;
	}
	size_t mapLength = listOfFeatureTrain.size();
	int *map = new int[mapLength];
	memset(map, 1, mapLength * sizeof(int));
	/*std::vector<std::string> listOfFilePositiveTrainTest, listOfFileNegativeTrainTest;
	std::vector<std::string> listOfFilePositiveTest, listOfFileNegativeTest;*/
	std::vector<std::vector<float>> listOfFeaturePositiveTrainTest, listOfFeatureNegativeTrainTest;
	std::vector<std::vector<float>> listOfFeaturePositiveTest, listOfFeatureNegativeTest;
	//for (size_t i = 0; i < this->listOfFeatureOfClass.size(); i++){
	for (size_t i = numClassifier-1; i < numClassifier; i++){
		std::cout << "Class : " << this->listLabels[i] << " C = " << SVMlight::getInstance()->learn_parm->svm_c << std::endl;
		if (SVMlight::getInstance()->getKernelType() == RBF) {
			std::cout << "Gamma = " << SVMlight::getInstance()->getRbfGamma() << std::endl;
		}
		trainBinaryBaseMap(file_input, map, mapLength, double(i));
		//init hog to testing
		/*cv::HOGDescriptor hog_test;
		hog_test.winSize = this->hog.winSize;
		hog_test.blockSize = this->hog.blockSize;
		hog_test.blockStride = this->hog.blockStride;
		hog_test.cellSize = this->hog.cellSize;
		hog_test.setSVMDetector(descriptorVector);
		descriptorVector.clear();*/
		
		//select file positive & negative train test
		for (size_t j = 0; j < listOfFeatureTrain.size(); j++)
		{
			if (listOfFeatureTrain[j].label == int(i))
				listOfFeaturePositiveTrainTest.push_back(listOfFeatureTrain[j].featureVector);
			else
				listOfFeatureNegativeTrainTest.push_back(listOfFeatureTrain[j].featureVector);
		}

		for (size_t j = 0; j < listOfFeatureTest.size(); j++)
		{
			if (listOfFeatureTest[j].label == int(i))
				listOfFeaturePositiveTest.push_back(listOfFeatureTest[j].featureVector);
			else
				listOfFeatureNegativeTest.push_back(listOfFeatureTest[j].featureVector);
		}

		int tp, tn, fp, fn;
		evaluate(listOfFeaturePositiveTrainTest, listOfFeatureNegativeTrainTest, tp, tn, fp, fn);
		_Accuracy acc;
		acc.train_error = double(fp + fn) / (tp + tn + fp + fn);
		acc.train_precision = double(tp) / (tp + fp);
		acc.train_recall = double(tp) / (tp + fn);
		std::cout << "train err: " << acc.train_error << " train_precision: " << acc.train_precision << " train_recall: " << acc.train_recall << std::endl;
		evaluate(listOfFeaturePositiveTest, listOfFeatureNegativeTest, tp, tn, fp, fn);
		acc.test_error = double(fp + fn) / (tp + tn + fp + fn);
		acc.test_precision = double(tp) / (tp + fp);
		acc.test_recall = double(tp) / (tp + fn);

		std::cout << "test err: " << acc.test_error << " test_precision: " << acc.test_precision << " test_recall: " << acc.test_recall << std::endl;
		std::string classLabel;
		if (this->listLabels.size() == 2) {
			classLabel = this->listLabels[0] + "v" + this->listLabels[1];
		}
		else
		{
			classLabel = listLabels[i];
		}
		if (SVMlight::getInstance()->getKernelType() == LINEAR) 
		{
			File << classLabel << "," << KERNER_TYPE[SVMlight::getInstance()->getKernelType()] << "," << SVMlight::getInstance()->getSvmC() << "," << SVMlight::getInstance()->getSvNum() << "," << SVMlight::getInstance()->getVcDim() << "," << acc.train_error << "," << acc.train_precision << "," << acc.train_recall << "," << acc.test_error << "," << acc.test_precision << "," << acc.test_recall << std::endl;
		}
		else if (SVMlight::getInstance()->getKernelType() == RBF)
		{
			File << classLabel << "," << KERNER_TYPE[SVMlight::getInstance()->getKernelType()] << "," << SVMlight::getInstance()->getRbfGamma() << "," << SVMlight::getInstance()->getSvmC() << "," << SVMlight::getInstance()->getSvNum() << "," << SVMlight::getInstance()->getVcDim() << "," << acc.train_error << "," << acc.train_precision << "," << acc.train_recall << "," << acc.test_error << "," << acc.test_precision << "," << acc.test_recall << std::endl;
		}
		//clearListOfString(listOfFileNegativeTest);
		SVMlight::getInstance()->release();
		clearListOfVectorFloat(listOfFeatureNegativeTest);
		clearListOfVectorFloat(listOfFeaturePositiveTest);
		clearListOfVectorFloat(listOfFeaturePositiveTrainTest);
		clearListOfVectorFloat(listOfFeatureNegativeTrainTest);
	}

	delete[] map;
}
bool SvmHogTrain::detectTest(const cv::HOGDescriptor& hog_test, cv::Mat& imageData) {
	std:: vector<cv::Rect> found;
	std::vector<cv::Point> foundLocations, searchLocations;
	std::vector<double> weights;
	int groupThreshold = 2;
	cv::Size padding(cv::Size(2, 2));
	cv::Size winStride(cv::Size(1, 1));
	double hitThreshold = -0.0; // tolerance
	
	cv::resize(imageData, imageData, hog_test.winSize);

	hog_test.detect(imageData, foundLocations, weights, hitThreshold, winStride, padding, searchLocations);

	if (foundLocations.size() > 0)
	{
		/*std::cout << "YES" << foundLocations.size() << std::endl;
		for (size_t i = 0; i < weights.size(); i++)
		{
			std::cout << weights[i] << "  ";
		}
		std::cout << std::endl;*/
		return true;
	}
	else
	{
		//printf("NO\n");
		return false;
	}
}

void SvmHogTrain::evaluate(const std::vector<std::vector<float>>& vValPosFiles, const std::vector<std::vector<float> >& vValNegFiles, int &tp, int &tn, int &fp, int &fn) {
	int iNumSamples = vValPosFiles.size() + vValNegFiles.size();
	std::vector<bool> posLabels(vValPosFiles.size(), true);
	std::vector<bool> negLabels(vValNegFiles.size(), false);
	std::vector<bool> labels;
	labels.reserve(iNumSamples);
	labels.insert(labels.end(), posLabels.begin(), posLabels.end());
	labels.insert(labels.end(), negLabels.begin(), negLabels.end());
	posLabels.clear();
	negLabels.clear();

	cv::Mat imgTest;

	tp = 0;
	tn = 0;
	fp = 0;
	fn = 0;
	std::cout << "Begin test" << std::endl;
	// Iterate over sample images
	for (unsigned long currentFile = 0; currentFile < iNumSamples; ++currentFile) {
		//storeCursor();
		std::vector<float> featureVector;
		// Get positive or negative sample image file path
		const std::vector<float> currentVectorFeature = (currentFile < vValPosFiles.size() ? vValPosFiles.at(currentFile) : vValNegFiles.at(currentFile - vValPosFiles.size()));

		/*imgTest = cv::imread(currentImageFile, CV_LOAD_IMAGE_GRAYSCALE);
		if (imgTest.empty())
		{
			std::cout << "error read file:" << currentImageFile << std::endl;
			continue;
		}*/
		bool bTest;/* = detectTest(hog_test, imgTest);*/
		if (SVMlight::getInstance()->classify(currentVectorFeature) > 0)
			bTest = true;
		else
			bTest = false;
		bool bGT = labels[currentFile];

		if (bTest && bGT) {
			tp++;
		}
		else if (!bTest && !bGT)
		{
			tn++;
		}
		else if (bTest && !bGT)
		{
			fp++;
		}
		else
		{
			fn++;
		}
#ifdef IS_SHOW
		imshow("imgTest", imgTest);
		waitKey(0);
#endif
	}
	std::cout << "Test Done" << std::endl;
}

void SvmHogTrain::saveAccuracyValidation(const std::string & file_name){
	std::fstream File;
	File.open(file_name.c_str(), std::ios::out);
	if (File.good() && File.is_open()) {
		File << "Class, Train Error, Train Precision, Train Recall, Test Error, Test Precision, Test Recall" << std::endl;
		for (size_t i = 0; i < listAcc.size(); ++i){
			File << i << "," << listAcc[i].train_error << "," 
				<< listAcc[i].train_precision << "," << listAcc[i].train_recall 
				<< listAcc[i].test_error << "," << listAcc[i].test_precision 
				<< "," << listAcc[i].test_recall << std::endl;
			}
		}
	
	File.close();
}
void SvmHogTrain::separateData(float trainingPercent, std::vector<FeatureHog>& listOfFeatureTrain, std::vector<FeatureHog>& listOfFeatureTest){
	if (trainingPercent <0 || trainingPercent >1.0){
		std::cout << "trainingPercent must be in (0, 1)" << std::endl;
		return;
	}
	listOfFeatureTrain.clear();
	listOfFeatureTest.clear();
	for (size_t i = 0; i < this->listOfFeatureOfClass.size(); i++){
		size_t numOfFeatureTrain = (size_t)(trainingPercent * this->listOfFeatureOfClass[i].size());
		for (size_t j = 0; j < numOfFeatureTrain; j++)
		{
			listOfFeatureTrain.push_back(this->listOfFeatureOfClass[i][j]);
		}
		for (size_t j = numOfFeatureTrain; j < this->listOfFeatureOfClass[i].size(); j++)
		{
			listOfFeatureTest.push_back(this->listOfFeatureOfClass[i][j]);
		}
	}
}

void SvmHogTrain::saveMultiClassFeatureToFile(const std::vector<FeatureHog>& listOfFeatureTrain, const std::string& sMulClassFeatureFile){
	std::fstream File;
	std::cout << "Saving multi-class features into 1 file..." << std::endl;
	File.open(sMulClassFeatureFile.c_str(), std::ios::out);

	for (size_t i = 0; i < listOfFeatureTrain.size(); i++)
	{
		std::string label = std::to_string(listOfFeatureTrain[i].label);
			std::cout << ".";
		std::string sFeatures = label;
		std::vector<float> featureVector = listOfFeatureTrain[i].featureVector;
		for (size_t feature = 0; feature < featureVector.size(); feature++)
		{
			sFeatures = sFeatures + " " + std::to_string(feature + 1) + ":" + std::to_string(featureVector.at(feature));
		}
		File << sFeatures << std::endl;
		
		//std::cout << std::endl;
	}
	std::cout << std::endl;
	File.flush();
	File.close();
}

void SvmHogTrain::learningCurveAnalysis(const std::string& file_dataInput, const std::vector<FeatureHog>& listOfFeatureTrain, const std::vector<FeatureHog>& listOfFeatureTest, int sampleStep, int classLabel){
	if (this->listOfFeatureOfClass.size() != this->listLabels.size()){
		std::cout << "Num Of class must be equal size of listLabels" << std::endl;
		return;
	}
	if (size_t(classLabel) < 0 || size_t(classLabel) >= this->listOfFeatureOfClass.size()){
		std::cout << "out of class" << std::endl;
		return;
	}
	if (sampleStep < 0 || sampleStep >= listOfFeatureTrain.size()){
		std::cout << "sample test out of size listOfFeatureTrain" << std::endl;
		return;
	}
	size_t mapLength = listOfFeatureTrain.size();
	int *map = new int[mapLength];
	memset(map, 0, mapLength * sizeof(int));
	std::vector<long> bufferMap;
	for (size_t i = 0; i < mapLength; i++){
		bufferMap.push_back(long(i));
	}
	//
	std::vector<std::vector<float>> listOfFeaturePositiveTrainTest, listOfFeatureNegativeTrainTest;
	std::vector<std::vector<float>> listOfFeaturePositiveTest, listOfFeatureNegative, listOfFeatureNegativeTest;
	for (size_t i = 0; i < listOfFeatureTest.size(); i++){
		if (listOfFeatureTest[i].label == int(classLabel)){
			listOfFeaturePositiveTest.push_back(listOfFeatureTest[i].featureVector);
		}
		else{
			listOfFeatureNegative.push_back(listOfFeatureTest[i].featureVector);
		}
	}
	//init random
	srand(time(NULL));

	std::vector<size_t> bufferNegativeTest;
	std::vector<float> vTrainingErrors, vTestErrors;
	while (!bufferMap.empty()){
		
		for (size_t i = 0; i < sampleStep; i++)
		{
			// get random in  list feature train
			if (bufferMap.size() == 0)
				break;
			int randomNumber = rand() % int(bufferMap.size());
			if (randomNumber < 0 || randomNumber >= bufferMap.size()){
				continue;
			}
			if (bufferMap[randomNumber] >= mapLength)
			{
				std::cout << "errr" << std::endl;
			}
			else{
				map[bufferMap[randomNumber]] = 1;
			}
			
			if (listOfFeatureTrain[bufferMap[randomNumber]].label == int(classLabel)){
				listOfFeaturePositiveTrainTest.push_back(listOfFeatureTrain[bufferMap[randomNumber]].featureVector);
			}
			else{
				listOfFeatureNegativeTrainTest.push_back(listOfFeatureTrain[bufferMap[randomNumber]].featureVector);
			}
			bufferMap.erase(bufferMap.begin() + randomNumber);
		}

		
		trainBinaryBaseMap(file_dataInput,  map, mapLength, double (classLabel) );

		//init hog to testing
		/*cv::HOGDescriptor hog_test;
		hog_test.winSize = this->hog.winSize;
		hog_test.blockSize = this->hog.blockSize;
		hog_test.blockStride = this->hog.blockStride;
		hog_test.cellSize = this->hog.cellSize;
		hog_test.setSVMDetector(descriptorVector);
		descriptorVector.clear();*/
		// calc train err, 
		int tp, tn, fp, fn;
		evaluate(listOfFeaturePositiveTrainTest, listOfFeatureNegativeTrainTest, tp, tn, fp, fn);
		_Accuracy acc;
		acc.train_error = double(fp + fn) / (tp + tn + fp + fn);
		acc.train_precision = double(tp) / (tp + fp);
		acc.train_recall = double(tp) / (tp + fn);
		std::cout << "train err: " << acc.train_error << " train_precision: " << acc.train_precision << " train_recall: " << acc.train_recall << std::endl;

		for (size_t j = 0; j < listOfFeatureNegative.size(); j++)
		{
			bufferNegativeTest.push_back(j);
		}
		for (size_t j = 0; j < listOfFeaturePositiveTest.size(); j++)
		{
			int random_index = rand() % int(bufferNegativeTest.size());
			if (random_index < 0 || random_index >= bufferNegativeTest.size()){
				continue;
			}
			listOfFeatureNegativeTest.push_back(listOfFeatureNegative[bufferNegativeTest[random_index]]);
		}

		evaluate(listOfFeaturePositiveTest, listOfFeatureNegativeTest, tp, tn, fp, fn);
		acc.test_error = double(fp + fn) / (tp + tn + fp + fn);
		acc.test_precision = double(tp) / (tp + fp);
		acc.test_recall = double(tp) / (tp + fn);

		vTrainingErrors.push_back(acc.train_error);
		vTestErrors.push_back(acc.test_error);
		std::cout << "test err: " << acc.test_error << " test_precision: " << acc.test_precision << " test_recall: " << acc.test_recall << std::endl;
		//SVMlight::getInstance()->release();
		clearListOfVectorFloat(listOfFeatureNegativeTest);
	}

	CvPlot::plot("Learning Curve Analysis", vTrainingErrors, vTrainingErrors.size(), 1, 0, 255, 255);
	CvPlot::plot("Learning Curve Analysis", vTestErrors, vTestErrors.size(), 1, 255, 0, 0);
	//cleanup mem
	clearListOfVectorFloat(listOfFeaturePositiveTrainTest);
	clearListOfVectorFloat(listOfFeatureNegativeTrainTest);
	clearListOfVectorFloat(listOfFeaturePositiveTest);
	clearListOfVectorFloat(listOfFeatureNegative);
	delete[] map;
}