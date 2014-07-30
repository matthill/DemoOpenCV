#ifdef DETECT_MEM_LEAK
#include <vld.h>
#endif
#include "svm_util.h"

#include "dirent.h"

std::string toLowerCase(const std::string& in) {
	std::string t;
	for (std::string::const_iterator i = in.begin(); i != in.end(); ++i) {
		t += tolower(*i);
	}
	return t;
}
void getFilesInDirectory(const std::string& dirName, std::vector<std::string>& fileNames, const std::vector<std::string>& validExtensions){
	printf("Opening directory %s\n", dirName.c_str());
	struct dirent* ep;
	size_t extensionLocation;
	DIR* dp = opendir(dirName.c_str());
	if (dp != NULL) {
		while ((ep = readdir(dp))) {
			// Ignore (sub-)directories like . , .. , .svn, etc.
			if (ep->d_type & DT_DIR) {
				continue;
			}
			extensionLocation = std::string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
			// Check if extension is matching the wanted ones
			std::string tempExt = toLowerCase(std::string(ep->d_name).substr(extensionLocation + 1));
			if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
				printf("Found matching data file '%s'\n", ep->d_name);
				fileNames.push_back((std::string)dirName + ep->d_name);
			}
			else {
				printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
			}
		}
		(void)closedir(dp);
	}
	else {
		printf("Error opening directory '%s'!\n", dirName.c_str());
	}
	return;

}

void clearListOfString(std::vector<std::string> &vs) {
	for (size_t i = 0; i < vs.size(); i++)
	{
		vs[i].clear();
	}
	vs.clear();
}

void clearListOfVectorFloat(std::vector< std::vector<float> > &vs){
	for (size_t i = 0; i < vs.size(); i++)
	{
		vs[i].clear();
	}
	vs.clear();
}
bool isFileExist(const std::string& strFile) {
	struct stat info;

	//check item exits
	if ((stat(strFile.c_str(), &info) == 0) && (info.st_mode & S_IFMT))
		return true;

	return false;
}
bool isDirExist(const std::string& strDir) {
	struct stat info;

	//check item exits
	if ((stat(strDir.c_str(), &info) == 0) && (info.st_mode & S_IFDIR))
		return true;

	return false;
}

void groupRectangles(std::vector<DetectionObject>& objList, int groupThreshold, double eps, GROUP_TYPE type)
{
	if (groupThreshold <= 0 || objList.empty() || objList.size() != objList.size())
	{
		/*size_t i, sz = objList.size();
		scores.resize(sz);
		for (i = 0; i < sz; i++)
			scores[i] = 1;*/
		return;
	}
	

	std::vector<int> labels;
	int nclasses = cv::partition(objList, labels, FTS_SimilarRects(eps));

	std::vector<DetectionObject> rObjs(nclasses);
	std::vector<int> rweights(nclasses, 0);
	std::vector<double> rscores(nclasses, 0);
	std::vector<int> rejectLevels(nclasses, 0);
	std::vector<int> rejectLevelsMax(nclasses, 0);
	int i, j, nlabels = (int)labels.size();
	//if (type == GROUP_AVG )
	{
		for (i = 0; i < nlabels; i++)
		{
			int cls = labels[i];
			rObjs[cls].boundingBox.x += int(objList[i].boundingBox.x * (objList)[i].score);
			rObjs[cls].boundingBox.y += int(objList[i].boundingBox.y * (objList)[i].score);
			rObjs[cls].boundingBox.width += int(objList[i].boundingBox.width * (objList)[i].score);
			rObjs[cls].boundingBox.height += int(objList[i].boundingBox.height * (objList)[i].score);
			rObjs[cls].label = objList[i].label;
			rweights[cls]++;
			rscores[cls] += (objList)[i].score;
		}
	}
	
	
	for (i = 0; i < nlabels; i++)
	{
		int cls = labels[i];
		if ((objList)[i].score > rejectLevels[cls])
		{
			rejectLevels[cls] = (objList)[i].score;
			rejectLevelsMax[cls] = i;
		}
	}
	

	for (i = 0; i < nclasses; i++)
	{
		if (type == GROUP_MAX){
			rObjs[i] = objList[rejectLevelsMax[i]];
		}
		else {
			DetectionObject r = rObjs[i];
			float s = 1.f / rscores[i];
			rObjs[i] = DetectionObject(cv::saturate_cast<int>(r.boundingBox.x*s),
				cv::saturate_cast<int>(r.boundingBox.y*s),
				cv::saturate_cast<int>(r.boundingBox.width*s),
				cv::saturate_cast<int>(r.boundingBox.height*s), rscores[i], rObjs[i].label);
		}
	}

	objList.clear();
	
	//scores.clear();

	for (i = 0; i < nclasses; i++)
	{
		DetectionObject r1 = rObjs[i];
		int n1 = rweights[i];
		
		if (n1 <= groupThreshold)
			continue;
		// filter out small face rectangles inside large rectangles
		for (j = 0; j < nclasses; j++)
		{
			int n2 = rweights[j];

			if (j == i || n2 <= groupThreshold)
				continue;
			DetectionObject r2 = rObjs[j];

			int dx = cv::saturate_cast<int>(r2.boundingBox.width * eps);
			int dy = cv::saturate_cast<int>(r2.boundingBox.height * eps);

			if (i != j &&
				r1.boundingBox.x >= r2.boundingBox.x - dx &&
				r1.boundingBox.y >= r2.boundingBox.y - dy &&
				r1.boundingBox.x + r1.boundingBox.width <= r2.boundingBox.x + r2.boundingBox.width + dx &&
				r1.boundingBox.y + r1.boundingBox.height <= r2.boundingBox.y + r2.boundingBox.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3))
				break;
		}

		if (j == nclasses)
		{
			objList.push_back(r1);
			
		}
	}
}

//++28.06 trung add to build success on vs2010
#if _MSC_VER <= 1600
namespace std
{
	std::string to_string(int i)
	{
		return std::to_string(static_cast<long long>(i)); 
	}
	
	std::string to_string(size_t i)
	{
		return std::to_string(static_cast<long long>(i)); 
	}
	
	std::string to_string(float i)
	{
		return std::to_string(static_cast<long double>(i)); 
	}
}
#endif
//--