/*
 * fts_anpr_pcaocr.cpp
 *
 *  Created on: May 11, 2014
 *      Author: sensen
 */

#include "fts_anpr_pcaocr.h"

FTS_ANPR_PcaOcr::FTS_ANPR_PcaOcr()
	: m_voPca()
	, m_oStandardCharSize( STANDARD_PCA_CHAR_WIDTH, STANDARD_PCA_CHAR_HEIGHT )
	, m_oCharClasses()
{
	// Nothing
}

FTS_ANPR_PcaOcr::~FTS_ANPR_PcaOcr()
{
	// Nothing
}

Mat FTS_ANPR_PcaOcr::formatImagesForPCA( const vector<Mat>& data )
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}

bool FTS_ANPR_PcaOcr::load( const string& sTrainPath )
{
	// vectors to hold training images of each class
	vector<Mat> images;

	/*
	 * Load training data and caculate PCA matrix for each class
	 */
	DIR *dir;
	struct dirent *ent;
	int class_index = 0;
	if ((dir = opendir (sTrainPath.c_str())) != NULL)
	{
		DIR *subdir;
		struct dirent *subent;
		while ((ent = readdir (dir)) != NULL)
		{
			if( strcmp(ent->d_name, ".")  && strcmp(ent->d_name, "..")  ) // ignore . and .. folders
			{
			  if ((subdir = opendir ((sTrainPath + "/" + ent->d_name).c_str())) != NULL) // dir subdirectory
			  {
					cout << "Training class " << class_index++ << " : " << ent->d_name << endl;

					// start load images of each class
					images.clear();
					while ((subent = readdir (subdir)) != NULL)
					{
						std::string filename = subent->d_name;
						bool b = ( filename.substr(filename.find_last_of(".") + 1) == "jpg" );

						if( strcmp(subent->d_name, ".")  && strcmp(subent->d_name, "..") && b )
						{
							Mat img = imread((sTrainPath + "/" + ent->d_name + "/" + subent->d_name).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
							if(! img.data ) // Check for invalid input
							{
								cout <<  "Could not open or find the image" << endl ;
								return -1;
							}
							Mat img_resize;
							resize(img, img_resize, m_oStandardCharSize, 0, 0, INTER_CUBIC);
							images.push_back(img_resize);
						}
					}

					// Reshape and stack images into a rowMatrix
					Mat data = formatImagesForPCA(images);

					// Perform PCA for each class
					PCA pca = PCA(data, cv::Mat(), CV_PCA_DATA_AS_ROW, NUM_OF_COMPONENT);
					m_voPca.push_back(pca);

					// Add character class
					m_oCharClasses.push_back( ent->d_name );
			  }
			}
		}
		closedir (dir);
	}
	else
	{
		/* could not open directory */
		perror ("Coundn't open directory");
		return false;
	}

	return true;
}

string FTS_ANPR_PcaOcr::ocr( const cv::Mat& img ) const
{
	Mat img_resize;
	resize(img, img_resize, m_oStandardCharSize, 0, 0, INTER_CUBIC);
	Mat test_data(1, img_resize.rows*img_resize.cols, CV_32F);
	Mat image_row = img_resize.clone().reshape(1,1);
	Mat row_i = test_data.row(0);
	image_row.convertTo(row_i,CV_32F);

	// Calculate error score
	double minScore = 10000000;
	int predClass = 1;
	for (unsigned i=0; i<m_voPca.size(); i++)
	{
		Mat point = m_voPca[i].project(test_data.row(0)); // project into the eigenspace, thus the image becomes a "point"
		Mat reconstruction = m_voPca[i].backProject(point); // re-create the image from the "point"
		double error = norm(test_data.row(0), reconstruction, NORM_L2)/test_data.cols;
//		cout << "Reconstruction error (L2) on class " << (i) << " = " << error << endl;
		if (error < minScore)
		{
			minScore = error;
			predClass = i;
		}
	}
	cout << "Recognized class = Class " << predClass << endl;

	return m_oCharClasses[predClass];
}
