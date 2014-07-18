//
//  main.cpp
//  DemoOpenCV-01
//
//  Created by BANG NGUYEN on 7/18/14.
//  Copyright (c) 2014 BANG NGUYEN. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, const char * argv[])
{
    cv::Mat img = cv::imread("/Users/bang/Desktop/Screen Shot 2014-01-17 at 18.50.07.png");
    
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    
    cv::imshow("Img", img);

    cv::waitKey(0);
    return 0;
}

