#include <omp.h>
#include <exception>
#include <thread>
#include <vld.h>

#include "util.h"
#include "LogInit.h"
#include "AlgorithmInitial.h"
#include "FTSCamera.h"
#include "FTSVideoAlgorithm.h"
#include "CVLine.h"


const unsigned int MAX_WORKER = std::max(std::thread::hardware_concurrency() - 1, 4U);
const unsigned int MAX_QUEUE = 100;

std::vector<std::thread> workers;
std::queue<ViolationEvent> taskQueue;
std::mutex mutexTaskQueue;

void doTask() {
//	for (;cv::waitKey(1) != 27;) {
	for (; ;) {
		if (!taskQueue.empty()) {
			if (workers.size() < MAX_WORKER) {
				mutexTaskQueue.lock();
				if (!taskQueue.empty()) {
					ViolationEvent e = taskQueue.front();
					taskQueue.pop();
					FTSANPR anpr;
					workers.push_back(std::thread(anpr, e));
					std::cout << "Start new worker! Number of task remaining: " << taskQueue.size() << std::endl << std::endl;
				}
				mutexTaskQueue.unlock();
			} else {
				std::cout << "All worker is busy. Waiting for 1s." << std::endl << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds(1));

				for (size_t i = 0; i < workers.size(); i++) {
					if (workers[i].joinable()) {
						workers[i].join();
						workers.erase(workers.begin() + i);
						std::cout << "Work done!" << std::endl << std::endl;
					}
				}
			}
		} else {
			std::cout << "Free task. Waiting for 5s." << std::endl << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(5));
		}
	}
}

int main(int argc, char *argv[]) {
	//init own algorithms
	initFTSAlgorithm();

	//init log framework
	initLog();
	boost::log::sources::severity_channel_logger< severity_level > lg;

	//read file configure 
	std::vector<FTSCamera> list;
	if (argc > 1) {
		readCameraFile(list, argv[1]);
	} else {
		readCameraFile(list, "camera.yml");
	}

	std::thread t1 = std::thread(doTask);

	//start camera process
	BOOST_LOG_CHANNEL_SEV(lg, "main", LOG_INFO) << "Number of cameras to process:" << list.size();
#pragma omp parallel for
	for (int i = 0; i < list.size(); i++) {
		try {
			BOOST_LOG_CHANNEL_SEV(lg, "main", LOG_TRACE) << "Camera " << i << " running with thread Id: " << std::this_thread::get_id();
			BOOST_LOG_CHANNEL_SEV(lg, "main", LOG_INFO) << "Reading param file for camera: " << list[i].strCameraId;

			cv::Ptr<FTSVideoAlgorithm> al = cv::Algorithm::create<FTSVideoAlgorithm>(list[i].strAlgorithm);
			readParamFile(al, list[i].strParamFile);
			(*al)(list[i], taskQueue);
		} catch (std::exception& e) {
			BOOST_LOG_CHANNEL_SEV(lg, "main", LOG_ERROR) << e.what();
		}
	}

	t1.join();
}
