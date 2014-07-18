#ifndef _FTS_TASK_QUEUE_H_
#define _FTS_TASK_QUEUE_H_


#include <queue>
#include <vector>
#include <mutex>
#include <thread>


#include "util.h"
#include "FTSANPR.h"
#include "ViolationEvent.h"

class FTSTaskQueue {
public:
	static std::string const className() { return "FTSTaskQueue"; }
	//static const unsigned int MAX_WORKER = std::max(std::thread::hardware_concurrency() - 2, 2U); //at least 2 worker
	static const unsigned int MAX_WORKER = 2;
	static const unsigned int MAX_QUEUE = 100;
	static std::vector<std::thread> workers;
	static std::queue<ViolationEvent> taskQueue;
	static std::mutex mutexTaskQueue;
	FTSTaskQueue() {};
	~FTSTaskQueue() {};

	//virtual void operator() () {
	void doTask() {
		for (;;) {
			if (!taskQueue.empty()) {
				if (workers.size() < MAX_WORKER) {
					mutexTaskQueue.lock();
					if (!taskQueue.empty()) {
						ViolationEvent e = taskQueue.front();
						taskQueue.pop();
						FTSANPR anpr;
						workers.push_back(std::thread(anpr, e));
					}
					mutexTaskQueue.unlock();
				} else {
					std::this_thread::sleep_for(std::chrono::seconds(1));

					std::cout << "All worker is busy. Waiting for 1s." << std::endl << std::endl;

					for (size_t i = 0; i < workers.size(); i++) {
						if (workers[i].joinable()) {
							workers[i].join();
							workers.erase(workers.begin() + i);
						}
					}
				}
			} else {
				std::cout << "Free task. Waiting for 5s." << std::endl << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds(5));
			}
		}
	}

	void pushTask(ViolationEvent e) {
		taskQueue.push(e);
		std::cout << "New Task" << std::endl;
	}
};


#endif