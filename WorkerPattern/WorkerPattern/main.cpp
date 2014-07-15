//
//  main.cpp
//  WorkerPattern
//
//  Created by BANG NGUYEN on 7/15/14.
//  Copyright (c) 2014 BANG NGUYEN. All rights reserved.
//

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <random>

std::vector<std::thread> worker;

//TODO: create struct for this
std::queue<std::string> task_queue;
std::mutex task_queue_mutex;

unsigned int MAX_WORKER = std::max(std::thread::hardware_concurrency() - 2, 5U);
unsigned int MAX_QUEUE = 100;

void puttask() {
	for (size_t i = 0; i < 10; i++) {
		int task_num = std::rand() % 20 + 1;
        
		for (size_t i = 0; i < task_num; i++) {
			task_queue.push(std::string(i + 1, 'x'));
		}
        
		std::cout << "New " << task_num << " task!!Number of task : " << task_queue.size() << std::endl << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
	
}

void dequeue() {
	task_queue_mutex.lock();
	if (!task_queue.empty()) {
		std::string data = task_queue.front();
		task_queue.pop();
		std::cout << "Data: " << data.c_str() << std::endl;
	}
	else {
		std::cout << "Collision: " << std::endl;
	}
	task_queue_mutex.unlock();
    
	std::cout << "Number of task remaining: " << task_queue.size() << std::endl << std::endl;
}

void distributer() {
	for ( ; ; ) {
		if (!task_queue.empty()) {
			if (worker.size() < MAX_WORKER) {
				worker.push_back(std::thread(dequeue));
			}
			else {
				std::this_thread::sleep_for(std::chrono::seconds(1));
                
				std::cout << "All worker is busy. Waiting for 1s." << std::endl << std::endl;
                
				for (size_t i = 0; i < worker.size(); i++) {
					if (worker[i].joinable()) {
						worker[i].join();
						worker.erase(worker.begin() + i);
					}
				}
			}
		}
		else {
			std::cout << "Free task. Waiting for 5s." << std::endl << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(5));
		}
	}
}

int main(int argc, char *argv[]) {
	std::srand(time(NULL));
	std::cout << "Max threads supported: " << MAX_WORKER << std::endl << std::endl;
    
	std::thread task = std::thread(puttask);
	std::thread work = std::thread(distributer);
	task.join();
	work.join();
    
}