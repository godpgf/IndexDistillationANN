#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <map>

namespace parlayANN{

void printProgressBar(double progress, int width = 50) {
    std::cout << "[";
    int pos = width * progress;
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << " %\r";
    std::cout.flush();
}

struct TimerCount{
    void start(std::string tag){
        auto it = timer_map.find(tag);
        if(it == timer_map.end()){
            auto res = std::pair<std::chrono::high_resolution_clock::time_point, double>(std::chrono::high_resolution_clock::now(), 0.0);
            timer_map[tag] = res;
        } else {
            auto res = std::pair<std::chrono::high_resolution_clock::time_point, double>(std::chrono::high_resolution_clock::now(), it->second.second);
            timer_map[tag] = res;
        }
    }

    double end(std::string tag){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto it = timer_map.find(tag);
        double dt = 0;
        if(it != timer_map.end()){
            auto begin = it->second.first;
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
            dt = it->second.second + time_span.count();
            auto res = std::pair<std::chrono::high_resolution_clock::time_point, double>(begin, dt);
            timer_map[tag] = res;
        }
        return dt;
    }

    double count(std::string tag){
         std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto it = timer_map.find(tag);
        double dt = 0;
        if(it != timer_map.end()){
            auto begin = it->second.first;
            dt = it->second.second;
        }
        return dt;       
    }

    std::map<std::string, std::pair<std::chrono::high_resolution_clock::time_point, double>> timer_map;

};

}