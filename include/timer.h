#pragma once

#include <chrono>

class Timer {
public:
    Timer() = default;
    ~Timer() = default;

    static std::chrono::high_resolution_clock::time_point tic() {
        return std::chrono::high_resolution_clock::now();
    }

    template<typename T = std::chrono::milliseconds>
    static T toc(std::chrono::high_resolution_clock::time_point& start_time) {
        return std::chrono::duration_cast<T>(
            std::chrono::high_resolution_clock::now() - start_time
        );
    }

};