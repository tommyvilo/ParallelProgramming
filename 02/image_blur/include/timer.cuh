#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32 // Se stai compilando su Windows
#include <windows.h>
#else
#include <time.h> // Per Linux/macOS
#endif

class timerGPU
{
public:
    timerGPU() = default;
    ~timerGPU();

    void start();
    void stop();
    float elapsed_ms();

private:
    cudaEvent_t m_start, m_end;
};

class timerCPU
{
public:
    void start();
    void stop();
    float elapsed_ms();

private:
#ifdef _WIN32
    LARGE_INTEGER m_start, m_end, freq;
#else
    struct timespec m_start, m_end;
#endif
};

#ifdef PPROG_TIMER

// ==================== GPU TIMER ====================
timerGPU::~timerGPU()
{
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_end);
}

void timerGPU::start()
{
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_end);
    cudaEventRecord(m_start, 0);
}

void timerGPU::stop()
{
    cudaEventRecord(m_end, 0);
    cudaEventSynchronize(m_end);
}

float timerGPU::elapsed_ms()
{
    float ms;
    cudaEventElapsedTime(&ms, m_start, m_end);
    return ms;
}

// ==================== CPU TIMER ====================
void timerCPU::start()
{
#ifdef _WIN32
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&m_start);
#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &m_start);
#endif
}

void timerCPU::stop()
{
#ifdef _WIN32
    QueryPerformanceCounter(&m_end);
#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &m_end);
#endif
}

float timerCPU::elapsed_ms()
{
#ifdef _WIN32
    return (double)(m_end.QuadPart - m_start.QuadPart) * 1000.0 / freq.QuadPart;
#else
    long ns = (m_end.tv_sec - m_start.tv_sec) * (long)1e9 + (m_end.tv_nsec - m_start.tv_nsec);
    return ns * 1e-6;
#endif
}

#endif
