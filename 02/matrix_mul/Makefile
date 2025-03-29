build:
	nvcc --std=c++11 matrix_mul.cu -Iinclude -o matrix_mul.exe -Xcompiler "/MD"

run: build
	.\matrix_mul.exe

CXXFLAGS += -DPPROG_TIMER

