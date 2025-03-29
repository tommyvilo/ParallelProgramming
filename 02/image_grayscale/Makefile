build:
	nvcc --std=c++11 image_grayscale.cu -Iinclude -o grayscale.exe -Xcompiler "/MD"

run: build
	.\grayscale.exe
	fc /b result_cpu.png result_gpu.png > nul && echo ### SUCCESS: Files Are Identical! ### || echo ### WARNING: Files Are Different! ###
