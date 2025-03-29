build:
	nvcc --std=c++11 image_blur.cu -Iinclude -o blur.exe -Xcompiler "/MD"

run: build
	.\blur.exe
	fc /b result_cpu.png result_gpu.png > nul && echo ### SUCCESS: Files Are Identical! ### || echo ### WARNING: Files Are Different! ###
