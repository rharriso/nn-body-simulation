CC:=/usr/local/cuda-9.1/bin/nvcc
PROF:=/usr/local/cuda-9.1/bin/nvprof

run: n-body
	./n-body

prof: n-body
	${PROF} ./n-body
	@echo
	time -v ./n-body

n-body: main.cu
	${CC} -ccbin g++-6 main.cu -o n-body

