CC:=/usr/local/cuda-9.1/bin/nvcc
PROF:=/usr/local/cuda-9.1/bin/nvprof

run: n-body
	./n-body ./Test.png

prof: n-body
	${PROF} ./n-body ./Test.png
	@echo
	time -v ./n-body ./Test.png

n-body: main.cu
	${CC} -ccbin g++-6 main.cu -o n-body `pkg-config --libs opencv`

main-image: main-image.cu
	${CC} -ccbin g++-6 main-image.cu -o main-image `pkg-config --libs opencv`


