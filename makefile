CXX = g++
SRCS = matrix_nn.cpp activation_function.cpp sigmoid.cpp sigmoid_derivative.cpp main.cpp mse.cpp softmax.cpp softmax_finalizer.cpp relu.cpp relu_derivative.cpp mnist_iterator.cpp
EXE_FILE = matrix_nn

$EXE_FILE:
	$(CXX) --std=c++11 $(SRCS) -o $(EXE_FILE)

clean:
	rm -f *.o $(EXE_FILE)
