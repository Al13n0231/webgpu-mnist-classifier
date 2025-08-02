CXX = clang++
CXXFLAGS = -std=c++17 -I../../ -I../../third_party/headers -O3 -Wall -Wextra
LDFLAGS = -L../../third_party/lib -Wl,-rpath,../../third_party/lib
LIBS = -lwebgpu_dawn

TARGETS = nn_layer mnist_classifier mnist_real

all: $(TARGETS)

nn_layer: nn_layer.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

mnist_classifier: mnist_classifier.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

mnist_real: mnist_real.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

run-nn: nn_layer
	./nn_layer

run-mnist: mnist_classifier
	./mnist_classifier

run-mnist-real: mnist_real
	./mnist_real

clean:
	rm -f $(TARGETS)

.PHONY: all run-nn run-mnist run-mnist-real clean