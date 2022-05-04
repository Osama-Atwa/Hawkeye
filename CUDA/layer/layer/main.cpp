#define AF_DEBUG
#define AF_CPU
#include <iostream>;
#include "Convolution.h";
int main() {
	Convolution conv2d = Convolution(3, 3, 1, 3, 3, 1, 1, 1);
	vector<float> weights;
	weights.push_back(-1.0);
	weights.push_back(0.0);
	weights.push_back(1.0);
	weights.push_back(-2.0);
	weights.push_back(0.0);
	weights.push_back(2.0);
	weights.push_back(-1.0);
	weights.push_back(0.0);
	weights.push_back(1.0);

	vector<float> input;
	input.push_back(1.0);
	input.push_back(4.0);
	input.push_back(7.0);
	input.push_back(2.0);
	input.push_back(5.0);
	input.push_back(8.0);
	input.push_back(3.0);
	input.push_back(6.0);
	input.push_back(9.0);
	vector<float> output;

	conv2d.load_parameters(weights);
	conv2d.execute(input, output);
	
	//{
	//	return 0;
	//	af::array signal = constant(1.f, 3, 3);
	//	signal(0,0) = 1; signal(0, 1) = 2; signal(0, 2) = 3; signal(1, 0) = 4; signal(1, 1) = 5; signal(1, 2) = 6; signal(2, 0) = 7; signal(2, 1) = 8; signal(2, 2) = 9;
	//	af::array filter = constant(0 , 3, 3);
	//	//filter(0, 0) = 1; filter(0, 1) = 2; filter(0, 2) = 3; filter(1, 0) = 4; filter(1, 1) = 5; filter(1, 2) = 6; filter(2, 0) = 7; filter(2, 1) = 8; filter(2, 2) = 9;
	//	//filter(0, 0) = -1; filter(0, 1) = -2; filter(0, 2) = -1; filter(1, 0) = 0; filter(1, 1) = 0; filter(1, 2) = 0; filter(2, 0) = 1; filter(2, 1) = 2; filter(2, 2) = 1;
	//	filter(1, 1) = 1;
	//	filter(2, 1) = 1;
	//	dim4 strides(1, 1), dilation(0, 0, 0, 0);
	//	dim4 padding(1, 1, 1, 1);

	//	af::array convolved = convolve2NN(signal, filter, strides, padding, dilation);
	//	af::print("signal", signal);
	//	af::print("filter", filter);
	//	af::print("convolved", convolved);
	//	return 0;
	//}

	for (int i = 0; i < 9; i++)
	{
		std::cout << output[i]<<endl;
	}
	return 0;
}