#define AF_DEBUG
#define AF_CPU
#include <iostream>;
#include "Convolution.h";
int main() {
	Convolution conv2d = Convolution(3, 3, 1, 3, 3, 1, 1, 0);
	vector<float> weights;
	weights.push_back(-1.0);
	weights.push_back(-2.0);
	weights.push_back(-1.0);
	weights.push_back(0.0);
	weights.push_back(0.0);
	weights.push_back(0.0);
	weights.push_back(1.0);
	weights.push_back(2.0);
	weights.push_back(1.0);

	vector<float> input;
	input.push_back(1.0);
	input.push_back(2.0);
	input.push_back(3.0);
	input.push_back(4.0);
	input.push_back(5.0);
	input.push_back(6.0);
	input.push_back(7.0);
	input.push_back(8.0);
	input.push_back(9.0);
	vector<float> output;

	conv2d.load_parameters(weights);
	conv2d.execute(input, output);
	for (int i = 0; i < 9; i++)
	{
		std::cout << output[i]<<endl;
	}
	return 0;
}