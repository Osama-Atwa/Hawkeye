#define AF_DEBUG
#define AF_CPU
#include <iostream>;
#include "Convolution.h";
#include "Avgpool.h";
#include "Maxpool.h";

vector<vector<float>> convert(vector<float> v_input)
{
	int i_w = 3;
	int i_h = 3;
	vector<vector<float>> v_output;
	v_output.resize(i_h);

	for (int i = 0; i < i_w; i++)
	{
		v_output[i].resize(i_w);
	}
	for (int i = 0; i < v_input.size(); i++)
	{
		int row = i / i_h;
		int col = i % i_w;
		v_output[row][col] = v_input[i];
	}
	return v_output;
}
int main() {
	Convolution conv2d = Convolution(3, 3, 1, 3, 3, 1, 1, 1);
	Avgpool Avgpool2d = Avgpool(3, 3, 1, 3, 1, 0);
	Maxpool maxpool = Maxpool(3, 3, 1, 3, 1, 0);

	vector<float> weights;
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);
	weights.push_back(1.0);

	vector<vector<float>> v = Avgpool2d.convert(weights);
	vector<vector<float>> result = Avgpool2d.vector_padding(v, 2, true);
	for (int i = 0; i < result.size(); i++)
	{
		for (int j = 0; j < result[i].size(); j++)
		{
			std::cout << result[i][j]<<" ";
		}
		std::cout << endl;
	}
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

	//conv2d.load_parameters(weights);
	//conv2d.execute(input, output);
	//
	vector<vector<float>> vv = convert(input);
	std::cout << vv.size() << vv[0].size() << endl;
	vector<vector<float>> vout = Avgpool2d.mean_filter(v,1);
	std::cout << vout.size() <<"  " << vout[0].size() <<"  "<< 1 << endl;
	//vector<vector<float>> vout = maxpool.max_filter(v, 1);

	//Avgpool2d.load_parameters(weights);
	//Avgpool2d.execute(input, output);

	vector<float> input2{ 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 };
	vector<float> output2;
	//Avgpool Avgpool2d2 = Avgpool(4, 4, 1, 3, 1, 0);
	//Avgpool2d2.execute(input2, output2);
	Maxpool Maxpool2d2 = Maxpool(4, 4, 1, 3, 1, 0);
	//Maxpool2d2.execute(input2, output2);
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

	for (int i = 0; i < vout.size(); i++)
	{
		for (int j = 0; j < vout[i].size(); j++)
		{
			std::cout << vout[i][j];
		}
		std::cout << endl;
	}
	return 0;
}