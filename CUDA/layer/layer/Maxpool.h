#pragma once
#include "layer.h"

class Maxpool :public layer {
private:
	int window_size;
	int stride;
	int padding;
	vector<float> weights;
public:
	Maxpool(int i_w, int i_h, int ch,int w_s, int s, int p);

	void set_window_size(int w_s);
	void set_stride(int s);
	void set_padding(int p);

	int get_window_size();
	int get_stride();
	int get_padding();

	vector<vector<float>> convert(vector<float> v);
	vector<vector<float>> max_filter(vector<vector<float>> v, int s = 1);

	af::array af_max_filter(af::array v_input, int isz, int wsz, int stride, int padding);

	void load_parameters(vector<float>& V);
	void execute(vector<float>& v_input,vector<float>& v_output);
};