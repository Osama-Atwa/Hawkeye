#pragma once
#include "layer.h"

class Maxpool :public layer {
private:
	int window_size;
	int stride;
	int padding;
	Array<float> weights;
public:
	Maxpool(int i_h, int i_w,  int ch,int w_s, int s);

	void set_window_size(int w_s);
	void set_stride(int s);
	void set_padding(int p);

	int get_window_size();
	int get_stride();
	int get_padding();

	vector<vector<float>> convert(vector<float> v);
	vector<vector<float>> max_filter(vector<vector<float>> v, int s = 1);

	//af::array af_max_filter(af::array v_input, int isz, int wsz, int stride, int padding);
	Array<float> HM_execute(Array<float> v_input, int stride, int DEPTH);
	void load_parameters(Array<float>& V);
	void execute(Array<float>& v_input, Array<float>& v_output);
};