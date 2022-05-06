#pragma once
#include"layer.h"

class Avgpool :public layer {
private:
	int window_size;
	int stride;
	int padding;
	Array<float> weights;
public:
	Avgpool(int i_w, int i_h, int ch, int w_s, int s, int p);
	
	void set_window_size(int w_s);
	void set_stride(int s);
	void set_padding(int p);
	
	int get_window_size();
	int get_stride();
	int get_padding();

	vector<vector<float>> convert(vector<float> v);
	vector<vector<float>> vector_padding(vector<vector<float>> v, int p , bool zero); // true means zero padding and false means on padding
	vector<vector<float>> mean_filter(vector<vector<float>> v, int s = 1 );

	af::array af_mean_filter(af::array v_input, int osz, int wsz, int stride, int padding);
	
	void load_parameters(Array<float>& V);
	void execute(Array<float>& v_input, Array<float>& v_output);
};