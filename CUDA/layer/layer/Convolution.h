#pragma once
#define AFAPI   __attribute__((visibility("default")))
#include "layer.h";

class Convolution :public layer {
private:
	int no_filters;
	int filters_w;
	int filters_h;
	int strid;
	int padding;
	//vector<float> weights;
	vector<Array<float>> weights;
public:
	Convolution(int i_h, int i_w, int i_ch, int f_h, int f_w, int f_ch);
	
	void set_no_filters(int n);
	void set_filters_w_h(int f_w,int f_h);
	void set_stride(int s);
	void set_padding(int p);

	int get_no_filters();
	int get_filters_w();
	int get_filters_h();
	int get_stride();
	int get_padding();
	vector<float> Flatten(vector<vector<float>> v);
	void load_parameters(vector<Array<float>>& V);
	void execute(Array<float>& v_input,Array<float>& v_output);
	
	vector<vector<float>> HM_excute(vector<vector<float>> v_input, int strid);
	Array<float> HM_excute_Array(Array<float> v_input, int strid);
	Array<float> HM_excute_Array_Depth(Array<float> v_input, vector<int> stride, int p_bits, bool zero, int DEPTH);

	vector<vector<float>> vector_padding(vector<vector<float>> v, int p_bits, bool zero);
	vector<float> convert_2d_2_1d(vector<vector<float>>v);
};