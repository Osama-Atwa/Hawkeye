#pragma once
#define AFAPI   __attribute__((visibility("default")))
#include"layer.h";

class Convolution :public layer {
private:
	int no_filters;
	int filters_w;
	int filters_h;
	int stride;
	int padding;
	vector<float> weights;
	
public:
	Convolution(int i_w, int i_h, int i_ch, int f_w,int f_h, int no_f, int s, int p);
	
	void set_no_filters(int n);
	void set_filters_w_h(int f_w,int f_h);
	void set_stride(int s);
	void set_padding(int p);

	int get_no_filters();
	int get_filters_w();
	int get_filters_h();
	int get_stride();
	int get_padding();

	void load_parameters(vector<float>& V);
	void execute(vector<float>& v_input,vector<float>& v_output);
};