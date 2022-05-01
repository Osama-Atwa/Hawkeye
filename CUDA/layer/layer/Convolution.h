#pragma once
#include"layer.h";

class Convolution :public layer {
private:
	int no_filters;
	int filters_w;
	int filters_h;
	vector<vector<vector<float>>> weights ;
public:
	Convolution(int i_w, int i_h, int i_ch, int no_f, int f_w,int f_h);
	
	void set_no_filters(int n);
	void set_filters_w_h(int f_w,int f_h);
	
	int get_no_filters();
	int get_filters_w();
	int get_filters_h();

	void load_parameters(vector<float>& V);
	void execute(vector<float>& v_input,vector<float>& v_output);
};