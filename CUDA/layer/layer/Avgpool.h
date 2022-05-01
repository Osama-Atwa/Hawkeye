#pragma once
#include"layer.h"

class Avgpool :public layer {
private:
	int window_size;
public:
	Avgpool(int i_w, int i_h, int ch, int w_s);
	void load_parameters(vector<float>& V);
	void execute(vector<float>& v_input,vector<float>& v_output);
};