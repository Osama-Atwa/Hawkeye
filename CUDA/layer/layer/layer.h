#pragma once
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <numeric>
#include <cstdlib>
#include"Array.h"
#include <algorithm>

using namespace std;
using std::vector;

class layer {
private:
	int input_width;
	int input_height;
	int input_channels;
public:
	layer(int i_h, int i_w, int i_ch);

	void set_input_width(int i_w);
	void set_input_height(int i_h);
	void set_input_channels(int i_ch);

	int get_input_width();
	int get_input_height();
	int get_input_channels();

	virtual void load_parameters(vector<float>& V);
	virtual void execute(vector<float>& v_input,vector<float>& v_output);
};