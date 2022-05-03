#include "Convolution.h"

Convolution::Convolution(int i_w, int i_h, int ch, int no_f, int f_w,int f_h):layer(i_w, i_h, ch) {
	set_no_filters(no_f);
	set_filters_w_h(f_w,f_h);
}

void Convolution::set_no_filters(int n) {
	no_filters = n;
}
void Convolution::set_filters_w_h(int f_w,int f_h) {
	filters_h = f_h;
	filters_w = f_w;
}

int Convolution::get_no_filters() { return no_filters; }
int Convolution::get_filters_w() { return filters_w; }
int Convolution::get_filters_h() { return filters_h; }

void Convolution::load_parameters(vector<float>& V) {

	int n_f = get_no_filters();
	int f_w = get_filters_w();
	int f_h = get_filters_h();
	vector<float> F1;
	vector<vector<float>> F2;
	for (int i = 0; i < n_f; i++)
	{
		for (int j = 0; j < f_w; j++)
		{
			for (int l = 0; l < f_h; l++)
			{
				F1.push_back(V[i + j + l]);
			}
			F2.push_back(F1);
			F1.clear();
		}
		weights.push_back(F2);
		F2.clear();
	}
}
void Convolution::execute(vector<float>& v_input,vector<float>& v_output) {
	v_output = convolve2NN(v_input, weights);
}