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

}
void Convolution::execute(vector<float>& v_input,vector<float>& v_output) {

}