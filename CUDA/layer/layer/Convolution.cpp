#include "Convolution.h"

Convolution::Convolution(int i_w, int i_h, int i_ch, int f_w, int f_h, int no_f, int s, int p):layer(i_w, i_h, i_ch) {
	set_no_filters(no_f);
	set_filters_w_h(f_w,f_h);
	set_stride(s);
	set_padding(p);
}

void Convolution::set_no_filters(int n) {
	no_filters = n;
}
void Convolution::set_filters_w_h(int f_w,int f_h) {
	filters_h = f_h;
	filters_w = f_w;
}
void Convolution::set_stride(int s)
{
	stride = s;
}
void Convolution::set_padding(int p)
{
	padding = p;
}

int Convolution::get_no_filters() { return no_filters; }
int Convolution::get_filters_w() { return filters_w; }
int Convolution::get_filters_h() { return filters_h; }
int Convolution::get_stride() { return stride; }
int Convolution::get_padding() { return padding; }
void Convolution::load_parameters(vector<float>& V) {

	weights = V;
	/*int n_f = get_no_filters();
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
	}*/
}
void Convolution::execute(vector<float>& v_input,vector<float>& v_output) {
	int n_f = get_no_filters();
	int f_w = get_filters_w();
	int f_h = get_filters_h();

	int i_w = get_input_width();
	int i_h = get_input_height();
	int i_ch = get_input_channels();

	af::dim4 s (get_stride(), get_stride());
	af::dim4 p (get_padding(), get_padding());
	af::dim4 dil (0, 0,0,0);
//	std::cout << s << p << endl;
	const af::array af_v_input = af::array( i_w, i_h, i_ch, v_input.data());
	const af::array af_weights = af::array(f_w,f_h, n_f, this->weights.data());
	af::array af_v_output = convolve2NN(af_v_input, af_weights,s,p,dil);
	
	int arrlen = af_v_output.elements();
	float* dbl_ptr = af_v_output.host<float>();

	vector<float> values(dbl_ptr, dbl_ptr + arrlen);
	v_output = values;

}