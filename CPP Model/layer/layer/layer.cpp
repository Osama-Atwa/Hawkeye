#include"layer.h"

layer::layer(int i_h, int i_w, int i_ch) {
	set_input_width(i_w);
	set_input_height(i_h);
	set_input_channels(i_ch);
}

void layer::set_input_width(int i_w) {
	input_width = i_w;
}
void layer::set_input_height(int i_h) {
	input_height = i_h;
}
void layer::set_input_channels(int ch) {
	input_channels = ch;
}

int layer::get_input_width() { return input_width; }
int layer::get_input_height() { return input_height; }
int layer::get_input_channels() { return input_channels; }

void layer::load_parameters(vector<float>& V) {}
void layer::execute(vector<float>& v_input,vector<float>& v_output){}