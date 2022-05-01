#include"Avgpool.h";

Avgpool::Avgpool(int i_w, int i_h, int ch, int w_s) :layer(i_w, i_h, ch) { window_size = w_s; }
void Avgpool::load_parameters(vector<float>& V) {}
void Avgpool::execute(vector<float>& v_input,vector<float>& v_output) {}
