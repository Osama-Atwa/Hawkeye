#include"Maxpool.h";

Maxpool::Maxpool(int i_w, int i_h, int ch, int w_s) :layer(i_w, i_h, ch) { window_size = w_s; }
void Maxpool::load_parameters(vector<float>& V){}
void Maxpool::execute(vector<float>& v_input,vector<float>& v_output){}
