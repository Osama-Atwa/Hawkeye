#include"Avgpool.h";

Avgpool::Avgpool(int i_w, int i_h, int ch, int w_s, int s, int p) :layer(i_w, i_h, ch) { set_window_size(w_s); }

void Avgpool::set_window_size(int w_s)
{
	window_size = w_s;
}
void Avgpool::set_stride(int s)
{
	stride = s;
}
void Avgpool::set_padding(int p)
{
	padding = p;
}

int Avgpool::get_window_size() { return window_size; }
int Avgpool::get_stride() { return stride; }
int Avgpool::get_padding() { return padding; }

vector<vector<float>> Avgpool::convert(vector<float> v_input)
{
	int i_w = get_input_width();
	int i_h = get_input_height();
	vector<vector<float>> v_output;
	v_output.resize(i_h);

	for (int i = 0; i < i_w; i++)
	{
		v_output[i].resize(i_w);
	}
	for (int i = 0; i < v_input.size(); i++)
	{
		int row = i / i_h;
		int col = i % i_w;
		v_output[row][col] = v_input[i];
	}
	return v_output;
}


vector<vector<float>> Avgpool::mean_filter(vector<vector<float>> v_input)
{
	const int numrows = get_input_width();
	int numcols = get_input_height();
	int row, col;
	vector<vector<float> > v_output(numrows,vector<float>(numcols));

	vector<float> window;
	window.resize(9);
	
	float avg;
	for (row = 1; row < numrows-1; row++)
	{
		for (col = 1; col < numcols-1; col++)
		{
			//neighbor pixel values are stored in window including this pixel
			window[0] = v_input[row - 1][col - 1];
			window[1] = v_input[row - 1][col];
			window[2] = v_input[row - 1][col + 1];
			window[3] = v_input[row][col - 1];
			window[4] = v_input[row][col];
			window[5] = v_input[row][col + 1];
			window[6] = v_input[row + 1][col - 1];
			window[7] = v_input[row + 1][col];
			window[8] = v_input[row + 1][col + 1];

			avg = accumulate(window.begin(), window.end(), 0.0) / window.size();

			v_output[row][col] = avg;
		}
	}
	return v_output;
}

void Avgpool::load_parameters(vector<float>& V) 
{
	weights = V;
}
void Avgpool::execute(vector<float>& v_input,vector<float>& v_output) 
{
	int n_f = get_input_channels();
	int f_w = get_window_size();
	int f_h = get_window_size();

	int i_w = get_input_width();
	int i_h = get_input_height();
	int i_ch = get_input_channels();

	const af::array af_v_input = af::array(i_w, i_h, 1, i_ch, v_input.data());
	const af::array af_weights = af::array(f_w, f_h, 1, n_f, this->weights.data());
	
	af::array af_v_output = af::mean(af_v_input, af_weights);
	
	af::print("input", af_v_input);
	af::print("output", af_v_output);

	int arrlen = af_v_output.elements();
	float* dbl_ptr = af_v_output.host<float>();
	v_output = vector<float>(dbl_ptr, dbl_ptr + arrlen);
}
