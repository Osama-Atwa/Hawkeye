#include"Avgpool.h";

Avgpool::Avgpool(int i_h, int i_w, int ch, int w_s, int s) :layer(i_h, i_w,  ch)
{
	set_window_size(w_s);
	set_stride(s);
	
}

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


vector<vector<float>> Avgpool::mean_filter(vector<vector<float>> v_input, int s )
{
	const int numrows = get_input_width();
	int numcols = get_input_height();
	int row, col;
	vector<vector<float> > v_output(numrows,vector<float>(numcols));

	vector<float> window;
	int window_size = get_window_size();
	window.resize(window_size * window_size);
	int x = floor(window_size / 2);
	float avg;
	int index = 0;
	for (row = x; row < numrows-x; row++)
	{
		for (col = x; col < numcols-x; col+=s)
		{
			if (window_size % 2 == 0)
			{
				for (int i = row - x; i < row + x; i++)
				{
					for (int j = col - x; j < col + x; j++)
					{
						window[index] = v_input[i][j];
						index++;
					}
				}
			}
			else
			{
				for (int i = row - x; i < row + x + 1; i++)
				{
					for (int j = col - x; j < col + x + 1; j++)
					{
						window[index] = v_input[i][j];
						index++;
					}
				}
			}
			
			avg = accumulate(window.begin(), window.end(), 0.0) / window.size();
			v_output[row][col] = avg;
			index = 0;
		}
	}
	return v_output;
}

vector<vector<float>> Avgpool::vector_padding(vector<vector<float>> v, int p_bits, bool zero)
{
	vector<vector<float>> result;
	result.resize(v.size() + 2*p_bits);

	vector<float> padding_vec;
	padding_vec.resize(v[0].size() + 2 * p_bits, 0.0);
	
	for (int i = 0; i < p_bits; i++)
	{
		if (zero)
		{
			vector<float> vv(v[0].size() + 2 * p_bits, 0.0);
			result[i] = vv;
			padding_vec[i] = 0.0;
		}
		else
		{
			vector<float> vv(v[0].size() + 2 * p_bits, 1.0);
			result[i] = vv;
			padding_vec[i] = 1.0;
		}

	}
	int index = p_bits + v[0].size();
	for (int l = 0; l < p_bits; l++)
	{
		padding_vec[l + index] = zero ? 0.0 : 1.0;
	}

	for (int i = 0; i < v.size(); i++)
	{
		for (int j = 0; j < v[0].size(); j++)
		{
			padding_vec[j + p_bits] = v[i][j];
		}
		
		result[p_bits + i] = padding_vec;
	}

	for (int i = 0; i < p_bits; i++)
	{
		if (zero)
		{
			vector<float> vv(v[0].size() + 2 * p_bits, 0.0);
			result[i+p_bits+v[0].size()] = vv;
		}
		else
		{
			vector<float> vv(v[0].size() + 2 * p_bits, 1.0);
			result[i + p_bits + v[0].size()] = vv;
		}

	}
	this->set_input_height(result.size());
	this->set_input_width(result[0].size());
	return result;
}

Array<float> Avgpool::HM_execute(Array<float> v_input, int s, int DEPTH)
{
	int  numcols = get_input_width();
	int  numrows = get_input_height();
	int row, col;
	
	vector<vector<float> > v_output(numrows, vector<float>(numcols));
	vector<float> V_out;
	vector<float> window;

	int window_size = get_window_size();
	window.resize(window_size * window_size);
	int x = floor(window_size / 2);
	float avg;
	int index = 0;
	for (int d = 0; d < DEPTH; d++)
	{
		for (row = x; row < numrows - x; row += s)
		{
			for (col = x; col < numcols - x; col += s)
			{
				if (window_size % 2 == 0)
				{
					for (int i = row - x; i < row + x; i++)
					{
						for (int j = col - x; j < col + x; j++)
						{
							window[index] = v_input(j,i,d);
							index++;
						}
					}
				}
				else
				{
					for (int i = row - x; i < row + x + 1; i++)
					{
						for (int j = col - x; j < col + x + 1; j++)
						{
							window[index] = v_input(j, i, d);
							index++;
						}
					}
				}

				avg = accumulate(window.begin(), window.end(), 0.0) / window.size();
				V_out.push_back(avg);
				index = 0;
			}
		}
	}
	// W2=(W1−F)/S+1
	// H2 = (H1−F) / S + 1
	// D2 = D1
	int W2 = ((numrows - window_size) / s) + 1;
	int H2 = ((numcols - window_size) / s) + 1;
	int D2 = DEPTH;
	vector<int> dim_img({ W2,H2,D2 });
	Array<float> output(dim_img);
	output.fill_data(V_out);
	return output;
}
void Avgpool::load_parameters(Array<float>& V)
{
	weights.fill_data(V.get_data());
}
void Avgpool::execute(Array<float>& v_input,Array<float>& v_output)
{
	//int n_f = get_input_channels();
	//int f_w = get_window_size();
	//int f_h = get_window_size();

	//int i_w = get_input_width();
	//int i_h = get_input_height();
	//int i_ch = get_input_channels();

	//int osz = (i_w - f_w + 2 * get_padding()) / get_stride() + 1;
	//int o_ch = n_f;

	//const af::array af_v_input = af::array(i_w, i_h, 1, i_ch, v_input.get_data().data());
	////const af::array af_weights = af::array(f_w, f_h, 1, n_f, this->weights.data());
	//
	////af::array af_v_output = af::mean(af_v_input, af_weights);
	//af::array af_v_output(osz, osz, o_ch);
	//for (int i = 0; i < o_ch; i++)
	//{
	//	af_v_output(span, span, i) = af_mean_filter(af_v_input(span, span, i), osz, f_w, get_stride(), get_padding());
	//}
	////af::array af_v_output = af_mean_filter(af_v_input, osz, f_w, get_stride(), get_padding());
	//
	//af::print("input", af_v_input);
	//af::print("output", af_v_output);

	//int arrlen = af_v_output.elements();
	//float* dbl_ptr = af_v_output.host<float>();
	//v_output.fill_data(vector<float>(dbl_ptr, dbl_ptr + arrlen));
}
