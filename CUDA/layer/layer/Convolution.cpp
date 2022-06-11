#include "Convolution.h"

Convolution::Convolution(int i_h, int i_w, int i_ch, int f_h, int f_w, int f_ch):layer(i_w, i_h, i_ch) {
	set_no_filters(f_ch);
	set_filters_w_h(f_w,f_h);
	vector<int> dim = { f_w,f_h,f_ch };
	for (int i = 0; i < weights.size(); i++)
	{
		weights[i].set_dim(dim);
	}

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
	strid = s;
}
void Convolution::set_padding(int p)
{
	padding = p;
}

int Convolution::get_no_filters() { return no_filters; }
int Convolution::get_filters_w() { return filters_w; }
int Convolution::get_filters_h() { return filters_h; }
int Convolution::get_stride() { return strid; }
int Convolution::get_padding() { return padding; }
void Convolution::load_parameters(vector<Array<float>>& V) {

	weights = V;
}

vector<float> Convolution::Flatten(vector<vector<float>> v)
{
	int row = v.size();
	int col = v[0].size();
	vector<float> res;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			res.push_back(v[i][j]);
		}
	}
	return res;
}


vector<vector<float>> Convolution::HM_excute(vector<vector<float>> v_input, int strid)
{
	const int numrows = get_input_width();
	int numcols = get_input_height();
	int row, col;
	vector<vector<float> > v_output(numrows, vector<float>(numcols));

	vector<float> window;

	int window_size = get_filters_h();
	window.resize(window_size * window_size);
	int x = floor(window_size / 2);
	vector<float> w = weights[0].get_data();
	vector<float> res;
	res.resize(w.size());
	int v;
	int index = 0;
	for (row = x; row < numrows - x; row++)
	{
		for (col = x; col < numcols - x; col+= strid)
		{
			for (int i = row - x; i < row + x + 1; i++)
			{
				for (int j = col - x; j < col + x + 1; j ++)
				{
					window[index] = v_input[i][j];
					index++;
				}
			}
			for (int i = 0; i < w.size(); i++)
			{
				res[i] = window[i] * w[i];
			}
			v = accumulate(res.begin(), res.end(), 0.0);
			v_output[row][col] = v;
			index = 0;
		}
	}
	return v_output;
}

Array<float> Convolution::HM_excute_Array(Array<float> v_input, int strid)
{
	const int numrows = get_input_height();
	int numcols = get_input_width();
	int row, col;

	vector<float> window;
	vector<vector<float> > v_output(numrows, vector<float>(numcols));
	int window_size = get_filters_h();
	window.resize(window_size * window_size);
	int x = floor(window_size / 2);
	vector<float> w = weights[0].get_data();
	vector<float> res;
	res.resize(w.size());
	int v;
	int index = 0;
	for (row = x; row < numrows - x; row++)
	{
		for (col = x; col < numcols - x; col += strid)
		{
			for (int i = row - x; i < row + x + 1; i++)
			{
				for (int j = col - x; j < col + x + 1; j++)
				{
					window[index] = v_input(i,j);
					index++;
				}
			}
			for (int i = 0; i < w.size(); i++)
			{
				res[i] = window[i] * w[i];
			}
			v = accumulate(res.begin(), res.end(), 0.0);
			v_output[row][col] = v;
			index = 0;
		}
	}
	
	vector<int> dim_img({ 3,3 });
	Array<float> output(dim_img);
	vector<float> V_out = Convolution::Flatten(v_output);
	output.fill_data(V_out);
	return output;
}

vector<float> Convolution::convert_2d_2_1d(vector<vector<float>>v)
{
	vector<float> output;
	for (int i = 0; i < v.size(); i++)
	{
		for (int j = 0; j < v[0].size(); j++)
		{
			output.push_back(v[i][j]);
		}
	}
	return output;
}

Array<float> Convolution::HM_excute_Array_Depth(Array<float> _input, Array<float> bias, vector<int> strid, int p_bits, bool zero, int DEPTH)
{

	

	int numrows = get_input_height();
	int numcols = get_input_width();
	int row, col;

	vector<float> vec_input;
	
	for (int i = 0; i < DEPTH; i++)
	{
		vector<vector<float>> channel;
		vector<float>row_;
		vector<float> channel_1d;
		for (int r = 0; r < numrows; r++)
		{
			for (int c = 0; c < numcols; c++)
			{
				row_.push_back(_input(c, r, i));
			}
			channel.push_back(row_);
			row_.clear();
		}
		channel = vector_padding(channel, p_bits, zero);
		channel_1d = convert_2d_2_1d(channel);
		if (DEPTH == 0)
		{
			vec_input = channel_1d;
		}
		else {
			vec_input.insert(vec_input.end(), channel_1d.begin(), channel_1d.end());
		}
	}
	int new_w = 2 * p_bits + numcols;
	int new_h = 2 * p_bits + numrows;
	vector<int> diminsion({ new_h,new_w,DEPTH });

	Array<float> v_input(diminsion);
	v_input.fill_data(vec_input);
	
	this->set_input_height(new_h);
	this->set_input_width(new_w);

	numcols = new_w;
	numrows = new_h;

	vector<float> window;
	int window_h = get_filters_h();
	int window_w = get_filters_w();
	window.resize(window_h * window_h);
	int x = floor(window_h / 2);

	vector<vector<float> > v_output(numrows, vector<float>(numcols));
	vector<float> c_output;
//	vector<float> w = weights.get_data();
	vector<float> res;
	res.resize(window_h * window_w);

	int v;
	int index = 0;
	vector<float> V_out;
	vector<float> result;

	bool first_time = true;
	bool first_time_res = true;

	for (int filter_index = 0; filter_index < weights.size(); filter_index++)
	{
		for (int d = 0; d < DEPTH; d++)
		{
			for (row = x; row < numrows - x; row += strid[0])
			{
				for (col = x; col < numcols - x; col += strid[1])
				{
					for (int i = row - x; i < row + x + 1; i++)
					{
						for (int j = col - x; j < col + x + 1; j++)
						{
							/*if (i == 3 && j == 0 )
							{
								int x = weights(i, j, d);
							}*/
							window[index] = v_input(j, i, d);
							index++;
						}
					}
					for (int i = 0; i < window_h; i++)
					{
						for (int j = 0; j < window_w; j++)
						{
							//cout << i << " " << j << " " << d << endl;
							int x = i * window_w + j;
							//cout << x << " "<< window[x]<<" "<< weights(i, j, d)<<" " <<res[x]<< endl;
							res[x] = window[x] * weights[filter_index](j, i, d);
						}
					}
					v = accumulate(res.begin(), res.end(), 0.0);

					//v_output[col][row] = v;
					
					c_output.push_back(v);
					/*if (col == 110)
					{
						cout << row << " " << col << endl;
					}*/
					index = 0;
				}
			}
			//vector<float> vector2 = Convolution::Flatten(v_output);
			
			if (first_time)
			{
				V_out = c_output;
				first_time = false;
			}
			else
			{
				std::transform(V_out.begin(), V_out.end(), c_output.begin(), V_out.begin(), std::plus<float>());
			}

			c_output.clear();
			
		}

		first_time = true;

		for (float& x_ : V_out) // if you want to add 10 to each element
		{
			x_ += bias(filter_index);
			//relu
			if (x_ < 0)
			{
				x_ = 0;
			}
		}

		if (first_time_res)
		{
			result = V_out;
			first_time_res = false;
		}
		else
		{
			result.insert(result.end(), V_out.begin(), V_out.end());
			V_out.clear();
		}
	}


	int output_h = ((numrows - window_h + 2 * p_bits) / strid[0]) + 1;
	int output_w = ((numcols - window_w + 2 * p_bits) / strid[1]) + 1;
	vector<int> dim_img({ output_h,output_w,(int)weights.size() });
	Array<float> output(dim_img);
	output.fill_data(result);
	return output;
}
vector<vector<float>> Convolution::vector_padding(vector<vector<float>> v, int p_bits, bool zero)
{
	vector<vector<float>> result;
	result.resize(v.size() + 2 * p_bits);

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
			result[i + p_bits + v[0].size()] = vv;
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