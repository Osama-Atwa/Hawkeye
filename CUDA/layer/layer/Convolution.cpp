#include "Convolution.h"

Convolution::Convolution(int i_w, int i_h, int i_ch, int f_w, int f_h, int no_f, int s, int p):layer(i_w, i_h, i_ch) {
	set_no_filters(no_f);
	set_filters_w_h(f_w,f_h);
	set_stride(s);
	set_padding(p);
	vector<int> dim = { f_w,f_h,no_f };
	weights.set_dim(dim);
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
void Convolution::load_parameters(Array<float>& V) {

	weights.fill_data(V.get_data());
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
af::array my_convolve2_unwrap(const af::array & signal, const af::array & filter,
	const dim4& stride, const dim4& padding,
	const dim4& dilation) {
	dim4 sDims = signal.dims();
	dim4 fDims = filter.dims();

	dim_t outputWidth =
		1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
		stride[0];
	dim_t outputHeight =
		1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
		stride[1];

	const bool retCols = false;
	af::array unwrapped =
		unwrap(signal, fDims[0], fDims[1], stride[0], stride[1], padding[0],
			padding[1], retCols);

	print("unwrapped", unwrapped);
	unwrapped = reorder(unwrapped, 1, 2, 0, 3);
	print("unwrapped reorder", unwrapped);
	dim4 uDims = unwrapped.dims();
	unwrapped =
		moddims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));
	print("unwrapped moddims", unwrapped);

	af::array collapsedFilter = flip(filter, 0);
	collapsedFilter = flip(collapsedFilter, 1);
	print("collapsedFilter", collapsedFilter);

	collapsedFilter = moddims(collapsedFilter,
		dim4(fDims[0] * fDims[1] * fDims[2], fDims[3]));
	print("collapsedFilter moddims", collapsedFilter);

	af::array res =
		matmul(unwrapped, collapsedFilter, AF_MAT_TRANS, AF_MAT_NONE);
	print("res", res);

	cout << "res moddims new dims " << outputWidth << " " << outputHeight << " " << sDims[3] << " " << fDims[3] << endl;

	res = moddims(res, dim4(outputWidth, outputHeight, sDims[3], fDims[3]));
	print("res moddims", res);
	af::array out = reorder(res, 0, 1, 3, 2);
	print("out", out);

	return out;
}

af::array myconvolve2NN(
	const af::array& signal, const af::array& filter,
	const dim4 stride,      // NOLINT(performance-unnecessary-value-param)
	const dim4 padding,     // NOLINT(performance-unnecessary-value-param)
	const dim4 dilation) {  // NOLINT(performance-unnecessary-value-param)
	af_array out = 0;
	af_convolve2_nn(&out, signal.get(), filter.get(), 2, stride.get(),
		2, padding.get(), 2, dilation.get());
	return af::array(out);
}

void Convolution::execute(Array<float>& v_input,Array<float>& v_output) {
	int n_f = get_no_filters();
	int f_w = get_filters_w();
	int f_h = get_filters_h();

	int i_w = get_input_width();
	int i_h = get_input_height();
	int i_ch = get_input_channels();

	af::dim4 s (get_stride(), get_stride());
	af::dim4 p (get_padding(), get_padding(), 1, 1);
	af::dim4 dil (1, 1,0,0);
	//std::cout << s << p << endl;
	const af::array af_v_input = af::array( i_w, i_h, 1, i_ch, v_input.get_data().data());
	const af::array af_weights = af::array(f_w,f_h, 1, n_f, this->weights.get_data().data());
	//cout << "stride " << s[0] << " " << s[1] << " " << s[2] << " " << s[3] << endl;

	//cout << "padding " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << endl;


	//cout << "dilation " << dil[0] << " " << dil[1] << " " << dil[2] << " " << dil[3] << endl;

	//af::array af_v_output = my_convolve2_unwrap(af_v_input, af_weights,s,p,dil);
	//af::array af_v_output = convolve2NN(af_v_input, af_weights,s,p,dil);
	af::array af_v_output = convolve2NN(af_v_input, af_weights,s,p,dil);

	//af::print("input", af_v_input);
	//af::print("weights", af_weights);
	//af::print("output", af_v_output);

	int arrlen = af_v_output.elements();
	float* dbl_ptr = af_v_output.host<float>();

	v_output.fill_data(vector<float>(dbl_ptr, dbl_ptr + arrlen));
	//v_output = values;
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
	vector<float> w = weights.get_data();
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
	const int numrows = get_input_width();
	int numcols = get_input_height();
	int row, col;

	vector<float> window;
	vector<vector<float> > v_output(numrows, vector<float>(numcols));
	int window_size = get_filters_h();
	window.resize(window_size * window_size);
	int x = floor(window_size / 2);
	vector<float> w = weights.get_data();
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
Array<float> Convolution::HM_excute_Array_Depth(Array<float> v_input, int strid, int DEPTH)
{
	const int numrows = get_input_width();
	int numcols = get_input_height();
	int row, col;

	vector<float> window;
	int window_h = get_filters_h();
	int window_w = get_filters_w();
	window.resize(window_h * window_h);
	int x = floor(window_h / 2);

	vector<vector<float> > v_output(numrows, vector<float>(numcols));

	vector<float> w = weights.get_data();
	vector<float> res;
	res.resize(window_h * window_w);

	int v;
	int index = 0;
	vector<float> V_out(numrows* numcols);

	for (int d = 0; d < DEPTH; d++)
	{
		for (row = x; row < numrows - x; row++)
		{
			for (col = x; col < numcols - x; col += strid)
			{
				for (int i = row - x; i < row + x + 1; i++)
				{
					for (int j = col - x; j < col + x + 1; j++)
					{
						if (i == 3 && j == 0 )
						{
							int x = weights(i, j, d);
						}
						window[index] = v_input(i, j, d);
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
						res[x] = window[x] * weights(i,j,d);
					}
				}
				v = accumulate(res.begin(), res.end(), 0.0);

				v_output[col][row] = v;

				/*if (col == 110)
				{
					cout << row << " " << col << endl;
				}*/
				index = 0;
			}
		}
		vector<float> vector2 = Convolution::Flatten(v_output);
		std::transform(V_out.begin(), V_out.end(), vector2.begin(), V_out.begin(), std::plus<float>());
	}
	vector<int> dim_img({ 112,112 });
	Array<float> output(dim_img);
	output.fill_data(V_out);
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