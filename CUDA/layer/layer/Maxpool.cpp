#include"Maxpool.h";

Maxpool::Maxpool(int i_w, int i_h, int ch, int w_s, int s, int p) :layer(i_w, i_h, ch) { set_window_size(w_s); }

void Maxpool::set_window_size(int w_s) { window_size = w_s; }
void Maxpool::set_stride(int s) { stride = s; }
void Maxpool::set_padding(int p) { padding = p; }

int Maxpool::get_window_size() { return window_size; }
int Maxpool::get_stride() { return stride; }
int Maxpool::get_padding() { return padding; }


vector<vector<float>> Maxpool::convert(vector<float> v_input)
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


vector<vector<float>> Maxpool::max_filter(vector<vector<float>> v_input, int s)
{
	const int numrows = get_input_width();
	int numcols = get_input_height();
	int row, col;
	vector<vector<float> > v_output(numrows, vector<float>(numcols));

	vector<float> window;
	int window_size = get_window_size();
	window.resize(window_size * window_size);
	int x = floor(window_size / 2);
	float maximum = INT_MIN;
	for (row = x; row < numrows - x; row++)
	{
		for (col = x; col < numcols - x; col++)
		{
			for (int i = row - x; i < row + x + 1; i++)
			{
				for (int j = col - x; j < col + x + 1; j += s)
				{
					if (maximum < v_input[i][j])
					{
						maximum = v_input[i][j];
					}
				}
			}
			v_output[row][col] = maximum;
			maximum = INT_MIN;
		}
	}
	return v_output;
}


void Maxpool::load_parameters(vector<float>& V){}
void Maxpool::execute(vector<float>& v_input,vector<float>& v_output){}
