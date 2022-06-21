#define AF_DEBUG
#define AF_CPU
#include <iostream>;
#include "Convolution.h";
#include "Avgpool.h";
#include "Maxpool.h";
#include <opencv2/opencv.hpp>
#include <fstream>
#include "json.hpp"
#include <chrono>
using namespace cv;
using namespace std;

vector<vector<float>> convert(vector<float> v_input)
{
    int i_w = 112;
    int i_h = 112;
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

uchar *  convert_char(vector<float> v_input)
{
    uchar* result = new uchar[v_input.size()];
    for (int i = 0; i < v_input.size(); i++)
    {
        result[i] = v_input[i];
    }
    
    return result;
}

uint8_t** setupHMM(vector<vector<float> >& vals, int N, int M)
{
    uint8_t** temp;
    temp = new uint8_t * [N];
    for (unsigned i = 0; (i < N); i++)
    {
        temp[i] = new uint8_t[M];
        for (unsigned j = 0; (j < M); j++)
        {
            temp[i][j] = (uint8_t)vals[i][j];
        }
    }
    return temp;
}
Array<float> FireModule(Array<float> input, vector<vector<Array<float>>> weights, vector<Array<float>> bias)
{
    int i_h = input.get_dim()[0];
    int i_w = input.get_dim()[1];
    int i_depth = input.get_dim()[2];
    
    Convolution conv2d_1 = Convolution(i_h, i_w, i_depth, 1, 1, i_depth);
    conv2d_1.load_parameters(weights[0]);
    Array<float> squeeze = conv2d_1.HM_excute_Array_Depth(input, bias[0], {1, 1}, 0, true, i_depth);

    int s_h = squeeze.get_dim()[0];
    int s_w = squeeze.get_dim()[1];
    int s_depth = squeeze.get_dim()[2];

    Convolution conv2d_2 = Convolution(s_h, s_w, s_depth, 1, 1, s_depth);
    conv2d_2.load_parameters(weights[1]);
    Array<float> expand1 = conv2d_2.HM_excute_Array_Depth(squeeze, bias[1], { 1, 1 }, 0, true, s_depth);

    Convolution conv2d_3 = Convolution(s_h, s_w, s_depth, 3, 3, s_depth);
    conv2d_3.load_parameters(weights[2]);
    Array<float> expand2 = conv2d_3.HM_excute_Array_Depth(squeeze, bias[2], { 1, 1 }, 2, true, squeeze.get_dim()[2]);
    
    int x_h = expand1.get_dim()[0];
    int x_w = expand1.get_dim()[1];
    int x_depth = expand1.get_dim()[2];
    int x2_depth = expand2.get_dim()[2];

    vector<float> V_out;
    V_out = expand1.get_data();
    V_out.insert(V_out.end(), expand2.get_data().begin(), expand2.get_data().end());

    vector<int> dim({ x_h,x_w,x_depth + x2_depth });

    Array<float> result;

    result.set_dim(dim);
    result.fill_data(V_out);

    return result;
}

Array<float> SqueezeNetV1_1(Array<float> v_input,vector<vector<Array<float>>> weights, vector<Array<float>> bias ,int nb_classes)
{
    int i_h = v_input.get_dim()[0];
    int i_w = v_input.get_dim()[1];
    int i_depth = v_input.get_dim()[2];
    Convolution conv1 = Convolution(i_h, i_w, i_depth, 3, 3, i_depth);
    conv1.load_parameters(weights[0]);
    Array<float> conv1_out = conv1.HM_excute_Array_Depth(v_input, bias[0], { 2,2 }, 0, false, i_depth);

    int c1_h = conv1_out.get_dim()[0];
    int c1_w = conv1_out.get_dim()[1];
    int c1_depth = conv1_out.get_dim()[2];

    cout << endl << "Layer : conv1_out" << "(" << c1_h << "," << c1_w << "," << c1_depth << ")" << endl;

    Maxpool maxpool1 = Maxpool(c1_h, c1_w, c1_depth, 3, 2);
    Array<float> max1_out = maxpool1.HM_execute(conv1_out, 2, c1_depth);
    cout << endl << "Layer : max1_out" << "(" << max1_out.get_dim()[0] << "," << max1_out.get_dim()[1] << "," << max1_out.get_dim()[2] << ")" << endl;
    
    vector<vector<Array<float>>> fire1_weights;
    fire1_weights.push_back(weights[1]);
    fire1_weights.push_back(weights[2]);
    fire1_weights.push_back(weights[3]);
    vector<Array<float>> f1_bias;
    f1_bias.push_back(bias[1]);
    f1_bias.push_back(bias[2]);
    f1_bias.push_back(bias[3]);
    Array<float> Fire1 = FireModule(max1_out, fire1_weights, f1_bias);
    cout << endl << "Layer : Fire1" << "(" << Fire1.get_dim()[0] << "," << Fire1.get_dim()[1] << "," << Fire1.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire2_weights;
    fire2_weights.push_back(weights[4]);
    fire2_weights.push_back(weights[5]);
    fire2_weights.push_back(weights[6]);
    vector<Array<float>> f2_bias;
    f2_bias.push_back(bias[4]);
    f2_bias.push_back(bias[5]);
    f2_bias.push_back(bias[6]);
    Array<float> Fire2 = FireModule(Fire1, fire2_weights, f2_bias);
    cout << endl << "Layer : Fire2" << "(" << Fire2.get_dim()[0] << "," << Fire2.get_dim()[1] << "," << Fire2.get_dim()[2] << ")" << endl;

    int f2_h = Fire2.get_dim()[0];
    int f2_w = Fire2.get_dim()[1];
    int f2_depth = Fire2.get_dim()[2];
    Maxpool maxpool2 = Maxpool(f2_h, f2_w, f2_depth, 3, 2);
    Array<float> max2_out = maxpool2.HM_execute(Fire2, 2, f2_depth);
    cout << endl << "Layer : max2_out" << "(" << max2_out.get_dim()[0] << "," << max2_out.get_dim()[1] << "," << max2_out.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire3_weights;
    fire3_weights.push_back(weights[7]);
    fire3_weights.push_back(weights[8]);
    fire3_weights.push_back(weights[9]);
    vector<Array<float>> f3_bias;
    f3_bias.push_back(bias[7]);
    f3_bias.push_back(bias[8]);
    f3_bias.push_back(bias[9]);
    Array<float> Fire3 = FireModule(max2_out, fire3_weights, f3_bias);
    cout << endl << "Layer : Fire3" << "(" << Fire3.get_dim()[0] << "," << Fire3.get_dim()[1] << "," << Fire3.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire4_weights;
    fire4_weights.push_back(weights[10]);
    fire4_weights.push_back(weights[11]);
    fire4_weights.push_back(weights[12]);
    vector<Array<float>> f4_bias;
    f4_bias.push_back(bias[10]);
    f4_bias.push_back(bias[11]);
    f4_bias.push_back(bias[12]);
    Array<float> Fire4 = FireModule(Fire3, fire4_weights,f4_bias);
    cout << endl << "Layer : Fire4" << "(" << Fire4.get_dim()[0] << "," << Fire4.get_dim()[1] << "," << Fire4.get_dim()[2] << ")" << endl;

    int f4_h = Fire4.get_dim()[0];
    int f4_w = Fire4.get_dim()[1];
    int f4_depth = Fire4.get_dim()[2];
    Maxpool maxpool3 = Maxpool(f4_h, f4_w, f4_depth, 3, 2);
    Array<float> max3_out = maxpool3.HM_execute(Fire4, 2, f4_depth);
    cout << endl << "Layer : max3_out" << "(" << max3_out.get_dim()[0] << "," << max3_out.get_dim()[1] << "," << max3_out.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire5_weights;
    fire5_weights.push_back(weights[13]);
    fire5_weights.push_back(weights[14]);
    fire5_weights.push_back(weights[15]);
    vector<Array<float>> f5_bias;
    f5_bias.push_back(bias[13]);
    f5_bias.push_back(bias[14]);
    f5_bias.push_back(bias[15]);
    Array<float> Fire5 = FireModule(max3_out, fire5_weights, f5_bias);
    cout << endl << "Layer : Fire5" << "(" << Fire5.get_dim()[0] << "," << Fire5.get_dim()[1] << "," << Fire5.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire6_weights;
    fire6_weights.push_back(weights[16]);
    fire6_weights.push_back(weights[17]);
    fire6_weights.push_back(weights[18]);
    vector<Array<float>> f6_bias;
    f6_bias.push_back(bias[16]);
    f6_bias.push_back(bias[17]);
    f6_bias.push_back(bias[18]);
    Array<float> Fire6 = FireModule(Fire5, fire6_weights, f6_bias);
    cout << endl << "Layer : Fire6" << "(" << Fire6.get_dim()[0] << "," << Fire6.get_dim()[1] << "," << Fire6.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire7_weights;
    fire7_weights.push_back(weights[19]);
    fire7_weights.push_back(weights[20]);
    fire7_weights.push_back(weights[21]);
    vector<Array<float>> f7_bias;
    f7_bias.push_back(bias[19]);
    f7_bias.push_back(bias[20]);
    f7_bias.push_back(bias[21]);
    Array<float> Fire7 = FireModule(Fire6, fire7_weights, f7_bias);
    cout << endl << "Layer : Fire7" << "(" << Fire7.get_dim()[0] << "," << Fire7.get_dim()[1] << "," << Fire7.get_dim()[2] << ")" << endl;

    vector<vector<Array<float>>> fire8_weights;
    fire8_weights.push_back(weights[22]);
    fire8_weights.push_back(weights[23]);
    fire8_weights.push_back(weights[24]);
    vector<Array<float>> f8_bias;
    f8_bias.push_back(bias[22]);
    f8_bias.push_back(bias[23]);
    f8_bias.push_back(bias[24]);
    Array<float> Fire8 = FireModule(Fire7, fire8_weights, f8_bias);
    cout << endl << "Layer : Fire8" << "(" << Fire8.get_dim()[0] << "," << Fire8.get_dim()[1] << "," << Fire8.get_dim()[2] << ")" << endl;

    int f8_h = Fire8.get_dim()[0];
    int f8_w = Fire8.get_dim()[1];
    int f8_depth = Fire8.get_dim()[2];
    Convolution conv2 = Convolution(f8_h, f8_w, f8_depth, 1, 1, f8_depth);
    conv2.load_parameters(weights[25]);
    Array<float> conv2_out = conv2.HM_excute_Array_Depth(Fire8, bias[25], { 1,1 }, 0, false, f8_depth);
    cout << endl << "Layer : conv2_out" << "(" << conv2_out.get_dim()[0] << "," << conv2_out.get_dim()[1] << "," << conv2_out.get_dim()[2] << ")" << endl;

    int c2_h = conv2_out.get_dim()[0];
    int c2_w = conv2_out.get_dim()[1];
    int c2_d = conv2_out.get_dim()[2];
    Avgpool avg1 = Avgpool(c2_h, c2_w, c2_d, 13, 1);
    Array<float> av1_out = avg1.HM_execute(conv2_out, 1, c2_d);
    cout << endl << "Layer : av1_out" << "(" << av1_out.get_dim()[0] << "," << av1_out.get_dim()[1] << "," << av1_out.get_dim()[2] << ")" << endl;
    return av1_out;
}

void LoadWeightsForLayer(string layername, int layernum, int out_ch, int in_ch, int rows, int cols, vector<vector<Array<float>>>& weights, nlohmann::json& jo)
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    weights[layernum].clear();
    weights[layernum].resize(out_ch);
    for (size_t i = 0; i < out_ch; i++)
    {
        weights[layernum][i] = Array<float>({ rows, cols, in_ch });
    }

    for (size_t och = 0; och < out_ch; och++) // for each output channel
    {
        for (size_t ich = 0; ich < in_ch; ich++) // for each input channel
        {
            for (size_t r = 0; r < rows; r++)// for each row
            {
                for (size_t c = 0; c < cols; c++)// for each col
                {
                    double v = jo[layername][och][ich][r][c];
                    weights[layernum][och](r, c, ich) = v;
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "layer " << layernum << " weight fill time in milliseconds:" << duration.count() << endl;

}

void LoadBiasesForLayer(string layername, int layernum, int out_ch, vector<Array<float>>& biases, nlohmann::json& jo)
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    biases[layernum] = Array<float>({ out_ch });

    for (size_t och = 0; och < out_ch; och++) // for each output channel
    {
        double v = jo[layername][och];
        biases[layernum](och) = v;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "layer " << layernum << " bias fill time in milliseconds:" << duration.count() << endl;
}

bool LoadWeights(string weightfilename, vector<vector<Array<float>>>& weights, vector<Array<float>>& biases)
{
    std::ifstream t(weightfilename);
    assert(!!t);
    t.seekg(0, std::ios::end);
    size_t size = t.tellg();
    std::string buffer(size, ' ');
    t.seekg(0);
    t.read(&buffer[0], size);

    //vector<vector<Array<float>>>& weights
    //layers outch  colxrowxinch
    //dims: out_ch, in_ch, rows, cols


    using json = nlohmann::json;

    using namespace std::chrono;

    // Use auto keyword to avoid typing long
    // type definitions to get the timepoint
    // at this instant use function now()
    auto start = high_resolution_clock::now();

    auto jo = json::parse(buffer);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end - start);

    cout << "json parse time in seconds:" << duration.count() << endl;

    weights.clear();
    weights.resize(26);

    biases.clear();
    biases.resize(26);


    //features.0.weight : torch.Size([64, 3, 3, 3])
    LoadWeightsForLayer("features.0.weight", 0, 64, 3, 3, 3, weights, jo);
    //features.0.bias : torch.Size([64])
    LoadBiasesForLayer("features.0.bias", 0, 64, biases, jo);

    //features.3.squeeze.weight : torch.Size([16, 64, 1, 1])
    LoadWeightsForLayer("features.3.squeeze.weight", 1, 16, 64, 1, 1, weights, jo);
    //features.3.squeeze.bias : torch.Size([16])
    LoadBiasesForLayer("features.3.squeeze.bias", 1, 16, biases, jo);

    //features.3.expand1x1.weight : torch.Size([64, 16, 1, 1])
    LoadWeightsForLayer("features.3.expand1x1.weight", 2, 64, 16, 1, 1, weights, jo);
    //features.3.expand1x1.bias : torch.Size([64])
    LoadBiasesForLayer("features.3.expand1x1.bias", 2, 64, biases, jo);

    //features.3.expand3x3.weight : torch.Size([64, 16, 3, 3])
    LoadWeightsForLayer("features.3.expand3x3.weight", 3, 64, 16, 3, 3, weights, jo);
    //features.3.expand3x3.bias : torch.Size([64])
    LoadBiasesForLayer("features.3.expand3x3.bias", 3, 64, biases, jo);

    //features.4.squeeze.weight : torch.Size([16, 128, 1, 1])
    LoadWeightsForLayer("features.4.squeeze.weight", 4, 16, 128, 1, 1, weights, jo);
    //features.4.squeeze.bias : torch.Size([16])
    LoadBiasesForLayer("features.4.squeeze.bias", 4, 16, biases, jo);

    //features.4.expand1x1.weight : torch.Size([64, 16, 1, 1])
    LoadWeightsForLayer("features.4.expand1x1.weight", 5, 64, 16, 1, 1, weights, jo);
    //features.4.expand1x1.bias : torch.Size([64])
    LoadBiasesForLayer("features.4.expand1x1.bias", 5, 64, biases, jo);

    //features.4.expand3x3.weight : torch.Size([64, 16, 3, 3])
    LoadWeightsForLayer("features.4.expand3x3.weight", 6, 64, 16, 3, 3, weights, jo);
    //features.4.expand3x3.bias : torch.Size([64])
    LoadBiasesForLayer("features.4.expand3x3.bias", 6, 64, biases, jo);

    //features.6.squeeze.weight : torch.Size([32, 128, 1, 1])
    LoadWeightsForLayer("features.6.squeeze.weight", 7, 32, 128, 1, 1, weights, jo);
    //features.6.squeeze.bias : torch.Size([32])
    LoadBiasesForLayer("features.6.squeeze.bias", 7, 32, biases, jo);

    //features.6.expand1x1.weight : torch.Size([128, 32, 1, 1])
    LoadWeightsForLayer("features.6.expand1x1.weight", 8, 128, 32, 1, 1, weights, jo);
    //features.6.expand1x1.bias : torch.Size([128])
    LoadBiasesForLayer("features.6.expand1x1.bias", 8, 128, biases, jo);

    //features.6.expand3x3.weight : torch.Size([128, 32, 3, 3])
    LoadWeightsForLayer("features.6.expand3x3.weight", 9, 128, 32, 3, 3, weights, jo);
    //features.6.expand3x3.bias : torch.Size([128])
    LoadBiasesForLayer("features.6.expand3x3.bias", 9, 128, biases, jo);

    //features.7.squeeze.weight : torch.Size([32, 256, 1, 1])
    LoadWeightsForLayer("features.7.squeeze.weight", 10, 32, 256, 1, 1, weights, jo);
    //features.7.squeeze.bias : torch.Size([32])
    LoadBiasesForLayer("features.7.squeeze.bias", 10, 32, biases, jo);
    //features.7.expand1x1.weight : torch.Size([128, 32, 1, 1])

    LoadWeightsForLayer("features.7.expand1x1.weight", 11, 128, 32, 1, 1, weights, jo);
    //features.7.expand1x1.bias : torch.Size([128])
    LoadBiasesForLayer("features.7.expand1x1.bias", 11, 128, biases, jo);

    //features.7.expand3x3.weight : torch.Size([128, 32, 3, 3])
    LoadWeightsForLayer("features.7.expand3x3.weight", 12, 128, 32, 3, 3, weights, jo);
    //features.7.expand3x3.bias : torch.Size([128])
    LoadBiasesForLayer("features.7.expand3x3.bias", 12, 128, biases, jo);

    //features.9.squeeze.weight : torch.Size([48, 256, 1, 1])
    LoadWeightsForLayer("features.9.squeeze.weight", 13, 48, 256, 1, 1, weights, jo);
    //features.9.squeeze.bias : torch.Size([48])
    LoadBiasesForLayer("features.9.squeeze.bias", 13, 48, biases, jo);

    //features.9.expand1x1.weight : torch.Size([192, 48, 1, 1])
    LoadWeightsForLayer("features.9.expand1x1.weight", 14, 192, 48, 1, 1, weights, jo);
    //features.9.expand1x1.bias : torch.Size([192])
    LoadBiasesForLayer("features.9.expand1x1.bias", 14, 192, biases, jo);

    //features.9.expand3x3.weight : torch.Size([192, 48, 3, 3])
    LoadWeightsForLayer("features.9.expand3x3.weight", 15, 192, 48, 3, 3, weights, jo);
    //features.9.expand3x3.bias : torch.Size([192])
    LoadBiasesForLayer("features.9.expand3x3.bias", 15, 192, biases, jo);

    //features.10.squeeze.weight : torch.Size([48, 384, 1, 1])
    LoadWeightsForLayer("features.10.squeeze.weight", 16, 48, 384, 1, 1, weights, jo);
    //features.10.squeeze.bias : torch.Size([48])
    LoadBiasesForLayer("features.10.squeeze.bias", 16, 48, biases, jo);

    //features.10.expand1x1.weight : torch.Size([192, 48, 1, 1])
    LoadWeightsForLayer("features.10.expand1x1.weight", 17, 192, 48, 1, 1, weights, jo);
    //features.10.expand1x1.bias : torch.Size([192])
    LoadBiasesForLayer("features.10.expand1x1.bias", 17, 192, biases, jo);

    //features.10.expand3x3.weight : torch.Size([192, 48, 3, 3])
    LoadWeightsForLayer("features.10.expand3x3.weight", 18, 192, 48, 3, 3, weights, jo);
    //features.10.expand3x3.bias : torch.Size([192])
    LoadBiasesForLayer("features.10.expand3x3.bias", 18, 192, biases, jo);

    //features.11.squeeze.weight : torch.Size([64, 384, 1, 1])
    LoadWeightsForLayer("features.11.squeeze.weight", 19, 64, 384, 1, 1, weights, jo);
    //features.11.squeeze.bias : torch.Size([64])
    LoadBiasesForLayer("features.11.squeeze.bias", 19, 64, biases, jo);

    //features.11.expand1x1.weight : torch.Size([256, 64, 1, 1])
    LoadWeightsForLayer("features.11.expand1x1.weight", 20, 256, 64, 1, 1, weights, jo);
    //features.11.expand1x1.bias : torch.Size([256])
    LoadBiasesForLayer("features.11.expand1x1.bias", 20, 256, biases, jo);

    //features.11.expand3x3.weight : torch.Size([256, 64, 3, 3])
    LoadWeightsForLayer("features.11.expand3x3.weight", 21, 256, 64, 3, 3, weights, jo);
    //features.11.expand3x3.bias : torch.Size([256])
    LoadBiasesForLayer("features.11.expand3x3.bias", 21, 256, biases, jo);

    //features.12.squeeze.weight : torch.Size([64, 512, 1, 1])
    LoadWeightsForLayer("features.12.squeeze.weight", 22, 64, 512, 1, 1, weights, jo);
    //features.12.squeeze.bias : torch.Size([64])
    LoadBiasesForLayer("features.12.squeeze.bias", 22, 64, biases, jo);

    //features.12.expand1x1.weight : torch.Size([256, 64, 1, 1])
    LoadWeightsForLayer("features.12.expand1x1.weight", 23, 256, 64, 1, 1, weights, jo);
    //features.12.expand1x1.bias : torch.Size([256])
    LoadBiasesForLayer("features.12.expand1x1.bias", 23, 256, biases, jo);

    //features.12.expand3x3.weight : torch.Size([256, 64, 3, 3])
    LoadWeightsForLayer("features.12.expand3x3.weight", 24, 256, 64, 3, 3, weights, jo);
    //features.12.expand3x3.bias : torch.Size([256])
    LoadBiasesForLayer("features.12.expand3x3.bias", 24, 256, biases, jo);

    //classifier.1.weight : torch.Size([3, 512, 1, 1])
    LoadWeightsForLayer("classifier.1.weight", 25, 3, 512, 1, 1, weights, jo);
    //classifier.1.bias : torch.Size([3])
    LoadBiasesForLayer("classifier.1.bias", 25, 3, biases, jo);

    return false;
}

void main()
{
    Mat image = imread("F:/graduation_project/CPP Model/bump.jpg");
    int down_width = 224;
    int down_height = 224;

    //Mat nnew_image;
    Mat different_Channels[3];
    split(image, different_Channels);
    
    Mat b = different_Channels[0];//loading blue channels//
    Mat g = different_Channels[1];//loading green channels//
    Mat r = different_Channels[2];//loading red channels//  
    
    Mat r_image;
    
    resize(r, r_image, Size(down_width, down_height), INTER_LINEAR);
        
    //cv::imshow("Grey Copied Image", image);
    //waitKey(0);
    
    r_image.convertTo(r_image, CV_32F);
    r_image = r_image.t();
    std::vector<float> r_vec((float*)r_image.data, (float*)r_image.data + r_image.rows * r_image.cols);
    
    Mat g_image;
    resize(g, g_image, Size(down_width, down_height), INTER_LINEAR);
    g_image.convertTo(g_image, CV_32F);
    g_image = g_image.t();
    std::vector<float> g_vec((float*)g_image.data, (float*)g_image.data + g_image.rows * g_image.cols);
    
    Mat b_image;
    resize(b, b_image, Size(down_width, down_height), INTER_LINEAR);
    b_image.convertTo(b_image, CV_32F);
    b_image = b_image.t();
    std::vector<float> b_vec((float*)b_image.data, (float*)b_image.data + b_image.rows * b_image.cols);
    
    //means: [0.485, 0.456, 0.406]
    //stdevs : [0.229, 0.224, 0.225]
    for (float& x_ : r_vec) 
    {
        x_ = ((x_/255) - 0.485) / 0.229;
    }
    for (float& x_ : g_vec)
    {
        x_ = ((x_ / 255) - 0.456) / 0.224;
    }
    for (float& x_ : b_vec)
    {
        x_ = ((x_ / 255) - 0.406) / 0.225;
    }
    vector<float> image_vec;
    image_vec = r_vec;
    image_vec.insert(image_vec.end(), g_vec.begin(), g_vec.end());
    image_vec.insert(image_vec.end(), b_vec.begin(), b_vec.end());
    
    vector<int> dim_img({ down_height,down_width,3 });
    Array<float> img(dim_img);
    img.fill_data(image_vec);

    vector<float> weights;
    {
        weights.push_back(1.0);
        weights.push_back(1.0);
        weights.push_back(1.0);
        weights.push_back(0.0);
        weights.push_back(0.0);
        weights.push_back(0.0);
        weights.push_back(-1.0);
        weights.push_back(-1.0);
        weights.push_back(-1.0);

        weights.push_back(1.0);
        weights.push_back(1.0);
        weights.push_back(1.0);
        weights.push_back(0.0);
        weights.push_back(0.0);
        weights.push_back(0.0);
        weights.push_back(-1.0);
        weights.push_back(-1.0);
        weights.push_back(-1.0);
        
        weights.push_back(1.0);
        weights.push_back(1.0);
        weights.push_back(1.0);
        weights.push_back(0.0);
        weights.push_back(0.0);
        weights.push_back(0.0);
        weights.push_back(-1.0);
        weights.push_back(-1.0);
        weights.push_back(-1.0); 
    }

    vector<float> input;//3*3*3
    {
        input.push_back(1.0);
        input.push_back(2.0);
        input.push_back(3.0);
        input.push_back(4.0);
        input.push_back(5.0);
        input.push_back(6.0);
        input.push_back(7.0);
        input.push_back(8.0);
        input.push_back(9.0);
        
        input.push_back(1.0);
        input.push_back(2.0);
        input.push_back(3.0);
        input.push_back(4.0);
        input.push_back(5.0);
        input.push_back(6.0);
        input.push_back(7.0);
        input.push_back(8.0);
        input.push_back(9.0);
        
        input.push_back(1.0);
        input.push_back(2.0);
        input.push_back(3.0);
        input.push_back(4.0);
        input.push_back(5.0);
        input.push_back(6.0);
        input.push_back(7.0);
        input.push_back(8.0);
        input.push_back(9.0);
    }
    vector<float> vv_input= input;
    vv_input.insert(vv_input.end(), input.begin(), input.end());
    vv_input.insert(vv_input.end(), input.begin(), input.end());

    vector<float> vvv_input = vv_input;
    vvv_input.insert(vvv_input.end(), vv_input.begin(), vv_input.end());
    vvv_input.insert(vvv_input.end(), vv_input.begin(), vv_input.end());

    int in = 0;
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            cout << vv_input[in] << " ";
            in++;
        }
        cout << endl;
    }
    vector<int> dim({ 3,3,3 });
    Array<float> w(dim);

    vector<int> dim_in({ 9,9,3 });
    Array<float> v_in(dim_in);

    v_in.fill_data(vvv_input);
    w.fill_data(weights);
    vector<Array<float>> wei;
    wei.push_back(w);
    Array<float> biass({ 3 });
    biass.fill_data({ 1,1,1 });
    
    Convolution conv1 = Convolution(9, 9, 3, 3, 3, 3);
    Avgpool avg1 = Avgpool(9, 9, 3, 3, 3);
    Maxpool max1 = Maxpool(9, 9, 3, 3, 3);
    conv1.load_parameters(wei);
    
    vector<Array<float>> ww;
    ww.push_back(w);
    ww.push_back(w);
    ww.push_back(w);

    vector<vector<Array<float>>> www;
    www.push_back(ww);
    www.push_back(ww);
    www.push_back(ww);
    
    //-------------------------test convolution, average, and maxpool layers----------------------- 
    Array<float> conv1_out = conv1.HM_excute_Array_Depth(v_in, biass, { 1,1 }, 0, false, 3);
    Array<float> avg1_out = avg1.HM_execute(v_in, 3, 3);
    Array<float> max1_out = max1.HM_execute(v_in, 3, 3);
    // 
    //
    // 
    // 
    //  
    //____________________load weigts______________________
    vector<vector<Array<float>>> Model_Weigts;
    vector<Array<float>> bias;
    LoadWeights("F:/graduation_project/CPP Model/layer/outmodel.json", Model_Weigts, bias);

    SqueezeNetV1_1(img, Model_Weigts, bias, 3);

    //------------------------------THE END------------------------------//
}