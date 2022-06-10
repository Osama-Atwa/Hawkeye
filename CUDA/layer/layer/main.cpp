#define AF_DEBUG
#define AF_CPU
#include <iostream>;
#include "Convolution.h";
#include "Avgpool.h";
#include "Maxpool.h";
#include <opencv2/opencv.hpp>
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
Array<float> FireModule(Array<float> input, vector<vector<Array<float>>> weights)
{
    int i_h = input.get_dim()[0];
    int i_w = input.get_dim()[1];
    int i_depth = input.get_dim()[2];

    Convolution conv2d_1 = Convolution(i_h, i_w, i_depth, 1, 1, i_depth);
    conv2d_1.load_parameters(weights[0]);
    Array<float> squeeze = conv2d_1.HM_excute_Array_Depth(input, {1, 1}, 1, TRUE, i_depth);

    int s_h = squeeze.get_dim()[0];
    int s_w = squeeze.get_dim()[1];
    int s_depth = squeeze.get_dim()[2];

    Convolution conv2d_2 = Convolution(s_h, s_w, s_depth, 1, 1, s_depth);
    conv2d_2.load_parameters(weights[1]);
    Array<float> expand1 = conv2d_2.HM_excute_Array_Depth(squeeze, { 1, 1 }, 1, TRUE, s_depth);

    Convolution conv2d_3 = Convolution(s_h, s_w, s_depth, 3, 3, s_depth);
    conv2d_3.load_parameters(weights[2]);
    Array<float> expand2 = conv2d_3.HM_excute_Array_Depth(squeeze, { 1, 1 }, 1, TRUE, squeeze.get_dim()[2]);
    
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
void SqueezeNetV1_1(Array<float> v_input,vector<vector<Array<float>>> weights ,int nb_classes)
{
    int i_h = v_input.get_dim()[0];
    int i_w = v_input.get_dim()[1];
    int i_depth = v_input.get_dim()[2];
    Convolution conv1 = Convolution(i_h, i_w, i_depth, 3, 3, i_depth);
    conv1.load_parameters(weights[0]);
    Array<float> conv1_out = conv1.HM_excute_Array_Depth(v_input, { 2,2 }, 0, FALSE, i_depth);

    int c1_h = conv1_out.get_dim()[0];
    int c1_w = conv1_out.get_dim()[1];
    int c1_depth = conv1_out.get_dim()[2];
    Maxpool maxpool1 = Maxpool(c1_h, c1_w, c1_depth, 3, 2);
    Array<float> max1_out = maxpool1.HM_execute(conv1_out, 2, c1_depth);

    vector<vector<Array<float>>> fire1_weights;
    fire1_weights.push_back(weights[1]);
    fire1_weights.push_back(weights[2]);
    fire1_weights.push_back(weights[3]);
    Array<float> Fire1 = FireModule(max1_out, fire1_weights);

    vector<vector<Array<float>>> fire2_weights;
    fire2_weights.push_back(weights[4]);
    fire2_weights.push_back(weights[5]);
    fire2_weights.push_back(weights[6]);
    Array<float> Fire2 = FireModule(Fire1, fire2_weights);

    int f2_h = Fire2.get_dim()[0];
    int f2_w = Fire2.get_dim()[1];
    int f2_depth = Fire2.get_dim()[2];
    Maxpool maxpool2 = Maxpool(f2_h, f2_w, f2_depth, 3, 2);
    Array<float> max2_out = maxpool2.HM_execute(Fire2, 2, f2_depth);

    vector<vector<Array<float>>> fire3_weights;
    fire3_weights.push_back(weights[7]);
    fire3_weights.push_back(weights[8]);
    fire3_weights.push_back(weights[9]);
    Array<float> Fire3 = FireModule(max2_out, fire3_weights);

    vector<vector<Array<float>>> fire4_weights;
    fire4_weights.push_back(weights[10]);
    fire4_weights.push_back(weights[11]);
    fire4_weights.push_back(weights[12]);
    Array<float> Fire4 = FireModule(Fire3, fire4_weights);

    int f4_h = Fire4.get_dim()[0];
    int f4_w = Fire4.get_dim()[1];
    int f4_depth = Fire4.get_dim()[2];
    Maxpool maxpool3 = Maxpool(f4_h, f4_w, f4_depth, 3, 2);
    Array<float> max3_out = maxpool3.HM_execute(Fire4, 2, f4_depth);

    vector<vector<Array<float>>> fire5_weights;
    fire5_weights.push_back(weights[13]);
    fire5_weights.push_back(weights[14]);
    fire5_weights.push_back(weights[15]);
    Array<float> Fire5 = FireModule(max3_out, fire5_weights);

    vector<vector<Array<float>>> fire6_weights;
    fire6_weights.push_back(weights[16]);
    fire6_weights.push_back(weights[17]);
    fire6_weights.push_back(weights[18]);
    Array<float> Fire6 = FireModule(Fire5, fire6_weights);

    vector<vector<Array<float>>> fire7_weights;
    fire7_weights.push_back(weights[19]);
    fire7_weights.push_back(weights[20]);
    fire7_weights.push_back(weights[21]);
    Array<float> Fire7 = FireModule(Fire6, fire7_weights);

    vector<vector<Array<float>>> fire8_weights;
    fire8_weights.push_back(weights[22]);
    fire8_weights.push_back(weights[23]);
    fire8_weights.push_back(weights[24]);
    Array<float> Fire8 = FireModule(Fire7, fire8_weights);

    int f8_h = Fire8.get_dim()[0];
    int f8_w = Fire8.get_dim()[1];
    int f8_depth = Fire8.get_dim()[2];
    Convolution conv2 = Convolution(f8_h, f8_w, f8_depth, 1, 1, f8_depth);
    conv2.load_parameters(weights[25]);
    Array<float> conv2_out = conv2.HM_excute_Array_Depth(Fire8, { 1,1 }, 1, FALSE, f8_depth);

}

void main() {
    Mat image = imread("dog.jpg");
    int down_width = 112;
    int down_height = 112;

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
    std::vector<float> r_vec((float*)r_image.data, (float*)r_image.data + r_image.rows * r_image.cols);
    
    Mat g_image;
    resize(g, g_image, Size(down_width, down_height), INTER_LINEAR);
    g_image.convertTo(g_image, CV_32F);
    std::vector<float> g_vec((float*)g_image.data, (float*)g_image.data + g_image.rows * g_image.cols);
    
    Mat b_image;
    resize(b, b_image, Size(down_width, down_height), INTER_LINEAR);
    b_image.convertTo(b_image, CV_32F);
    std::vector<float> b_vec((float*)b_image.data, (float*)b_image.data + b_image.rows * b_image.cols);
    
    vector<float> image_vec;
    image_vec = b_vec;
    image_vec.insert(image_vec.end(), g_vec.begin(), g_vec.end());
    image_vec.insert(image_vec.end(), r_vec.begin(), r_vec.end());
    
    vector<int> dim_img({ 112,112,3 });
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

    vector<int> dim({ 3,3,3 });
    Array<float> w(dim);

    w.fill_data(weights);
    vector<Array<float>> ww;
    ww.push_back(w);
    ww.push_back(w);
    ww.push_back(w);

    vector<vector<Array<float>>> www;
    www.push_back(ww);
    www.push_back(ww);
    www.push_back(ww);

    
    SqueezeNetV1_1(img, www, 3);

}