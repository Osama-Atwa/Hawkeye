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
    int i_w = 3;
    int i_h = 3;
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
// Driver code
int main()
{
    // Read the image file as
    // imread("default.jpg");
    Mat image = imread("C:/Users/ashra/Desktop/dog.jpg", IMREAD_GRAYSCALE);
    int down_width = 112;
    int down_height = 112;
    //Mat nnew_image;
    Mat new_image;
    resize(image, new_image, Size(down_width, down_height), INTER_LINEAR);
    new_image.convertTo(new_image, CV_32F);
    std::vector<float> c((float*)new_image.data, (float*)new_image.data + new_image.rows * new_image.cols);

    Convolution conv2d = Convolution(112, 112, 1, 3, 3, 1, 1, 1);
    
    vector<float> input;
    input.push_back(1.0);
    input.push_back(2.0);
    input.push_back(3.0);
    input.push_back(4.0);
    input.push_back(5.0);
    input.push_back(6.0);
    input.push_back(7.0);
    input.push_back(8.0);
    input.push_back(9.0);


    vector<float> weights;
    weights.push_back(1.0 );
    weights.push_back(1.0 );
    weights.push_back(1.0 );
    weights.push_back(0.0);
    weights.push_back(0.0);
    weights.push_back(0.0);
    weights.push_back(-1.0);
    weights.push_back(-1.0);
    weights.push_back(-1.0);

    vector<int> dim({ 3,3 });
    vector<int> dim_img({ new_image.rows,new_image.cols });

    Array<float> w(dim);
    Array<float> img(dim_img);
    Array<float> output(dim_img);
    Array<float> input_(dim);

    input_.fill_data(input);
    w.fill_data(weights);
    img.fill_data(c);

    conv2d.load_parameters(w);
    conv2d.execute(img, output);
    //vector<vector<float>> vec = convert(output.get_data());
    //vector<vector<float>> input_2d = convert(input);

    const vector<float>& vec1 = output.get_data();

    //uint8_t** greyArr = setupHMM(vec, 112, 112);
    //uchar* data = convert_char(output.get_data());
    float arr[] = { 1,2,3,4,5,6,7,8,9 };

    cv::Mat greyImgForArrCopy = cv::Mat(112, 112, CV_32FC1, (float*)vec1.data(),cv::Mat::AUTO_STEP);

    
    //nnew_image.data = data;
    // 
    greyImgForArrCopy.convertTo(greyImgForArrCopy, CV_8U);
    //cout << greyImgForArrCopy;

    cv::imshow("Grey Copied Image", greyImgForArrCopy);

    //if (output_img.empty()) {
    //    cout << "Image File "
    //        << "Not Found" << endl;

    //    // wait for any key press
    //    cin.get();
    //    return -1;
    //}

    //// Show Image inside a window with
    //// the name provided
    //imshow("Window Name", greyImg);

    //// Wait for any keystroke
    /*for (int i = 0; i < input_2d.size(); i++)
    {
        for (int j = 0; j < input_2d[i].size(); j++)
        {
            std::cout << input_2d[i][j] << " ";
        }
        std::cout << endl;
    }*/

    /*for (int i = 0; i < vec.size(); i++)
    {
        for (int j = 0; j < vec[i].size(); j++)
        {
        	std::cout << vec[i][j]<<" ";
        }
        std::cout << endl;
    }*/
    waitKey(0);
    return 0;
}
//int main() {
//	Convolution conv2d = Convolution(3, 3, 1, 3, 3, 1, 1, 1);
//	Avgpool Avgpool2d = Avgpool(3, 3, 1, 3, 1, 0);
//	Maxpool maxpool = Maxpool(3, 3, 1, 3, 1, 0);
//
//	vector<float> weights;
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//	weights.push_back(1.0);
//
//	vector<vector<float>> v = Avgpool2d.convert(weights);
//	vector<vector<float>> result = Avgpool2d.vector_padding(v, 2, true);
//	for (int i = 0; i < result.size(); i++)
//	{
//		for (int j = 0; j < result[i].size(); j++)
//		{
//			std::cout << result[i][j]<<" ";
//		}
//		std::cout << endl;
//	}
//	vector<float> input;
//	input.push_back(1.0);
//	input.push_back(4.0);
//	input.push_back(7.0);
//	input.push_back(2.0);
//	input.push_back(5.0);
//	input.push_back(8.0);
//	input.push_back(3.0);
//	input.push_back(6.0);
//	input.push_back(9.0);
//	vector<float> output;
//
//	//conv2d.load_parameters(weights);
//	//conv2d.execute(input, output);
//	//
//	vector<vector<float>> vv = convert(input);
//	std::cout << vv.size() << vv[0].size() << endl;
//	vector<vector<float>> vout = Avgpool2d.mean_filter(v,1);
//	std::cout << vout.size() <<"  " << vout[0].size() <<"  "<< 1 << endl;
//	//vector<vector<float>> vout = maxpool.max_filter(v, 1);
//
//	//Avgpool2d.load_parameters(weights);
//	//Avgpool2d.execute(input, output);
//
//	vector<float> input2{ 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 };
//	vector<float> output2;
//	//Avgpool Avgpool2d2 = Avgpool(4, 4, 1, 3, 1, 0);
//	//Avgpool2d2.execute(input2, output2);
//	Maxpool Maxpool2d2 = Maxpool(4, 4, 1, 3, 1, 0);
//	//Maxpool2d2.execute(input2, output2);
//	//{
//	//	return 0;
//	//	af::array signal = constant(1.f, 3, 3);
//	//	signal(0,0) = 1; signal(0, 1) = 2; signal(0, 2) = 3; signal(1, 0) = 4; signal(1, 1) = 5; signal(1, 2) = 6; signal(2, 0) = 7; signal(2, 1) = 8; signal(2, 2) = 9;
//	//	af::array filter = constant(0 , 3, 3);
//	//	//filter(0, 0) = 1; filter(0, 1) = 2; filter(0, 2) = 3; filter(1, 0) = 4; filter(1, 1) = 5; filter(1, 2) = 6; filter(2, 0) = 7; filter(2, 1) = 8; filter(2, 2) = 9;
//	//	//filter(0, 0) = -1; filter(0, 1) = -2; filter(0, 2) = -1; filter(1, 0) = 0; filter(1, 1) = 0; filter(1, 2) = 0; filter(2, 0) = 1; filter(2, 1) = 2; filter(2, 2) = 1;
//	//	filter(1, 1) = 1;
//	//	filter(2, 1) = 1;
//	//	dim4 strides(1, 1), dilation(0, 0, 0, 0);
//	//	dim4 padding(1, 1, 1, 1);
//
//	//	af::array convolved = convolve2NN(signal, filter, strides, padding, dilation);
//	//	af::print("signal", signal);
//	//	af::print("filter", filter);
//	//	af::print("convolved", convolved);
//	//	return 0;
//	//}
//
//	for (int i = 0; i < vout.size(); i++)
//	{
//		for (int j = 0; j < vout[i].size(); j++)
//		{
//			std::cout << vout[i][j];
//		}
//		std::cout << endl;
//	}
//	return 0;
//}