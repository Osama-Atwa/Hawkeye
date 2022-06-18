//int main()
//{
//    // Read the image file as
//    // imread("default.jpg");
//    Mat image = imread("dog.jpg");
//    int down_width = 112;
//    int down_height = 112;
//    //Mat nnew_image;
//    Mat different_Channels[3];
//    split(image, different_Channels);
//
//    Mat b = different_Channels[0];//loading blue channels//
//    Mat g = different_Channels[1];//loading green channels//
//    Mat r = different_Channels[2];//loading red channels//  
//
//    Mat r_image;
//
//    resize(r, r_image, Size(down_width, down_height), INTER_LINEAR);
//    
//    //cv::imshow("Grey Copied Image", image);
//    //waitKey(0);
//
//    r_image.convertTo(r_image, CV_32F);
//    std::vector<float> r_vec((float*)r_image.data, (float*)r_image.data + r_image.rows * r_image.cols);
//
//    Mat g_image;
//    resize(g, g_image, Size(down_width, down_height), INTER_LINEAR);
//    g_image.convertTo(g_image, CV_32F);
//    std::vector<float> g_vec((float*)g_image.data, (float*)g_image.data + g_image.rows * g_image.cols);
//
//    Mat b_image;
//    resize(b, b_image, Size(down_width, down_height), INTER_LINEAR);
//    b_image.convertTo(b_image, CV_32F);
//    std::vector<float> b_vec((float*)b_image.data, (float*)b_image.data + b_image.rows * b_image.cols);
//
//    vector<float> image_vec;
//    image_vec = b_vec;
//    image_vec.insert(image_vec.end(), g_vec.begin(), g_vec.end());
//    image_vec.insert(image_vec.end(), r_vec.begin(), r_vec.end());
//
//    Convolution conv2d = Convolution(112, 112, 3, 3, 3, 3, 1, 1);
//    Avgpool avgpool = Avgpool(112, 112, 3, 3, 1, 0);
//    Maxpool maxpool = Maxpool(112, 112, 3, 3, 1, 0);
//    vector<float> weights;
//    weights.push_back(1.0);
//    weights.push_back(1.0);
//    weights.push_back(1.0);
//    weights.push_back(0.0);
//    weights.push_back(0.0);
//    weights.push_back(0.0);
//    weights.push_back(-1.0);
//    weights.push_back(-1.0);
//    weights.push_back(-1.0);
//    weights.push_back(1.0);
//    weights.push_back(1.0);
//    weights.push_back(1.0);
//    weights.push_back(0.0);
//    weights.push_back(0.0);
//    weights.push_back(0.0);
//    weights.push_back(-1.0);
//    weights.push_back(-1.0);
//    weights.push_back(-1.0);
//    weights.push_back(1.0);
//    weights.push_back(1.0);
//    weights.push_back(1.0);
//    weights.push_back(0.0);
//    weights.push_back(0.0);
//    weights.push_back(0.0);
//    weights.push_back(-1.0);
//    weights.push_back(-1.0);
//    weights.push_back(-1.0);
//    
//    vector<float> input;
//    input.push_back(1.0);
//    input.push_back(4.0);
//    input.push_back(7.0);
//    input.push_back(2.0);
//    input.push_back(5.0);
//    input.push_back(8.0);
//    input.push_back(3.0);
//    input.push_back(6.0);
//    input.push_back(9.0);
//
//    input.push_back(1.0);
//    input.push_back(4.0);
//    input.push_back(7.0);
//    input.push_back(2.0);
//    input.push_back(5.0);
//    input.push_back(8.0);
//    input.push_back(3.0);
//    input.push_back(6.0);
//    input.push_back(9.0);
//
//    input.push_back(1.0);
//    input.push_back(4.0);
//    input.push_back(7.0);
//    input.push_back(2.0);
//    input.push_back(5.0);
//    input.push_back(8.0);
//    input.push_back(3.0);
//    input.push_back(6.0);
//    input.push_back(9.0);
//
//
//    vector<int> dim({ 3,3,3 });
//    vector<int> dim_img({ 112,112,3 });
//    vector<int> i_dim({ 3,3,3 });
//
//    Array<float> w(dim);
//    Array<float> img(dim_img);
//    Array<float> input_(i_dim);
//
//    w.fill_data(weights);
//    vector<Array<float>> ww;
//    ww.push_back(w);
//    ww.push_back(w);
//    ww.push_back(w);
//    img.fill_data(image_vec);
//    input_.fill_data(input);
//    conv2d.load_parameters(ww);
//    //vector<vector<float>> out = conv2d.HM_excute(input_2d, 1);
//
//    Array<float> out_img = conv2d.HM_excute_Array_Depth(img, 1,0,false,3);
//    Array<float> AVG_image = avgpool.HM_execute(img, 1, 3);
//    Array<float> MAX_image = maxpool.HM_execute(img, 1, 3);
//
//    //vector<vector<float>> veeeeee({ {1.0,2.0,3.0},{4.0,5.0,6.0},{4.0,5.0,6.0} });
//    //vector<float> vv = conv2d.Flatten(veeeeee);
//    //vector<vector<float>> vec;
//    // 
//    // 
//    //vector<vector<float>> input_2d = convert(input);
//    /*vector<float> f_output;
//    for (int i = 0; i < out.size(); i++)
//    {
//        for (int j = 0; j < out[0].size(); j++)
//        {
//            f_output.push_back(out[i][j]);
//        }
//    }*/
//    //const vector<float>& vec1 = output.get_data();
//
//    //uint8_t** greyArr = setupHMM(vec, 112, 112);
//    //uchar* data = convert_char(output.get_data());
//
//
//    //cv::Mat greyImgForArrCopy = cv::Mat(112, 112, CV_32FC1, (float*)f_output.data(),cv::Mat::AUTO_STEP);
//
//    cv::Mat greyImgForArrCopy = cv::Mat(110, 110, CV_32FC1, (float*)out_img.get_data().data(), cv::Mat::AUTO_STEP);
//
//
//    //nnew_image.data = data;
//    // 
//    greyImgForArrCopy.convertTo(greyImgForArrCopy, CV_8U);
//    //cout << greyImgForArrCopy;
//
//    cv::imshow("Grey Copied Image", greyImgForArrCopy);
//    
//    //if (output_img.empty()) {
//    //    cout << "Image File "
//    //        << "Not Found" << endl;
//
//    //    // wait for any key press
//    //    cin.get();
//    //    return -1;
//    //}
//
//    //// Show Image inside a window with
//    //// the name provided
//    //imshow("Window Name", greyImg);
//
//    //// Wait for any keystroke
//    /*for (int i = 0; i < input_2d.size(); i++)
//    {
//        for (int j = 0; j < input_2d[i].size(); j++)
//        {
//            std::cout << input_2d[i][j] << " ";
//        }
//        std::cout << endl;
//    }*/
//
//    /*for (int i = 0; i < vec.size(); i++)
//    {
//        for (int j = 0; j < vec[i].size(); j++)
//        {
//            std::cout << vec[i][j]<<" ";
//        }
//        std::cout << endl;
//    }*/
//    waitKey(0);
//    return 0;
//}