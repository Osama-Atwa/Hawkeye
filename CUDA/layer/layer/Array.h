#pragma once
#include <iostream>
//#include <complex>
#include <vector>

using namespace std;
//todo: proxies

template<typename T>
class Array{
private:
    vector<T> data;
    vector<int> dim;
public:
    Array(vector<int> dimemsions);
};

