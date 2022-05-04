#pragma once
#include <iostream>
//#include <complex>
#include <vector>

using namespace std;


template<typename T>
class Array{
private:
    vector<T> data;
    vector<int> dim;
public:
    Array(vector<int> dimemsions);
    Array(T arr[], int l);
    Array(T** arr, int dim0, int dim1);
    Array(T*** arr, int dim0, int dim1, int dim2);
    Array(T**** arr, int dim0, int dim1, int dim2, int dim3);

};

