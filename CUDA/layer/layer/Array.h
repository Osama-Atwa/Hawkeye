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
    void fill_data(vector<T> arr);
    vector<T> get_data();
    Array(T** arr, int dim0, int dim1);
    Array(T*** arr, int dim0, int dim1, int dim2);
    Array(T**** arr, int dim0, int dim1, int dim2, int dim3);
    T operator()(int i0, int i1 = -1, int i2 = -1, int i3 = -1);
};

template<typename T>
Array<T>::Array(vector<int> dimemsions)
{
    dim = dimemsions;
    int nelem = 0;
    for (size_t i = 0; i < dimemsions.size(); i++)
    {
        nelem += dimemsions[i];
    }
    data.resize(nelem);
}

template<typename T> void Array<T>::fill_data(vector<T> arr)
{
    data = arr;
}
template<typename T> vector<T> Array<T>::get_data()
{
    return data;
}
template<typename T> Array<T>::Array(T** arr, int dim0, int dim1)
{
   
}

template<typename T> Array<T>::Array(T*** arr, int dim0, int dim1, int dim2)
{

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0; k < dim2; k++)
            {
                data.push_back(arr[i][j][k]);

            }
        }
    }
}

template<typename T> Array<T>::Array(T**** arr, int dim0, int dim1, int dim2, int dim3)
{

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0; k < dim2; k++)
            {
                for (int l = 0; l < dim3; l++)
                {
                    data.push_back(arr[i][j][k][l]);
                }
            }
        }
    }
}

template<typename T> T Array<T>::operator()(int i0, int i1, int i2, int i3)
{
    if (i1 == -1)
        return data[i0];
    else if (i2 == -1)
    {
        int x = i0* dim[1] + i1 ;
        return data[x];
    }
    else if (i3 == -1 && i2 != -1)
    {
        int x = i0 * (dim[0] * dim[1]) + i1 * dim[1] + i2;
        return data[x];

    }
    else if (i3 != -1)
    {
        int x = i0 * (dim[0] * dim[1] * dim[2]) + i1 * (dim[0] * dim[1]) + i2 * dim[2] + i3;
        return data[x];
    }
        
}
