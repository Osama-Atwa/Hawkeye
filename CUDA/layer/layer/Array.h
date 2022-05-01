#pragma once
#include <iostream>
using namespace std;

template<typename T> class Array{
private:
    T * data;
    int length;
public:
    Array(T arr[], int l);
};
template<typename T> Array<T>::Array(T arr[], int l)
{
    data = new T[l];
    length = l;
    for (int i - 0; i < length; i++)
    {
        data[i] = arr[i];
    }
}
