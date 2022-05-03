#include "Array.h"

template<typename T>
Array<T>::Array(vector<int> dimemsions)
{
	dim = dimemsions;
	int nelem = 0;
	for (size_t i = 0; i < disjunction.size(); i++)
	{
		nelem += dimemsions[i];
	}
	data.resize(nelem);
}

template<typename T> Array<T>::Array(T arr[], int l)
{
    data = new T[l];
    for (int i - 0; i < length; i++)
    {
        data.push_back(arr[i]);
    }
}

template<typename T> Array<T>::Array(T** arr, int dim0, int dim1)
{
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++)
        {
            data.push_back(arr[i][j]);

        }
    }
}

template<typename T> Array<T>::Array(T*** arr, int dim0, int dim1, int dim2)
{

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0k < dim2; k++)
            {
                data.push_back(arr[i][j]);

            }
        }
    }
}

template<typename T> Array<T>::Array(T**** arr, int dim0, int dim1, int dim2, int dim3)
{

    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0k < dim2; k++)
            {
                for (int l = 0; l < dim3; l++)
                {
                    data.push_back(arr[i][j]);
                }
            }
        }
    }
}

