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

