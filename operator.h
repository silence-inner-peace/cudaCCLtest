#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda.h>

#include <gdal_priv.h>
#include <cpl_conv.h>
#include "GDALRead.h"
#include "GDALWrite.h"
#include "basestruct.h"
#include "timer.h"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
using namespace std;
#define CLASS_MAX 256

void createPatchMap(int* img_data, int* labelMap, int *pixNum, int *perimeter, int height, int width, std::map<int, Patch>& mapPatch, int data_start, int data_end, int task_start, int task_end);
void mergePatchMap(std::map<int, Patch>& mapPatch1, std::map<int, Patch>& mapPatch2);

void outputPatchClass(PClass *pClass, int &_pClscount);

template <class T>
void covertInt(T* img_data, int* AllDataHost, int size)
{
	for (int i = 0; i < size; i++)
	{
		AllDataHost[i] = (int)img_data[i];
	}
}

int findMerge(int width, int BGvalue, int* Meg, int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel);
void MergePatch( int* Meg, int Meg_count, UF &Quf);
void outputPatchToFile(std::map<int, Patch>& mapPatch, std::string Name);
void outputPatchToFile(std::map<int, Patch>& mapPatch, UF &Quf, std::string Name);
void MergePatchArray(int width, int blockNum, int BGvalue, int* mergeStructArray,UF &Quf);
