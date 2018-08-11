#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda.h>

#include <gdal_priv.h>
#include <cpl_conv.h>
#include "GDALRead.h"
#include "GDALWrite.h"
#include "basestruct.h"
#include "operator.h"
#include "timer.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
using namespace std;


double LineCCLNoSplit(int *allData, int width, int height, dim3 blockSize, dim3 gridSize, int * labelMap, int *pixNum, int *perimeter, int nodata);
// double LineSplitCCL2(int *allData, int width, int data_height, int task_height, dim3 blockSize, dim3 gridSize, int *h_labelMap, int *pixNum, int *perimeter, int nodata, int labelStart, int data_start, int data_end, int task_start, int task_end);
// double LineSplitCCL(int *allData, int width,  int data_height, int task_height, dim3 blockSize, dim3 gridSize, int *h_labelMap, int *pixNum, int *perimeter, int nodata, int labelStart, int data_start, int data_end, int task_start, int task_end);


double LineSplitCCL(int** h_subDataNextBlock, dataBlock &dataBlockNext, CGDALRead* pread, 
	int *allData, int width, int data_height, int task_height, dim3 blockSize, dim3 gridSize, 
	int *h_labelMap, int *pixNum, int *perimeter, int nodata, int labelStart, int data_start, int data_end, int task_start, int task_end);
double LineSplitCCL2(int** h_subDataNextBlock, dataBlock* dataBlockArray, int iBlock, CGDALRead* pread, 
	int *allData, int width, int data_height, int task_height, dim3 blockSize, dim3 gridSize, 
	int *h_labelMap, int *pixNum, int *perimeter, int nodata, int labelStart, int data_start, int data_end, int task_start, int task_end);


void loadBlockData(int width, int data_height, int data_start, int** h_subData, CGDALRead* pread)
{
	size_t nBytes_data = data_height * width * sizeof(int);
    checkCudaErrors(cudaHostAlloc((void **)h_subData, nBytes_data, cudaHostAllocDefault));
    memset(*h_subData, 0, data_height * width);
    switch (pread->datatype())
	{
		case GDT_Byte:
		{
				pread->readDataBlock<unsigned char>(width,data_height,0,data_start, *h_subData);
				break;
		}
		case GDT_UInt16:
		{
				pread->readDataBlock<unsigned short>(width,data_height,0,data_start,*h_subData);
				break;
		}
		case GDT_Int16:
		{
				pread->readDataBlock<short>(width,data_height,0,data_start,*h_subData);
				break;
		}
		case GDT_UInt32:
		{
				pread->readDataBlock<unsigned int>(width,data_height,0,data_start,*h_subData);
				break;
		}
		case GDT_Int32:
		{
				pread->readDataBlock<int>(width,data_height,0,data_start,*h_subData);
				break;
		}
		case GDT_Float32:
		{
				float* allData = pread->transforData<float>();
				break;
		}
		case GDT_Float64:
		{
				double* allData = pread->transforData<double>();
				break;
		}
		default:
		{
			   cout << "transfor data type false!" << endl;
		}
	}
}

int getDevideInfo(int width, int height, dataBlock** dataBlockArray)
{
	int maxnum;		//可以读入的像元的个数
	size_t freeGPU, totalGPU;
	cudaMemGetInfo(&freeGPU, &totalGPU);//size_t* free, size_t* total
	cout << "(free,total)" << freeGPU << "," << totalGPU << endl;

	maxnum = (freeGPU) / (sizeof(int)* 6);//每个pixel基本上要开辟6个中间变量，变量类型都是int
	int sub_height = maxnum / width - 5;	//每个分块的高度sub_height
	int blockNum = height / sub_height + 1;	//总的分块个数
	
	*dataBlockArray = new dataBlock[blockNum];
	
	int subIdx = 0;
	for (int height_all = 0; height_all < height; height_all += sub_height)
	{
		int task_start = subIdx*sub_height;
		int task_end;
		if ((subIdx + 1)*sub_height - height <= 0)
			task_end = (subIdx + 1)*sub_height - 1;
		else
			task_end = height - 1;
		int data_start, data_end;
		if (task_start - 1 <= 0)
			data_start = 0;
		else
			data_start = task_start - 1;
		if (task_end + 1 >= height - 1)
			data_end = height - 1;
		else
			data_end = task_end + 1;
		int data_height = data_end - data_start + 1;
		int task_height = task_end - task_start + 1;

		(*dataBlockArray)[subIdx].dataStart = data_start;
		(*dataBlockArray)[subIdx].dataEnd = data_end;
		(*dataBlockArray)[subIdx].taskStart = task_start;
		(*dataBlockArray)[subIdx].taskEnd = task_end;
		(*dataBlockArray)[subIdx].subTaskHeight = task_height;
		(*dataBlockArray)[subIdx].subDataHeight = data_height;

		subIdx++;
	}
	return blockNum;
}


//GPU执行上一个分块的任务，同时读下一个分块的数据
void exeGPUAndMemcpy( CGDALRead* pread, std::map<int, Patch> &mapPatch, UF &Quf, bool &split)
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	cudaSetDevice(0);
	int width = pread->cols();
	int height = pread->rows();



	dataBlock* dataBlockArray = NULL;		//blockInfo
	int* mergeStructArray = (int* )malloc(sizeof(int) * width * 4 * width);	//mergeInfo
	int blockNum;
	blockNum = getDevideInfo(width, height, &dataBlockArray);//get information of all the blocks,allocate memory for mergeArray

	int iBlock;

	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
	dim3 blockDim(blockSize.x, blockSize.y, 1);
	dim3 gridDim((pread->cols() + blockSize.x - 1) / blockSize.x, (pread->rows() + blockSize.y - 1) / blockSize.y, 1);

	int* h_subDataNextBlock = NULL;//host端读下一个分块

    cudaStream_t stream[blockNum];

    for (int i = 0; i < blockNum; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

	for(iBlock = 0; iBlock < blockNum; iBlock++)
	{
		//当前分块info
		int data_start = dataBlockArray[iBlock].dataStart;
		int data_end = dataBlockArray[iBlock].dataEnd;
		int task_start = dataBlockArray[iBlock].taskStart;
		int task_end = dataBlockArray[iBlock].taskEnd;
		int data_height = dataBlockArray[iBlock].subDataHeight;
		int task_height = dataBlockArray[iBlock].subTaskHeight;

		size_t nBytes_task = data_height * width * sizeof(int);
		int* h_labelMap;
    	checkCudaErrors(cudaHostAlloc((void **)&h_labelMap, nBytes_task, cudaHostAllocDefault));
	    memset(h_labelMap, 0, data_height * width);

		int* h_PixelNum;
    	checkCudaErrors(cudaHostAlloc((void **)&h_PixelNum, nBytes_task, cudaHostAllocDefault));
	    memset(h_PixelNum, 0, data_height * width);

		int* h_Peri;
    	checkCudaErrors(cudaHostAlloc((void **)&h_Peri, nBytes_task, cudaHostAllocDefault));
	    memset(h_Peri, 0, data_height * width);
		std::map<int, Patch> sub_mapPatch;

		if(iBlock == 0)
		{
			int* h_subData = NULL;
			loadBlockData(width, data_height, data_start, &h_subData, pread);
			
			if (!dataBlockArray[iBlock].isSplit()) // no need to split
			{
				split = 0;
				cout << "do not need devide the picture" << endl;
				LineCCLNoSplit(h_subData, width, height, blockDim, gridDim, h_labelMap, h_PixelNum, h_Peri, (int)pread->invalidValue());
				createPatchMap(h_subData, h_labelMap, h_PixelNum, h_Peri, data_height, width, mapPatch, data_start, data_end, task_start, task_end);
			}
			else
			{
				split = 1;
				LineSplitCCL(&h_subDataNextBlock, dataBlockArray[iBlock+1], pread ,h_subData, width, data_height, task_height, blockDim, gridDim, h_labelMap, h_PixelNum, h_Peri, (int)pread->invalidValue(), task_start*width, data_start, data_end, task_start, task_end);
				createPatchMap(h_subData, h_labelMap, h_PixelNum, h_Peri, data_height, width, sub_mapPatch, data_start, data_end, task_start, task_end);
				mergePatchMap(mapPatch, sub_mapPatch);

				memcpy(mergeStructArray+iBlock, h_labelMap + (task_height - 1) * width , sizeof(int) * width);//mlastRowLabel
			}
		    checkCudaErrors(cudaFreeHost(h_subData));
		}
		else
		{
			int* h_subData = h_subDataNextBlock;//当前分块的值
			h_subDataNextBlock = NULL;
			
			LineSplitCCL2(&h_subDataNextBlock, dataBlockArray, iBlock, pread, h_subData, width, data_height,task_height, blockDim, gridDim, h_labelMap, h_PixelNum, h_Peri, (int)pread->invalidValue(), task_start*width, data_start, data_end, task_start, task_end);
			createPatchMap(h_subData, h_labelMap, h_PixelNum, h_Peri, data_height, width, sub_mapPatch, data_start, data_end, task_start, task_end);
			mergePatchMap(mapPatch, sub_mapPatch);
			
			memcpy(mergeStructArray+(iBlock-1)*width*4+width, h_labelMap, sizeof(int)*width);//mfirstRowLabel
			memcpy(mergeStructArray+(iBlock-1)*width*4+width*2, h_subData, sizeof(int)*width);//mh_subDataFirst
			memcpy(mergeStructArray+(iBlock-1)*width*4+width*3, h_subData+width, sizeof(int)*width);//mh_subDataSecond
			if(iBlock != blockNum - 1)
			{			
				memcpy(mergeStructArray+iBlock*width*4, h_labelMap + width*(task_height - 1), sizeof(int)*width);//mlastRowLabel用完了之后将当前块的最后一行保存下来，下次用
			}
		    checkCudaErrors(cudaFreeHost(h_subData));
		}
	    checkCudaErrors(cudaFreeHost(h_labelMap));
	    checkCudaErrors(cudaFreeHost(h_PixelNum));
	    checkCudaErrors(cudaFreeHost(h_Peri));
	}
	MergePatchArray(width, blockNum, (int)pread->invalidValue(), mergeStructArray, Quf);
	// delete[] mergeStructArray;
	// mergeStructArray = NULL;
	free(mergeStructArray);
	delete[] dataBlockArray;
	dataBlockArray = NULL;
    for (int i = 0; i < blockNum; ++i)
    {
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }
}
int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		cout << "please input I/O filename. exit." << endl;
		return -1;
	}
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CGDALRead* pread = new CGDALRead;
	// load all data to unsigned char array when read data;
	
	if (!pread->loadMetaData(argv[1]))
	{
		cout << "load error!" << endl;
	}
	cout << "rows:" << pread->rows() << endl;
	cout << "cols:" << pread->cols() << endl;
	cout << "bandnum:" << pread->bandnum() << endl;
	cout << "datalength:" << pread->datalength() << endl;
	cout << "invalidValue:" << pread->invalidValue() << endl;
	cout << "datatype:" << GDALGetDataTypeName(pread->datatype()) << endl;
	cout << "projectionRef:" << pread->projectionRef() << endl;
	cout << "perPixelSize:" << pread->perPixelSize() << endl;

	int imgSize = pread->rows()*pread->cols();
	// int mergeStruct::width = pread->cols();

	//分块
	std::map<int, Patch> mapPatch;
	UF	Quf(imgSize);//QuickUnion算法，合并斑块序号
	bool split = 0;
	exeGPUAndMemcpy(pread, mapPatch, Quf, split);
	if(split)
		outputPatchToFile(mapPatch, Quf, "LineUF");
	else
		outputPatchToFile(mapPatch, "LineUF");
	return 0;
}
// void devideImg( CGDALRead* pread, std::map<int, Patch> &mapPatch, UF &Quf, bool &split)
// {
// 	int devicesCount;
// 	cudaGetDeviceCount(&devicesCount);
// 	cudaSetDevice(0);

// 	int width = pread->cols();
// 	int height = pread->rows();

// 	int maxnum;		//可以读入的像元的个数
// 	size_t freeGPU, totalGPU;
// 	cudaMemGetInfo(&freeGPU, &totalGPU);//size_t* free, size_t* total
// 	cout << "(free,total)" << freeGPU << "," << totalGPU << endl;

// 	maxnum = (freeGPU) / (sizeof(int)* 6);//每个pixel基本上要开辟6个中间变量，变量类型都是int
// 	int sub_height = maxnum / width - 5;	//每个分块的高度sub_height
// 	int subIdx = 0;
// 	int* lastRowLabel = new int[width];
// 	int* firstRowLabel = new int[width];


// 	int *Meg = NULL;	//合并数组
// 	int Merge_count = 0;	//合并计数
// 	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
// 	dim3 blockDim(blockSize.x, blockSize.y, 1);
// 	dim3 gridDim((pread->cols() + blockSize.x - 1) / blockSize.x, (pread->rows() + blockSize.y - 1) / blockSize.y, 1);

// {
// 	// //the first block----------------------start------------------------------------------
// 	// //------------------------------------------------------------------------------------
// 	// //------------------------------------------------------------------------------------
// 	// int task_start0 = 0;
// 	// int task_end0;
// 	// if ((subIdx + 1)*sub_height - height <= 0)
// 	// 	task_end0 = (subIdx + 1)*sub_height - 1;
// 	// else
// 	// 	task_end0 = height - 1;
// 	// int data_start0, data_end0;
// 	// if (task_start0 - 1 <= 0)
// 	// 	data_start0 = 0;
// 	// else
// 	// 	data_start0 = task_start0 - 1;
// 	// if (task_end0 + 1 >= height - 1)
// 	// 	data_end0 = height - 1;
// 	// else
// 	// 	data_end0 = task_end0 + 1;
// 	// int data_height0 = data_end0 - data_start0 + 1;
// 	// int task_height0 = task_end0 - task_start0 + 1;

// 	// int* h_subData0 = NULL;
// 	// loadBlockData(width, data_height0, data_start0, &h_subData0, pread);	

// 	// size_t nBytes_task0 = task_height0 * width * sizeof(int);
// 	// int* h_labelMap0;
// 	// checkCudaErrors(cudaHostAlloc((void **)&h_labelMap0, nBytes_task0, cudaHostAllocDefault));
//  //    memset(h_labelMap0, 0, task_height0 * width);

// 	// int* h_PixelNum0;
// 	// checkCudaErrors(cudaHostAlloc((void **)&h_PixelNum0, nBytes_task0, cudaHostAllocDefault));
//  //    memset(h_PixelNum0, 0, task_height0 * width);

// 	// int* h_Peri0;
// 	// checkCudaErrors(cudaHostAlloc((void **)&h_Peri0, nBytes_task0, cudaHostAllocDefault));
//  //    memset(h_Peri0, 0, task_height0 * width);

// 	// cout << "subIdx ,data_height:" << subIdx << "," << data_height0 << endl;
// 	// cout << "data_start:" << data_start0 << endl;
// 	// cout << "data___end:" << data_end0 << endl;
// 	// cout << "task_start:" << task_start0 << endl;
// 	// cout << "task___end:" << task_end0 << endl;
// 	// cout << "-------------------------------------------" << endl;
// 	// std::map<int, Patch> sub_mapPatch0;
// 	// if ((task_start0 == data_start0) && (task_end0 == data_end0))
// 	// {
// 	// 	split = 0;
// 	// 	cout << "do not need devide the picture" << endl;
// 	// 	LineCCLNoSplit(h_subData0, width, height, blockDim, gridDim, h_labelMap0, h_PixelNum0, h_Peri0, (int)pread->invalidValue());
// 	// 	createPatchMap(h_subData0, h_labelMap0, h_PixelNum0, h_Peri0, data_height0, width, mapPatch, data_start0, data_end0, task_start0, task_end0);
// 	// }
// 	// else
// 	// {
// 	// 	split = 1;
// 	// 	if (task_start == 0)
// 	// 	{
// 	// 		LineSplitCCL(h_subData0, width, data_height0, task_height0, blockDim, gridDim, h_labelMap0, h_PixelNum0, h_Peri0, (int)pread->invalidValue(), task_start0*width, data_start0, data_end0, task_start0, task_end0);
// 	// 		createPatchMap(h_subData0, h_labelMap0, h_PixelNum0, h_Peri0, data_height0, width, sub_mapPatch0, data_start0, data_end0, task_start0, task_end0);
// 	// 		mergePatchMap(mapPatch, sub_mapPatch0);
// 	// 		memcpy(lastRowLabel , h_labelMap0 + (task_height0-1) * width , sizeof(int)*width);
// 	// 	}
// 	// }

// 	// //the first block----------------------end------------------------------------------
// 	// //----------------------------------------------------------------------------------
// 	// //----------------------------------------------------------------------------------
// }		
	
// 	for (int height_all = 0; height_all < height; height_all += sub_height)
// 	{
// 		int task_start = subIdx*sub_height;
// 		int task_end;
// 		if ((subIdx + 1)*sub_height - height <= 0)
// 			task_end = (subIdx + 1)*sub_height - 1;
// 		else
// 			task_end = height - 1;
// 		int data_start, data_end;
// 		if (task_start - 1 <= 0)
// 			data_start = 0;
// 		else
// 			data_start = task_start - 1;
// 		if (task_end + 1 >= height - 1)
// 			data_end = height - 1;
// 		else
// 			data_end = task_end + 1;
// 		int data_height = data_end - data_start + 1;
// 		int task_height = task_end - task_start + 1;

// 		int* h_subData = NULL;
// 		loadBlockData(width, data_height, data_start, &h_subData, pread);	

// 		size_t nBytes_task = task_height * width * sizeof(int);
// 		int* h_labelMap;
//     	checkCudaErrors(cudaHostAlloc((void **)&h_labelMap, nBytes_task, cudaHostAllocDefault));
// 	    memset(h_labelMap, 0, task_height * width);

// 		int* h_PixelNum;
//     	checkCudaErrors(cudaHostAlloc((void **)&h_PixelNum, nBytes_task, cudaHostAllocDefault));
// 	    memset(h_PixelNum, 0, task_height * width);

// 		int* h_Peri;
//     	checkCudaErrors(cudaHostAlloc((void **)&h_Peri, nBytes_task, cudaHostAllocDefault));
// 	    memset(h_Peri, 0, task_height * width);

// 		cout << "subIdx ,data_height:" << subIdx << "," << data_height << endl;
// 		cout << "data_start:" << data_start << endl;
// 		cout << "data___end:" << data_end << endl;
// 		cout << "task_start:" << task_start << endl;
// 		cout << "task___end:" << task_end << endl;
// 		cout << "-------------------------------------------" << endl;

// 		//至此，每个分块的初始数据已经保存在h_subData中,下面调用核函数，进行计算，并将数据传回

// 		std::map<int, Patch> sub_mapPatch;

// 		if ((task_start == data_start) && (task_end == data_end))
// 		{
// 			split = 0;
// 			cout << "do not need devide the picture" << endl;
// 			LineCCLNoSplit(h_subData, width, height, blockDim, gridDim, h_labelMap, h_PixelNum, h_Peri, (int)pread->invalidValue());
// 			createPatchMap(h_subData, h_labelMap, h_PixelNum, h_Peri, data_height, width, mapPatch, data_start, data_end, task_start, task_end);
// 		}
// 		else
// 		{
// 			split = 1;
// 			if (task_start == 0)
// 			{
// 				LineSplitCCL(h_subData, width, data_height, task_height, blockDim, gridDim, h_labelMap, h_PixelNum, h_Peri, (int)pread->invalidValue(), task_start*width, data_start, data_end, task_start, task_end);
// 				createPatchMap(h_subData, h_labelMap, h_PixelNum, h_Peri, data_height, width, sub_mapPatch, data_start, data_end, task_start, task_end);
// 				mergePatchMap(mapPatch, sub_mapPatch);
// 				memcpy(lastRowLabel , h_labelMap + (task_height-1) * width , sizeof(int)*width);
// 			}
// 			else
// 			{
// 				//所有分块共同维护
// 				//mapPatch---存储所有局部patch的value,area,peri
// 				//Quf---存储要合并的信息

// 				LineSplitCCL2(h_subData, width, data_height,task_height, blockDim, gridDim, h_labelMap, h_PixelNum, h_Peri, (int)pread->invalidValue(), task_start*width, data_start, data_end, task_start, task_end);
// 				createPatchMap(h_subData, h_labelMap, h_PixelNum, h_Peri, data_height, width, sub_mapPatch, data_start, data_end, task_start, task_end);
// 				mergePatchMap(mapPatch, sub_mapPatch);
				

// 				memcpy(firstRowLabel, h_labelMap, sizeof(int)*width);//取当前块的第一行用来与上一分块中的标记nextTaskStartLabel做对比生成集合
// 				int *h_subDataFirst = new int[width];
// 				int *h_subDataSecond = new int[width];
// 				memcpy(h_subDataFirst, h_subData, sizeof(int)*width);//取当前块的第一行用来与上一分块中的标记nextTaskStartLabel做对比生成集合
// 				memcpy(h_subDataSecond, h_subData+width, sizeof(int)*width);//取当前块的第一行用来与上一分块中的标记nextTaskStartLabel做对比生成集合
				

// 				//现在有上个分块最后一行的标识lastRowLabel
// 				//当前分块第一行的标识firstRowLabel,将这两行用来生成Meg数组，并构造Union-find
// 				Meg = (int *)malloc(sizeof(int) * width * 2);
// 				Merge_count = findMerge(width,(int)pread->invalidValue(), Meg, h_subDataFirst,h_subDataSecond,lastRowLabel,firstRowLabel);
// 				MergePatch( Meg, Merge_count, Quf);

// 				memcpy(lastRowLabel, h_labelMap + width*(task_height - 1), sizeof(int)*width);//用完了之后将当前块的最后一行保存下来，下次用
// 				delete[] Meg;
// 				Meg = NULL;
// 				Merge_count = 0;
// 			}
// 		}
// 	    checkCudaErrors(cudaFreeHost(h_subData));
// 	    checkCudaErrors(cudaFreeHost(h_labelMap));
// 	    checkCudaErrors(cudaFreeHost(h_PixelNum));
// 	    checkCudaErrors(cudaFreeHost(h_Peri));

// 		subIdx++;
// 	}

// }



// int main(int argc, char *argv[])
// {
// 	if (argc < 2)
// 	{
// 		cout << "please input I/O filename. exit." << endl;
// 		return -1;
// 	}
// 	GDALAllRegister();
// 	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
// 	CGDALRead* pread = new CGDALRead;
// 	// load all data to unsigned char array when read data;
	
// 	if (!pread->loadMetaData(argv[1]))
// 	{
// 		cout << "load error!" << endl;
// 	}
// 	cout << "rows:" << pread->rows() << endl;
// 	cout << "cols:" << pread->cols() << endl;
// 	cout << "bandnum:" << pread->bandnum() << endl;
// 	cout << "datalength:" << pread->datalength() << endl;
// 	cout << "invalidValue:" << pread->invalidValue() << endl;
// 	cout << "datatype:" << GDALGetDataTypeName(pread->datatype()) << endl;
// 	cout << "projectionRef:" << pread->projectionRef() << endl;
// 	cout << "perPixelSize:" << pread->perPixelSize() << endl;

// 	int imgSize = pread->rows()*pread->cols();
// 	//分块
// 	std::map<int, Patch> mapPatch;
// 	UF	Quf(imgSize);//QuickUnion算法，合并斑块序号
// 	bool split = 0;
// 	devideImg(pread, mapPatch, Quf, split);
// 	if(split)
// 		outputPatchToFile(mapPatch, Quf, "LineUF");
// 	else
// 		outputPatchToFile(mapPatch, "LineUF");
// 	return 0;
// }

// int main()
// {
// 	int imgSize = 25;
// 	// int array[25] = { 1, 1, 2, 2, 2, 1, 2, 2, 3, 2, 1, 2, 1, 3, 2, 2, 1, 3, 3, 3, 2, 3, 3, 3, 3 };
// 	int array[25] = { 1, 3, 3, 3, 3, 
// 					  1, 3, 3, 1, 3,
// 					  1, 2, 1, 3, 2,
// 					  2, 1, 3, 2, 3, 
// 					  1, 2, 2, 3, 2 };
// 	int* AllDataHost = new int[25];
// 	for (int i = 0; i < 25; i++)
// 	{
// 		AllDataHost[i] = array[i];
// 	}

// 	std::map<int, Patch> mapPatch;
// 	UF	Quf(imgSize);//QuickUnion算法，合并斑块序号
// 	// vector<set<int> >mergeSet;
// 	PRead *pread = new PRead(5, 5, 0);
// 	devideImg1(AllDataHost, pread, mapPatch,Quf);
// 	outputPatchToFile(mapPatch,Quf, "LineUF");

// 	return 0;
// }