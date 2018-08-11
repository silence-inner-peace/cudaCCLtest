#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include "timer.h"
#include "GDALRead.h"

#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "utils.h"
#include "basestruct.h"
#include "operator.h"
#include <unistd.h>


void loadBlockData(int width, int data_height, int data_start, int** h_subData, CGDALRead* pread);

__device__ int find(int * localLabel, int p)
{
	if (localLabel[p] != -1)
	{
		while (p != localLabel[p])
		{
			p = localLabel[p];
		}
		return p;
	}
	else
		return -1;
}
__device__ void findAndUnion(int* buf, int g1, int g2) {
	bool done;
	do {

		g1 = find(buf, g1);
		g2 = find(buf, g2);

		// it should hold that g1 == buf[g1] and g2 == buf[g2] now

		if (g1 < g2) {
			int old = atomicMin(&buf[g2], g1);
			done = (old == g2);
			g2 = old;
		}
		else if (g2 < g1) {
			int old = atomicMin(&buf[g1], g2);
			done = (old == g1);
			g1 = old;
		}
		else {
			done = true;
		}

	} while (!done);
}


__global__ void gpuLineLocal(int* devSrcData, int * devLabelMap, int width, int task_height, int nodata)
{
	//int id = threadIdx.x + threadIdx.y * blockDim.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;
	//if (id > imgDimension.x * imgDimension.y)	return;

	//int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int tid = threadIdx.x;

	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;

	bool limits = x < width && y < task_height;
	int id = x + y * width;


	__shared__ int localLabel[32 * 16];


	if (limits)
	{
		localLabel[tid] = tid;
		__syncthreads();

		int focusP = devSrcData[x + y * width];
		if (focusP != nodata && threadIdx.x > 0 && focusP == devSrcData[x - 1 + y * width])
			localLabel[tid] = localLabel[tid - 1];
		__syncthreads();

		int buf = tid;

		while (buf != localLabel[buf])
		{
			buf = localLabel[buf];
			localLabel[tid] = buf;
		}

		int globalL = (blockIdx.x * blockDim.x + buf) + (blockIdx.y) * width;
		devLabelMap[id] = globalL;

		if (focusP == nodata)
			devLabelMap[id] = -1;
	}

}

__global__ void gpuLineUfGlobal(int* devSrcData, int * devLabelMap,  int width, int task_height, int nodata)
{
	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;

	if (in_limits)
	{
		int center = devSrcData[gid];
		if (center != nodata)
		{
			// search neighbour, left and up
			//if (x > 0 && threadIdx.x == 0 && center == devSrcData[x - 1 + y * imgDimension.x])
			//	findAndUnion(devLabelMap, gid, x - 1 + y * imgDimension.x); // left
			//if (y > 0 && threadIdx.y == 0 && center == devSrcData[x + (y - 1) * imgDimension.x])
			//	findAndUnion(devLabelMap, gid, x + (y - 1) * imgDimension.x); // up

			if (x > 0 && threadIdx.x == 0)//&& center == left
			{
				if (center == devSrcData[x - 1 + y * width])
					findAndUnion(devLabelMap, gid, x - 1 + y * width); // left
			}
			if (y > 0 && threadIdx.y == 0)//&& center == up 
			{
				if (center == devSrcData[x + (y - 1) * width])
					findAndUnion(devLabelMap, gid, x + (y - 1) * width); // up
			}
			if (y > 0 && x > 0 && threadIdx.y == 0)// && center == leftup
			{
				if (center == devSrcData[x - 1 + (y - 1) * width])
					findAndUnion(devLabelMap, gid, x - 1 + (y - 1) * width); // up-left
			}
			if (y > 0 && x < width - 1 && threadIdx.y == 0)// && center == rightup
			{
				if (center == devSrcData[x + 1 + (y - 1) * width])
					findAndUnion(devLabelMap, gid, x + 1 + (y - 1) * width); // up-right
			}
		}
	}
}


__global__ void gpuLineUfFinal(int * labelMap, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	bool limits = x < width && y < task_height;

	int gid = x + y * width;
	if (limits)
		labelMap[gid] = find(labelMap, gid);
}

__global__ void getEachPixelPeriTop1(int *dev_iData, int *dev_iData_last,int*dev_PixelPerimeter, int width, int task_height, int data_start, int data_end, int task_start, int task_end)
{
	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;//×îºóÒ»ÐÐ²»ÊôÓÚµ±Ç°¿éµÄtaskÇøÓò£¬²»×ö¼ÆËã
	if (in_limits)
	{
		int center = dev_iData[gid];
		if (x == 0)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x == width - 1)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (y == 0)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if(y == task_height - 1)
		{
			if(center!=dev_iData_last[x])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}

		if (x>0)
		{
			if (center != dev_iData[gid - 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (x < width - 1)
		{
			if (center != dev_iData[gid + 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y > 0)
		{
			if (center != dev_iData[gid - width])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if(y < task_height - 1)
		{
			if (center != dev_iData[gid + width])//down
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
	}
}
__global__ void getEachPixelPeriMid2(int *dev_iData, int*data_startValue,int *dev_dataLastValue, int*dev_PixelPerimeter, int width, int task_height, int data_start, int data_end, int task_start, int task_end)
{
	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;//×îºóÒ»ÐÐ²»ÊôÓÚµ±Ç°¿éµÄtaskÇøÓò£¬²»×ö¼ÆËã
	if (in_limits)
	{
		int center = dev_iData[gid];
		if (x == 0)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x == width - 1)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x>0)
		{
			if (center != dev_iData[gid - 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (x < width - 1)
		{
			if (center != dev_iData[gid + 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y>0)
		{
			if (center != dev_iData[gid - width])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y == 0)
		{
			if (center != data_startValue[x])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if(y == task_height - 1)
		{
			if(center != dev_dataLastValue[x])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if(y < task_height-1)
		{
			if (center != dev_iData[gid + width])//down
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
	}
}
__global__ void getEachPixelPeriBottom(int *dev_iData, int*data_startValue, int*dev_PixelPerimeter, int width, int task_height, int data_start, int data_end, int task_start, int task_end)
{
	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;//×îºóÒ»ÐÐÊôÓÚµ±Ç°¿éµÄtaskÇøÓò£¬ÐèÒª×ö¼ÆËã
	if (in_limits)
	{
		int center = dev_iData[gid];
		if (x == 0)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x == width - 1)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (y == task_height - 1)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x > 0)
		{
			if (center != dev_iData[gid - 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (x < width - 1)
		{
			if (center != dev_iData[gid + 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y > 0)
		{
			if (center != dev_iData[gid - width])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y == 0)
		{
			if (center != data_startValue[x])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y < task_height - 1)
		{
			if (center != dev_iData[gid + width])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}

	}
}

__global__ void getEachPixelPeriNoSplit(int *dev_iData, int*dev_PixelPerimeter, int width, int height)
{
	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < height;//×îºóÒ»ÐÐÊôÓÚµ±Ç°¿éµÄtaskÇøÓò£¬ÐèÒª×ö¼ÆËã
	if (in_limits)
	{
		int center = dev_iData[gid];
		if (x == 0)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x == width - 1)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (y == 0)
		{
				dev_PixelPerimeter[gid] += 1;
		}
		if (y == height - 1)
		{
			dev_PixelPerimeter[gid] += 1;
		}
		if (x > 0)
		{
			if (center != dev_iData[gid - 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (x < width - 1)
		{
			if (center != dev_iData[gid + 1])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}
		if (y > 0)
		{
			if (center != dev_iData[gid - width])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}

		if (y < height - 1)
		{
			if (center != dev_iData[gid + width])
			{
				dev_PixelPerimeter[gid] += 1;
			}
		}

	}
}

__global__ void getPixNumAndPeri(int* dOutPixNum, int* dOutPeri, int* dev_labelMap, int *dev_pixelPerimeter, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		int regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		if (regLabel >= 0)
		{
			atomicAdd(dOutPixNum + regLabel, 1);//get area
			atomicAdd(dOutPeri + regLabel, dev_pixelPerimeter[gid]);
		}

	}
}


__global__ void updateDevLabel(int * dev_labelMap, int labelStart, int task_height, int width)
{
	//heightÊÇdev_labelMapµÄ¸ß¶È
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;
	bool limits = x < width && y < task_height;
	if (limits)
	{
		dev_labelMap[gid] += labelStart;
	}
}


double LineCCLNoSplit(int *allData, int width, int height, dim3 blockSize, dim3 gridSize, int * labelMap, int *pixNum, int *perimeter, int nodata)
{
	cout << "LineCCLNoSplit" << endl;
	int2 imgSize; imgSize.x = width;  imgSize.y = height;

	// device data
	int 		  * dev_iData;
	int			  * dev_labelMap;
	int			  * dev_pixNum;
	int			  * dev_perimeter;
	//ÖÐ¼ä±äÁ¿
	int	  * dev_PixelPerimeter;	//Éè±¸¶ËÔÝ´æÖÜ³¤(task_start task_end)

	checkCudaErrors(cudaMalloc((void**)&dev_iData, sizeof(int)* width * height));
	checkCudaErrors(cudaMalloc((void**)&dev_labelMap, sizeof(int)* width * height));
	checkCudaErrors(cudaMalloc((void**)&dev_pixNum, sizeof(int)* width * height));
	checkCudaErrors(cudaMalloc((void**)&dev_perimeter, sizeof(int)* width * height));
	checkCudaErrors(cudaMalloc((void**)&dev_PixelPerimeter, sizeof(int)* width * height));

	// copy data
	checkCudaErrors(cudaMemcpy(dev_iData, allData, sizeof(int)* width * height, cudaMemcpyHostToDevice));
	// set data
	checkCudaErrors(cudaMemset(dev_pixNum, 0, sizeof(int)* width * height));
	checkCudaErrors(cudaMemset(dev_perimeter, 0, sizeof(int)* width * height));
	checkCudaErrors(cudaMemset(dev_PixelPerimeter, 0, sizeof(int)* width * height));

	// reconfigue the dimension of block and grid
	const int blockSizeX = blockSize.x * blockSize.y;
	const int blockSizeY = 1;
	dim3 blockSizeLine(blockSizeX, blockSizeY, 1);
	dim3 gridSizeLine((imgSize.x + blockSizeX - 1) / blockSizeX, (imgSize.y + blockSizeY - 1) / blockSizeY, 1);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	gpuLineLocal << < gridSizeLine, blockSizeLine >> > (dev_iData, dev_labelMap, width, height, nodata);
	gpuLineUfGlobal << <gridSizeLine, blockSizeLine >> > (dev_iData, dev_labelMap, width, height, nodata);
	gpuLineUfFinal << < gridSizeLine, blockSizeLine >> > (dev_labelMap, width, height);
	
	getEachPixelPeriNoSplit << <gridSizeLine, blockSizeLine >> >(dev_iData, dev_PixelPerimeter, width, height);
	getPixNumAndPeri << <gridSizeLine, blockSizeLine >> >(dev_pixNum, dev_perimeter, dev_labelMap, dev_PixelPerimeter, width, height);
	
	cudaEventRecord(stop);

	checkCudaErrors(cudaMemcpy(labelMap, dev_labelMap, sizeof(int)* width * height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(perimeter, dev_perimeter, sizeof(int)* width * height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pixNum, dev_pixNum, sizeof(int)* width * height, cudaMemcpyDeviceToHost));


	cudaFree(dev_iData);
	cudaFree(dev_labelMap);
	cudaFree(dev_perimeter);
	cudaFree(dev_pixNum);
	cudaFree(dev_PixelPerimeter);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//std::cout << "milliseconds = " << milliseconds << std::endl;
	return	milliseconds;

}

double LineSplitCCL(int** h_subDataNextBlock, dataBlock &dataBlockNext, CGDALRead* pread, 
	int *allData, int width, int data_height, int task_height, dim3 blockSize, dim3 gridSize, 
	int *h_labelMap, int *pixNum, int *perimeter, int nodata, int labelStart, int data_start, int data_end, int task_start, int task_end)
{
	/*
	input:
	allData: all data from data_start to data_end
	width: width
	height: data_end-data_start
	kernel configuration:
	blockSize
	gridSize
	output:
	nextTaskStartLabel: the label of data_end row
	*/
	cout << "LineSplitCCL" << endl;
	// device data
	//input
	int   * dev_iData;		//½ÓÊÕÔ­Ê¼Í¼ÏñÖµ(task_start task_end)
	int   * dev_iData_last;	//½ÓÊÕÔ­Ê¼Í¼ÏñµÄ×îºóÒ»ÐÐ
	//ÖÐ¼ä±äÁ¿
	int	  * dev_PixelPerimeter;	//Éè±¸¶ËÔÝ´æÖÜ³¤(task_start task_end)

	//ÓÃÓÚÉú³É×îºósubPatchµÄÊý×é
	int	  * dev_labelMap;	//Éè±¸¶Ë¾Ö²¿±ê¼Ç(task_start task_end)
	int	  * dev_pixNum;		//Éè±¸¶ËÔÝ´æÃæ»ý(task_start task_end)
	int	  * dev_perimeter;	//Éè±¸¶ËÔÝ´æÖÜ³¤(task_start task_end)

	//allocate size
	checkCudaErrors(cudaMalloc((void**)&dev_iData, sizeof(int)* width * task_height));
	checkCudaErrors(cudaMalloc((void**)&dev_iData_last, sizeof(int)* width));

	checkCudaErrors(cudaMalloc((void**)&dev_PixelPerimeter, sizeof(int)* width * (task_height)));

	checkCudaErrors(cudaMalloc((void**)&dev_labelMap, sizeof(int)* width * task_height));

	checkCudaErrors(cudaMalloc((void**)&dev_pixNum, sizeof(int)* width * (task_height)));
	checkCudaErrors(cudaMalloc((void**)&dev_perimeter, sizeof(int)* width * (task_height)));

	// copy data
	checkCudaErrors(cudaMemcpyAsync(dev_iData, allData, sizeof(int)* width * task_height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(dev_iData_last, allData + width * task_height, sizeof(int)* width, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(dev_iData, allData, sizeof(int)* width * task_height, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(dev_iData_last, allData + width * task_height, sizeof(int)* width, cudaMemcpyHostToDevice));

	// set data
	checkCudaErrors(cudaMemset(dev_PixelPerimeter, 0, sizeof(int)* width * task_height));
	checkCudaErrors(cudaMemset(dev_pixNum, 0, sizeof(int)* width * task_height));
	checkCudaErrors(cudaMemset(dev_perimeter, 0, sizeof(int)*  width * task_height));

	// reconfigue the dimension of block and grid
	const int blockSizeX = blockSize.x * blockSize.y;
	const int blockSizeY = 1;
	dim3 blockSizeLine(blockSizeX, blockSizeY, 1);
	dim3 gridSizeLine((width + blockSizeX - 1) / blockSizeX, (task_height + blockSizeY - 1) / blockSizeY, 1);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	gpuLineLocal << < gridSizeLine, blockSizeLine >> > (dev_iData, dev_labelMap, width, task_height, nodata);
	gpuLineUfGlobal << <gridSizeLine, blockSizeLine >> > (dev_iData, dev_labelMap, width, task_height, nodata);
	gpuLineUfFinal << < gridSizeLine, blockSizeLine >> > (dev_labelMap, width, task_height);
	cout << "--------------label---finish---------------------------------" << endl;
	if (task_start == data_start)
		getEachPixelPeriTop1 << < gridSizeLine, blockSizeLine >> >(dev_iData,dev_iData_last,dev_PixelPerimeter, width, task_height, data_start, data_end, task_start, task_end);
	cout << "--------------getEachPixelPeriTop----finish------------------" << endl;

	cudaFree(dev_iData);
	getPixNumAndPeri << <gridSizeLine, blockSizeLine >> >(dev_pixNum, dev_perimeter, dev_labelMap, dev_PixelPerimeter, width, task_height);
	cout << "--------------getPixNumAndPeri----finish--------------------" << endl;


	// checkCudaErrors(cudaMemcpy(h_labelMap, dev_labelMap, sizeof(int)* task_height * width, cudaMemcpyDeviceToHost));
	// cout << "--------------cudaMemcpyDeviceToHost--h_labelMap--finish--------------------" << endl;
	// checkCudaErrors(cudaMemcpy(perimeter, dev_perimeter, sizeof(int)* task_height * width, cudaMemcpyDeviceToHost));
	// cout << "--------------cudaMemcpyDeviceToHost--perimeter--finish--------------------" << endl;
	// checkCudaErrors(cudaMemcpy(pixNum, dev_pixNum, sizeof(int)* task_height * width, cudaMemcpyDeviceToHost));
	// cout << "--------------cudaMemcpyDeviceToHost--pixNum--finish--------------------" << endl;

	checkCudaErrors(cudaMemcpyAsync(h_labelMap, dev_labelMap, sizeof(int)* task_height * width, cudaMemcpyDeviceToHost));
	cout << "--------------cudaMemcpyDeviceToHost--h_labelMap--finish--------------------" << endl;
	checkCudaErrors(cudaMemcpyAsync(perimeter, dev_perimeter, sizeof(int)* task_height * width, cudaMemcpyDeviceToHost));
	cout << "--------------cudaMemcpyDeviceToHost--perimeter--finish--------------------" << endl;
	checkCudaErrors(cudaMemcpyAsync(pixNum, dev_pixNum, sizeof(int)* task_height * width, cudaMemcpyDeviceToHost));
	cout << "--------------cudaMemcpyDeviceToHost--pixNum--finish--------------------" << endl;


	cudaEventRecord(stop);

    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    loadBlockData(width, dataBlockNext.subDataHeight, dataBlockNext.dataStart, h_subDataNextBlock, pread);

    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);


	cudaFree(dev_PixelPerimeter);
	cudaFree(dev_labelMap);
	cudaFree(dev_perimeter);
	cudaFree(dev_pixNum);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "milliseconds = " << milliseconds << std::endl;
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
	return	milliseconds;


}

double LineSplitCCL2(int** h_subDataNextBlock, dataBlock* dataBlockArray, int iBlock, CGDALRead* pread, 
	int *allData, int width, int data_height, int task_height, dim3 blockSize, dim3 gridSize, 
	int *h_labelMap, int *pixNum, int *perimeter, int nodata, int labelStart, int data_start, int data_end, int task_start, int task_end)
{
	/*
	input:
	allData: all data from data_start to data_end
	firstRowLabel:ÓÃÓÚ¸üÐÂµ±Ç°¿éµÄ±êÇ©ÎªÈ«¾Ö±êÇ©
	width: width
	height: data_end-data_start
	kernel configuration:
	blockSize
	gridSize
	output:
	nextTaskStartLabel: the label of data_end row


	*/
	cout << "LineSplitCCL2" << endl;
	// device data
	//input
	int   * dev_iData;		//½ÓÊÕÔ­Ê¼Í¼ÏñÖµ(task_start task_end)
	int   * dev_dataStartValue;//½ÓÊÕÔ­Ê¼Í¼ÏñÖµdata_start
	int   * dev_dataLastValue;//½ÓÊÕÔ­Ê¼Í¼ÏñÖµdata_end

	//ÖÐ¼ä±äÁ¿
	int	  * dev_PixelPerimeter;	//Éè±¸¶ËÔÝ´æÖÜ³¤(task_start task_end)

	//ÓÃÓÚÉú³É×îºósubPatchµÄÊý×é
	int	  * dev_labelMap;	//Éè±¸¶Ë¾Ö²¿±ê¼Ç(task_start task_end)
	int	  * dev_pixNum;		//Éè±¸¶ËÔÝ´æÃæ»ý(task_start task_end)
	int	  * dev_perimeter;	//Éè±¸¶ËÔÝ´æÖÜ³¤(task_start task_end)

	//allocate size
	checkCudaErrors(cudaMalloc((void**)&dev_iData, sizeof(int)* width * task_height));
	checkCudaErrors(cudaMalloc((void**)&dev_dataStartValue, sizeof(int)* width));

	checkCudaErrors(cudaMalloc((void**)&dev_PixelPerimeter, sizeof(int)* width * task_height));

	checkCudaErrors(cudaMalloc((void**)&dev_labelMap, sizeof(int)* width * task_height));

	checkCudaErrors(cudaMalloc((void**)&dev_pixNum, sizeof(int)* width * task_height));
	checkCudaErrors(cudaMalloc((void**)&dev_perimeter, sizeof(int)* width * task_height));

	// copy data
	// checkCudaErrors(cudaMemcpy(dev_dataStartValue, allData, sizeof(int)* width, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(dev_iData, allData + width, sizeof(int)* width * task_height, cudaMemcpyHostToDevice));
	// if(task_end != data_end)
	// {
	// 	checkCudaErrors(cudaMalloc((void**)&dev_dataLastValue, sizeof(int)* width));
	// 	checkCudaErrors(cudaMemcpy(dev_dataLastValue, allData+width*(task_height+1), sizeof(int)* width, cudaMemcpyHostToDevice));
	// }

	checkCudaErrors(cudaMemcpyAsync(dev_dataStartValue, allData, sizeof(int)* width, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(dev_iData, allData + width, sizeof(int)* width * task_height, cudaMemcpyHostToDevice));
	if(task_end != data_end)
	{
		checkCudaErrors(cudaMalloc((void**)&dev_dataLastValue, sizeof(int)* width));
		checkCudaErrors(cudaMemcpyAsync(dev_dataLastValue, allData+width*(task_height+1), sizeof(int)* width, cudaMemcpyHostToDevice));
	}

	// set data
	checkCudaErrors(cudaMemset(dev_PixelPerimeter, 0, sizeof(int)* width *task_height));
	checkCudaErrors(cudaMemset(dev_pixNum, 0, sizeof(int)* width * task_height));
	checkCudaErrors(cudaMemset(dev_perimeter, 0, sizeof(int)*  width * task_height));

	// reconfigue the dimension of block and grid
	const int blockSizeX = blockSize.x * blockSize.y;
	const int blockSizeY = 1;
	dim3 blockSizeLine(blockSizeX, blockSizeY, 1);
	dim3 gridSizeLine((width + blockSizeX - 1) / blockSizeX, (task_height + blockSizeY - 1) / blockSizeY, 1);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	gpuLineLocal << < gridSizeLine, blockSizeLine >> > (dev_iData, dev_labelMap,  width, task_height, nodata);
	gpuLineUfGlobal << <gridSizeLine, blockSizeLine >> > (dev_iData, dev_labelMap, width, task_height, nodata);
	gpuLineUfFinal << < gridSizeLine, blockSizeLine >> > (dev_labelMap, width, task_height);


	if (task_end == data_end)
		getEachPixelPeriBottom << < gridSizeLine, blockSizeLine >> >(dev_iData, dev_dataStartValue, dev_PixelPerimeter, width, task_height, data_start, data_end, task_start, task_end);
	else
		getEachPixelPeriMid2 << < gridSizeLine, blockSizeLine >> >(dev_iData, dev_dataStartValue, dev_dataLastValue, dev_PixelPerimeter, width, task_height, data_start, data_end, task_start, task_end);


	cudaFree(dev_iData);

	getPixNumAndPeri << <gridSizeLine, blockSizeLine >> >(dev_pixNum, dev_perimeter, dev_labelMap, dev_PixelPerimeter, width, task_height);

	updateDevLabel << <gridSizeLine, blockSizeLine >> > (dev_labelMap, labelStart, task_height, width);
	
	cout << "--------------updateDevLabeling----finish--------------------" << endl;

    // have CPU do some work while waiting for stage 1 to finish


	// checkCudaErrors(cudaMemcpy(h_labelMap, dev_labelMap, sizeof(int)* width* task_height, cudaMemcpyDeviceToHost));
	// cout << "--------------cudaMemcpyDeviceToHost--dev_labelMap--finish--------------------" << endl;
	// checkCudaErrors(cudaMemcpy(perimeter, dev_perimeter, sizeof(int)* width* task_height, cudaMemcpyDeviceToHost));
	// cout << "--------------cudaMemcpyDeviceToHost--dev_perimeter--finish--------------------" << endl;
	// checkCudaErrors(cudaMemcpy(pixNum, dev_pixNum, sizeof(int)* width* task_height, cudaMemcpyDeviceToHost));
	// cout << "--------------cudaMemcpyDeviceToHost--pixNum--finish--------------------" << endl;
	
	checkCudaErrors(cudaMemcpyAsync(h_labelMap, dev_labelMap, sizeof(int)* width* task_height, cudaMemcpyDeviceToHost));
	cout << "--------------cudaMemcpyDeviceToHost--dev_labelMap--finish--------------------" << endl;
	checkCudaErrors(cudaMemcpyAsync(perimeter, dev_perimeter, sizeof(int)* width* task_height, cudaMemcpyDeviceToHost));
	cout << "--------------cudaMemcpyDeviceToHost--dev_perimeter--finish--------------------" << endl;
	checkCudaErrors(cudaMemcpyAsync(pixNum, dev_pixNum, sizeof(int)* width* task_height, cudaMemcpyDeviceToHost));
	cout << "--------------cudaMemcpyDeviceToHost--pixNum--finish--------------------" << endl;

	cudaEventRecord(stop);

    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }
	if (task_end != data_end)
	    loadBlockData(width, dataBlockArray[iBlock+1].subDataHeight, dataBlockArray[iBlock+1].dataStart, h_subDataNextBlock, pread);

    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);


	cudaFree(dev_dataStartValue);
	if(task_end != data_end)
		cudaFree(dev_dataLastValue);
	cudaFree(dev_PixelPerimeter);
	cudaFree(dev_labelMap);
	cudaFree(dev_perimeter);
	cudaFree(dev_pixNum);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
	
	std::cout << "milliseconds = " << milliseconds << std::endl;
	return	milliseconds;
}

