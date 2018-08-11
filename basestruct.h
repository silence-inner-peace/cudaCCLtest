
#define NODATA -9999 //默认nodata值
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
//image metadata图像元数据,用GDAL获取
typedef struct
{
	char* imagename;//图像名称
	unsigned char *imagedata;//图像数组指针
	int nXSize;//图像X坐标大小
	int nYSize;//图像Y坐标大小
	double cellxsize;//像素x坐标分辨率
	double cellysize;//像素y坐标分辨率
	double adfMinMax[2];//最大最小值
	int nodata;//nodata值
}IMAGE_METADATA;

//patch结构
typedef struct
{
	int PatchIndex;//斑块序号

	int PatchValue;//斑块颜色
	//char* PatchType;//斑块类型（class）

	int PatchCount;//斑块数量
	double PatchArea;//斑块面积

	int PatchContour;//斑块轮廓
	double PatchPerimeter;//斑块周长

}Patch;

class dataBlock
{
public:
	int dataStart;
	int dataEnd;
	int taskStart;
	int taskEnd;
	int subDataHeight;
	int subTaskHeight;
	bool isFirstBlock()
	{
		if(dataStart == taskStart)
			return true;
		else
			return false;
	}
	bool isLastBlock()
	{
		if(dataEnd == taskEnd)
			return true;
		else
			return false;
	}
	bool isSplit()
	{
		if((dataStart == taskStart) && (dataEnd == taskEnd))
			return false;
		else
			return true;
	}
};
class mergeStruct
{
public:
	// int width;
	int *mlastRowLabel;		//交界处第一行标记值
	int *mfirstRowLabel;	//交界处第二行标记值
	int *mh_subDataFirst;	//交界处第一行原始值
	int *mh_subDataSecond;	//交界处第二行原始值

	mergeStruct()
	{
		mh_subDataFirst = NULL;
		mh_subDataSecond = NULL;
		mlastRowLabel = NULL;
		mfirstRowLabel = NULL;
	}
	mergeStruct(int width)
	{
		mh_subDataFirst = new int[width];
		mh_subDataSecond = new int[width];
		mlastRowLabel = new int[width];
		mfirstRowLabel = new int[width];
	}
	void setLength(int width)
	{
		mh_subDataFirst = new int[width];
		mh_subDataSecond = new int[width];
		mlastRowLabel = new int[width];
		mfirstRowLabel = new int[width];
	}
	~mergeStruct()
	{
		delete[] mh_subDataFirst;
		mh_subDataFirst = NULL;
		delete[] mh_subDataSecond;
		mh_subDataSecond = NULL;
		delete[] mlastRowLabel;
		mlastRowLabel = NULL;
		delete[] mfirstRowLabel;
		mfirstRowLabel = NULL;		
	}
};
//类结构
typedef struct
{
	
	int PClassValue;//类斑块颜色
	int PClassCount;//类斑块个数

	int PClassIndex;//类序号
	int PClassPos; //记录顺序Patch数组中的首位置
	char* PClassType;//类斑块类型（class）
	
	double PClassArea;//类斑块面积
	double PClassPerimeter;//类斑块周长
	double PClassProLand;//类面积占总景观面积值
}PClass;


class PRead
{
public:
	PRead(int col, int row, int invalid);
public:
	int col;
	int row;
	int invalid;
public:
	int cols();
	int rows();
	int invalidValue();
};

class UF
{  
private: 
	int* id; // access to component id (site indexed)  
private: 
	int count; // number of components 
	int	pluscount; //增加新类数量
	int sum;//数组大小
public:
	UF(int N)  
    {  
		// Initialize component id array. 
		sum = N;
		count = N;  
		id = new int[N];  
		//随机分配任意分类，即初始化为N种类别
		for (int i = 0; i < N; i++)  
			id[i] = i;  
    }
    UF()
    {
    	sum = 0;
		count = 0;  
		id = NULL;
    }

    UF(int N, int a)//初始化重新计数，a无用只为标记是重载
	{
		// Initialize component id array.  
		sum = N;
		count = 0;  
		pluscount = 0;
		id = new int[N];  
		//随机分配任意分类，即初始化为N种类别
		for (int i = 0; i < N; i++)  
			id[i] = -1;  
	}
    ~UF(){delete[] id;}
public:

	int Count()  
    {
    	return count; 
    } 
        
    /*void Initialize()
	{
		for (int i = 0; i < sum; i++)  
			id[i] = -1;  
		count=0;
	}*/
	
	int PlusCount()//计数增加
	{
		count++;
		pluscount++;
		return pluscount;
	}
    int SetRoot(int p, int q)//将索引为p的分类标签手动设置为q，即将树p合并到分类节点q下
	{
		if(p >= sum || q >= sum)
			return 1;
		id[p] = q;
		return 0;
	}
	int FindParent(int p)//查找索引p的父节点
	{
		return id[p];
	}
public:
	bool Connected(int p, int q)
    {
    	return Find(p) == Find(q); 
    }  
public:
	int Find(int p)//路径压缩查找  
	{   
	    //寻找p节点所在组的根节点，根节点具有性质id[root] = root  
	    while (p != id[p]) //p = id[p];  
		{
			//将节点的父节点id[p]设置为它的爷爷节点id[id[p]]，实现边找根节点，边路径压缩，使树扁平化
			id[p] = id[id[p]];
			p = id[p];
			//p = id[p];  
		}
	    return p;  
	}
public: 
	void QUnion(int p, int q)  
	{   
		// Give p and q the same root.  
		int pRoot = Find(p);  
		int qRoot = Find(q);  
		if (pRoot == qRoot)   
			return; 
		//将较大的序号合并到较小根序号中
		if(pRoot>qRoot)
			id[pRoot] = qRoot;    // 将一颗树p(即一个组)变成另外一课树q(即一个组)的子树   
		else
			id[qRoot] = pRoot; 
		count--;  
	}
	/*void Out(int n)//输出指定部分的分类结果
	{
		for(int i=0; i<n; i++)
			cout<<id[i]<<" ";
		cout<<endl;
	}*/
}; 