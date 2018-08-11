#include "operator.h"

void createPatchMap(int* img_data, int* labelMap, int *pixNum, int *perimeter, int data_height, int width, std::map<int, Patch>& mapPatch, int data_start, int data_end, int task_start, int task_end)
{
	if((data_start == task_start)&&(data_end == task_end))
	{
		for (int i = 0; i < data_height * width; i++)
		{
			if (pixNum[i]>0)
			{
				int tempPatchIndex = labelMap[i];//labeling
				int tempPatchValue = (int)img_data[i];
				int tempPatchCount = pixNum[i];
				int tempPatchContour = perimeter[i];

				Patch temp_patch;
				temp_patch.PatchIndex = tempPatchIndex;
				temp_patch.PatchValue = tempPatchValue;
				temp_patch.PatchCount = tempPatchCount;
				temp_patch.PatchContour = tempPatchContour;
				mapPatch.insert(pair<int, Patch>(tempPatchIndex, temp_patch));
				// mapPatch.insert(pair<int, Patch>(mapPatchCount + mapPatch.size(), temp_patch));

			}
		}
	}
	else if (data_start == task_start)
	{
		for (int i = 0; i < (data_height - 1)*width; i++)
		{
			if (pixNum[i]>0)
			{
				int tempPatchIndex = labelMap[i];//labeling
				int tempPatchValue = (int)img_data[i];
				int tempPatchCount = pixNum[i];
				int tempPatchContour = perimeter[i];

				Patch temp_patch;
				temp_patch.PatchIndex = tempPatchIndex;
				temp_patch.PatchValue = tempPatchValue;
				temp_patch.PatchCount = tempPatchCount;
				temp_patch.PatchContour = tempPatchContour;
				mapPatch.insert(pair<int, Patch>(tempPatchIndex, temp_patch));
				// mapPatch.insert(pair<int, Patch>(mapPatchCount + mapPatch.size(), temp_patch));

			}
		}
	}
	else if (data_end == task_end)
	{
		img_data += width;
		for (int i = 0; i < (data_height - 1)*width; i++)
		{
			if (pixNum[i]>0)
			{
				int tempPatchIndex = labelMap[i];//labeling
				int tempPatchValue = (int)img_data[i];
				int tempPatchCount = pixNum[i];
				int tempPatchContour = perimeter[i];

				Patch temp_patch;
				temp_patch.PatchIndex = tempPatchIndex;
				temp_patch.PatchValue = tempPatchValue;
				temp_patch.PatchCount = tempPatchCount;
				temp_patch.PatchContour = tempPatchContour;
				mapPatch.insert(pair<int, Patch>(tempPatchIndex, temp_patch));
				// mapPatch.insert(pair<int, Patch>(mapPatchCount + mapPatch.size(), temp_patch));

			}
		}
	}
	else
	{
		img_data += width;
		for (int i = 0; i < (data_height - 2)*width; i++)
		{
			if (pixNum[i]>0)
			{
				int tempPatchIndex = labelMap[i];//labeling
				int tempPatchValue = (int)img_data[i];
				int tempPatchCount = pixNum[i];
				int tempPatchContour = perimeter[i];

				Patch temp_patch;
				temp_patch.PatchIndex = tempPatchIndex;
				temp_patch.PatchValue = tempPatchValue;
				temp_patch.PatchCount = tempPatchCount;
				temp_patch.PatchContour = tempPatchContour;
				mapPatch.insert(pair<int, Patch>(tempPatchIndex, temp_patch));
				// mapPatch.insert(pair<int, Patch>(mapPatchCount + mapPatch.size(), temp_patch));

			}
		}
	}
	 //map<int, Patch>::iterator iter;
	 //for(iter=mapPatch.begin();iter!=mapPatch.end();iter++)
	 //{
	 //	std::cout << iter->first << "," << iter->second.PatchIndex<< "," << iter->second.PatchValue<< "," << iter->second.PatchCount<< "," << iter->second.PatchContour << std::endl;
	 //}

}


void mergePatchMap(std::map<int, Patch>& mapPatch1, std::map<int, Patch>& mapPatch2)
{
	// for (map<int, Patch>::iterator it2 = mapPatch2.begin(); it2 != mapPatch2.end(); ++it2)
	// {
	// 	int findKey = it2->first;
	// 	map<int, Patch>::iterator it = mapPatch1.find(findKey);
	// 	if (mapPatch1.end() == it)
	// 	{
	// 		//û���ҵ�����mapPatch2�и�Ԫ�ؼ���mapPatch1
	// 		mapPatch1[findKey] = it2->second;
	// 	}
	// 	else
	// 	{
	// 		it->second.PatchCount += it2->second.PatchCount;
	// 		it->second.PatchContour += it2->second.PatchContour;
	// 	}
	// }
	for (map<int, Patch>::iterator it2 = mapPatch2.begin(); it2 != mapPatch2.end(); ++it2)
	{
		//û���ҵ�����mapPatch2�и�Ԫ�ؼ���mapPatch1
		mapPatch1[it2->first] = it2->second;
	}
}
void outputPatchToFile(std::map<int, Patch>& mapPatch, std::string Name)
{
	std::ofstream f;
	char filename[20];//�˴��������ܳ��ֵ���󳤶ȣ��������ֶ�ջ�����
	strcpy(filename, (Name + "_patch.csv").c_str());
	f.open(filename, std::ios::out);
	f << "ID" << ","
		<< "pixelValue" << ","
		<< "pixelNum" << ","
		<< "perimeter" << ","
		<< std::endl;
	for (map<int, Patch>::iterator it = mapPatch.begin(); it != mapPatch.end(); ++it)
	{
		f << it->first << ","
			<< it->second.PatchValue << ","
			<< it->second.PatchCount << ","
			<< it->second.PatchContour << endl;
	}
	f.close();
}
void outputPatchToFile(std::map<int, Patch>& mapPatch, UF &Quf, std::string Name)
{
	std::ofstream f;
	char filename[20];//�˴��������ܳ��ֵ���󳤶ȣ��������ֶ�ջ�����
	strcpy(filename, (Name + "_patch.csv").c_str());
	f.open(filename, std::ios::out);
	f << "ID" << ","
		<< "pixelValue" << ","
		<< "pixelNum" << ","
		<< "perimeter" << ","
		<< std::endl;
	map<int, Patch>::iterator it = mapPatch.begin();
	for (; it != mapPatch.end(); )
	{
		int curPatIdx = it->first;
		if(curPatIdx!=Quf.FindParent(curPatIdx))//���Ǹ��ڵ�
		{
			int root=Quf.Find(curPatIdx);
			map<int, Patch>::iterator it1 = mapPatch.find(root);
			it1->second.PatchCount += it->second.PatchCount;
			it1->second.PatchContour += it->second.PatchContour;
			mapPatch.erase(it++);
		}
		else
		{
			++it;
		}
	}

	for (map<int, Patch>::iterator it = mapPatch.begin(); it != mapPatch.end(); ++it)
	{
		f << it->first << ","
			<< it->second.PatchValue << ","
			<< it->second.PatchCount << ","
			<< it->second.PatchContour << endl;
	}

	f.close();
}
void outputPatchClass(PClass *pClass, int &_pClscount)
{
	ofstream ofile("pclass.csv", ios::out | ios::trunc);//patch_mpi7out
	ofile << "pClass_Total=" << _pClscount << endl;
	ofile << "PClassValue,PClassCount\n";
	int j = 0;
	for (int i = 0; i < CLASS_MAX; ++i)
	{
		if (pClass[i].PClassCount != 0)
		{
			ofile << j << "," << pClass[i].PClassValue << "," << pClass[i].PClassCount << "\n";
			j++;
		}
	}
	ofile.close();
}


int findMerge(int width, int BGvalue, int* Meg, int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel)
{
	int Meg_count = 0;//��ʼ����Meg
	int center;
	for(int i = 0; i < width; i++)
	{
		int	LastLabel = -1;//�ϴα�����
		int	CurLabel = -1;
		center = h_subDataFirst[i];//����һ����ÿ��pixelΪ���ģ�����ģ�����
		if(center == BGvalue)
			continue;
		if(center == h_subDataSecond[i])//ͬһ������һ����������һ��ͼ������һ��
		{
			LastLabel = lastRowLabel[i];//�ϴα�����
			CurLabel  = firstRowLabel[i];
			int	repetition = 0;//�Ƿ��ظ�
			for(int i = 0; i < Meg_count; i++)
			{
				if((Meg[2*i] == LastLabel) && (Meg[2*i+1] == CurLabel))
				{
					repetition = 1;
					break;
				}
			}
			if(!repetition)
			{
				Meg[Meg_count*2] = LastLabel;
				Meg[Meg_count*2+1] = CurLabel;
				Meg_count++;
			}	
		}
		if((i - 1 >= 0) && (center == h_subDataSecond[i-1]))//��һ�������������һ��ͼ������һ��
		{	
			LastLabel = lastRowLabel[i];//�ϴα�����
			CurLabel = firstRowLabel[i-1];
									
			int	repetition = 0;//�Ƿ��ظ�
			for(int i = 0; i < Meg_count; i++)
			{
				if((Meg[2*i] == LastLabel) && (Meg[2*i+1] == CurLabel))
				{
					repetition=1;
					break;
				}
			}
			if(!repetition)
			{
				Meg[Meg_count*2]=LastLabel;
				Meg[Meg_count*2+1]=CurLabel;
				Meg_count++;
			}	
		}
		if((i + 1 < width) && (center == h_subDataSecond[i+1]))//��һ���������ұ���һ��ͼ������һ��
		{	
			LastLabel = lastRowLabel[i];//�ϴα�����
			CurLabel = firstRowLabel[i+1];
								
			int	repetition = 0;//�Ƿ��ظ�
			for(int i = 0; i < Meg_count; i++)
			{
				if((Meg[2*i] == LastLabel) && (Meg[2*i+1] == CurLabel))
				{
					repetition=1;
					break;
				}
			}
			if(!repetition)
			{
				Meg[Meg_count*2] = LastLabel;
				Meg[Meg_count*2+1] = CurLabel;
				Meg_count++;
			}	
		}
	}
	return Meg_count;
}

void MergePatch( int* Meg, int Meg_count, UF &Quf)
{
	for(int i = 0; i < Meg_count; i++)
	{
		if(Meg[2*i] != -1)
		{
			int	cur_index = Meg[2*i+1];
			int	last_index = Quf.Find(Meg[2*i]);
			for(int j = i + 1; j < Meg_count; j++)//��������ĺϲ������Ƿ��к͵�ǰ��cur_indexһ���ģ���ͨU�ͣ�
			{
				if(Meg[j*2+1] == cur_index)
				{
					//merge
					int	cur_lastindex = Quf.Find(Meg[j*2]);
					Quf.QUnion( cur_lastindex, cur_index );//�ϲ����
					Meg[j*2] = Meg[j*2+1] = -1;//�������
				}
			}
			//merge 
			Quf.QUnion( last_index, cur_index );
			Meg[i*2] = Meg[i*2+1] = -1;//����Ѻϲ�
		}
	}
}

void MergePatchArray(int width, int blockNum, int BGvalue, int* mergeStructArray,UF &Quf)
{
	int *Meg = NULL;	//�ϲ�����
	int Merge_count = 0;	//�ϲ�����
	int i;
	for(i = 0; i< blockNum - 1; i++)	//mergeStructArraySize = blockNum-1
	{
		Meg = (int *)malloc(sizeof(int) * width * 2);
		// int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel
		Merge_count = findMerge(width, BGvalue, Meg, mergeStructArray+i*width*4+width*2,mergeStructArray+i*width*4+width*3, mergeStructArray+i*width*4, mergeStructArray+i*width*4+width*1);
		MergePatch(Meg, Merge_count, Quf);
		delete[] Meg;
		Meg = NULL;
		Merge_count = 0;
	}
}