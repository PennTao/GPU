#ifndef PART1_INL
#define PART1_INL

#include <cutil_inline.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>



//template <typename K>
int cmp(const void *a, const void *b)
{
//	return (*(K*)a - *(K*)b);
//	const int *x = a;
//	const int *y = b;
//	if(*x > *y)
//		return 1;
//	else 
//		return (*x < *y)? -1 : 0;
	if(*(int*)a > *(int*)b)
		return 1;
	else 
		return *(int*)a<*(int*)b ? -1 : 0;
}
template <typename K>
void
qsortWrapper(K * i_data,
        K * o_data,
        int num,
        Processor src,
        Processor dst) 
{
	// TODO
	if(dst == Host)
	{
		if (src == Host)
		{
			
//			for(int i=0; i < num; i++)
//			{
//				printf("%d\n",i_data[i]);
//			}
//		        
		//	unsigned short vec[] ={19,34,16,64,26,25,89,32,167,42};	
//			printf("------Start Sort H H--------\n");
		//	qsort(vec, 10, sizeof(vec[0]),cmp);
//			printf("------Sort Finished---------\n");
			
			qsort(i_data, num, sizeof(i_data[0]),cmp);
//			printf("------Sort Finished---------\n");
//			for(int i=0; i < num; i++)
//			{
//				printf("%d\n",i_data[i]);
//			}
//			printf("-----------------\n");
			cudaMemcpy(o_data, i_data, sizeof(K)*num,cudaMemcpyHostToHost);
		//	for(int i; i < num; i++)
		//	{
		//		printf("%d\n",o_data[i]);
		//	}	
		}
		else if(src == Device)
		{
			K *hst;
		//	cutilSafeCall(cudaMalloc(&hst, sizeof(K)*num));
			hst = (K*)malloc(sizeof(K)*num);
			cudaMemcpy(hst, i_data, sizeof(K)*num,cudaMemcpyDeviceToHost);
			qsort(hst, num, sizeof(K), cmp);
			cudaMemcpy(o_data, hst, sizeof(K)*num,cudaMemcpyHostToHost);
			free(hst);
		}
	}
	else if(dst == Device)
	{
		if (src == Host)
		{
			qsort(i_data, num, sizeof(K),cmp);

			cudaMemcpy(o_data, i_data, sizeof(K)*num,cudaMemcpyHostToDevice);
	//		o_data = downloadToDev(i_data, num);
//			cutilSafeCall(cudaMemcpy(o_data, i_data, sizeof(K)*num,cudaMemcpyHostToDevice));
	//		o_data = downloadToDev(i_data, num);
//			printf("HtoD, finished\n");
		}
		else if(src == Device)
		{
			K * hst;
		//	cutilSafeCall(cudaMalloc(&hst, sizeof(K)*num));
			hst = (K*)malloc(sizeof(K)*num);
			cudaMemcpy(hst, i_data, sizeof(K)*num,cudaMemcpyDeviceToHost);
			qsort(hst, num, sizeof(K), cmp);
			cudaMemcpy(o_data, hst, sizeof(K)*num,cudaMemcpyHostToDevice);
			
	//		o_data = downloadToDev(i_data, num);
			free(hst);
		}
	}
	
}

template <typename K>
void
stlWrapper(K * i_data,
        K * o_data,
        int num,
        Processor src,
        Processor dst)
{
	
	// TODO
//	printf("stl sort start:\r\n");
	if(dst == Host)
	{
		if (src == Host)
		{
			std::sort(i_data, i_data + num);
			cudaMemcpy(o_data, i_data, sizeof(K)*num,cudaMemcpyHostToHost);
		
		}
		else if(src == Device)
		{
	//		printf(" src dev, dst hst start!\n");
			K * hst;
			hst = (K*)malloc(sizeof(K)*num);		
//	cudaMalloc(&hst, sizeof(K)*num);
	//		printf("space malloced\r\n");
			cudaMemcpy(hst, i_data, sizeof(K)*num,cudaMemcpyDeviceToHost);
	//		printf("data copied from device to Host\r\n");
			std::sort(hst, hst + num);
	//		printf("device sort finished \r\n");
			cudaMemcpy(o_data, hst, sizeof(K)*num,cudaMemcpyHostToHost);
			free(hst);
		}
	}
	else if(dst == Device)
	{
		if (src == Host)
		{
			std::sort(i_data, i_data + num);
			cudaMemcpy(o_data, i_data, sizeof(K)*num,cudaMemcpyHostToDevice);
		//	o_data = downloadToDev(i_data, num); 
	//		printf("src hst, dst dev finished\r\n");
			
		}
		else if(src == Device)
		{
	//		printf("src dev, dst dev");
			K * hst;
			hst = (K*)malloc(sizeof(K)*num);	
	//	cudaMalloc(&hst, sizeof(K)*num);
			cudaMemcpy(hst, i_data, sizeof(K)*num,cudaMemcpyDeviceToHost);
			std::sort(hst, hst + num);
			cudaMemcpy(o_data, hst, sizeof(K)*num,cudaMemcpyHostToDevice);
			free(hst);
		}
	}

}

template <typename K>
void
thrustWrapper(K * i_data, 
        K * o_data, 
        int num,
        Processor src,
        Processor dst) 
{
	if(src == Host)
	{
		K* dev;
		cudaMalloc(&dev,num*sizeof(K));
		cudaMemcpy(dev, i_data, num*sizeof(K), cudaMemcpyHostToDevice);
		thrust::device_ptr<K> dev_ptr = thrust::device_pointer_cast(dev);
		thrust::sort(dev_ptr, dev_ptr + num);
		if(dst == Host)
		{
			cudaMemcpy(o_data, dev, num*sizeof(K), cudaMemcpyDeviceToHost);
		}
		else if(dst ==Device)
		{
			cudaMemcpy(o_data, dev, num*sizeof(K), cudaMemcpyDeviceToDevice);
		}
		thrust::device_free(dev_ptr);
		cudaFree(dev);
	}
	else if(src ==Device)
	{
		K* dev;
		cudaMalloc(&dev,num*sizeof(K));
		cudaMemcpy(dev, i_data, num*sizeof(K), cudaMemcpyDeviceToDevice);
		thrust::device_ptr<K> dev_ptr = thrust::device_pointer_cast(dev);
//		printf("sec == dev, dev_ptr created\n");	
		thrust::sort(dev_ptr, dev_ptr + num);
//		printf("src ==dev, dev_ptr sort finished\n");
		if(dst ==Host)
		{
			cudaMemcpy(o_data, dev, num*sizeof(K), cudaMemcpyDeviceToHost);
//			printf("dev to host copied\n");
		}
		else if(dst == Device)
		{
//			printf("just before DtoD\n");
			cudaMemcpy(o_data, dev, num*sizeof(K), cudaMemcpyDeviceToDevice);
//			printf("dev to dev copied\n");
		}
		thrust::device_free(dev_ptr);
		cudaFree(dev);
	}

}

#endif //PART1_INL
