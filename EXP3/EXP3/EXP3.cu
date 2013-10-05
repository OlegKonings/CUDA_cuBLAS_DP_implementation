#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <cuda.h>
#include <cublas_v2.h>
#include <ctime>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIN32_LEAN_AND_MEAN
#define pb push_back 
#define all(c) (c).begin(),(c).end()
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
using namespace std;

#define _DTH cudaMemcpyDeviceToHost
#define _HTD cudaMemcpyHostToDevice
#define _DTD cudaMemcpyDeviceToDevice

#define THREADS 256
#define INF (1<<30)
#define NUM_CANDIES 50
#define PIECES 100
#define SWAPS 10000
#define DO_GPU 1


const double eps=1e-8;

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

inline bool _eq(double a,double b){return a+eps>=b && a-eps<=b;}
inline long long choose2(int n){return n>0 ? ((long long(n)*long long(n-1))>>1LL):0LL;}//n needs to be unsigned 32 bit
void showArr(const double *Arr, const int sz);

const int Scores[NUM_CANDIES]={1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99};

void CPU_version(const int C, double *ret,const int S,const int sz){

	const int totc=sz*C;
	const double tot_num_pairs= double(choose2(totc));//total number of pairs that can be made from totc candies
	const double other_num_pairs=double(choose2((C*(sz-1))));//num that are of a different type
	const double prob_diff_pair=other_num_pairs/tot_num_pairs;//

	const double tot_type_pairs=double(C*C);
	const double prob_tt=tot_type_pairs/tot_num_pairs;

	const double num_same_type_pairs=double(choose2(C));
	const double prob_same_type=num_same_type_pairs/tot_num_pairs;

	const double p=1./double(C);
	const double q=(1.-p);
	double *A=(double *)malloc(sz*sizeof(double));
	double *temp=(double *)malloc(sz*sizeof(double));

	memcpy(A,ret,sz*sizeof(double));

	for(int i=0;i<S;i++){
		for(int j=0;j<sz;j++){
			temp[j]=prob_diff_pair*A[j];
			for(int k=0;k<sz;k++){
				if(k!=j)temp[j]+=prob_tt*(p*A[k]+q*A[j]);
				else
					temp[j]+=prob_same_type*A[k];
			}
		}
		memcpy(A,temp,sz*sizeof(double));
	}
	memcpy(ret,A,sz*sizeof(double));
	free(A);
	free(temp);
}

//__device__ inline int d_choose2(int n){return n>0 ? ((n*(n-1))>>1):0;}
void inline checkError(cublasStatus_t status, const char *msg);

__global__ void GPU_eye(double *M,const int N){
	const int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=N)return;
	const int j=blockIdx.y;
	M[i*N+j]= (i==j) ? 1.0:0.0;
}

__global__ void GPU_step0(double *M, const int N,const double psame,const double deft){
	const int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=N)return;
	const int j=blockIdx.y;
	M[i*N+j]= (i==j) ? psame:deft;
}

int main(){
	char ch;
	srand(time(NULL));
	const int N=NUM_CANDIES,tot=NUM_CANDIES*PIECES,swaps=SWAPS,C=PIECES;
	double *CPU_ans=(double *)malloc(N*sizeof(double));
	double *GPU_ans=(double *)malloc(N*sizeof(double));

	for(int i=0;i<N;i++){
		CPU_ans[i]=GPU_ans[i]=double(Scores[i]);
	}
	//CPU
	UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime=timeGetTime();

	CPU_version(C,CPU_ans,swaps,N);
	
	DWORD endTime = timeGetTime();
	CPU_time=endTime-startTime;
	cout<<"CPU solution timing: "<<CPU_time<<'\n';
	DestroyMMTimer(wTimerRes, init);
	cout<<"CPU_ans= ";
	showArr(CPU_ans,N);

	//GPU
	int compute_capability=0;
	cudaDeviceProp deviceProp;
	cudaError_t err=cudaGetDeviceProperties(&deviceProp, compute_capability);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	string ss= (deviceProp.major>=3 && deviceProp.minor>=5) ? "Capable!\n":"Not Sufficient compute capability!\n";
	cout<<ss;

	cublasHandle_t handle;
	cublasStatus_t cur;
	cur = cublasCreate_v2(&handle);
	if(cur != CUBLAS_STATUS_SUCCESS){
		printf("cublasCreate returned error code %d, line(%d)\n", cur, __LINE__);
		exit(EXIT_FAILURE);
	}


	if(DO_GPU && (deviceProp.major>=3 && deviceProp.minor>=5)){


		const int num_bytes_m=N*N*sizeof(double),num_bytes_v=N*sizeof(double);
		const double alpha=1.0, beta=0.0,denom=double(choose2(tot));
		double *D_M, *D_temp,*D_N,*D_ans,*D_vt;
		err=cudaMalloc((void**)&D_M,num_bytes_m);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_temp,num_bytes_m);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_N,num_bytes_m);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_ans,num_bytes_v);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_vt,num_bytes_v);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		dim3 dimGrid0((N+THREADS-1)/THREADS,N,1);
		int temp=0;

		const double deft=double(C)/denom;
		const double p=(double(choose2(C))+double(choose2(C*(N-1))))/denom;
		const double psame=p+((1.-p)*(double(C-1)/double(C)));

		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();

		err=cudaMemcpy(D_vt,GPU_ans,num_bytes_v,_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemset(D_M,0,num_bytes_m);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		GPU_eye<<<dimGrid0,THREADS>>>(D_temp,N);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		GPU_step0<<<dimGrid0,THREADS>>>(D_M,N,psame,deft);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		for(;(1<<temp)<=swaps;temp++){//result will be D_temp
			if(swaps&(1<<temp)){
				cur=cublasDgemm_v2(handle,CUBLAS_OP_N, CUBLAS_OP_N,N,N,N,&alpha,D_temp,N,D_M,N,&beta,D_N,N);//D_temp=D_rez*D_A
				if(cur != CUBLAS_STATUS_SUCCESS){
					printf("cublasDgemm returned error code %d, line(%d)\n", cur, __LINE__);
					exit(EXIT_FAILURE);
				}
				err=cudaMemcpy(D_temp,D_N,num_bytes_m,_DTD);
				if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
			}
			cur=cublasDgemm_v2(handle,CUBLAS_OP_N, CUBLAS_OP_N,N,N,N,&alpha,D_M,N,D_M,N,&beta,D_N,N);
			if(cur != CUBLAS_STATUS_SUCCESS){
				printf("cublasDgemm returned error code %d, line(%d)\n", cur, __LINE__);
				exit(EXIT_FAILURE);
			}
			err=cudaMemcpy(D_M,D_N,num_bytes_m,_DTD);
			if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		}

		cur=cublasDgemv_v2(handle,CUBLAS_OP_N,N,N,&alpha,D_temp,N,D_vt,1,&beta,D_ans,1);
		if(cur != CUBLAS_STATUS_SUCCESS){
			printf("cublasDgemm returned error code %d, line(%d)\n", cur, __LINE__);
			exit(EXIT_FAILURE);
		}


		err=cudaMemcpy(GPU_ans,D_ans,num_bytes_v,_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		endTime = timeGetTime();
		GPU_time=endTime-startTime;
		cout<<"GPU timing: "<<GPU_time<<'\n';

		err=cudaFree(D_M);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_temp);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_N);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_ans);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_vt);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	cout<<"GPU_ans= ";
	showArr(GPU_ans,N);
	int c=0;
	for(int i=0;i<N;i++){
		if(!_eq(GPU_ans[i],CPU_ans[i]))c++;
	}
	if(GPU_time==0)GPU_time=1;
	if(c==0){//NOTE: sometimes for this problem GPU time is less than 1 ms, so for division round up to 1
		cout<<"\nSuccess! CUDA GPU implementation matches serial CPU implementation. GPU implementation was "<<double(CPU_time)/double(GPU_time)<<" faster than serial CPU version.\n";

	}else{
		cout<<"\nError in calculation!\n";

	}

	free(CPU_ans);
	free(GPU_ans);
	std::cin>>ch;
	return 0;
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}
void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}
void inline checkError(cublasStatus_t status, const char *msg){
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("%s", msg);
        exit(EXIT_FAILURE);
    }
}
void showArr(const double *Arr, const int sz){
	cout<<'\n';
	for(int i=0;i<sz;i++){
		cout<<Arr[i]<<' ';
	}
	cout<<'\n';
}





