/*
Author: Yihao Liang
Ph.D candidate
School of Physics and Astronomy and  Institute of Natural Sciences, Shanghai Jiao Tong University
Email: liangyihao AT sjtu.edu.cn
LICENSE: GNU GENERAL PUBLIC LICENSE Version 3
*/
/*
本代码旨在对粒子进行快速移动
External function:
void init_move();//用于初始化MC的运行环境
void move();//用于将所有的粒子进行一次移动
*/
#include "MC_public.h"
#define TILESIZE 512
#define BLOCKSIZE TILESIZE
#define BLOCKNUM (NUMBER/TILESIZE)

#define MASK 0xffffffe0 //Warpsize==32专用
#define MOVER 1000.0///单次移动的范围
extern float4*particles_host,*particles_dev;/*particles.x y z分别记录粒子的坐标 particles.w记录粒子的价数*/
extern float4*rnum_dev;//存储程序运行时所需随机数的空间,大小为4*NUMBER
int*isready;//放在Global Memory 上,用于标示thread block是否完成工作

__device__ double atomicAdd(double* address, double val)//定义double型的AtomicAdd
{
unsigned long long int* address_as_ull =
(unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;
do {
assumed = old;
old = atomicCAS(address_as_ull, assumed,
__double_as_longlong(val +
__longlong_as_double(assumed)));
// Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN)
} while (assumed != old);
return __longlong_as_double(old);
}

__device__ double Pairwise_dE(float4 X,float4 NX,float4 YY,uchar*isoverlap){
	double dx,dy,dz,inv_dr;
	dx=NX.x-YY.x;
	dy=NX.y-YY.y;
	dz=NX.z-YY.z;
	inv_dr=rsqrt(dx*dx+dy*dy+dz*dz);
	isoverlap[0]=(isoverlap[0]||(inv_dr*2*TAU>1));
	dx=X.x-YY.x;
	dy=X.y-YY.y;
	dz=X.z-YY.z;
	inv_dr-=rsqrt(dx*dx+dy*dy+dz*dz);
	return X.w*YY.w*inv_dr;
}

__global__ void MULTIMOVE(float4*X_dev,float4*rnum_dev,int*isready)//X_dev为坐标,rnum_dev为所需的随机数,isready显示每个块是否更新
{
__shared__ float4 Y[TILESIZE];
float4 X,NX;

double dE;
uchar isoverlap;
dE=0;isoverlap=0;
X=X_dev[blockIdx.x*TILESIZE+threadIdx.x];
NX=rnum_dev[blockIdx.x*TILESIZE+threadIdx.x];
NX.x=X.x+(NX.x-0.5)*MOVER;
NX.y=X.y+(NX.y-0.5)*MOVER;
NX.z=X.z+(NX.z-0.5)*MOVER;
//(X.x,X.y,X.z)为粒子的旧坐标,(NX.x,NX.y,NX.z)为粒子的新坐标,X.w为粒子的电荷价数, NX.w为移动该粒子时所用的概率p
isoverlap=(NX.x*NX.x+NX.y*NX.y+NX.z*NX.z>L*L);

//下面求上对角块对当下粒子的作用
for(int Tile=blockIdx.x+1;Tile<BLOCKNUM;Tile++){
	Y[threadIdx.x]=X_dev[Tile*TILESIZE+threadIdx.x];
	__syncthreads();
	#pragma unroll 128
	for(int l=0;l<TILESIZE;l++)dE+=Pairwise_dE(X,NX,Y[l],&isoverlap);
    __syncthreads();
}

//下面求对角快中上对角元素作用
Y[threadIdx.x]=X;
__syncthreads();
for(int l=threadIdx.x&MASK;l<TILESIZE;l++)if(l>threadIdx.x)dE+=Pairwise_dE(X,NX,Y[l],&isoverlap);
//下面求下对角块对当下粒子的作用
for(int Tile=0;Tile<blockIdx.x;Tile++){
	if(threadIdx.x==0)while(!atomicAdd(&isready[Tile],0));
	__syncthreads();
	Y[threadIdx.x]=X_dev[Tile*TILESIZE+threadIdx.x];
	__syncthreads();
	#pragma unroll 128
	for(int l=0;l<TILESIZE;l++)dE+=Pairwise_dE(X,NX,Y[l],&isoverlap);
    __syncthreads();
}


//下面进行移动并求对角块中下对角元素的作用
Y[threadIdx.x]=X;
__syncthreads();

//下面的循环将对本组待移动粒子进行移动,每次循环将确定一个粒子(i)的位置,并且将i对所有粒子j(j>i)的作用计入寄存器///////
for(int l=0;l<TILESIZE;l++){
	///值班线程tid==l要等待其它Block在l位置的原子计算完成后决定是否移动,如果移动,则更新shared mem中的变量
	if((threadIdx.x==l)&&(!isoverlap)&&(NX.w<exp(-Lambda*dE)))
		  {
		    NX.w=X.w;Y[l]=NX;
		  }
	__syncthreads();
	///其他线程等待值班线程,然后对于tid>l的线程,将一起更新自己的能量项和overlap项
	if(threadIdx.x>l)dE+=Pairwise_dE(X,NX,Y[l],&isoverlap);
}
///将新位置信息写入global mem中
X_dev[blockIdx.x*BLOCKSIZE+threadIdx.x]=Y[threadIdx.x];
__syncthreads();
if((threadIdx.x==0)&&(blockIdx.x!=BLOCKNUM-1))atomicAdd(&isready[blockIdx.x],1);
if(blockIdx.x==BLOCKNUM-1)//若为最后一个block,原理上讲是最后一个结束,结束后应当将就绪标记清零
	for(int k=threadIdx.x;k<BLOCKNUM;k+=BLOCKSIZE)if(k<BLOCKNUM)isready[k]=0;
}
void init_move()
{
	cudaMalloc((void**)&isready,BLOCKNUM*sizeof(int));
	int isready_host[BLOCKNUM];
	for(int k=0;k<BLOCKNUM;k++)isready_host[k]=0;
	cudaMemcpy(isready,isready_host,sizeof(isready_host),cudaMemcpyHostToDevice);
}
void multimove()
{
	MULTIMOVE<<<BLOCKNUM,BLOCKSIZE>>>(particles_dev,rnum_dev,isready);
	//cudaThreadSynchronize();fflush(stdout);
	//HANDLE_ERROR(cudaPeekAtLastError());
}
