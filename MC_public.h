/*
Author: Yihao Liang
Ph.D candidate
School of Physics and Astronomy and  Institute of Natural Sciences, Shanghai Jiao Tong University
Email: liangyihao AT sjtu.edu.cn
LICENSE: GNU GENERAL PUBLIC LICENSE Version 3
*/

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include<unistd.h>
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

#define Lambda 7.117071094
#define TAU 3.75//离子的半径
#define L 1000.0//系统半径
#define NUMBER 65536   //256*256
#define PI 3.14159265358979323846
 static void HandleError( cudaError_t err,const char *file,int line ) {
          if (err != cudaSuccess) {
              printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
              printf("Error code: %d\n",err);
              exit( EXIT_FAILURE );
          }
  }
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


typedef unsigned int uint;
typedef unsigned char uchar;

