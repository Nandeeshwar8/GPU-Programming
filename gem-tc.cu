%%writefile gem-tc.cu
#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;
// size of the tile (for simplicity use 16×16×16)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void init(half *A,half *B){
 int tid=blockIdx.x * blockDim.x + threadIdx.x;
 A[tid]=tid;
 B[tid]=tid;
}

// A: (M×K) ; B: (K×N) ; C: (M×N)
__global__ void tensorCoreGemmKernel(half *A, half *B, float *C, int M, int N, int K) {
   // each warp computes one tile of output C
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;// warps per block is one, 
    int tileRow = warpM / 4;
    int tileCol = warpM% 4;
    int mTile = tileRow * WMMA_M;
    int nTile = tileCol * WMMA_N;//value is zero.
   
   if( threadIdx.x==0)printf(" %d \n", warpM);
    if (mTile >= M || nTile >= N) return;

    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    // Initialize the output to zero
    fill_fragment(cFrag, 0.0f);
    //Compute C(1,1) of the 32x32 matrix.
        for (int i=0;i<4;i++){
               	int bind=i*16*N+nTile;//i=0-> 16, i=1 -> 16x(1+32) (0,1) ,  (1,1)
                int aind=mTile*K+i*16;//i=0->32*16, i=1 --> 16+32*16  (1,0) , (1,1)
//if(threadIdx.x==0)		printf("i=%d bind=%d aind=%d \n", i, bind,aind);
           load_matrix_sync(aFrag, A +aind,K);//load submatrix A  into registers
            load_matrix_sync(bFrag, B + bind,N);//load submatrix B into registers
            mma_sync(cFrag, aFrag, bFrag, cFrag);//do the multiplication.
         }
// writting result in to first element of tile C(0,0)//incorrect
   int cind = mTile * N + nTile;      
   store_matrix_sync(C+cind, cFrag, N, mem_row_major);
 //index in to tile C(1,1)
  // float *loc= C+32*16+16;//starting address of the tile C(1,1), element at 32nd row, 16th column
// writting result in to first element of tile C(1,1).correct
 
}

int main() {
	//half : one bit for sign, 5 bit for exponent, Fraction/Mantisa: 10 bits
	//single: one bit for sign, 8 bits for exponent, 23 bits for fraction.
    int M = 64, N = 64, K = 64;  // multiple of 16 expected by GPU tensor core.
    half *devA;
    half *devB;
    float *devC,*hostC;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&devA, M * K * sizeof(half));
    cudaMalloc(&devB,K * N * sizeof(half));
    cudaMalloc(&devC, M * N * sizeof(float));
    hostC=(float *)malloc(M*N*sizeof(float));

    cudaError_t    err=cudaGetLastError();
    init<<<4,1024>>>(devA, devB);
    cudaDeviceSynchronize();
    err=cudaGetLastError();
     if(err!=cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    err=cudaGetLastError();
    //code computes C(1,1) and stores it in C(0,0) and C(1,1), see two calls to store_matrix_sync inside the CUDA kernel
    cudaEventRecord(start, 0);//starttime recorded.
    tensorCoreGemmKernel<<<1, 512>>>(devA, devB, devC, M, N, K);//one warp, computes c(1,1)
    cudaEventRecord(stop, 0);//endtime recoorded
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f milli seconds\n", elapsedTime);
    err=cudaGetLastError();
    if(err!=cudaSuccess) printf("%s\n", cudaGetErrorString(err));
    cudaMemcpy(hostC,devC,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    printf("C[0][0]=%f\n",hostC[0*N+0]);
    printf("C[0][1]=%f\n",hostC[0*N+1]);
    printf("C[15][15]=%f\n",hostC[15*N+15]);
    printf("C[16][16]=%f\n",hostC[16*N+16]);
    printf("C[31][31]=%f\n",hostC[31*N+31]);
    printf("C[32][32]=%f\n",hostC[32*N+32]);
    printf("C[63][63]=%f\n",hostC[63*N+63]);
    int res=0;
    int  startrow=32*31;
    int startcol= 30;
    for(int i=0;i<M;i++)res+=(startrow+i)*(startcol+i*32);
    printf("expected result %d\n", res);
    res=0;
      startrow=64*63;
     startcol= 62;
    for(int i=0;i<M;i++)res+=(startrow+i)*(startcol+i*32);
    printf("expected result %d\n", res);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    free(hostC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;

}


//for a matrix of size 64x64. We have 16 tiles of size 16x16.
//(0,0) (0,1) (0,2) (0,3)
//(1,0) (1,1) (1,2) (1,3)
//(2,0) (2,1) (2,2) (2,3)
//(3,0) (3,1) (3,2) (3,3)
//starting address of the tiles
//(0,0) - 0 , (0,1) - 16, (0,2)-32, (0,3)-48
//(1,0) - X , (1,1) - X+16 , (1,2) - X+32, (1,3)-X+48
//(2,0) - Y , (2,1) - Y+16 , (2,2) - Y+32, (2,3)-Y+48
//(3,0) - Z , (3,1) - Z+16 , (3,2) - Z+32, (3,3)-Z+48
//X=64x16, Y=64x32, Z= 64x48
//warp-id/4+warp-id*16

// (0,0) (0,1)       (0,0) (0,1)
// (1,0) (1,1)       (1,0) (1,)
