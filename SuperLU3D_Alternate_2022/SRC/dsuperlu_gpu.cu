

/*! @file
 * \brief Descriptions and declarations for structures used in GPU
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 * March 14, 2021 version 7.0.0
 * </pre>
 */

//#define GPU_DEBUG

#define PRNTlevel 2
#define DEBUGlevel 1

#include "mpi.h"
// #include "sec_structs.h"
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "math.h"

#undef Reduce
#include "cub/cub.cuh"
//#include <thrust/system/cuda/detail/cub/cub.cuh>

#include "dlustruct_gpu.h"

#ifdef test
#include <string.h>
typedef int integer;
typedef int logical;

typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;
#endif

#undef Torch
//extern "C" {
//	void cblas_daxpy(const int N, const double alpha, const double *X,
//	                 const int incX, double *Y, const int incY);
//}

/*error reporting functions */
//static
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


// cublasStatus_t checkCublas(cublasStatus_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
// 	if (result != CUBLAS_STATUS_SUCCESS)
// 	{
// 		fprintf(stderr, "CUDA Blas Runtime Error: %s\n", cublasGetErrorString(result));
// 		assert(result == CUBLAS_STATUS_SUCCESS);
// 	}
// #endif
// 	return result;
// }


// #define UNIT_STRIDE

#ifdef Torch
__global__ void setzero_int(int_t size, int_t *data)
{
	int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < size) {
		data[idx] = 0;
	}
}

void zeros_int(int_t *data, int_t size)
{
	__global__ void setzero_int(int_t size, int_t *data);

	dim3 block(32);
	dim3 grid(size / block.x + 1);
	
	setzero_int << <grid, block >> >(size,data);
}

void fillvalue_int(int_t *data, int_t size,int_t value)
{
	__global__ void setvalue_int(int_t size, int_t *data, int_t value);

	dim3 block(32);
	dim3 grid(size / block.x + 1);
	
	setvalue_int << <grid, block >> >(size, data,value);
}

__global__ void setvalue_int(int_t size, int_t *data, int_t value)
{
	int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx] = value;
	}
}


#endif

#ifdef Torch_0412
void isnan_double(double *data, int_t size, bool *result)
{
	__global__ void isnan_test(int_t size, double *data, bool *result);

	dim3 block(32);
	dim3 grid(size / block.x + 1);
	bool h_result = true;
	bool *d_result;
	cudaMalloc((void**)&d_result, sizeof(bool));
	cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
	
	isnan_test << <grid, block >> >(size, data, d_result);

	cudaMemcpy(result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
}

__global__ void isnan_test(int_t size, double *data, bool *result)
{

	int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		if (isnan(data[idx])){
			*result = false;
		}
	}
}

void isnan_double_host(double *data, int_t size, bool *result)
{
	__global__ void isnan_test(int_t size, double *data, bool *result);

	dim3 block(32);
	dim3 grid(size / block.x + 1);
	bool h_result = true;
	bool *d_result;
	cudaMalloc((void**)&d_result, sizeof(bool));
	cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);

	double *d_data;
	cudaMalloc((void**)&d_data, size * sizeof(double));
	cudaMemcpy(d_data, &data, size * sizeof(double), cudaMemcpyHostToDevice);
	
	isnan_test << <grid, block >> >(size, d_data, d_result);

	cudaMemcpy(result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_data);

}
#endif

#if 0  ////////// this routine is not used anymore
__device__ inline
void device_scatter_l (int_t thread_id,
                       int_t nsupc, int_t temp_nbrow,
                       int_t *usub, int_t iukp, int_t klst,
                       double *nzval, int_t ldv,
                       double *tempv, int_t nbrow,
                       // int_t *indirect2_thread
                       int *indirect2_thread
                      )
{


	int_t segsize, jj;

	for (jj = 0; jj < nsupc; ++jj)
	{
		segsize = klst - usub[iukp + jj];
		if (segsize)
		{
			if (thread_id < temp_nbrow)
			{

#ifndef UNIT_STRIDE
				nzval[indirect2_thread[thread_id]] -= tempv[thread_id];
#else
				nzval[thread_id] -= tempv[thread_id]; /*making access unit strided*/
#endif
			}
			tempv += nbrow;
		}
		nzval += ldv;
	}
}
#endif ///////////// not used

#define THREAD_BLOCK_SIZE  512  /* Sherry: was 192. should be <= MAX_SUPER_SIZE */

__device__ inline
void ddevice_scatter_l_2D (int thread_id,
                          int nsupc, int temp_nbrow,
                          int_t *usub, int iukp, int_t klst,
                          double *nzval, int ldv,
                          const double *tempv, int nbrow,
                          int *indirect2_thread,
                          int nnz_cols, int ColPerBlock,
                          int *IndirectJ3
                         )
{
    int i;
    if ( thread_id < temp_nbrow * ColPerBlock )	{
	int thread_id_x  = thread_id % temp_nbrow;
	int thread_id_y  = thread_id / temp_nbrow;

#define UNROLL_ITER 8

#pragma unroll 4
	for (int col = thread_id_y; col < nnz_cols ; col += ColPerBlock)
	{
   	    i = ldv * IndirectJ3[col] + indirect2_thread[thread_id_x];
	    nzval[i] -= tempv[nbrow * col + thread_id_x];
	}
    }
}

/* Sherry: this routine is not used */
#if 0
__global__
void cub_scan_test(void)
{
	int thread_id = threadIdx.x;
	typedef cub::BlockScan<int, MAX_SUPER_SIZE > BlockScan; /*1D int data type*/

	__shared__ typename BlockScan::TempStorage temp_storage; /*storage temp*/

	__shared__ int IndirectJ1[MAX_SUPER_SIZE];
	__shared__ int IndirectJ2[MAX_SUPER_SIZE];

	if (thread_id < MAX_SUPER_SIZE)
	{
		IndirectJ1[thread_id] = (thread_id + 1) % 2;
	}

	__syncthreads();
	if (thread_id < MAX_SUPER_SIZE)
		BlockScan(temp_storage).InclusiveSum (IndirectJ1[thread_id], IndirectJ2[thread_id]);


	if (thread_id < MAX_SUPER_SIZE)
		printf("%d %d\n", thread_id, IndirectJ2[thread_id]);

}
#endif  // not used


__device__ inline
void device_scatter_u_2D (int thread_id,
                          int temp_nbrow,  int nsupc,
                          double * ucol,
                          int_t * usub, int iukp,
                          int_t ilst, int_t klst,
                          int_t * index, int iuip_lib,
                          double * tempv, int nbrow,
                          int *indirect,
                          int nnz_cols, int ColPerBlock,
                          int *IndirectJ1,
                          int *IndirectJ3
                         )
{
    int i;

    if ( thread_id < temp_nbrow * ColPerBlock )
    {    
	/* 1D threads are logically arranged in 2D shape. */
	int thread_id_x  = thread_id % temp_nbrow;
	int thread_id_y  = thread_id / temp_nbrow;

#pragma unroll 4
	for (int col = thread_id_y; col < nnz_cols ; col += ColPerBlock)
	{
           i = IndirectJ1[IndirectJ3[col]] + indirect[thread_id_x];
	    ucol[i] -= tempv[nbrow * col + thread_id_x];
	}
    }
}


__device__ inline
void device_scatter_u (int_t thread_id,
                       int_t temp_nbrow,  int_t nsupc,
                       double * ucol,
                       int_t * usub, int_t iukp,
                       int_t ilst, int_t klst,
                       int_t * index, int_t iuip_lib,
                       double * tempv, int_t nbrow,
                       // int_t *indirect
                       int *indirect
                      )
{
	int_t segsize, fnz, jj;
	for (jj = 0; jj < nsupc; ++jj)
	{
	    segsize = klst - usub[iukp + jj];
	    fnz = index[iuip_lib++];
	    ucol -= fnz;
	    if (segsize) {            /* Nonzero segment in U(k.j). */
		if (thread_id < temp_nbrow)
		{
#ifndef UNIT_STRIDE
		    ucol[indirect[thread_id]] -= tempv[thread_id];
#else
		    /* making access unit strided;
		       it doesn't work; it is for measurements */
		    ucol[thread_id] -= tempv[thread_id];
#endif
		}
		tempv += nbrow;
	    }
	    ucol += ilst ;
	}
}

#ifdef SuperLargeScaleGPU
__global__
void Scatter_GPU_kernel_predict(int_t streamId, int_t ii_st, int_t jj_st, int_t npcol, int_t nprow, dLUstruct_gpu_t * A_gpu)
{
	/* thread block assignment: this thread block is
	   assigned to block (lb, j) in 2D grid */
	int lb = blockIdx.x + ii_st;
	int j  = blockIdx.y + jj_st;

	int_t *xsup = A_gpu->xsup;
	Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info;
	Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info;

	int jb = Ublock_info[j].jb;
	int nsupc = SuperSize (jb);
	int ljb = jb / npcol;

	int ib   = Remain_info[lb].ib;
	
	if (ib < jb)  /*scatter U code */
	{
		int lib =  ib / nprow;   /* local index of row block ib */		
	}
	else     /* ib >= jb, scatter L code */
	{
		
		if(A_gpu->isGPUUsed_Lnzval_bc_ptr[ljb] == GPUUnused){
			A_gpu->isGPUUsed_Lnzval_bc_ptr[ljb] = GPUNeeded;
		}
		
	}
}

__global__
void Scatter_GPU_kernel_test(int_t streamId, int_t ii_st, int_t jj_st, int_t npcol, int_t nprow, dLUstruct_gpu_t * A_gpu)
{
	/* thread block assignment: this thread block is
	   assigned to block (lb, j) in 2D grid */
	int lb = blockIdx.x + ii_st;
	int j  = blockIdx.y + jj_st;

	double *LnzvalVec = A_gpu->LnzvalVec;
	int_t *LnzvalPtr = A_gpu->LnzvalPtr;

	int_t *xsup = A_gpu->xsup;
	Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info;
	Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info;

	int jb = Ublock_info[j].jb;
	int nsupc = SuperSize (jb);
	int ljb = jb / npcol;

	int ib   = Remain_info[lb].ib;
	
	if (ib < jb)  /*scatter U code */
	{
		int lib =  ib / nprow;   /* local index of row block ib */		
	}
	else     /* ib >= jb, scatter L code */
	{
		
		if(A_gpu->isEmpty){
			double *nzval = &LnzvalVec[LnzvalPtr[ljb]];
			nzval[0] ++;
			*(A_gpu->testdata) = 2;	
		}
		
	}	
}
#endif

__global__
void Scatter_GPU_kernel(
    int_t streamId,
    int_t ii_st, int_t ii_end,
    int_t jj_st, int_t jj_end, /* defines rectangular Schur block to be scatter */
    int_t klst,
    int_t jj0,   /* 0 on entry */
    int_t nrows, int_t ldt, int_t npcol, int_t nprow,
    dLUstruct_gpu_t * A_gpu)
{

	/* initializing pointers */
	int_t *xsup = A_gpu->xsup;
	int_t *UrowindPtr = A_gpu->UrowindPtr;
	int_t *UrowindVec = A_gpu->UrowindVec;
	int_t *UnzvalPtr = A_gpu->UnzvalPtr;
	double *UnzvalVec = A_gpu->UnzvalVec;
	int_t *LrowindPtr = A_gpu->LrowindPtr;
	int_t *LrowindVec = A_gpu->LrowindVec;
	int_t *LnzvalPtr = A_gpu->LnzvalPtr;	
	double *LnzvalVec = A_gpu->LnzvalVec;
	double *bigV = A_gpu->scubufs[streamId].bigV;
	local_l_blk_info_t *local_l_blk_infoVec = A_gpu->local_l_blk_infoVec;
	local_u_blk_info_t *local_u_blk_infoVec = A_gpu->local_u_blk_infoVec;
	int_t *local_l_blk_infoPtr = A_gpu->local_l_blk_infoPtr;
	int_t *local_u_blk_infoPtr = A_gpu->local_u_blk_infoPtr;
	Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info;
	Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info;
	int_t *lsub  = A_gpu->scubufs[streamId].lsub;
	int_t *usub  = A_gpu->scubufs[streamId].usub;

	/* thread block assignment: this thread block is
	   assigned to block (lb, j) in 2D grid */
	int lb = blockIdx.x + ii_st;
	int j  = blockIdx.y + jj_st;
	__shared__ int indirect_thread[MAX_SUPER_SIZE];  /* row-wise */
	__shared__ int indirect2_thread[MAX_SUPER_SIZE]; /* row-wise */
	__shared__ int IndirectJ1[THREAD_BLOCK_SIZE];    /* column-wise */
	__shared__ int IndirectJ3[THREAD_BLOCK_SIZE];    /* column-wise */

	/* see CUB page https://nvlabs.github.io/cub/. Implement threads collectives */
	typedef cub::BlockScan<int, THREAD_BLOCK_SIZE> BlockScan; /*1D int data type*/
	__shared__ typename BlockScan::TempStorage temp_storage; /*storage temp*/

	int thread_id = threadIdx.x;

	int iukp = Ublock_info[j].iukp;
	int jb = Ublock_info[j].jb;
	int nsupc = SuperSize (jb);
	int ljb = jb / npcol;

	double *tempv1;
	if (jj_st == jj0)
	{
		tempv1 = (j == jj_st) ? bigV
		         : bigV + Ublock_info[j - 1].full_u_cols * nrows;
	}
	else
	{
		tempv1 = (j == jj_st) ? bigV
		         : bigV + (Ublock_info[j - 1].full_u_cols -
		                   Ublock_info[jj_st - 1].full_u_cols) * nrows;
	}

	/* # of nonzero columns in block j  */
	int nnz_cols = (j == 0) ? Ublock_info[j].full_u_cols
	               : (Ublock_info[j].full_u_cols - Ublock_info[j - 1].full_u_cols);
	int cum_ncol = (j == 0) ? 0	: Ublock_info[j - 1].full_u_cols;

	int lptr = Remain_info[lb].lptr;
	int ib   = Remain_info[lb].ib;
	int temp_nbrow = lsub[lptr + 1]; /* number of rows in the current L block */
	lptr += LB_DESCRIPTOR;

	int_t cum_nrow;
	if (ii_st == 0)
	{
		cum_nrow = (lb == 0 ? 0 : Remain_info[lb - 1].FullRow);
	}
	else
	{
		cum_nrow = (lb == 0 ? 0 : Remain_info[lb - 1].FullRow - Remain_info[ii_st - 1].FullRow);
	}

	tempv1 += cum_nrow;

	if (ib < jb)  /*scatter U code */
	{
		int ilst = FstBlockC (ib + 1);
		int lib =  ib / nprow;   /* local index of row block ib */
		int_t *index = &UrowindVec[UrowindPtr[lib]];

		int num_u_blocks = index[0];

		int ljb = (jb) / npcol; /* local index of column block jb */

		/* Each thread is responsible for one block column */
		__shared__ int ljb_ind;
		/*do a search ljb_ind at local row lib*/
		int blks_per_threads = CEILING(num_u_blocks, THREAD_BLOCK_SIZE);
		for (int i = 0; i < blks_per_threads; ++i)
			/* each thread is assigned a chunk of consecutive U blocks to search */
		{
			/* only one thread finds the block index matching ljb */
			if (thread_id * blks_per_threads + i < num_u_blocks &&
			        local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + thread_id * blks_per_threads + i ].ljb == ljb)
			{
				ljb_ind = thread_id * blks_per_threads + i;
			}
		}
		__syncthreads();

		int iuip_lib = local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + ljb_ind].iuip;
		int ruip_lib = local_u_blk_infoVec[ local_u_blk_infoPtr[lib] + ljb_ind].ruip;
		iuip_lib += UB_DESCRIPTOR;
		double *Unzval_lib = &UnzvalVec[UnzvalPtr[lib]];
		double *ucol = &Unzval_lib[ruip_lib];

		if (thread_id < temp_nbrow) /* row-wise */
		{
			/* cyclically map each thread to a row */
			indirect_thread[thread_id] = (int) lsub[lptr + thread_id];
		}

		/* column-wise: each thread is assigned one column */
		if (thread_id < nnz_cols)
			IndirectJ3[thread_id] = A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
		/* indirectJ3[j] == kk means the j-th nonzero segment
		   points to column kk in this supernode */

		__syncthreads();

		/* threads are divided into multiple columns */
		int ColPerBlock = THREAD_BLOCK_SIZE / temp_nbrow;

		if (thread_id < THREAD_BLOCK_SIZE)
			IndirectJ1[thread_id] = 0;

		if (thread_id < THREAD_BLOCK_SIZE)
		{
			if (thread_id < nsupc)
			{
				/* fstnz subscript of each column in the block */
				IndirectJ1[thread_id] = index[iuip_lib + thread_id];
			}
		}

		/* perform an inclusive block-wide prefix sum among all threads */
		if (thread_id < THREAD_BLOCK_SIZE)
			BlockScan(temp_storage).InclusiveSum(IndirectJ1[thread_id], IndirectJ1[thread_id]);

		if (thread_id < THREAD_BLOCK_SIZE)
			IndirectJ1[thread_id] = -IndirectJ1[thread_id] + ilst * thread_id;

		__syncthreads();

		device_scatter_u_2D (
		    thread_id,
		    temp_nbrow,  nsupc,
		    ucol,
		    usub, iukp,
		    ilst, klst,
		    index, iuip_lib,
		    tempv1, nrows,
		    indirect_thread,
		    nnz_cols, ColPerBlock,
		    IndirectJ1,
		    IndirectJ3 );

	}
	else     /* ib >= jb, scatter L code */
	{

		int rel;
		double *nzval;
		int_t *index = &LrowindVec[LrowindPtr[ljb]];
		int num_l_blocks = index[0];
		int ldv = index[1];

		int fnz = FstBlockC (ib);
		int lib = ib / nprow;

		__shared__ int lib_ind;
		/*do a search lib_ind for lib*/
		int blks_per_threads = CEILING(num_l_blocks, THREAD_BLOCK_SIZE);
		for (int i = 0; i < blks_per_threads; ++i)
		{
			if (thread_id * blks_per_threads + i < num_l_blocks &&
			        local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + thread_id * blks_per_threads + i ].lib == lib)
			{
				lib_ind = thread_id * blks_per_threads + i;
			}
		}
		__syncthreads();

		int lptrj = local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + lib_ind].lptrj;
		int luptrj = local_l_blk_infoVec[ local_l_blk_infoPtr[ljb] + lib_ind].luptrj;
		lptrj += LB_DESCRIPTOR;
		int dest_nbrow = index[lptrj - 1];

		if (thread_id < dest_nbrow)
		{
			rel = index[lptrj + thread_id] - fnz;
			indirect_thread[rel] = thread_id;
		}
		__syncthreads();

		/* can be precalculated */
		if (thread_id < temp_nbrow)
		{
			rel = lsub[lptr + thread_id] - fnz;
			indirect2_thread[thread_id] = indirect_thread[rel];
		}
		if (thread_id < nnz_cols)
			IndirectJ3[thread_id] = (int) A_gpu->scubufs[streamId].usub_IndirectJ3[cum_ncol + thread_id];
		__syncthreads();

		int ColPerBlock = THREAD_BLOCK_SIZE / temp_nbrow;
		
		nzval = &LnzvalVec[LnzvalPtr[ljb]] + luptrj;

		ddevice_scatter_l_2D(
		    thread_id,
		    nsupc, temp_nbrow,
		    usub, iukp, klst,
		    nzval, ldv,
		    tempv1, nrows, indirect2_thread,
		    nnz_cols, ColPerBlock,
		    IndirectJ3);
	} /* end else ib >= jb */

} /* end Scatter_GPU_kernel */


#define GPU_2D_SCHUDT  /* Not used */

int dSchurCompUpdate_GPU(
    int_t streamId,
    int_t jj_cpu, /* 0 on entry, pointing to the start of Phi part */
    int_t nub,    /* jj_cpu on entry, pointing to the end of the Phi part */
    int_t klst, int_t knsupc,
    int_t Rnbrow, int_t RemainBlk,
    int_t Remain_lbuf_send_size,
    int_t bigu_send_size, int_t ldu,
    int_t mcb,    /* num_u_blks_hi */
    int_t buffer_size, int_t lsub_len, int_t usub_len,
    int_t ldt, int_t k0,
    dsluGPU_t *sluGPU, gridinfo_t *grid
)
{

	dLUstruct_gpu_t * A_gpu = sluGPU->A_gpu;
	dLUstruct_gpu_t * dA_gpu = sluGPU->dA_gpu;
	int_t nprow = grid->nprow;
	int_t npcol = grid->npcol;

	cudaStream_t FunCallStream = sluGPU->funCallStreams[streamId];
	cublasHandle_t cublas_handle0 = sluGPU->cublasHandles[streamId];
	int_t * lsub = A_gpu->scubufs[streamId].lsub_buf;
	int_t * usub = A_gpu->scubufs[streamId].usub_buf;
	Remain_info_t *Remain_info = A_gpu->scubufs[streamId].Remain_info_host;
	double * Remain_L_buff = A_gpu->scubufs[streamId].Remain_L_buff_host;
	Ublock_info_t *Ublock_info = A_gpu->scubufs[streamId].Ublock_info_host;
	double * bigU = A_gpu->scubufs[streamId].bigU_host;
	
	A_gpu->isOffloaded[k0] = 1;
	/* start by sending data to  */
	int_t *xsup = A_gpu->xsup_host;
	int_t col_back = (jj_cpu == 0) ? 0 : Ublock_info[jj_cpu - 1].full_u_cols;
	// if(nub<1) return;
	int_t ncols  = Ublock_info[nub - 1].full_u_cols - col_back;

	/* Sherry: can get max_super_size from sp_ienv(3) */
	int_t indirectJ1[MAX_SUPER_SIZE]; // 0 indicates an empry segment
	int_t indirectJ2[MAX_SUPER_SIZE]; // # of nonzero segments so far
	int_t indirectJ3[MAX_SUPER_SIZE]; /* indirectJ3[j] == k means the
					 j-th nonzero segment points
					 to column k in this supernode */
	/* calculate usub_indirect */
	for (int jj = jj_cpu; jj < nub; ++jj)
	{
	    int_t iukp = Ublock_info[jj].iukp;
	    int_t jb = Ublock_info[jj].jb;
	    int_t nsupc = SuperSize (jb);
	    int_t addr = (jj == 0) ? 0
	             : Ublock_info[jj - 1].full_u_cols - col_back;

	    for (int_t kk = 0; kk < nsupc; ++kk) // old: MAX_SUPER_SIZE
	    {
	    	indirectJ1[kk] = 0;
	    }

	    for (int_t kk = 0; kk < nsupc; ++kk)
	    {
	 	indirectJ1[kk] = ((klst - usub[iukp + kk]) == 0) ? 0 : 1;
	    }

	    /*prefix sum - indicates # of nonzero segments up to column kk */
	    indirectJ2[0] = indirectJ1[0];
	    for (int_t kk = 1; kk < nsupc; ++kk) // old: MAX_SUPER_SIZE
	    {
	 	indirectJ2[kk] = indirectJ2[kk - 1] + indirectJ1[kk];
	    }

	    /* total number of nonzero segments in this supernode */
	    int nnz_col = indirectJ2[nsupc - 1]; // old: MAX_SUPER_SIZE

	    /* compactation */
	    for (int_t kk = 0; kk < nsupc; ++kk) // old: MAX_SUPER_SIZE
	    {
	    	if (indirectJ1[kk]) /* kk is a nonzero segment */
		{
		    /* indirectJ3[j] == kk means the j-th nonzero segment
		       points to column kk in this supernode */
		    indirectJ3[indirectJ2[kk] - 1] = kk;
		}
	    }

    	    for (int i = 0; i < nnz_col; ++i)
	    {
	        /* addr == total # of full columns before current block jj */
		A_gpu->scubufs[streamId].usub_IndirectJ3_host[addr + i] = indirectJ3[i];
	    }
	} /* end for jj ... calculate usub_indirect */

	//printf("dSchurCompUpdate_GPU[3]: jj_cpu %d, nub %d\n", jj_cpu, nub); fflush(stdout);

	/*sizeof RemainLbuf = Rnbuf*knsupc */
	double tTmp = SuperLU_timer_();
	cudaEventRecord(A_gpu->ePCIeH2D[k0], FunCallStream);

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].usub_IndirectJ3,
	                          A_gpu->scubufs[streamId].usub_IndirectJ3_host,
	                          ncols * sizeof(int_t), cudaMemcpyHostToDevice,
	                          FunCallStream)) ;

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].Remain_L_buff, Remain_L_buff,
	                          Remain_lbuf_send_size * sizeof(double),
	                          cudaMemcpyHostToDevice, FunCallStream)) ;

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].bigU, bigU,
	                          bigu_send_size * sizeof(double),
	                          cudaMemcpyHostToDevice, FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].Remain_info, Remain_info,
	                          RemainBlk * sizeof(Remain_info_t),
	                          cudaMemcpyHostToDevice, FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].Ublock_info, Ublock_info,
	                          mcb * sizeof(Ublock_info_t), cudaMemcpyHostToDevice,
	                          FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].lsub, lsub,
	                          lsub_len * sizeof(int_t), cudaMemcpyHostToDevice,
	                          FunCallStream) );

	checkCuda(cudaMemcpyAsync(A_gpu->scubufs[streamId].usub, usub,
	                          usub_len * sizeof(int_t), cudaMemcpyHostToDevice,
	                          FunCallStream) );

	A_gpu->tHost_PCIeH2D += SuperLU_timer_() - tTmp;
	A_gpu->cPCIeH2D += Remain_lbuf_send_size * sizeof(double)
	                   + bigu_send_size * sizeof(double)
	                   + RemainBlk * sizeof(Remain_info_t)
	                   + mcb * sizeof(Ublock_info_t)
	                   + lsub_len * sizeof(int_t)
	                   + usub_len * sizeof(int_t);

	double alpha = 1.0, beta = 0.0;

	int_t ii_st  = 0;
	int_t ii_end = 0;
	int_t maxGemmBlockDim = (int) sqrt(buffer_size);
	// int_t maxGemmBlockDim = 8000;

	/* Organize GEMM by blocks of [ii_st : ii_end, jj_st : jj_end] that
	   fits in the buffer_size  */
	while (ii_end < RemainBlk) {
    	    ii_st = ii_end;
	    ii_end = RemainBlk;
	    int_t nrow_max = maxGemmBlockDim;
// nrow_max = Rnbrow;
	    int_t remaining_rows = (ii_st == 0) ? Rnbrow : Rnbrow - Remain_info[ii_st - 1].FullRow;
	    nrow_max = (remaining_rows / nrow_max) > 0 ? remaining_rows / CEILING(remaining_rows,  nrow_max) : nrow_max;

	    int_t ResRow = (ii_st == 0) ? 0 : Remain_info[ii_st - 1].FullRow;
	    for (int_t i = ii_st; i < RemainBlk - 1; ++i)
    	    {
		if ( Remain_info[i + 1].FullRow > ResRow + nrow_max)
		{
		    ii_end = i;
		    break;  /* row dimension reaches nrow_max */
		}
	    }

	    int_t nrows;   /* actual row dimension for GEMM */
	    int_t st_row;
	    if (ii_st > 0)
	    {
		nrows = Remain_info[ii_end - 1].FullRow - Remain_info[ii_st - 1].FullRow;
		st_row = Remain_info[ii_st - 1].FullRow;
	    }
	    else
	    {
		nrows = Remain_info[ii_end - 1].FullRow;
		st_row = 0;
	    }

	    int jj_st = jj_cpu;
	    int jj_end = jj_cpu;

	    while (jj_end < nub && nrows > 0 )
	    {
		int_t remaining_cols = (jj_st == jj_cpu) ? ncols : ncols - Ublock_info[jj_st - 1].full_u_cols;
		if ( remaining_cols * nrows < buffer_size)
		{
			jj_st = jj_end;
			jj_end = nub;
		}
		else  /* C matrix cannot fit in buffer, need to break into pieces */
		{
		    int_t ncol_max = buffer_size / nrows;
		    /** Must revisit **/
		    ncol_max = SUPERLU_MIN(ncol_max, maxGemmBlockDim);
		    ncol_max = (remaining_cols / ncol_max) > 0 ?
		           remaining_cols / CEILING(remaining_cols,  ncol_max)
		           : ncol_max;

		    jj_st = jj_end;
		    jj_end = nub;

		    int_t ResCol = (jj_st == 0) ? 0 : Ublock_info[jj_st - 1].full_u_cols;
		    for (int_t j = jj_st; j < nub - 1; ++j)
		    {
			if (Ublock_info[j + 1].full_u_cols > ResCol + ncol_max)
			{
				jj_end = j;
				break;
			}
		    }
	    	} /* end-if-else */

		int ncols;
		int st_col;
		if (jj_st > 0)
		{
		    ncols = Ublock_info[jj_end - 1].full_u_cols - Ublock_info[jj_st - 1].full_u_cols;
		    st_col = Ublock_info[jj_st - 1].full_u_cols;
		    if (ncols == 0) exit(0);
		}
		else
		{
		    ncols = Ublock_info[jj_end - 1].full_u_cols;
		    st_col = 0;
		}

		/* none of the matrix dimension is zero. */
		if (nrows > 0 && ldu > 0 && ncols > 0)
		{
		    if (nrows * ncols > buffer_size) {
			printf("!! Matrix size %lld x %lld exceeds buffer_size \n",
			       nrows, ncols, buffer_size);
			fflush(stdout);
		    }
		    assert(nrows * ncols <= buffer_size);
		    cublasSetStream(cublas_handle0, FunCallStream);
		    cudaEventRecord(A_gpu->GemmStart[k0], FunCallStream);

		    cublasDgemm(cublas_handle0, CUBLAS_OP_N, CUBLAS_OP_N,
		            nrows, ncols, ldu, &alpha,
		            &A_gpu->scubufs[streamId].Remain_L_buff[(knsupc - ldu) * Rnbrow + st_row], Rnbrow,
		            &A_gpu->scubufs[streamId].bigU[st_col * ldu], ldu,
		            &beta, A_gpu->scubufs[streamId].bigV, nrows);
			

// #define SCATTER_OPT
#ifdef SCATTER_OPT
		    cudaStreamSynchronize(FunCallStream);
#warning this function is synchronous
#endif
		    cudaEventRecord(A_gpu->GemmEnd[k0], FunCallStream);

		    A_gpu->GemmFLOPCounter += 2.0 * (double) nrows * ncols * ldu;

		    /*
		     * Scattering the output
		     */
  		    dim3 dimBlock(THREAD_BLOCK_SIZE);   // 1d thread

		    dim3 dimGrid(ii_end - ii_st, jj_end - jj_st);
			
		    Scatter_GPU_kernel <<< dimGrid, dimBlock, 0, FunCallStream>>>
			(streamId, ii_st, ii_end,  jj_st, jj_end, klst,
			 0, nrows, ldt, npcol, nprow, dA_gpu);
#ifdef SCATTER_OPT
		    cudaStreamSynchronize(FunCallStream);
#warning this function is synchrnous
#endif

		    cudaEventRecord(A_gpu->ScatterEnd[k0], FunCallStream);

		    A_gpu->ScatterMOPCounter +=  3.0 * (double) nrows * ncols;
		} /* endif ... none of the matrix dimension is zero. */

	    } /* end while jj_end < nub */

	} /* end while (ii_end < RemainBlk) */

	return 0;
} /* end dSchurCompUpdate_GPU */


static void print_occupancy()
{
    int blockSize;   // The launch configurator returned block size
    int minGridSize; /* The minimum grid size needed to achieve the
    		        best potential occupancy  */

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                        Scatter_GPU_kernel, 0, 0);
    printf("Occupancy: MinGridSize %d blocksize %d \n", minGridSize, blockSize);
}

static void printDevProp(cudaDeviceProp devProp)
{
	size_t mfree, mtotal;
	cudaMemGetInfo	(&mfree, &mtotal);
	
	printf("pciBusID:                      %d\n",  devProp.pciBusID);
	printf("pciDeviceID:                   %d\n",  devProp.pciDeviceID);
	printf("GPU Name:                      %s\n",  devProp.name);
	printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
	printf("Total free memory:             %zu\n",  mfree);
	printf("Clock rate:                    %d\n",  devProp.clockRate);

	return;
}


static size_t get_acc_memory ()
{

	size_t mfree, mtotal;
	cudaMemGetInfo	(&mfree, &mtotal);
#if 0
	printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif
	return (size_t) (0.9 * (double) mfree) / get_mpi_process_per_gpu ();


}

int dfree_LUstruct_gpu (dLUstruct_gpu_t * A_gpu)
{
	checkCuda(cudaFree(A_gpu->LrowindVec));
	checkCuda(cudaFree(A_gpu->LrowindPtr));

	checkCuda(cudaFree(A_gpu->LnzvalVec));
	checkCuda(cudaFree(A_gpu->LnzvalPtr));
	free(A_gpu->LnzvalPtr_host);
	/*freeing the pinned memory*/
	int_t streamId = 0;
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].Remain_info_host));
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].Ublock_info_host));
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].Remain_L_buff_host));
	checkCuda (cudaFreeHost (A_gpu->scubufs[streamId].bigU_host));

	checkCuda(cudaFreeHost(A_gpu->acc_L_buff));
	checkCuda(cudaFreeHost(A_gpu->acc_U_buff));
	checkCuda(cudaFreeHost(A_gpu->scubufs[streamId].lsub_buf));
	checkCuda(cudaFreeHost(A_gpu->scubufs[streamId].usub_buf));

	#ifndef SuperLargeScaleGPUBuffer
	free(A_gpu->isOffloaded);
	#endif
	free(A_gpu->GemmStart);
	free(A_gpu->GemmEnd);
	free(A_gpu->ScatterEnd);
	free(A_gpu->ePCIeH2D);

	free(A_gpu->ePCIeD2H_Start);
	free(A_gpu->ePCIeD2H_End);

	checkCuda(cudaFree(A_gpu->UrowindVec));
	checkCuda(cudaFree(A_gpu->UrowindPtr));

	#ifndef SuperLargeScaleGPUBuffer
	free(A_gpu->UrowindPtr_host);
	#endif

	checkCuda(cudaFree(A_gpu->UnzvalVec));
	checkCuda(cudaFree(A_gpu->UnzvalPtr));

	checkCuda(cudaFree(A_gpu->grid));

	checkCuda(cudaFree(A_gpu->scubufs[streamId].bigV));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].bigU));

	checkCuda(cudaFree(A_gpu->scubufs[streamId].Remain_L_buff));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].Ublock_info));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].Remain_info));

	// checkCuda(cudaFree(A_gpu->indirect));
	// checkCuda(cudaFree(A_gpu->indirect2));
	checkCuda(cudaFree(A_gpu->xsup));

	checkCuda(cudaFree(A_gpu->scubufs[streamId].lsub));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].usub));

	checkCuda(cudaFree(A_gpu->local_l_blk_infoVec));
	checkCuda(cudaFree(A_gpu->local_l_blk_infoPtr));
	checkCuda(cudaFree(A_gpu->jib_lookupVec));
	checkCuda(cudaFree(A_gpu->jib_lookupPtr));
	checkCuda(cudaFree(A_gpu->local_u_blk_infoVec));
	checkCuda(cudaFree(A_gpu->local_u_blk_infoPtr));
	checkCuda(cudaFree(A_gpu->ijb_lookupVec));
	checkCuda(cudaFree(A_gpu->ijb_lookupPtr));

	return 0;
}

#ifdef SuperLargeScaleGPUBuffer
int dfree_LUstruct_gpu_Buffer (dLUstruct_gpu_t * A_gpu)
{
	int_t streamId = 0;
	checkCuda(cudaFree(A_gpu->LnzvalVec));
	checkCuda(cudaFree(A_gpu->UnzvalVec));
	checkCuda(cudaFree(A_gpu->scubufs[streamId].bigV));
}
#endif

void dPrint_matrix( char *desc, int_t m, int_t n, double * dA, int_t lda )
{
	double *cPtr = (double *) malloc(sizeof(double) * lda * n);
	checkCuda(cudaMemcpy( cPtr, dA,
	                      lda * n * sizeof(double), cudaMemcpyDeviceToHost)) ;

	int_t i, j;
	printf( "\n %s\n", desc );
	for ( i = 0; i < m; i++ )
	{
		for ( j = 0; j < n; j++ ) printf( " %.3e", cPtr[i + j * lda] );
		printf( "\n" );
	}
	free(cPtr);
}

void dPrint_imatrix( char *desc, int_t m, int_t n, int_t * dA, int_t lda )
{
	int_t *cPtr = (int_t *) malloc(sizeof(int_t) * lda * n);
	checkCuda(cudaMemcpy( cPtr, dA,
	                      lda * n * sizeof(int_t), cudaMemcpyDeviceToHost)) ;

	int_t i, j;
	printf( "\n %s\n", desc );
	for ( i = 0; i < m; i++ )
	{
		for ( j = 0; j < n; j++ ) printf( " %ld", cPtr[i + j * lda] );
		printf( "\n" );
	}
	free(cPtr);
}

void dprintGPUStats(dLUstruct_gpu_t * A_gpu)
{
	double tGemm = 0;
	double tScatter = 0;
	double tPCIeH2D = 0;
	double tPCIeD2H = 0;

	for (int_t i = 0; i < A_gpu->nsupers; ++i)
	{
	    float milliseconds = 0;

	    if (A_gpu->isOffloaded[i])
		{
			cudaEventElapsedTime(&milliseconds, A_gpu->ePCIeH2D[i], A_gpu->GemmStart[i]);
			tPCIeH2D += 1e-3 * (double) milliseconds;
			milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, A_gpu->GemmStart[i], A_gpu->GemmEnd[i]);
			tGemm += 1e-3 * (double) milliseconds;
			milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, A_gpu->GemmEnd[i], A_gpu->ScatterEnd[i]);
			tScatter += 1e-3 * (double) milliseconds;
		}

		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, A_gpu->ePCIeD2H_Start[i], A_gpu->ePCIeD2H_End[i]);
		tPCIeD2H += 1e-3 * (double) milliseconds;
	}

	printf("GPU: Flops offloaded %.3e Time spent %lf Flop rate %lf GF/sec \n",
	       A_gpu->GemmFLOPCounter, tGemm, 1e-9 * A_gpu->GemmFLOPCounter / tGemm  );
	printf("GPU: Mop offloaded %.3e Time spent %lf Bandwidth %lf GByte/sec \n",
	       A_gpu->ScatterMOPCounter, tScatter, 8e-9 * A_gpu->ScatterMOPCounter / tScatter  );
	printf("PCIe Data Transfer H2D:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
	       1e-9 * A_gpu->cPCIeH2D, A_gpu->tHost_PCIeH2D, tPCIeH2D, 1e-9 * A_gpu->cPCIeH2D / tPCIeH2D  );
	printf("PCIe Data Transfer D2H:\n\tData Sent %.3e(GB)\n\tTime observed from CPU %lf\n\tActual time spent %lf\n\tBandwidth %lf GByte/sec \n",
	       1e-9 * A_gpu->cPCIeD2H, A_gpu->tHost_PCIeD2H, tPCIeD2H, 1e-9 * A_gpu->cPCIeD2H / tPCIeD2H  );
	fflush(stdout);

} /* end printGPUStats */


int dinitSluGPU3D_t(
    dsluGPU_t *sluGPU,
    dLUstruct_t *LUstruct,
    gridinfo3d_t * grid3d,
    int_t* perm_c_supno,
    int_t n,
    int_t buffer_size,    /* read from env variable MAX_BUFFER_SIZE */
    int_t bigu_size,
    int_t ldt             /* NSUP read from sp_ienv(3) */
)
{
    checkCudaErrors(cudaDeviceReset ());

	gridinfo_t *grid  = &(grid3d->grid2d);
	
	char processor_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
	MPI_Get_processor_name(processor_name,&namelen);
	int *deviceid = (int*)malloc(grid3d->npcol * grid3d->nprow * grid3d->npdep);

	#ifdef SuperLargeScaleGPUBuffer
	int count;
	cudaGetDeviceCount(&count);
	// cudaSetDevice(grid->iam%4);
	// cudaSetDevice(grid3d->iam%4);
	// cudaSetDevice(((grid3d->iam/grid3d->npcol)%grid3d->npdep)%count);
	cudaSetDevice((grid->iam/grid3d->npcol)%count);
	// cudaSetDevice((grid->iam/grid3d->nprow)%count);
	#ifdef Torch_0419_Case2
	// cudaSetDevice((grid->iam/grid3d->nprow)%count);
	int id = 0;
    deviceid[id++] = 0;	
	deviceid[id++] = 0;
	deviceid[id++] = 0;
	deviceid[id++] = 0;
	deviceid[id++] = 1;
	deviceid[id++] = 1;
	
	cudaSetDevice(deviceid[grid->iam]);
	#endif
	int device;
	cudaGetDevice(&device);
	printf("%d gpu %d\n", grid3d->iam, device);
	#else
	int count;
	cudaGetDeviceCount(&count);
	// cudaSetDevice(grid3d->iam%4);
	// cudaSetDevice(grid->iam%4);
	// cudaSetDevice(grid3d->zscp.Iam%4);
	// cudaSetDevice(grid3d->iam/4 + 1);
	cudaSetDevice((grid3d->iam/grid3d->npcol)%count);
	#endif
	
	

	#if 0
	int id = 0;
    deviceid[id++] = 1;	
	deviceid[id++] = 2;
	deviceid[id++] = 3;
	deviceid[id++] = 0;
	deviceid[id++] = 1;
	deviceid[id++] = 2;
	deviceid[id++] = 3;
	deviceid[id++] = 0;
	deviceid[id++] = 1;
	deviceid[id++] = 2;
	deviceid[id++] = 3;
	deviceid[id++] = 1;
	
	cudaSetDevice(deviceid[grid3d->iam]);
    #else
    // if(!strcmp(processor_name,"node3")){
	// 	switch(grid3d->iam)
	// 	{
	// 		case 0:
	// 			cudaSetDevice(2);
	// 			break;
	// 		case 1:
	// 			cudaSetDevice(3);
	// 			break;
			
	// 		default:
	// 			break;
	// 	}
    // }
	// if(!strcmp(processor_name,"node4")){
	// 	switch(grid3d->iam)
	// 	{
	// 		case 5:
	// 			cudaSetDevice(1);
	// 			break;
	// 		case 6:
	// 			cudaSetDevice(2);
	// 			break;
	// 		case 7:
	// 			cudaSetDevice(3);
	// 			break;
			
	// 		default:
	// 			break;
	// 	}
    // }
	// if(!strcmp(processor_name,"node2")){
	// 	switch(grid3d->iam)
	// 	{
	// 		case 22:
	// 			cudaSetDevice(0);
	// 			break;
	// 		case 23:
	// 			cudaSetDevice(1);
	// 			break;
	// 	}
    // }
	#endif	

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* isNodeInMyGrid = sluGPU->isNodeInMyGrid;

    sluGPU->nCudaStreams = getnCudaStreams();
    if (!grid3d->iam)
    {
	printf("%d:dinitSluGPU3D_t: Using hardware acceleration, with %d cuda streams \n", grid3d->iam,sluGPU->nCudaStreams);
	fflush(stdout);
	if ( MAX_SUPER_SIZE < ldt )
	{
		ABORT("MAX_SUPER_SIZE smaller than requested NSUP");
	}
    }

    cudaStreamCreate(&(sluGPU->CopyStream));

    for (int streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
    {
	cudaStreamCreate(&(sluGPU->funCallStreams[streamId]));
	cublasCreate(&(sluGPU->cublasHandles[streamId]));
	sluGPU->lastOffloadStream[streamId] = -1;
    }

    sluGPU->A_gpu = (dLUstruct_gpu_t *) malloc (sizeof(dLUstruct_gpu_t));
    sluGPU->A_gpu->perm_c_supno = perm_c_supno;
    dCopyLUToGPU3D ( isNodeInMyGrid,
	        Llu,             /* referred to as A_host */
	        sluGPU, Glu_persist, n, grid3d, buffer_size, bigu_size, ldt
	);

    return 0;
} /* end dinitSluGPU3D_t */

int dinitD2Hreduce(
    int next_k,  d2Hreduce_t* d2Hred, int last_flag, HyP_t* HyP,
    dsluGPU_t *sluGPU, gridinfo_t *grid, dLUstruct_t *LUstruct, SCT_t* SCT
)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;

    // int_t next_col = SUPERLU_MIN (k0 + num_look_aheads + 1, nsupers - 1);
    // int_t next_k = perm_c_supno[next_col];  /* global block number for next colum*/
    int_t mkcol, mkrow;
    
    int_t kljb = LBj( next_k, grid );   /*local block number for next block*/
    int_t kijb = LBi( next_k, grid );   /*local block number for next block*/
    
    int_t *kindexL ;                     /*for storing index vectors*/
    int_t *kindexU ;
    mkrow = PROW (next_k, grid);
    mkcol = PCOL (next_k, grid);
    int_t ksup_size = SuperSize(next_k);
    
    int_t copyL_kljb = 0;
    int_t copyU_kljb = 0;
    int_t l_copy_len = 0;
    int_t u_copy_len = 0;
    
    if (mkcol == mycol &&  Lrowind_bc_ptr[kljb] != NULL  && last_flag)
    {
	if (HyP->Lblock_dirty_bit[kljb] > -1)
	    {
		copyL_kljb = 1;
		int_t lastk0 = HyP->Lblock_dirty_bit[kljb];
		int_t streamIdk0Offload =  lastk0 % sluGPU->nCudaStreams;
		if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
		    {
			// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
			double ttx = SuperLU_timer_();
			cudaStreamSynchronize(sluGPU->funCallStreams[streamIdk0Offload]);
			SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
			sluGPU->lastOffloadStream[streamIdk0Offload] = -1;
		    }
	    }

	kindexL = Lrowind_bc_ptr[kljb];
	l_copy_len = kindexL[1] * ksup_size;
    }

    if ( mkrow == myrow && Ufstnz_br_ptr[kijb] != NULL    && last_flag )
    {
	if (HyP->Ublock_dirty_bit[kijb] > -1)
	    {
		copyU_kljb = 1;
		int_t lastk0 = HyP->Ublock_dirty_bit[kijb];
		int_t streamIdk0Offload =  lastk0 % sluGPU->nCudaStreams;
		if (sluGPU->lastOffloadStream[streamIdk0Offload] == lastk0 && lastk0 != -1)
		    {
			// printf("Waiting for Offload =%d to finish StreamId=%d\n", lastk0, streamIdk0Offload);
			double ttx = SuperLU_timer_();
			cudaStreamSynchronize(sluGPU->funCallStreams[streamIdk0Offload]);
			SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
			sluGPU->lastOffloadStream[streamIdk0Offload] = -1;
		    }
	    }
	// copyU_kljb = HyP->Ublock_dirty_bit[kijb]>-1? 1: 0;
	kindexU = Ufstnz_br_ptr[kijb];
	u_copy_len = kindexU[1];
    }

    // wait for streams if they have not been finished
    
    // d2Hred->next_col = next_col;
    d2Hred->next_k = next_k;
    d2Hred->kljb = kljb;
    d2Hred->kijb = kijb;
    d2Hred->copyL_kljb = copyL_kljb;
    d2Hred->copyU_kljb = copyU_kljb;
    d2Hred->l_copy_len = l_copy_len;
    d2Hred->u_copy_len = u_copy_len;
    d2Hred->kindexU = kindexU;
    d2Hred->kindexL = kindexL;
    d2Hred->mkrow = mkrow;
    d2Hred->mkcol = mkcol;
    d2Hred->ksup_size = ksup_size;
    return 0;
} /* dinitD2Hreduce */

int dreduceGPUlu(
    int last_flag,
    d2Hreduce_t* d2Hred,
    dsluGPU_t *sluGPU,
    SCT_t *SCT,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct
)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    
    cudaStream_t CopyStream;
    dLUstruct_gpu_t *A_gpu;
    A_gpu = sluGPU->A_gpu;
    CopyStream = sluGPU->CopyStream;

    int_t kljb = d2Hred->kljb;
    int_t kijb = d2Hred->kijb;
    int_t copyL_kljb = d2Hred->copyL_kljb;
    int_t copyU_kljb = d2Hred->copyU_kljb;
    int_t mkrow = d2Hred->mkrow;
    int_t mkcol = d2Hred->mkcol;
    int_t ksup_size = d2Hred->ksup_size;
    int_t *kindex;
    if ((copyL_kljb || copyU_kljb) && last_flag )
	{
	    double ttx = SuperLU_timer_();
	    cudaStreamSynchronize(CopyStream);
	    SCT->PhiWaitTimer_2 += SuperLU_timer_() - ttx;
	}

    double tt_start = SuperLU_timer_();

    if (last_flag) {
	if (mkcol == mycol && Lrowind_bc_ptr[kljb] != NULL )
	    {
		kindex = Lrowind_bc_ptr[kljb];
		int_t len = kindex[1];

		if (copyL_kljb)
		    {
			double *nzval_host;			

			#ifdef Torch
			int_t iPart=0;
			#ifdef SuperLargeScale			

			if(Llu->isSave){
				if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[kljb][iPart]) == Unused)){
					Lnzval_bc_ptr[kljb] = load_Lnzval_bc_ptr(kljb,Llu);
				}
			}
			if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[kljb][iPart]) == Unused){
				SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[kljb][iPart], Used);
			}
			#endif
						
			#endif
			nzval_host = Lnzval_bc_ptr[kljb];

			int_t llen = ksup_size * len;
			double alpha = 1;
			superlu_daxpy (llen, alpha, A_gpu->acc_L_buff, 1, nzval_host, 1);

			#ifdef Torch
			#ifdef SuperLargeScale
			if(Llu->isSave){
				if(Llu->core_status == OutOfCore && Llu->Lnzval_bc_ptr_ilen[kljb]){
					if(set_iLnzval_bc_ptr_txt(nzval_host,kljb,0,llen,Llu)){
						if(SUPERLU_FREE(nzval_host)){
							ABORT("failed in set_iLnzval_bc_ptr_txt() of dreduceGPUlu(). \n");
						}
						SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[kljb][iPart], Unused);
					}
					else{
						ABORT("failed in set_iLnzval_bc_ptr_txt() of dreduceGPUlu(). \n");
					}
				}
				else{
					if(Llu->Lnzval_bc_ptr_ilen[kljb]){
						SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[kljb][iPart], Changed);
					}
					else{
						SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[kljb][iPart], Unused);
					}
				}
			}
			#endif
			#endif
		    }

	    }
    }
    if (last_flag) {
	if (mkrow == myrow && Ufstnz_br_ptr[kijb] != NULL )
	    {
		kindex = Ufstnz_br_ptr[kijb];
		int_t len = kindex[1];

		if (copyU_kljb)
		    {
			double *nzval_host;

			#ifdef Torch
			int_t iPart=0;
			#ifdef SuperLargeScale

			if(Llu->isSave){
				if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Unzval_br_ptr[kijb][iPart]) == Unused)){
					Unzval_br_ptr[kijb] = load_Unzval_br_ptr(kijb,Llu);
				}
			}
			if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[kijb][iPart]) == Unused){
				SetVectorStatus(Llu->isUsed_Unzval_br_ptr[kijb][iPart], Used);
			}
			#endif
			
			
			#endif
			nzval_host = Unzval_br_ptr[kijb];

			double alpha = 1;
			superlu_daxpy (len, alpha, A_gpu->acc_U_buff, 1, nzval_host, 1);

			#ifdef Torch
			#ifdef SuperLargeScale
			if(Llu->isSave){
				if(Llu->core_status == OutOfCore && Llu->Unzval_br_ptr_ilen[kijb]){
					if(set_iUnzval_br_ptr_txt(nzval_host,kijb,0,len,Llu)){
						SUPERLU_FREE(nzval_host);
						SetVectorStatus(Llu->isUsed_Unzval_br_ptr[kijb][iPart], Unused);
					}
					else{
						ABORT("failed in set_iUnzval_br_ptr_txt() of dreduceGPUlu(). \n");
					}
				}
				else{
					if(Llu->Unzval_br_ptr_ilen[kijb]){
						SetVectorStatus(Llu->isUsed_Unzval_br_ptr[kijb][iPart], Changed);
					}
					else{
						SetVectorStatus(Llu->isUsed_Unzval_br_ptr[kijb][iPart], Unused);
					}
				}
			}
			#endif
			#endif
		    }
	    }
    }

    double tt_end = SuperLU_timer_();
    SCT->AssemblyTimer += tt_end - tt_start;
    return 0;
} /* dreduceGPUlu */


int dwaitGPUscu(int streamId, dsluGPU_t *sluGPU, SCT_t *SCT)
{
    double ttx = SuperLU_timer_();
    cudaStreamSynchronize(sluGPU->funCallStreams[streamId]);
    SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
    return 0;
}

int dsendLUpanelGPU2HOST(
    int_t k0,
    d2Hreduce_t* d2Hred,
    dsluGPU_t *sluGPU
)
{
    int_t kljb = d2Hred->kljb;
    int_t kijb = d2Hred->kijb;
    int_t copyL_kljb = d2Hred->copyL_kljb;
    int_t copyU_kljb = d2Hred->copyU_kljb;
    int_t l_copy_len = d2Hred->l_copy_len;
    int_t u_copy_len = d2Hred->u_copy_len;
    cudaStream_t CopyStream = sluGPU->CopyStream;
    dLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
    double tty = SuperLU_timer_();
    cudaEventRecord(A_gpu->ePCIeD2H_Start[k0], CopyStream);
    if (copyL_kljb)
	{
		#ifdef SuperLargeScaleGPUBuffer
		if(A_gpu->isEmpty){
			if(A_gpu->isGPUUsed_Lnzval_bc_ptr_host[kljb] == GPUUnused){
				if(A_gpu->isCPUUsed_Lnzval_bc_ptr_host[kljb] == CPUUnused){
					memset(A_gpu->acc_L_buff, 0, l_copy_len * sizeof(double));
				}
				else{
					memcpy(A_gpu->acc_L_buff, A_gpu->LnzvalVec_host[kljb], l_copy_len * sizeof(double));
				}				
			}
			else{
				checkCuda(cudaMemcpyAsync(A_gpu->acc_L_buff, &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[kljb]],
					l_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
			}
		}
		else{
			checkCuda(cudaMemcpyAsync(A_gpu->acc_L_buff, &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[kljb]],
				l_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
		}
		#else
		checkCuda(cudaMemcpyAsync(A_gpu->acc_L_buff, &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[kljb]],
				  l_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
		#endif
	}

	#ifdef SuperLargeScaleGPUBuffer
	if (copyU_kljb)
	{		
		if(A_gpu->isEmpty){
			if(A_gpu->isGPUUsed_Unzval_br_ptr_host[kijb] == GPUUnused){
				if(A_gpu->isCPUUsed_Unzval_br_ptr_host[kijb] == CPUUnused){
					memset(A_gpu->acc_U_buff, 0, u_copy_len * sizeof(double));
				}
				else{
					memcpy(A_gpu->acc_U_buff, A_gpu->UnzvalVec_host[kijb], u_copy_len * sizeof(double));
				}				
			}
			else{
				checkCuda(cudaMemcpyAsync(A_gpu->acc_U_buff, &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[kijb]],
					u_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
			}
		}
		else{
			checkCuda(cudaMemcpyAsync(A_gpu->acc_U_buff, &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[kijb]],
				u_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
		}		
	}
	#else
    if (copyU_kljb)
	checkCuda(cudaMemcpyAsync(A_gpu->acc_U_buff, &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[kijb]],
				  u_copy_len * sizeof(double), cudaMemcpyDeviceToHost, CopyStream ) );
	#endif

    cudaEventRecord(A_gpu->ePCIeD2H_End[k0], CopyStream);
    A_gpu->tHost_PCIeD2H += SuperLU_timer_() - tty;
    A_gpu->cPCIeD2H += u_copy_len * sizeof(double) + l_copy_len * sizeof(double);

    return 0;
}

/* Copy L and U panel data structures from host to the host part of the
   data structures in A_gpu.
   GPU is not involved in this routine. */
int dsendSCUdataHost2GPU(
    int_t streamId,
    int_t* lsub,
    int_t* usub,
    double* bigU,
    int_t bigu_send_size,
    int_t Remain_lbuf_send_size,
    dsluGPU_t *sluGPU,
    HyP_t* HyP
)
{
    //{printf("....[enter] dsendSCUdataHost2GPU, bigu_send_size %d\n", bigu_send_size); fflush(stdout);}

    int_t usub_len = usub[2];
    int_t lsub_len = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
    //{printf("....[2] in dsendSCUdataHost2GPU, lsub_len %d\n", lsub_len); fflush(stdout);}
    dLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
    memcpy(A_gpu->scubufs[streamId].lsub_buf, lsub, sizeof(int_t)*lsub_len);
    memcpy(A_gpu->scubufs[streamId].usub_buf, usub, sizeof(int_t)*usub_len);
    memcpy(A_gpu->scubufs[streamId].Remain_info_host, HyP->Remain_info,
	   sizeof(Remain_info_t)*HyP->RemainBlk);
    memcpy(A_gpu->scubufs[streamId].Ublock_info_host, HyP->Ublock_info_Phi,
	   sizeof(Ublock_info_t)*HyP->num_u_blks_Phi);
    memcpy(A_gpu->scubufs[streamId].Remain_L_buff_host, HyP->Remain_L_buff,
	   sizeof(double)*Remain_lbuf_send_size);
    memcpy(A_gpu->scubufs[streamId].bigU_host, bigU,
	   sizeof(double)*bigu_send_size);

    return 0;
}

/* Sherry: not used ?*/
#if 0
int freeSluGPU(dsluGPU_t *sluGPU)
{
    return 0;
}
#endif

void dCopyLUToGPU3D (
    int_t* isNodeInMyGrid,
    dLocalLU_t *A_host, /* distributed LU structure on host */
    dsluGPU_t *sluGPU,
    Glu_persist_t *Glu_persist, int_t n,
    gridinfo3d_t *grid3d,
    int_t buffer_size, /* bigV size on GPU for Schur complement update */
    int_t bigu_size,
    int_t ldt
)
{
	#if ( DEBUGlevel>=1 )
		CHECK_MALLOC (grid3d->iam, "Enter dCopyLUToGPU3D()");
	#endif

    gridinfo_t* grid = &(grid3d->grid2d);
    dLUstruct_gpu_t * A_gpu =  sluGPU->A_gpu;
    dLUstruct_gpu_t **dA_gpu =  &(sluGPU->dA_gpu);

	#ifdef SuperLargeScaleGPU
	if(A_host->isRecordingForGPU == LoadedRecordForGPU){
		A_gpu->isEmpty = 1;		
	}
	else{
		A_gpu->isEmpty = 0;
	}
	#endif

#if ( PRNTlevel>=1 )
    if ( grid3d->iam == 0 ) print_occupancy();
#endif

#ifdef GPU_DEBUG
    // if ( grid3d->iam == 0 )
    {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printDevProp(devProp);
    }
#endif
    int_t *xsup ;
    xsup = Glu_persist->xsup;
    int iam = grid->iam;
    int nsupers = Glu_persist->supno[n - 1] + 1;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t mrb =    (nsupers + Pr - 1) / Pr;
    int_t mcb =    (nsupers + Pc - 1) / Pc;
    int_t remain_l_max = A_host->bufmax[1];

    /*copies of scalars for easy access*/
    A_gpu->nsupers = nsupers;
    A_gpu->ScatterMOPCounter = 0;
    A_gpu->GemmFLOPCounter = 0;
    A_gpu->cPCIeH2D = 0;
    A_gpu->cPCIeD2H = 0;
    A_gpu->tHost_PCIeH2D = 0;
    A_gpu->tHost_PCIeD2H = 0;

    /*initializing memory*/
    size_t max_gpu_memory = get_acc_memory ();
    size_t gpu_mem_used = 0;

    void *tmp_ptr;

    A_gpu->xsup_host = xsup;

    int_t nCudaStreams = sluGPU->nCudaStreams;
    /*pinned memory allocations.
      Paged-locked memory by cudaMallocHost is accessible to the device.*/
    for (int streamId = 0; streamId < nCudaStreams; streamId++ ) {
	void *tmp_ptr;
	checkCudaErrors(cudaMallocHost(  &tmp_ptr, (n) * sizeof(int_t) )) ;
	A_gpu->scubufs[streamId].usub_IndirectJ3_host = (int_t*) tmp_ptr;

	checkCudaErrors(cudaMalloc( &tmp_ptr,  ( n) * sizeof(int_t) ));
	A_gpu->scubufs[streamId].usub_IndirectJ3 =  (int_t*) tmp_ptr;
	gpu_mem_used += ( n) * sizeof(int_t);
	checkCudaErrors(cudaMallocHost(  &tmp_ptr, mrb * sizeof(Remain_info_t) )) ;
	A_gpu->scubufs[streamId].Remain_info_host = (Remain_info_t*)tmp_ptr;
	checkCudaErrors(cudaMallocHost(  &tmp_ptr, mcb * sizeof(Ublock_info_t) )) ;
	A_gpu->scubufs[streamId].Ublock_info_host = (Ublock_info_t*)tmp_ptr;
	checkCudaErrors(cudaMallocHost(  &tmp_ptr,  remain_l_max * sizeof(double) )) ;
	A_gpu->scubufs[streamId].Remain_L_buff_host = (double *) tmp_ptr;
	checkCudaErrors(cudaMallocHost(  &tmp_ptr,  bigu_size * sizeof(double) )) ;
	A_gpu->scubufs[streamId].bigU_host = (double *) tmp_ptr;

	checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(double) * (A_host->bufmax[1])));
	A_gpu->acc_L_buff = (double *) tmp_ptr;
	checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(double) * (A_host->bufmax[3])));
	A_gpu->acc_U_buff = (double *) tmp_ptr;
	checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(int_t) * (A_host->bufmax[0])));
	A_gpu->scubufs[streamId].lsub_buf =  (int_t *) tmp_ptr;
	checkCudaErrors(cudaMallocHost ( &tmp_ptr, sizeof(int_t) * (A_host->bufmax[2])));
	A_gpu->scubufs[streamId].usub_buf = (int_t *) tmp_ptr;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  remain_l_max * sizeof(double) )) ;
	A_gpu->scubufs[streamId].Remain_L_buff = (double *) tmp_ptr;
	gpu_mem_used += remain_l_max * sizeof(double);
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  bigu_size * sizeof(double) )) ;
	A_gpu->scubufs[streamId].bigU = (double *) tmp_ptr;
	gpu_mem_used += bigu_size * sizeof(double);
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  mcb * sizeof(Ublock_info_t) )) ;
	A_gpu->scubufs[streamId].Ublock_info = (Ublock_info_t *) tmp_ptr;
	gpu_mem_used += mcb * sizeof(Ublock_info_t);
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  mrb * sizeof(Remain_info_t) )) ;
	A_gpu->scubufs[streamId].Remain_info = (Remain_info_t *) tmp_ptr;
	gpu_mem_used += mrb * sizeof(Remain_info_t);

	printf("%d: buffer_size %ld, gpu_mem_used %ld\n", grid3d->iam, buffer_size, gpu_mem_used);
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  buffer_size * sizeof(double))) ;
	A_gpu->scubufs[streamId].bigV = (double *) tmp_ptr;
	gpu_mem_used += buffer_size * sizeof(double);
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  A_host->bufmax[0]*sizeof(int_t))) ;
	A_gpu->scubufs[streamId].lsub = (int_t *) tmp_ptr;
	gpu_mem_used += A_host->bufmax[0] * sizeof(int_t);
	checkCudaErrors(cudaMalloc(  &tmp_ptr,  A_host->bufmax[2]*sizeof(int_t))) ;
	A_gpu->scubufs[streamId].usub = (int_t *) tmp_ptr;
	gpu_mem_used += A_host->bufmax[2] * sizeof(int_t);
	
    } /* endfor streamID ... allocate paged-locked memory */

	#ifdef SuperLargeScaleGPUBuffer
	// printf("scubufs[streamId].usub_IndirectJ3 %ld, scubufs[streamId].Remain_L_buff %ld, scubufs[streamId].bigU %ld, scubufs[streamId].Ublock_info %ld, scubufs[streamId].Remain_info %ld, scubufs[streamId].bigV %ld, scubufs[streamId].lsub %ld, scubufs[streamId].usub %ld\n", ( n) * sizeof(int_t), remain_l_max * sizeof(double), bigu_size * sizeof(double), mcb * sizeof(Ublock_info_t), mrb * sizeof(Remain_info_t), buffer_size * sizeof(double), A_host->bufmax[0] * sizeof(int_t), A_host->bufmax[2] * sizeof(int_t));
	#endif

    A_gpu->isOffloaded = (int *) SUPERLU_MALLOC (sizeof(int) * nsupers);
    A_gpu->GemmStart  = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
    A_gpu->GemmEnd  = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
    A_gpu->ScatterEnd  = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
    A_gpu->ePCIeH2D = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
    A_gpu->ePCIeD2H_Start = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
    A_gpu->ePCIeD2H_End = (cudaEvent_t *) malloc(sizeof(cudaEvent_t) * nsupers);
    
    for (int i = 0; i < nsupers; ++i)
	{
	    A_gpu->isOffloaded[i] = 0;
	    checkCudaErrors(cudaEventCreate(&(A_gpu->GemmStart[i])));
	    checkCudaErrors(cudaEventCreate(&(A_gpu->GemmEnd[i])));
	    checkCudaErrors(cudaEventCreate(&(A_gpu->ScatterEnd[i])));
	    checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeH2D[i])));
		#ifdef Torch
	    // checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeH2D[i])));
		#endif
	    checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeD2H_Start[i])));
	    checkCudaErrors(cudaEventCreate(&(A_gpu->ePCIeD2H_End[i])));
	}

    /*---- Copy L data structure to GPU ----*/

    /*pointers and address of local blocks for easy accessibility */
    local_l_blk_info_t  *local_l_blk_infoVec;
    int_t  * local_l_blk_infoPtr;
    local_l_blk_infoPtr =  (int_t *) malloc( CEILING(nsupers, Pc) * sizeof(int_t ) );

    /* First pass: count total L blocks */
    int_t cum_num_l_blocks = 0;  /* total number of L blocks I own */
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
	{
	    /* going through each block column I own */

	    if (A_host->Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[i * Pc + mycol] == 1)
		{
		    int_t *index = A_host->Lrowind_bc_ptr[i];
		    int_t num_l_blocks = index[0];
		    cum_num_l_blocks += num_l_blocks;
		}
	}

    /*allocating memory*/
    local_l_blk_infoVec =  (local_l_blk_info_t *) malloc(cum_num_l_blocks * sizeof(local_l_blk_info_t));

    /* Second pass: set up the meta-data for the L structure */
    cum_num_l_blocks = 0;

    /*initialzing vectors */
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
	{
	    if (A_host->Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[i * Pc + mycol] == 1)
		{
		    int_t *index = A_host->Lrowind_bc_ptr[i];
		    int_t num_l_blocks = index[0]; /* # L blocks in this column */

		    if (num_l_blocks > 0)
			{

			    local_l_blk_info_t *local_l_blk_info_i = local_l_blk_infoVec + cum_num_l_blocks;
			    local_l_blk_infoPtr[i] = cum_num_l_blocks;

			    int_t lptrj = BC_HEADER;
			    int_t luptrj = 0;

			    for (int_t j = 0; j < num_l_blocks ; ++j)
				{

				    int_t ijb = index[lptrj];

				    local_l_blk_info_i[j].lib = ijb / Pr;
				    local_l_blk_info_i[j].lptrj = lptrj;
				    local_l_blk_info_i[j].luptrj = luptrj;
				    luptrj += index[lptrj + 1];
				    lptrj += LB_DESCRIPTOR + index[lptrj + 1];
					
				}
			}
		    cum_num_l_blocks += num_l_blocks;
		}

	} /* endfor all block columns */

    /* Allocate L memory on GPU, and copy the values from CPU to GPU */
    checkCudaErrors(cudaMalloc(  &tmp_ptr,  cum_num_l_blocks * sizeof(local_l_blk_info_t))) ;
    A_gpu->local_l_blk_infoVec = (local_l_blk_info_t *) tmp_ptr;
    gpu_mem_used += cum_num_l_blocks * sizeof(local_l_blk_info_t);
    checkCudaErrors(cudaMemcpy( (A_gpu->local_l_blk_infoVec), local_l_blk_infoVec, cum_num_l_blocks * sizeof(local_l_blk_info_t), cudaMemcpyHostToDevice)) ;

    checkCudaErrors(cudaMalloc(  &tmp_ptr,  CEILING(nsupers, Pc)*sizeof(int_t))) ;
    A_gpu->local_l_blk_infoPtr = (int_t *) tmp_ptr;
    gpu_mem_used += CEILING(nsupers, Pc) * sizeof(int_t);
    checkCudaErrors(cudaMemcpy( (A_gpu->local_l_blk_infoPtr), local_l_blk_infoPtr, CEILING(nsupers, Pc)*sizeof(int_t), cudaMemcpyHostToDevice)) ;

	#ifdef SuperLargeScaleGPUBuffer
	// printf("local_l_blk_infoVec %ld, local_l_blk_infoPtr %ld\n", cum_num_l_blocks * sizeof(local_l_blk_info_t), CEILING(nsupers, Pc) * sizeof(int_t));
	#endif

    /*---- Copy U data structure to GPU ----*/

    local_u_blk_info_t  *local_u_blk_infoVec;
    int_t  * local_u_blk_infoPtr;
    local_u_blk_infoPtr =  (int_t *) malloc( CEILING(nsupers, Pr) * sizeof(int_t ) );

    /* First pass: count total U blocks */
    int_t cum_num_u_blocks = 0;

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
	{

	    if (A_host->Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
		{
		    int_t *index = A_host->Ufstnz_br_ptr[i];
		    int_t num_u_blocks = index[0];
		    cum_num_u_blocks += num_u_blocks;

		}
	}

	local_u_blk_infoVec =  (local_u_blk_info_t *) malloc(cum_num_u_blocks * sizeof(local_u_blk_info_t));

	/* Second pass: set up the meta-data for the U structure */
	cum_num_u_blocks = 0;

	for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
	{
	    if (A_host->Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
		{
		    int_t *index = A_host->Ufstnz_br_ptr[i];
		    int_t num_u_blocks = index[0];

		    if (num_u_blocks > 0)
			{
			    local_u_blk_info_t  *local_u_blk_info_i = local_u_blk_infoVec + cum_num_u_blocks;
			    local_u_blk_infoPtr[i] = cum_num_u_blocks;

			    int_t iuip_lib, ruip_lib;
			    iuip_lib = BR_HEADER;
			    ruip_lib = 0;

			    for (int_t j = 0; j < num_u_blocks ; ++j)
				{

				    int_t ijb = index[iuip_lib];
				    local_u_blk_info_i[j].ljb = ijb / Pc;
				    local_u_blk_info_i[j].iuip = iuip_lib;
				    local_u_blk_info_i[j].ruip = ruip_lib;

				    ruip_lib += index[iuip_lib + 1];
				    iuip_lib += UB_DESCRIPTOR + SuperSize (ijb);

				}
			}
		    cum_num_u_blocks +=  num_u_blocks;
		}
	}

	checkCudaErrors(cudaMalloc( &tmp_ptr,  cum_num_u_blocks * sizeof(local_u_blk_info_t))) ;
	A_gpu->local_u_blk_infoVec = (local_u_blk_info_t *) tmp_ptr;
	gpu_mem_used += cum_num_u_blocks * sizeof(local_u_blk_info_t);
	checkCudaErrors(cudaMemcpy( (A_gpu->local_u_blk_infoVec), local_u_blk_infoVec, cum_num_u_blocks * sizeof(local_u_blk_info_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc( &tmp_ptr,  CEILING(nsupers, Pr)*sizeof(int_t))) ;
	A_gpu->local_u_blk_infoPtr = (int_t *) tmp_ptr;
	gpu_mem_used += CEILING(nsupers, Pr) * sizeof(int_t);
	checkCudaErrors(cudaMemcpy( (A_gpu->local_u_blk_infoPtr), local_u_blk_infoPtr, CEILING(nsupers, Pr)*sizeof(int_t), cudaMemcpyHostToDevice)) ;	

	#ifdef SuperLargeScaleGPUBuffer
	// printf("local_u_blk_infoVec %ld, local_u_blk_infoPtr %ld\n", cum_num_u_blocks * sizeof(local_u_blk_info_t), CEILING(nsupers, Pr) * sizeof(int_t));
	#endif

	/* Copy the actual L indices and values */
	int_t l_k = CEILING( nsupers, grid->npcol ); /* # of local block columns */
	int_t *temp_LrowindPtr    = (int_t *) malloc(sizeof(int_t) * l_k);
	int_t *temp_LnzvalPtr     = (int_t *) malloc(sizeof(int_t) * l_k);
	int_t *Lnzval_size = (int_t *) malloc(sizeof(int_t) * l_k);
	int_t l_ind_len = 0;
	int_t l_val_len = 0;	

	for (int_t jb = 0; jb < nsupers; ++jb) /* for each block column ... */
	{
	    int_t pc = PCOL( jb, grid );
	    if (mycol == pc && isNodeInMyGrid[jb] == 1)
		{
		    int_t ljb = LBj( jb, grid ); /* Local block number */
		    int_t  *index_host;
		    index_host = A_host->Lrowind_bc_ptr[ljb];

		    temp_LrowindPtr[ljb] = l_ind_len;
		    temp_LnzvalPtr[ljb] = l_val_len;        // ###
		    Lnzval_size[ljb] = 0;       //###
		    if (index_host != NULL)
			{
			    int_t nrbl  = index_host[0];   /* number of L blocks */
			    int_t len   = index_host[1];   /* LDA of the nzval[] */
			    int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;

			    /* Global block number is mycol +  ljb*Pc */
			    int_t nsupc = SuperSize(jb);

			    l_ind_len += len1;
			    l_val_len += len * nsupc;
			    Lnzval_size[ljb] = len * nsupc ; // ###				
			}
		    else
			{
			    Lnzval_size[ljb] = 0 ; // ###
			}
		}
	} /* endfor jb = 0 ... */

	/* Copy the actual U indices and values */
	int_t u_k = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
	int_t *temp_UrowindPtr    = (int_t *) malloc(sizeof(int_t) * u_k);
	int_t *temp_UnzvalPtr     = (int_t *) malloc(sizeof(int_t) * u_k);
	int_t *Unzval_size = (int_t *) malloc(sizeof(int_t) * u_k);
	int_t u_ind_len = 0;
	int_t u_val_len = 0;
	for ( int_t lb = 0; lb < u_k; ++lb)
	{
	    int_t *index_host;
	    index_host =  A_host->Ufstnz_br_ptr[lb];
	    temp_UrowindPtr[lb] = u_ind_len;
	    temp_UnzvalPtr[lb] = u_val_len;
	    Unzval_size[lb] = 0;
	    if (index_host != NULL && isNodeInMyGrid[lb * Pr + myrow] == 1)
		{
		    int_t len = index_host[1];
		    int_t len1 = index_host[2];
		    
		    u_ind_len += len1;
		    u_val_len += len;
		    Unzval_size[lb] = len;
		}
	    else
		{
		    Unzval_size[lb] = 0;
		}
	}

	gpu_mem_used += l_ind_len * sizeof(int_t);
	gpu_mem_used += 2 * l_k * sizeof(int_t);
	gpu_mem_used += u_ind_len * sizeof(int_t);
	gpu_mem_used += 2 * u_k * sizeof(int_t);

	#ifdef SuperLargeScaleGPUBuffer
	// printf("l_ind_len %ld, 2 * l_k %ld\n, u_ind_len %ld, 2 * u_k %ld\n", l_ind_len * sizeof(int_t), 2 * l_k * sizeof(int_t), u_ind_len * sizeof(int_t), 2 * u_k * sizeof(int_t));
	#endif

	/*left memory shall be divided among the two */

	for (int_t i = 0;  i < l_k; ++i)
	{
	    temp_LnzvalPtr[i] = -1;
	}

	for (int_t i = 0; i < u_k; ++i)
	{
	    temp_UnzvalPtr[i] = -1;
	}

	/*setting these pointers back */
	l_val_len = 0;
	u_val_len = 0;

	int_t num_gpu_l_blocks = 0;
	int_t num_gpu_u_blocks = 0;
	size_t mem_l_block, mem_u_block;

	/* Find the trailing matrix size that can fit into GPU memory */
	for (int_t i = nsupers - 1; i > -1; --i)
	{
	    /* ulte se chalte hai eleimination tree  */
	    /* bottom up ordering  */
	    int_t i_sup = A_gpu->perm_c_supno[i];

	    int_t pc = PCOL( i_sup, grid );
	    if (isNodeInMyGrid[i_sup] == 1)
		{
		    if (mycol == pc )
			{
			    int_t ljb  = LBj(i_sup, grid);
			    mem_l_block = sizeof(double) * Lnzval_size[ljb];
			    if (gpu_mem_used + mem_l_block > max_gpu_memory)
				{
				    break;
				}
				else
				{
				    gpu_mem_used += mem_l_block;
				    temp_LnzvalPtr[ljb] = l_val_len;
				    l_val_len += Lnzval_size[ljb];
				    num_gpu_l_blocks++;
				    A_gpu->first_l_block_gpu = i;
				}
			}

			int_t pr = PROW( i_sup, grid );
			if (myrow == pr)
			{
			    int_t lib  = LBi(i_sup, grid);
			    mem_u_block = sizeof(double) * Unzval_size[lib];
			    if (gpu_mem_used + mem_u_block > max_gpu_memory)
				{
				    break;
				}
			    else
				{
				    gpu_mem_used += mem_u_block;
				    temp_UnzvalPtr[lib] = u_val_len;
				    u_val_len += Unzval_size[lib];
				    num_gpu_u_blocks++;
				    A_gpu->first_u_block_gpu = i;
				}
			}
		} /* endif */

	} /* endfor i .... nsupers */

	#ifdef SuperLargeScaleGPUBuffer	
	free(Lnzval_size);	
	free(Unzval_size);
	#endif	

	#ifdef Torch
	// while(gpu_mem_used > max_gpu_memory)
	// {
	// 	sleep(1);
	// 	max_gpu_memory = get_acc_memory ();
	// }
	max_gpu_memory = get_acc_memory ();
	#ifdef SOLVE_1
	if(gpu_mem_used * 1e-9 > max_gpu_memory * 1e-9 + 1.0){
		double error = 3;
		int_t ierror = 3;
		delete_idata_vector_txt(0,1007);
        save_idata_vector_binary(&ierror,1,0,1009);
		save_ddata_vector_txt(&error,1,0,1200);
	}
	#endif
	#endif

#if (PRNTlevel>=2)
	printf("(%d) Number of L blocks in GPU %d, U blocks %d\n",
	       grid3d->iam, num_gpu_l_blocks, num_gpu_u_blocks );
	printf("(%d) elimination order of first block in GPU: L block %d, U block %d\n",
	       grid3d->iam, A_gpu->first_l_block_gpu, A_gpu->first_u_block_gpu);
	printf("(%d) Memory of L %.1f GB, memory for U %.1f GB, Total device memory used %.1f GB, Memory allowed %.1f GB \n", grid3d->iam,
	       l_val_len * sizeof(double) * 1e-9,
	       u_val_len * sizeof(double) * 1e-9,
	       gpu_mem_used * 1e-9, max_gpu_memory * 1e-9);
	fflush(stdout);
#endif

	/* Assemble index vector on temp */
	int_t *indtemp = (int_t *) malloc(sizeof(int_t) * l_ind_len);
	for (int_t jb = 0; jb < nsupers; ++jb)   /* for each block column ... */
	{
	    int_t pc = PCOL( jb, grid );
	    if (mycol == pc && isNodeInMyGrid[jb] == 1)
		{
		    int_t ljb = LBj( jb, grid ); /* Local block number */
		    int_t  *index_host;
		    index_host = A_host->Lrowind_bc_ptr[ljb];

		    if (index_host != NULL)
			{
			    int_t nrbl  =   index_host[0]; /* number of L blocks */
			    int_t len   = index_host[1];   /* LDA of the nzval[] */
			    int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;

			    memcpy(&indtemp[temp_LrowindPtr[ljb]] , index_host, len1 * sizeof(int_t)) ;
			}
		}
	}

	checkCudaErrors(cudaMalloc( &tmp_ptr,  l_ind_len * sizeof(int_t))) ;
	A_gpu->LrowindVec = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->LrowindVec), indtemp, l_ind_len * sizeof(int_t), cudaMemcpyHostToDevice)) ;	

	#ifdef SuperLargeScaleGPUBuffer
	free (indtemp);
	#endif

	#ifdef SuperLargeScaleGPU
	if(A_gpu->isEmpty){
		
		int_t len_max = (GPUMemLimit + sizeof(double))/sizeof(double);		
		A_gpu->len_max = len_max;
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  len_max * sizeof(double)));
		A_gpu->LnzvalVec = (double *) tmp_ptr;
		checkCudaErrors(cudaMemset( (A_gpu->LnzvalVec), 0, len_max * sizeof(double)));
		A_gpu->Lnzval_bc_ptr_len = 0;

		A_gpu->LnzvalVec_host = (double**)malloc(l_k * sizeof(double*));

		A_gpu->isGPUUsed_Lnzval_bc_ptr_host = (int_t*)malloc(l_k * sizeof(int_t));
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_k * sizeof(int_t)));
		A_gpu->isGPUUsed_Lnzval_bc_ptr = (int_t *)tmp_ptr;		
		fillvalue_int(A_gpu->isGPUUsed_Lnzval_bc_ptr, l_k, GPUUnused);

		A_gpu->isCPUUsed_Lnzval_bc_ptr_host = (int_t*)malloc(l_k * sizeof(int_t));
		for(int_t i = 0; i < l_k; i++)
		{
			A_gpu->isCPUUsed_Lnzval_bc_ptr_host[i] = CPUUnused;
			A_gpu->isGPUUsed_Lnzval_bc_ptr_host[i] = GPUUnused;
		}		

		A_gpu->UsedOrder_Lnzval = (int_t**)malloc(2 * sizeof(int_t*));
		
	}
	else{		
		
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_val_len * sizeof(double)));
		A_gpu->LnzvalVec = (double *) tmp_ptr;
		checkCudaErrors(cudaMemset( (A_gpu->LnzvalVec), 0, l_val_len * sizeof(double)));
	}
	#else	

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_val_len * sizeof(double)));
	A_gpu->LnzvalVec = (double *) tmp_ptr;
	checkCudaErrors(cudaMemset( (A_gpu->LnzvalVec), 0, l_val_len * sizeof(double)));
	#endif

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_k * sizeof(int_t))) ;
	A_gpu->LrowindPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->LrowindPtr), temp_LrowindPtr, l_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	#ifdef SuperLargeScaleGPUBuffer
	free (temp_LrowindPtr);
	#endif

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  l_k * sizeof(int_t))) ;
	A_gpu->LnzvalPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->LnzvalPtr), temp_LnzvalPtr, l_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	A_gpu->LnzvalPtr_host = temp_LnzvalPtr;

	int_t *indtemp1 = (int_t *) malloc(sizeof(int_t) * u_ind_len);
	for ( int_t lb = 0; lb < u_k; ++lb)
	{
	    int_t *index_host;
	    index_host =  A_host->Ufstnz_br_ptr[lb];

	    if (index_host != NULL && isNodeInMyGrid[lb * Pr + myrow] == 1)
		{
		    int_t len1 = index_host[2];
		    memcpy(&indtemp1[temp_UrowindPtr[lb]] , index_host, sizeof(int_t)*len1);
		}
	}

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_ind_len * sizeof(int_t))) ;
	A_gpu->UrowindVec = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->UrowindVec), indtemp1, u_ind_len * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	#ifdef SuperLargeScaleGPUBuffer
	free (indtemp1);
	#endif

	#ifdef SuperLargeScaleGPU
	if(A_gpu->isEmpty){
		
		int_t len_max = A_gpu->len_max;
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  len_max * sizeof(double)));
		A_gpu->UnzvalVec = (double *) tmp_ptr;
		checkCudaErrors(cudaMemset( (A_gpu->UnzvalVec), 0, len_max * sizeof(double)));
		A_gpu->Unzval_br_ptr_len = 0;

		A_gpu->UnzvalVec_host = (double**)malloc(u_k * sizeof(double*));

		A_gpu->isGPUUsed_Unzval_br_ptr_host = (int_t*)malloc(u_k * sizeof(int_t));
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_k * sizeof(int_t)));
		A_gpu->isGPUUsed_Unzval_br_ptr = (int_t *)tmp_ptr;		
		fillvalue_int(A_gpu->isGPUUsed_Unzval_br_ptr, u_k, GPUUnused);

		A_gpu->isCPUUsed_Unzval_br_ptr_host = (int_t*)malloc(u_k * sizeof(int_t));
		for(int_t i = 0; i < u_k; i++)
		{
			A_gpu->isCPUUsed_Unzval_br_ptr_host[i] = CPUUnused;
			A_gpu->isGPUUsed_Unzval_br_ptr_host[i] = GPUUnused;
		}	

		A_gpu->UsedOrder_Unzval = (int_t**)malloc(2 * sizeof(int_t*));
			
	}
	else{		
		
		checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_val_len * sizeof(double)));
		A_gpu->UnzvalVec = (double *) tmp_ptr;
		checkCudaErrors(cudaMemset( (A_gpu->UnzvalVec), 0, u_val_len * sizeof(double)));
	}
	#else

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_val_len * sizeof(double)));
	A_gpu->UnzvalVec = (double *) tmp_ptr;
	checkCudaErrors(cudaMemset( (A_gpu->UnzvalVec), 0, u_val_len * sizeof(double)));
	#endif

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_k * sizeof(int_t))) ;
	A_gpu->UrowindPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->UrowindPtr), temp_UrowindPtr, u_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	#ifdef SuperLargeScaleGPUBuffer
	free (temp_UrowindPtr);
	#endif

	A_gpu->UnzvalPtr_host = temp_UnzvalPtr;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  u_k * sizeof(int_t))) ;
	A_gpu->UnzvalPtr = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->UnzvalPtr), temp_UnzvalPtr, u_k * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc(  &tmp_ptr,  (nsupers + 1)*sizeof(int_t))) ;
	A_gpu->xsup = (int_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( (A_gpu->xsup), xsup, (nsupers + 1)*sizeof(int_t), cudaMemcpyHostToDevice)) ;

	checkCudaErrors(cudaMalloc( &tmp_ptr,  sizeof(dLUstruct_gpu_t))) ;
	*dA_gpu = (dLUstruct_gpu_t *) tmp_ptr;
	checkCudaErrors(cudaMemcpy( *dA_gpu, A_gpu, sizeof(dLUstruct_gpu_t), cudaMemcpyHostToDevice)) ;

	#ifndef SuperLargeScaleGPUBuffer
	free (temp_LrowindPtr);
	free (temp_UrowindPtr);
	free (indtemp1);
	free (indtemp);
	#endif

	#if ( DEBUGlevel>=1 )
		CHECK_MALLOC (grid3d->iam, "Exit dCopyLUToGPU3D()");
	#endif

} /* end dCopyLUToGPU3D */



int dreduceAllAncestors3d_GPU(int_t ilvl, int_t* myNodeCount,
				int_t** treePerm,
				dLUValSubBuf_t*LUvsb,
				   dLUstruct_t* LUstruct,
			           gridinfo3d_t* grid3d,
				   dsluGPU_t *sluGPU,
				   d2Hreduce_t* d2Hred,
				   factStat_t *factStat,
				   HyP_t* HyP, SCT_t* SCT )
{
	#if ( DEBUGlevel>=1 )
		CHECK_MALLOC (grid3d->iam, "Enter dreduceAllAncestors3d_GPU()");
	#endif

    // first synchronize all cuda streams
    int superlu_acc_offload =   HyP->superlu_acc_offload;

    int_t maxLvl = log2i( (int_t) grid3d->zscp.Np) + 1;
    int_t myGrid = grid3d->zscp.Iam;
    gridinfo_t* grid = &(grid3d->grid2d);
    int_t* gpuLUreduced = factStat->gpuLUreduced;

	#ifdef SuperLargeScaleGPUBuffer
	if(LUstruct->Llu->isEmpty == 1)
	{
		myGrid = LUstruct->Llu->tempiam;
	}
	#endif

    int_t sender;
    if ((myGrid % (1 << (ilvl + 1))) == 0)
	{
	    sender = myGrid + (1 << ilvl);
	    
	}
    else
	{
	    sender = myGrid;
	}

    /*Reduce all the ancestors from the GPU*/
    if (myGrid == sender && superlu_acc_offload)
    {
        for (int_t streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
	{
	    double ttx = SuperLU_timer_();
	    cudaStreamSynchronize(sluGPU->funCallStreams[streamId]);
	    SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
	    sluGPU->lastOffloadStream[streamId] = -1;
	}

	for (int_t alvl = ilvl + 1; alvl < maxLvl; ++alvl)
	{
	    /* code */
	    // int_t atree = myTreeIdxs[alvl];
	    int_t nsAncestor = myNodeCount[alvl];
	    int_t* cAncestorList = treePerm[alvl];

	    for (int_t node = 0; node < nsAncestor; node++ )
	    {
	        int_t k = cAncestorList[node];
	        if (!gpuLUreduced[k])
		{
		    dinitD2Hreduce(k, d2Hred, 1,
				  HyP, sluGPU, grid, LUstruct, SCT);
		    int_t copyL_kljb = d2Hred->copyL_kljb;
		    int_t copyU_kljb = d2Hred->copyU_kljb;

		    double tt_start1 = SuperLU_timer_();
		    SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
		    if (copyL_kljb || copyU_kljb) SCT->PhiMemCpyCounter++;
		    dsendLUpanelGPU2HOST(k, d2Hred, sluGPU);
		    /*
		      Reduce the LU panels from GPU
		    */
		    dreduceGPUlu(1, d2Hred, sluGPU, SCT, grid, LUstruct);
		    gpuLUreduced[k] = 1;
		}
	    }
	}
    } /*if (myGrid == sender)*/

	#ifndef SuperLargeScaleGPUBuffer
    dreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
	                      LUvsb, LUstruct, grid3d, SCT );
	#endif
		
	#if ( DEBUGlevel>=1 )
		CHECK_MALLOC (grid3d->iam, "Exit dreduceAllAncestors3d_GPU()");
	#endif
    return 0;
} /* dreduceAllAncestors3d_GPU */


void dsyncAllfunCallStreams(dsluGPU_t* sluGPU, SCT_t* SCT)
{
    for (int streamId = 0; streamId < sluGPU->nCudaStreams; streamId++)
    {
        double ttx = SuperLU_timer_();
        cudaStreamSynchronize(sluGPU->funCallStreams[streamId]);
        SCT->PhiWaitTimer += SuperLU_timer_() - ttx;
        sluGPU->lastOffloadStream[streamId] = -1;
     }
}

#ifdef test
// void dblock_gemm_scatterTopLeft_GPU( double* bigV, int_t knsupc,  int_t klst,
// 				 int_t* lsub, int_t * usub, int_t ldt,
// 				 int* indirect, int* indirect2, HyP_t* HyP,
//                                  dLUstruct_t *LUstruct,
//                                  gridinfo_t* grid,
//                                  SCT_t*SCT, SuperLUStat_t *stat
//                                )
// {
// 	__global__ 
// 	void dblock_gemm_scatterTopLeft_kernel( double* bigV, int_t knsupc,  int_t klst,
// 					int_t* lsub, int_t * usub, int_t ldt,
// 					int* indirect, int* indirect2, HyP_t* HyP,
// 									dLUstruct_t *LUstruct,
// 									gridinfo_t* grid,
// 									SCT_t*SCT, SuperLUStat_t *stat
// 								);

// 	dim3 dimBlock(THREAD_BLOCK_SIZE);   // 1d thread
// 	dim3 dimGrid(HyP->lookAheadBlk * HyP->num_u_blks / dimBlock.x + 1);
// 	dblock_gemm_scatterTopLeft_kernel<<<dimBlock,dimBlock>>>(bigV, knsupc, klst, lsub,
// 		usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
// }


// __global__
// void dblock_gemm_scatterTopLeft_kernel( double* bigV, int_t knsupc,  int_t klst,
// 				 int_t* lsub, int_t * usub, int_t ldt,
// 				 int* indirect, int* indirect2, HyP_t* HyP,
//                                  dLUstruct_t *LUstruct,
//                                  gridinfo_t* grid,
//                                  SCT_t*SCT, SuperLUStat_t *stat
//                                )
// {
// 	__device__ inline
// 	void
// 	device_dblock_gemm_scatter( int_t lb, int_t j,
// 						Ublock_info_t *Ublock_info,
// 						Remain_info_t *Remain_info,
// 						double *L_mat, int ldl,
// 						double *U_mat, int ldu,
// 						double *bigV,
// 						// int_t jj0,
// 						int_t knsupc,  int_t klst,
// 						int_t *lsub, int_t *usub, int_t ldt,
// 						int_t thread_id,
// 						int *indirect,
// 						int *indirect2,
// 						int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
// 						int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
// 						int_t *xsup, int npcol, int nprow
// 	#ifdef SCATTER_PROFILE
// 						, double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
// 	#endif
// 					);
//     Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
//     dLocalLU_t *Llu = LUstruct->Llu;
//     int_t* xsup = Glu_persist->xsup;
//     int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
//     int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
//     double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;
//     double** Unzval_br_ptr = Llu->Unzval_br_ptr;

// 	int thread_id = threadIdx.x;

// 	if(thread_id<HyP->lookAheadBlk * HyP->num_u_blks)
// 	{
// 		int_t j = thread_id / HyP->lookAheadBlk;
// 		int_t lb = thread_id % HyP->lookAheadBlk;
		
// 	//    printf("Thread's ID %lld \n", thread_id);
// 		//unsigned long long t1 = _rdtsc();
// 		// double t1 = SuperLU_timer_();
// 		device_dblock_gemm_scatter( lb, j, HyP->Ublock_info, HyP->lookAhead_info,
// 				HyP->lookAhead_L_buff, HyP->Lnbrow,
// 							HyP->bigU_host, HyP->ldu,
// 							bigV, knsupc,  klst, lsub,  usub, ldt, thread_id,
// 				indirect, indirect2,
// 							Lrowind_bc_ptr, Lnzval_bc_ptr, Ufstnz_br_ptr, Unzval_br_ptr,
// 				xsup, grid->npcol, grid->nprow
// 	#ifdef SCATTER_PROFILE
// 							, SCT->Host_TheadScatterMOP, SCT->Host_TheadScatterTimer
// 	#endif
// 						);
// 		//unsigned long long t2 = _rdtsc();
// 		// double t2 = SuperLU_timer_();
// 		// SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double) (t2 - t1);
// 	}
// } /* dgemm_scatterTopLeft */

// __device__ inline
// void
// device_dblock_gemm_scatter( int_t lb, int_t j,
//                     Ublock_info_t *Ublock_info,
//                     Remain_info_t *Remain_info,
//                     double *L_mat, int ldl,
//                     double *U_mat, int ldu,
//                     double *bigV,
//                     // int_t jj0,
//                     int_t knsupc,  int_t klst,
//                     int_t *lsub, int_t *usub, int_t ldt,
//                     int_t thread_id,
//                     int *indirect,
//                     int *indirect2,
//                     int_t **Lrowind_bc_ptr, double **Lnzval_bc_ptr,
//                     int_t **Ufstnz_br_ptr, double **Unzval_br_ptr,
//                     int_t *xsup, int npcol, int nprow
// #ifdef SCATTER_PROFILE
//                     , double *Host_TheadScatterMOP, double *Host_TheadScatterTimer
// #endif
//                   )
// {
// 	__device__ inline
// 	void device_superlu_dgemm(char *transa, char *transb,
// 		int m, int n, int k, double alpha, double *a,
// 		int lda, double *b, int ldb, double beta, double *c, int ldc);
// 	__device__ inline
// 	void   // SHERRY: ALMOST the same as dscatter_u in dscatter.c
// 	device_scatter_u2 (int_t ib,
// 				int_t jb,
// 				int_t nsupc,
// 				int_t iukp,
// 				int_t *xsup,
// 				int_t klst,
// 				int_t nbrow,
// 				int_t lptr,
// 				int_t temp_nbrow,
// 				int_t *lsub,
// 				int_t *usub,
// 				double *tempv,
// 				int *indirect,
// 				int_t **Ufstnz_br_ptr, double **Unzval_br_ptr, int nprow);
// 	__device__ inline
// 	void
// 	device_dscatter_l (
// 				int ib,    /* row block number of source block L(i,k) */
// 				int ljb,   /* local column block number of dest. block L(i,j) */
// 				int nsupc, /* number of columns in destination supernode */
// 				int_t iukp, /* point to destination supernode's index[] */
// 				int_t* xsup,
// 				int klst,
// 				int nbrow,  /* LDA of the block in tempv[] */
// 				int_t lptr, /* Input, point to index[] location of block L(i,k) */
// 			int temp_nbrow, /* number of rows of source block L(i,k) */
// 				int_t* usub,
// 				int_t* lsub,
// 				double *tempv,
// 				int* indirect_thread,int* indirect2,
// 				int_t ** Lrowind_bc_ptr, double **Lnzval_bc_ptr);

//     int *indirect_thread = indirect + ldt * thread_id;
//     int *indirect2_thread = indirect2 + ldt * thread_id;
//     double *tempv1 = bigV + thread_id * ldt * ldt;

//     /* Getting U block information */

//     int_t iukp =  Ublock_info[j].iukp;
//     int_t jb   =  Ublock_info[j].jb;
//     int_t nsupc = SuperSize(jb);
//     int_t ljb = jb/npcol;  /*LBj (jb, grid)*/
//     int_t st_col;
//     int ncols;
//     // if (j > jj0)
//     if (j > 0)
//     {
//         ncols  = Ublock_info[j].full_u_cols - Ublock_info[j - 1].full_u_cols;
//         st_col = Ublock_info[j - 1].full_u_cols;
//     }
//     else
//     {
//         ncols  = Ublock_info[j].full_u_cols;
//         st_col = 0;
//     }

//     /* Getting L block information */
//     int_t lptr = Remain_info[lb].lptr;
//     int_t ib   = Remain_info[lb].ib;
//     int temp_nbrow = lsub[lptr + 1];
//     lptr += LB_DESCRIPTOR;
//     int cum_nrow = (lb == 0 ? 0 : Remain_info[lb - 1].FullRow);
//     double alpha = 1.0, beta = 0.0;

//     /* calling DGEMM */
//     // printf(" m %d n %d k %d ldu %d ldl %d st_col %d \n",temp_nbrow,ncols,ldu,ldl,st_col );
//     device_superlu_dgemm("N", "N", temp_nbrow, ncols, ldu, alpha,
//                 &L_mat[(knsupc - ldu)*ldl + cum_nrow], ldl,
//                 &U_mat[st_col * ldu], ldu,
//                 beta, tempv1, temp_nbrow);
    
//     // printf("SCU update: (%d, %d)\n",ib,jb );
// #ifdef SCATTER_PROFILE
//     double ttx = SuperLU_timer_();
// #endif
//     /*Now scattering the block*/
//     if (ib < jb)
//     {
//         device_scatter_u2 (
//             ib, jb,
//             nsupc, iukp, xsup,
//             klst, temp_nbrow,
//             lptr, temp_nbrow, lsub,
//             usub, tempv1,
//             indirect_thread,
//             Ufstnz_br_ptr,
//             Unzval_br_ptr,
//             nprow
//         );
//     }
//     else
//     {
//         //scatter_l (    Sherry
// 		device_dscatter_l (
//             ib, ljb, nsupc, iukp, xsup, klst, temp_nbrow, lptr,
//             temp_nbrow, usub, lsub, tempv1,
//             indirect_thread, indirect2_thread,
//             Lrowind_bc_ptr, Lnzval_bc_ptr
//         );

//     }

//     // #pragma omp atomic
//     // stat->ops[FACT] += 2*temp_nbrow*ncols*ldu + temp_nbrow*ncols;

// #ifdef SCATTER_PROFILE
//     double t_s = SuperLU_timer_() - ttx;
//     Host_TheadScatterMOP[thread_id * ((192 / 8) * (192 / 8)) + ((CEILING(temp_nbrow, 8) - 1)   +  (192 / 8) * (CEILING(ncols, 8) - 1))]
//     += 3.0 * (double ) temp_nbrow * (double ) ncols;
//     Host_TheadScatterTimer[thread_id * ((192 / 8) * (192 / 8)) + ((CEILING(temp_nbrow, 8) - 1)   +  (192 / 8) * (CEILING(ncols, 8) - 1))]
//     += t_s;
// #endif
// } /* dblock_gemm_scatter */

// __device__ inline
// void device_superlu_dgemm(char *transa, char *transb,
// 	int m, int n, int k, double alpha, double *a,
// 	int lda, double *b, int ldb, double beta, double *c, int ldc)
// {
// 	#if 1
// 	__device__ inline
// /* Subroutine */ void device_dgemm_(char *transa, char *transb, integer *m, integer *
// 	n, integer *k, doublereal *alpha, doublereal *a, integer *lda, 
// 	doublereal *b, integer *ldb, doublereal *beta, doublereal *c, integer 
// 	*ldc);
// #ifdef _CRAY
// _fcd ftcs = _cptofcd(transa, strlen(transa));
// _fcd ftcs1 = _cptofcd(transb, strlen(transb));
// return SGEMM(ftcs, ftcs1, &m, &n, &k,
//    &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
// #elif defined(USE_VENDOR_BLAS)
// 	device_dgemm_(transa, transb, &m, &n, &k,
// 	&alpha, a, &lda, b, &ldb, &beta, c, &ldc);
// #else
// 	device_dgemm_(transa, transb, &m, &n, &k,
// 	&alpha, a, &lda, b, &ldb, &beta, c, &ldc);
// #endif
// 	#endif
// }

// __device__ inline
// /* Subroutine */ void device_dgemm_(char *transa, char *transb, integer *m, integer *
// 	n, integer *k, doublereal *alpha, doublereal *a, integer *lda, 
// 	doublereal *b, integer *ldb, doublereal *beta, doublereal *c, integer 
// 	*ldc)
// {


//     /* System generated locals */
//     integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, 
// 	    i__3;

//     /* Local variables */
//     static integer info;
//     static logical nota, notb;
//     static doublereal temp;
//     static integer i, j, l, ncola;
//     static integer nrowa, nrowb;


// /*  Purpose   
//     =======   

//     DGEMM  performs one of the matrix-matrix operations   

//        C := alpha*op( A )*op( B ) + beta*C,   

//     where  op( X ) is one of   

//        op( X ) = X   or   op( X ) = X',   

//     alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
  
//     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix. 
  

//     Parameters   
//     ==========   

//     TRANSA - CHARACTER*1.   
//              On entry, TRANSA specifies the form of op( A ) to be used in 
  
//              the matrix multiplication as follows:   

//                 TRANSA = 'N' or 'n',  op( A ) = A.   

//                 TRANSA = 'T' or 't',  op( A ) = A'.   

//                 TRANSA = 'C' or 'c',  op( A ) = A'.   

//              Unchanged on exit.   

//     TRANSB - CHARACTER*1.   
//              On entry, TRANSB specifies the form of op( B ) to be used in 
  
//              the matrix multiplication as follows:   

//                 TRANSB = 'N' or 'n',  op( B ) = B.   

//                 TRANSB = 'T' or 't',  op( B ) = B'.   

//                 TRANSB = 'C' or 'c',  op( B ) = B'.   

//              Unchanged on exit.   

//     M      - INTEGER.   
//              On entry,  M  specifies  the number  of rows  of the  matrix 
  
//              op( A )  and of the  matrix  C.  M  must  be at least  zero. 
  
//              Unchanged on exit.   

//     N      - INTEGER.   
//              On entry,  N  specifies the number  of columns of the matrix 
  
//              op( B ) and the number of columns of the matrix C. N must be 
  
//              at least zero.   
//              Unchanged on exit.   

//     K      - INTEGER.   
//              On entry,  K  specifies  the number of columns of the matrix 
  
//              op( A ) and the number of rows of the matrix op( B ). K must 
  
//              be at least  zero.   
//              Unchanged on exit.   

//     ALPHA  - DOUBLE PRECISION.   
//              On entry, ALPHA specifies the scalar alpha.   
//              Unchanged on exit.   

//     A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is 
  
//              k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.   
//              Before entry with  TRANSA = 'N' or 'n',  the leading  m by k 
  
//              part of the array  A  must contain the matrix  A,  otherwise 
  
//              the leading  k by m  part of the array  A  must contain  the 
  
//              matrix A.   
//              Unchanged on exit.   

//     LDA    - INTEGER.   
//              On entry, LDA specifies the first dimension of A as declared 
  
//              in the calling (sub) program. When  TRANSA = 'N' or 'n' then 
  
//              LDA must be at least  max( 1, m ), otherwise  LDA must be at 
  
//              least  max( 1, k ).   
//              Unchanged on exit.   

//     B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is 
  
//              n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.   
//              Before entry with  TRANSB = 'N' or 'n',  the leading  k by n 
  
//              part of the array  B  must contain the matrix  B,  otherwise 
  
//              the leading  n by k  part of the array  B  must contain  the 
  
//              matrix B.   
//              Unchanged on exit.   

//     LDB    - INTEGER.   
//              On entry, LDB specifies the first dimension of B as declared 
  
//              in the calling (sub) program. When  TRANSB = 'N' or 'n' then 
  
//              LDB must be at least  max( 1, k ), otherwise  LDB must be at 
  
//              least  max( 1, n ).   
//              Unchanged on exit.   

//     BETA   - DOUBLE PRECISION.   
//              On entry,  BETA  specifies the scalar  beta.  When  BETA  is 
  
//              supplied as zero then C need not be set on input.   
//              Unchanged on exit.   

//     C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).   
//              Before entry, the leading  m by n  part of the array  C must 
  
//              contain the matrix  C,  except when  beta  is zero, in which 
  
//              case C need not be set on entry.   
//              On exit, the array  C  is overwritten by the  m by n  matrix 
  
//              ( alpha*op( A )*op( B ) + beta*C ).   

//     LDC    - INTEGER.   
//              On entry, LDC specifies the first dimension of C as declared 
  
//              in  the  calling  (sub)  program.   LDC  must  be  at  least 
  
//              max( 1, m ).   
//              Unchanged on exit.   


//     Level 3 Blas routine.   

//     -- Written on 8-February-1989.   
//        Jack Dongarra, Argonne National Laboratory.   
//        Iain Duff, AERE Harwell.   
//        Jeremy Du Croz, Numerical Algorithms Group Ltd.   
//        Sven Hammarling, Numerical Algorithms Group Ltd.   



//        Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not 
  
//        transposed and set  NROWA, NCOLA and  NROWB  as the number of rows 
  
//        and  columns of  A  and the  number of  rows  of  B  respectively. 
  

    
//    Parameter adjustments   
//        Function Body */

// #define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]
// #define B(I,J) b[(I)-1 + ((J)-1)* ( *ldb)]
// #define C(I,J) c[(I)-1 + ((J)-1)* ( *ldc)]

//     nota = (transa=="N");
//     notb = (transb=="N");
//     if (nota) {
// 	nrowa = *m;
// 	ncola = *k;
//     } else {
// 	nrowa = *k;
// 	ncola = *m;
//     }
//     if (notb) {
// 	nrowb = *k;
//     } else {
// 	nrowb = *n;
//     }

// /*     Test the input parameters. */

//     info = 0;
//     if (! nota && transa!="C" && transa!="T") {
// 	info = 1;
//     } else if (! notb && transb!="C" && transb!="T") {
// 	info = 2;
//     } else if (*m < 0) {
// 	info = 3;
//     } else if (*n < 0) {
// 	info = 4;
//     } else if (*k < 0) {
// 	info = 5;
//     } else if (*lda < max(1,nrowa)) {
// 	info = 8;
//     } else if (*ldb < max(1,nrowb)) {
// 	info = 10;
//     } else if (*ldc < max(1,*m)) {
// 	info = 13;
//     }
//     if (info != 0) {
// 		printf("** On entry to DGEMM, parameter number %2d had an illegal value\n",
// 			info);
// 		return;
//     }

// /*     Quick return if possible. */

//     if (*m == 0 || *n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
// 	return;
//     }

// /*     And if  alpha.eq.zero. */

//     if (*alpha == 0.) {
// 	if (*beta == 0.) {
// 	    i__1 = *n;
// 	    for (j = 1; j <= *n; ++j) {
// 		i__2 = *m;
// 		for (i = 1; i <= *m; ++i) {
// 		    C(i,j) = 0.;
// /* L10: */
// 		}
// /* L20: */
// 	    }
// 	} else {
// 	    i__1 = *n;
// 	    for (j = 1; j <= *n; ++j) {
// 		i__2 = *m;
// 		for (i = 1; i <= *m; ++i) {
// 		    C(i,j) = *beta * C(i,j);
// /* L30: */
// 		}
// /* L40: */
// 	    }
// 	}
// 	return;
//     }

// /*     Start the operations. */

//     if (notb) {
// 	if (nota) {

// /*           Form  C := alpha*A*B + beta*C. */

// 	    i__1 = *n;
// 	    for (j = 1; j <= *n; ++j) {
// 		if (*beta == 0.) {
// 		    i__2 = *m;
// 		    for (i = 1; i <= *m; ++i) {
// 			C(i,j) = 0.;
// /* L50: */
// 		    }
// 		} else if (*beta != 1.) {
// 		    i__2 = *m;
// 		    for (i = 1; i <= *m; ++i) {
// 			C(i,j) = *beta * C(i,j);
// /* L60: */
// 		    }
// 		}
// 		i__2 = *k;
// 		for (l = 1; l <= *k; ++l) {
// 		    if (B(l,j) != 0.) {
// 			temp = *alpha * B(l,j);
// 			i__3 = *m;
// 			for (i = 1; i <= *m; ++i) {
// 			    C(i,j) += temp * A(i,l);
// /* L70: */
// 			}
// 		    }
// /* L80: */
// 		}
// /* L90: */
// 	    }
// 	} else {

// /*           Form  C := alpha*A'*B + beta*C */

// 	    i__1 = *n;
// 	    for (j = 1; j <= *n; ++j) {
// 		i__2 = *m;
// 		for (i = 1; i <= *m; ++i) {
// 		    temp = 0.;
// 		    i__3 = *k;
// 		    for (l = 1; l <= *k; ++l) {
// 			temp += A(l,i) * B(l,j);
// /* L100: */
// 		    }
// 		    if (*beta == 0.) {
// 			C(i,j) = *alpha * temp;
// 		    } else {
// 			C(i,j) = *alpha * temp + *beta * C(i,j);
// 		    }
// /* L110: */
// 		}
// /* L120: */
// 	    }
// 	}
//     } else {
// 	if (nota) {

// /*           Form  C := alpha*A*B' + beta*C */

// 	    i__1 = *n;
// 	    for (j = 1; j <= *n; ++j) {
// 		if (*beta == 0.) {
// 		    i__2 = *m;
// 		    for (i = 1; i <= *m; ++i) {
// 			C(i,j) = 0.;
// /* L130: */
// 		    }
// 		} else if (*beta != 1.) {
// 		    i__2 = *m;
// 		    for (i = 1; i <= *m; ++i) {
// 			C(i,j) = *beta * C(i,j);
// /* L140: */
// 		    }
// 		}
// 		i__2 = *k;
// 		for (l = 1; l <= *k; ++l) {
// 		    if (B(j,l) != 0.) {
// 			temp = *alpha * B(j,l);
// 			i__3 = *m;
// 			for (i = 1; i <= *m; ++i) {
// 			    C(i,j) += temp * A(i,l);
// /* L150: */
// 			}
// 		    }
// /* L160: */
// 		}
// /* L170: */
// 	    }
// 	} else {

// /*           Form  C := alpha*A'*B' + beta*C */

// 	    i__1 = *n;
// 	    for (j = 1; j <= *n; ++j) {
// 		i__2 = *m;
// 		for (i = 1; i <= *m; ++i) {
// 		    temp = 0.;
// 		    i__3 = *k;
// 		    for (l = 1; l <= *k; ++l) {
// 			temp += A(l,i) * B(j,l);
// /* L180: */
// 		    }
// 		    if (*beta == 0.) {
// 			C(i,j) = *alpha * temp;
// 		    } else {
// 			C(i,j) = *alpha * temp + *beta * C(i,j);
// 		    }
// /* L190: */
// 		}
// /* L200: */
// 	    }
// 	}
//     }

// /*     End of DGEMM . */

// } /* dgemm_ */

// __device__ inline
// void   // SHERRY: ALMOST the same as dscatter_u in dscatter.c
// device_scatter_u2 (int_t ib,
//            int_t jb,
//            int_t nsupc,
//            int_t iukp,
//            int_t *xsup,
//            int_t klst,
//            int_t nbrow,
//            int_t lptr,
//            int_t temp_nbrow,
//            int_t *lsub,
//            int_t *usub,
//            double *tempv,
//            int *indirect,
//            int_t **Ufstnz_br_ptr, double **Unzval_br_ptr, int nprow)
// {
// #ifdef PI_DEBUG
//     printf ("A(%d,%d) goes to U block \n", ib, jb);
// #endif
//     int_t jj, i, fnz;
//     int_t segsize;
//     double *ucol;
//     int_t ilst = FstBlockC (ib + 1);
//     int_t lib = ib/nprow; //LBi (ib, grid);
//     int_t *index = Ufstnz_br_ptr[lib];

//     /* reinitialize the pointer to each row of U */
//     int_t iuip_lib, ruip_lib;
//     iuip_lib = BR_HEADER;
//     ruip_lib = 0;

//     int_t ijb = index[iuip_lib];
//     while (ijb < jb)            /* Search for dest block. */
//     {
//         ruip_lib += index[iuip_lib + 1];

//         iuip_lib += UB_DESCRIPTOR + SuperSize (ijb);
//         ijb = index[iuip_lib];
//     }
//     /* Skip descriptor.  Now point_t to fstnz index of
//        block U(i,j). */

//     for (i = 0; i < temp_nbrow; ++i)
//     {
//         indirect[i] = lsub[lptr + i] ;
//     }

//     iuip_lib += UB_DESCRIPTOR;

//     ucol = &Unzval_br_ptr[lib][ruip_lib];
//     for (jj = 0; jj < nsupc; ++jj)
//     {
//         segsize = klst - usub[iukp + jj];
//         fnz = index[iuip_lib++];
//         ucol -= fnz;
//         if (segsize)            /* Nonzero segment in U(k.j). */
//         {
//             for (i = 0; i < temp_nbrow; ++i)
//             {
//                 ucol[indirect[i]] -= tempv[i];
//             }                   /* for i=0..temp_nbropw */
//             tempv += nbrow;

//         } /*if segsize */
//         ucol += ilst ;

//     } /*for jj=0:nsupc */

// }

// __device__ inline
// void
// device_dscatter_l (
//            int ib,    /* row block number of source block L(i,k) */
//            int ljb,   /* local column block number of dest. block L(i,j) */
//            int nsupc, /* number of columns in destination supernode */
//            int_t iukp, /* point to destination supernode's index[] */
//            int_t* xsup,
//            int klst,
//            int nbrow,  /* LDA of the block in tempv[] */
//            int_t lptr, /* Input, point to index[] location of block L(i,k) */
// 	   int temp_nbrow, /* number of rows of source block L(i,k) */
//            int_t* usub,
//            int_t* lsub,
//            double *tempv,
//            int* indirect_thread,int* indirect2,
//            int_t ** Lrowind_bc_ptr, double **Lnzval_bc_ptr)
// {

//     int_t rel, i, segsize, jj;
//     double *nzval;
//     int_t *index = Lrowind_bc_ptr[ljb];
//     int_t ldv = index[1];       /* LDA of the destination lusup. */
//     int_t lptrj = BC_HEADER;
//     int_t luptrj = 0;
//     int_t ijb = index[lptrj];

//     while (ijb != ib)  /* Search for destination block L(i,j) */
//     {
//         luptrj += index[lptrj + 1];
//         lptrj += LB_DESCRIPTOR + index[lptrj + 1];
//         ijb = index[lptrj];
//     }

//     /*
//      * Build indirect table. This is needed because the indices are not sorted
//      * in the L blocks.
//      */
//     int_t fnz = FstBlockC (ib);
//     int_t dest_nbrow;
//     lptrj += LB_DESCRIPTOR;
//     dest_nbrow=index[lptrj - 1];

//     for (i = 0; i < dest_nbrow; ++i) {
//         rel = index[lptrj + i] - fnz;
//         indirect_thread[rel] = i;

//     }

//     /* can be precalculated? */
//     for (i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
//         rel = lsub[lptr + i] - fnz;
//         indirect2[i] =indirect_thread[rel];
//     }

//     nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */

//     for (jj = 0; jj < nsupc; ++jj) {
//         segsize = klst - usub[iukp + jj];
//         if (segsize) {

//             for (i = 0; i < temp_nbrow; ++i) {
//                 nzval[indirect2[i]] -= tempv[i];
//             }
//             tempv += nbrow;
//         }
//         nzval += ldv;
//     }

// } /* dscatter_l */


#endif

#if 0
__global__
void test_gpu()
{
	__device__ inline void hello_gpu();
	int thread_id=threadIdx.x;
	hello_gpu();
}

__device__ inline
void hello_gpu()
{
	__device__ inline void hello_gpu2();
	printf("hello1\n");
	hello_gpu2();
}

__device__ inline
void hello_gpu2()
{
	printf("hello2\n");
}
#endif

