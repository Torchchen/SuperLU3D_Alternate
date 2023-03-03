/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Driver program for PDGSSVX example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * December 6, 2018
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cusparse.h"
#include "cusolverSp.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

// static void load_ddata_markders_vector_txt(double *data,int_t n,int_t ntimestep,int_t index);
// static void load_idata_markders_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index);
// static void load_idata_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index);
// static void load_ddata_vector_txt(double *data,int_t n,int_t ntimestep,int_t index);

#define FORTRAN_TXT

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PDDRIVE.
 *
 * This example illustrates how to use PDGSSVX with the full
 * (default) options to solve a linear system.
 * 
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pdgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> pddrive -r <proc rows> -c <proc columns> big.rua
 * </pre>
 */

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dSOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
    double   *b, *xtrue;
    int    m, n;
    int      nprow, npcol;
    int      iam, info, ldb, ldx, nrhs;
    char     **cpp, c, *postfix;;
    FILE *fp, *fopen();
    int cpp_defs();
    int ii, omp_mpi_level;

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    MPI_Init( &argc, &argv );
    // MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &omp_mpi_level); 

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	cudaSetDevice(rank%4);
    // omp_set_num_threads(8);

	

#if ( VAMPIR>=1 )
    VT_traceoff(); 
#endif

#if ( VTUNE>=1 )
	__itt_pause();
#endif
	
    /* Parse command line argv[]. */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows    (default %4d)\n", nprow);
		  printf("\t-c <int>: process columns (default %4d)\n", npcol);
		  exit(0);
		  break;
	      case 'r': nprow = atoi(*cpp);
		        break;
	      case 'c': npcol = atoi(*cpp);
		        break;
	    }
	} else { /* Last arg is considered a filename */
	    if ( !(fp = fopen(*cpp, "r")) ) {
                ABORT("File does not exist");
            }
	    break;
	}
    }

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID. 
       ------------------------------------------------------------*/
    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);
	
    if(grid.iam==0){
        // MPI_Query_thread(&omp_mpi_level);
        // switch (omp_mpi_level) {
        // case MPI_THREAD_SINGLE:
        //     printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
        //     fflush(stdout);
        // break;
        // case MPI_THREAD_FUNNELED:
        //     printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
        //     fflush(stdout);
        // break;
        // case MPI_THREAD_SERIALIZED:
        //     printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
        //     fflush(stdout);
        // break;
        // case MPI_THREAD_MULTIPLE:
        //     printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
        //     fflush(stdout);
        // break;
        // }
        fflush(stdout);
	}
	
    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if ( iam >= nprow * npcol || iam ==-1 ) goto out;

    if ( !iam ) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid:\t\t%d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    for(ii = 0;ii<strlen(*cpp);ii++){
	if((*cpp)[ii]=='.'){
		postfix = &((*cpp)[ii+1]);
	}
    }
    // printf("%s\n", postfix);
	
    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
       ------------------------------------------------------------*/
    dcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, postfix, &grid);

    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES;
	options.DiagInv           = NO;
     */
    set_default_options_dist(&options);
    options.ReplaceTinyPivot = YES;
    options.ParSymbFact       = YES;
    options.ColPerm           =PARMETIS;
    
#if 0
    options.RowPerm = NOROWPERM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO; 
    options.ReplaceTinyPivot = YES;
#endif

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    m = A.nrow;
    n = A.ncol;

    /* Initialize ScalePermstruct and LUstruct. */
    dScalePermstructInit(m, n, &ScalePermstruct);
    dLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

   
    /* Call the linear equation solver. */
    pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
	    &LUstruct, &SOLVEstruct, berr, &stat, &info);

    if(!iam){
        printf("info:%d\n");
    }    

    /* Check the accuracy of the solution. */
    // pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
	// 	     nrhs, b, ldb, xtrue, ldx, grid.comm);

    double *r;
    int_t ntimestep,index,matrixlen[3];
    int_t nonz,nrow,ncol;
    index=1004;
    ntimestep=0;

    int slen=strlen("/home/412-23/collision_model18/data/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    char *filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/collision_model18/data/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/collision_model18/data/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    nonz=0;
    nrow=0;
    ncol=0;
    ntimestep=0;
    if(!access(filename,0)){
        load_idata_vector_txt(matrixlen,3,0,1004);
        nonz=matrixlen[0];
        nrow=matrixlen[1];
        ncol=matrixlen[1];
        ntimestep=matrixlen[2];
    }

    MPI_Barrier(grid.comm);

    // get x_global
    double *x_global;
    x_global=doubleMalloc_dist(nrow);
    int_t m_loc, fst_row, nnz_loc,m_loc_fst,inx,i,j;
    double *tmp;
    tmp=doubleMalloc_dist(nrow);

    MPI_Barrier(grid.comm);

    for(i=0;i<nrow;i++)
    {
        x_global[i]=0.0;
    }

    MPI_Barrier(grid.comm);

    /* Compute the number of rows to be distributed to local process */
    m_loc = nrow / (grid.nprow * grid.npcol); 
    m_loc_fst = m_loc;
    /* When nrow / procs is not an integer */
    if ((m_loc * grid.nprow * grid.npcol) != nrow) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
        if (iam == (grid.nprow * grid.npcol - 1)) /* last proc. gets all*/
	        m_loc = nrow - m_loc * (grid.nprow * grid.npcol - 1);
    } 

    MPI_Barrier(grid.comm);

    fst_row = iam * m_loc_fst;
    /* Get the local part of x_global */
    for (j = 0; j < nrhs; ++j) 
    {
        for (i = 0; i < m_loc; ++i)
        {            
            x_global[i + fst_row + j*ncol] = b[i + j*m_loc];
        }
    }

    MPI_Barrier(grid.comm);

    MPI_Barrier(grid.comm);    
    MPI_Status status;

    if(iam){
        for(j=0;j<nrow;j++)
        {
            tmp[j]=x_global[j];
        }
        MPI_Send(tmp,nrow,MPI_DOUBLE,0,0,grid.comm);
    }
    else{
        for(i=1;i<nprow*npcol;i++)
        {
            MPI_Recv(tmp,nrow,MPI_DOUBLE,i,0,grid.comm,&status);
            for(j=0;j<nrow;j++)
            {
                x_global[j]+=tmp[j];
            }
        }
    }

    MPI_Barrier(grid.comm);
    
    // if(!iam){
        
    //     for(i=0;i<nprow*npcol;i++)
    //     {
    //         for(j=0;j<100;j++)
    //         {
    //             printf("rank %d:x[%d]=%f\n",iam,i*m_loc+j,x_global[i*m_loc+j]);
    //         }
    //     }
    // }    

    MPI_Barrier(grid.comm);

    if(!iam){

        if(nrow>0 && nonz>0 && ntimestep>0){
            r=doubleMalloc_dist(nrow);
            load_ddata_vector_txt(r,nrow,ntimestep,1003);
        }

        if(nrow>0 && nonz>0 && ntimestep>0){
            int_t *colptr,*rowind;
            double *nzval;
            colptr=(int_t*)malloc((ncol+1)*sizeof(int_t));
            rowind=(int_t*)malloc(nonz*sizeof(int_t));
            nzval=doubleMalloc_dist(nonz);
            load_idata_vector_txt(colptr,ncol+1,ntimestep,1001);
            load_idata_markders_vector_txt(rowind,nonz,ntimestep,1002);
            load_ddata_markders_vector_txt(nzval,nonz,ntimestep,1000);

            #if defined (FORTRAN_TXT)
            int_t i;
            for(i=0;i<ncol+1;i++)
            {
                colptr[i]--;
            }
            for(i=0;i<nonz;i++)
            {
                rowind[i]--;
            }
            #endif

            cusparseHandle_t handle=0;
            cusparseMatDescr_t descr = 0;
            cusparseCreate(&handle);
            cusparseCreateMatDescr(&descr);
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

            cublasHandle_t cublasHandle = 0;
            cublasCreate(&cublasHandle);

            // matrix A, b, x on device
            double *d_csrValA, *d_b, *d_x, *d_r;
            int *d_csrRowIndA, *d_csrColPtrA, *d_cooRowIndA;
            cudaMalloc((void **)&d_csrValA, nonz * sizeof(double));
            cudaMalloc((void **)&d_csrRowIndA, (nrow + 1) * sizeof(int));
            cudaMalloc((void **)&d_csrColPtrA, nonz * sizeof(int));
            cudaMalloc((void **)&d_b, nrow * sizeof(double));
            cudaMalloc((void **)&d_x, nrow * sizeof(double));
            cudaMalloc((void **)&d_r, ncol * sizeof(double));
            cudaMemcpy(d_x, x_global, nrow * sizeof(double), cudaMemcpyHostToDevice);
            // transfer data from host to device
            cudaMemcpy(d_csrValA, nzval, nonz * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, r, nrow * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_csrRowIndA, colptr, (nrow + 1) * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_csrColPtrA, rowind, nonz * sizeof(int), cudaMemcpyHostToDevice);

            // local variables
            double c_zero=0;
            double c_one=1;
            double c_neg_one=-1;
            double residual,residual_b;
            // r=b-A*x
            cublasDnrm2(cublasHandle, ncol, d_b, 1, &residual_b);
            cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nrow, nrow, nonz, &c_one, descr, d_csrValA, d_csrRowIndA, d_csrColPtrA, d_x, &c_zero, d_r);
            cublasDaxpy(cublasHandle, ncol, &c_neg_one, d_b, 1, d_r, 1);
            cublasDnrm2(cublasHandle, ncol, d_r, 1, &residual);
            residual/=residual_b;

            printf("residual:%f\n",residual);

            SUPERLU_FREE(rowind);
            SUPERLU_FREE(colptr);
            SUPERLU_FREE(nzval);
            SUPERLU_FREE(r);
            cudaFree(d_csrValA);
            cudaFree(d_csrRowIndA);
            cudaFree(d_csrColPtrA);
            cudaFree(d_b);
            cudaFree(d_x);
            cudaFree(d_r);
            cusparseDestroy(handle);
            cusparseDestroyMatDescr(descr);
            cublasDestroy(cublasHandle);
        }
        
    }

    SUPERLU_FREE(x_global);

    return;

    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    PStatFree(&stat);
    Destroy_CompRowLoc_Matrix_dist(&A);
    dScalePermstructFree(&ScalePermstruct);
    dDestroy_LU(n, &grid, &LUstruct);
    dLUstructFree(&LUstruct);
    // if ( options.SolveInitialized ) {
    //     dSolveFinalize(&options, &SOLVEstruct);
    // }
    
    SUPERLU_FREE(b);
    SUPERLU_FREE(xtrue);
    SUPERLU_FREE(berr);
    fclose(fp);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    superlu_gridexit(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
    MPI_Finalize();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

}


int cpp_defs()
{
    printf(".. CPP definitions:\n");
#if ( PRNTlevel>=1 )
    printf("\tPRNTlevel = %d\n", PRNTlevel);
#endif
#if ( DEBUGlevel>=1 )
    printf("\tDEBUGlevel = %d\n", DEBUGlevel);
#endif
#if ( PROFlevel>=1 )
    printf("\tPROFlevel = %d\n", PROFlevel);
#endif
#if ( StaticPivot>=1 )
    printf("\tStaticPivot = %d\n", StaticPivot);
#endif
    printf("....\n");
    return 0;
}

// static void load_ddata_markders_vector_txt(double *data,int_t n,int_t ntimestep,int_t index)
// {
//     int_t minstep=500000;
//     int_t i,j,maxstep;
//     char buf[100];
//     char *filename;
//     int slen;
//     FILE *fp;

//     // omp_set_num_threads(8);

//     #pragma omp parallel for private(i,maxstep,slen,filename,buf,fp)
//     for(j=1;j<=n/minstep+1;j++)
//     {
//         maxstep=j*minstep;
//         if(maxstep>=n){
//             maxstep=n;
//         }
//         slen=strlen("/home/412-23/collision_model18/data/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
//         slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
//         slen=slen+(index==0?1:(int)log(index)+1);
//         slen=slen+(j==0?1:(int)log(j)+1);
//         filename=(char*)malloc(slen+1);
//         #if defined (_LONGINT)
//         sprintf(filename,"/home/412-23/collision_model18/data/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
//         #else /* Default */
//         sprintf(filename,"/home/412-23/collision_model18/data/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
//         #endif        

//         if(access(filename,0)){
//             printf("%s do not exist.\n",filename);
//         }
//         else{
//             fp=fopen(filename,"r");
            
//             for(i=(j-1)*minstep+1;i<=maxstep;i++)
//             {
//                 fscanf(fp,"%s",buf);
//                 data[i-1]=atof(buf);
//             }

//             fclose(fp);
//         }
        
//     }

//     // omp_set_num_threads(1);

// }

// static void load_idata_markders_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index)
// {
//     int_t minstep=500000;
//     int_t i,j,maxstep;
//     char buf[100];
//     char *filename;
//     int slen;
//     FILE *fp;

//     // omp_set_num_threads(8);

//     #pragma omp parallel for private(i,maxstep,slen,filename,buf,fp)
//     for(j=1;j<=n/minstep+1;j++)
//     {
//         maxstep=j*minstep;
//         if(maxstep>=n){
//             maxstep=n;
//         }
//         slen=strlen("/home/412-23/collision_model18/data/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
//         slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
//         slen=slen+(index==0?1:(int)log(index)+1);
//         slen=slen+(j==0?1:(int)log(j)+1);
//         filename=(char*)malloc(slen+1);
//         #if defined (_LONGINT)
//         sprintf(filename,"/home/412-23/collision_model18/data/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
//         #else /* Default */
//         sprintf(filename,"/home/412-23/collision_model18/data/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
//         #endif        

//         if(access(filename,0)){
//             printf("%s do not exist.\n",filename);
//         }
//         else{
//             fp=fopen(filename,"r");
            
//             for(i=(j-1)*minstep+1;i<=maxstep;i++)
//             {
//                 fscanf(fp,"%s",buf);
//                 data[i-1]=atoi(buf);
//             }

//             fclose(fp);
//         }
        
//     }

//     // omp_set_num_threads(1);

// }

// static void load_idata_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index)
// {
//     int_t i;
//     char buf[100];
//     char *filename;
//     int slen;
//     FILE *fp;

//     slen=strlen("/home/412-23/collision_model18/data/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
//     slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
//     slen=slen+(index==0?1:(int)log(index)+1);
//     filename=(char*)malloc(slen+1);
//     #if defined (_LONGINT)
//     sprintf(filename,"/home/412-23/collision_model18/data/data-%ld_%ld.txt\0",index,ntimestep);
//     #else /* Default */
//     sprintf(filename,"/home/412-23/collision_model18/data/data-%8d_%8d.txt\0",index,ntimestep);
//     #endif

//     if(access(filename,0)){
//         printf("%s do not exist.\n",filename);
//     }
//     else{
//         fp=fopen(filename,"r");

//         for(i=0;i<n;i++)
//         {
//             fscanf(fp,"%s",buf);
//             data[i]=atoi(buf);
//         }

//         fclose(fp);
//     }

// }

// static void load_ddata_vector_txt(double *data,int_t n,int_t ntimestep,int_t index)
// {
//     int_t i;
//     char buf[100];
//     char *filename;
//     int slen;
//     FILE *fp;

//     slen=strlen("/home/412-23/collision_model18/data/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
//     slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
//     slen=slen+(index==0?1:(int)log(index)+1);
//     filename=(char*)malloc(slen+1);
//     #if defined (_LONGINT)
//     sprintf(filename,"/home/412-23/collision_model18/data/data-%ld_%ld.txt\0",index,ntimestep);
//     #else /* Default */
//     sprintf(filename,"/home/412-23/collision_model18/data/data-%8d_%8d.txt\0",index,ntimestep);
//     #endif

//     if(access(filename,0)){
//         printf("%s do not exist.\n",filename);
//     }
//     else{
//         fp=fopen(filename,"r");

//         for(i=0;i<n;i++)
//         {
//             fscanf(fp,"%s",buf);
//             data[i]=atof(buf);
//         }

//         fclose(fp);
//     }

// }