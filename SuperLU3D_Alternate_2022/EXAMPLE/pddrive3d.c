/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Driver program for PDGSSVX3D example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology.
 * May 10, 2019
 *
 */
#include "superlu_ddefs.h" 

#define FORTRAN_TXT
// #define pddrive3d2

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PDDRIVE3D.
 *
 * This example illustrates how to use PDGSSVX3D with the full
 * (default) options to solve a linear system.
 *
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pdgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * The program may be run by typing
 *    mpiexec -np <p> pddrive3d -r <proc rows> -c <proc columns> \
 *                                   -d <proc Z-dimension> <input_file>
 * NOTE: total number of processes p = r * c * d
 *       d must be a power-of-two, e.g., 1, 2, 4, ...
 *
 * </pre>
 */
 
static void matCheck(int n, int m, double* A, int LDA,
       double* B, int LDB)
{
    for(int j=0; j<m;j++)
        for (int i = 0; i < n; ++i) {
	    assert(A[i+ LDA*j] == B[i+ LDB*j]);
	}
    printf("B check passed\n");
    return;
}

static void checkNRFMT(NRformat_loc*A, NRformat_loc*B)
{
    /*
    int_t nnz_loc;
    int_t m_loc;
    int_t fst_row;
    void  *nzval;
    int_t *rowptr;
    int_t *colind;
    */

    assert(A->nnz_loc == B->nnz_loc);
    assert(A->m_loc == B->m_loc);
    assert(A->fst_row == B->fst_row);

#if 0
    double *Aval = (double *)A->nzval, *Bval = (double *)B->nzval;
    PrintDouble5("A", A->nnz_loc, Aval);
    PrintDouble5("B", B->nnz_loc, Bval);
    fflush(stdout);
#endif

    double * Aval = (double *) A->nzval;
    double * Bval = (double *) B->nzval;
    for (int_t i = 0; i < A->nnz_loc; i++)
    {
        assert( Aval[i] == Bval[i] );
        assert((A->colind)[i] == (B->colind)[i]);
	printf("colind[] correct\n");
    }

    for (int_t i = 0; i < A->m_loc + 1; i++)
    {
        assert((A->rowptr)[i] == (B->rowptr)[i]);
    }

    printf("Matrix check passed\n");

}

#if 1
int
main (int argc, char *argv[])
{
    double norm2(int_t n,double *vec);
    void solution_error(int iam, int_t m, int_t n, int_t nrhs, double b[], int *info, SuperMatrix GA, gridinfo3d_t grid, int ntimestep);

    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;  // Now, A is on all 3D processes 
    SuperMatrix GA; /* global A */ 
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dSOLVEstruct_t SOLVEstruct;
    gridinfo3d_t grid;
    double *berr;
    double *b, *xtrue;
    int_t m, n;
    int nprow, npcol, npdep;
    int iam, info, ldb, ldx, nrhs;
    char **cpp, c, *suffix;
    FILE *fp, *fopen ();
    extern int cpp_defs ();
    int ii, omp_mpi_level;
    int_t ntimestep;

    nprow = 1;            /* Default process rows.      */
    npcol = 1;            /* Default process columns.   */
    npdep = 1;            /* replication factor must be power of two */
    nrhs = 1;             /* Number of right-hand side. */

    #ifdef SOLVE_2
    // if(!isexist_idata_vector_txt(0,1008)){
    //     return 0;
    // }
    #endif

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------ */
    MPI_Init (&argc, &argv);
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    // MPI_Init_thread(&argc, &argv, required, &provided);
    // if (provided < required)
    // {
    //     int rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     printf("%d", rank);
    //     if (!rank) printf("The MPI library doesn't provide MPI_THREAD_MULTIPLE \n");
    // }

    /* Parse command line argv[]. */
    for (cpp = argv + 1; *cpp; ++cpp)
    {
        if (**cpp == '-')
        {
            c = *(*cpp + 1);
            ++cpp;
            switch (c)
            {
            case 'h':
                printf ("Options:\n");
                printf ("\t-r <int>: process rows    (default %d)\n", nprow);
                printf ("\t-c <int>: process columns (default %d)\n", npcol);
                printf ("\t-d <int>: process Z-dimension (default %d)\n", npdep);
                exit (0);
                break;
            case 'r':
                nprow = atoi (*cpp);
                break;
            case 'c':
                npcol = atoi (*cpp);
                break;
            case 'd':
                npdep = atoi (*cpp);
                break;
            }
        }
        else
        {   /* Last arg is considered a filename */
            if (!(fp = fopen (*cpp, "r")))
            {
                ABORT ("File does not exist");
            }
            break;
        }
    }

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
    #ifdef Torch
	#ifdef SuperLargeScale
    grid.nSave = 0;
    #endif
    #endif
    superlu_gridinit3d (MPI_COMM_WORLD, nprow, npcol, npdep, &grid);

    if(grid.iam==0) {
	MPI_Query_thread(&omp_mpi_level);
	switch (omp_mpi_level) {
	case MPI_THREAD_SINGLE:
	    printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
	    fflush(stdout);
	    break;
	case MPI_THREAD_FUNNELED:
	    printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
	    fflush(stdout);
	    break;
	case MPI_THREAD_SERIALIZED:
	    printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
	    fflush(stdout);
	    break;
	case MPI_THREAD_MULTIPLE:
	    printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
	    fflush(stdout);
	    break;
	}
    }
	
    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if (iam == -1)     goto out;
    if (!iam) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	printf("Input matrix file:\t%s\n", *cpp);
	printf("3D process grid: %d X %d X %d\n", nprow, npcol, npdep);
	//printf("2D Process grid: %d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE.
       ------------------------------------------------------------ */
    for (ii = 0; ii<strlen(*cpp); ii++) {
	if((*cpp)[ii]=='.'){
	    suffix = &((*cpp)[ii+1]);
	    // printf("%s\n", suffix);
	}
    }

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
	MPI_Get_processor_name(processor_name,&namelen);
    printf("%d, %s, %d, %d, %d\n",grid.iam, processor_name, grid.zscp.Iam, grid.rscp.Iam, grid.cscp.Iam);

#define NRFRMT
#ifndef NRFRMT
    if ( grid.zscp.Iam == 0 )  // only in process layer 0
	dcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, suffix, &(grid.grid2d));
	
#else
    // *fp0 = *fp;
    dcreate_matrix_postfix3d2(&A, &GA, nrhs, &b, &ldb, &ldx, suffix, &(grid), &ntimestep);
    //printf("ldx %d, ldb %d\n", ldx, ldb);
    
#if 0  // following code is only for checking *Gather* routine
    NRformat_loc *Astore, *Astore0;
    double* B2d;
    NRformat_loc Atmp = dGatherNRformat_loc(
                            (NRformat_loc *) A.Store,
                            b, ldb, nrhs, &B2d,
                            &grid);
    Astore = &Atmp;
    SuperMatrix Aref;
    double *bref, *xtrueref;
    if ( grid.zscp.Iam == 0 )  // only in process layer 0
    {
        dcreate_matrix_postfix(&Aref, nrhs, &bref, &ldb,
                               &xtrueref, &ldx, fp0, 
                               suffix, &(grid.grid2d));
        Astore0 = (NRformat_loc *) Aref.Store;

	/*
	if ( (grid.grid2d).iam == 0 ) {
	    printf(" iam %d\n", 0); 
	    checkNRFMT(Astore, Astore0);
	} else if ((grid.grid2d).iam == 1 ) {
	    printf(" iam %d\n", 1); 
	    checkNRFMT(Astore, Astore0);
	} 
	*/
    
	// bref, xtrueref are created on 2D
        matCheck(Astore->m_loc, nrhs, B2d, Astore->m_loc, bref, ldb);
    }
    // MPI_Finalize(); exit(0);
    #endif
#endif

    if (!(berr = doubleMalloc_dist (nrhs)))
        ABORT ("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------ */

    /* Set the default input options:
       options.Fact              = DOFACT;
       options.Equil             = YES;
       options.ParSymbFact       = NO;
       options.ColPerm           = METIS_AT_PLUS_A;
       options.RowPerm           = LargeDiag_MC64;
       options.ReplaceTinyPivot  = YES;
       options.IterRefine        = DOUBLE;
       options.Trans             = NOTRANS;
       options.SolveInitialized  = NO;
       options.RefineInitialized = NO;
       options.PrintStat         = YES;
       options->num_lookaheads    = 10;
       options->lookahead_etree   = NO;
       options->SymPattern        = NO;
       options.DiagInv           = NO;
     */
    set_default_options_dist (&options);

    #ifdef Torch_ReplaceTinyPivot
    options.ReplaceTinyPivot = YES;    
    #endif

    #ifdef Torch_SymPattern
    options.SymPattern = YES;
    #endif

    #ifdef Torch_0320
    // options.Fact = SamePattern;
    #endif
#if 0
    options.RowPerm = NOROWPERM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO;
    options.ReplaceTinyPivot = NO;
#endif

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

#ifdef NRFRMT  // matrix is on 3D process grid
    m = A.nrow;
    n = A.ncol;
#else
    if ( grid.zscp.Iam == 0 )  // Process layer 0
    {
	m = A.nrow;
        n = A.ncol;
    }
    // broadcast m, n to all the process layers;
    MPI_Bcast( &m, 1, mpi_int_t, 0,  grid.zscp.comm);
    MPI_Bcast( &n, 1, mpi_int_t, 0,  grid.zscp.comm);
#endif    

    /* Initialize ScalePermstruct and LUstruct. */
    dScalePermstructInit (m, n, &ScalePermstruct);
    dLUstructInit (n, &LUstruct);
    
    // MPI_Request *send_req;
    // if ( !(send_req = (MPI_Request *)
	//    SUPERLU_MALLOC(sizeof(MPI_Request))))
    //     ABORT("Malloc fails for send_req[].");
    // int_t **senddata = (int_t**)SUPERLU_MALLOC(200 * sizeof(int_t*));
    // int tag = 0;
    // MPI_Status status;
    // if (!grid.iam)
    // {       
    //     for (int_t j = 0; j < 200; j++)
    //     {
    //         senddata[j] = intCalloc_dist(200);
    //         for (int_t i = 0; i < 200; i++)
    //         {
    //             senddata[j][i] = i;
    //         }
    //         MPI_Isend(senddata[j], 200, mpi_int_t, 1, tag + j, grid.comm, send_req);
    //     }   
        
         
    // }
    // if (grid.iam == 1)
    // {       

    //     for (int_t j = 0; j < 200; j++)
    //     {
    //         senddata[j] = intCalloc_dist(200);
    //         MPI_Irecv(senddata[j], 200, mpi_int_t, 0, tag + j, grid.comm, send_req);
    //         MPI_Wait(send_req, &status);
    //     }

    //     PrintInt10("recv", 200, senddata[150]);
    // }
    
    

    #ifdef Torch_0320
    int_t *perm_c0 = intMalloc_dist(n);
    load_idata_vector_binary_c(perm_c0, n, 0, perm_c_index + grid.iam);
    if (grid.iam == 0)
    {
       PrintInt10("perm_c0", n, perm_c0);
    }
    
    if (grid.zscp.Iam == 0 && options.Fact == SamePattern)
    {
        load_idata_vector_binary_c(ScalePermstruct.perm_c, n, 0, perm_c_index + grid.iam);
    }    
    #endif

    #ifdef Torch
	#ifdef SuperLargeScale
    // if(iam == 0 /*||iam == 1 ||iam == 2 ||iam == 3 ||iam == 4 ||iam == 5 || iam == 6 */|| iam == 7){
    //     LUstruct.Llu->isSave=1;        
    // }
    // if(iam >= MAXNODE){
    //     LUstruct.Llu->isSave=1;
    // }
    LUstruct.Llu->ncol = n;
    #endif

    #ifdef SuperLargeScaleGPU
    // if(iam == 7 /*|| iam == 1 || iam == 2 || iam == 3*/){
    //     LUstruct.Llu->isRecordingForGPU = RecordingForGPU;
    //     LUstruct.Llu->isRecordingForGPU = LoadedRecordForGPU;
    // }    
    LUstruct.Llu->LimitGPUMemory = GPUMemLimit;
    #endif    
    #endif

    /* Initialize the statistics variables. */
    PStatInit (&stat);

    /* Call the linear equation solver. */    
    
    pdgssvx3d (&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
               &LUstruct, &SOLVEstruct, berr, &stat, &info);

    if(!iam){
        printf("info=%d\n",info);
    }

    /* Check the accuracy of the solution. */
    // pdinf_norm_error (iam, ((NRformat_loc *) A.Store)->m_loc,
    //                       nrhs, b, ldb, xtrue, ldx, grid.comm);
    // fflush(stdout);

    MPI_Barrier(grid.comm);

    if(info==0){
        if ( grid.zscp.Iam == 0 ){
            PStatPrint (&options, &stat, &(grid.grid2d));
        }        
        
        solution_error(iam, m, n, nrhs, b, &info, GA, grid, ntimestep);
    } 
    else
    {
        if (!grid.iam)
        {            
            #ifdef SOLVE_1
            double error = (double)info;
            delete_idata_vector_txt(0,1007);
            save_idata_vector_binary(&info,1,0,1009);
            save_ddata_vector_txt(&error,1,0,1200);
            #else
            double error = (double)info;
            delete_idata_vector_txt(0,1007);
            save_idata_vector_binary(&info,1,0,1009);
            save_ddata_vector_txt(&error,1,0,1200);
            #endif
        }
        return 0;        
    }
    

    MPI_Barrier(grid.comm); 

    #ifdef pddrive3d2

    while (1)
    {
        
        

    if ( grid.zscp.Iam == 0 ) { // process layer 0

	PStatPrint (&options, &stat, &(grid.grid2d)); /* Print 2D statistics.*/

        dDestroy_LU (n, &(grid.grid2d), &LUstruct);
        if (options.SolveInitialized) {
            dSolveFinalize (&options, &SOLVEstruct);
        }
    } else { // Process layers not equal 0
        dDeAllocLlu_3d(n, &LUstruct, &grid);
        dDeAllocGlu_3d(&LUstruct);
    }

    Destroy_CompRowLoc_Matrix_dist (&A);    
    SUPERLU_FREE (b);    
    PStatFree (&stat);

    MPI_Barrier(grid.comm); 

    /* ------------------------------------------------------------
       NOW WE SOLVE ANOTHER LINEAR SYSTEM.
       ONLY THE SPARSITY PATTERN OF MATRIX A IS THE SAME.
       ------------------------------------------------------------*/

    options.Fact = SamePattern;

    if (iam==0) {
	print_options_dist(&options);
#if ( PRNTlevel>=2 )
	PrintInt10("perm_r", m, ScalePermstruct.perm_r);
	PrintInt10("perm_c", n, ScalePermstruct.perm_c);
#endif
    }

    /* Get the matrix from file, perturbed some diagonal entries to force
       a different perm_r[]. Set up the right-hand side.   */
    if ( !(fp = fopen(*cpp, "r")) ) ABORT("File does not exist");
    dcreate_matrix_postfix3d3(&A, &GA, nrhs, &b, &ldb, &ldx, suffix, &(grid), &ntimestep);

    PStatInit(&stat); /* Initialize the statistics variables. */

     /* Solve the linear system. */
    pdgssvx3d (&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
               &LUstruct, &SOLVEstruct, berr, &stat, &info);
 
    if(info==0){
        solution_error(iam, m, n, nrhs, b, &info, GA, grid, ntimestep);
    }  

    }







    #endif

    #ifdef pddrive3d3

    while (1)
    {
        
        

    if ( grid.zscp.Iam == 0 ) { // process layer 0

	PStatPrint (&options, &stat, &(grid.grid2d)); /* Print 2D statistics.*/

        dDestroy_LU (n, &(grid.grid2d), &LUstruct);
        if (options.SolveInitialized) {
            dSolveFinalize (&options, &SOLVEstruct);
        }
    } else { // Process layers not equal 0
        dDeAllocLlu_3d(n, &LUstruct, &grid);
        dDeAllocGlu_3d(&LUstruct);
    }

    Destroy_CompRowLoc_Matrix_dist (&A);    
    SUPERLU_FREE (b);
    SUPERLU_FREE(x_global);
    SUPERLU_FREE(tmp);
    PStatFree (&stat);

    dScalePermstructFree (&ScalePermstruct);
    dLUstructFree (&LUstruct);
    Destroy_CompCol_Matrix_dist(&GA);

    MPI_Barrier(grid.comm); 

    /* ------------------------------------------------------------
       NOW WE SOLVE ANOTHER LINEAR SYSTEM.
       ONLY THE SPARSITY PATTERN OF MATRIX A IS THE SAME.
       ------------------------------------------------------------*/

    dcreate_matrix_postfix3d2(&A, &GA, nrhs, &b, &ldb, &ldx, suffix, &(grid), &ntimestep);

    set_default_options_dist (&options);

    if (iam==0) {
	print_options_dist(&options);
#if ( PRNTlevel>=2 )
	PrintInt10("perm_r", m, ScalePermstruct.perm_r);
	PrintInt10("perm_c", n, ScalePermstruct.perm_c);
#endif
    }

    /* Initialize ScalePermstruct and LUstruct. */
    dScalePermstructInit (m, n, &ScalePermstruct);
    dLUstructInit (n, &LUstruct);

    /* Get the matrix from file, perturbed some diagonal entries to force
       a different perm_r[]. Set up the right-hand side.   */
    if ( !(fp = fopen(*cpp, "r")) ) ABORT("File does not exist");    

    PStatInit(&stat); /* Initialize the statistics variables. */

     /* Solve the linear system. */
    pdgssvx3d (&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
               &LUstruct, &SOLVEstruct, berr, &stat, &info);

    nrow=m;
    ncol=n;
    Astore = (NCformat *) GA.Store;
    nnz=Astore->nnz;
    x_global=doubleMalloc_dist(nrow);
    tmp=doubleMalloc_dist(nrow);

    MPI_Barrier(grid.comm);

    for(i=0;i<nrow;i++)
    {
        x_global[i]=0.0;
    }

    MPI_Barrier(grid.comm);

    /* Compute the number of rows to be distributed to local process */
    m_loc = nrow / (grid.nprow * grid.npcol * grid.npdep); 
    m_loc_fst = m_loc;
    /* When nrow / procs is not an integer */
    if ((m_loc * grid.nprow * grid.npcol * grid.npdep) != nrow) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
        if (iam == (grid.nprow * grid.npcol * grid.npdep - 1)) /* last proc. gets all*/
	        m_loc = nrow - m_loc * (grid.nprow * grid.npcol * grid.npdep - 1);
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

    if(iam){
        for(j=0;j<nrow;j++)
        {
            tmp[j]=x_global[j];
        }
        MPI_Send(tmp,nrow,MPI_DOUBLE,0,0,grid.comm);
    }
    else{
        for(i=1;i<nprow*npcol*npdep;i++)
        {
            MPI_Recv(tmp,nrow,MPI_DOUBLE,i,0,grid.comm,&status);
            for(j=0;j<nrow;j++)
            {
                x_global[j]+=tmp[j];
            }
        }
    }

    MPI_Barrier(grid.comm);

    if(!iam){

        if(nrow>0 && nnz>0 && ntimestep>0){  

            double *r=doubleMalloc_dist(nrow);
            load_ddata_vector_txt(r,nrow,ntimestep,1003);

            double alpha,beta;
            alpha=1;
            beta=-1;
            int_t incx=1;
            int_t incy=1;
            char trans='N';
            double rnrm=norm2(ncol,r);
            sp_dgemv_dist(&trans, alpha, &GA, x_global, 1, beta, r, 1);
            double dnrm=norm2(ncol,r);
            double res=dnrm/rnrm;
            printf("res=%e,%e,%e\n",res,dnrm,rnrm);
            if(res<1e-6){
                save_ddata_vector_txt(x_global,ncol,ntimestep,1200);
            }

            SUPERLU_FREE(r);
            
        }
    }

    MPI_Barrier(grid.comm);

    }

    #endif
    

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------ */

    if ( grid.zscp.Iam == 0 ) { // process layer 0

	PStatPrint (&options, &stat, &(grid.grid2d)); /* Print 2D statistics.*/

        dDestroy_LU (n, &(grid.grid2d), &LUstruct);
        if (options.SolveInitialized) {
            dSolveFinalize (&options, &SOLVEstruct);
        }
    } else { // Process layers not equal 0
        dDeAllocLlu_3d(n, &LUstruct, &grid);
        dDeAllocGlu_3d(&LUstruct);
    }
    

    Destroy_CompRowLoc_Matrix_dist (&A);
    Destroy_CompCol_Matrix_dist(&GA);
    SUPERLU_FREE (b);
    SUPERLU_FREE (berr);
    dScalePermstructFree (&ScalePermstruct);
    dLUstructFree (&LUstruct);
    PStatFree (&stat);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
out:
    superlu_gridexit3d (&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------ */
    MPI_Finalize ();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit main()");
#endif

}

#else

int
main (int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;  // Now, A is on all 3D processes  
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dSOLVEstruct_t SOLVEstruct;
    gridinfo3d_t grid;
    double *berr;
    double *b, *xtrue;
    int_t m, n;
    int nprow, npcol, npdep;
    int iam, info, ldb, ldx, nrhs;
    char **cpp, c, *suffix;
    FILE *fp, *fopen ();
    extern int cpp_defs ();
    int ii, omp_mpi_level;

    nprow = 1;            /* Default process rows.      */
    npcol = 1;            /* Default process columns.   */
    npdep = 1;            /* replication factor must be power of two */
    nrhs = 1;             /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------ */
    // MPI_Init (&argc, &argv);
    MPI_Init (&argc, &argv);
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    // MPI_Init_thread(&argc, &argv, required, &provided);
    // if (provided < required)
    // {
    //     int rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     if (!rank) printf("The MPI library doesn't provide MPI_THREAD_MULTIPLE \n");
    // }

    /* Parse command line argv[]. */
    for (cpp = argv + 1; *cpp; ++cpp)
    {
        if (**cpp == '-')
        {
            c = *(*cpp + 1);
            ++cpp;
            switch (c)
            {
            case 'h':
                printf ("Options:\n");
                printf ("\t-r <int>: process rows    (default %d)\n", nprow);
                printf ("\t-c <int>: process columns (default %d)\n", npcol);
                printf ("\t-d <int>: process Z-dimension (default %d)\n", npdep);
                exit (0);
                break;
            case 'r':
                nprow = atoi (*cpp);
                break;
            case 'c':
                npcol = atoi (*cpp);
                break;
            case 'd':
                npdep = atoi (*cpp);
                break;
            }
        }
        else
        {   /* Last arg is considered a filename */
            if (!(fp = fopen (*cpp, "r")))
            {
                ABORT ("File does not exist");
            }
            break;
        }
    }

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
    superlu_gridinit3d (MPI_COMM_WORLD, nprow, npcol, npdep, &grid);

    if(grid.iam==0) {
	MPI_Query_thread(&omp_mpi_level);
	switch (omp_mpi_level) {
	case MPI_THREAD_SINGLE:
	    printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
	    fflush(stdout);
	    break;
	case MPI_THREAD_FUNNELED:
	    printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
	    fflush(stdout);
	    break;
	case MPI_THREAD_SERIALIZED:
	    printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
	    fflush(stdout);
	    break;
	case MPI_THREAD_MULTIPLE:
	    printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
	    fflush(stdout);
	    break;
	}
    }
	
    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if (iam == -1)     goto out;
    if (!iam) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	printf("Input matrix file:\t%s\n", *cpp);
	printf("3D process grid: %d X %d X %d\n", nprow, npcol, npdep);
	//printf("2D Process grid: %d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE.
       ------------------------------------------------------------ */
    for (ii = 0; ii<strlen(*cpp); ii++) {
	if((*cpp)[ii]=='.'){
	    suffix = &((*cpp)[ii+1]);
	    // printf("%s\n", suffix);
	}
    }

#define NRFRMT
#ifndef NRFRMT
    if ( grid.zscp.Iam == 0 )  // only in process layer 0
	dcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, suffix, &(grid.grid2d));
	
#else
    // *fp0 = *fp;
    dcreate_matrix_postfix3d(&A, nrhs, &b, &ldb,
                             &xtrue, &ldx, fp, suffix, &(grid));
    //printf("ldx %d, ldb %d\n", ldx, ldb);
    
#if 0  // following code is only for checking *Gather* routine
    NRformat_loc *Astore, *Astore0;
    double* B2d;
    NRformat_loc Atmp = dGatherNRformat_loc(
                            (NRformat_loc *) A.Store,
                            b, ldb, nrhs, &B2d,
                            &grid);
    Astore = &Atmp;
    SuperMatrix Aref;
    double *bref, *xtrueref;
    if ( grid.zscp.Iam == 0 )  // only in process layer 0
    {
        dcreate_matrix_postfix(&Aref, nrhs, &bref, &ldb,
                               &xtrueref, &ldx, fp0, 
                               suffix, &(grid.grid2d));
        Astore0 = (NRformat_loc *) Aref.Store;

	/*
	if ( (grid.grid2d).iam == 0 ) {
	    printf(" iam %d\n", 0); 
	    checkNRFMT(Astore, Astore0);
	} else if ((grid.grid2d).iam == 1 ) {
	    printf(" iam %d\n", 1); 
	    checkNRFMT(Astore, Astore0);
	} 
	*/
    
	// bref, xtrueref are created on 2D
        matCheck(Astore->m_loc, nrhs, B2d, Astore->m_loc, bref, ldb);
    }
    // MPI_Finalize(); exit(0);
    #endif
#endif

    if (!(berr = doubleMalloc_dist (nrhs)))
        ABORT ("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------ */

    /* Set the default input options:
       options.Fact              = DOFACT;
       options.Equil             = YES;
       options.ParSymbFact       = NO;
       options.ColPerm           = METIS_AT_PLUS_A;
       options.RowPerm           = LargeDiag_MC64;
       options.ReplaceTinyPivot  = YES;
       options.IterRefine        = DOUBLE;
       options.Trans             = NOTRANS;
       options.SolveInitialized  = NO;
       options.RefineInitialized = NO;
       options.PrintStat         = YES;
       options->num_lookaheads    = 10;
       options->lookahead_etree   = NO;
       options->SymPattern        = NO;
       options.DiagInv           = NO;
     */
    set_default_options_dist (&options);
#if 0
    options.RowPerm = NOROWPERM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO;
    options.ReplaceTinyPivot = NO;
#endif

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

#ifdef NRFRMT  // matrix is on 3D process grid
    m = A.nrow;
    n = A.ncol;
#else
    if ( grid.zscp.Iam == 0 )  // Process layer 0
    {
	m = A.nrow;
        n = A.ncol;
    }
    // broadcast m, n to all the process layers;
    MPI_Bcast( &m, 1, mpi_int_t, 0,  grid.zscp.comm);
    MPI_Bcast( &n, 1, mpi_int_t, 0,  grid.zscp.comm);
#endif    

    /* Initialize ScalePermstruct and LUstruct. */
    dScalePermstructInit (m, n, &ScalePermstruct);
    dLUstructInit (n, &LUstruct);

    #ifdef Torch
	#ifdef SuperLargeScale
    if(iam>=MAXNODE){
        LUstruct.Llu->isSave=1;
    }
    #endif
    #ifdef SuperLargeScaleGPU
    // if(iam == 0 || iam == 1 || iam == 2 || iam == 3){
    //     LUstruct.Llu->isRecordingForGPU = RecordingForGPU;
    //     LUstruct.Llu->isRecordingForGPU = LoadedRecordForGPU;
    // }    
    LUstruct.Llu->LimitGPUMemory = GPUMemLimit;
    #endif
    #endif

    /* Initialize the statistics variables. */
    PStatInit (&stat);

    /* Call the linear equation solver. */
    pdgssvx3d (&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
               &LUstruct, &SOLVEstruct, berr, &stat, &info);

    /* Check the accuracy of the solution. */
    pdinf_norm_error (iam, ((NRformat_loc *) A.Store)->m_loc,
                          nrhs, b, ldb, xtrue, ldx, grid.comm);
    fflush(stdout);    

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------ */

    if ( grid.zscp.Iam == 0 ) { // process layer 0

	PStatPrint (&options, &stat, &(grid.grid2d)); /* Print 2D statistics.*/

        while (1)
        {
            /* code */
        }
        dDestroy_LU (n, &(grid.grid2d), &LUstruct);
        if (options.SolveInitialized) {
            dSolveFinalize (&options, &SOLVEstruct);
        }
    } else { // Process layers not equal 0
        while (1)
        {
            /* code */
        }
        dDeAllocLlu_3d(n, &LUstruct, &grid);
        dDeAllocGlu_3d(&LUstruct);
    }

    while (1)
    {
        /* code */
    }
    

    Destroy_CompRowLoc_Matrix_dist (&A);
    SUPERLU_FREE (b);
    SUPERLU_FREE (xtrue);
    SUPERLU_FREE (berr);
    dScalePermstructFree (&ScalePermstruct);
    dLUstructFree (&LUstruct);
    PStatFree (&stat);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------ */
out:
    superlu_gridexit3d (&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------ */
    MPI_Finalize ();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit main()");
#endif

}

#endif

int
cpp_defs ()
{
    printf (".. CPP definitions:\n");
#if ( PRNTlevel>=1 )
    printf ("\tPRNTlevel = %d\n", PRNTlevel);
#endif
#if ( DEBUGlevel>=1 )
    printf ("\tDEBUGlevel = %d\n", DEBUGlevel);
#endif
#if ( PROFlevel>=1 )
    printf ("\tPROFlevel = %d\n", PROFlevel);
#endif
    printf ("....\n");
    return 0;
}

double norm2(int_t n,double *vec)
{
    double normal2=0;;
    for(int_t i=0;i<n;i++)
    {
        normal2 += (vec[i] * vec[i]);
    }

    return sqrt(normal2);
}

#if 1
void solution_error(int iam, int_t m, int_t n, int_t nrhs, double b[], int *info, SuperMatrix GA, gridinfo3d_t grid, int ntimestep)
{
    // get x_global
    double *x_global;
    int_t nrow,ncol,nnz;
    nrow=m;
    ncol=n;
    NCformat *Astore;
    Astore = (NCformat *) GA.Store;
    nnz=Astore->nnz;
    x_global=doubleMalloc_dist(nrow);
    int_t m_loc, fst_row, nnz_loc,m_loc_fst,i,j;
    double *tmp;
    tmp=doubleMalloc_dist(nrow);

    MPI_Barrier(grid.comm);

    for(i=0;i<nrow;i++)
    {
        x_global[i]=0.0;
    }

    MPI_Barrier(grid.comm);

    /* Compute the number of rows to be distributed to local process */
    m_loc = nrow / (grid.nprow * grid.npcol * grid.npdep); 
    m_loc_fst = m_loc;
    /* When nrow / procs is not an integer */
    if ((m_loc * grid.nprow * grid.npcol * grid.npdep) != nrow) {
        /*m_loc = m_loc+1;
          m_loc_fst = m_loc;*/
        if (iam == (grid.nprow * grid.npcol * grid.npdep - 1)) /* last proc. gets all*/
	        m_loc = nrow - m_loc * (grid.nprow * grid.npcol * grid.npdep - 1);
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
    MPI_Status status;

    if(iam){
        for(j=0;j<nrow;j++)
        {
            tmp[j]=x_global[j];
        }
        MPI_Send(tmp,nrow,MPI_DOUBLE,0,0,grid.comm);
    }
    else{
        for(i=1;i<grid.nprow*grid.npcol*grid.npdep;i++)
        {
            MPI_Recv(tmp,nrow,MPI_DOUBLE,i,0,grid.comm,&status);
            for(j=0;j<nrow;j++)
            {
                x_global[j]+=tmp[j];
            }
        }
    }

    MPI_Barrier(grid.comm);

    if(!iam){

        if(nrow>0 && nnz>0 && ntimestep>0){  

            for ( i = 0; i < ncol; i++)
            {
                if (isnan(x_global[i]) || isinf(x_global[i] == 0))
                {
                    printf("x is nan or inf.\n");
                    break;
                }
                
            }

            double *r=doubleMalloc_dist(nrow);
            #ifdef FILE_BINARY
            load_ddata_vector_binary(r,nrow,ntimestep,1003);
            #else
            load_ddata_vector_txt(r,nrow,ntimestep,1003);
            #endif

            double alpha,beta;
            alpha=1;
            beta=-1;
            int_t incx=1;
            int_t incy=1;
            char trans[1];
            *trans ='N';
            double rnrm=norm2(ncol,r);
            sp_dgemv_dist(trans, alpha, &GA, x_global, 1, beta, r, 1);            
            double dnrm=norm2(ncol,r);
            double res = dnrm/rnrm;          

            printf("res=%e,%e,%e\n",res,dnrm,rnrm);
            if(res<1e-6){
                #ifndef Torch_stat
                #ifdef FILE_BINARY
                save_ddata_vector_binary(x_global,ncol,ntimestep,1006);                
                #else
                save_ddata_vector_txt(x_global,ncol,ntimestep,1006);
                #endif
                save_ddata_vector_txt(x_global,1,ntimestep,1005);
                *info=0;
                delete_idata_vector_txt(0,1007);
                #endif
            }
            else{
                *info=-1;
            }

            SUPERLU_FREE(r);
            
        }
    }

    SUPERLU_FREE(x_global);
    SUPERLU_FREE(tmp);
}

#endif

