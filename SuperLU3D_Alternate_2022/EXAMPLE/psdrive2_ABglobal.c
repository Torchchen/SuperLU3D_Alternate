/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Driver program for psgssvx_ABglobal example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * April 5, 2015
 * </pre>
 */

#include <math.h>
#include "superlu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program psdrive2_ABglobal.
 *
 * This example illustrates how to use psgssvx_ABglobal to solve
 * systems repeatedly with the same sparsity pattern of matrix A.
 * In this case, the column permutation vector ScalePermstruct->perm_c is
 * computed once.  The following data structures will be reused in the
 * subsequent call to psgssvx_ABglobal:
 *        ScalePermstruct : perm_c
 *        LUstruct        : etree
 *
 * On an IBM SP, the program may be run by typing:
 *    poe psdrive2_ABglobal -r <proc rows> -c <proc columns> <input_matrix> -procs <p>
 * </pre>
 */

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    gridinfo_t grid;
    float   *berr;
    float   *a, *a1, *b, *b1, *xtrue;
    int_t    *asub, *asub1, *xa, *xa1;
    int_t    i, j, m, n, nnz;
    int_t    nprow, npcol;
    int      iam, info, ldb, ldx, nrhs;
    char     trans[1];
    char     **cpp, c;
    FILE *fp, *fopen();
    extern int cpp_defs();

    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT. 
       ------------------------------------------------------------*/
    MPI_Init( &argc, &argv );

    /* Parse command line argv[]. */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows    (default " IFMT ")\n", nprow);
		  printf("\t-c <int>: process columns (default " IFMT ")\n", npcol);
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

    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if ( iam == -1 )	goto out;
    
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       Process 0 reads the matrix A, and then broadcasts it to all
       the other processes.
       ------------------------------------------------------------*/
    if ( !iam ) {
	/* Print the CPP definitions. */
	cpp_defs();
	
	/* Read the matrix stored on disk in Harwell-Boeing format. */
	sreadhb_dist(iam, fp, &m, &n, &nnz, &a, &asub, &xa);
	
	printf("Input matrix file: %s\n", *cpp);
	printf("\tDimension\t" IFMT "x" IFMT "\t # nonzeros " IFMT "\n", m, n, nnz);
	printf("\tProcess grid\t%d X %d\n", (int) grid.nprow, (int) grid.npcol);

	/* Broadcast matrix A to the other PEs. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( a,    nnz, MPI_FLOAT, 0, grid.comm );
	MPI_Bcast( asub, nnz, mpi_int_t,  0, grid.comm );
	MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid.comm );
    } else {
	/* Receive matrix A from PE 0. */
	MPI_Bcast( &m,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &n,   1,   mpi_int_t,  0, grid.comm );
	MPI_Bcast( &nnz, 1,   mpi_int_t,  0, grid.comm );

	/* Allocate storage for compressed column representation. */
	sallocateA_dist(n, nnz, &a, &asub, &xa);

	MPI_Bcast( a,    nnz, MPI_FLOAT, 0, grid.comm );
	MPI_Bcast( asub, nnz, mpi_int_t,  0, grid.comm );
	MPI_Bcast( xa,   n+1, mpi_int_t,  0, grid.comm );
    }
	
    /* Create compressed column matrix for A. */
    sCreate_CompCol_Matrix_dist(&A, m, n, nnz, a, asub, xa,
				SLU_NC, SLU_S, SLU_GE);

    /* Generate the exact solution and compute the right-hand side. */
    if (!(b=floatMalloc_dist(m * nrhs))) ABORT("Malloc fails for b[]");
    if (!(xtrue=floatMalloc_dist(n*nrhs))) ABORT("Malloc fails for xtrue[]");
    *trans = 'N';
    ldx = n;
    ldb = m;
    sGenXtrue_dist(n, nrhs, xtrue, ldx);
    sFillRHS_dist(trans, nrhs, xtrue, ldx, &A, b, ldb);

    /* Save a copy of the right-hand side. */  
    if ( !(b1 = floatMalloc_dist(m * nrhs)) ) ABORT("Malloc fails for b1[]");
    for (j = 0; j < nrhs; ++j)
	for (i = 0; i < m; ++i) b1[i+j*ldb] = b[i+j*ldb];
    
    if ( !(berr = floatMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* Save a copy of the matrix A. */
    sallocateA_dist(n, nnz, &a1, &asub1, &xa1);
    for (i = 0; i < nnz; ++i) { a1[i] = a[i]; asub1[i] = asub[i]; }
    for (i = 0; i < n+1; ++i) xa1[i] = xa[i];


    /* ------------------------------------------------------------
       WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME.
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact = DOFACT;
        options.Equil = YES;
        options.ColPerm = METIS_AT_PLUS_A;
        options.RowPerm = LargeDiag_MC64;
        options.ReplaceTinyPivot = YES;
        options.Trans = NOTRANS;
        options.IterRefine = DOUBLE;
        options.SolveInitialized = NO;
        options.RefineInitialized = NO;
        options.PrintStat = YES;
     */
    set_default_options_dist(&options);

    if (!iam) {
	print_sp_ienv_dist(&options);
	print_options_dist(&options);
    }

    /* Initialize ScalePermstruct and LUstruct. */
    sScalePermstructInit(m, n, &ScalePermstruct);
    sLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    /* Call the linear equation solver: factorize and solve. */
    psgssvx_ABglobal(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
		     &LUstruct, berr, &stat, &info);

    /* Check the accuracy of the solution. */
    if ( !iam ) {
	sinf_norm_error_dist(n, nrhs, b, ldb, xtrue, ldx, &grid);
    }
    
    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
    PStatFree(&stat);
    Destroy_CompCol_Matrix_dist(&A); /* Deallocate storage of matrix A.     */
    sDestroy_LU(n, &grid, &LUstruct); /* Deallocate storage associated with 
					the L and U matrices.               */
    SUPERLU_FREE(b);                 /* Free storage of right-hand side.    */


    /* ------------------------------------------------------------
       NOW WE SOLVE ANOTHER LINEAR SYSTEM.
       ONLY THE SPARSITY PATTERN OF MATRIX A IS THE SAME.
       ------------------------------------------------------------*/
    options.Fact = SamePattern;
    PStatInit(&stat); /* Initialize the statistics variables. */

    /* Create compressed column matrix for A. */
    sCreate_CompCol_Matrix_dist(&A, m, n, nnz, a1, asub1, xa1,
				SLU_NC, SLU_S, SLU_GE);

    /* Solve the linear system. */
    psgssvx_ABglobal(&options, &A, &ScalePermstruct, b1, ldb, nrhs, &grid,
		     &LUstruct, berr, &stat, &info);

    /* Check the accuracy of the solution. */
    if ( !iam ) {
	printf("Solve the system with the same sparsity pattern.\n");
	sinf_norm_error_dist(n, nrhs, b1, ldb, xtrue, ldx, &grid);
    }

    /* Print the statistics. */
    PStatPrint(&options, &stat, &grid);

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/
    PStatFree(&stat);
    Destroy_CompCol_Matrix_dist(&A); /* Deallocate storage of matrix A.     */
    sDestroy_LU(n, &grid, &LUstruct); /* Deallocate storage associated with    
					the L and U matrices.               */
    sScalePermstructFree(&ScalePermstruct);
    sLUstructFree(&LUstruct);         /* Deallocate the structure of L and U.*/
    SUPERLU_FREE(b1);	             /* Free storage of right-hand side.    */
    SUPERLU_FREE(xtrue);             /* Free storage of the exact solution. */
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
