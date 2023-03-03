/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
#define statistics_time

/*! @file
 * \brief Improves the computed solution and provies error bounds
 *
 * <pre>
 * -- Distributed SuperLU routine (version 4.3) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 *
 * Last modified:
 * December 31, 2015  version 4.3
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"

/*-- Function prototypes --*/
static void gather_1rhs_diag_to_all(int_t, double [], Glu_persist_t *,
                                    dLocalLU_t *, gridinfo_t *, int_t, int_t [],
				    int_t [], double [], double []);
static void redist_all_to_diag(int_t, double [], Glu_persist_t *,
                               dLocalLU_t *, gridinfo_t *, int_t [], double []);

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * pdgsrfs_ABXglobal improves the computed solution to a system of linear
 * equations and provides error bounds and backward error estimates
 * for the solution.
 *
 * Arguments
 * =========
 *
 * n      (input) int (global)
 *        The order of the system of linear equations.
 *
 * A      (input) SuperMatrix*
 *	  The original matrix A, or the scaled A if equilibration was done.
 *        A is also permuted into the form Pc*Pr*A*Pc', where Pr and Pc
 *        are permutation matrices. The type of A can be:
 *        Stype = SLU_NCP; Dtype = SLU_D; Mtype = SLU_GE.
 *
 *        NOTE: Currently, A must reside in all processes when calling
 *              this routine.
 *
 * anorm  (input) double
 *        The norm of the original matrix A, or the scaled A if
 *        equilibration was done.
 *
 * LUstruct (input) dLUstruct_t*
 *        The distributed data structures storing L and U factors.
 *        The L and U factors are obtained from pdgstrf for
 *        the possibly scaled and permuted matrix A.
 *        See superlu_ddefs.h for the definition of 'dLUstruct_t'.
 *
 * grid   (input) gridinfo_t*
 *        The 2D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and my process rank. It is an input argument to all the
 *        parallel routines.
 *        Grid can be initialized by subroutine SUPERLU_GRIDINIT.
 *        See superlu_ddefs.h for the definition of 'gridinfo_t'.
 *
 * B      (input) double* (global)
 *        The N-by-NRHS right-hand side matrix of the possibly equilibrated
 *        and row permuted system.
 *
 *        NOTE: Currently, B must reside on all processes when calling
 *              this routine.
 *
 * ldb    (input) int (global)
 *        Leading dimension of matrix B.
 *
 * X      (input/output) double* (global)
 *        On entry, the solution matrix X, as computed by PDGSTRS.
 *        On exit, the improved solution matrix X.
 *        If DiagScale = COL or BOTH, X should be premultiplied by diag(C)
 *        in order to obtain the solution to the original system.
 *
 *        NOTE: Currently, X must reside on all processes when calling
 *              this routine.
 *
 * ldx    (input) int (global)
 *        Leading dimension of matrix X.
 *
 * nrhs   (input) int
 *        Number of right-hand sides.
 *
 * berr   (output) double*, dimension (nrhs)
 *         The componentwise relative backward error of each solution
 *         vector X(j) (i.e., the smallest relative change in
 *         any element of A or B that makes X(j) an exact solution).
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics about the refinement steps.
 *        See util.h for the definition of SuperLUStat_t.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *
 * Internal Parameters
 * ===================
 *
 * ITMAX is the maximum number of steps of iterative refinement.
 * </pre>
 */

void
pdgsrfs_ABXglobal(int_t n, SuperMatrix *A, double anorm, dLUstruct_t *LUstruct,
		  gridinfo_t *grid, double *B, int_t ldb, double *X, int_t ldx,
		  int nrhs, double *berr, SuperLUStat_t *stat, int *info)
{


#define ITMAX 100
#define PRNTlevel 1

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    /*
     * Data structures used by matrix-vector multiply routine.
     */
    int_t  N_update; /* Number of variables updated on this process */
    int_t  *update;  /* vector elements (global index) updated
			on this processor.                     */
    int_t  *bindx;
    double *val;
    int_t *mv_sup_to_proc;  /* Supernode to process mapping in
			       matrix-vector multiply.  */
    /*-- end data structures for matrix-vector multiply --*/
    double *b, *ax, *R, *B_col, *temp, *work, *X_col,
           *x_trs, *dx_trs;
    int_t count, ii, j, jj, k, knsupc, lk, lwork,
          nprow, nsupers, nz, p;
    int   i, iam, pkk;
    int_t *ilsum, *xsup;
    double eps, lstres;
    double s, safmin, safe1, safe2;

    /* NEW STUFF */
    int_t num_diag_procs, *diag_procs; /* Record diagonal process numbers. */
    int_t *diag_len; /* Length of the X vector on diagonal processes. */

    /*-- Function prototypes --*/
    extern void pdgstrs1(int_t, dLUstruct_t *, gridinfo_t *,
			 double *, int, SuperLUStat_t *, int *);

    /* Test the input parameters. */
    *info = 0;
    if ( n < 0 ) *info = -1;
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
	      A->Stype != SLU_NCP || A->Dtype != SLU_D || A->Mtype != SLU_GE )
	*info = -2;
    else if ( ldb < SUPERLU_MAX(0, n) ) *info = -10;
    else if ( ldx < SUPERLU_MAX(0, n) )	*info = -12;
    else if ( nrhs < 0 ) *info = -13;
    if (*info != 0) {
	i = -(*info);
	pxerr_dist("pdgsrfs_ABXglobal", grid, i);
	return;
    }

    /* Quick return if possible. */
    if ( n == 0 || nrhs == 0 ) {
	return;
    }

    /* Initialization. */
    iam = grid->iam;
    nprow = grid->nprow;
    nsupers = Glu_persist->supno[n-1] + 1;
    xsup = Glu_persist->xsup;
    ilsum = Llu->ilsum;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter pdgsrfs_ABXglobal()");
#endif

    get_diag_procs(n, Glu_persist, grid, &num_diag_procs,
		   &diag_procs, &diag_len);
#if ( PRNTlevel>=1 )
    if ( !iam ) {
	printf(".. number of diag processes = " IFMT "\n", num_diag_procs);
	PrintInt10("diag_procs", num_diag_procs, diag_procs);
	PrintInt10("diag_len", num_diag_procs, diag_len);
    // super_stats_dist(nsupers,xsup);
    }    
#endif

    if ( !(mv_sup_to_proc = intCalloc_dist(nsupers)) )
	ABORT("Calloc fails for mv_sup_to_proc[]");

    pdgsmv_AXglobal_setup(A, Glu_persist, grid, &N_update, &update,
		          &val, &bindx, mv_sup_to_proc);

    i = CEILING( nsupers, nprow ); /* Number of local block rows */
    ii = Llu->ldalsum + i * XK_H;
    k = SUPERLU_MAX(N_update, sp_ienv_dist(3));
    jj = diag_len[0];
    for (j = 1; j < num_diag_procs; ++j) jj = SUPERLU_MAX( jj, diag_len[j] );
    jj = SUPERLU_MAX( jj, N_update );
    lwork = N_update         /* For ax and R */
	  + ii               /* For dx_trs */
	  + ii               /* For x_trs */
          + k                /* For b */
	  + jj;              /* for temp */
    if ( !(work = doubleMalloc_dist(lwork)) )
	ABORT("Malloc fails for work[]");
    ax = R = work;
    dx_trs = work + N_update;
    x_trs  = dx_trs + ii;
    b      = x_trs + ii;
    temp   = b + k;

#if ( DEBUGlevel>=2 )
    {
	double *dwork = doubleMalloc_dist(n);
	for (i = 0; i < n; ++i) {
	    if ( i & 1 ) dwork[i] = 1.;
	    else dwork[i] = 2.;
        }
	/* Check correctness of matrix-vector multiply. */
	pdgsmv_AXglobal(N_update, update, val, bindx, dwork, ax);
	Printdouble5("Mult A*x", N_update, ax);
	SUPERLU_FREE(dwork);
    }
#endif


    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */
    nz     = A->ncol + 1;
    eps    = dmach_dist("Epsilon");
    safmin = dmach_dist("Safe minimum");

    /* Set SAFE1 essentially to be the underflow threshold times the
       number of additions in each row. */
    safe1  = nz * safmin;
    safe2  = safe1 / eps;

#if ( DEBUGlevel>=1 )
    if ( !iam ) printf(".. eps = %e\tanorm = %e\tsafe1 = %e\tsafe2 = %e\n",
		       eps, anorm, safe1, safe2);
#endif

	#ifdef statistics_time
	double times[6],t_tmep;
	for(int iii=0;iii<6;iii++)times[iii]=0;
	#endif

    /* Do for each right-hand side ... */
    for (j = 0; j < nrhs; ++j) {
	count = 0;
	lstres = 3.;

	/* Copy X into x on the diagonal processes. */
	B_col = &B[j*ldb];
	X_col = &X[j*ldx];
	for (p = 0; p < num_diag_procs; ++p) {
	    pkk = diag_procs[p];
	    if ( iam == pkk ) {
		for (k = p; k < nsupers; k += num_diag_procs) {
		    knsupc = SuperSize( k );
		    lk = LBi( k, grid );
		    ii = ilsum[lk] + (lk+1)*XK_H;
		    jj = FstBlockC( k );
		    for (i = 0; i < knsupc; ++i) x_trs[i+ii] = X_col[i+jj];
		    dx_trs[ii-XK_H] = k;/* Block number prepended in header. */
		}
	    }
	}
	/* Copy B into b distributed the same way as matrix-vector product. */
        if ( N_update ) ii = update[0];
	for (i = 0; i < N_update; ++i) b[i] = B_col[i + ii];

	while (1) { /* Loop until stopping criterion is satisfied. */
		
	    /* Compute residual R = B - op(A) * X,
	       where op(A) = A, A**T, or A**H, depending on TRANS. */

		#ifdef statistics_time
		t_tmep=SuperLU_timer_();
		#endif

	    /* Matrix-vector multiply. */
	    pdgsmv_AXglobal(N_update, update, val, bindx, X_col, ax);

		#ifdef statistics_time
		times[0]=times[0]+SuperLU_timer_()-t_tmep;
		#endif

	    /* Compute residual. */
	    for (i = 0; i < N_update; ++i) R[i] = b[i] - ax[i];

		#ifdef statistics_time
		t_tmep=SuperLU_timer_();
		#endif

	    /* Compute abs(op(A))*abs(X) + abs(B). */
	    pdgsmv_AXglobal_abs(N_update, update, val, bindx, X_col, temp);

		#ifdef statistics_time
		times[1]=times[1]+SuperLU_timer_()-t_tmep;
		#endif

	    for (i = 0; i < N_update; ++i) temp[i] += fabs(b[i]);

	    s = 0.0;
	    for (i = 0; i < N_update; ++i) {
		if ( temp[i] > safe2 ) {
		    s = SUPERLU_MAX(s, fabs(R[i]) / temp[i]);
		} else if ( temp[i] != 0.0 ) {
                    /* Adding SAFE1 to the numerator guards against
                       spuriously zero residuals (underflow). */
		    s = SUPERLU_MAX(s, (safe1 + fabs(R[i])) / temp[i]);
                }
                /* If temp[i] is exactly 0.0 (computed by PxGSMV), then
                   we know the true residual also must be exactly 0.0. */
	    }
	    MPI_Allreduce( &s, &berr[j], 1, MPI_DOUBLE, MPI_MAX, grid->comm );

#if ( PRNTlevel>= 1 )
	    if ( !iam )
		printf("(%2d) .. Step " IFMT ": berr[j] = %e\n", iam, count, berr[j]);
#endif
	    if ( /*berr[j] > eps &&*/ (berr[j] * 1.1 <= lstres || berr[j] > eps*100) && count < ITMAX ) {
		/* Compute new dx. */

		#ifdef statistics_time
		t_tmep=SuperLU_timer_();
		#endif

		redist_all_to_diag(n, R, Glu_persist, Llu, grid,
				   mv_sup_to_proc, dx_trs);

		#ifdef statistics_time
		times[2]=times[2]+SuperLU_timer_()-t_tmep;
		#endif

		#ifdef statistics_time
		t_tmep=SuperLU_timer_();
		#endif

		pdgstrs1(n, LUstruct, grid, dx_trs, 1, stat, info);

		#ifdef statistics_time
		times[3]=times[3]+SuperLU_timer_()-t_tmep;
		#endif

		#ifdef statistics_time
		t_tmep=SuperLU_timer_();
		#endif

		/* Update solution. */
		for (p = 0; p < num_diag_procs; ++p)
		    if ( iam == diag_procs[p] )
			for (k = p; k < nsupers; k += num_diag_procs) {
			    lk = LBi( k, grid );
			    ii = ilsum[lk] + (lk+1)*XK_H;
			    knsupc = SuperSize( k );
			    for (i = 0; i < knsupc; ++i)
				x_trs[i + ii] += dx_trs[i + ii];
			}
		lstres = berr[j];
		++count;

		#ifdef statistics_time
		times[4]=times[4]+SuperLU_timer_()-t_tmep;
		#endif

		#ifdef statistics_time
		t_tmep=SuperLU_timer_();
		#endif

		/* Transfer x_trs (on diagonal processes) into X
		   (on all processes). */
		gather_1rhs_diag_to_all(n, x_trs, Glu_persist, Llu, grid,
					num_diag_procs, diag_procs, diag_len,
					X_col, temp);
	    } else {
		break;
	    }

		#ifdef statistics_time
		times[5]=times[5]+SuperLU_timer_()-t_tmep;
		#endif

		if(berr[j] > 0.1 && count > ITMAX/5){
			break;
		}

		if(berr[j] < eps*100 && count > ITMAX/2 && ITMAX>10){
			break;
		}
	} /* end while */

	stat->RefineSteps = count;

    } /* for j ... */

	#ifdef statistics_time
	if(!iam){
		printf("pdgsmv_AXglobal:%f\n",times[0]);
		printf("pdgsmv_AXglobal_abs:%f\n",times[1]);
		printf("redist_all_to_diag:%f\n",times[2]);
		printf("pdgstrs1:%f\n",times[3]);
		printf("Update solution:%f\n",times[4]);
		printf("gather_1rhs_diag_to_all:%f\n",times[5]);
	}
	#endif


    /* Deallocate storage used by matrix-vector multiplication. */
    SUPERLU_FREE(diag_procs);
    SUPERLU_FREE(diag_len);
    if ( N_update ) {
	SUPERLU_FREE(update);
	SUPERLU_FREE(bindx);
	SUPERLU_FREE(val);
    }
    SUPERLU_FREE(mv_sup_to_proc);
    SUPERLU_FREE(work);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit pdgsrfs_ABXglobal()");
#endif

} /* PDGSRFS_ABXGLOBAL */


/*! \brief
 *
 * <pre>
 * r[] is the residual vector distributed the same way as
 * matrix-vector product.
 * </pre>
 */
static void
redist_all_to_diag(int_t n, double r[], Glu_persist_t *Glu_persist,
		   dLocalLU_t *Llu, gridinfo_t *grid, int_t mv_sup_to_proc[],
		   double work[])
{
    int_t i, ii, k, lk, lr, nsupers;
    int_t *ilsum, *xsup;
    int iam, knsupc, psrc, pkk;
    MPI_Status status;

    iam = grid->iam;
    nsupers = Glu_persist->supno[n-1] + 1;
    xsup = Glu_persist->xsup;
    ilsum = Llu->ilsum;
    lr = 0;

    for (k = 0; k < nsupers; ++k) {
	pkk = PNUM( PROW( k, grid ), PCOL( k, grid ), grid );
	psrc = mv_sup_to_proc[k];
	knsupc = SuperSize( k );
	lk = LBi( k, grid );
	ii = ilsum[lk] + (lk+1)*XK_H;
	if ( iam == psrc ) {
	    if ( iam != pkk ) { /* Send X component. */
		MPI_Send( &r[lr], knsupc, MPI_DOUBLE, pkk, Xk,
			 grid->comm );
	    } else { /* Local copy. */
		for (i = 0; i < knsupc; ++i)
		    work[i + ii] = r[i + lr];
	    }
	    lr += knsupc;
	} else {
	    if ( iam == pkk ) { /* Recv X component. */
		MPI_Recv( &work[ii], knsupc, MPI_DOUBLE, psrc, Xk,
			 grid->comm, &status );
	    }
	}
    }
} /* REDIST_ALL_TO_DIAG */


/*! \brief
 *
 * <pre>
 * Gather the components of x vector on the diagonal processes
 * onto all processes, and combine them into the global vector y.
 * </pre>
 */
static void
gather_1rhs_diag_to_all(int_t n, double x[],
			Glu_persist_t *Glu_persist, dLocalLU_t *Llu,
			gridinfo_t *grid, int_t num_diag_procs,
			int_t diag_procs[], int_t diag_len[],
			double y[], double work[])
{
    int_t i, ii, k, lk, lwork, nsupers, p;
    int_t *ilsum, *xsup;
    int iam, knsupc, pkk;

    iam = grid->iam;
    nsupers = Glu_persist->supno[n-1] + 1;
    xsup = Glu_persist->xsup;
    ilsum = Llu->ilsum;

    for (p = 0; p < num_diag_procs; ++p) {
	pkk = diag_procs[p];
	if ( iam == pkk ) {
	    /* Copy x vector into a buffer. */
	    lwork = 0;
	    for (k = p; k < nsupers; k += num_diag_procs) {
		knsupc = SuperSize( k );
		lk = LBi( k, grid );
		ii = ilsum[lk] + (lk+1)*XK_H;
		for (i = 0; i < knsupc; ++i) work[i+lwork] = x[i+ii];
		lwork += knsupc;
	    }
	    MPI_Bcast( work, lwork, MPI_DOUBLE, pkk, grid->comm );
	} else {
	    MPI_Bcast( work, diag_len[p], MPI_DOUBLE, pkk, grid->comm );
	}
	/* Scatter work[] into global y vector. */
	lwork = 0;
	for (k = p; k < nsupers; k += num_diag_procs) {
	    knsupc = SuperSize( k );
	    ii = FstBlockC( k );
	    for (i = 0; i < knsupc; ++i) y[i+ii] = work[i+lwork];
	    lwork += knsupc;
	}
    }
} /* GATHER_1RHS_DIAG_TO_ALL */

