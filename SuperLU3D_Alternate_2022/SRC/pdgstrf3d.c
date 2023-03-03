/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file
 * \brief Performs LU factorization in 3D process grid.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology.
 * May 10, 2019
 */

#include "superlu_ddefs.h"
#if 0
#include "pdgstrf3d.h"
#include "trfCommWrapper.h"
#include "trfAux.h"
//#include "load-balance/supernodal_etree.h"
//#include "load-balance/supernodalForest.h"
#include "supernodal_etree.h"
#include "supernodalForest.h"
#include "p3dcomm.h"
#include "treeFactorization.h"
#include "ancFactorization.h"
#include "xtrf3Dpartition.h"
#endif

#ifdef MAP_PROFILE
#include  "mapsampler_api.h"
#endif

// #undef GPU_ACC

#ifdef GPU_ACC
#include "dlustruct_gpu.h"
//#include "acc_aux.c"  //no need anymore
#endif

// #define test

// #define IEXCHANGE

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * PDGSTRF3D performs the LU factorization in parallel using 3D process grid,
 * which is a communication-avoiding algorithm compared to the 2D algorithm.
 *
 * Arguments
 * =========
 *
 * options (input) superlu_dist_options_t*
 *         The structure defines the input parameters to control
 *         how the LU decomposition will be performed.
 *         The following field should be defined:
 *         o ReplaceTinyPivot (yes_no_t)
 *           Specifies whether to replace the tiny diagonals by
 *           sqrt(epsilon)*norm(A) during LU factorization.
 *
 * m      (input) int
 *        Number of rows in the matrix.
 *
 * n      (input) int
 *        Number of columns in the matrix.
 *
 * anorm  (input) double
 *        The norm of the original matrix A, or the scaled A if
 *        equilibration was done.
 *
 * trf3Dpartition (input) trf3Dpartition*
 *        Matrix partitioning information in 3D process grid.
 *
 * SCT    (input/output) SCT_t*
 *        Various statistics of 3D factorization.
 *
 * LUstruct (input/output) dLUstruct_t*
 *         The data structures to store the distributed L and U factors.
 *         The following fields should be defined:
 *
 *         o Glu_persist (input) Glu_persist_t*
 *           Global data structure (xsup, supno) replicated on all processes,
 *           describing the supernode partition in the factored matrices
 *           L and U:
 *         xsup[s] is the leading column of the s-th supernode,
 *             supno[i] is the supernode number to which column i belongs.
 *
 *         o Llu (input/output) dLocalLU_t*
 *           The distributed data structures to store L and U factors.
 *           See superlu_ddefs.h for the definition of 'dLocalLU_t'.
 *
 * grid3d (input) gridinfo3d_t*
 *        The 3D process mesh. It contains the MPI communicator, the number
 *        of process rows (NPROW), the number of process columns (NPCOL),
 *        and replication factor in Z-dimension. It is an input argument to all
 *        the 3D parallel routines.
 *        Grid3d can be initialized by subroutine SUPERLU_GRIDINIT3D.
 *        See superlu_defs.h for the definition of 'gridinfo3d_t'.
 *
 * stat   (output) SuperLUStat_t*
 *        Record the statistics on runtime and floating-point operation count.
 *        See util.h for the definition of 'SuperLUStat_t'.
 *
 * info   (output) int*
 *        = 0: successful exit
 *        < 0: if info = -i, the i-th argument had an illegal value
 *        > 0: if info = i, U(i,i) is exactly zero. The factorization has
 *             been completed, but the factor U is exactly singular,
 *             and division by zero will occur if it is used to solve a
 *             system of equations.
 * </pre>
 */
// #define pdgstrf3d_normal
// #define BigMem_pdgstrf3d
#define pdgstrf3d_cpugpu

// #undef SuperLargeScaleGPUBuffer

#ifdef SuperLargeScaleGPUBuffer
int_t dBcastRecv_sforest(sForest_t* sforest, int sendrank, int recrank, gridinfo3d_t *grid3d, int_t nsupers)
{
    int tag = 200;
    MPI_Status status;

    if (grid3d->zscp.Iam == sendrank)
    {       
        MPI_Send(&(sforest->nNodes), 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        MPI_Send(&(sforest->numTrees), 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        MPI_Send(&(sforest->weight), 1, MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm);
        
        if (sforest->nNodes > 0)
        {
            MPI_Send(sforest->nodeList, sforest->nNodes, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        }
        treeTopoInfo_t *treeTopoInfo = &(sforest->topoInfo);

        if (sforest->nNodes >= 0)
        {
            MPI_Send(treeTopoInfo->myIperm, nsupers, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
            MPI_Send(&(treeTopoInfo->numLvl), 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
            MPI_Send(treeTopoInfo->eTreeTopLims, treeTopoInfo->numLvl + 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        }

        
    }

    if (grid3d->zscp.Iam == recrank)
    {               
        MPI_Recv(&(sforest->nNodes), 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        MPI_Recv(&(sforest->numTrees), 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        MPI_Recv(&(sforest->weight), 1, MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm, &status);

        int_t *nodeList;
        if (sforest->nNodes > 0)
        {
            nodeList = (int_t*) SUPERLU_MALLOC (sforest->nNodes * sizeof (int_t));
            assert(nodeList);
            MPI_Recv(nodeList, sforest->nNodes, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        }
        else
        {
            nodeList = NULL;
        }

        sforest->nodeList = nodeList;        
        treeTopoInfo_t ttI;
        int_t* myTopLims;
        if (sforest->nNodes >= 0)
        {
            ttI.myIperm = INT_T_ALLOC(nsupers);
            MPI_Recv(ttI.myIperm, nsupers, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
            MPI_Recv(&(ttI.numLvl), 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
            myTopLims = INT_T_ALLOC(ttI.numLvl + 1);
            MPI_Recv(myTopLims, ttI.numLvl + 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        }
        else
        {
            ttI.myIperm = NULL;
            myTopLims = NULL;
        }                
        
        ttI.eTreeTopLims = myTopLims;
        sforest->topoInfo = ttI;
    }   
    
    return 0;
}

void backup_LUstruct(dLocalLU_t *Llu_temp, dLocalLU_t *Llu, gridinfo3d_t *grid3d, int_t nsupers)
{
    int_t i;

    int_t Pc = grid3d->npcol;
    int_t Pr = grid3d->nprow;
    
    int_t nbc = CEILING(nsupers, Pc);
    int_t nbr = CEILING(nsupers, Pr);

    int_t *Lrowind_bc_ptr_ilen = Llu->Lrowind_bc_ptr_ilen;
    int_t *Lnzval_bc_ptr_ilen = Llu->Lnzval_bc_ptr_ilen;
    int_t *Ufstnz_br_ptr_ilen = Llu->Ufstnz_br_ptr_ilen;
    int_t *Unzval_br_ptr_ilen = Llu->Unzval_br_ptr_ilen;

    Llu_temp->Lrowind_bc_ptr_ilen = intCalloc_dist(nbc);
    Llu_temp->Lnzval_bc_ptr_ilen = intCalloc_dist(nbc);
    Llu_temp->Ufstnz_br_ptr_ilen = intCalloc_dist(nbr);
    Llu_temp->Unzval_br_ptr_ilen = intCalloc_dist(nbr);
    memcpy(Llu_temp->Lrowind_bc_ptr_ilen, Lrowind_bc_ptr_ilen, nbc * sizeof(int_t));
    memcpy(Llu_temp->Lnzval_bc_ptr_ilen, Lnzval_bc_ptr_ilen, nbc * sizeof(int_t));
    memcpy(Llu_temp->Ufstnz_br_ptr_ilen, Ufstnz_br_ptr_ilen, nbr * sizeof(int_t));
    memcpy(Llu_temp->Unzval_br_ptr_ilen, Unzval_br_ptr_ilen, nbr * sizeof(int_t));

    int_t **isUsed_Lnzval_bc_ptr =(int_t**) SUPERLU_MALLOC(sizeof(int_t*)*nbc);

    Llu_temp->Lrowind_bc_ptr = (int_t**)SUPERLU_MALLOC(nbc * sizeof(int_t*));
    Llu_temp->Lnzval_bc_ptr = (double**)SUPERLU_MALLOC(nbc * sizeof(double*));

    #pragma omp for
    for (i = 0; i < nbc ; ++i)
    {   
        Llu_temp->Lrowind_bc_ptr[i] = NULL;                      
        if (Lrowind_bc_ptr_ilen[i])
        {            
            int_t *iLrowind_bc_ptr = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
            memcpy(iLrowind_bc_ptr, Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i] * sizeof(int_t));
            SUPERLU_FREE(Llu->Lrowind_bc_ptr[i]);
            Llu_temp->Lrowind_bc_ptr[i] = iLrowind_bc_ptr;            
            
        }
        Llu_temp->Lnzval_bc_ptr[i] = NULL;
        if (Lnzval_bc_ptr_ilen[i])
        {
            double *iLnzval_bc_ptr = doubleCalloc_dist(Lnzval_bc_ptr_ilen[i]);
            memcpy(iLnzval_bc_ptr, Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i] * sizeof(double));
            SUPERLU_FREE(Llu->Lnzval_bc_ptr[i]);
            Llu_temp->Lnzval_bc_ptr[i] = iLnzval_bc_ptr;            
        }

        isUsed_Lnzval_bc_ptr[i]=intCalloc_dist(nPart+1);
    }

    int_t **isUsed_Unzval_br_ptr = (int_t**)SUPERLU_MALLOC(nbr * sizeof(int_t*));
    Llu_temp->Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(nbr * sizeof(int_t*));
    Llu_temp->Unzval_br_ptr = (double**)SUPERLU_MALLOC(nbr * sizeof(double*));

    #pragma omp for
    for (i = 0; i < nbr ; ++i)
    {
        Llu_temp->Ufstnz_br_ptr[i] = NULL;       
        if (Ufstnz_br_ptr_ilen[i])
        {
            int_t *iUfstnz_br_ptr = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
            memcpy(iUfstnz_br_ptr, Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] * sizeof(int_t));
            SUPERLU_FREE(Llu->Ufstnz_br_ptr[i]);
            Llu_temp->Ufstnz_br_ptr[i] = iUfstnz_br_ptr;
            
        }
        Llu_temp->Unzval_br_ptr[i] = NULL;
        if (Unzval_br_ptr_ilen[i])
        {
            double *iUnzval_br_ptr = doubleCalloc_dist(Unzval_br_ptr_ilen[i]);
            memcpy(iUnzval_br_ptr, Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i] * sizeof(double));
            SUPERLU_FREE(Llu->Unzval_br_ptr[i]);
            Llu_temp->Unzval_br_ptr[i] = iUnzval_br_ptr;
            
        }

        isUsed_Unzval_br_ptr[i]=intCalloc_dist(nPart+1);
    }

    Llu_temp->isUsed_Lnzval_bc_ptr = isUsed_Lnzval_bc_ptr;
	Llu_temp->isUsed_Unzval_br_ptr = isUsed_Unzval_br_ptr;
}

void restore_LUstruct(dLocalLU_t *Llu, dLocalLU_t *Llu_temp, gridinfo3d_t *grid3d, int_t nsupers)
{
    int_t i;

    int_t Pc = grid3d->npcol;
    int_t Pr = grid3d->nprow;
    
    int_t nbc = CEILING(nsupers, Pc);
    int_t nbr = CEILING(nsupers, Pr);

    int_t *Lrowind_bc_ptr_ilen = Llu->Lrowind_bc_ptr_ilen;
    int_t *Lnzval_bc_ptr_ilen = Llu->Lnzval_bc_ptr_ilen;
    int_t *Ufstnz_br_ptr_ilen = Llu->Ufstnz_br_ptr_ilen;
    int_t *Unzval_br_ptr_ilen = Llu->Unzval_br_ptr_ilen;

    SUPERLU_FREE(Llu_temp->Lrowind_bc_ptr_ilen);
    SUPERLU_FREE(Llu_temp->Lnzval_bc_ptr_ilen);
    SUPERLU_FREE(Llu_temp->Ufstnz_br_ptr_ilen);
    SUPERLU_FREE(Llu_temp->Unzval_br_ptr_ilen);

    #pragma omp for
    for (i = 0; i < nbc ; ++i)
    {         
        Llu->Lrowind_bc_ptr[i] = NULL;               
        if (Lrowind_bc_ptr_ilen[i])
        {            
            int_t *iLrowind_bc_ptr = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
            memcpy(iLrowind_bc_ptr, Llu_temp->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i] * sizeof(int_t));
            SUPERLU_FREE(Llu_temp->Lrowind_bc_ptr[i]);
            Llu->Lrowind_bc_ptr[i] = iLrowind_bc_ptr;
            
        }
        Llu->Lnzval_bc_ptr[i] = NULL;
        if (Lnzval_bc_ptr_ilen[i])
        {
            double *iLnzval_bc_ptr = doubleCalloc_dist(Lnzval_bc_ptr_ilen[i]);
            memcpy(iLnzval_bc_ptr, Llu_temp->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i] * sizeof(double));
            SUPERLU_FREE(Llu_temp->Lnzval_bc_ptr[i]);
            Llu->Lnzval_bc_ptr[i] = iLnzval_bc_ptr;
            
        }

    }

    #pragma omp for
    for (i = 0; i < nbr ; ++i)
    {
        Llu->Ufstnz_br_ptr[i] = NULL;           
        if (Ufstnz_br_ptr_ilen[i])
        {
            int_t *iUfstnz_br_ptr = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
            memcpy(iUfstnz_br_ptr, Llu_temp->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] * sizeof(int_t));
            SUPERLU_FREE(Llu_temp->Ufstnz_br_ptr[i]);
            Llu->Ufstnz_br_ptr[i] = iUfstnz_br_ptr;
            
        }
        Llu->Unzval_br_ptr[i] = NULL;
        if (Unzval_br_ptr_ilen[i])
        {
            double *iUnzval_br_ptr = doubleCalloc_dist(Unzval_br_ptr_ilen[i]);
            memcpy(iUnzval_br_ptr, Llu_temp->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i] * sizeof(double));
            SUPERLU_FREE(Llu_temp->Unzval_br_ptr[i]);
            Llu->Unzval_br_ptr[i] = iUnzval_br_ptr;            
        }
    }

    SUPERLU_FREE(Llu_temp->isUsed_Lnzval_bc_ptr);
	SUPERLU_FREE(Llu_temp->isUsed_Unzval_br_ptr);
}

void dBcastRecv_LUstruct(dLUstruct_t *LUstruct, int sendrank, int recrank, gridinfo3d_t *grid3d, int_t nsupers)
{
    int tag = 500;
    MPI_Status status;
    int_t i;

    int_t Pc = grid3d->npcol;
    int_t Pr = grid3d->nprow;
    
    int_t nbc = CEILING(nsupers, Pc);
    int_t nbr = CEILING(nsupers, Pr);

    dLocalLU_t *Llu = LUstruct->Llu;
    int_t *Lrowind_bc_ptr_ilen = Llu->Lrowind_bc_ptr_ilen;
    int_t *Lnzval_bc_ptr_ilen = Llu->Lnzval_bc_ptr_ilen;
    int_t *Ufstnz_br_ptr_ilen = Llu->Ufstnz_br_ptr_ilen;
    int_t *Unzval_br_ptr_ilen = Llu->Unzval_br_ptr_ilen;

    if (grid3d->zscp.Iam == sendrank)
    {
        Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
        
        MPI_Send(&(Glu_persist->xsup_len), 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        MPI_Send(&(Glu_persist->supno_len), 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        MPI_Send(Glu_persist->xsup, Glu_persist->xsup_len, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        MPI_Send(Glu_persist->supno, Glu_persist->supno_len, mpi_int_t, recrank, tag++, grid3d->zscp.comm); 
        
        for (i = 0; i < nbc ; ++i)
        {                        
            if (Lrowind_bc_ptr_ilen[i])
            {
                int_t *iLrowind_bc_ptr = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
                MPI_Recv(iLrowind_bc_ptr, Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm, &status);
                MPI_Send(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm);
                SUPERLU_FREE(Llu->Lrowind_bc_ptr[i]);
                Llu->Lrowind_bc_ptr[i] = iLrowind_bc_ptr;
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                double *iLnzval_bc_ptr = doubleCalloc_dist(Lnzval_bc_ptr_ilen[i]);
                MPI_Recv(iLnzval_bc_ptr, Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm, &status);
                MPI_Send(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm);
                SUPERLU_FREE(Llu->Lnzval_bc_ptr[i]);
                Llu->Lnzval_bc_ptr[i] = iLnzval_bc_ptr;
            }
        }            
        
        for (i = 0; i < nbr ; ++i)
        {
            if (Ufstnz_br_ptr_ilen[i])
            {
                int_t *iUfstnz_br_ptr = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
                MPI_Recv(iUfstnz_br_ptr, Ufstnz_br_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm, &status);
                MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm);
                SUPERLU_FREE(Llu->Ufstnz_br_ptr[i]);
                Llu->Ufstnz_br_ptr[i] = iUfstnz_br_ptr;
            }
            if (Unzval_br_ptr_ilen[i])
            {
                double *iUnzval_br_ptr = doubleCalloc_dist(Unzval_br_ptr_ilen[i]);
                MPI_Recv(iUnzval_br_ptr, Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm, &status);
                MPI_Send(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm);
                SUPERLU_FREE(Llu->Unzval_br_ptr[i]);
                Llu->Unzval_br_ptr[i] = iUnzval_br_ptr;
            }
        }

        MPI_Send(Llu->ToRecv, nsupers, MPI_INT, recrank, tag++, grid3d->zscp.comm);
        MPI_Send(Llu->ToSendD, nbr, MPI_INT, recrank, tag++, grid3d->zscp.comm);
        for (i = 0; i < nbc; ++i)
        {
            /* code */
            MPI_Send(Llu->ToSendR[i], Pc, MPI_INT, recrank, tag++, grid3d->zscp.comm);
        }

        MPI_Send( Llu->bufmax, NBUFFERS, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        #ifdef SuperLargeScale
        MPI_Send( &(Llu->isSave), 1, MPI_INT, recrank, tag++, grid3d->zscp.comm);
        #endif
        // MPI_Send( &(Llu->isEmpty), 1, MPI_INT, recrank, tag++, grid3d->zscp.comm);

    }

    if (grid3d->zscp.Iam == recrank)
    {        
        
        Glu_persist_t *Glu_persist = LUstruct->Glu_persist;        

        MPI_Recv(&(Glu_persist->xsup_len), 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        MPI_Recv(&(Glu_persist->supno_len), 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        Glu_persist->xsup = intMalloc_dist(Glu_persist->xsup_len);
        Glu_persist->supno = intMalloc_dist(Glu_persist->supno_len);
        MPI_Recv(Glu_persist->xsup, Glu_persist->xsup_len, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        MPI_Recv(Glu_persist->supno, Glu_persist->supno_len, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);

        for (i = 0; i < nbc ; ++i)
        {            
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Send(Llu->Lrowind_bc_ptr[i], Llu->Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm);
                MPI_Recv(Llu->Lrowind_bc_ptr[i], Llu->Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                MPI_Send(Llu->Lnzval_bc_ptr[i], Llu->Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm);
                MPI_Recv(Llu->Lnzval_bc_ptr[i], Llu->Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm, &status);
            }
        }

        for (i = 0; i < nbr ; ++i)
        {
            if (Ufstnz_br_ptr_ilen[i])
            {                
                if (grid3d->zscp.Iam)
                {
                    MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm);
                    MPI_Recv(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
                }
                else
                {
                    MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] - 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm);
                    MPI_Recv(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] - 1, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
                }
            }
            if (Unzval_br_ptr_ilen[i])
            {
                MPI_Send(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm);
                MPI_Recv(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm, &status);
            }
            
        }

        /* Recv from no one (0), left (1), and up (2).*/
        int *ToRecv = SUPERLU_MALLOC(nsupers * sizeof(int));
        MPI_Recv(ToRecv, nsupers, MPI_INT, sendrank, tag++, grid3d->zscp.comm, &status);
                    /* Whether need to send down block row. */
        int *ToSendD = SUPERLU_MALLOC(nbr * sizeof(int));
        MPI_Recv(ToSendD, nbr, MPI_INT, sendrank, tag++, grid3d->zscp.comm, &status);
                    /* List of processes to send right block col. */
        int **ToSendR = (int **) SUPERLU_MALLOC(nbc * sizeof(int*));

        for (i = 0; i < nbc; ++i)
        {
            /* code */
            //ToSendR[i] = INT_T_ALLOC(Pc);
            ToSendR[i] = SUPERLU_MALLOC(Pc * sizeof(int));
            MPI_Recv(ToSendR[i], Pc, MPI_INT, sendrank, tag++, grid3d->zscp.comm, &status);
        }

        MPI_Recv(Llu->bufmax, NBUFFERS, mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        #ifdef SuperLargeScale
        MPI_Recv(&(Llu->isSave), 1, MPI_INT, sendrank, tag++, grid3d->zscp.comm, &status);
        #endif
        // MPI_Recv(&(Llu->isEmpty), 1, MPI_INT, sendrank, tag++, grid3d->zscp.comm, &status);

        Llu->ToRecv = ToRecv ;
        Llu->ToSendD = ToSendD ;
        Llu->ToSendR = ToSendR ;
        
    }

}

void dBcastRecv_LUstruct1(dLUstruct_t *LUstruct, int sendrank, int recrank, gridinfo3d_t *grid3d, int_t nsupers)
{
    int tag = 800;
    MPI_Status status;
    MPI_Request *send_req, *recv_req;
    int_t i;    
    int nireq, ndreq;

    int_t Pc = grid3d->npcol;
    int_t Pr = grid3d->nprow;
    
    int_t nbc = CEILING(nsupers, Pc);
    int_t nbr = CEILING(nsupers, Pr);

    int_t maxnb = SUPERLU_MAX(nbc, nbr);

    
    // for (i = 0; i < 2; i++)
    // {
    //     send_req[i] = MPI_REQUEST_NULL;
    // }
    // send_req = MPI_REQUEST_NULL;
    
    // if ( !(recv_req = (MPI_Request *)
	//    SUPERLU_MALLOC(2*maxnb *sizeof(MPI_Request))))
    //     ABORT("Malloc fails for recv_req[].");
    // recv_req = send_req + maxnb;

    // int_t **senddata = (int_t**)SUPERLU_MALLOC(200 * sizeof(int_t*));
    // if (grid3d->zscp.Iam == sendrank)
    // {       
    //     for (int_t j = 0; j < 200; j++)
    //     {
    //         senddata[j] = intCalloc_dist(200);
    //         for (int_t i = 0; i < 200; i++)
    //         {
    //             senddata[j][i] = i;
    //         }
    //         MPI_Isend(senddata[j], 200, mpi_int_t, 1, tag + j, grid3d->zscp.comm, send_req);
    //     }   
        
         
    // }
    // if (grid3d->zscp.Iam == recrank)
    // {       

    //     for (int_t j = 0; j < 200; j++)
    //     {
    //         senddata[j] = intCalloc_dist(200);
    //         MPI_Irecv(senddata[j], 200, mpi_int_t, 0, tag + j, grid3d->zscp.comm, send_req);
    //         MPI_Wait(send_req, &status);
    //     }

    //     PrintInt10("recv", 200, senddata[150]);
    // }

    // while (1)
    // {
    //     /* code */
    // }
    

    dLocalLU_t *Llu = LUstruct->Llu;
        
    int_t *Lrowind_bc_ptr_ilen = Llu->Lrowind_bc_ptr_ilen;
    int_t *Lnzval_bc_ptr_ilen = Llu->Lnzval_bc_ptr_ilen;
    int_t *Ufstnz_br_ptr_ilen = Llu->Ufstnz_br_ptr_ilen;
    int_t *Unzval_br_ptr_ilen = Llu->Unzval_br_ptr_ilen;

    int_t **Lrowind_bc_ptr;
    double **Lnzval_bc_ptr;

    #ifdef Test

    nireq = 0;
    if (grid3d->zscp.Iam == sendrank)
    {
        #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Isend(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
                // MPI_Send(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm);
                // printf("%d: send %d, %d\n", grid3d->iam, i, Lrowind_bc_ptr_ilen[i]);
                // PrintInt10("iLrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Llu->Lrowind_bc_ptr[i]);
                // if (i % 2000 == 0)
                // {
                //     printf("%d: send %d, %d\n", grid3d->iam, i, Lrowind_bc_ptr_ilen[i]);
                //     PrintInt10("iLrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Llu->Lrowind_bc_ptr[i]);
                // }
            }
            // if (Lnzval_bc_ptr_ilen[i])
            // {
            //     MPI_Isend(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, i * 2 + 1, grid3d->zscp.comm, &send_req[i + nbc]);
            //     ndreq++;
            // }
            
            // PrintInt10("Lrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Llu->Lrowind_bc_ptr[i]);
        }   

        // #pragma omp for
        // for (i = 100; i < 200 ; ++i)
        // {
        //     if (Lrowind_bc_ptr_ilen[i])
        //     {
        //         MPI_Isend(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
        //         // MPI_Send(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        //         // printf("%d: send %d, %d\n", grid3d->iam, i, Lrowind_bc_ptr_ilen[i]);
        //         // PrintInt10("iLrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Llu->Lrowind_bc_ptr[i]);
        //     }
        //     // if (Lnzval_bc_ptr_ilen[i])
        //     // {
        //     //     MPI_Isend(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, i * 2 + 1, grid3d->zscp.comm, &send_req[i + nbc]);
        //     //     ndreq++;
        //     // }
            
        //     // PrintInt10("Lrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Llu->Lrowind_bc_ptr[i]);
        // }   

        #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Wait(send_req, &status);
            }
        }
          
    }
    else
    {
        Lrowind_bc_ptr = (int_t**)SUPERLU_MALLOC(nbc * sizeof(int_t*));
        Lnzval_bc_ptr = (double**)SUPERLU_MALLOC(nbc * sizeof(double*));

        double time1 = SuperLU_timer_();
        printf("%d: time %e\n", grid3d->iam, time1);
        #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                Lrowind_bc_ptr[i] = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
            }
        }
        #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                // Lrowind_bc_ptr[i] = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
                // printf("%d: recv0 %d, %d\n", grid3d->iam, i, Lrowind_bc_ptr_ilen[i]);
                MPI_Irecv(Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag + i, grid3d->zscp.comm, send_req);
                // Lrowind_bc_ptr[i] = iLrowind_bc_ptr;
                // MPI_Wait(send_req, &status);
                
                // if (i % 2000 == 0)
                // {
                //     printf("%d: recv %d, %d\n", grid3d->iam, i, Lrowind_bc_ptr_ilen[i]);
                //     PrintInt10("iLrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Lrowind_bc_ptr[i]);
                // }
                
                
            }
            // if (Lnzval_bc_ptr_ilen[i])
            // {
            //     double *iLnzval_bc_ptr = doubleCalloc_dist(Lnzval_bc_ptr_ilen[i]);
            //     MPI_Irecv(iLnzval_bc_ptr, Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, i * 2 + 1, grid3d->zscp.comm, &recv_req[i + nbc]);
            //     Lnzval_bc_ptr[i] = iLnzval_bc_ptr;
            //     ndreq++;
            // }
            
            // PrintInt10("Lrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Lrowind_bc_ptr[i]);
        }

        #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Wait(send_req, &status);
                
                // if (i % 2000 == 0)
                // {
                //     printf("%d: recv %d, %d\n", grid3d->iam, i, Lrowind_bc_ptr_ilen[i]);
                //     PrintInt10("iLrowind_bc_ptr", Lrowind_bc_ptr_ilen[i], Lrowind_bc_ptr[i]);
                // }
            }
        }

        printf("%d: time %e\n", grid3d->iam, SuperLU_timer_() - time1);
        
    }

    // MPI_Waitall(nireq, send_req, &status);

    if (grid3d->zscp.Iam != sendrank)
    {
        PrintInt10("Lrowind_bc_ptr2", Lrowind_bc_ptr_ilen[nbc - 1], Lrowind_bc_ptr[nbc - 1]);
    }   
    
    while (1)
    {
        /* code */
    }

    #endif

    MPI_Barrier(grid3d->zscp.comm);
    if (grid3d->zscp.Iam == sendrank)
    {
        
        Glu_persist_t *Glu_persist = LUstruct->Glu_persist;                
        SUPERLU_FREE(Glu_persist->xsup);
        SUPERLU_FREE(Glu_persist->supno);
        SUPERLU_FREE(Glu_persist);
        
        // for (i = 0; i < nbc ; ++i)
        // {
        //     double time1;
        //     if (grid3d->iam == 0)
        //     {
        //         time1 = SuperLU_timer_();
        //     }            
            
        //     if (Lrowind_bc_ptr_ilen[i])
        //     {
        //         MPI_Send(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        //         MPI_Recv(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm, &status);
        //     }
        //     if (Lnzval_bc_ptr_ilen[i])
        //     {
        //         MPI_Send(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm);
        //         MPI_Recv(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm, &status);                
        //     }

        //     if (grid3d->iam == 0)
        //     {
        //         printf("%d: time %e, %ld, %ld\n", grid3d->iam, SuperLU_timer_() - time1, Lrowind_bc_ptr_ilen[i], Lnzval_bc_ptr_ilen[i]);
        //     }
        // }
        
        // for (i = 0; i < nbr ; ++i)
        // {
        //     // printf("%d: time3 %e\n", grid3d->iam, SuperLU_timer_());
        //     if (Ufstnz_br_ptr_ilen[i])
        //     {                
        //         if (grid3d->zscp.Iam)
        //         {
        //             MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        //             MPI_Recv(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, recrank, tag++, grid3d->zscp.comm, &status);
        //         }
        //         else
        //         {
        //             MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] - 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm);
        //             MPI_Recv(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] - 1, mpi_int_t, recrank, tag++, grid3d->zscp.comm, &status);
        //         }                
                
        //     }
        //     if (Unzval_br_ptr_ilen[i])
        //     {
        //         MPI_Send(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm);
        //         MPI_Recv(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag++, grid3d->zscp.comm, &status);
        //     }
        //     // printf("%d: time4 %e\n", grid3d->iam, SuperLU_timer_());
            
        // }  

        #ifdef IEXCHANGE
        idynamicarray_iexchange(Llu->Lrowind_bc_ptr, Lrowind_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        ddynamicarray_iexchange(Llu->Lnzval_bc_ptr, Lnzval_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        Ufstnz_br_ptr_iexchange(Llu->Ufstnz_br_ptr, Ufstnz_br_ptr_ilen, nbr, sendrank, recrank, grid3d);

        ddynamicarray_iexchange(Llu->Unzval_br_ptr, Unzval_br_ptr_ilen, nbr, sendrank, recrank, grid3d);

        #else

        idynamicarray_exchange(Llu->Lrowind_bc_ptr, Lrowind_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        ddynamicarray_exchange(Llu->Lnzval_bc_ptr, Lnzval_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        Ufstnz_br_ptr_exchange(Llu->Ufstnz_br_ptr, Ufstnz_br_ptr_ilen, nbr, sendrank, recrank, grid3d);

        ddynamicarray_exchange(Llu->Unzval_br_ptr, Unzval_br_ptr_ilen, nbr, sendrank, recrank, grid3d);

        #endif







        #ifdef Test
        // #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Isend(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                MPI_Isend(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, tag + i + nbc, grid3d->zscp.comm, send_req);
            }
        }

        // #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Wait(send_req, &status);
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                MPI_Wait(send_req, &status);
            }
        }

        for (i = 0; i < nbc ; ++i)
        {            
            if (Lrowind_bc_ptr_ilen[i])
            {
                
                if (Lrowind_bc_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Lrowind_bc_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Lrowind_bc_ptr_ilen[i])
                        {
                            len = Lrowind_bc_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Irecv(Llu->Lrowind_bc_ptr[i] + minstep, len, mpi_int_t, recrank, tag + i + nbc * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, recrank, tag + i + nbc, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        } 

        for (i = 0; i < nbc ; ++i)
        {            
            if (Lnzval_bc_ptr_ilen[i])
            {
                
                if (Lnzval_bc_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Lnzval_bc_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Lnzval_bc_ptr_ilen[i])
                        {
                            len = Lnzval_bc_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Irecv(Llu->Lnzval_bc_ptr[i] + minstep, len, MPI_DOUBLE, recrank, tag + i + nbc * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, recrank, tag + i + nbc, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        }
        
        // for (i = 0; i < 100 ; ++i)
        // {            
        //     if (Lnzval_bc_ptr_ilen[i])
        //     {
        //         MPI_Wait(send_req, &status);
        //     }
        // }

        PrintInt10("recv Lrowind_bc_ptr", Llu->Lrowind_bc_ptr_ilen[nbc - 1], Llu->Lrowind_bc_ptr[nbc - 1]); 
        
        // for (i = 0; i < nbr ; ++i)
        // {
        //     if (Ufstnz_br_ptr_ilen[i])
        //     {
        //         if (grid3d->zscp.Iam)
        //         {
        //             MPI_Isend(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
        //         }
        //         else
        //         {                    
        //             MPI_Isend(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] - 1, mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
        //         }
        //         MPI_Wait(send_req, &status);
        //     }
        //     if (Unzval_br_ptr_ilen[i])
        //     {
        //         MPI_Isend(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag + i + nbr, grid3d->zscp.comm, send_req);
        //         MPI_Wait(send_req, &status);
        //     }
            
        // }

        printf("%d, break1\n", grid3d->iam);

        for (i = 0; i < nbr ; ++i)
        {        
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }
            
            if (ilen > 0)
            {
                // printf("%d: %ld, isend %ld\n", grid3d->iam, i, ilen);
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(Llu->Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, recrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                                       
                    MPI_Isend(Llu->Ufstnz_br_ptr[i], ilen, mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        }

        printf("%d, break2\n", grid3d->iam);
        

        for (i = 0; i < nbr ; ++i)
        {            
            if (Unzval_br_ptr_ilen[i])
            {
                
                if (Unzval_br_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Unzval_br_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Unzval_br_ptr_ilen[i])
                        {
                            len = Unzval_br_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Isend(Llu->Unzval_br_ptr[i] + minstep, len, MPI_DOUBLE, recrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag + i + nbr, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        }

        for (i = 0; i < nbr ; ++i)
        {        
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }
            
            if (ilen > 0)
            {
                
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Irecv(Llu->Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, recrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(Llu->Ufstnz_br_ptr[i], ilen, mpi_int_t, recrank, tag + i + nbr, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        } 

        for (i = 0; i < nbr ; ++i)
        {            
            if (Unzval_br_ptr_ilen[i])
            {
                
                if (Unzval_br_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Unzval_br_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Unzval_br_ptr_ilen[i])
                        {
                            len = Unzval_br_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(Llu->Unzval_br_ptr[i] + minstep, len, MPI_DOUBLE, recrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag + i + nbr, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        }









        
        
        // for (i = 0; i < nbr ; ++i)
        // {
        //     if (Ufstnz_br_ptr_ilen[i])
        //     {
        //         if (grid3d->zscp.Iam)
        //         {                    
        //             MPI_Irecv(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
        //         }
        //         else
        //         {
        //             MPI_Irecv(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i] - 1, mpi_int_t, recrank, tag + i, grid3d->zscp.comm, send_req);
        //         }
                
        //     }           
            
        // }

        // for (i = 0; i < nbr ; ++i)
        // {
        //     if (Ufstnz_br_ptr_ilen[i])
        //     {
        //         MPI_Wait(send_req, &status);                
        //     }       
        // }  

        // for (i = 0; i < nbr ; ++i)
        // {
        //     if (Unzval_br_ptr_ilen[i])
        //     {                
        //         MPI_Irecv(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, recrank, tag + i + nbr, grid3d->zscp.comm, send_req);
        //     }
        // }

        #endif

        SUPERLU_FREE(Llu->ToRecv);
        SUPERLU_FREE(Llu->ToSendD);
        for (i = 0; i < nbc; ++i)
        {
            SUPERLU_FREE(Llu->ToSendR[i]);
        }
        SUPERLU_FREE(Llu->ToSendR);

    }

    if (grid3d->zscp.Iam == recrank)
    {        
        
        // for (i = 0; i < nbc ; ++i)
        // {                        
        //     if (Lrowind_bc_ptr_ilen[i])
        //     {
        //         int_t *iLrowind_bc_ptr = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
        //         MPI_Recv(iLrowind_bc_ptr, Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        //         MPI_Send(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm);
        //         SUPERLU_FREE(Llu->Lrowind_bc_ptr[i]);
        //         Llu->Lrowind_bc_ptr[i] = iLrowind_bc_ptr;
        //     }
        //     if (Lnzval_bc_ptr_ilen[i])
        //     {
        //         double *iLnzval_bc_ptr = doubleCalloc_dist(Lnzval_bc_ptr_ilen[i]);
        //         MPI_Recv(iLnzval_bc_ptr, Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm, &status);
        //         MPI_Send(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm);
        //         SUPERLU_FREE(Llu->Lnzval_bc_ptr[i]);
        //         Llu->Lnzval_bc_ptr[i] = iLnzval_bc_ptr;
        //     }
        // }        
        
        // for (i = 0; i < nbr ; ++i)
        // {
            
        //     if (Ufstnz_br_ptr_ilen[i])
        //     {
        //         int_t *iUfstnz_br_ptr = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
        //         MPI_Recv(iUfstnz_br_ptr, Ufstnz_br_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm, &status);
        //         MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm);
        //         SUPERLU_FREE(Llu->Ufstnz_br_ptr[i]);
        //         Llu->Ufstnz_br_ptr[i] = iUfstnz_br_ptr;
        //     }
        //     if (Unzval_br_ptr_ilen[i])
        //     {
        //         double *iUnzval_br_ptr = doubleCalloc_dist(Unzval_br_ptr_ilen[i]);
        //         MPI_Recv(iUnzval_br_ptr, Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm, &status);
        //         MPI_Send(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm);
        //         SUPERLU_FREE(Llu->Unzval_br_ptr[i]);
        //         Llu->Unzval_br_ptr[i] = iUnzval_br_ptr;
        //     }
        // }

        #ifdef IEXCHANGE
        idynamicarray_iexchange(Llu->Lrowind_bc_ptr, Lrowind_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        ddynamicarray_iexchange(Llu->Lnzval_bc_ptr, Lnzval_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        Ufstnz_br_ptr_iexchange(Llu->Ufstnz_br_ptr, Ufstnz_br_ptr_ilen, nbr, sendrank, recrank, grid3d);

        ddynamicarray_iexchange(Llu->Unzval_br_ptr, Unzval_br_ptr_ilen, nbr, sendrank, recrank, grid3d);
        #else
        idynamicarray_exchange(Llu->Lrowind_bc_ptr, Lrowind_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        ddynamicarray_exchange(Llu->Lnzval_bc_ptr, Lnzval_bc_ptr_ilen, nbc, sendrank, recrank, grid3d);

        Ufstnz_br_ptr_exchange(Llu->Ufstnz_br_ptr, Ufstnz_br_ptr_ilen, nbr, sendrank, recrank, grid3d);

        ddynamicarray_exchange(Llu->Unzval_br_ptr, Unzval_br_ptr_ilen, nbr, sendrank, recrank, grid3d);
        #endif























        #ifdef Test

        Lrowind_bc_ptr = (int_t**)SUPERLU_MALLOC(nbc * sizeof(int_t*));
        Lnzval_bc_ptr = (double**)SUPERLU_MALLOC(nbc * sizeof(double*));

        gridinfo_t* grid = &(grid3d->grid2d);        

        // #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                Lrowind_bc_ptr[i] = intCalloc_dist(Lrowind_bc_ptr_ilen[i]);
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                Lnzval_bc_ptr[i] = doubleCalloc_dist(Lnzval_bc_ptr_ilen[i]);
            }
        }

        // #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {                        
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Irecv(Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag + i, grid3d->zscp.comm, send_req);
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                MPI_Irecv(Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, tag + i + nbc, grid3d->zscp.comm, send_req);
            }
        }

        // #pragma omp for
        for (i = 0; i < nbc ; ++i)
        {
            if (Lrowind_bc_ptr_ilen[i])
            {
                MPI_Wait(send_req, &status);
            }
            if (Lnzval_bc_ptr_ilen[i])
            {
                MPI_Wait(send_req, &status);
            }
        }

        Printdouble5("recv Lnzval_bc_ptr", Llu->Lnzval_bc_ptr_ilen[nbc - 1], Lnzval_bc_ptr[nbc - 1]);

        double time1;
        if (grid->iam == 0)
        {
            time1 = SuperLU_timer_();
            printf("%d: time %e\n", grid3d->iam, time1);
        }

        for (i = 0; i < nbc ; ++i)
        {            
            
            if (Lrowind_bc_ptr_ilen[i])
            {                
                if (Lrowind_bc_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Lrowind_bc_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Lrowind_bc_ptr_ilen[i])
                        {
                            len = Lrowind_bc_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(Llu->Lrowind_bc_ptr[i] + minstep, len, mpi_int_t, sendrank, tag + i + nbc * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Llu->Lrowind_bc_ptr[i], Lrowind_bc_ptr_ilen[i], mpi_int_t, sendrank, tag + i + nbc, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                SUPERLU_FREE(Llu->Lrowind_bc_ptr[i]);
                Llu->Lrowind_bc_ptr[i] = Lrowind_bc_ptr[i];
            }
        }        

        for (i = 0; i < nbc ; ++i)
        {            
            
            if (Lnzval_bc_ptr_ilen[i])
            {
                if (Lnzval_bc_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Lnzval_bc_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Lnzval_bc_ptr_ilen[i])
                        {
                            len = Lnzval_bc_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(Llu->Lnzval_bc_ptr[i] + minstep, len, MPI_DOUBLE, sendrank, tag + i + nbc * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Llu->Lnzval_bc_ptr[i], Lnzval_bc_ptr_ilen[i], MPI_DOUBLE, sendrank, tag + i + nbc, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                SUPERLU_FREE(Llu->Lnzval_bc_ptr[i]);
                Llu->Lnzval_bc_ptr[i] = Lnzval_bc_ptr[i];
            }
        }       
        

        if (grid->iam == 0)
        {
            printf("%d: time %e\n", grid3d->iam, SuperLU_timer_() - time1);
        }

        Printdouble5("new Lnzval_bc_ptr", Llu->Lnzval_bc_ptr_ilen[nbc - 1], Llu->Lnzval_bc_ptr[nbc - 1]);
        
        int_t **Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(nbr * sizeof(int_t*));
        double **Unzval_br_ptr = (double**)SUPERLU_MALLOC(nbr * sizeof(double*));

        for (i = 0; i < nbr ; ++i)
        {
            if (Ufstnz_br_ptr_ilen[i])
            {
                Ufstnz_br_ptr[i] = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
            }
            if (Unzval_br_ptr_ilen[i])
            {
                Unzval_br_ptr[i] = doubleCalloc_dist(Unzval_br_ptr_ilen[i]);
            }
        }
        
        // for (i = 0; i < nbr ; ++i)
        // {            
        //     if (Ufstnz_br_ptr_ilen[i])            
        //     {                
        //         MPI_Irecv(Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, sendrank, tag + i, grid3d->zscp.comm, send_req);
        //         MPI_Wait(send_req, &status);
        //     }
        //     if (Unzval_br_ptr_ilen[i])
        //     {
        //         MPI_Irecv(Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag + i + nbr, grid3d->zscp.comm, send_req);
        //         MPI_Wait(send_req, &status);
        //     }
        // }

        // while (1)
        // {
        //     /* code */
        // }

        printf("%d, break1\n", grid3d->iam);


        for (i = 0; i < nbr ; ++i)
        {        
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }
            
            if (ilen > 0)
            {
                // printf("%d: %ld, irecv %ld\n", grid3d->iam, i, ilen);
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Irecv(Llu->Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, sendrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                                       
                    MPI_Irecv(Llu->Ufstnz_br_ptr[i], ilen, mpi_int_t, sendrank, tag + i, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        }

        printf("%d, break2\n", grid3d->iam);        

        for (i = 0; i < nbr ; ++i)
        {            
            if (Unzval_br_ptr_ilen[i])
            {
                
                if (Unzval_br_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Unzval_br_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Unzval_br_ptr_ilen[i])
                        {
                            len = Unzval_br_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(Llu->Unzval_br_ptr[i] + minstep, len, MPI_DOUBLE, sendrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag + i + nbr, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                
            }
        }       
        

        // while (1)
        // {
        //     /* code */
        // }
        
        
        // for (i = 0; i < nbr ; ++i)
        // {            
        //     if (Ufstnz_br_ptr_ilen[i])            
        //     {                
        //         MPI_Wait(send_req, &status);
        //     }
        //     if (Unzval_br_ptr_ilen[i])
        //     {
        //         MPI_Wait(send_req, &status);
        //     }
        // }       


        for (i = 0; i < nbr ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {                
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(Llu->Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, sendrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Llu->Ufstnz_br_ptr[i], ilen, mpi_int_t, sendrank, tag + i + nbr, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                SUPERLU_FREE(Llu->Ufstnz_br_ptr[i]);
                Llu->Ufstnz_br_ptr[i] = Ufstnz_br_ptr[i];
            }
        }        

        for (i = 0; i < nbr ; ++i)
        {            
            
            if (Unzval_br_ptr_ilen[i])
            {
                if (Unzval_br_ptr_ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < Unzval_br_ptr_ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= Unzval_br_ptr_ilen[i])
                        {
                            len = Unzval_br_ptr_ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(Llu->Unzval_br_ptr[i] + minstep, len, MPI_DOUBLE, sendrank, tag + i + nbr * (j + 1), grid3d->zscp.comm, send_req);
                            MPI_Wait(send_req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag + i + nbr, grid3d->zscp.comm, send_req);
                    MPI_Wait(send_req, &status);
                }
                SUPERLU_FREE(Llu->Unzval_br_ptr[i]);
                Llu->Unzval_br_ptr[i] = Unzval_br_ptr[i];
            }
        }


        
        
        // for (i = 0; i < nbr ; ++i)
        // {
            
        //     if (Ufstnz_br_ptr_ilen[i])
        //     {
        //         MPI_Send(Llu->Ufstnz_br_ptr[i], Ufstnz_br_ptr_ilen[i], mpi_int_t, sendrank, tag++, grid3d->zscp.comm);
        //         SUPERLU_FREE(Llu->Ufstnz_br_ptr[i]);
        //         Llu->Ufstnz_br_ptr[i] = Ufstnz_br_ptr[i];
        //     }
        //     if (Unzval_br_ptr_ilen[i])
        //     {                
        //         MPI_Send(Llu->Unzval_br_ptr[i], Unzval_br_ptr_ilen[i], MPI_DOUBLE, sendrank, tag++, grid3d->zscp.comm);
        //         SUPERLU_FREE(Llu->Unzval_br_ptr[i]);
        //         Llu->Unzval_br_ptr[i] = Unzval_br_ptr[i];
        //     }
        // }

        #endif           
        
    }

}

void idynamicarray_exchange(int_t **data, int_t *ilen, int_t row, int sendrank, int recrank, gridinfo3d_t *grid3d)
{
    int tag = 800;
    MPI_Status status;
    int_t i;

    if (grid3d->zscp.Iam == sendrank)
    {
        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {                
                         
                MPI_Send(data[i], ilen[i], mpi_int_t, recrank, tag + i, grid3d->zscp.comm);
                MPI_Recv(data[i], ilen[i], mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &status);
            }
        }

        // for (i = 0; i < row ; ++i)
        // {            
        //     if (ilen[i] > 0)
        //     {
                
        //         MPI_Recv(data[i], ilen[i], mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &status);
                
        //     }
        // }
    }
    else
    {
        int_t **temp_data = (int_t**)SUPERLU_MALLOC(row * sizeof(int_t*));

        // for (i = 0; i < row ; ++i)
        // {        
        //     if (ilen[i] > 0)
        //     {
        //         temp_data[i] = intCalloc_dist(ilen[i]);
        //     }
        // }

        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                temp_data[i] = intCalloc_dist(ilen[i]);                
                MPI_Recv(temp_data[i], ilen[i], mpi_int_t, sendrank, tag + i, grid3d->zscp.comm, &status);
                MPI_Send(data[i], ilen[i], mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm);
                SUPERLU_FREE(data[i]);
                data[i] = temp_data[i];
            }
        } 

        // for (i = 0; i < row ; ++i)
        // {            
            
        //     if (ilen[i] > 0)
        //     {
                
        //         MPI_Send(data[i], ilen[i], mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm);
        //         SUPERLU_FREE(data[i]);
        //         data[i] = temp_data[i];
        //     }
        // }   
    }
    
}

void Ufstnz_br_ptr_exchange(int_t **Ufstnz_br_ptr, int_t *Ufstnz_br_ptr_ilen, int_t row, int sendrank, int recrank, gridinfo3d_t *grid3d)
{
    int tag = 800;
    MPI_Status status;
    int_t i;

    if (grid3d->zscp.Iam == sendrank)
    {
        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                                         
                MPI_Send(Ufstnz_br_ptr[i], ilen, mpi_int_t, recrank, tag + i, grid3d->zscp.comm);
                
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                                
                MPI_Recv(Ufstnz_br_ptr[i], ilen, mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &status);
                
            }
        }
    }
    else
    {
        int_t **temp_Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(row * sizeof(int_t*));

        for (i = 0; i < row ; ++i)
        {        
            if (Ufstnz_br_ptr_ilen[i])
            {
                temp_Ufstnz_br_ptr[i] = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                                
                MPI_Recv(temp_Ufstnz_br_ptr[i], ilen, mpi_int_t, sendrank, tag + i, grid3d->zscp.comm, &status);
                
            }
        } 

        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                
                MPI_Send(Ufstnz_br_ptr[i], ilen, mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm);
                SUPERLU_FREE(Ufstnz_br_ptr[i]);
                Ufstnz_br_ptr[i] = temp_Ufstnz_br_ptr[i];
            }
        }   
    }
    
}

void ddynamicarray_exchange(double **data, int_t *ilen, int_t row, int sendrank, int recrank, gridinfo3d_t *grid3d)
{
    int tag = 800;
    MPI_Status status;
    int_t i;    

    if (grid3d->zscp.Iam == sendrank)
    {
        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                              
                MPI_Send(data[i], ilen[i], MPI_DOUBLE, recrank, tag + i, grid3d->zscp.comm);
                MPI_Recv(data[i], ilen[i], MPI_DOUBLE, recrank, tag + i + row, grid3d->zscp.comm, &status);
                
            }
        }

        // for (i = 0; i < row ; ++i)
        // {            
        //     if (ilen[i] > 0)
        //     {
                                
        //         MPI_Recv(data[i], ilen[i], MPI_DOUBLE, recrank, tag + i + row, grid3d->zscp.comm, &status);
                
        //     }
        // }
    }
    else
    {
        double **temp_data = (double**)SUPERLU_MALLOC(row * sizeof(double*));

        // for (i = 0; i < row ; ++i)
        // {        
        //     if (ilen[i] > 0)
        //     {
        //         temp_data[i] = doubleCalloc_dist(ilen[i]);
        //     }
        // }

        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                temp_data[i] = doubleCalloc_dist(ilen[i]);                
                MPI_Recv(temp_data[i], ilen[i], MPI_DOUBLE, sendrank, tag + i, grid3d->zscp.comm, &status);
                MPI_Send(data[i], ilen[i], MPI_DOUBLE, sendrank, tag + i + row, grid3d->zscp.comm);
                SUPERLU_FREE(data[i]);
                data[i] = temp_data[i];
            }
        } 

        // for (i = 0; i < row ; ++i)
        // {            
            
        //     if (ilen[i] > 0)
        //     {
                     
        //         MPI_Send(data[i], ilen[i], MPI_DOUBLE, sendrank, tag + i + row, grid3d->zscp.comm);
        //         SUPERLU_FREE(data[i]);
        //         data[i] = temp_data[i];
        //     }
        // }   
    }
    
}

void idynamicarray_iexchange(int_t **data, int_t *ilen, int_t row, int sendrank, int recrank, gridinfo3d_t *grid3d)
{
    int tag = 800;
    MPI_Status status;
    MPI_Request req;
    int_t i;

    if (grid3d->zscp.Iam == sendrank)
    {
        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Isend(data[i] + minstep, len, mpi_int_t, recrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(data[i], ilen[i], mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(data[i] + minstep, len, mpi_int_t, recrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(data[i], ilen[i], mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        }
    }
    else
    {
        int_t **temp_data = (int_t**)SUPERLU_MALLOC(row * sizeof(int_t*));

        for (i = 0; i < row ; ++i)
        {        
            if (ilen[i] > 0)
            {
                temp_data[i] = intCalloc_dist(ilen[i]);
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(temp_data[i] + minstep, len, mpi_int_t, sendrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(temp_data[i], ilen[i], mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        } 

        for (i = 0; i < row ; ++i)
        {            
            
            if (ilen[i] > 0)
            {
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(data[i] + minstep, len, mpi_int_t, sendrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(data[i], ilen[i], mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                SUPERLU_FREE(data[i]);
                data[i] = temp_data[i];
            }
        }   
    }
    
}

void Ufstnz_br_ptr_iexchange(int_t **Ufstnz_br_ptr, int_t *Ufstnz_br_ptr_ilen, int_t row, int sendrank, int recrank, gridinfo3d_t *grid3d)
{
    int tag = 800;
    MPI_Status status;
    MPI_Request req;
    int_t i;

    if (grid3d->zscp.Iam == sendrank)
    {
        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Isend(Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, recrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Ufstnz_br_ptr[i], ilen, mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, recrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(Ufstnz_br_ptr[i], ilen, mpi_int_t, recrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        }
    }
    else
    {
        int_t **temp_Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(row * sizeof(int_t*));

        for (i = 0; i < row ; ++i)
        {        
            if (Ufstnz_br_ptr_ilen[i])
            {
                temp_Ufstnz_br_ptr[i] = intCalloc_dist(Ufstnz_br_ptr_ilen[i]);
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(temp_Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, sendrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(temp_Ufstnz_br_ptr[i], ilen, mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        } 

        for (i = 0; i < row ; ++i)
        {            
            int_t ilen;
            if (grid3d->zscp.Iam)
            {
                ilen = Ufstnz_br_ptr_ilen[i];
            }    
            else
            {
                ilen = Ufstnz_br_ptr_ilen[i] - 1;
            }

            if (ilen > 0)
            {
                if (ilen > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen)
                        {
                            len = ilen - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(Ufstnz_br_ptr[i] + minstep, len, mpi_int_t, sendrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(Ufstnz_br_ptr[i], ilen, mpi_int_t, sendrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                SUPERLU_FREE(Ufstnz_br_ptr[i]);
                Ufstnz_br_ptr[i] = temp_Ufstnz_br_ptr[i];
            }
        }   
    }
    
}

void ddynamicarray_iexchange(double **data, int_t *ilen, int_t row, int sendrank, int recrank, gridinfo3d_t *grid3d)
{
    int tag = 800;
    MPI_Status status;
    MPI_Request req;
    int_t i;    

    if (grid3d->zscp.Iam == sendrank)
    {
        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Isend(data[i] + minstep, len, MPI_DOUBLE, recrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(data[i], ilen[i], MPI_DOUBLE, recrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(data[i] + minstep, len, MPI_DOUBLE, recrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(data[i], ilen[i], MPI_DOUBLE, recrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        }
    }
    else
    {
        double **temp_data = (double**)SUPERLU_MALLOC(row * sizeof(double*));

        for (i = 0; i < row ; ++i)
        {        
            if (ilen[i] > 0)
            {
                temp_data[i] = doubleCalloc_dist(ilen[i]);
            }
        }

        for (i = 0; i < row ; ++i)
        {            
            if (ilen[i] > 0)
            {
                
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {
                            MPI_Irecv(temp_data[i] + minstep, len, MPI_DOUBLE, sendrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Irecv(temp_data[i], ilen[i], MPI_DOUBLE, sendrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                
            }
        } 

        for (i = 0; i < row ; ++i)
        {            
            
            if (ilen[i] > 0)
            {
                if (ilen[i] > MaxBuffer)
                {
                    for (int_t j = 0; j < ilen[i]/MaxBuffer + 1; j++)
                    {
                        int_t minstep = j * MaxBuffer;
                        int_t len = MaxBuffer;
                        if (minstep + len >= ilen[i])
                        {
                            len = ilen[i] - minstep;
                        }
                        
                        if (len)
                        {                            
                            MPI_Isend(data[i] + minstep, len, MPI_DOUBLE, sendrank, tag + i + row * (j + 1), grid3d->zscp.comm, &req);
                            MPI_Wait(&req, &status);
                        }
                        
                    }				
                }
                else
                {                   
                    MPI_Isend(data[i], ilen[i], MPI_DOUBLE, sendrank, tag + i + row, grid3d->zscp.comm, &req);
                    MPI_Wait(&req, &status);
                }
                SUPERLU_FREE(data[i]);
                data[i] = temp_data[i];
            }
        }   
    }
    
}

int dsendAllLUpanelGPU2HOST(dLUstruct_t *LUstruct, dsluGPU_t *sluGPU, int_t nsupers, gridinfo_t* grid)
{
    dLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
    dLocalLU_t *Llu = LUstruct->Llu;

    A_gpu->LnzvalVec_host = (double**)malloc(CEILING(nsupers, grid->npcol) * sizeof(double*));
    A_gpu->isGPUUsed_Lnzval_bc_ptr_host = (int_t*)malloc(CEILING(nsupers, grid->npcol) * sizeof(int_t));
    A_gpu->isCPUUsed_Lnzval_bc_ptr_host = (int_t*)malloc(CEILING(nsupers, grid->npcol) * sizeof(int_t));

    A_gpu->UnzvalVec_host = (double**)malloc(CEILING(nsupers, grid->nprow) * sizeof(double*));
    A_gpu->isGPUUsed_Unzval_br_ptr_host = (int_t*)malloc(CEILING(nsupers, grid->nprow) * sizeof(int_t));
    A_gpu->isCPUUsed_Unzval_br_ptr_host = (int_t*)malloc(CEILING(nsupers, grid->nprow) * sizeof(int_t));

    #pragma omp parallel for
    for(int_t i=0; i<CEILING(nsupers, grid->npcol); i++)
    {
        A_gpu->LnzvalVec_host[i] = doubleMalloc_dist(Llu->Lnzval_bc_ptr_ilen[i]);        
        SetVectorStatus(A_gpu->isGPUUsed_Lnzval_bc_ptr_host[i], GPUUnused);
        SetVectorStatus(A_gpu->isCPUUsed_Lnzval_bc_ptr_host[i], CPUUsed);
        checkCuda(cudaMemcpy(A_gpu->LnzvalVec_host[i], &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[i]], Llu->Lnzval_bc_ptr_ilen[i] * sizeof(double), cudaMemcpyDeviceToHost));
    }
    #pragma omp parallel for
    for(int_t i=0; i<CEILING(nsupers, grid->nprow); i++)
    {       
        A_gpu->UnzvalVec_host[i] = doubleMalloc_dist(Llu->Unzval_br_ptr_ilen[i]); 
        SetVectorStatus(A_gpu->isGPUUsed_Unzval_br_ptr_host[i], GPUUnused);
        SetVectorStatus(A_gpu->isCPUUsed_Unzval_br_ptr_host[i], CPUUsed);
        checkCuda(cudaMemcpy(A_gpu->UnzvalVec_host[i], &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[i]], Llu->Unzval_br_ptr_ilen[i] * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

// int dsendAllLUpanelGPU2HOST(dsluGPU_t *sluGPU, int_t nsupers, gridinfo_t* grid)
// {
//     dLUstruct_gpu_t *A_gpu = sluGPU->A_gpu;
//     #pragma omp parallel for
//     for(int_t i=0; i<CEILING(nsupers, grid->npcol); i++)
//     {        
//         SetVectorStatus(A_gpu->isGPUUsed_Lnzval_bc_ptr_host[i], GPUUsed);
//         SetVectorStatus(A_gpu->isCPUUsed_Lnzval_bc_ptr_host[i], CPUUsed);
//         checkCuda(cudaMemcpy(A_gpu->LnzvalVec_host[i], &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[i]], Llu->Lnzval_bc_ptr_ilen[i] * sizeof(double), cudaMemcpyDeviceToHost));
//     }
//     #pragma omp parallel for
//     for(int_t i=0; i<CEILING(nsupers, grid->nprow); i++)
//     {        
//         SetVectorStatus(A_gpu->isGPUUsed_Unzval_br_ptr_host[i], GPUUsed);
//         SetVectorStatus(A_gpu->isCPUUsed_Unzval_br_ptr_host[i], CPUUsed);
//         checkCuda(cudaMemcpy(A_gpu->UnzvalVec_host[i], &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[i]], Llu->Unzval_br_ptr_ilen[i] * sizeof(double), cudaMemcpyDeviceToHost));
//     }
// }
#endif

#ifdef pdgstrf3d_normal
int_t pdgstrf3d(superlu_dist_options_t *options, int m, int n, double anorm,
		trf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		dLUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info)
{
    gridinfo_t* grid = &(grid3d->grid2d);
    dLocalLU_t *Llu = LUstruct->Llu;

    // problem specific contants
    int_t ldt = sp_ienv_dist (3);     /* Size of maximum supernode */
    //    double s_eps = slamch_ ("Epsilon");  -Sherry
    double s_eps = smach_dist("Epsilon");
    double thresh = s_eps * anorm;    

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter pdgstrf3d()");
#endif

    // Initilize stat
    stat->ops[FACT] = 0;
    stat->current_buffer = 0.0;
    stat->peak_buffer    = 0.0;
    stat->gpu_buffer     = 0.0;
    //if (!grid3d->zscp.Iam && !grid3d->iam) printf("Using NSUP=%d\n", (int) ldt);

    //getting Nsupers
    int_t nsupers = getNsupers(n, LUstruct->Glu_persist);

    // Grid related Variables
    int_t iam = grid->iam; // in 2D grid
    int num_threads = getNumThreads(grid3d->iam);

    factStat_t factStat;
    initFactStat(nsupers, &factStat);

#if 0  // sherry: not used
    ddiagFactBufs_t dFBuf;
    dinitDiagFactBufs(ldt, &dFBuf);

    commRequests_t comReqs;   
    initCommRequests(&comReqs, grid);

    msgs_t msgs;
    initMsgs(&msgs);
#endif

    SCT->tStartup = SuperLU_timer_();
    packLUInfo_t packLUInfo;
    initPackLUInfo(nsupers, &packLUInfo);

    dscuBufs_t scuBufs;
    dinitScuBufs(ldt, num_threads, nsupers, &scuBufs, LUstruct, grid);

    factNodelists_t  fNlists;
    initFactNodelists( ldt, num_threads, nsupers, &fNlists);

    // tag_ub initialization
    int tag_ub = set_tag_ub();
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;    

#if ( PRNTlevel>=1 )
    if (grid3d->iam == 0) {
        printf ("MPI tag upper bound = %d\n", tag_ub); fflush(stdout);
    }
#endif

    // trf3Dpartition_t*  trf3Dpartition = initTrf3Dpartition(nsupers, options, LUstruct, grid3d);
    gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
    int_t* iperm_c_supno = trf3Dpartition->iperm_c_supno;
    int_t* myNodeCount = trf3Dpartition->myNodeCount;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t** treePerm = trf3Dpartition->treePerm ;
    dLUValSubBuf_t *LUvsb = trf3Dpartition->LUvsb;

    /* Initializing factorization specific buffers */

    int_t numLA = getNumLookAhead(options);
    dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);
    msgs_t**msgss = initMsgsArr(numLA);
    int_t mxLeafNode    = 0;
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode )
            mxLeafNode    = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
    }
    ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
    commRequests_t** comReqss = initCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), ldt, grid);

    /* Setting up GPU related data structures */

    int_t first_l_block_acc = 0;
    int_t first_u_block_acc = 0;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t mrb =    (nsupers + Pr - 1) / Pr;
    int_t mcb =    (nsupers + Pc - 1) / Pc;
    HyP_t *HyP = (HyP_t *) SUPERLU_MALLOC(sizeof(HyP_t));

    dInit_HyP(HyP, Llu, mcb, mrb);
    HyP->first_l_block_acc = first_l_block_acc;
    HyP->first_u_block_acc = first_u_block_acc;

    int superlu_acc_offload = HyP->superlu_acc_offload;

    #ifdef AllCPU
    superlu_acc_offload = 0;
    #endif

    //int_t bigu_size = getBigUSize(nsupers, grid, LUstruct);
    int_t bigu_size = getBigUSize(nsupers, grid,
    	  	                  LUstruct->Llu->Lrowind_bc_ptr);
    HyP->bigu_size = bigu_size;
    int_t buffer_size =sp_ienv_dist(8); // get_max_buffer_size ();
    HyP->buffer_size = buffer_size;
    HyP->nsupers = nsupers;

#ifdef GPU_ACC

    /*Now initialize the GPU data structure*/
    dLUstruct_gpu_t *A_gpu, *dA_gpu;

    d2Hreduce_t d2HredObj;
    d2Hreduce_t* d2Hred = &d2HredObj;
    dsluGPU_t sluGPUobj;
    dsluGPU_t *sluGPU = &sluGPUobj;
    sluGPU->isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);
    if (superlu_acc_offload)
    {
#if 0 	/* Sherry: For GPU code on titan, we do not need performance 
	   lookup tables since due to difference in CPU-GPU performance,
	   it didn't make much sense to do any Schur-complement update
	   on CPU, except for the lookahead-update on CPU. Same should
	   hold for summit as well. (from Piyush)   */

        /*Initilize the lookup tables */
        LookUpTableInit(iam);
        acc_async_cost = get_acc_async_cost();
#ifdef GPU_DEBUG
        if (!iam) printf("Using MIC async cost of %lf \n", acc_async_cost);
#endif
#endif

	//OLD: int_t* perm_c_supno = getPerm_c_supno(nsupers, options, LUstruct, grid);
	int_t* perm_c_supno = getPerm_c_supno(nsupers, options,
					      LUstruct->etree,
					      LUstruct->Glu_persist,
					      LUstruct->Llu->Lrowind_bc_ptr,
					      LUstruct->Llu->Ufstnz_br_ptr,
					      grid);

	/* Initialize GPU data structures */
        dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                        n, buffer_size, bigu_size, ldt);

        HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
        HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
        HyP->nCudaStreams = sluGPU->nCudaStreams;
    }

#endif  // end GPU_ACC

    /*====  starting main factorization loop =====*/
    MPI_Barrier( grid3d->comm);
    SCT->tStartup = SuperLU_timer_() - SCT->tStartup;
    // int_t myGrid = grid3d->zscp.Iam;

#ifdef ITAC_PROF
    VT_traceon();
#endif
#ifdef MAP_PROFILE
    allinea_start_sampling();
#endif
    SCT->pdgstrfTimer = SuperLU_timer_();

    #if 0
    num_threads=(double)num_threads/pow(2,maxLvl-1);
    omp_set_num_threads(num_threads);
    #endif   

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            //int_t tree = myTreeIdxs[ilvl];

            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /* main loop over all the supernodes */
            if (sforest) /* 2D factorization at individual subtree */
            {
                double tilvl = SuperLU_timer_();
                if(superlu_acc_offload)
                {
                    dsparseTreeFactor_ASYNC_GPU(
                    sforest,
                    comReqss, &scuBufs,  &packLUInfo,
                    msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                    &gEtreeInfo, options,  iperm_c_supno, ldt,
                    sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                    thresh,  SCT, tag_ub, info);
                }
                else{
                    dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
					msgss, LUvsbs, dFBufs, &factStat, &fNlists,
					&gEtreeInfo, options, iperm_c_supno, ldt,
					HyP, LUstruct, grid3d, stat,
					thresh,  SCT, tag_ub, info );
                }

                /*now reduce the updates*/
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
            }

            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
            {
                if(superlu_acc_offload)
                {
                    dreduceAllAncestors3d_GPU(
                    ilvl, myNodeCount, treePerm, LUvsb,
                    LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                    SCT );
                }    
                else{
                    dreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
                                      LUvsb, LUstruct, grid3d, SCT );
                }

                

            }            

        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/

        SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
	    : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];

        #if 0
        num_threads=num_threads*1.5;
        omp_set_num_threads(num_threads);
        #endif

    } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

    MPI_Barrier( grid3d->comm);
    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;    

#ifdef ITAC_PROF
    VT_traceoff();
#endif

#ifdef MAP_PROFILE
    allinea_stop_sampling();
#endif

    reduceStat(FACT, stat, grid3d);

    // sherry added
    /* Deallocate factorization specific buffers */
    freePackLUInfo(&packLUInfo);
    dfreeScuBufs(&scuBufs);
    freeFactStat(&factStat);
    freeFactNodelists(&fNlists);
    freeMsgsArr(numLA, msgss);
    freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
    dLluBufFreeArr(numLA, LUvsbs);
    dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
    Free_HyP(HyP);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit pdgstrf3d()");
#endif
    return 0;

} /* pdgstrf3d */

#endif

#ifdef BigMem_pdgstrf3d
int_t pdgstrf3d(superlu_dist_options_t *options, int m, int n, double anorm,
		trf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		dLUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info)
{
    gridinfo_t* grid = &(grid3d->grid2d);
    dLocalLU_t *Llu = LUstruct->Llu;

    // problem specific contants
    int_t ldt = sp_ienv_dist (3);     /* Size of maximum supernode */
    //    double s_eps = slamch_ ("Epsilon");  -Sherry
    double s_eps = smach_dist("Epsilon");
    double thresh = s_eps * anorm;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter pdgstrf3d()");
#endif

    // Initilize stat
    stat->ops[FACT] = 0;
    stat->current_buffer = 0.0;
    stat->peak_buffer    = 0.0;
    stat->gpu_buffer     = 0.0;
    //if (!grid3d->zscp.Iam && !grid3d->iam) printf("Using NSUP=%d\n", (int) ldt);

    //getting Nsupers
    int_t nsupers = getNsupers(n, LUstruct->Glu_persist);

    // Grid related Variables
    int_t iam = grid->iam; // in 2D grid
    int num_threads = getNumThreads(grid3d->iam);

    factStat_t factStat;
    initFactStat(nsupers, &factStat);

#if 0  // sherry: not used
    ddiagFactBufs_t dFBuf;
    dinitDiagFactBufs(ldt, &dFBuf);

    commRequests_t comReqs;   
    initCommRequests(&comReqs, grid);

    msgs_t msgs;
    initMsgs(&msgs);
#endif

    SCT->tStartup = SuperLU_timer_();
    packLUInfo_t packLUInfo;
    initPackLUInfo(nsupers, &packLUInfo);

    dscuBufs_t scuBufs;
    dinitScuBufs(ldt, num_threads, nsupers, &scuBufs, LUstruct, grid);

    factNodelists_t  fNlists;
    initFactNodelists( ldt, num_threads, nsupers, &fNlists);

    // tag_ub initialization
    int tag_ub = set_tag_ub();
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;    

#if ( PRNTlevel>=1 )
    if (grid3d->iam == 0) {
        printf ("MPI tag upper bound = %d\n", tag_ub); fflush(stdout);
    }
#endif

    // trf3Dpartition_t*  trf3Dpartition = initTrf3Dpartition(nsupers, options, LUstruct, grid3d);
    gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
    int_t* iperm_c_supno = trf3Dpartition->iperm_c_supno;
    int_t* myNodeCount = trf3Dpartition->myNodeCount;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t** treePerm = trf3Dpartition->treePerm ;
    dLUValSubBuf_t *LUvsb = trf3Dpartition->LUvsb;

    /* Initializing factorization specific buffers */

    int_t numLA = getNumLookAhead(options);
    dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);
    msgs_t**msgss = initMsgsArr(numLA);
    int_t mxLeafNode    = 0;
    for (int j = 0; j < maxLvl; ++j) {
        if (sForests[myTreeIdxs[j]] && sForests[myTreeIdxs[j]]->topoInfo.eTreeTopLims[1] > mxLeafNode )
            mxLeafNode    = sForests[myTreeIdxs[j]]->topoInfo.eTreeTopLims[1];
    }
    ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
    commRequests_t** comReqss = initCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), ldt, grid);

    /* Setting up GPU related data structures */

    int_t first_l_block_acc = 0;
    int_t first_u_block_acc = 0;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t mrb =    (nsupers + Pr - 1) / Pr;
    int_t mcb =    (nsupers + Pc - 1) / Pc;
    HyP_t *HyP = (HyP_t *) SUPERLU_MALLOC(sizeof(HyP_t));

    dInit_HyP(HyP, Llu, mcb, mrb);
    HyP->first_l_block_acc = first_l_block_acc;
    HyP->first_u_block_acc = first_u_block_acc;

    int superlu_acc_offload = HyP->superlu_acc_offload;

    //int_t bigu_size = getBigUSize(nsupers, grid, LUstruct);
    int_t bigu_size = getBigUSize(nsupers, grid,
                            LUstruct->Llu->Lrowind_bc_ptr);
    HyP->bigu_size = bigu_size;
    int_t buffer_size =sp_ienv_dist(8); // get_max_buffer_size ();
    HyP->buffer_size = buffer_size;
    HyP->nsupers = nsupers;

    /*====  starting main factorization loop =====*/
    MPI_Barrier( grid3d->comm);
    SCT->tStartup = SuperLU_timer_() - SCT->tStartup;
    // int_t myGrid = grid3d->zscp.Iam;

    #ifdef ITAC_PROF
        VT_traceon();
    #endif
    #ifdef MAP_PROFILE
        allinea_start_sampling();
    #endif
        SCT->pdgstrfTimer = SuperLU_timer_();

    MPI_Status status;

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        #if 0
        num_threads=(double)num_threads/pow(2,maxLvl-1);
        omp_set_num_threads(num_threads);
        #endif

        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            //int_t tree = myTreeIdxs[ilvl];

                sForest_t* sforest = sForests[myTreeIdxs[ilvl]];            

                int flag=0;

                if(grid->zscp.Iam){

                    /* main loop over all the supernodes */
                    if (sforest) /* 2D factorization at individual subtree */
                    {
                        
                        MPI_Recv(&flag,1,MPI_INT,0,0,grid3d->zscp.comm,&status);

                        // Send data to grid3d->zscp.Iam=1 for dinitSluGPU3D_t()
                        //***code***//
                    }

                }
                else{

                    

                    for(int i=0;i<grid3d->zscp.Np;i++)
                    {

                        /* main loop over all the supernodes */
                        if (sforest) /* 2D factorization at individual subtree */
                        {

#ifdef GPU_ACC

                            /*Now initialize the GPU data structure*/
                            dLUstruct_gpu_t *A_gpu, *dA_gpu;

                            d2Hreduce_t d2HredObj;
                            d2Hreduce_t* d2Hred = &d2HredObj;
                            dsluGPU_t sluGPUobj;
                            dsluGPU_t *sluGPU = &sluGPUobj;
                            sluGPU->isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);
                            if (superlu_acc_offload)
                            {
                                #if 0 	/* Sherry: For GPU code on titan, we do not need performance 
                                    lookup tables since due to difference in CPU-GPU performance,
                                    it didn't make much sense to do any Schur-complement update
                                    on CPU, except for the lookahead-update on CPU. Same should
                                    hold for summit as well. (from Piyush)   */

                                        /*Initilize the lookup tables */
                                        LookUpTableInit(iam);
                                        acc_async_cost = get_acc_async_cost();
                                #ifdef GPU_DEBUG
                                        if (!iam) printf("Using MIC async cost of %lf \n", acc_async_cost);
                                #endif
                                #endif

                                //OLD: int_t* perm_c_supno = getPerm_c_supno(nsupers, options, LUstruct, grid);
                                int_t* perm_c_supno = getPerm_c_supno(nsupers, options,
                                                    LUstruct->etree,
                                                    LUstruct->Glu_persist,
                                                    LUstruct->Llu->Lrowind_bc_ptr,
                                                    LUstruct->Llu->Ufstnz_br_ptr,
                                                    grid);

                                /* Initialize GPU data structures */
                                    dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                                    n, buffer_size, bigu_size, ldt);

                                    HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
                                    HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
                                    HyP->nCudaStreams = sluGPU->nCudaStreams;
                            }
#endif  // end GPU_ACC

#ifdef GPU_ACC
                            dsparseTreeFactor_ASYNC_GPU(
                                sforest,
                                comReqss, &scuBufs,  &packLUInfo,
                                msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                                &gEtreeInfo, options,  iperm_c_supno, ldt,
                                sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                                thresh,  SCT, tag_ub, info);
#else
                            dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                                msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                                &gEtreeInfo, options, iperm_c_supno, ldt,
                                HyP, LUstruct, grid3d, stat,
                                thresh,  SCT, tag_ub, info );
#endif
                            }
                    }

                    

                    

                /* if I participate in this level */
                if (!myZeroTrIdxs[ilvl])
                {
                    //int_t tree = myTreeIdxs[ilvl];

                    sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

                    /* main loop over all the supernodes */
                    if (sforest) /* 2D factorization at individual subtree */
                    {
                        double tilvl = SuperLU_timer_();
        #ifdef GPU_ACC
                        dsparseTreeFactor_ASYNC_GPU(
                            sforest,
                            comReqss, &scuBufs,  &packLUInfo,
                            msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                            &gEtreeInfo, options,  iperm_c_supno, ldt,
                            sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                            thresh,  SCT, tag_ub, info);
        #else
                        dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                            msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                            &gEtreeInfo, options, iperm_c_supno, ldt,
                            HyP, LUstruct, grid3d, stat,
                            thresh,  SCT, tag_ub, info );
        #endif

                        /*now reduce the updates*/
                        SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                        sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
                    }

                }
            
            
        }

    

    



    

      

    
        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            //int_t tree = myTreeIdxs[ilvl];

            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

            /* main loop over all the supernodes */
            if (sforest) /* 2D factorization at individual subtree */
            {
                double tilvl = SuperLU_timer_();
#ifdef GPU_ACC
                dsparseTreeFactor_ASYNC_GPU(
                    sforest,
                    comReqss, &scuBufs,  &packLUInfo,
                    msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                    &gEtreeInfo, options,  iperm_c_supno, ldt,
                    sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                    thresh,  SCT, tag_ub, info);
#else
                dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
					msgss, LUvsbs, dFBufs, &factStat, &fNlists,
					&gEtreeInfo, options, iperm_c_supno, ldt,
					HyP, LUstruct, grid3d, stat,
					thresh,  SCT, tag_ub, info );
#endif

                /*now reduce the updates*/
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
            }

            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
            {
#ifdef GPU_ACC
                dreduceAllAncestors3d_GPU(
                    ilvl, myNodeCount, treePerm, LUvsb,
                    LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                    SCT );
#else

                dreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
                                      LUvsb, LUstruct, grid3d, SCT );
#endif

            }            

        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/

        SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
	    : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];

        #if 0
        num_threads=num_threads*1.5;
        omp_set_num_threads(num_threads);
        #endif

    } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

    MPI_Barrier( grid3d->comm);
    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer;    

#ifdef ITAC_PROF
    VT_traceoff();
#endif

#ifdef MAP_PROFILE
    allinea_stop_sampling();
#endif

    reduceStat(FACT, stat, grid3d);

    // sherry added
    /* Deallocate factorization specific buffers */
    freePackLUInfo(&packLUInfo);
    dfreeScuBufs(&scuBufs);
    freeFactStat(&factStat);
    freeFactNodelists(&fNlists);
    freeMsgsArr(numLA, msgss);
    freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
    dLluBufFreeArr(numLA, LUvsbs);
    dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
    Free_HyP(HyP);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit pdgstrf3d()");
#endif
    return 0;

} /* pdgstrf3d */

#endif

#ifdef pdgstrf3d_cpugpu
#define GPU_ACC
#endif

#ifdef pdgstrf3d_cpugpu
int_t pdgstrf3d(superlu_dist_options_t *options, int m, int n, double anorm,
		trf3Dpartition_t*  trf3Dpartition, SCT_t *SCT,
		dLUstruct_t *LUstruct, gridinfo3d_t * grid3d,
		SuperLUStat_t *stat, int *info)
{
    gridinfo_t* grid = &(grid3d->grid2d);
    dLocalLU_t *Llu = LUstruct->Llu;

    // problem specific contants
    int_t ldt = sp_ienv_dist (3);     /* Size of maximum supernode */
    //    double s_eps = slamch_ ("Epsilon");  -Sherry
    #ifndef Torch_dmach_dist
    double s_eps = smach_dist("Epsilon");
    #else
    double s_eps = dmach_dist("Epsilon");
    #endif

    #ifdef Torch_dmach_dist
    s_eps = 1e-6;
    #endif

    double thresh = s_eps * anorm;
    
    #ifdef Torch_dmach_dist
    printf("thresh %e, s_eps %e, anorm %e\n", thresh, s_eps ,anorm);
    #endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Enter pdgstrf3d()");
#endif

    // Initilize stat
    stat->ops[FACT] = 0;
    stat->current_buffer = 0.0;
    stat->peak_buffer    = 0.0;
    stat->gpu_buffer     = 0.0;
    //if (!grid3d->zscp.Iam && !grid3d->iam) printf("Using NSUP=%d\n", (int) ldt);

    //getting Nsupers
    int_t nsupers = getNsupers(n, LUstruct->Glu_persist);

    // Grid related Variables
    int_t iam = grid->iam; // in 2D grid
    int num_threads = getNumThreads(grid3d->iam);

    factStat_t factStat;
    initFactStat(nsupers, &factStat);

#if 0  // sherry: not used
    ddiagFactBufs_t dFBuf;
    dinitDiagFactBufs(ldt, &dFBuf);

    commRequests_t comReqs;   
    initCommRequests(&comReqs, grid);

    msgs_t msgs;
    initMsgs(&msgs);
#endif

    SCT->tStartup = SuperLU_timer_();
    packLUInfo_t packLUInfo;
    initPackLUInfo(nsupers, &packLUInfo);

    dscuBufs_t scuBufs;
    dinitScuBufs(ldt, num_threads, nsupers, &scuBufs, LUstruct, grid);

    factNodelists_t  fNlists;
    initFactNodelists( ldt, num_threads, nsupers, &fNlists);

    // tag_ub initialization
    int tag_ub = set_tag_ub();
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;    

#if ( PRNTlevel>=1 )
    if (grid3d->iam == 0) {
        printf ("MPI tag upper bound = %d\n", tag_ub); fflush(stdout);
    }
#endif

    // trf3Dpartition_t*  trf3Dpartition = initTrf3Dpartition(nsupers, options, LUstruct, grid3d);
    gEtreeInfo_t gEtreeInfo = trf3Dpartition->gEtreeInfo;
    int_t* iperm_c_supno = trf3Dpartition->iperm_c_supno;
    int_t* myNodeCount = trf3Dpartition->myNodeCount;
    int_t* myTreeIdxs = trf3Dpartition->myTreeIdxs;
    int_t* myZeroTrIdxs = trf3Dpartition->myZeroTrIdxs;
    sForest_t** sForests = trf3Dpartition->sForests;
    int_t** treePerm = trf3Dpartition->treePerm ;
    dLUValSubBuf_t *LUvsb = trf3Dpartition->LUvsb;

    /* Initializing factorization specific buffers */

    int_t numLA = getNumLookAhead(options);
    #ifndef SuperLargeScaleGPUBuffer
    dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);
    #endif
    msgs_t**msgss = initMsgsArr(numLA);
    int_t mxLeafNode    = 0;
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode )
            mxLeafNode    = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
    }  

    #ifndef SuperLargeScaleGPUBuffer
    ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
    #endif
    commRequests_t** comReqss = initCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), ldt, grid);
    /* Setting up GPU related data structures */

    int_t first_l_block_acc = 0;
    int_t first_u_block_acc = 0;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t mrb =    (nsupers + Pr - 1) / Pr;
    int_t mcb =    (nsupers + Pc - 1) / Pc;
    HyP_t *HyP = (HyP_t *) SUPERLU_MALLOC(sizeof(HyP_t));

    dInit_HyP(HyP, Llu, mcb, mrb);
    HyP->first_l_block_acc = first_l_block_acc;
    HyP->first_u_block_acc = first_u_block_acc; 

    int superlu_acc_offload = HyP->superlu_acc_offload;

    #ifdef AllCPU
    superlu_acc_offload = 0;
    #endif

    #ifdef CPU_GPU
    if (grid3d->zscp.Iam)
    {        
        superlu_acc_offload = 0;
    }
    else
    {
        superlu_acc_offload = 1;
    }
    
    
    #endif

    #ifdef AllGPU
    superlu_acc_offload = 1;
    #endif

    #ifdef SuperLargeScaleGPU
    if(LUstruct->Llu->isRecordingForGPU == RecordingForGPU){
        superlu_acc_offload = 0;
    }
    #endif

    //int_t bigu_size = getBigUSize(nsupers, grid, LUstruct);
    int_t bigu_size = getBigUSize(nsupers, grid,
    	  	                  LUstruct->Llu->Lrowind_bc_ptr);
    HyP->bigu_size = bigu_size;
    int_t buffer_size =sp_ienv_dist(8); // get_max_buffer_size ();
    HyP->buffer_size = buffer_size;
    HyP->nsupers = nsupers;

    //OLD: int_t* perm_c_supno = getPerm_c_supno(nsupers, options, LUstruct, grid);
	int_t* perm_c_supno = getPerm_c_supno(nsupers, options,
					      LUstruct->etree,
					      LUstruct->Glu_persist,
					      LUstruct->Llu->Lrowind_bc_ptr,
					      LUstruct->Llu->Ufstnz_br_ptr,
					      grid);    
    
    #ifdef Torch
    d2Hreduce_t* d2Hred;
    dsluGPU_t *sluGPU;
    d2Hreduce_t d2HredObj;
    d2Hred = &d2HredObj;
    dsluGPU_t sluGPUobj;
    sluGPU = &sluGPUobj;
    #endif
    
    if (superlu_acc_offload)
    {

        /*Now initialize the GPU data structure*/
        dLUstruct_gpu_t *A_gpu, *dA_gpu;
        #ifndef Torch
        d2Hreduce_t* d2Hred;
        dsluGPU_t *sluGPU;
        d2Hreduce_t d2HredObj;
        d2Hred = &d2HredObj;
        dsluGPU_t sluGPUobj;
        sluGPU = &sluGPUobj;
        #endif
        
        sluGPU->isNodeInMyGrid = getIsNodeInMyGrid(nsupers, maxLvl, myNodeCount, treePerm);       
        
    
        #if 0 	
        /* Sherry: For GPU code on titan, we do not need performance 
            lookup tables since due to difference in CPU-GPU performance,
            it didn't make much sense to do any Schur-complement update
            on CPU, except for the lookahead-update on CPU. Same should
            hold for summit as well. (from Piyush)   */

                /*Initilize the lookup tables */
                LookUpTableInit(iam);
                acc_async_cost = get_acc_async_cost();
        #ifdef GPU_DEBUG
                if (!iam) printf("Using MIC async cost of %lf \n", acc_async_cost);
        #endif
        #endif	
        #ifndef SuperLargeScaleGPUBuffer
	/* Initialize GPU data structures */
        dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                        n, buffer_size, bigu_size, ldt);

        HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
        HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
        HyP->nCudaStreams = sluGPU->nCudaStreams;
        #endif
    }

    #ifdef SuperLargeScaleGPUBuffer
    #ifdef Torch_0419_Case2
    if (!grid3d->zscp.Iam)
    {
        Llu->isEmpty = 2;
    }
    else{
        Llu->isEmpty = 1;
    }        
    #else
    Llu->isEmpty = 0;
    #endif
    #endif

    /*====  starting main factorization loop =====*/
    MPI_Barrier( grid3d->comm);
    SCT->tStartup = SuperLU_timer_() - SCT->tStartup;
    // int_t myGrid = grid3d->zscp.Iam;   
    

#ifdef ITAC_PROF
    VT_traceon();
#endif
#ifdef MAP_PROFILE
    allinea_start_sampling();
#endif
    SCT->pdgstrfTimer = SuperLU_timer_();

    #if 0
    num_threads=(double)num_threads/pow(2,maxLvl-1);
    omp_set_num_threads(num_threads);
    #endif

    #ifdef SuperLargeScale
    Llu->core_status = OutOfCore;            
    #endif     

    #ifdef SuperLargeScaleGPU
    if (superlu_acc_offload == 0 && Llu->isRecordingForGPU == RecordingForGPU)
    {
        Llu->isUsed_Lnzval_bc_ptr_Record = intMalloc_dist(CEILING(nsupers, grid->npcol));
        Llu->isUsed_Unzval_br_ptr_Record = intMalloc_dist(CEILING(nsupers, grid->nprow));
        #pragma omp parallel for
        for(int_t i=0; i<CEILING(nsupers, grid->npcol); i++)
        {        
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr_Record[i], Unused);
        }
        #pragma omp parallel for
        for(int_t i=0; i<CEILING(nsupers, grid->nprow); i++)
        {        
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr_Record[i], Unused);
        }
        Llu->MaxGPUMemory = 0;
    }
    if (superlu_acc_offload && Llu->isRecordingForGPU == LoadedRecordForGPU)
    {
        
        sluGPU->A_gpu->nexttopoLvl_Lnzval = -1;
        sluGPU->A_gpu->nexttopoLvl_Unzval = -1;
        sluGPU->A_gpu->nextk0_Lnzval = -1;
        sluGPU->A_gpu->nextk0_Unzval = -1;
        sluGPU->A_gpu->pretopoLvl_Lnzval = 0;
        sluGPU->A_gpu->pretopoLvl_Unzval = 0;
        sluGPU->A_gpu->pre_lrecordid = 1;
        sluGPU->A_gpu->pre_urecordid = 1;
        
        #pragma omp parallel for
        for(int_t i=0; i<CEILING(nsupers, grid->npcol); i++)
        {        
            SetVectorStatus(sluGPU->A_gpu->isGPUUsed_Lnzval_bc_ptr_host[i], GPUUnused);
            (sluGPU->A_gpu->isCPUUsed_Lnzval_bc_ptr_host[i], CPUUnused);
        }
        #pragma omp parallel for
        for(int_t i=0; i<CEILING(nsupers, grid->nprow); i++)
        {        
            SetVectorStatus(sluGPU->A_gpu->isGPUUsed_Unzval_br_ptr_host[i], GPUUnused);
            SetVectorStatus(sluGPU->A_gpu->isCPUUsed_Unzval_br_ptr_host[i], CPUUnused);
        }
        
        load_RecordMatrix_txt(grid3d, LUstruct, &(sluGPU->A_gpu->nexttopoLvl_Lnzval), &(sluGPU->A_gpu->nexttopoLvl_Unzval), &(sluGPU->A_gpu->nextk0_Lnzval), &(sluGPU->A_gpu->nextk0_Unzval), sluGPU->A_gpu->LnzvalPtr_host, sluGPU->A_gpu->UnzvalPtr_host, sluGPU->A_gpu->isGPUUsed_Lnzval_bc_ptr_host, sluGPU->A_gpu->isGPUUsed_Unzval_br_ptr_host, &(sluGPU->A_gpu->Lnzval_bc_ptr_len), &(sluGPU->A_gpu->Unzval_br_ptr_len), sluGPU->A_gpu->UsedOrder_Lnzval, sluGPU->A_gpu->UsedOrder_Unzval);
        printf("break2\n");
        checkCudaErrors(cudaMemcpy( (sluGPU->A_gpu->LnzvalPtr), sluGPU->A_gpu->LnzvalPtr_host, CEILING(nsupers, grid->npcol) * sizeof(int_t), cudaMemcpyHostToDevice)) ;
        checkCudaErrors(cudaMemcpy( (sluGPU->A_gpu->UnzvalPtr), sluGPU->A_gpu->UnzvalPtr_host, CEILING(nsupers, grid->nprow) * sizeof(int_t), cudaMemcpyHostToDevice)) ;
        // printf("break2\n");

        // save_RecordMatrix_txt(grid3d, LUstruct);
        // while (1)
        // {
        //     /* code */
        // }
        
    } 

    #endif

    #ifdef SuperLargeScaleGPUBuffer
    int_t numForests = (1 << maxLvl) - 1;
    MPI_Status status;
    #endif

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        #ifndef SuperLargeScaleGPUBuffer
        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            //int_t tree = myTreeIdxs[ilvl];
            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];            

            /* main loop over all the supernodes */
            if (sforest) /* 2D factorization at individual subtree */
            {
                #ifdef Torch 
                
                int_t nb = CEILING(nsupers, grid->npcol);
                // for(int_t i=0; i<nb; i++)
                // {                            
                //     for (int_t j = 0; j < nPart; j++)
                //     {
                //         SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[i][j], Unused);
                //     }                    
                // }

                // nb = CEILING(nsupers, grid->nprow);
                // for(int_t i=0; i<nb; i++)
                // {                    
                //     for (int_t j = 0; j < nPart; j++)
                //     {
                //         SetVectorStatus(Llu->isUsed_Unzval_br_ptr[i][j], Unused);
                //     }
                // }

                #ifdef SuperLargeScaleGPU
                Llu->ilvl = ilvl;
                #endif

                #endif

                double tilvl = SuperLU_timer_();

                if(superlu_acc_offload){

                    #ifdef SuperLargeScaleGPUBuffer
                    if (ilvl == 0)
                    {
                       
                    /* Initialize GPU data structures */
                        dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                        n, buffer_size, bigu_size, ldt);

                        HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
                        HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
                        HyP->nCudaStreams = sluGPU->nCudaStreams;
                    }
                    #endif
                    
                    dsparseTreeFactor_ASYNC_GPU(
                        sforest,
                        comReqss, &scuBufs,  &packLUInfo,
                        msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                        &gEtreeInfo, options,  iperm_c_supno, ldt,
                        sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                        thresh,  SCT, tag_ub, info);
                }
                
                else{
                    dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                        msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                        &gEtreeInfo, options, iperm_c_supno, ldt,
                        HyP, LUstruct, grid3d, stat,
                        thresh,  SCT, tag_ub, info );
                }

                /*now reduce the updates*/
                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];

                #ifdef Torch_stat
                printf("%d: tFactor3D[%d] = %e s\n", grid3d->iam, ilvl, SCT->tFactor3D[ilvl]);
                #endif
            }
            else
            {
                #ifdef SuperLargeScaleGPU
                Llu->ilvl = ilvl;
                if (Llu->isRecordingForGPU == RecordingForGPU)
                {
                    Llu->Lnzval_RecordMatrix[Llu->ilvl] = (int_t***)SUPERLU_MALLOC(sizeof(int_t**));
                    int_t **LRecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
                    int_t *iLnzval_RecordMatrix = intMalloc_dist(1);
                    iLnzval_RecordMatrix[0] = 1;
                    
                    LRecordMatrix[0] = iLnzval_RecordMatrix;
                    Llu->Lnzval_RecordMatrix[Llu->ilvl][0] = LRecordMatrix;

                    Llu->Unzval_RecordMatrix[Llu->ilvl] = (int_t***)SUPERLU_MALLOC(sizeof(int_t**));
                    int_t **URecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
                    int_t *iUnzval_RecordMatrix = intMalloc_dist(1);
                    iUnzval_RecordMatrix[0] = 1;
                    
                    URecordMatrix[0] = iUnzval_RecordMatrix;
                    Llu->Unzval_RecordMatrix[Llu->ilvl][0] = URecordMatrix;
                }
                #endif
            }
            

            #ifdef SuperLargeScale
            Llu->core_status = OutOfCore;
            #endif
            
            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
            {
                if(superlu_acc_offload){
                    dreduceAllAncestors3d_GPU(
                        ilvl, myNodeCount, treePerm, LUvsb,
                        LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                        SCT );

                    #ifdef SuperLargeScaleGPUBuffer
                    dreduceAllAncestors3d(ilvl, myNodeCount, treePerm, 	                      LUvsb, LUstruct, grid3d, SCT );
                    #endif
                }
                else{

                    dreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
                                        LUvsb, LUstruct, grid3d, SCT );
                }

            }       

        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/
        else
        {            
            #ifdef SuperLargeScaleGPU
            Llu->ilvl = ilvl;

            if (Llu->isRecordingForGPU == RecordingForGPU)
            {
                Llu->Lnzval_RecordMatrix[Llu->ilvl] = (int_t***)SUPERLU_MALLOC(sizeof(int_t**));
                int_t **LRecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
                int_t *iLnzval_RecordMatrix = intMalloc_dist(1);
                iLnzval_RecordMatrix[0] = 1;
                
                LRecordMatrix[0] = iLnzval_RecordMatrix;
                Llu->Lnzval_RecordMatrix[Llu->ilvl][0] = LRecordMatrix;

                Llu->Unzval_RecordMatrix[Llu->ilvl] = (int_t***)SUPERLU_MALLOC(sizeof(int_t**));
                int_t **URecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
                int_t *iUnzval_RecordMatrix = intMalloc_dist(1);
                iUnzval_RecordMatrix[0] = 1;
                
                URecordMatrix[0] = iUnzval_RecordMatrix;
                Llu->Unzval_RecordMatrix[Llu->ilvl][0] = URecordMatrix;
            }
            #endif
        }

        #else        

        if (ilvl == 0)
        {
            
            for (int i = grid3d->npdep - 1; i >= 0; i--)
            {
                MPI_Barrier(grid3d->comm);
                            
                if (i > 0)
                {
                    if ((i == grid3d->zscp.Iam && !Llu->isEmpty)  || (grid3d->zscp.Iam == 0 && !Llu->isEmpty))
                    {
                        if (grid3d->zscp.Iam == 0)
                        {
                            continue;
                        }
                        
                        if(superlu_acc_offload){

                            #ifdef Torch_batch_init

                            for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                            {
                                MPI_Barrier(grid->comm);
                                if (j == grid->iam)
                                {
                                    /* Initialize GPU data structures */
                                    dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                                    n, buffer_size, bigu_size, ldt);
                                }
                                
                            }

                            #else
                            /* Initialize GPU data structures */
                            dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                            n, buffer_size, bigu_size, ldt);
                            #endif

                            HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
                            HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
                            HyP->nCudaStreams = sluGPU->nCudaStreams;
                            sluGPU->A_gpu->isEmpty = 0;
                        }
                            
                        
                        /* if I participate in this level */
                        if (!myZeroTrIdxs[ilvl])
                        {
                            //int_t tree = myTreeIdxs[ilvl];
                            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];            

                            /* main loop over all the supernodes */
                            if (sforest) /* 2D factorization at individual subtree */
                            {                
                                double tilvl = SuperLU_timer_();
                                ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
                                dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);

                                if(superlu_acc_offload){
                                    
                                    dsparseTreeFactor_ASYNC_GPU(
                                        sforest,
                                        comReqss, &scuBufs,  &packLUInfo,
                                        msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                                        &gEtreeInfo, options,  iperm_c_supno, ldt,
                                        sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                                        thresh,  SCT, tag_ub, info);
                                }
                                
                                else{
                                    dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                                        msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                                        &gEtreeInfo, options, iperm_c_supno, ldt,
                                        HyP, LUstruct, grid3d, stat,
                                        thresh,  SCT, tag_ub, info );
                                }

                                freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
                                dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
                                dLluBufFreeArr(numLA, LUvsbs);

                                /*now reduce the updates*/
                                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];

                                #ifdef Torch_stat
                                printf("%d: tFactor3D[%d] = %e s\n", grid3d->iam, ilvl, SCT->tFactor3D[ilvl]);
                                #endif
                            }
                            
                            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                            {
                                if(superlu_acc_offload){
                                    
                                    dreduceAllAncestors3d_GPU(
                                        ilvl, myNodeCount, treePerm, LUvsb,
                                        LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                                        SCT );
                                    
                                    if (maxLvl > 2)
                                    {
                                        dsendAllLUpanelGPU2HOST(LUstruct, sluGPU, nsupers,  grid);
                                    }
                                    
                                    if (maxLvl == 2) {
                                        dfree_LUstruct_gpu(sluGPU->A_gpu);
                                        dfreeScuBufs(&scuBufs);
                                    }
                                    else
                                    {
                                        dfree_LUstruct_gpu_Buffer(sluGPU->A_gpu);
                                    }
                                    
                                }                        

                            }                            

                            freePackLUInfo(&packLUInfo);
                            freeFactStat(&factStat);
                            freeFactNodelists(&fNlists);       

                        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/
                    }

                    if ((i == grid3d->zscp.Iam && Llu->isEmpty) || grid3d->zscp.Iam == 0)
                    {
                        int_t nNodes;

                        if (Llu->isEmpty == 1)
                        {

                            MPI_Send(&myZeroTrIdxs[ilvl], 1, mpi_int_t, 0, 100, grid3d->zscp.comm);

                            MPI_Send(myTreeIdxs, maxLvl, mpi_int_t, 0, 101, grid3d->zscp.comm);

                            if (!myZeroTrIdxs[ilvl])
                            {  
                                // sForests
                                int_t *sForests_nNodes = intCalloc_dist(numForests);

                                for (int j = 0; j < numForests; ++j)
                                {
                                    if (sForests[j])
                                    {
                                        sForests_nNodes[j] = sForests[j]->nNodes;
                                    }                                
                                }
                                MPI_Send(sForests_nNodes, numForests, mpi_int_t, 0, 102, grid3d->zscp.comm);
                                
                                for (int j = 0; j < numForests; ++j)
                                {
                                    if (sForests[j])
                                    {
                                        dBcastRecv_sforest(sForests[j], i, 0, grid3d, nsupers);
                                    }
                                }

                                SUPERLU_FREE(sForests_nNodes);
                                sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

                                if(superlu_acc_offload){
                                    int tag = 400;

                                    // for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                                    // {
                                    //     // LUstruct
                                    //     MPI_Barrier(grid->comm);
                                    //     if (j == grid->iam)
                                    //     {
                                            /* code */
                                            dBcastRecv_LUstruct(LUstruct, i, 0, grid3d, nsupers);
                                    //     }                                       
                                        
                                    // }                                  
                                    

                                    // perm_c_supno
                                    MPI_Send(perm_c_supno, nsupers, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                    // bigu_size
                                    MPI_Send(&bigu_size, 1, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                    // sluGPU                                    
                                    MPI_Send(sluGPU->isNodeInMyGrid, nsupers, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                }

                                /* main loop over all the supernodes */
                                if (sforest) /* 2D factorization at individual subtree */
                                {
                                    int tag = 600;

                                    // comReqss
                                    MPI_Send(&mxLeafNode, 1, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                    // bigu_size
                                    MPI_Send(&bigu_size, 1, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                    // scuBufs
                                    MPI_Send(scuBufs.bigV, 8 * ldt * ldt * num_threads, MPI_DOUBLE, 0, tag++, grid3d->zscp.comm);
                                    MPI_Send(scuBufs.bigU, bigu_size, MPI_DOUBLE, 0, tag++, grid3d->zscp.comm);

                                    // gEtreeInfo
                                    MPI_Send(gEtreeInfo.setree, nsupers, mpi_int_t, 0, tag++, grid3d->zscp.comm);
                                    MPI_Send(gEtreeInfo.numChildLeft, nsupers, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                    // iperm_c_supno
                                    MPI_Send(iperm_c_supno, nsupers, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                    // scuBufs
                                    // MPI_Recv(scuBufs.bigV, 8 * ldt * ldt * num_threads, MPI_DOUBLE, 0, tag++, grid3d->zscp.comm, &status);
                                    // MPI_Recv(scuBufs.bigU, bigu_size, MPI_DOUBLE, 0, tag++, grid3d->zscp.comm, &status);
                                    
                                }

                                if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                                {
                                    if(superlu_acc_offload){
                                        int tag = 700;
                                        // myNodeCount
                                        MPI_Send(myNodeCount, maxLvl, mpi_int_t, 0, tag++, grid3d->zscp.comm);

                                        // myGrid
                                        MPI_Send(&(Llu->isEmpty), 1, MPI_INT, 0, tag++, grid3d->zscp.comm);

                                        // iam
                                        MPI_Send(&(grid3d->zscp.Iam), 1, MPI_INT, 0, tag++, grid3d->zscp.comm);
                                        
                                    }
                                }                                

                                // for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                                // {
                                //     MPI_Barrier(grid->comm);
                                //     if (j == grid->iam)
                                //     {                                        
                                        // exchange LUStruct
                                        dBcastRecv_LUstruct1(LUstruct, 0, i, grid3d, nsupers);
                                //     }
                                // }
                            }
                            MPI_Barrier(grid->comm);
                            
                        }
                        else
                        {
                            // grid3d->zscp.Iam == 0
                            int_t myZeroTrIdxs_ilvl;
                            MPI_Recv(&myZeroTrIdxs_ilvl, 1, mpi_int_t, i, 100, grid3d->zscp.comm, &status);

                            HyP_t *HyP_temp = (HyP_t *) SUPERLU_MALLOC(sizeof(HyP_t));
                            dLUstruct_t LUstruct_temp;                            

                            d2Hreduce_t* d2Hred_temp;
                            dsluGPU_t *sluGPU_temp;

                            int_t* myTreeIdxs_temp = (int_t*) SUPERLU_MALLOC (maxLvl * sizeof (int_t));
                            MPI_Recv(myTreeIdxs_temp, maxLvl, mpi_int_t, i, 101, grid3d->zscp.comm, &status);

                            #ifdef Torch_stat
                            double time_exchange = SuperLU_timer_();

                            SCT->tExchange[ilvl] = 0;
                            #endif

                            if (!myZeroTrIdxs_ilvl)
                            {                                

                                // sForests
                                int_t *sForests_nNodes = intCalloc_dist(numForests);
                                MPI_Recv(sForests_nNodes, numForests, mpi_int_t, i, 102, grid3d->zscp.comm, &status);
                                
                                sForest_t **sForests_temp = (sForest_t**)SUPERLU_MALLOC (sizeof (sForest_t*) * numForests);    
                                
                                for (int j = 0; j < numForests; ++j)
                                {
                                    sForest_t *sforest_temp = NULL;
                                    if (sForests_nNodes[j])
                                    {
                                        sforest_temp = SUPERLU_MALLOC (sizeof (sForest_t));
                                        dBcastRecv_sforest(sforest_temp, i, 0, grid3d, nsupers);
                                    }
                                    sForests_temp[j] = sforest_temp;
                                }

                                SUPERLU_FREE(sForests_nNodes);

                                sForest_t *sforest_temp = sForests_temp[myTreeIdxs_temp[ilvl]];                                

                                int_t *perm_c_supno_temp;  
                                d2Hreduce_t d2HredObj_temp;
                                d2Hred_temp = &d2HredObj_temp;
                                dsluGPU_t sluGPUobj_temp;
                                sluGPU_temp = &sluGPUobj_temp;                

                                if(superlu_acc_offload){

                                    int tag = 400;                                    

                                    // LUstruct
                                    dLUstructInit (grid3d->npcol, &LUstruct_temp);
                                    backup_LUstruct(LUstruct_temp.Llu, Llu, grid3d, nsupers);

                                    // for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                                    // {
                                    //     // LUstruct
                                    //     MPI_Barrier(grid->comm);
                                    //     if (j == grid->iam)
                                    //     {
                                            dBcastRecv_LUstruct(&LUstruct_temp, i, 0, grid3d, nsupers);
                                    //     }
                                        
                                    // }    

                                    // perm_c_supno
                                    perm_c_supno_temp  = SUPERLU_MALLOC( nsupers * sizeof(int_t) );
                                    MPI_Recv(perm_c_supno_temp, nsupers, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                    // bigu_size
                                    int_t bigu_size_temp;
                                    MPI_Recv(&bigu_size_temp, 1, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                    // sluGPU
                                    sluGPU_temp->isNodeInMyGrid = INT_T_ALLOC (nsupers);
                                    MPI_Recv(sluGPU_temp->isNodeInMyGrid, nsupers, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                    // HyP
                                    dInit_HyP(HyP_temp, LUstruct_temp.Llu, mcb, mrb);
                                    
                                    #ifdef Torch_batch_init
                                    for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                                    {
                                        MPI_Barrier(grid->comm);
                                        if (j == grid->iam)
                                        {
                                            /* Initialize GPU data structures */
                                            dinitSluGPU3D_t(sluGPU_temp, &LUstruct_temp, grid3d, perm_c_supno_temp,
                                                            n, buffer_size, bigu_size_temp, ldt);
                                        }
                                    }
                                    #else
                                    
                                    /* Initialize GPU data structures */
                                    dinitSluGPU3D_t(sluGPU_temp, &LUstruct_temp, grid3d, perm_c_supno_temp,
                                                    n, buffer_size, bigu_size_temp, ldt);
                                    #endif
                                    
                                    HyP_temp->bigu_size = bigu_size_temp;
                                    HyP_temp->buffer_size = buffer_size;
                                    HyP_temp->nsupers = nsupers;
                                    HyP_temp->first_u_block_acc = sluGPU_temp->A_gpu->first_u_block_gpu;
                                    HyP_temp->first_l_block_acc = sluGPU_temp->A_gpu->first_l_block_gpu;
                                    HyP_temp->nCudaStreams = sluGPU_temp->nCudaStreams;
                                    sluGPU_temp->A_gpu->isEmpty = 0;

                                    SUPERLU_FREE(perm_c_supno_temp);
                                                                    
                                }                                

                                // packLUInfo
                                packLUInfo_t packLUInfo_temp;
                                initPackLUInfo(nsupers, &packLUInfo_temp);

                                // factStat
                                factStat_t factStat_temp;
                                initFactStat(nsupers, &factStat_temp);

                                // fNlists
                                factNodelists_t  fNlists_temp;
                                initFactNodelists( ldt, num_threads, nsupers, &fNlists_temp);

                                gEtreeInfo_t gEtreeInfo_temp;
                                int_t* iperm_c_supno_temp;

                                /* main loop over all the supernodes */
                                if (sforest_temp) /* 2D factorization at individual subtree */
                                {
                                    int tag = 600;

                                    // comReqss
                                    int_t mxLeafNode_temp;
                                    MPI_Recv(&mxLeafNode_temp, 1, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);
                                    commRequests_t** comReqss_temp = initCommRequestsArr(SUPERLU_MAX(mxLeafNode_temp, numLA), ldt, grid);

                                    // bigu_size
                                    int_t bigu_size_temp;
                                    MPI_Recv(&bigu_size_temp, 1, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                    // scuBufs
                                    dscuBufs_t scuBufs_temp;
                                    scuBufs_temp.bigV = dgetBigV(ldt, num_threads);
                                    scuBufs_temp.bigU = doubleMalloc_dist(bigu_size_temp);
                                    MPI_Recv(scuBufs_temp.bigV, 8 * ldt * ldt * num_threads, MPI_DOUBLE, i, tag++, grid3d->zscp.comm, &status);
                                    MPI_Recv(scuBufs_temp.bigU, bigu_size_temp, MPI_DOUBLE, i, tag++, grid3d->zscp.comm, &status);

                                    // dFBufs
                                    ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode_temp, ldt, grid);

                                    // gEtreeInfo                                
                                    int_t *setree = intMalloc_dist(nsupers);
                                    MPI_Recv(setree, nsupers, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);
                                    gEtreeInfo_temp.setree = setree;
                                    gEtreeInfo_temp.numChildLeft = (int_t* ) SUPERLU_MALLOC(sizeof(int_t) * nsupers);
                                    MPI_Recv(gEtreeInfo_temp.numChildLeft, nsupers, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                    // iperm_c_supno
                                    iperm_c_supno_temp = INT_T_ALLOC(nsupers);
                                    MPI_Recv(iperm_c_supno_temp, nsupers, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                    dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), &LUstruct_temp);

                                    #ifdef Torch_stat
                                    SCT->tExchange[ilvl] += (SuperLU_timer_() - time_exchange);
                                    #endif

                                    double tilvl = SuperLU_timer_();
                                    
                                    if(superlu_acc_offload){
                                        dsparseTreeFactor_ASYNC_GPU(
                                        sforest_temp,
                                        comReqss_temp, &scuBufs_temp,  &packLUInfo_temp,
                                        msgss, LUvsbs, dFBufs,  &factStat_temp, &fNlists_temp,
                                        &gEtreeInfo_temp, options,  iperm_c_supno_temp, ldt,
                                        sluGPU_temp,  d2Hred_temp,  HyP_temp, &LUstruct_temp, grid3d, stat,
                                        thresh,  SCT, tag_ub, info);
                                    }                                    

                                    // scuBufs
                                    // MPI_Send(scuBufs_temp.bigV, 8 * ldt * ldt * num_threads, MPI_DOUBLE, i, tag++, grid3d->zscp.comm);
                                    // MPI_Send(scuBufs_temp.bigU, bigu_size, MPI_DOUBLE, i, tag++, grid3d->zscp.comm);

                                    freeCommRequestsArr(SUPERLU_MAX(mxLeafNode_temp, numLA), comReqss_temp);
                                    dfreeScuBufs(&scuBufs_temp);
                                    dfreeDiagFactBufsArr(mxLeafNode_temp, dFBufs);
                                    dLluBufFreeArr(numLA, LUvsbs);
                                    SUPERLU_FREE(gEtreeInfo_temp.setree);
                                    SUPERLU_FREE(gEtreeInfo_temp.numChildLeft);
                                    SUPERLU_FREE(iperm_c_supno_temp);

                                    /*now reduce the updates*/
                                    SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                                    sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];

                                    #ifdef Torch_stat
                                    printf("%d: tFactor3D[%d] = %e s\n", grid3d->iam, ilvl, SCT->tFactor3D[ilvl]);
                                    #endif
                                    
                                }                                

                                if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                                {
                                    if(superlu_acc_offload){
                                        int tag = 700;

                                        #ifdef Torch_stat
                                        time_exchange = SuperLU_timer_();
                                        #endif

                                        // myNodeCount
                                        int_t* myNodeCount_temp = INT_T_ALLOC(maxLvl);
                                        MPI_Recv(myNodeCount_temp, maxLvl, mpi_int_t, i, tag++, grid3d->zscp.comm, &status);

                                        // treePerm
                                        int_t** treePerm_temp = getTreePermFr( myTreeIdxs_temp, sForests_temp, grid3d);

                                        // myGrid
                                        MPI_Recv(&(LUstruct_temp.Llu->isEmpty), 1, MPI_INT, i, tag++, grid3d->zscp.comm, &status);

                                        // iam
                                        MPI_Recv(&(LUstruct_temp.Llu->tempiam), 1, MPI_INT, i, tag++, grid3d->zscp.comm, &status);

                                        #ifdef Torch_stat
                                        SCT->tExchange[ilvl] += (SuperLU_timer_() - time_exchange);
                                        #endif

                                        dreduceAllAncestors3d_GPU(
                                            ilvl, myNodeCount_temp, treePerm_temp, LUvsb,
                                            &LUstruct_temp, grid3d, sluGPU_temp, d2Hred_temp, &factStat_temp, HyP_temp,
                                            SCT );

                                        LUstruct_temp.Llu->isEmpty = 2;

                                        if (maxLvl == 2) {
                                            dfree_LUstruct_gpu(sluGPU_temp->A_gpu);
                                        }
                                        else
                                        {
                                            dfree_LUstruct_gpu_Buffer(sluGPU_temp->A_gpu);
                                        }

                                        SUPERLU_FREE(myNodeCount_temp);
                                        SUPERLU_FREE(myTreeIdxs_temp);
                                    }
                                }                              
               
                                #ifdef Torch_stat
                                time_exchange = SuperLU_timer_();
                                #endif

                                // for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                                // {
                                //     MPI_Barrier(grid->comm);
                                //     if (j == grid->iam)
                                //     {
                                        // exchange LUStruct
                                        dBcastRecv_LUstruct1(&LUstruct_temp, 0, i, grid3d, nsupers);
                                //     }
                                // }
                                
                                MPI_Barrier(grid->comm);
                                restore_LUstruct(Llu, LUstruct_temp.Llu, grid3d, nsupers);

                                freePackLUInfo(&packLUInfo_temp);
                                freeFactStat(&factStat_temp);
                                freeFactNodelists(&fNlists_temp);

                                #ifdef Torch_stat
                                SCT->tExchange[ilvl] += (SuperLU_timer_() - time_exchange);

                                printf("%d: tExchange[%d] = %e s\n", grid3d->iam, ilvl, SCT->tExchange[ilvl]);
                                #endif
                            }

                            Free_HyP(HyP_temp);
                            
                        }
                        
                        
                    }
                    
                }
                else
                {
                    if (grid3d->zscp.Iam == 0)
                    {   
                        if(superlu_acc_offload){
                            
                            #ifdef Torch_batch_init
                            for (int j = 0; j < grid3d->npcol * grid3d->nprow; j++)
                            {
                                MPI_Barrier(grid->comm);
                                if (j == grid->iam)
                                {
                                    /* Initialize GPU data structures */
                                    dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                                    n, buffer_size, bigu_size, ldt);
                                }
                            }

                            MPI_Barrier(grid->comm);

                            #else
                            /* Initialize GPU data structures */
                            dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                            n, buffer_size, bigu_size, ldt);
                            #endif

                            HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
                            HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
                            HyP->nCudaStreams = sluGPU->nCudaStreams;
                            sluGPU->A_gpu->isEmpty = 0;  
                        }               
                        
                        /* if I participate in this level */
                        if (!myZeroTrIdxs[ilvl])
                        {
                            
                            //int_t tree = myTreeIdxs[ilvl];
                            sForest_t* sforest = sForests[myTreeIdxs[ilvl]];  

                            /* main loop over all the supernodes */
                            if (sforest) /* 2D factorization at individual subtree */
                            { 
                                double tilvl = SuperLU_timer_();
                                ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
                                dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);

                                if(superlu_acc_offload){
                                    
                                    dsparseTreeFactor_ASYNC_GPU(
                                        sforest,
                                        comReqss, &scuBufs,  &packLUInfo,
                                        msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                                        &gEtreeInfo, options,  iperm_c_supno, ldt,
                                        sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                                        thresh,  SCT, tag_ub, info);
                                }
                                
                                else{
                                    dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                                        msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                                        &gEtreeInfo, options, iperm_c_supno, ldt,
                                        HyP, LUstruct, grid3d, stat,
                                        thresh,  SCT, tag_ub, info );
                                }

                                dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
                                dLluBufFreeArr(numLA, LUvsbs);

                                /*now reduce the updates*/
                                SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                                sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];

                                #ifdef Torch_stat
                                printf("%d: tFactor3D[%d] = %e s\n", grid3d->iam, ilvl, SCT->tFactor3D[ilvl]);
                                #endif
                            }
                            
                            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                            {
                                if(superlu_acc_offload){
                                    dreduceAllAncestors3d_GPU(
                                        ilvl, myNodeCount, treePerm, LUvsb,
                                        LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                                        SCT );
                                }                        

                            }       

                        } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/
                    }
                }
                
                #if 0
                if (i == grid3d->zscp.Iam)
                {
                    /* if I participate in this level */
                    if (!myZeroTrIdxs[ilvl])
                    {
                        //int_t tree = myTreeIdxs[ilvl];
                        sForest_t* sforest = sForests[myTreeIdxs[ilvl]];            

                        /* main loop over all the supernodes */
                        if (sforest) /* 2D factorization at individual subtree */
                        {                
                            double tilvl = SuperLU_timer_();

                            if(superlu_acc_offload){
                                
                                if (ilvl == 0)
                                {
                                
                                /* Initialize GPU data structures */
                                    dinitSluGPU3D_t(sluGPU, LUstruct, grid3d, perm_c_supno,
                                                    n, buffer_size, bigu_size, ldt);

                                    HyP->first_u_block_acc = sluGPU->A_gpu->first_u_block_gpu;
                                    HyP->first_l_block_acc = sluGPU->A_gpu->first_l_block_gpu;
                                    HyP->nCudaStreams = sluGPU->nCudaStreams;
                                }
                                
                                dsparseTreeFactor_ASYNC_GPU(
                                    sforest,
                                    comReqss, &scuBufs,  &packLUInfo,
                                    msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                                    &gEtreeInfo, options,  iperm_c_supno, ldt,
                                    sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                                    thresh,  SCT, tag_ub, info);
                            }
                            
                            else{
                                dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                                    msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                                    &gEtreeInfo, options, iperm_c_supno, ldt,
                                    HyP, LUstruct, grid3d, stat,
                                    thresh,  SCT, tag_ub, info );
                            }

                            /*now reduce the updates*/
                            SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                            sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];
                        }
                        
                        if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                        {
                            if(superlu_acc_offload){
                                dreduceAllAncestors3d_GPU(
                                    ilvl, myNodeCount, treePerm, LUvsb,
                                    LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                                    SCT );
                                
                            }                        

                        }       

                    } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/

                }
                #endif
            }
        }
        else
        {
            if (grid3d->zscp.Iam)
            {
                /* if I participate in this level */
                if (!myZeroTrIdxs[ilvl])
                {
                    if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                    {
                        if(superlu_acc_offload){
                            sluGPU->A_gpu->isEmpty = 1;
                            dreduceAllAncestors3d_GPU(
                                ilvl, myNodeCount, treePerm, LUvsb,
                                LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                                SCT );
                        }

                    }  
                }    
            }
            else
            {
                /* if I participate in this level */
                if (!myZeroTrIdxs[ilvl])
                {
                    //int_t tree = myTreeIdxs[ilvl];
                    sForest_t* sforest = sForests[myTreeIdxs[ilvl]];

                    /* main loop over all the supernodes */
                    if (sforest) /* 2D factorization at individual subtree */
                    {                
                        double tilvl = SuperLU_timer_();

                        ddiagFactBufs_t** dFBufs = dinitDiagFactBufsArr(mxLeafNode, ldt, grid);
                        dLUValSubBuf_t** LUvsbs = dLluBufInitArr( SUPERLU_MAX( numLA, grid3d->zscp.Np ), LUstruct);

                        if(superlu_acc_offload){
                            
                            dsparseTreeFactor_ASYNC_GPU(
                                sforest,
                                comReqss, &scuBufs,  &packLUInfo,
                                msgss, LUvsbs, dFBufs,  &factStat, &fNlists,
                                &gEtreeInfo, options,  iperm_c_supno, ldt,
                                sluGPU,  d2Hred,  HyP, LUstruct, grid3d, stat,
                                thresh,  SCT, tag_ub, info);
                        }
                        
                        else{
                            dsparseTreeFactor_ASYNC(sforest, comReqss,  &scuBufs, &packLUInfo,
                                msgss, LUvsbs, dFBufs, &factStat, &fNlists,
                                &gEtreeInfo, options, iperm_c_supno, ldt,
                                HyP, LUstruct, grid3d, stat,
                                thresh,  SCT, tag_ub, info );
                        }

                        dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
                        dLluBufFreeArr(numLA, LUvsbs);

                        /*now reduce the updates*/
                        SCT->tFactor3D[ilvl] = SuperLU_timer_() - tilvl;
                        sForests[myTreeIdxs[ilvl]]->cost = SCT->tFactor3D[ilvl];

                        #ifdef Torch_stat
                        printf("%d: tFactor3D[%d] = %e s\n", grid3d->iam, ilvl, SCT->tFactor3D[ilvl]);
                        #endif
                    }
                    
                    if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
                    {
                        if(superlu_acc_offload){
                            dreduceAllAncestors3d_GPU(
                                ilvl, myNodeCount, treePerm, LUvsb,
                                LUstruct, grid3d, sluGPU, d2Hred, &factStat, HyP,
                                SCT );
                            
                        }                        

                    }       

                } /*if (!myZeroTrIdxs[ilvl])  ... If I participate in this level*/
            }
            
        }
                

        /* if I participate in this level */
        if (!myZeroTrIdxs[ilvl])
        {
            if (ilvl < maxLvl - 1)     /*then reduce before factorization*/
            {
                if(superlu_acc_offload){

                    dreduceAllAncestors3d(ilvl, myNodeCount, treePerm, 	                      LUvsb, LUstruct, grid3d, SCT );
                }
                else{

                    dreduceAllAncestors3d(ilvl, myNodeCount, treePerm,
                                        LUvsb, LUstruct, grid3d, SCT );
                }

            }
        }        
        
        
        printf("%d: ilvl %d\n", grid3d->iam, ilvl);        
        MPI_Barrier( grid3d->comm);
        #endif       
        

        SCT->tSchCompUdt3d[ilvl] = ilvl == 0 ? SCT->NetSchurUpTimer
	    : SCT->NetSchurUpTimer - SCT->tSchCompUdt3d[ilvl - 1];

        #ifdef Torch_stat
        printf("%d: tSchCompUdt3d[%d] = %e s\n", grid3d->iam, ilvl, SCT->tSchCompUdt3d[ilvl]);
        #endif
        
        #if 0
        num_threads=num_threads*1.5;
        omp_set_num_threads(num_threads);
        #endif

    } /*for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)*/

    MPI_Barrier( grid3d->comm);
    
    #ifdef SuperLargeScaleGPU
    if (Llu->isRecordingForGPU == RecordingForGPU)
    {
        if (Llu->MaxGPUMemory <= Llu->LimitGPUMemory)
        {
            printf("(%d) MaxGPUMemory %f, LimitGPUMemory %f, LimitGPUMemory is appropriate.\n", grid3d->iam, Llu->MaxGPUMemory, Llu->LimitGPUMemory);
        }
        else
        {
            printf("(%d) MaxGPUMemory %f, LimitGPUMemory %f, LimitGPUMemory is not appropriate.\n", grid3d->iam, Llu->MaxGPUMemory, Llu->LimitGPUMemory);
        }       
        
    }
    #endif
    
    SCT->pdgstrfTimer = SuperLU_timer_() - SCT->pdgstrfTimer; 
    #ifdef test_0318
    MPI_Barrier(grid3d->comm);

    
    sleep(10);
    MPI_Barrier(grid3d->comm);
    for (int i = 0; i < grid3d->npcol*grid3d->nprow*grid3d->npdep; i++)
    {
        
        /* code */
        if(grid3d->iam==i){
            dPrintLblocks(i, HyP->nsupers, grid, LUstruct->Glu_persist, LUstruct->Llu);
            dPrintUblocks(i, HyP->nsupers, grid, LUstruct->Glu_persist, LUstruct->Llu);
        }
        MPI_Barrier(grid3d->comm);
        sleep(5);
    }
    
    #endif    

#ifdef ITAC_PROF
    VT_traceoff();
#endif

#ifdef MAP_PROFILE
    allinea_stop_sampling();
#endif

    reduceStat(FACT, stat, grid3d);
    // sherry added
    /* Deallocate factorization specific buffers */
    #ifndef SuperLargeScaleGPUBuffer
    freePackLUInfo(&packLUInfo);
    
    freeFactStat(&factStat);
    
    freeFactNodelists(&fNlists);
    freeMsgsArr(numLA, msgss);
    freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
    dfreeScuBufs(&scuBufs);
    dLluBufFreeArr(numLA, LUvsbs);
    #else
    
    if (grid3d->iam == 0)
    {
        freePackLUInfo(&packLUInfo);
        freeFactStat(&factStat);
        dfreeScuBufs(&scuBufs);
        freeFactNodelists(&fNlists);
        freeCommRequestsArr(SUPERLU_MAX(mxLeafNode, numLA), comReqss);
    }
    freeMsgsArr(numLA, msgss);

    #endif

    #ifndef SuperLargeScaleGPUBuffer
    dfreeDiagFactBufsArr(mxLeafNode, dFBufs);
    #endif
    
    Free_HyP(HyP);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (grid3d->iam, "Exit pdgstrf3d()");
#endif
    return 0;

} /* pdgstrf3d */
#endif

