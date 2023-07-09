

/*! @file
 * \brief Factorization routines for the subtree using 2D process grid, with GPUs.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley,
 * Georgia Institute of Technology, Oak Ridge National Laboratory
 * May 12, 2021
 * </pre>
 */
// #include "treeFactorization.h"
// #include "trfCommWrapper.h"
#include "dlustruct_gpu.h"

//#include "cblas.h"
#define Torch

#ifdef GPU_ACC ///////////////// enable GPU

/* 
/-- num_u_blks--\ /-- num_u_blks_Phi --\
----------------------------------------
|  host_cols    ||    GPU   |   host   |
----------------------------------------
                  ^          ^
                  0          jj_cpu
*/
static int_t getAccUPartition(HyP_t *HyP)
{
    /* Sherry: what if num_u_blks_phi == 0 ? Need to fix the bug */
    int_t total_cols_1 = HyP->Ublock_info_Phi[HyP->num_u_blks_Phi - 1].full_u_cols;

    // int_t host_cols = HyP->Ublock_info[HyP->num_u_blks - 1].full_u_cols;
    // double cpu_time_0 = estimate_cpu_time(HyP->Lnbrow, total_cols_1, HyP->ldu_Phi) +
    //                     estimate_cpu_time(HyP->Rnbrow, host_cols, HyP->ldu) + estimate_cpu_time(HyP->Lnbrow, host_cols, HyP->ldu);

    int jj_cpu;

#if 0 /* Ignoe those estimates */
    jj_cpu = tuned_partition(HyP->num_u_blks_Phi, HyP->Ublock_info_Phi,
                                   HyP->Remain_info, HyP->RemainBlk, cpu_time_0, HyP->Rnbrow, HyP->ldu_Phi );
#else /* Sherry: new */
    jj_cpu = HyP->num_u_blks_Phi;
#endif

    if (jj_cpu != 0 && HyP->Rnbrow > 0) // ###
    {
        HyP->offloadCondition = 1;
    }
    else
    {
        HyP->offloadCondition = 0;
        jj_cpu = 0; // ###
    }

    return jj_cpu;
}

#ifdef SuperLargeScaleGPU
void buffermemoryrefresh_Lnzval(dLUstruct_gpu_t *A_gpu, dLUstruct_t *LUstruct, int_t nsupers, int_t npcol)
{
	dLocalLU_t *Llu = LUstruct->Llu;    
	int_t *mLnzval_RecordMatrix = &(LNZVAL_RECORDMATRIX(Llu->ilvl, A_gpu->nexttopoLvl_Lnzval + 1, A_gpu->nextk0_Lnzval + 1, 0));
    int_t sumUsed_Lnzval_bc_ptr_Record = mLnzval_RecordMatrix[0];
	int_t l_val_len = 0;

    #pragma omp parallel for
	for (int_t m = 1; m < sumUsed_Lnzval_bc_ptr_Record + 1; m++)
	{
		int_t index = mLnzval_RecordMatrix[m];
		// exchange data from GPU to host
        if(A_gpu->isCPUUsed_Lnzval_bc_ptr_host[index] != CPUUsed){
			A_gpu->LnzvalVec_host[index] = doubleMalloc_dist(Llu->Lnzval_bc_ptr_ilen[index]);
		}
		checkCuda(cudaMemcpy(A_gpu->LnzvalVec_host[index], &A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[index]], Llu->Lnzval_bc_ptr_ilen[index] * sizeof(double), cudaMemcpyDeviceToHost));
		
		A_gpu->isCPUUsed_Lnzval_bc_ptr_host[index] = CPUUsed;
		A_gpu->isGPUUsed_Lnzval_bc_ptr_host[index] = GPUUnused;
	}

    if (A_gpu->pre_lrecordid + 2 >= A_gpu->UsedOrder_Lnzval[Llu->ilvl][0])
    {
        return;
    }
    
	A_gpu->pre_lrecordid += 2;
	A_gpu->nexttopoLvl_Lnzval = A_gpu->UsedOrder_Lnzval[Llu->ilvl][A_gpu->pre_lrecordid];
	A_gpu->nextk0_Lnzval = A_gpu->UsedOrder_Lnzval[Llu->ilvl][A_gpu->pre_lrecordid + 1];

	mLnzval_RecordMatrix = &(LNZVAL_RECORDMATRIX(Llu->ilvl, A_gpu->nexttopoLvl_Lnzval + 1, A_gpu->nextk0_Lnzval + 1, 0));
	sumUsed_Lnzval_bc_ptr_Record = mLnzval_RecordMatrix[0];

	for (int_t m = 1; m < sumUsed_Lnzval_bc_ptr_Record + 1; m++)
	{
		int_t index = mLnzval_RecordMatrix[m];
		A_gpu->isGPUUsed_Lnzval_bc_ptr_host[index] = GPUUsed;
		// exchange data from host to GPU
        A_gpu->LnzvalPtr_host[index] = l_val_len;
		if(A_gpu->isCPUUsed_Lnzval_bc_ptr_host[index] == CPUUsed)
		{
			checkCuda(cudaMemcpy(&A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[index]], A_gpu->LnzvalVec_host[index], Llu->Lnzval_bc_ptr_ilen[index] * sizeof(double), cudaMemcpyHostToDevice));
		}
        else
        {
            checkCuda(cudaMemset(&A_gpu->LnzvalVec[A_gpu->LnzvalPtr_host[index]], 0, Llu->Lnzval_bc_ptr_ilen[index] * sizeof(double)));
        }
        		
		l_val_len += Llu->Lnzval_bc_ptr_ilen[index];
	}

    checkCuda(cudaMemcpy( (A_gpu->LnzvalPtr), A_gpu->LnzvalPtr_host, CEILING(nsupers, npcol) * sizeof(int_t), cudaMemcpyHostToDevice)) ;        

	A_gpu->Lnzval_bc_ptr_len = l_val_len;
}

void buffermemoryrefresh_Unzval(dLUstruct_gpu_t *A_gpu, dLUstruct_t *LUstruct, int_t nsupers, int_t nprow)
{
	dLocalLU_t *Llu = LUstruct->Llu;
	int_t *mUnzval_RecordMatrix = &(UNZVAL_RECORDMATRIX(Llu->ilvl, A_gpu->nexttopoLvl_Unzval + 1, A_gpu->nextk0_Unzval + 1, 0));
    int_t sumUsed_Unzval_br_ptr_Record = mUnzval_RecordMatrix[0];
	int_t u_val_len = 0;

    #pragma omp parallel for
	for (int_t m = 1; m < sumUsed_Unzval_br_ptr_Record + 1; m++)
	{
		int_t index = mUnzval_RecordMatrix[m];
		// exchange data from GPU to host
        if(A_gpu->isCPUUsed_Unzval_br_ptr_host[index] != CPUUsed){
			A_gpu->UnzvalVec_host[index] = doubleMalloc_dist(Llu->Unzval_br_ptr_ilen[index]);
		}
		checkCuda(cudaMemcpy(A_gpu->UnzvalVec_host[index], &A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[index]], Llu->Unzval_br_ptr_ilen[index] * sizeof(double), cudaMemcpyDeviceToHost));
		
		A_gpu->isCPUUsed_Unzval_br_ptr_host[index] = CPUUsed;
		A_gpu->isGPUUsed_Unzval_br_ptr_host[index] = GPUUnused;
	}

    if (A_gpu->pre_urecordid + 2 >= A_gpu->UsedOrder_Unzval[Llu->ilvl][0])
    {
        return;
    }

	A_gpu->pre_urecordid += 2;
	A_gpu->nexttopoLvl_Unzval = A_gpu->UsedOrder_Unzval[Llu->ilvl][A_gpu->pre_urecordid];
	A_gpu->nextk0_Unzval = A_gpu->UsedOrder_Unzval[Llu->ilvl][A_gpu->pre_urecordid + 1];

	mUnzval_RecordMatrix = &(UNZVAL_RECORDMATRIX(Llu->ilvl, A_gpu->nexttopoLvl_Unzval + 1, A_gpu->nextk0_Unzval + 1, 0));
	sumUsed_Unzval_br_ptr_Record = mUnzval_RecordMatrix[0];

	for (int_t m = 1; m < sumUsed_Unzval_br_ptr_Record + 1; m++)
	{
		int_t index = mUnzval_RecordMatrix[m];
		A_gpu->isGPUUsed_Unzval_br_ptr_host[index] = GPUUsed;
		// exchange data from host to GPU
        A_gpu->UnzvalPtr_host[index] = u_val_len;
		if(A_gpu->isCPUUsed_Unzval_br_ptr_host[index] == CPUUsed)
		{
			checkCuda(cudaMemcpy(&A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[index]], A_gpu->UnzvalVec_host[index], Llu->Unzval_br_ptr_ilen[index] * sizeof(double), cudaMemcpyHostToDevice));
		}
        else
        {
            checkCuda(cudaMemset(&A_gpu->UnzvalVec[A_gpu->UnzvalPtr_host[index]], 0, Llu->Unzval_br_ptr_ilen[index] * sizeof(double)));
        }
		u_val_len += Llu->Unzval_br_ptr_ilen[index];		
	}

    checkCuda(cudaMemcpy( (A_gpu->UnzvalPtr), A_gpu->UnzvalPtr_host, CEILING(nsupers, nprow) * sizeof(int_t), cudaMemcpyHostToDevice)) ;

	A_gpu->Unzval_br_ptr_len = u_val_len;
}
#endif

int dsparseTreeFactor_ASYNC_GPU(
    sForest_t *sforest,
    commRequests_t **comReqss, // lists of communication requests,
                               // size = maxEtree level
    dscuBufs_t *scuBufs,        // contains buffers for schur complement update
    packLUInfo_t *packLUInfo,
    msgs_t **msgss,          // size = num Look ahead
    dLUValSubBuf_t **LUvsbs, // size = num Look ahead
    ddiagFactBufs_t **dFBufs, // size = maxEtree level
    factStat_t *factStat,
    factNodelists_t *fNlists,
    gEtreeInfo_t *gEtreeInfo, // global etree info
    superlu_dist_options_t *options,
    int_t *gIperm_c_supno,
    int ldt,
    dsluGPU_t *sluGPU,
    d2Hreduce_t *d2Hred,
    HyP_t *HyP,
    dLUstruct_t *LUstruct, gridinfo3d_t *grid3d, SuperLUStat_t *stat,
    double thresh, SCT_t *SCT, int tag_ub,
    int *info)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid3d->iam, "Enter dsparseTreeFactor_ASYNC_GPU()");
    #endif

    // sforest.nNodes, sforest.nodeList,
    // &sforest.topoInfo,
    int_t nnodes = sforest->nNodes; // number of nodes in supernodal etree
    if (nnodes < 1)
    {
        return 1;
    }

    int_t *perm_c_supno = sforest->nodeList; // list of nodes in the order of factorization
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t *myIperm = treeTopoInfo->myIperm;

    gridinfo_t *grid = &(grid3d->grid2d);
    /*main loop over all the levels*/

    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;
    int_t *IrecvPlcd_D = factStat->IrecvPlcd_D;
    int_t *factored_D = factStat->factored_D;
    int_t *factored_L = factStat->factored_L;
    int_t *factored_U = factStat->factored_U;
    int_t *IbcastPanel_L = factStat->IbcastPanel_L;
    int_t *IbcastPanel_U = factStat->IbcastPanel_U;
    int_t *gpuLUreduced = factStat->gpuLUreduced;
    int_t *xsup = LUstruct->Glu_persist->xsup;

    // int_t numLAMax = getNumLookAhead();
    int_t numLAMax = getNumLookAhead(options);
    int_t numLA = numLAMax; // number of look-ahead panels
    int_t superlu_acc_offload = HyP->superlu_acc_offload;
    int_t last_flag = 1;                       /* for updating nsuper-1 only once */
    int_t nCudaStreams = sluGPU->nCudaStreams; // number of cuda streams

    #ifdef Torch
    int_t nsupers=HyP->nsupers;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t sum_Used_Lnzval_bc_ptr, sum_Used_Unzval_br_ptr, sum_Lnzval_bc_ptr_ilen, sum_Unzval_br_ptr_ilen, sum_Lnzval_bc_ptr_len, sum_Unzval_br_ptr_len;
    #endif   

    #ifdef SuperLargeScale    
    Llu->core_status = OutOfCore;
    #ifdef SuperLargeScale_limit
    Llu->core_status = InCore;
    
    #endif
    #endif   

    if (superlu_acc_offload)
        dsyncAllfunCallStreams(sluGPU, SCT);

    #ifdef Torch_0312
    int_t *tmpIbcastPanel_L = intMalloc_dist(nsupers);
    int_t *tmpIbcastPanel_U = intMalloc_dist(nsupers);
    int_t *tmpfactored_L = intMalloc_dist( nsupers); //INT_T_ALLOC( nsupers);
    int_t *tmpfactored_U = intMalloc_dist( nsupers); //INT_T_ALLOC( nsupers);

    int_t maxoffset = 0;
    for (int topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    {
        if (topoLvl < maxTopoLevel - 1) /* not root */
        {
            /*look-ahead LU factorization*/
            int kx_st = eTreeTopLims[topoLvl + 1];
            int kx_end = eTreeTopLims[topoLvl + 2];
            if (maxoffset < (kx_end - kx_st))
            {
                maxoffset = kx_end - kx_st;
            }
            
        }
    }
    int **recvLDiagMatrix = (int **)SUPERLU_MALLOC(maxoffset * sizeof(int*));
    int **recvUDiagMatrix = (int **)SUPERLU_MALLOC(maxoffset * sizeof(int*));
    for (int i = 0; i < maxoffset; i++)
    {
        recvLDiagMatrix[i] = SUPERLU_MALLOC( nsupers * sizeof(int));
        recvUDiagMatrix[i] = SUPERLU_MALLOC( nsupers * sizeof(int));
    }    
    #endif
        
    /* Go through each leaf node */
    for (int_t k0 = 0; k0 < eTreeTopLims[1]; ++k0)
    {
        
        int_t k = perm_c_supno[k0]; // direct computation no perm_c_supno
        int_t offset = k0;
        /* k-th diagonal factorization */

        /* If LU panels from GPU are not reduced, then reduce
	   them before diagonal factorization */
        if (!gpuLUreduced[k] && superlu_acc_offload)
        {
            double tt_start1 = SuperLU_timer_();

            dinitD2Hreduce(k, d2Hred, last_flag,
                          HyP, sluGPU, grid, LUstruct, SCT);
            int_t copyL_kljb = d2Hred->copyL_kljb;
            int_t copyU_kljb = d2Hred->copyU_kljb;

            if (copyL_kljb || copyU_kljb)
                SCT->PhiMemCpyCounter++;
            dsendLUpanelGPU2HOST(k, d2Hred, sluGPU);

            dreduceGPUlu(last_flag, d2Hred, sluGPU, SCT, grid, LUstruct);

            gpuLUreduced[k] = 1;
            SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
           
        }

        double t1 = SuperLU_timer_();

        /*Now factor and broadcast diagonal block*/
        // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
        //                 options, thresh, LUstruct, stat, info, SCT);

#if 0
        sDiagFactIBCast(k,  dFBufs[offset], factStat, comReqss[offset], grid,
                        options, thresh, LUstruct, stat, info, SCT, tag_ub);
#else
        dDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor, dFBufs[offset]->BlockLFactor,
                        factStat->IrecvPlcd_D,
                        comReqss[offset]->U_diag_blk_recv_req,
                        comReqss[offset]->L_diag_blk_recv_req,
                        comReqss[offset]->U_diag_blk_send_req,
                        comReqss[offset]->L_diag_blk_send_req,
                        grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif
        factored_D[k] = 1;

        // printf("2.%ld: %ld, %ld\n", grid3d->iam, k0, eTreeTopLims[1]);

        SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
    } /* for all leaves ... */

    // printf("%d, .. SparseFactor_GPU: after leaves %e\n", grid->iam, SuperLU_timer_()); fflush(stdout);
    
    #ifndef SuperLargeScale

    // int_t max_maxTopoLevel;
    // MPI_Status *status;
    // int_t last_topoLvl, next_topoLvl;
    // int last_rank, next_rank;
    // if (MPI_COMM_NULL != grid3d->gridsave.comm){

    //     int saveflag = 0;
    //     int_t **maxTopolevelVector = (int_t**)SUPERLU_MALLOC(grid3d->nSave * sizeof(int_t*));
    //     gridsaveinfo_t *gridSave = &(grid3d->gridsave);

    //     if (gridSave->iam == 0)
    //     {
    //         maxTopolevelVector[0] = intMalloc_dist(2);
    //         maxTopolevelVector[0][0] = 0;
    //         maxTopolevelVector[0][1] = maxTopoLevel;
    //         for (int i = 1; i < grid3d->nSave; i++)
    //         {
    //             maxTopolevelVector[i] = intMalloc_dist(2);
    //             maxTopolevelVector[i][0] = i;
    //             MPI_Recv(&maxTopolevelVector[i][1], 1, mpi_int_t, i, i, gridSave->comm, status);
    //         }

    //         for (int i = 0; i < grid3d->nSave; i++)
    //         {
                
    //             for (int j = i + 1; j < grid3d->nSave; j++)
    //             {
    //                 if (maxTopolevelVector[i][1] > maxTopolevelVector[j][1])
    //                 {
    //                     int_t temp_maxtopoLvl = maxTopolevelVector[i][1];
    //                     int tempid = maxTopolevelVector[i][0];
    //                     maxTopolevelVector[i][1] = maxTopolevelVector[j][1];
    //                     maxTopolevelVector[i][0] = maxTopolevelVector[j][0];
    //                     maxTopolevelVector[j][1] = temp_maxtopoLvl;
    //                     maxTopolevelVector[j][0] = tempid;
    //                 }
                    
    //             }
                
    //         }

    //         int_t max_maxTopoLevel = maxTopolevelVector[grid3d->nSave - 1][1];
    //         int_t last_topoLvl0, next_topoLvl0;
    //         int last_rank0, next_rank0;

    //         for (int i = 0; i < grid3d->nSave; i++)
    //         {
    //             if (i == 0)
    //             {
    //                 last_topoLvl = 0;
    //                 last_rank = -1;
    //             }
    //             else
    //             {
    //                 last_topoLvl = maxTopolevelVector[i-1][1];
    //                 last_rank = maxTopolevelVector[i-1][0];
    //             }

    //             if (i == grid3d->nSave - 1)
    //             {
    //                 next_topoLvl = max_maxTopoLevel;
    //                 next_rank = grid3d->nSave;
    //             }
    //             else
    //             {
    //                 next_topoLvl = maxTopolevelVector[i+1][1];
    //                 next_rank = maxTopolevelVector[i+1][0];
    //             }
                
    //             if (maxTopolevelVector[i][0] != 0)
    //             {
    //                 MPI_Send(&last_topoLvl, 1, mpi_int_t, maxTopolevelVector[i][0], maxTopolevelVector[i][0], gridSave->comm);
    //                 MPI_Send(&next_topoLvl, 1, mpi_int_t, maxTopolevelVector[i][0], maxTopolevelVector[i][0], gridSave->comm);
    //                 MPI_Send(&last_rank, 1, MPI_INT, maxTopolevelVector[i][0], maxTopolevelVector[i][0], gridSave->comm);
    //                 MPI_Send(&next_rank, 1, MPI_INT, maxTopolevelVector[i][0], maxTopolevelVector[i][0], gridSave->comm);
    //             }
    //             else
    //             {
    //                 last_rank0 = last_rank;
    //                 last_topoLvl0 = last_topoLvl;
    //                 next_rank0 = next_rank;
    //                 next_topoLvl0 = next_topoLvl;
    //             }
                
    //         }
            
    //         last_rank = last_rank0;
    //         last_topoLvl = last_topoLvl0;
    //         next_rank = next_rank0;
    //         next_topoLvl = next_topoLvl0;

    //     }
    //     else{
    //         MPI_Send(&maxTopoLevel, 1, mpi_int_t, 0, gridSave->iam, gridSave->comm);
    //         MPI_Recv(&last_topoLvl, 1, mpi_int_t, 0, gridSave->iam, gridSave->comm, status);
    //         MPI_Recv(&next_topoLvl, 1, mpi_int_t, 0, gridSave->iam, gridSave->comm, status);
    //         MPI_Recv(&last_rank, 1, MPI_INT, 0, gridSave->iam, gridSave->comm, status);
    //         MPI_Recv(&next_rank, 1, MPI_INT, 0, gridSave->iam, gridSave->comm, status);
    //     }

    //     printf("%d:last_topolvl %ld, next_topolvl %ld, last_rank %d, next_rank %d, maxTopoLevel %ld\n", gridSave->iam, last_topoLvl, next_topoLvl, last_rank, next_rank, maxTopoLevel);
    //     MPI_Barrier(gridSave->comm);
                
    // }

    #endif    

    /* Process supernodal etree level by level */
    for (int topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
    // for (int_t topoLvl = 0; topoLvl < 1; ++topoLvl)
    {

        #ifdef Torch
        int_t nb;     
        #ifdef SuperLargeScale
        nb = CEILING(nsupers, grid->npcol);
        for(int_t i=0; i<nb; i++)
        {        
            Llu->isUsed_Lnzval_bc_ptr[i][nPart]=topoLvl;
        }

        nb = CEILING(nsupers, grid->nprow);
        for(int_t i=0; i<nb; i++)
        {
            Llu->isUsed_Unzval_br_ptr[i][nPart]=topoLvl;
        }   
        #endif 
        #endif
       
        // printf("(%d) factor level %d, maxTopoLevel %d, %e\n",grid3d->iam,topoLvl,maxTopoLevel, SuperLU_timer_()); fflush(stdout);
        
        /* code */
        int k_st = eTreeTopLims[topoLvl];
        int k_end = eTreeTopLims[topoLvl + 1];
        
        #ifdef Torch_0312
        for (int k0 = k_st; k0 < k_end; ++k0)
        {
            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 - k_st;

            if (!factored_D[k])
            {
                /*If LU panels from GPU are not reduced then reduce
		  them before diagonal factorization*/
                if (!gpuLUreduced[k] && superlu_acc_offload)
                {
                    double tt_start1 = SuperLU_timer_();
                    dinitD2Hreduce(k, d2Hred, last_flag,
                                     HyP, sluGPU, grid, LUstruct, SCT);
                    int_t copyL_kljb = d2Hred->copyL_kljb;
                    int_t copyU_kljb = d2Hred->copyU_kljb;

                    if (copyL_kljb || copyU_kljb)
                        SCT->PhiMemCpyCounter++;
                    dsendLUpanelGPU2HOST(k, d2Hred, sluGPU);
                    /*
                        Reduce the LU panels from GPU
                    */
                    dreduceGPUlu(last_flag, d2Hred, sluGPU, SCT, grid,
		    		     LUstruct);

                    gpuLUreduced[k] = 1;
                    SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
                }

                double t1 = SuperLU_timer_();
                /* Factor diagonal block on CPU */
                // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
                //                 options, thresh, LUstruct, stat, info, SCT);

                SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
            }
        } /* for all nodes in this level */                 

        for (int_t i = 0; i < NUMPART; i++)
        {
            for (int k0 = k_st; k0 < k_end; ++k0)
            {
                int k = perm_c_supno[k0]; // direct computation no perm_c_supno
                int offset = k0 - k_st;

                if (!factored_D[k])
                {
                    /*If LU panels from GPU are not reduced then reduce
            them before diagonal factorization*/               

                    double t1 = SuperLU_timer_();
                    /* Factor diagonal block on CPU */
                    // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
                    //                 options, thresh, LUstruct, stat, info, SCT);
    #if 0
            sDiagFactIBCast(k,  dFBufs[offset], factStat, comReqss[offset], grid,
                            options, thresh, LUstruct, stat, info, SCT, tag_ub);
    #else
                    idLDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor, dFBufs[offset]->BlockLFactor,
                                    factStat->IrecvPlcd_D,
                                    comReqss[offset]->U_diag_blk_recv_req,
                                    comReqss[offset]->L_diag_blk_recv_req,
                                    comReqss[offset]->U_diag_blk_send_req,
                                    comReqss[offset]->L_diag_blk_send_req,
                                    grid, options, thresh, LUstruct, stat, info, SCT, tag_ub, i);
    #endif
                    SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
                }
            } /* for all nodes in this level */ 

            for (int k0 = k_st; k0 < k_end; ++k0)
            {
                int k = perm_c_supno[k0]; // direct computation no perm_c_supno
                int offset = k0 - k_st;

                if (!factored_D[k])
                {
                    /*If LU panels from GPU are not reduced then reduce
            them before diagonal factorization*/               

                    double t1 = SuperLU_timer_();
                    /* Factor diagonal block on CPU */
                    // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
                    //                 options, thresh, LUstruct, stat, info, SCT);
    #if 0
            sDiagFactIBCast(k,  dFBufs[offset], factStat, comReqss[offset], grid,
                            options, thresh, LUstruct, stat, info, SCT, tag_ub);
    #else
                    idUDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor, dFBufs[offset]->BlockLFactor,
                                    factStat->IrecvPlcd_D,
                                    comReqss[offset]->U_diag_blk_recv_req,
                                    comReqss[offset]->L_diag_blk_recv_req,
                                    comReqss[offset]->U_diag_blk_send_req,
                                    comReqss[offset]->L_diag_blk_send_req,
                                    grid, options, thresh, LUstruct, stat, info, SCT, tag_ub, i);
    #endif
                    SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
                }
            } /* for all nodes in this level */ 
        }

        
        #else
        /* Process all the nodes in 'topoLvl': diagonal factorization */
        for (int k0 = k_st; k0 < k_end; ++k0)
        {
            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 - k_st;

            if (!factored_D[k])
            {
                /*If LU panels from GPU are not reduced then reduce
		  them before diagonal factorization*/
                if (!gpuLUreduced[k] && superlu_acc_offload)
                {
                    double tt_start1 = SuperLU_timer_();
                    dinitD2Hreduce(k, d2Hred, last_flag,
                                     HyP, sluGPU, grid, LUstruct, SCT);
                    int_t copyL_kljb = d2Hred->copyL_kljb;
                    int_t copyU_kljb = d2Hred->copyU_kljb;

                    if (copyL_kljb || copyU_kljb)
                        SCT->PhiMemCpyCounter++;
                    dsendLUpanelGPU2HOST(k, d2Hred, sluGPU);
                    /*
                        Reduce the LU panels from GPU
                    */
                    dreduceGPUlu(last_flag, d2Hred, sluGPU, SCT, grid,
		    		     LUstruct);

                    gpuLUreduced[k] = 1;
                    SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
                }

                double t1 = SuperLU_timer_();
                /* Factor diagonal block on CPU */
                // sDiagFactIBCast(k, dFBufs[offset], factStat, comReqss[offset], grid,
                //                 options, thresh, LUstruct, stat, info, SCT);
#if 0
        sDiagFactIBCast(k,  dFBufs[offset], factStat, comReqss[offset], grid,
                        options, thresh, LUstruct, stat, info, SCT, tag_ub);
#else
                dDiagFactIBCast(k, k, dFBufs[offset]->BlockUFactor, dFBufs[offset]->BlockLFactor,
                                factStat->IrecvPlcd_D,
                                comReqss[offset]->U_diag_blk_recv_req,
                                comReqss[offset]->L_diag_blk_recv_req,
                                comReqss[offset]->U_diag_blk_send_req,
                                comReqss[offset]->L_diag_blk_send_req,
                                grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif
                SCT->pdgstrf2_timer += (SuperLU_timer_() - t1);
            }
        } /* for all nodes in this level */ 
        #endif 
        
        // printf("%d: .. SparseFactor_GPU: after diag factorization,%e\n", grid->iam, SuperLU_timer_()); fflush(stdout);        

        double t_apt = SuperLU_timer_(); /* Async Pipe Timer */

        /* Process all the nodes in 'topoLvl': panel updates on CPU */
        
        for (int k0 = k_st; k0 < k_end; ++k0)
        {
            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 - k_st;

            /*L update */
            if (factored_L[k] == 0)
            {
#if 0
		sLPanelUpdate(k, dFBufs[offset], factStat, comReqss[offset],
			      grid, LUstruct, SCT);
#else
                dLPanelUpdate(k, factStat->IrecvPlcd_D, factStat->factored_L,
                              comReqss[offset]->U_diag_blk_recv_req,
                              dFBufs[offset]->BlockUFactor, grid, LUstruct, SCT);
#endif

                factored_L[k] = 1;
            }
            /*U update*/
            if (factored_U[k] == 0)
            {
#if 0
		sUPanelUpdate(k, ldt, dFBufs[offset], factStat, comReqss[offset],
			      scuBufs, packLUInfo, grid, LUstruct, stat, SCT);
#else
                dUPanelUpdate(k, factStat->factored_U, comReqss[offset]->L_diag_blk_recv_req,
                              dFBufs[offset]->BlockLFactor, scuBufs->bigV, ldt,
                              packLUInfo->Ublock_info, grid, LUstruct, stat, SCT);
#endif
                factored_U[k] = 1;
            }            
            
            
        } /* end panel update */

        // printf("%d: .. after CPU panel updates. numLA %d, %e\n", grid->iam, numLA, SuperLU_timer_()); fflush(stdout);

        #ifdef SuperLargeScale
        Llu->core_status = InCore;
        #endif 

        #ifdef Torch_0312
        #pragma omp parallel for
        for (int_t i = 0; i < nsupers; ++i)
        {
            /* code */                
            tmpIbcastPanel_L[i] = IbcastPanel_L[i];
            tmpIbcastPanel_U[i] = IbcastPanel_U[i];
        }
        /* Process all the panels in look-ahead window: 
	   broadcast L and U panels. */
        for (int k0 = k_st; k0 < SUPERLU_MIN(k_end, k_st + numLA); ++k0)
        {
            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 % numLA;
            /* diagonal factorization */

            /*L Ibcast*/
            if (IbcastPanel_L[k] == 0)
            {
#if 0
                sIBcastRecvLPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
                dIRecvLPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_req,
                                  comReqss[offset]->recv_req, LUvsbs[offset]->Lsub_buf,
                                  LUvsbs[offset]->Lval_buf, factStat->factored,
                                  grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_L[k] = 1; /*for consistancy; unused later*/
            }

            /*U Ibcast*/
            if (IbcastPanel_U[k] == 0)
            {
#if 0
                sIBcastRecvUPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
                dIRecvUPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_requ,
                                  comReqss[offset]->recv_requ, LUvsbs[offset]->Usub_buf,
                                  LUvsbs[offset]->Uval_buf, grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_U[k] = 1;
            }
        } /* end for panels in look-ahead window */

        #pragma omp parallel for
        for (int_t j = 0; j < nsupers; ++j)
        {
            /* code */                
            IbcastPanel_L[j] = tmpIbcastPanel_L[j];
            IbcastPanel_U[j] = tmpIbcastPanel_U[j];
        } 

        /* Process all the panels in look-ahead window: 
	   broadcast L and U panels. */
        for (int k0 = k_st; k0 < SUPERLU_MIN(k_end, k_st + numLA); ++k0)
        {
            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 % numLA;
            /* diagonal factorization */

            /*L Ibcast*/
            if (IbcastPanel_L[k] == 0)
            {
#if 0
                sIBcastRecvLPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
                dIBcastLPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_req,
                                  comReqss[offset]->recv_req, LUvsbs[offset]->Lsub_buf,
                                  LUvsbs[offset]->Lval_buf, factStat->factored,
                                  grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_L[k] = 1; /*for consistancy; unused later*/
            }

            /*U Ibcast*/
            if (IbcastPanel_U[k] == 0)
            {
#if 0
                sIBcastRecvUPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
                dIBcastUPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_requ,
                                  comReqss[offset]->recv_requ, LUvsbs[offset]->Usub_buf,
                                  LUvsbs[offset]->Uval_buf, grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_U[k] = 1;
            }
        } /* end for panels in look-ahead window */
        #else
        /* Process all the panels in look-ahead window: 
	   broadcast L and U panels. */
        for (int k0 = k_st; k0 < SUPERLU_MIN(k_end, k_st + numLA); ++k0)
        {
            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 % numLA;
            /* diagonal factorization */ 

            /*L Ibcast*/
            if (IbcastPanel_L[k] == 0)
            {
#if 0
                sIBcastRecvLPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
                dIBcastRecvLPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_req,
                                  comReqss[offset]->recv_req, LUvsbs[offset]->Lsub_buf,
                                  LUvsbs[offset]->Lval_buf, factStat->factored,
                                  grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_L[k] = 1; /*for consistancy; unused later*/
            }

            /*U Ibcast*/
            if (IbcastPanel_U[k] == 0)
            {
#if 0
                sIBcastRecvUPanel( k, comReqss[offset],  LUvsbs[offset],
                                   msgss[offset], factStat, grid, LUstruct, SCT, tag_ub );
#else
                dIBcastRecvUPanel(k, k, msgss[offset]->msgcnt, comReqss[offset]->send_requ,
                                  comReqss[offset]->recv_requ, LUvsbs[offset]->Usub_buf,
                                  LUvsbs[offset]->Uval_buf, grid, LUstruct, SCT, tag_ub);
#endif
                IbcastPanel_U[k] = 1;
            }
        } /* end for panels in look-ahead window */
        #endif

        // printf("%d: .. after CPU look-ahead updates, %e\n", grid->iam, SuperLU_timer_()); fflush(stdout);

        // if (topoLvl) SCT->tAsyncPipeTail += SuperLU_timer_() - t_apt;
        SCT->tAsyncPipeTail += (SuperLU_timer_() - t_apt);
        
        /* Process all the nodes in level 'topoLvl': Schur complement update
	   (no MPI communication)  */
        for (int k0 = k_st; k0 < k_end; ++k0)
        {
            
            // printf("%d:%d,%d,%d,%e\n",grid3d->iam,k0,k_st,k_end,SuperLU_timer_()); fflush(stdout);

            int k = perm_c_supno[k0]; // direct computation no perm_c_supno
            int offset = k0 % numLA;

            double tsch = SuperLU_timer_();            

#if 0
            sWaitL(k, comReqss[offset], msgss[offset], grid, LUstruct, SCT);
            /*Wait for U panel*/
            sWaitU(k, comReqss[offset], msgss[offset], grid, LUstruct, SCT);
#else
            dWaitL(k, msgss[offset]->msgcnt, msgss[offset]->msgcntU,
                   comReqss[offset]->send_req, comReqss[offset]->recv_req,
                   grid, LUstruct, SCT);
            dWaitU(k, msgss[offset]->msgcnt, comReqss[offset]->send_requ,
                   comReqss[offset]->recv_requ, grid, LUstruct, SCT);
#endif                  

            int_t LU_nonempty = dSchurComplementSetupGPU(k,
                                                         msgss[offset], packLUInfo,
                                                         myIperm, gIperm_c_supno, perm_c_supno,
                                                         gEtreeInfo, fNlists, scuBufs,
                                                         LUvsbs[offset], grid, LUstruct, HyP);
            // initializing D2H data transfer. D2H = Device To Host.
            int_t jj_cpu; /* limit between CPU and GPU */
            
#if 1
            if (superlu_acc_offload)
            {
                jj_cpu = HyP->num_u_blks_Phi; // -1 ??
                HyP->offloadCondition = 1;
            }
            else
            {
                /* code */
                HyP->offloadCondition = 0;
                jj_cpu = 0;
            }

#else
            if (superlu_acc_offload)
            {
                jj_cpu = getAccUPartition(HyP);

                if (jj_cpu > 0)
                    jj_cpu = HyP->num_u_blks_Phi;

                /* Sherry force this --> */
                jj_cpu = HyP->num_u_blks_Phi; // -1 ??
                HyP->offloadCondition = 1;
            }
            else
            {
                jj_cpu = 0;
            }
#endif

            // int_t jj_cpu = HyP->num_u_blks_Phi-1;
            // if (HyP->Rnbrow > 0 && jj_cpu>=0)
            //     HyP->offloadCondition = 1;
            // else
            //     HyP->offloadCondition = 0;
            //     jj_cpu=0;
#if 0
	    if ( HyP->offloadCondition ) {
	    printf("(%d) k=%d, nub=%d, nub_host=%d, nub_phi=%d, jj_cpu %d, offloadCondition %d\n",
		   grid3d->iam, k, HyP->num_u_blks+HyP->num_u_blks_Phi ,
		   HyP->num_u_blks, HyP->num_u_blks_Phi,
		   jj_cpu, HyP->offloadCondition);
	    fflush(stdout);
	    }
#endif            

            scuStatUpdate(SuperSize(k), HyP, SCT, stat);
            
            int_t offload_condition = HyP->offloadCondition;
            uPanelInfo_t *uPanelInfo = packLUInfo->uPanelInfo;
            lPanelInfo_t *lPanelInfo = packLUInfo->lPanelInfo;
            int_t *lsub = lPanelInfo->lsub;
            int_t *usub = uPanelInfo->usub;
            int *indirect = fNlists->indirect;
            int *indirect2 = fNlists->indirect2;

            /* Schur Complement Update */

            int_t knsupc = SuperSize(k);
            int_t klst = FstBlockC(k + 1);

            double *bigV = scuBufs->bigV;
            double *bigU = scuBufs->bigU;

            double t1 = SuperLU_timer_();

#pragma omp parallel /* Look-ahead update on CPU */
            {
                int_t thread_id = omp_get_thread_num();

#pragma omp for
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks; ++ij)
                {
                    int_t j = ij / HyP->lookAheadBlk;
                    int_t lb = ij % HyP->lookAheadBlk;
                    dblock_gemm_scatterTopLeft(lb, j, bigV, knsupc, klst, lsub,
                                               usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                }                

#pragma omp for
                for (int_t ij = 0; ij < HyP->lookAheadBlk * HyP->num_u_blks_Phi; ++ij)
                {
                    int_t j = ij / HyP->lookAheadBlk;
                    int_t lb = ij % HyP->lookAheadBlk;
                    dblock_gemm_scatterTopRight(lb, j, bigV, knsupc, klst, lsub,
                                                usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                }

#pragma omp for
                for (int_t ij = 0; ij < HyP->RemainBlk * HyP->num_u_blks; ++ij)
                {
                    int_t j = ij / HyP->RemainBlk;
                    int_t lb = ij % HyP->RemainBlk;
                    dblock_gemm_scatterBottomLeft(lb, j, bigV, knsupc, klst, lsub,
                                                  usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                } /* for int_t ij = ... */
            }     /* end parallel region ... end look-ahead update */

            SCT->lookaheadupdatetimer += (SuperLU_timer_() - t1);

            // printf("%d: ... after look-ahead update, topoLvl %d\t maxTopoLevel %d, %e\n", grid->iam, topoLvl, maxTopoLevel, SuperLU_timer_()); fflush(stdout);

            /* Reduce the L & U panels from GPU to CPU.       */
            #ifdef Torch
            if (topoLvl < maxTopoLevel - 1 && gEtreeInfo->setree[k]<nsupers)
            #else
            if (topoLvl < maxTopoLevel - 1)
            #endif
            { /* Not the root */
                int_t k_parent = gEtreeInfo->setree[k];
                gEtreeInfo->numChildLeft[k_parent]--;
                #ifdef Torch
                if (gEtreeInfo->numChildLeft[k_parent] == 0 && k_parent < nnodes)
                #else
                if (gEtreeInfo->numChildLeft[k_parent] == 0)
                #endif
                { /* if k is the last child in this level */

                    int_t k0_parent = myIperm[k_parent];
                    
                    if (k0_parent > 0)
                    {
                        /* code */
                        //      printf("Before assert: iam %d, k %d, k_parent %d, k0_parent %d, nnodes %d\n", grid3d->iam, k, k_parent, k0_parent, nnodes); fflush(stdout);
                        //	      exit(-1);
                        assert(k0_parent < nnodes);
                        int offset = k0_parent - k_end;
                        if (!gpuLUreduced[k_parent] && superlu_acc_offload)
                        {
                            double tt_start1 = SuperLU_timer_();

                            dinitD2Hreduce(k_parent, d2Hred, last_flag,
                                          HyP, sluGPU, grid, LUstruct, SCT);
                            int_t copyL_kljb = d2Hred->copyL_kljb;
                            int_t copyU_kljb = d2Hred->copyU_kljb;

                            if (copyL_kljb || copyU_kljb)
                                SCT->PhiMemCpyCounter++;
                            dsendLUpanelGPU2HOST(k_parent, d2Hred, sluGPU);

                            /* Reduce the LU panels from GPU */
                            dreduceGPUlu(last_flag, d2Hred,
                                        sluGPU, SCT, grid, LUstruct);

                            gpuLUreduced[k_parent] = 1;
                            SCT->PhiMemCpyTimer += SuperLU_timer_() - tt_start1;
                        }

                        /* Factorize diagonal block on CPU */
#if 0
                        sDiagFactIBCast(k_parent,  dFBufs[offset], factStat,
					comReqss[offset], grid, options, thresh,
					LUstruct, stat, info, SCT, tag_ub);
#else
                        
                        dDiagFactIBCast(k_parent, k_parent, dFBufs[offset]->BlockUFactor,
                                        dFBufs[offset]->BlockLFactor, factStat->IrecvPlcd_D,
                                        comReqss[offset]->U_diag_blk_recv_req,
                                        comReqss[offset]->L_diag_blk_recv_req,
                                        comReqss[offset]->U_diag_blk_send_req,
                                        comReqss[offset]->L_diag_blk_send_req,
                                        grid, options, thresh, LUstruct, stat, info, SCT, tag_ub);
#endif
                        factored_D[k_parent] = 1;
                    } /* end if k0_parent > 0 */
                } /* end if all children are done */                
            }     /* end if non-root */      
           
            // printf("%d:Reduce the L & U panels from GPU to CPU.%d,%d,%d,%e\n",grid3d->iam,k0,k_st,k_end, SuperLU_timer_()); fflush(stdout);

#pragma omp parallel
            {
                /* Master thread performs Schur complement update on GPU. */
#pragma omp master
                {
                    if (superlu_acc_offload)
                    {
                        int thread_id = omp_get_thread_num();
                        double t1 = SuperLU_timer_();

                        if (offload_condition)
                        {
                            SCT->datatransfer_count++;
                            int streamId = k0 % nCudaStreams;

                            /*wait for previous offload to get finished*/
                            if (sluGPU->lastOffloadStream[streamId] != -1)
                            {
                                dwaitGPUscu(streamId, sluGPU, SCT);
                                sluGPU->lastOffloadStream[streamId] = -1;
                            }

                            int_t Remain_lbuf_send_size = knsupc * HyP->Rnbrow;
                            int_t bigu_send_size = jj_cpu < 1 ? 0 : HyP->ldu_Phi * HyP->Ublock_info_Phi[jj_cpu - 1].full_u_cols;
                            assert(bigu_send_size < HyP->bigu_size);

                            /* !! Sherry add the test to avoid seg_fault inside
                                  sendSCUdataHost2GPU */
                            if (bigu_send_size > 0)
                            {
                                dsendSCUdataHost2GPU(streamId, lsub, usub,
                                              bigU, bigu_send_size,
                                              Remain_lbuf_send_size, sluGPU, HyP);

                                sluGPU->lastOffloadStream[streamId] = k0;
                                int_t usub_len = usub[2];
                                int_t lsub_len = lsub[1] + BC_HEADER + lsub[0] * LB_DESCRIPTOR;
                                //{printf("... before SchurCompUpdate_GPU, bigu_send_size %d\n", bigu_send_size); fflush(stdout);}
                                
                                dSchurCompUpdate_GPU(
                                    streamId, 0, jj_cpu, klst, knsupc, HyP->Rnbrow, HyP->RemainBlk,
                                    Remain_lbuf_send_size, bigu_send_size, HyP->ldu_Phi, HyP->num_u_blks_Phi,
                                    HyP->buffer_size, lsub_len, usub_len, ldt, k0, sluGPU, grid);
                            } /* endif bigu_send_size > 0 */

                            // sendLUpanelGPU2HOST( k0, d2Hred, sluGPU);

                            SCT->schurPhiCallCount++;
                            HyP->jj_cpu = jj_cpu;
                            updateDirtyBit(k0, HyP, grid);
                        } /* endif (offload_condition) */

                        double t2 = SuperLU_timer_();
                        SCT->SchurCompUdtThreadTime[thread_id * CACHE_LINE_SIZE] += (double)(t2 - t1); /* not used */
                        SCT->CPUOffloadTimer += (double)(t2 - t1);                                     // Sherry added

                    } /* endif (superlu_acc_offload) */

                } /* end omp master thread */                 

#pragma omp for
                /* The following update is on CPU. Should not be necessary now,
		   because we set jj_cpu equal to num_u_blks_Phi.      		*/
                for (int_t ij = 0; ij < HyP->RemainBlk * (HyP->num_u_blks_Phi - jj_cpu); ++ij)
                {
                    //printf(".. WARNING: should NOT get here\n");
                    int_t j = ij / HyP->RemainBlk + jj_cpu;
                    int_t lb = ij % HyP->RemainBlk;
                    dblock_gemm_scatterBottomRight(lb, j, bigV, knsupc, klst, lsub,
                                                   usub, ldt, indirect, indirect2, HyP, LUstruct, grid, SCT, stat);
                } /* for int_t ij = ... */

            } /* end omp parallel region */
            
            // printf("\n%d:Master thread performs Schur complement update on GPU,%d,%d,%d,%e\n",grid3d->iam,k0,k_st,k_end,SuperLU_timer_()); fflush(stdout);

            //SCT->NetSchurUpTimer += SuperLU_timer_() - tsch;

            // finish waiting for diag block send
            int_t abs_offset = k0 - k_st;
#if 0
            sWait_LUDiagSend(k,  comReqss[abs_offset], grid, SCT);
#else
            Wait_LUDiagSend(k, comReqss[abs_offset]->U_diag_blk_send_req,
                            comReqss[abs_offset]->L_diag_blk_send_req,
                            grid, SCT);
#endif         

            #ifdef SuperLargeScaleGPU    

            if (Llu->isRecordingForGPU == LoadedRecordForGPU)
            {
                if (sluGPU->A_gpu->nexttopoLvl_Lnzval != maxTopoLevel - 1 || sluGPU->A_gpu->nextk0_Lnzval != sluGPU->A_gpu->UsedOrder_Lnzval[Llu->ilvl][sluGPU->A_gpu->UsedOrder_Lnzval[Llu->ilvl][0] - 1])
                {
                   if ((k0 < k_end - 1 && sluGPU->A_gpu->nexttopoLvl_Lnzval == topoLvl && sluGPU->A_gpu->nextk0_Lnzval == k0 + 1 - k_st) || (k0 == k_end - 1 && sluGPU->A_gpu->nextk0_Lnzval == 0))
                   {
                       printf("1:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Lnzval, sluGPU->A_gpu->nexttopoLvl_Lnzval, k0 - k_st, topoLvl);
                       buffermemoryrefresh_Lnzval(sluGPU->A_gpu, LUstruct, nsupers, grid3d->npcol);
                       printf("2:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Lnzval, sluGPU->A_gpu->nexttopoLvl_Lnzval, k0 - k_st, topoLvl);
                   }
                   
                }
                else
                {
                    if (sluGPU->A_gpu->nexttopoLvl_Lnzval == topoLvl && k0 == k_end - 1)
                    {
                        printf("3:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Lnzval, sluGPU->A_gpu->nexttopoLvl_Lnzval, k0 - k_st, topoLvl);
                        buffermemoryrefresh_Lnzval(sluGPU->A_gpu, LUstruct, nsupers, grid3d->npcol);
                        printf("4:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Lnzval, sluGPU->A_gpu->nexttopoLvl_Lnzval, k0 - k_st, topoLvl);
                    }
                    
                }


                if (sluGPU->A_gpu->nexttopoLvl_Unzval != maxTopoLevel - 1 || sluGPU->A_gpu->nextk0_Unzval != sluGPU->A_gpu->UsedOrder_Unzval[Llu->ilvl][sluGPU->A_gpu->UsedOrder_Unzval[Llu->ilvl][0] - 1])
                {
                   if ((k0 < k_end - 1 && sluGPU->A_gpu->nexttopoLvl_Unzval == topoLvl && sluGPU->A_gpu->nextk0_Unzval == k0 + 1 - k_st) || (k0 == k_end - 1 && sluGPU->A_gpu->nextk0_Unzval == 0))
                   {  
                       printf("5:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Unzval, sluGPU->A_gpu->nexttopoLvl_Unzval, k0 - k_st, topoLvl);                     
                       buffermemoryrefresh_Unzval(sluGPU->A_gpu, LUstruct, nsupers, grid3d->npcol);
                       printf("6:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Unzval, sluGPU->A_gpu->nexttopoLvl_Unzval, k0 - k_st, topoLvl);
                   }
                   
                }
                else
                {
                    if (sluGPU->A_gpu->nexttopoLvl_Unzval == topoLvl && k0 == k_end - 1)
                    {     
                        printf("7:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Unzval, sluGPU->A_gpu->nexttopoLvl_Unzval, k0 - k_st, topoLvl);                   
                        buffermemoryrefresh_Unzval(sluGPU->A_gpu, LUstruct, nsupers, grid3d->npcol);
                        printf("8:%ld %ld %ld %ld\n", sluGPU->A_gpu->nextk0_Unzval, sluGPU->A_gpu->nexttopoLvl_Unzval, k0 - k_st, topoLvl);
                    }
                    
                }
                
                
            }
            
            #endif
           
            #ifdef Torch_0312
            #pragma omp parallel for
            for (int_t i = 0; i < nsupers; ++i)
            {
                /* code */                
                tmpIbcastPanel_L[i] = IbcastPanel_L[i];
                tmpIbcastPanel_U[i] = IbcastPanel_U[i];
            }

            /*Schedule next I bcasts within look-ahead window */
            for (int next_k0 = k0 + 1; next_k0 < SUPERLU_MIN(k0 + 1 + numLA, nnodes); ++next_k0)
            {               

                /* code */
                int_t next_k = perm_c_supno[next_k0];
                int_t offset = next_k0 % numLA;

                /*L Ibcast*/
                if (IbcastPanel_L[next_k] == 0 && factored_L[next_k])
                {
#if 0
                    sIBcastRecvLPanel( next_k, comReqss[offset], 
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
                    dIRecvLPanel(next_k, next_k, msgss[offset]->msgcnt,
                                      comReqss[offset]->send_req, comReqss[offset]->recv_req,
                                      LUvsbs[offset]->Lsub_buf, LUvsbs[offset]->Lval_buf,
                                      factStat->factored, grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_L[next_k] = 1; /*will be used later*/
                }
                /*U Ibcast*/
                if (IbcastPanel_U[next_k] == 0 && factored_U[next_k])
                {
#if 0
                    sIBcastRecvUPanel( next_k, comReqss[offset],
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
                    dIRecvUPanel(next_k, next_k, msgss[offset]->msgcnt,
                                      comReqss[offset]->send_requ, comReqss[offset]->recv_requ,
                                      LUvsbs[offset]->Usub_buf, LUvsbs[offset]->Uval_buf,
                                      grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_U[next_k] = 1;
                }
            } /* end for look-ahead window */

            #pragma omp parallel for
            for (int_t j = 0; j < nsupers; ++j)
            {
                /* code */                
                IbcastPanel_L[j] = tmpIbcastPanel_L[j];
                IbcastPanel_U[j] = tmpIbcastPanel_U[j];
            } 

            /*Schedule next I bcasts within look-ahead window */
            for (int next_k0 = k0 + 1; next_k0 < SUPERLU_MIN(k0 + 1 + numLA, nnodes); ++next_k0)
            {               

                /* code */
                int_t next_k = perm_c_supno[next_k0];
                int_t offset = next_k0 % numLA;

                /*L Ibcast*/
                if (IbcastPanel_L[next_k] == 0 && factored_L[next_k])
                {
#if 0
                    sIBcastRecvLPanel( next_k, comReqss[offset], 
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
                    dIBcastLPanel(next_k, next_k, msgss[offset]->msgcnt,
                                      comReqss[offset]->send_req, comReqss[offset]->recv_req,
                                      LUvsbs[offset]->Lsub_buf, LUvsbs[offset]->Lval_buf,
                                      factStat->factored, grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_L[next_k] = 1; /*will be used later*/
                }
                
            } /* end for look-ahead window */

            /*Schedule next I bcasts within look-ahead window */
            for (int next_k0 = k0 + 1; next_k0 < SUPERLU_MIN(k0 + 1 + numLA, nnodes); ++next_k0)
            {               

                /* code */
                int_t next_k = perm_c_supno[next_k0];
                int_t offset = next_k0 % numLA;
                
                /*U Ibcast*/
                if (IbcastPanel_U[next_k] == 0 && factored_U[next_k])
                {
#if 0
                    sIBcastRecvUPanel( next_k, comReqss[offset],
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
                    dIBcastUPanel(next_k, next_k, msgss[offset]->msgcnt,
                                      comReqss[offset]->send_requ, comReqss[offset]->recv_requ,
                                      LUvsbs[offset]->Usub_buf, LUvsbs[offset]->Uval_buf,
                                      grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_U[next_k] = 1;
                }
            } /* end for look-ahead window */
            #else
            /*Schedule next I bcasts within look-ahead window */
            for (int next_k0 = k0 + 1; next_k0 < SUPERLU_MIN(k0 + 1 + numLA, nnodes); ++next_k0)
            {               

                /* code */
                int_t next_k = perm_c_supno[next_k0];
                int_t offset = next_k0 % numLA;                

                /*L Ibcast*/
                if (IbcastPanel_L[next_k] == 0 && factored_L[next_k])
                {
#if 0
                    sIBcastRecvLPanel( next_k, comReqss[offset], 
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
                    dIBcastRecvLPanel(next_k, next_k, msgss[offset]->msgcnt,
                                      comReqss[offset]->send_req, comReqss[offset]->recv_req,
                                      LUvsbs[offset]->Lsub_buf, LUvsbs[offset]->Lval_buf,
                                      factStat->factored, grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_L[next_k] = 1; /*will be used later*/
                }
                /*U Ibcast*/
                if (IbcastPanel_U[next_k] == 0 && factored_U[next_k])
                {
#if 0
                    sIBcastRecvUPanel( next_k, comReqss[offset],
				       LUvsbs[offset], msgss[offset], factStat,
				       grid, LUstruct, SCT, tag_ub );
#else
                    dIBcastRecvUPanel(next_k, next_k, msgss[offset]->msgcnt,
                                      comReqss[offset]->send_requ, comReqss[offset]->recv_requ,
                                      LUvsbs[offset]->Usub_buf, LUvsbs[offset]->Uval_buf,
                                      grid, LUstruct, SCT, tag_ub);
#endif
                    IbcastPanel_U[next_k] = 1;
                }
            } /* end for look-ahead window */
            #endif
            
            // printf("\n%d: .end for look-ahead , %e\n", grid->iam, SuperLU_timer_()); fflush(stdout);
   
            if (topoLvl < maxTopoLevel - 1) /* not root */
            {
                /*look-ahead LU factorization*/
                int kx_st = eTreeTopLims[topoLvl + 1];
                int kx_end = eTreeTopLims[topoLvl + 2];

                #ifdef Torch_0312

                #pragma omp parallel for
                for (int_t i = 0; i < nsupers; ++i)
                {
                    /* code */                
                    tmpIbcastPanel_L[i] = IbcastPanel_L[i];
                    tmpIbcastPanel_U[i] = IbcastPanel_U[i];
                    tmpfactored_L[i] = factored_L[i];
                    tmpfactored_U[i] = factored_U[i];
                } 

                for (int k0x = kx_st; k0x < kx_end; k0x++)
                {
                    /* code */
                    int kx = perm_c_supno[k0x];
                    int offset = k0x - kx_st;
                    if (IrecvPlcd_D[kx] && !factored_L[kx])
                    {
                        /*check if received*/
                        int_t recvUDiag = checkRecvUDiag(kx, comReqss[offset],
                                                         grid, SCT);
                        recvUDiagMatrix[offset][kx] = recvUDiag;
                        if (recvUDiag)
                        {
#if 0
                            sLPanelTrSolve( kx,  dFBufs[offset],
                                            factStat, comReqss[offset],
                                            grid, LUstruct, SCT);
#else
                            dLPanelTrSolve(kx, factStat->factored_L,
                                           dFBufs[offset]->BlockUFactor, grid, LUstruct);
#endif

                            factored_L[kx] = 1;

                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_L[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within look-ahead window
                                factored_L[kx])
                            {
                                int_t offset1 = k0x % numLA;
#if 0
                                sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
                                                   msgss[offset1], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
                                dIRecvLPanel(kx, kx, msgss[offset1]->msgcnt,
                                                  comReqss[offset1]->send_req,
                                                  comReqss[offset1]->recv_req,
                                                  LUvsbs[offset1]->Lsub_buf,
                                                  LUvsbs[offset1]->Lval_buf,
                                                  factStat->factored,
                                                  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_L[kx] = 1; /*will be used later*/
                            }
                        }
                    }

                    if (IrecvPlcd_D[kx] && !factored_U[kx])
                    {
                        /*check if received*/
                        int_t recvLDiag = checkRecvLDiag(kx, comReqss[offset],
                                                         grid, SCT);
                        recvLDiagMatrix[offset][kx] = recvLDiag;
                        if (recvLDiag)
                        {
#if 0
                            sUPanelTrSolve( kx, ldt, dFBufs[offset], scuBufs, packLUInfo,
                                            grid, LUstruct, stat, SCT);
#else
                            dUPanelTrSolve(kx, dFBufs[offset]->BlockLFactor,
                                           scuBufs->bigV,
                                           ldt, packLUInfo->Ublock_info,
                                           grid, LUstruct, stat, SCT);
#endif
                            factored_U[kx] = 1;
                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_U[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within lookahead window
                                factored_U[kx])
                            {
                                int_t offset = k0x % numLA;
#if 0
                                sIBcastRecvUPanel( kx, comReqss[offset],
						   LUvsbs[offset],
						   msgss[offset], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
                                dIRecvUPanel(kx, kx, msgss[offset]->msgcnt,
                                                  comReqss[offset]->send_requ,
                                                  comReqss[offset]->recv_requ,
                                                  LUvsbs[offset]->Usub_buf,
                                                  LUvsbs[offset]->Uval_buf,
                                                  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_U[kx] = 1; /*will be used later*/
                            }
                        }
                    }
                    
                } /* end look-ahead */

                #pragma omp parallel for
                for (int_t j = 0; j < nsupers; ++j)
                {
                    /* code */                
                    IbcastPanel_L[j] = tmpIbcastPanel_L[j];
                    IbcastPanel_U[j] = tmpIbcastPanel_U[j];
                    factored_L[j] = tmpfactored_L[j];
                    factored_U[j] = tmpfactored_U[j];
                } 

                for (int k0x = kx_st; k0x < kx_end; k0x++)
                {
                    /* code */
                    int kx = perm_c_supno[k0x];
                    int offset = k0x - kx_st;
                    if (IrecvPlcd_D[kx] && !factored_L[kx])
                    {
                        /*check if received*/
                        int_t recvUDiag = recvUDiagMatrix[offset][kx];
                        
                        if (recvUDiag)
                        {

                            factored_L[kx] = 1;

                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_L[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within look-ahead window
                                factored_L[kx])
                            {
                                int_t offset1 = k0x % numLA;
#if 0
                                sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
                                                   msgss[offset1], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
                                dIBcastLPanel(kx, kx, msgss[offset1]->msgcnt,
                                                  comReqss[offset1]->send_req,
                                                  comReqss[offset1]->recv_req,
                                                  LUvsbs[offset1]->Lsub_buf,
                                                  LUvsbs[offset1]->Lval_buf,
                                                  factStat->factored,
                                                  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_L[kx] = 1; /*will be used later*/
                            }
                        }
                    }

                    if (IrecvPlcd_D[kx] && !factored_U[kx])
                    {
                        /*check if received*/
                        int_t recvLDiag = recvLDiagMatrix[offset][kx];
                        if (recvLDiag)
                        {
                            factored_U[kx] = 1;
                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_U[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within lookahead window
                                factored_U[kx])
                            {
                                int_t offset = k0x % numLA;
#if 0
                                sIBcastRecvUPanel( kx, comReqss[offset],
						   LUvsbs[offset],
						   msgss[offset], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
                                dIBcastUPanel(kx, kx, msgss[offset]->msgcnt,
                                                  comReqss[offset]->send_requ,
                                                  comReqss[offset]->recv_requ,
                                                  LUvsbs[offset]->Usub_buf,
                                                  LUvsbs[offset]->Uval_buf,
                                                  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_U[kx] = 1; /*will be used later*/
                            }
                        }
                    }
                    
                } /* end look-ahead */                    

    //             
    //             for (int i = 0; i < grid3d->npcol*grid3d->nprow; i++)
    //             {
    //                 MPI_Barrier(grid->comm);
    //                 if(grid->iam==i){
    //                     int_t *recvUDiagMatrix = intCalloc_dist(kx_end - kx_st);
    //                     for (int k0x = kx_st; k0x < kx_end; k0x++)
    //                     {
    //                         /* code */
    //                         int kx = perm_c_supno[k0x];
    //                         int offset = k0x - kx_st;
    //                         if (IrecvPlcd_D[kx] && !factored_L[kx])
    //                         {
    //                             /*check if received*/
    //                             int_t recvUDiag = checkRecvUDiag(kx, comReqss[offset],
    //                                                             grid, SCT);
    //                             recvUDiagMatrix[k0x - kx_st] = recvUDiag;
    //                             if (recvUDiag)
    //                             {
    // #if 0
    //                                 sLPanelTrSolve( kx,  dFBufs[offset],
    //                                                 factStat, comReqss[offset],
    //                                                 grid, LUstruct, SCT);
    // #else
    //                                 dLPanelTrSolve(kx, factStat->factored_L,
    //                                             dFBufs[offset]->BlockUFactor, grid, LUstruct);
    // #endif

    //                                 factored_L[kx] = 1;

    //                                 /*check if an L_Ibcast is possible*/

    // //                                 if (IbcastPanel_L[kx] == 0 &&
    // //                                     k0x - k0 < numLA + 1 && // is within look-ahead window
    // //                                     factored_L[kx])
    // //                                 {
    // //                                     int_t offset1 = k0x % numLA;
    // // #if 0
    // //                                     sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
    // //                                                     msgss[offset1], factStat,
    // //                             grid, LUstruct, SCT, tag_ub);
    // // #else
    // //                                     dIBcastRecvLPanel(kx, kx, msgss[offset1]->msgcnt,
    // //                                                     comReqss[offset1]->send_req,
    // //                                                     comReqss[offset1]->recv_req,
    // //                                                     LUvsbs[offset1]->Lsub_buf,
    // //                                                     LUvsbs[offset1]->Lval_buf,
    // //                                                     factStat->factored,
    // //                                                     grid, LUstruct, SCT, tag_ub);
    // // #endif
    // //                                     IbcastPanel_L[kx] = 1; /*will be used later*/
    // //                                 }
    //                             }
    //                         }
                            
    //                     } /* end look-ahead */ 

    //                     for (int k0x = kx_st; k0x < kx_end; k0x++)
    //                     {
    //                         /* code */
    //                         int kx = perm_c_supno[k0x];
    //                         int offset = k0x - kx_st;
    //                         if (IrecvPlcd_D[kx] && !factored_L[kx])
    //                         {
    //                             /*check if received*/
    //                             int_t recvUDiag = recvUDiagMatrix[k0x - kx_st];
    //                             if (recvUDiag)
    //                             {

    //                                 /*check if an L_Ibcast is possible*/

    //                                 if (IbcastPanel_L[kx] == 0 &&
    //                                     k0x - k0 < numLA + 1 && // is within look-ahead window
    //                                     factored_L[kx])
    //                                 {
    //                                     int_t offset1 = k0x % numLA;
    // #if 0
    //                                     sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
    //                                                     msgss[offset1], factStat,
    //                             grid, LUstruct, SCT, tag_ub);
    // #else
    //                                     dIBcastRecvLPanel_Lsub(kx, kx, msgss[offset1]->msgcnt,
    //                                                     comReqss[offset1]->send_req,
    //                                                     comReqss[offset1]->recv_req,
    //                                                     LUvsbs[offset1]->Lsub_buf,
    //                                                     LUvsbs[offset1]->Lval_buf,
    //                                                     factStat->factored,
    //                                                     grid, LUstruct, SCT, tag_ub);
    // #endif
    //                                     // IbcastPanel_L[kx] = 1; /*will be used later*/
    //                                 }
    //                             }
    //                         }
                            
    //                     } /* end look-ahead */ 

    //                     for (int k0x = kx_st; k0x < kx_end; k0x++)
    //                     {
    //                         /* code */
    //                         int kx = perm_c_supno[k0x];
    //                         int offset = k0x - kx_st;
    //                         if (IrecvPlcd_D[kx] && !factored_L[kx])
    //                         {
    //                             /*check if received*/
    //                             int_t recvUDiag = recvUDiagMatrix[k0x - kx_st];
    //                             if (recvUDiag)
    //                             {

    //                                 /*check if an L_Ibcast is possible*/

    //                                 if (IbcastPanel_L[kx] == 0 &&
    //                                     k0x - k0 < numLA + 1 && // is within look-ahead window
    //                                     factored_L[kx])
    //                                 {
    //                                     int_t offset1 = k0x % numLA;
    // #if 0
    //                                     sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
    //                                                     msgss[offset1], factStat,
    //                             grid, LUstruct, SCT, tag_ub);
    // #else
    //                                     dIBcastRecvLPanel_Lval(kx, kx, msgss[offset1]->msgcnt,
    //                                                     comReqss[offset1]->send_req,
    //                                                     comReqss[offset1]->recv_req,
    //                                                     LUvsbs[offset1]->Lsub_buf,
    //                                                     LUvsbs[offset1]->Lval_buf,
    //                                                     factStat->factored,
    //                                                     grid, LUstruct, SCT, tag_ub);
    // #endif
    //                                     IbcastPanel_L[kx] = 1; /*will be used later*/
    //                                 }
    //                             }
    //                         }
                            
    //                     } /* end look-ahead */

    //                     SUPERLU_FREE(recvUDiagMatrix);
    //                 }
    //             } 

    //             // printf("%d:break2\n",grid3d->iam); 

    //             for (int i = 0; i < grid3d->npcol*grid3d->nprow; i++)
    //             {
    //                 MPI_Barrier(grid->comm);
    //                 if(grid->iam==i){
    //                     for (int k0x = kx_st; k0x < kx_end; k0x++)
    //                     {
    //                         /* code */
    //                         int kx = perm_c_supno[k0x];
    //                         int offset = k0x - kx_st;                    

    //                         if (IrecvPlcd_D[kx] && !factored_U[kx])
    //                         {
    //                             /*check if received*/
    //                             int_t recvLDiag = checkRecvLDiag(kx, comReqss[offset],
    //                                                             grid, SCT);
    //                             if (recvLDiag)
    //                             {
    //     #if 0
    //                                 sUPanelTrSolve( kx, ldt, dFBufs[offset], scuBufs, packLUInfo,
    //                                                 grid, LUstruct, stat, SCT);
    //     #else
    //                                 dUPanelTrSolve(kx, dFBufs[offset]->BlockLFactor,
    //                                             scuBufs->bigV,
    //                                             ldt, packLUInfo->Ublock_info,
    //                                             grid, LUstruct, stat, SCT);
    //     #endif
    //                                 factored_U[kx] = 1;
    //                                 /*check if an L_Ibcast is possible*/

    //                                 if (IbcastPanel_U[kx] == 0 &&
    //                                     k0x - k0 < numLA + 1 && // is within lookahead window
    //                                     factored_U[kx])
    //                                 {
    //                                     int_t offset = k0x % numLA;
    //     #if 0
    //                                     sIBcastRecvUPanel( kx, comReqss[offset],
    //                             LUvsbs[offset],
    //                             msgss[offset], factStat,
    //                             grid, LUstruct, SCT, tag_ub);
    //     #else
    //                                     dIBcastRecvUPanel(kx, kx, msgss[offset]->msgcnt,
    //                                                     comReqss[offset]->send_requ,
    //                                                     comReqss[offset]->recv_requ,
    //                                                     LUvsbs[offset]->Usub_buf,
    //                                                     LUvsbs[offset]->Uval_buf,
    //                                                     grid, LUstruct, SCT, tag_ub);
    //     #endif
    //                                     IbcastPanel_U[kx] = 1; /*will be used later*/
    //                                 }
    //                             }
    //                         }
                            
    //                     } /* end look-ahead */   
    //                 }
    //             }    

                // printf("%d:break3\n",grid3d->iam);

                #else
                for (int k0x = kx_st; k0x < kx_end; k0x++)
                {
                    /* code */
                    int kx = perm_c_supno[k0x];
                    int offset = k0x - kx_st;
                    if (IrecvPlcd_D[kx] && !factored_L[kx])
                    {
                        /*check if received*/
                        int_t recvUDiag = checkRecvUDiag(kx, comReqss[offset],
                                                         grid, SCT);
                        if (recvUDiag)
                        {
#if 0
                            sLPanelTrSolve( kx,  dFBufs[offset],
                                            factStat, comReqss[offset],
                                            grid, LUstruct, SCT);
#else
                            dLPanelTrSolve(kx, factStat->factored_L,
                                           dFBufs[offset]->BlockUFactor, grid, LUstruct);
#endif

                            factored_L[kx] = 1;

                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_L[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within look-ahead window
                                factored_L[kx])
                            {
                                int_t offset1 = k0x % numLA;
#if 0
                                sIBcastRecvLPanel( kx, comReqss[offset1], LUvsbs[offset1],
                                                   msgss[offset1], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
                                dIBcastRecvLPanel(kx, kx, msgss[offset1]->msgcnt,
                                                  comReqss[offset1]->send_req,
                                                  comReqss[offset1]->recv_req,
                                                  LUvsbs[offset1]->Lsub_buf,
                                                  LUvsbs[offset1]->Lval_buf,
                                                  factStat->factored,
                                                  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_L[kx] = 1; /*will be used later*/
                            }
                        }
                    }

                    if (IrecvPlcd_D[kx] && !factored_U[kx])
                    {
                        /*check if received*/
                        int_t recvLDiag = checkRecvLDiag(kx, comReqss[offset],
                                                         grid, SCT);
                        if (recvLDiag)
                        {
#if 0
                            sUPanelTrSolve( kx, ldt, dFBufs[offset], scuBufs, packLUInfo,
                                            grid, LUstruct, stat, SCT);
#else
                            dUPanelTrSolve(kx, dFBufs[offset]->BlockLFactor,
                                           scuBufs->bigV,
                                           ldt, packLUInfo->Ublock_info,
                                           grid, LUstruct, stat, SCT);
#endif
                            factored_U[kx] = 1;
                            /*check if an L_Ibcast is possible*/

                            if (IbcastPanel_U[kx] == 0 &&
                                k0x - k0 < numLA + 1 && // is within lookahead window
                                factored_U[kx])
                            {
                                int_t offset = k0x % numLA;
#if 0
                                sIBcastRecvUPanel( kx, comReqss[offset],
						   LUvsbs[offset],
						   msgss[offset], factStat,
						   grid, LUstruct, SCT, tag_ub);
#else
                                dIBcastRecvUPanel(kx, kx, msgss[offset]->msgcnt,
                                                  comReqss[offset]->send_requ,
                                                  comReqss[offset]->recv_requ,
                                                  LUvsbs[offset]->Usub_buf,
                                                  LUvsbs[offset]->Uval_buf,
                                                  grid, LUstruct, SCT, tag_ub);
#endif
                                IbcastPanel_U[kx] = 1; /*will be used later*/
                            }
                        }
                    }
                    
                } /* end look-ahead */       
                #endif         

            } /* end if non-root level */ 

            /* end Schur complement update */
            SCT->NetSchurUpTimer += SuperLU_timer_() - tsch;

        } /* end Schur update for all the nodes in level 'topoLvl' */
        
        #ifdef Torch

        #ifdef Torch_debug
        sum_Used_Lnzval_bc_ptr = 0;
        sum_Used_Unzval_br_ptr = 0;
        sum_Lnzval_bc_ptr_ilen = 0;
        sum_Unzval_br_ptr_ilen = 0;
        sum_Lnzval_bc_ptr_len = 0;
        sum_Unzval_br_ptr_len = 0;

        int_t iPart = 0;
        nb = CEILING(nsupers, grid->npcol);
        
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:sum_Used_Lnzval_bc_ptr),reduction(+:sum_Lnzval_bc_ptr_ilen), reduction(+:sum_Lnzval_bc_ptr_len)
        #endif
        for(int_t i=0; i<nb; i++)
        {
            sum_Used_Lnzval_bc_ptr += (GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[i][iPart]) != Unused);
            sum_Lnzval_bc_ptr_ilen+=Llu->Lnzval_bc_ptr_ilen[i]*(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[i][iPart]) != Unused);
            sum_Lnzval_bc_ptr_len+=Llu->Lnzval_bc_ptr_ilen[i];
        }       

        nb = CEILING(nsupers, grid->nprow);

        
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:sum_Used_Unzval_br_ptr), reduction(+:sum_Unzval_br_ptr_ilen), reduction(+:sum_Unzval_br_ptr_len)
        #endif
        for(int_t i=0; i<nb; i++)
        {            
            sum_Used_Unzval_br_ptr += (GetVectorStatus(Llu->isUsed_Unzval_br_ptr[i][iPart]) != Unused);
            sum_Unzval_br_ptr_ilen+=Llu->Unzval_br_ptr_ilen[i]*(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[i][iPart]) != Unused);
            sum_Unzval_br_ptr_len+=Llu->Unzval_br_ptr_ilen[i];                
        }
        
        // printf("%d: %ld, %ld, %ld, %ld, %f%%, %ld, %ld, %ld, %f%%, %f%%\n", grid3d->iam, topoLvl, sum_Used_Lnzval_bc_ptr, sum_Lnzval_bc_ptr_ilen, sum_Lnzval_bc_ptr_len, (double)sum_Lnzval_bc_ptr_ilen/sum_Lnzval_bc_ptr_len*100, sum_Used_Unzval_br_ptr, sum_Unzval_br_ptr_ilen, sum_Unzval_br_ptr_len, (double)sum_Unzval_br_ptr_ilen/sum_Unzval_br_ptr_len*100, (double)(sum_Lnzval_bc_ptr_ilen+sum_Unzval_br_ptr_ilen)/(sum_Lnzval_bc_ptr_len+sum_Unzval_br_ptr_len)*100);

        #endif

        // #ifndef SuperLargeScale        

        // // for (int i = 0; i < grid3d->npcol*grid3d->nprow; i++)
        // // {
        // //     MPI_Barrier(grid->comm);
        // //     #ifdef SuperLargeScale_limit
        // //     if(Llu->isSave && grid->iam==i && ((sum_Lnzval_bc_ptr_ilen + sum_Unzval_br_ptr_ilen) * sizeof(double) * 1.0e-9 >= CPUMemLimit || topoLvl == maxTopoLevel -1))
        // //     #else
        // //     if(Llu->isSave && grid->iam==i)
        // //     #endif
        // //     {
        // //         printf("%d: topoLvl %ld\n", grid3d->iam, topoLvl);
        // //         save_Changed_LUstruct_txt(LUstruct->Llu->ncol, grid3d, LUstruct);
        // //     }
        // // }

        // if (MPI_COMM_NULL != grid3d->gridsave.comm)
        // {            
        //     int saveflag = 0;

        //     if (last_rank != -1)
        //     {
        //         if (topoLvl < last_topoLvl)
        //         {
        //             MPI_Recv(&saveflag, 1, MPI_INT, last_rank, 0, grid3d->gridsave.comm, status);
        //         }
                
        //     }

        //     printf("%d: topoLvl %ld %ld\n", grid3d->iam, topoLvl, maxTopoLevel);
            
        //     #ifdef SuperLargeScale_limit
        //     if(Llu->isSave && ((sum_Lnzval_bc_ptr_ilen + sum_Unzval_br_ptr_ilen) * sizeof(double) * 1.0e-9 >= CPUMemLimit || topoLvl == maxTopoLevel -1))
        //     #else
        //     if(Llu->isSave)
        //     #endif
        //     {
        //         printf("%d: topoLvl %ld\n", grid3d->iam, topoLvl);
                
        //         save_Changed_LUstruct_txt(LUstruct->Llu->ncol, grid3d, LUstruct); 
                
        //     }

        //     if (next_rank != grid3d->nSave)
        //     {
        //         MPI_Send(&saveflag, 1, MPI_INT, next_rank, 0, grid3d->gridsave.comm);
        //     }
        // }
        
        // Llu->core_status = OutOfCore;

        // #ifdef SuperLargeScale_limit
        // Llu->core_status = InCore;
        // #endif
        // #endif
        #endif

        #ifdef test_0111
        for (int i = 0; i < grid3d->npcol*grid3d->nprow; i++)
        {
            
            /* code */
            if(grid->iam==i){

                printf("%d: %ld, %ld, %ld, %ld, %f%%, %ld, %ld, %ld, %f%%, %f%%\n", grid3d->iam, topoLvl, sum_Used_Lnzval_bc_ptr, sum_Lnzval_bc_ptr_ilen, sum_Lnzval_bc_ptr_len, (double)sum_Lnzval_bc_ptr_ilen/sum_Lnzval_bc_ptr_len*100, sum_Used_Unzval_br_ptr, sum_Unzval_br_ptr_ilen, sum_Unzval_br_ptr_len, (double)sum_Unzval_br_ptr_ilen/sum_Unzval_br_ptr_len*100, (double)(sum_Lnzval_bc_ptr_ilen+sum_Unzval_br_ptr_ilen)/(sum_Lnzval_bc_ptr_len+sum_Unzval_br_ptr_len)*100);

            }
           
        }
        #endif
        
    } /* end for all levels of the tree */ 

    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid3d->iam, "Exit dsparseTreeFactor_ASYNC_GPU()");
    #endif

    #ifdef Torch_0312
    SUPERLU_FREE(tmpIbcastPanel_L);
    SUPERLU_FREE(tmpIbcastPanel_U);
    SUPERLU_FREE(tmpfactored_L);
    SUPERLU_FREE(tmpfactored_U);
    for (int_t i = 0; i < maxoffset; i++)
    {
        SUPERLU_FREE(recvLDiagMatrix[i]);
        SUPERLU_FREE(recvUDiagMatrix[i]);
    }
    SUPERLU_FREE(recvLDiagMatrix);
    SUPERLU_FREE(recvUDiagMatrix);
    #endif

    return 0;
} /* end dsparseTreeFactor_ASYNC_GPU */

#endif // matching: enable GPU
