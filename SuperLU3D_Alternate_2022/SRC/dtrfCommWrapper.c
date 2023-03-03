/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Communication wrapper routines for 2D factorization.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
 * Lawrence Berkeley National Lab, Georgia Institute of Technology,
 * Oak Ridge National Lab
 * May 12, 2021
 */

#include "superlu_ddefs.h"

#undef Torch
#undef DEBUGlevel

#if 0
#include "pdgstrf3d.h"
#include "trfCommWrapper.h"
#endif

//#include "cblas.h"

int_t dDiagFactIBCast(int_t k,  int_t k0,      // supernode to be factored
                     double *BlockUFactor,
                     double *BlockLFactor,
                     int_t* IrecvPlcd_D,
                     MPI_Request *U_diag_blk_recv_req,
                     MPI_Request *L_diag_blk_recv_req,
                     MPI_Request *U_diag_blk_send_req,
                     MPI_Request *L_diag_blk_send_req,
                     gridinfo_t *grid,
                     superlu_dist_options_t *options,
                     double thresh,
                     dLUstruct_t *LUstruct,
                     SuperLUStat_t *stat, int *info,
                     SCT_t *SCT,
		     int tag_ub
                    )
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dDiagFactIBCast()");
    #endif
    
    // unpacking variables
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);

    //xsup for supersize

    /*Place Irecvs first*/
    // if (IrecvPlcd_D[k] == 0 )
    // {
    int_t nsupc = SuperSize (k);
    if (mycol == kcol && iam != pkk)
    {
        dIRecv_UDiagBlock(k0, BlockUFactor,  /*pointer for the diagonal block*/
                         nsupc * nsupc, krow,
                         U_diag_blk_recv_req, grid, SCT, tag_ub);
    }

    if (myrow == krow && iam != pkk)
    {
        dIRecv_LDiagBlock(k0, BlockLFactor,  /*pointer for the diagonal block*/
                         nsupc * nsupc, kcol,
                         L_diag_blk_recv_req, grid, SCT, tag_ub);
    }
    IrecvPlcd_D[k] = 1;
    // }

    /*DiagFact and send */
    // if ( factored_D[k] == 0 )
    // {

    // int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    // int_t krow = PROW (k, grid);
    // int_t kcol = PCOL (k, grid);
    /*factorize the leaf node and broadcast them
     process row and process column*/
    if (iam == pkk)
    {
        // printf("Entering factorization %d\n", k);
        // int_t offset = (k0 - k_st); // offset is input
        /*factorize A[kk]*/
        Local_Dgstrf2(options, k, thresh,
                      BlockUFactor, /*factored U is over writen here*/
                      Glu_persist, grid, Llu, stat, info, SCT);

        /*Pack L[kk] into blockLfactor*/
        dPackLBlock(k, BlockLFactor, Glu_persist, grid, Llu);

        /*Isend U blocks to the process row*/
        int_t nsupc = SuperSize(k);
        dISend_UDiagBlock(k0, BlockLFactor,
                         nsupc * nsupc, U_diag_blk_send_req , grid, tag_ub);

        /*Isend L blocks to the process col*/
        dISend_LDiagBlock(k0, BlockLFactor,
                         nsupc * nsupc, L_diag_blk_send_req, grid, tag_ub);
        SCT->commVolFactor += 1.0 * nsupc * nsupc * (Pr + Pc);
        
    }
    // }
    return 0;
}

#ifdef Torch
int_t dLDiagFactIBCast(int_t k,  int_t k0,      // supernode to be factored
                     double *BlockUFactor,
                     double *BlockLFactor,
                     int_t* IrecvPlcd_D,
                     MPI_Request *U_diag_blk_recv_req,
                     MPI_Request *L_diag_blk_recv_req,
                     MPI_Request *U_diag_blk_send_req,
                     MPI_Request *L_diag_blk_send_req,
                     gridinfo_t *grid,
                     superlu_dist_options_t *options,
                     double thresh,
                     dLUstruct_t *LUstruct,
                     SuperLUStat_t *stat, int *info,
                     SCT_t *SCT,
		     int tag_ub
                    )
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dDiagFactIBCast()");
    #endif
    
    // unpacking variables
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);

    //xsup for supersize

    /*Place Irecvs first*/
    // if (IrecvPlcd_D[k] == 0 )
    // {
    int_t nsupc = SuperSize (k);
    // if (mycol == kcol && iam != pkk)
    // {
    //     dIRecv_UDiagBlock(k0, BlockUFactor,  /*pointer for the diagonal block*/
    //                      nsupc * nsupc, krow,
    //                      U_diag_blk_recv_req, grid, SCT, tag_ub);
    // }

    if (myrow == krow && iam != pkk)
    {
        dIRecv_LDiagBlock(k0, BlockLFactor,  /*pointer for the diagonal block*/
                         nsupc * nsupc, kcol,
                         L_diag_blk_recv_req, grid, SCT, tag_ub);
    }
    // IrecvPlcd_D[k] = 1;
    // }

    /*DiagFact and send */
    // if ( factored_D[k] == 0 )
    // {

    // int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    // int_t krow = PROW (k, grid);
    // int_t kcol = PCOL (k, grid);
    /*factorize the leaf node and broadcast them
     process row and process column*/
    if (iam == pkk)
    {
        // printf("Entering factorization %d\n", k);
        // int_t offset = (k0 - k_st); // offset is input
        /*factorize A[kk]*/
        Local_Dgstrf2(options, k, thresh,
                      BlockUFactor, /*factored U is over writen here*/
                      Glu_persist, grid, Llu, stat, info, SCT);

        /*Pack L[kk] into blockLfactor*/
        dPackLBlock(k, BlockLFactor, Glu_persist, grid, Llu);

        /*Isend U blocks to the process row*/
        int_t nsupc = SuperSize(k);
        // dISend_UDiagBlock(k0, BlockLFactor,
        //                  nsupc * nsupc, U_diag_blk_send_req , grid, tag_ub);

        /*Isend L blocks to the process col*/
        dISend_LDiagBlock(k0, BlockLFactor,
                         nsupc * nsupc, L_diag_blk_send_req, grid, tag_ub);
        SCT->commVolFactor += 1.0 * nsupc * nsupc * (Pr + Pc);
    }
    // }
    return 0;
}

int_t idLDiagFactIBCast(int_t k,  int_t k0,      // supernode to be factored
                     double *BlockUFactor,
                     double *BlockLFactor,
                     int_t* IrecvPlcd_D,
                     MPI_Request *U_diag_blk_recv_req,
                     MPI_Request *L_diag_blk_recv_req,
                     MPI_Request *U_diag_blk_send_req,
                     MPI_Request *L_diag_blk_send_req,
                     gridinfo_t *grid,
                     superlu_dist_options_t *options,
                     double thresh,
                     dLUstruct_t *LUstruct,
                     SuperLUStat_t *stat, int *info,
                     SCT_t *SCT,
		             int tag_ub,
                     int_t i
                    )
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dDiagFactIBCast()");
    #endif
    
    // unpacking variables
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);

    //xsup for supersize

    /*Place Irecvs first*/
    // if (IrecvPlcd_D[k] == 0 )
    // {
    int_t nsupc = SuperSize (k);
    // if (mycol == kcol && iam != pkk)
    // {
    //     dIRecv_UDiagBlock(k0, BlockUFactor,  /*pointer for the diagonal block*/
    //                      nsupc * nsupc, krow,
    //                      U_diag_blk_recv_req, grid, SCT, tag_ub);
    // }

    if (myrow == krow && iam != pkk)
    {
        int_t length, index;
        if ((nsupc * nsupc)%NUMPART)
        {
            length = floor((nsupc * nsupc)/NUMPART);
            index = i * length;
            if (i == NUMPART - 1)
            {
                length = nsupc * nsupc - i * length;
            }            
        }
        else
        {
            length = (nsupc * nsupc)/NUMPART;
            index = i * length;
        }
        dIRecv_LDiagBlock(k0, BlockLFactor + index,  /*pointer for the diagonal block*/
                         length, kcol,
                         L_diag_blk_recv_req, grid, SCT, tag_ub);
    }
    // IrecvPlcd_D[k] = 1;
    // }

    /*DiagFact and send */
    // if ( factored_D[k] == 0 )
    // {

    // int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    // int_t krow = PROW (k, grid);
    // int_t kcol = PCOL (k, grid);
    /*factorize the leaf node and broadcast them
     process row and process column*/
    if (iam == pkk)
    {
        if (i == 0)
        {
            // printf("Entering factorization %d\n", k);
            // int_t offset = (k0 - k_st); // offset is input
            /*factorize A[kk]*/
            Local_Dgstrf2(options, k, thresh,
                        BlockUFactor, /*factored U is over writen here*/
                        Glu_persist, grid, Llu, stat, info, SCT);

            /*Pack L[kk] into blockLfactor*/
            dPackLBlock(k, BlockLFactor, Glu_persist, grid, Llu);
        }
        
        
        /*Isend U blocks to the process row*/
        int_t nsupc = SuperSize(k);
        // dISend_UDiagBlock(k0, BlockLFactor,
        //                  nsupc * nsupc, U_diag_blk_send_req , grid, tag_ub);

        /*Isend L blocks to the process col*/
        int_t length, index;
        if ((nsupc * nsupc)%NUMPART)
        {
            length = floor((nsupc * nsupc)/NUMPART);
            index = i * length;
            if (i == NUMPART - 1)
            {
                length = nsupc * nsupc - i * length;
            }            
        }
        else
        {
            length = (nsupc * nsupc)/NUMPART;
            index = i * length;
        }
        
        dISend_LDiagBlock(k0, BlockLFactor + index,
                         length, L_diag_blk_send_req, grid, tag_ub);
        
        if (i == 0)
        {
            SCT->commVolFactor += 1.0 * nsupc * nsupc * (Pr + Pc);
        }
    }
    // }
    return 0;
}

int_t dUDiagFactIBCast(int_t k,  int_t k0,      // supernode to be factored
                     double *BlockUFactor,
                     double *BlockLFactor,
                     int_t* IrecvPlcd_D,
                     MPI_Request *U_diag_blk_recv_req,
                     MPI_Request *L_diag_blk_recv_req,
                     MPI_Request *U_diag_blk_send_req,
                     MPI_Request *L_diag_blk_send_req,
                     gridinfo_t *grid,
                     superlu_dist_options_t *options,
                     double thresh,
                     dLUstruct_t *LUstruct,
                     SuperLUStat_t *stat, int *info,
                     SCT_t *SCT,
		     int tag_ub
                    )
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dDiagFactIBCast()");
    #endif
    
    // unpacking variables
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);

    //xsup for supersize

    /*Place Irecvs first*/
    // if (IrecvPlcd_D[k] == 0 )
    // {
    int_t nsupc = SuperSize (k);
    if (mycol == kcol && iam != pkk)
    {
        dIRecv_UDiagBlock(k0, BlockUFactor,  /*pointer for the diagonal block*/
                         nsupc * nsupc, krow,
                         U_diag_blk_recv_req, grid, SCT, tag_ub);
    }

    // if (myrow == krow && iam != pkk)
    // {
    //     dIRecv_LDiagBlock(k0, BlockLFactor,  /*pointer for the diagonal block*/
    //                      nsupc * nsupc, kcol,
    //                      L_diag_blk_recv_req, grid, SCT, tag_ub);
    // }
    IrecvPlcd_D[k] = 1;
    // }

    /*DiagFact and send */
    // if ( factored_D[k] == 0 )
    // {

    // int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    // int_t krow = PROW (k, grid);
    // int_t kcol = PCOL (k, grid);
    /*factorize the leaf node and broadcast them
     process row and process column*/
    if (iam == pkk)
    {
        // printf("Entering factorization %d\n", k);
        // int_t offset = (k0 - k_st); // offset is input
        /*factorize A[kk]*/
        // Local_Dgstrf2(options, k, thresh,
        //               BlockUFactor, /*factored U is over writen here*/
        //               Glu_persist, grid, Llu, stat, info, SCT);

        /*Pack L[kk] into blockLfactor*/
        // dPackLBlock(k, BlockLFactor, Glu_persist, grid, Llu);

        /*Isend U blocks to the process row*/
        int_t nsupc = SuperSize(k);
        dISend_UDiagBlock(k0, BlockLFactor,
                         nsupc * nsupc, U_diag_blk_send_req , grid, tag_ub);

        /*Isend L blocks to the process col*/
        // dISend_LDiagBlock(k0, BlockLFactor,
        //                  nsupc * nsupc, L_diag_blk_send_req, grid, tag_ub);
        SCT->commVolFactor += 1.0 * nsupc * nsupc * (Pr + Pc);
    }
    // }
    return 0;
}

int_t idUDiagFactIBCast(int_t k,  int_t k0,      // supernode to be factored
                     double *BlockUFactor,
                     double *BlockLFactor,
                     int_t* IrecvPlcd_D,
                     MPI_Request *U_diag_blk_recv_req,
                     MPI_Request *L_diag_blk_recv_req,
                     MPI_Request *U_diag_blk_send_req,
                     MPI_Request *L_diag_blk_send_req,
                     gridinfo_t *grid,
                     superlu_dist_options_t *options,
                     double thresh,
                     dLUstruct_t *LUstruct,
                     SuperLUStat_t *stat, int *info,
                     SCT_t *SCT,
		             int tag_ub,
                     int_t i
                    )
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dDiagFactIBCast()");
    #endif
    
    // unpacking variables
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    int_t kcol = PCOL (k, grid);

    //xsup for supersize

    /*Place Irecvs first*/
    // if (IrecvPlcd_D[k] == 0 )
    // {
    int_t nsupc = SuperSize (k);
    if (mycol == kcol && iam != pkk)
    {
        int_t length, index;
        if ((nsupc * nsupc)%NUMPART)
        {
            length = floor((nsupc * nsupc)/NUMPART);
            index = i * length;
            if (i == NUMPART - 1)
            {
                length = nsupc * nsupc - i * length;
            }            
        }
        else
        {
            length = (nsupc * nsupc)/NUMPART;
            index = i * length;
        }
        dIRecv_UDiagBlock(k0, BlockUFactor + index,  /*pointer for the diagonal block*/
                         length, krow,
                         U_diag_blk_recv_req, grid, SCT, tag_ub);
    }

    // if (myrow == krow && iam != pkk)
    // {
    //     dIRecv_LDiagBlock(k0, BlockLFactor,  /*pointer for the diagonal block*/
    //                      nsupc * nsupc, kcol,
    //                      L_diag_blk_recv_req, grid, SCT, tag_ub);
    // }
    IrecvPlcd_D[k] = 1;
    // }

    /*DiagFact and send */
    // if ( factored_D[k] == 0 )
    // {

    // int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    // int_t krow = PROW (k, grid);
    // int_t kcol = PCOL (k, grid);
    /*factorize the leaf node and broadcast them
     process row and process column*/
    if (iam == pkk)
    {
        // printf("Entering factorization %d\n", k);
        // int_t offset = (k0 - k_st); // offset is input
        /*factorize A[kk]*/
        // Local_Dgstrf2(options, k, thresh,
        //               BlockUFactor, /*factored U is over writen here*/
        //               Glu_persist, grid, Llu, stat, info, SCT);

        /*Pack L[kk] into blockLfactor*/
        // dPackLBlock(k, BlockLFactor, Glu_persist, grid, Llu);

        /*Isend U blocks to the process row*/
        int_t nsupc = SuperSize(k);
        int_t length, index;
        if ((nsupc * nsupc)%NUMPART)
        {
            length = floor((nsupc * nsupc)/NUMPART);
            index = i * length;
            if (i == NUMPART - 1)
            {
                length = nsupc * nsupc - i * length;
            }            
        }
        else
        {
            length = (nsupc * nsupc)/NUMPART;
            index = i * length;
        }
        dISend_UDiagBlock(k0, BlockLFactor + index,
                         length, U_diag_blk_send_req , grid, tag_ub);

        /*Isend L blocks to the process col*/
        // dISend_LDiagBlock(k0, BlockLFactor,
        //                  nsupc * nsupc, L_diag_blk_send_req, grid, tag_ub);
        if (i == 0)
        {
            SCT->commVolFactor += 1.0 * nsupc * nsupc * (Pr + Pc);
        }        
        
    }
    // }
    return 0;
}
#endif

int_t dLPanelTrSolve( int_t k,   int_t* factored_L,
		      double* BlockUFactor,
		      gridinfo_t *grid,
		      dLUstruct_t *LUstruct)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dLPanelTrSolve()");
    #endif
    double alpha = 1.0;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;

    int_t iam = grid->iam;

    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t kcol = PCOL (k, grid);
    int_t mycol = MYCOL (iam, grid);
    int nsupc = SuperSize(k);

    /*factor the L panel*/
    if (mycol == kcol  && iam != pkk)
    {
        // factored_L[k] = 1;
        int_t lk = LBj (k, grid);

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale
        
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused)){
                Llu->Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }    
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }   
        #endif
        
        
        #endif
        
        double* lusup = Llu->Lnzval_bc_ptr[lk];

        int nsupr;
        if (Llu->Lrowind_bc_ptr[lk])
            nsupr = Llu->Lrowind_bc_ptr[lk][1];
        else
            nsupr = 0;
        /*wait for communication to finish*/

        // Wait_UDiagBlock_Recv( U_diag_blk_recv_req, SCT);
        // int_t flag = 0;
        // while (flag == 0)
        // {
        //     flag = Test_UDiagBlock_Recv( U_diag_blk_recv_req, SCT);
        // }

        int_t l = nsupr;
        double* ublk_ptr = BlockUFactor;
        int ld_ujrow = nsupc;

        // unsigned long long t1 = _rdtsc();

        // #pragma omp for schedule(dynamic) nowait
#define BL  32
        for (int i = 0; i < CEILING(l, BL); ++i)
        {
            #pragma omp task
            {
                int_t off = i * BL;
                // Sherry: int_t len = MY_MIN(BL, l - i * BL);
                int len = SUPERLU_MIN(BL, l - i * BL);

                superlu_dtrsm("R", "U", "N", "N", len, nsupc, alpha,
			      ublk_ptr, ld_ujrow, &lusup[off], nsupr);                

            }
        }

        #ifdef Torch        
        #ifdef SuperLargeScale
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore && Llu->Lnzval_bc_ptr_ilen[lk]){
                if(set_iLnzval_bc_ptr_txt(lusup,lk,0,Llu->Lnzval_bc_ptr_ilen[lk],Llu)){
                    SUPERLU_FREE(lusup);
                    SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Unused);
                }
                else{
                    ABORT("failed in set_iLnzval_bc_ptr_txt() of dLPanelTrSolve(). \n");
                }
            }
            else{
                if(Llu->Lnzval_bc_ptr_ilen[lk]){
                    SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Changed);
                }
                else{
                    SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Unused);
                }
            }
        }
        #endif
        #endif
    }

    if (iam == pkk)
    {
        /* if (factored_L[k] == 0)
         { */
        /* code */
        factored_L[k] = 1;
        int_t lk = LBj (k, grid);

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused)){
                Llu->Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }
        #endif        
        
        #endif
        double* lusup = Llu->Lnzval_bc_ptr[lk];

        int nsupr;
        if (Llu->Lrowind_bc_ptr[lk]) nsupr = Llu->Lrowind_bc_ptr[lk][1];
        else nsupr = 0;

        /*factorize A[kk]*/

        int_t l = nsupr - nsupc;

        double* ublk_ptr = BlockUFactor;
        int ld_ujrow = nsupc;
        // printf("%d: L update \n",k );

#define BL  32
        // #pragma omp parallel for
        for (int i = 0; i < CEILING(l, BL); ++i)
        {
            int_t off = i * BL;
            // Sherry: int_t len = MY_MIN(BL, l - i * BL);
            int len = SUPERLU_MIN(BL, (l - i * BL));
#pragma omp task
            {
                superlu_dtrsm("R", "U", "N", "N", len, nsupc, alpha,
			      ublk_ptr, ld_ujrow, &lusup[nsupc + off], nsupr);
            }
        }

        #ifdef Torch
        #ifdef SuperLargeScale
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore && Llu->Lnzval_bc_ptr_ilen[lk]){
                if(set_iLnzval_bc_ptr_txt(lusup,lk,0,Llu->Lnzval_bc_ptr_ilen[lk],Llu)){
                    SUPERLU_FREE(lusup);
                    SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Unused);
                }
                else{
                    ABORT("failed in set_iLnzval_bc_ptr_txt() of dLPanelTrSolve(). \n");
                }
            }
            else{
                if(Llu->Lnzval_bc_ptr_ilen[lk]){
                    SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Changed);
                }
                else{
                    SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Unused);
                }
            }
        }
        #endif
        #endif
    }

    return 0;
}  /* dLPanelTrSolve */

int_t dLPanelUpdate( int_t k,  int_t* IrecvPlcd_D, int_t* factored_L,
                    MPI_Request * U_diag_blk_recv_req,
                    double* BlockUFactor,
                    gridinfo_t *grid,
                    dLUstruct_t *LUstruct, SCT_t *SCT)
{

    dUDiagBlockRecvWait( k,  IrecvPlcd_D, factored_L,
                         U_diag_blk_recv_req, grid, LUstruct, SCT);

    dLPanelTrSolve( k, factored_L, BlockUFactor, grid, LUstruct );

    return 0;
}  /* dLPanelUpdate */

#define BL  32

int_t dUPanelTrSolve( int_t k,  
                     double* BlockLFactor,
                     double* bigV,
                     int_t ldt,
                     Ublock_info_t* Ublock_info,
                     gridinfo_t *grid,
                     dLUstruct_t *LUstruct,
                     SuperLUStat_t *stat, SCT_t *SCT)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t pkk = PNUM (PROW (k, grid), PCOL (k, grid), grid);
    int_t krow = PROW (k, grid);
    int_t nsupc = SuperSize(k);

    /*factor the U panel*/
    if (myrow == krow  && iam != pkk)
    {
        int_t lk = LBi (k, grid);         /* Local block number */

        #ifdef Torch
        #ifdef SuperLargeScale
        
        if(Llu->isSave){
            if(!Llu->Unzval_br_ptr_ilen[lk]){
                return 0;
            }
        }
        else{
            if (!Llu->Unzval_br_ptr[lk])
                return 0;
        }
        #else
        if (!Llu->Unzval_br_ptr[lk])
            return 0;
        
        #endif

        #else
        if (!Llu->Unzval_br_ptr[lk])
            return 0;
        #endif

        /* Initialization. */
        int_t klst = FstBlockC (k + 1);

        int_t *usub = Llu->Ufstnz_br_ptr[lk];  /* index[] of block row U(k,:) */        

        #ifdef Torch    
        int_t iPart=0;    
        #ifdef SuperLargeScale        

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused)){
                Llu->Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
        }
        #endif       
        
        #endif
        double *uval = Llu->Unzval_br_ptr[lk];

        int_t nb = usub[0];

        // int_t nsupr = Lsub_buf[1];   /* LDA of lusup[] */
        double *lusup = BlockLFactor;

        /* Loop through all the row blocks. to get the iukp and rukp*/
        Trs2_InitUblock_info(klst, nb, Ublock_info, usub, Glu_persist, stat );

        /* Loop through all the row blocks. */
        // #pragma omp for schedule(dynamic,2) nowait
        for (int_t b = 0; b < nb; ++b)
        {
            #pragma omp task
            {
#ifdef _OPENMP	    
                int_t thread_id = omp_get_thread_num();
#else		
                int_t thread_id = 0;
#endif		
                double *tempv = bigV +  thread_id * ldt * ldt;
                dTrs2_GatherTrsmScatter(klst, Ublock_info[b].iukp, Ublock_info[b].rukp,
				       usub, uval, tempv, nsupc, nsupc, lusup, Glu_persist);
            }
        }

        #ifdef Torch
        #ifdef SuperLargeScale
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore && Llu->Unzval_br_ptr_ilen[lk]){
                if(set_iUnzval_br_ptr_txt(uval,lk,0,Llu->Unzval_br_ptr_ilen[lk],Llu)){
                    SUPERLU_FREE(uval);
                    SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Unused);
                }
                else{
                    ABORT("failed in set_iUnzval_br_ptr_txt() of dUPanelTrSolve(). \n");
                }
            }
            else{
                if(Llu->Unzval_br_ptr_ilen[lk]){
                    SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Changed);
                }
                else{
                    SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Unused);
                }
            }
        }
        #endif
        #endif
    }

    /*factor the U panel*/
    if (iam == pkk)
    {
        /* code */
        // factored_U[k] = 1;
        int_t *Lsub_buf;
        double *Lval_buf;
        int_t lk = LBj (k, grid);
        Lsub_buf = Llu->Lrowind_bc_ptr[lk];        

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale        

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused)){
                Llu->Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }
        #endif       
        
        #endif

        Lval_buf = Llu->Lnzval_bc_ptr[lk];

        /* calculate U panel */
        // PDGSTRS2 (n, k0, k, Lsub_buf, Lval_buf, Glu_persist, grid, Llu,
        //           stat, HyP->Ublock_info, bigV, ldt, SCT);

        lk = LBi (k, grid);         /* Local block number */

        #ifdef Torch
        #ifdef SuperLargeScale
        if((Llu->isSave && Llu->Unzval_br_ptr_ilen[lk]) || (!Llu->isSave && Llu->Unzval_br_ptr[lk]))
        #else
        if (Llu->Unzval_br_ptr[lk])
        #endif
        #else
        if (Llu->Unzval_br_ptr[lk])
        #endif
        {
            /* Initialization. */
            int_t klst = FstBlockC (k + 1);

            int_t *usub = Llu->Ufstnz_br_ptr[lk];  /* index[] of block row U(k,:) */

            #ifdef Torch
            int_t iPart=0;
            #ifdef SuperLargeScale           

            if(Llu->isSave){
                if(Llu->core_status == OutOfCore || (Llu->core_status == InCore && GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused)){
                    Llu->Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
                }
            }
            if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
                SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
            }
            #endif
                        
            #endif

            double *uval = Llu->Unzval_br_ptr[lk];

            int_t nb = usub[0];

            // int_t nsupr = Lsub_buf[1];   /* LDA of lusup[] */
            int_t nsupr = Lsub_buf[1];   /* LDA of lusup[] */
            double *lusup = Lval_buf;

            /* Loop through all the row blocks. to get the iukp and rukp*/
            Trs2_InitUblock_info(klst, nb, Ublock_info, usub, Glu_persist, stat );

            /* Loop through all the row blocks. */
            // printf("%d :U update \n", k);
            for (int_t b = 0; b < nb; ++b)
            {
                #pragma omp task
                {
#ifdef _OPENMP		
                    int_t thread_id = omp_get_thread_num();
#else		    
                    int_t thread_id = 0;
#endif		    
                    double *tempv = bigV +  thread_id * ldt * ldt;
                    dTrs2_GatherTrsmScatter(klst, Ublock_info[b].iukp, Ublock_info[b].rukp,
					   usub, uval, tempv, nsupc, nsupr, lusup, Glu_persist);
                }

            }

            #ifdef Torch
            #ifdef SuperLargeScale
            if(Llu->isSave){
                if(Llu->core_status == OutOfCore && Llu->Unzval_br_ptr_ilen[lk]){
                    if(set_iUnzval_br_ptr_txt(uval,lk,0,Llu->Unzval_br_ptr_ilen[lk],Llu)){
                        SUPERLU_FREE(uval);
                        SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Unused);
                    }
                    else{
                        ABORT("failed in set_iUnzval_br_ptr_txt() of dUPanelTrSolve(). \n");
                    }
                }
                else{
                    if(Llu->Unzval_br_ptr_ilen[lk]){
                        SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Changed);
                    }
                    else{
                        SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Unused);
                    }
                }
            }
            #endif
            #endif
        }

        #ifdef Torch
        #ifdef SuperLargeScale
        lk = LBj (k, grid);
        if(Llu->isSave && Llu->Lnzval_bc_ptr_ilen[lk] && Llu->core_status == OutOfCore){
            SUPERLU_FREE(Lval_buf);
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Unused);
        }
        #endif
        #endif
    }

    return 0;
} /* dUPanelTrSolve */

int_t dUPanelUpdate( int_t k,  int_t* factored_U,
                    MPI_Request * L_diag_blk_recv_req,
                    double* BlockLFactor,
                    double* bigV,
                    int_t ldt,
                    Ublock_info_t* Ublock_info,
                    gridinfo_t *grid,
                    dLUstruct_t *LUstruct,
                    SuperLUStat_t *stat, SCT_t *SCT)
{

    LDiagBlockRecvWait( k, factored_U, L_diag_blk_recv_req, grid);

    dUPanelTrSolve( k, BlockLFactor, bigV, ldt, Ublock_info, grid,
                       LUstruct, stat, SCT);
    return 0;
}

int_t dIBcastRecvLPanel(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub
)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvLPanel()");
    #endif

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol == kcol)
    {
        /*send the L panel to myrow*/

        int_t lk = LBj (k, grid);     /* Local block number. */
        int_t* lsub = Lrowind_bc_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale        
        
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvLPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
                Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }

        #endif
        
        #endif
        double* lusup = Lnzval_bc_ptr[lk];
        
        dIBcast_LPanel (k, k0, lsub, lusup, grid, msgcnt, send_req,
		       ToSendR, xsup, tag_ub);        

        if (lsub)
        {
            int_t nrbl  =   lsub[0]; /*number of L blocks */
            int_t   len   = lsub[1];       /* LDA of the nzval[] */
            int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
            int_t len2  = SuperSize(lk) * len;
            SCT->commVolFactor += 1.0 * (Pc - 1) * (len1 * sizeof(int_t) + len2 * sizeof(double));
            
        }

    }
    else
    {
        /*receive factored L panels*/
        if (ToRecv[k] >= 1)     /* Recv block column L(:,0). */
        {            
            
            /*place Irecv*/
            dIrecv_LPanel (k, k0, Lsub_buf, Lval_buf, grid, recv_req, Llu, tag_ub);
            
        }
        else
        {
            msgcnt[0] = 0;
        }

    }
    factored[k] = 0;

    return 0;
}

#ifdef Torch
int_t dIBcastRecvLPanel_Lsub(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub
)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvLPanel()");
    #endif

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol == kcol)
    {
        /*send the L panel to myrow*/

        int_t lk = LBj (k, grid);     /* Local block number. */
        int_t* lsub = Lrowind_bc_ptr[lk];
        
        dIBcast_LPanel_Lsub (k, k0, lsub, grid, msgcnt, send_req,
		       ToSendR, xsup, tag_ub);

    }
    else
    {
        /*receive factored L panels*/
        if (ToRecv[k] >= 1)     /* Recv block column L(:,0). */
        {            
            
            /*place Irecv*/
            dIrecv_LPanel_Lsub (k, k0, Lsub_buf, grid, recv_req, Llu, tag_ub);
            
        }
        else
        {
            msgcnt[0] = 0;
        }

    }
    factored[k] = 0;

    return 0;
}

int_t dIBcastRecvLPanel_Lval(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub
)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvLPanel()");
    #endif

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol == kcol)
    {
        /*send the L panel to myrow*/

        int_t lk = LBj (k, grid);     /* Local block number. */
        int_t* lsub = Lrowind_bc_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale        
        
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvLPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
                Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }

        #endif        
        #endif
        double* lusup = Lnzval_bc_ptr[lk];
        
        dIBcast_LPanel_Lval (k, k0, lsub, lusup, grid, msgcnt, send_req,
		       ToSendR, xsup, tag_ub);

        if (lsub)
        {
            int_t nrbl  =   lsub[0]; /*number of L blocks */
            int_t   len   = lsub[1];       /* LDA of the nzval[] */
            int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
            int_t len2  = SuperSize(lk) * len;
            SCT->commVolFactor += 1.0 * (Pc - 1) * (len1 * sizeof(int_t) + len2 * sizeof(double));
        }

    }
    else
    {
        /*receive factored L panels*/
        if (ToRecv[k] >= 1)     /* Recv block column L(:,0). */
        {            
            
            /*place Irecv*/
            dIrecv_LPanel_Lval (k, k0, Lsub_buf, Lval_buf, grid, recv_req, Llu, tag_ub);
            
        }
        else
        {
            msgcnt[0] = 0;
        }

    }
    factored[k] = 0;

    return 0;
}

int_t dIBcastRecvLPanel0(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub
)
{
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol == kcol)
    {
        /*send the L panel to myrow*/

        int_t lk = LBj (k, grid);     /* Local block number. */
        int_t* lsub = Lrowind_bc_ptr[lk];

        if (lsub)
        {
            int_t nrbl  =   lsub[0]; /*number of L blocks */
            int_t   len   = lsub[1];       /* LDA of the nzval[] */
            int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
            int_t len2  = SuperSize(k) * len;
            SCT->commVolFactor += 1.0 * (Pc - 1) * (len1 * sizeof(int_t) + len2 * sizeof(double));

            msgcnt[0] = len1;
            msgcnt[1] = len2;
        }

    }
    return 0;
}

int_t idIBcastRecvLPanel(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub,
    int_t i
)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvLPanel()");
    #endif

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol == kcol)
    {
        /*send the L panel to myrow*/

        int_t lk = LBj (k, grid);     /* Local block number. */
        int_t* lsub = Lrowind_bc_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale        
        
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvLPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
                Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }

        #endif
        
        #endif
        double* lusup = Lnzval_bc_ptr[lk];
        
        idIBcast_LPanel (k, k0, lsub, lusup, grid, msgcnt, send_req,
		       ToSendR, xsup, tag_ub, i);

        if (lsub && i == 0)
        {
            int_t nrbl  =   lsub[0]; /*number of L blocks */
            int_t   len   = lsub[1];       /* LDA of the nzval[] */
            int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
            int_t len2  = SuperSize(lk) * len;
            SCT->commVolFactor += 1.0 * (Pc - 1) * (len1 * sizeof(int_t) + len2 * sizeof(double));
        }

    }
    else
    {
        /*receive factored L panels*/
        if (ToRecv[k] >= 1)     /* Recv block column L(:,0). */
        {            
            
            /*place Irecv*/
            idIrecv_LPanel (k, k0, Lsub_buf, Lval_buf, grid, recv_req, Llu, tag_ub, i, msgcnt);
            
        }
        else
        {
            msgcnt[0] = 0;
        }

    }
    factored[k] = 0;

    return 0;
}

int_t dIBcastLPanel(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub
)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvLPanel()");
    #endif

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol == kcol)
    {
        /*send the L panel to myrow*/

        int_t lk = LBj (k, grid);     /* Local block number. */
        int_t* lsub = Lrowind_bc_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale        
        
        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvLPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
                Lnzval_bc_ptr[lk] = load_Lnzval_bc_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[lk][iPart], Used);
        }

        #endif
        
        #endif
        double* lusup = Lnzval_bc_ptr[lk];
        
        dIBcast_LPanel (k, k0, lsub, lusup, grid, msgcnt, send_req,
		       ToSendR, xsup, tag_ub);

        if (lsub)
        {
            int_t nrbl  =   lsub[0]; /*number of L blocks */
            int_t   len   = lsub[1];       /* LDA of the nzval[] */
            int_t len1  = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
            int_t len2  = SuperSize(lk) * len;
            SCT->commVolFactor += 1.0 * (Pc - 1) * (len1 * sizeof(int_t) + len2 * sizeof(double));
        }

    }
    factored[k] = 0;

    return 0;
}

int_t dIRecvLPanel(
    int_t k,
    int_t k0,
    int* msgcnt,
    MPI_Request *send_req,
    MPI_Request *recv_req ,
    int_t* Lsub_buf,
    double* Lval_buf,
    int_t * factored,
    gridinfo_t *grid,
    dLUstruct_t *LUstruct,
    SCT_t *SCT,
    int tag_ub
)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvLPanel()");
    #endif

    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    /* code */
    if (mycol != kcol)
    {        
        /*receive factored L panels*/
        if (ToRecv[k] >= 1)     /* Recv block column L(:,0). */
        {            
            
            /*place Irecv*/
            dIrecv_LPanel (k, k0, Lsub_buf, Lval_buf, grid, recv_req, Llu, tag_ub);
            
        }
        else
        {
            msgcnt[0] = 0;
        }

    }

    return 0;
}
#endif

int_t dIBcastRecvUPanel(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow == krow)
    {
        /*send U panel to myrow*/
        int_t   lk = LBi (k, grid);
        int_t*  usub = Ufstnz_br_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale       

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvUPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
                Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
        }
        #endif
        
        #endif

        double* uval = Unzval_br_ptr[lk];
        
        dIBcast_UPanel(k, k0, usub, uval, grid, msgcnt,
                        send_requ, ToSendD, tag_ub);

        if (usub)
        {
            /* code */
            int_t lenv = usub[1];
            int_t lens = usub[2];
            SCT->commVolFactor += 1.0 * (Pr - 1) * (lens * sizeof(int_t) + lenv * sizeof(double));
        }
        
    }
    else
    {
        /*receive U panels */
        if (ToRecv[k] == 2)     /* Recv block row U(k,:). */
        {            
            dIrecv_UPanel (k, k0, Usub_buf, Uval_buf, Llu, grid, recv_requ, tag_ub);
        }
        else
        {
            msgcnt[2] = 0;
        }
    }

    return 0;
}

#ifdef Torch
int_t dIBcastRecvUPanel_Usub(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow == krow)
    {
        /*send U panel to myrow*/
        int_t   lk = LBi (k, grid);
        int_t*  usub = Ufstnz_br_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale       

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvUPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
                Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
        }
        #endif
        
        #endif

        double* uval = Unzval_br_ptr[lk];
        
        dIBcast_UPanel_Usub(k, k0, usub, grid, msgcnt,
                        send_requ, ToSendD, tag_ub);

        if (usub)
        {
            /* code */
            int_t lenv = usub[1];
            int_t lens = usub[2];
            SCT->commVolFactor += 1.0 * (Pr - 1) * (lens * sizeof(int_t) + lenv * sizeof(double));
        }
        
    }
    else
    {
        /*receive U panels */
        if (ToRecv[k] == 2)     /* Recv block row U(k,:). */
        {            
            dIrecv_UPanel_Usub (k, k0, Usub_buf, Llu, grid, recv_requ, tag_ub);
        }
        else
        {
            msgcnt[2] = 0;
        }
    }

    return 0;
}

int_t dIBcastRecvUPanel_Uval(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow == krow)
    {
        /*send U panel to myrow*/
        int_t   lk = LBi (k, grid);
        int_t*  usub = Ufstnz_br_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale       

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvUPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
                Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
        }
        #endif
        
        #endif

        double* uval = Unzval_br_ptr[lk];
        
        dIBcast_UPanel_Uval(k, k0, usub, uval, grid, msgcnt,
                        send_requ, ToSendD, tag_ub);

        if (usub)
        {
            /* code */
            int_t lenv = usub[1];
            int_t lens = usub[2];
            SCT->commVolFactor += 1.0 * (Pr - 1) * (lens * sizeof(int_t) + lenv * sizeof(double));
        }
        
    }
    else
    {
        /*receive U panels */
        if (ToRecv[k] == 2)     /* Recv block row U(k,:). */
        {            
            dIrecv_UPanel_Uval (k, k0, Usub_buf, Uval_buf, Llu, grid, recv_requ, tag_ub);
        }
        else
        {
            msgcnt[2] = 0;
        }
    }

    return 0;
}

int_t idIBcastRecvUPanel(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub, int_t i)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow == krow)
    {
        /*send U panel to myrow*/
        int_t   lk = LBi (k, grid);
        int_t*  usub = Ufstnz_br_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale       

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvUPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
                Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
        }
        #endif
        
        #endif

        double* uval = Unzval_br_ptr[lk];
        
        idIBcast_UPanel(k, k0, usub, uval, grid, msgcnt,
                        send_requ, ToSendD, tag_ub, i);

        if (usub && i == 0)
        {
            /* code */
            int_t lenv = usub[1];
            int_t lens = usub[2];
            SCT->commVolFactor += 1.0 * (Pr - 1) * (lens * sizeof(int_t) + lenv * sizeof(double));
        }
        
    }
    else
    {
        /*receive U panels */
        if (ToRecv[k] == 2)     /* Recv block row U(k,:). */
        {            
            idIrecv_UPanel (k, k0, Usub_buf, Uval_buf, Llu, grid, recv_requ, tag_ub, i, msgcnt);
        }
        else
        {
            msgcnt[2] = 0;
        }
    }

    return 0;
}

int_t dIBcastRecvUPanel0(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow == krow)
    {
        /*send U panel to myrow*/
        int_t   lk = LBi (k, grid);
        int_t*  usub = Ufstnz_br_ptr[lk];

        if (usub)
        {
            /* code */
            int_t lenv = usub[1];
            int_t lens = usub[2];
            SCT->commVolFactor += 1.0 * (Pr - 1) * (lens * sizeof(int_t) + lenv * sizeof(double));

            msgcnt[2] = usub[2];
            msgcnt[3] = usub[1];
        }
        
    }

    return 0;
}

int_t dIBcastUPanel(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow == krow)
    {
        /*send U panel to myrow*/
        int_t   lk = LBi (k, grid);
        int_t*  usub = Ufstnz_br_ptr[lk];

        #ifdef Torch
        int_t iPart=0;
        #ifdef SuperLargeScale       

        if(Llu->isSave){
            if(Llu->core_status == OutOfCore){
                ABORT("Error core status: dIBcastRecvUPanel() should be in core status.\n");
            }
            if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
                Unzval_br_ptr[lk] = load_Unzval_br_ptr(lk,Llu);
            }
        }
        if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart]) == Unused){
            SetVectorStatus(Llu->isUsed_Unzval_br_ptr[lk][iPart], Used);
        }
        #endif
        
        #endif

        double* uval = Unzval_br_ptr[lk];
        
        dIBcast_UPanel(k, k0, usub, uval, grid, msgcnt,
                        send_requ, ToSendD, tag_ub);

        if (usub)
        {
            /* code */
            int_t lenv = usub[1];
            int_t lens = usub[2];
            SCT->commVolFactor += 1.0 * (Pr - 1) * (lens * sizeof(int_t) + lenv * sizeof(double));
        }
        
    }    

    return 0;
}

int_t dIRecvUPanel(int_t k, int_t k0, int* msgcnt,
    			     MPI_Request *send_requ,
    			     MPI_Request *recv_requ,
    			     int_t* Usub_buf, double* Uval_buf,
    			     gridinfo_t *grid, dLUstruct_t *LUstruct,
    			     SCT_t *SCT, int tag_ub)
{
    #if ( DEBUGlevel>=1 )
        CHECK_MALLOC (grid->iam, "Enter dIBcastRecvUPanel()");
    #endif
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToSendD = Llu->ToSendD;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
    if (myrow != krow)
    {        
        /*receive U panels */
        if (ToRecv[k] == 2)     /* Recv block row U(k,:). */
        {            
            dIrecv_UPanel (k, k0, Usub_buf, Uval_buf, Llu, grid, recv_requ, tag_ub);
        }
        else
        {
            msgcnt[2] = 0;
        }
    }

    return 0;
}
#endif

int_t dWaitL( int_t k, int* msgcnt, int* msgcntU,
              MPI_Request *send_req, MPI_Request *recv_req,
    	      gridinfo_t *grid, dLUstruct_t *LUstruct, SCT_t *SCT)
{
    dLocalLU_t *Llu = LUstruct->Llu;
    int** ToSendR = Llu->ToSendR;
    int* ToRecv = Llu->ToRecv;
    int_t iam = grid->iam;
    int_t mycol = MYCOL (iam, grid);
    int_t kcol = PCOL (k, grid);
    if (mycol == kcol)
    {        
        /*send the L panel to myrow*/
        Wait_LSend (k, grid, ToSendR, send_req, SCT);
        
    }
    else
    {
        /*receive factored L panels*/
        if (ToRecv[k] >= 1)     /* Recv block column L(:,0). */
        {
            /*force wait for I recv to complete*/
            dWait_LRecv( recv_req,  msgcnt, msgcntU, grid, SCT);
        }
    }

    return 0;
}

int_t dWaitU( int_t k, int* msgcnt,
              MPI_Request *send_requ, MPI_Request *recv_requ,
    	      gridinfo_t *grid, dLUstruct_t *LUstruct, SCT_t *SCT)
{
    dLocalLU_t *Llu = LUstruct->Llu;

    int* ToRecv = Llu->ToRecv;
    int* ToSendD = Llu->ToSendD;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t krow = PROW (k, grid);
    if (myrow == krow)
    {
        int_t lk = LBi (k, grid);
        if (ToSendD[lk] == YES){

            Wait_USend(send_requ, grid, SCT);
            
        }
            
    }
    else
    {
        /*receive U panels */
        if (ToRecv[k] == 2)     /* Recv block row U(k,:). */
        {
            /*force wait*/
            dWait_URecv( recv_requ, msgcnt, SCT);
        }
    }
    return 0;
}
