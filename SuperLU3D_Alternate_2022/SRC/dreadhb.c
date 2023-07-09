/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Read a DOUBLE PRECISION matrix stored in Harwell-Boeing format
 *
 * <pre>
 * -- Distributed SuperLU routine (version 1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * September 1, 1999
 * </pre>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "superlu_ddefs.h"
#include <sys/io.h>

#define FORTRAN_TXT
#include <unistd.h>
#include <math.h>
/*
 * Prototypes
 */
static void ReadVector(FILE *, int_t, int_t *, int_t, int_t);
static void dReadValues(FILE *, int_t, double *, int_t, int_t);
extern void FormFullA(int_t, int_t *, double **, int_t **, int_t **);
static int DumpLine(FILE *);
static int ParseIntFormat(char *, int_t *, int_t *);
static int ParseFloatFormat(char *, int_t *, int_t *);

// void load_ddata_markders_vector_txt(double *data,int_t n,int_t ntimestep,int_t index);
// void load_idata_markders_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index);
// void load_idata_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index);

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 * 
 * Read a DOUBLE PRECISION matrix stored in Harwell-Boeing format 
 * as described below.
 * 
 * Line 1 (A72,A8) 
 *  	Col. 1 - 72   Title (TITLE) 
 *	Col. 73 - 80  Key (KEY) 
 * 
 * Line 2 (5I14) 
 * 	Col. 1 - 14   Total number of lines excluding header (TOTCRD) 
 * 	Col. 15 - 28  Number of lines for pointers (PTRCRD) 
 * 	Col. 29 - 42  Number of lines for row (or variable) indices (INDCRD) 
 * 	Col. 43 - 56  Number of lines for numerical values (VALCRD) 
 *	Col. 57 - 70  Number of lines for right-hand sides (RHSCRD) 
 *                    (including starting guesses and solution vectors 
 *		       if present) 
 *           	      (zero indicates no right-hand side data is present) 
 *
 * Line 3 (A3, 11X, 4I14) 
 *   	Col. 1 - 3    Matrix type (see below) (MXTYPE) 
 * 	Col. 15 - 28  Number of rows (or variables) (NROW) 
 * 	Col. 29 - 42  Number of columns (or elements) (NCOL) 
 *	Col. 43 - 56  Number of row (or variable) indices (NNZERO) 
 *	              (equal to number of entries for assembled matrices) 
 * 	Col. 57 - 70  Number of elemental matrix entries (NELTVL) 
 *	              (zero in the case of assembled matrices) 
 * Line 4 (2A16, 2A20) 
 * 	Col. 1 - 16   Format for pointers (PTRFMT) 
 *	Col. 17 - 32  Format for row (or variable) indices (INDFMT) 
 *	Col. 33 - 52  Format for numerical values of coefficient matrix (VALFMT) 
 * 	Col. 53 - 72 Format for numerical values of right-hand sides (RHSFMT) 
 *
 * Line 5 (A3, 11X, 2I14) Only present if there are right-hand sides present 
 *    	Col. 1 	      Right-hand side type: 
 *	         	  F for full storage or M for same format as matrix 
 *    	Col. 2        G if a starting vector(s) (Guess) is supplied. (RHSTYP) 
 *    	Col. 3        X if an exact solution vector(s) is supplied. 
 *	Col. 15 - 28  Number of right-hand sides (NRHS) 
 *	Col. 29 - 42  Number of row indices (NRHSIX) 
 *          	      (ignored in case of unassembled matrices) 
 *
 * The three character type field on line 3 describes the matrix type. 
 * The following table lists the permitted values for each of the three 
 * characters. As an example of the type field, RSA denotes that the matrix 
 * is real, symmetric, and assembled. 
 *
 * First Character: 
 *	R Real matrix 
 *	C Complex matrix 
 *	P Pattern only (no numerical values supplied) 
 *
 * Second Character: 
 *	S Symmetric 
 *	U Unsymmetric 
 *	H Hermitian 
 *	Z Skew symmetric 
 *	R Rectangular 
 *
 * Third Character: 
 *	A Assembled 
 *	E Elemental matrices (unassembled) 
 * </pre>
 */

#if 1
void
dreadhb_dist(int iam, FILE *fp, int_t *nrow, int_t *ncol, int_t *nonz,
	     double **nzval, int_t **rowind, int_t **colptr)
{    
    register int_t i, numer_lines, rhscrd = 0;
    int_t tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int_t sym;
    int_t matrixlen[3];

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Enter dreadhb_dist1()");
#endif    

    int_t ntimestep,index;
    index=1007;
    ntimestep=0;

    int slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    char *filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif
    
    *nonz=0;
    *nrow=0;
    *ncol=0;
    ntimestep=0;
    if(!access(filename,0)){
        sleep(5);
        load_idata_vector_txt(matrixlen,3,0,1004);
        *nonz=matrixlen[0];
        *nrow=matrixlen[1];
        *ncol=matrixlen[1];
        ntimestep=matrixlen[2];
    }

    if(*nrow>0 && *nonz>0 && ntimestep>0){
        /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
        dallocateA_dist(*ncol, *nonz, nzval, rowind, colptr);        
        load_idata_vector_txt(*colptr,*ncol+1,ntimestep,1001);
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read colptr[%d] = %d\n", *ncol, (*colptr)[*ncol]);
        #endif        
        load_idata_markders_vector_txt(*rowind,*nonz,ntimestep,1002);
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read rowind[%d] = %d\n", *nonz-1, (*rowind)[*nonz-1]);
        #endif
        load_ddata_markders_vector_txt(*nzval,*nonz,ntimestep,1000);

        #ifdef test
        srand((unsigned int)(time(NULL)));
        for (int_t i = 0; i < (*nonz); i++)
        {
            (*nzval)[i] += ((*nzval)[i] * (rand() / (double)(RAND_MAX) - 0.5));
        }
        
        #endif

        #if ( DEBUGlevel>=1 )
            if ( !iam ) printf("read nzval[%d] = %e\n", *nonz-1, (*nzval)[*nonz-1]);
        #endif

        #if defined (FORTRAN_TXT)
        for(i=0;i<*ncol+1;i++)
        {
            (*colptr)[i]--;
        }
        for(i=0;i<*nonz;i++)
        {
            (*rowind)[i]--;
        }

        #endif
        
    }

    

#if 0
    /* Line 1 */
    fgets(buf, 100, fp);

    /* Line 2 */
    for (i=0; i<5; i++) {
	fscanf(fp, "%14c", buf); buf[14] = 0;
	tmp = atoi(buf); /*sscanf(buf, "%d", &tmp);*/
	if (i == 3) numer_lines = tmp;
	if (i == 4 && tmp) rhscrd = tmp;
    }
    DumpLine(fp);

    /* Line 3 */
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#if ( DEBUGlevel>=1 )
    if ( !iam ) printf("Matrix type %s\n", type);
#endif
    
    fscanf(fp, "%14c", buf); *nrow = atoi(buf); 
    fscanf(fp, "%14c", buf); *ncol = atoi(buf); 
    fscanf(fp, "%14c", buf); *nonz = atoi(buf); 
    fscanf(fp, "%14c", buf); tmp = atoi(buf);   
    
    if (tmp != 0)
	if ( !iam ) printf("This is not an assembled matrix!\n");
    if (*nrow != *ncol)
	if ( !iam ) printf("Matrix is not square.\n");
    DumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    dallocateA_dist(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    fscanf(fp, "%16c", buf);
    ParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    ParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    ParseFloatFormat(buf, &valnum, &valsize);
    fscanf(fp, "%20c", buf);
    DumpLine(fp);

    /* Line 5: right-hand side */    
    if ( rhscrd ) DumpLine(fp); /* skip RHSFMT */

#if ( DEBUGlevel>=1 )
    if ( !iam ) {
	printf("%d rows, %d nonzeros\n", *nrow, *nonz);
	printf("colnum %d, colsize %d\n", colnum, colsize);
	printf("rownum %d, rowsize %d\n", rownum, rowsize);
	printf("valnum %d, valsize %d\n", valnum, valsize);
    }
#endif
    
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
#if ( DEBUGlevel>=1 )
    if ( !iam )	printf("read colptr[%d] = %d\n", *ncol, (*colptr)[*ncol]);
#endif
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
#if ( DEBUGlevel>=1 )
    if ( !iam )	printf("read rowind[%d] = %d\n", *nonz-1, (*rowind)[*nonz-1]);
#endif
    if ( numer_lines ) {
        dReadValues(fp, *nonz, *nzval, valnum, valsize);
#if ( DEBUGlevel>=1 )
	if ( !iam ) printf("read nzval[%d] = %e\n", *nonz-1, (*nzval)[*nonz-1]);
#endif
    }    

    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
	FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }
#endif

    fclose(fp);
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Exit dreadhb_dist()");
#endif
}

#else
void
dreadhb_dist(int iam, FILE *fp, int_t *nrow, int_t *ncol, int_t *nonz,
	     double **nzval, int_t **rowind, int_t **colptr)
{

    register int_t i, numer_lines, rhscrd = 0;
    int_t tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int_t sym;

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Enter dreadhb_dist()");
#endif

    /* Line 1 */
    fgets(buf, 100, fp);

    /* Line 2 */
    for (i=0; i<5; i++) {
	fscanf(fp, "%14c", buf); buf[14] = 0;
	tmp = atoi(buf); /*sscanf(buf, "%d", &tmp);*/
	if (i == 3) numer_lines = tmp;
	if (i == 4 && tmp) rhscrd = tmp;
    }
    DumpLine(fp);

    /* Line 3 */
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#if ( DEBUGlevel>=1 )
    if ( !iam ) printf("Matrix type %s\n", type);
#endif
    
    fscanf(fp, "%14c", buf); *nrow = atoi(buf); 
    fscanf(fp, "%14c", buf); *ncol = atoi(buf); 
    fscanf(fp, "%14c", buf); *nonz = atoi(buf); 
    fscanf(fp, "%14c", buf); tmp = atoi(buf);   
    
    if (tmp != 0)
	if ( !iam ) printf("This is not an assembled matrix!\n");
    if (*nrow != *ncol)
	if ( !iam ) printf("Matrix is not square.\n");
    DumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    dallocateA_dist(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    fscanf(fp, "%16c", buf);
    ParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    ParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    ParseFloatFormat(buf, &valnum, &valsize);
    fscanf(fp, "%20c", buf);
    DumpLine(fp);

    /* Line 5: right-hand side */    
    if ( rhscrd ) DumpLine(fp); /* skip RHSFMT */

#if ( DEBUGlevel>=1 )
    if ( !iam ) {
	printf("%d rows, %d nonzeros\n", *nrow, *nonz);
	printf("colnum %d, colsize %d\n", colnum, colsize);
	printf("rownum %d, rowsize %d\n", rownum, rowsize);
	printf("valnum %d, valsize %d\n", valnum, valsize);
    }
#endif
    
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
#if ( DEBUGlevel>=1 )
    if ( !iam )	printf("read colptr[%d] = %d\n", *ncol, (*colptr)[*ncol]);
#endif
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
#if ( DEBUGlevel>=1 )
    if ( !iam )	printf("read rowind[%d] = %d\n", *nonz-1, (*rowind)[*nonz-1]);
#endif
    if ( numer_lines ) {
        dReadValues(fp, *nonz, *nzval, valnum, valsize);
#if ( DEBUGlevel>=1 )
	if ( !iam ) printf("read nzval[%d] = %e\n", *nonz-1, (*nzval)[*nonz-1]);
#endif
    }

    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
	FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }
    fclose(fp);
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Exit dreadhb_dist()");
#endif
}
#endif

void
dreadhb_dist2(int iam, int_t *nrow, int_t *ncol, int_t *nonz,
	     double **nzval, int_t **rowind, int_t **colptr, double **b_global, int_t *ntimestep)
{

    register int_t i, numer_lines, rhscrd = 0;
    int_t tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int_t sym;
    int_t matrixlen[3];

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Enter dreadhb_dist()");
#endif

    int_t index;
    index=1007;
    *ntimestep=0;

    #ifdef Torch_0419_Case2
    index=1100;
    #else
    index=1007;
    #endif

    int slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+((*ntimestep)==0?1:(int)log(*ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    char *filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,*ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,*ntimestep);
    #endif
    
    *nonz=0;
    *nrow=0;
    *ncol=0;
    *ntimestep=0;

    while (access(filename,0))
    {
        sleep(5);
    }
    
    if(!access(filename,0)){
        sleep(5);
        load_idata_vector_txt(matrixlen,3,0,1004);
        *nonz=matrixlen[0];
        *nrow=matrixlen[1];
        *ncol=matrixlen[1];
        *ntimestep=matrixlen[2];
    }

    if(*nrow>0 && *nonz>0 && *ntimestep>0){
        /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
        dallocateA_dist(*ncol, *nonz, nzval, rowind, colptr);
        #ifdef FILE_TXT
        load_idata_vector_txt(*colptr,*ncol+1,*ntimestep,1001);
        #else
        #ifdef FILE_BINARY_MATLAB
        int_t *colptr_temp = intMalloc_dist(*nonz);
        load_idata_vector_binary(colptr_temp,*nonz,*ntimestep,1001);        
        (*colptr)[0] = 1;
        int_t j = 0;
        for (i = 1; i < *nonz; i++)
        {
            if (colptr_temp[i] != colptr_temp[i-1])
            {
                (*colptr)[++j] = i + 1;
                if (j == *ncol)
                {
                    break;
                }                
            }            
        }
        (*colptr)[++j] = *nonz + (*colptr)[0];
        SUPERLU_FREE(colptr_temp);
        #else
        load_idata_vector_binary(*colptr,*ncol+1,*ntimestep,1001);
        #endif
        #endif
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read colptr[%d] = %d\n", *ncol, (*colptr)[*ncol]);
        #endif
        #ifdef FILE_TXT
        load_idata_markders_vector_txt(*rowind,*nonz,*ntimestep,1002);
        #else
        load_idata_vector_binary(*rowind,*nonz,*ntimestep,1002);
        #endif
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read rowind[%d] = %d\n", *nonz-1, (*rowind)[*nonz-1]);
        #endif
        #ifdef FILE_TXT
        load_ddata_markders_vector_txt(*nzval,*nonz,*ntimestep,1000);
        #else        
        load_ddata_vector_binary(*nzval,*nonz,*ntimestep,1000);
        #endif
        #if ( DEBUGlevel>=1 )
            if ( !iam ) printf("read nzval[%d] = %e\n", *nonz-1, (*nzval)[*nonz-1]);
        #endif
        *b_global=(double *) doubleMalloc_dist(*ncol);
        #ifdef FILE_TXT
        load_ddata_vector_txt(*b_global,*ncol,*ntimestep,1003);
        #else
        load_ddata_vector_binary(*b_global,*ncol,*ntimestep,1003);
        #endif
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read b_global[%ld] = %e\n", (*ncol)-1, (*b_global)[(*ncol)-1]);
        #endif

        if ( !iam ){
            PrintInt10("ai", 100, *colptr);
            PrintInt10("aj", 100, *rowind);
            Printdouble5("nnz", 100, *nzval);
            Printdouble5("b", 100, *b_global);
                       

            for ( i = 0; i < *nonz; i++)
            {
                if (isnan((*nzval)[i]) || isinf((*nzval)[i] == 0))
                {
                    printf("nzval[%ld] is nan or inf.\n", i);
                }
                
            }
            for ( i = 0; i < *ncol; i++)
            {
                if (isnan((*b_global)[i]) || isinf((*b_global)[i] == 0))
                {
                    printf("b[%ld] is nan or inf.\n", i);
                }
                
            }
            
        }        

        #if defined (FORTRAN_TXT)
        for(i=0;i<*ncol+1;i++)
        {
            (*colptr)[i]--;
        }
        for(i=0;i<*nonz;i++)
        {
            (*rowind)[i]--;
        }

        #endif
        
    }
}

void
dreadhb_dist3(int iam, int_t *nrow, int_t *ncol, int_t *nonz, double **nzval, double **b_global, int_t *ntimestep)
{

    register int_t i, numer_lines, rhscrd = 0;
    int_t tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int_t sym;
    int_t matrixlen[3];

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Enter dreadhb_dist()");
#endif

    int_t index;
    index=1007;
    *ntimestep=0;

    int slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+((*ntimestep)==0?1:(int)log(*ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    char *filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,*ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,*ntimestep);
    #endif
    
    *nonz=0;
    *nrow=0;
    *ncol=0;
    *ntimestep=0;
    if(!access(filename,0)){
        sleep(5);
        load_idata_vector_txt(matrixlen,3,0,1004);
        *nonz=matrixlen[0];
        *nrow=matrixlen[1];
        *ncol=matrixlen[1];
        *ntimestep=matrixlen[2];
    }

    if(*nrow>0 && *nonz>0 && *ntimestep>0){
        /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
        *nzval=(double *) doubleMalloc_dist(*nonz);
        load_ddata_markders_vector_txt(*nzval,*nonz,*ntimestep,1000);
        #if ( DEBUGlevel>=1 )
            if ( !iam ) printf("read nzval[%d] = %e\n", *nonz-1, (*nzval)[*nonz-1]);
        #endif
        *b_global=(double *) doubleMalloc_dist(*ncol);
        load_ddata_vector_txt(*b_global,*ncol,*ntimestep,1003);
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read b_global[%ld] = %e\n", (*ncol)-1, (*b_global)[(*ncol)-1]);
        #endif
        
    }
}

#if 0
void
dreadhb_dist4(int iam, int_t *nrow, int_t *ncol, int_t *nonz, double **nzval, double **b_global, int_t *ntimestep)
{

    register int_t i, numer_lines, rhscrd = 0;
    int_t tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int_t sym;
    int_t matrixlen[3];

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Enter dreadhb_dist()");
#endif

    int_t ntimestep,index;
    index=1007;
    ntimestep=0;

    int slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    char *filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif
    
    *nonz=0;
    *nrow=0;
    *ncol=0;
    ntimestep=0;
    if(!access(filename,0)){
        sleep(5);
        load_idata_vector_txt(matrixlen,3,0,1004);
        *nonz=matrixlen[0];
        *nrow=matrixlen[1];
        *ncol=matrixlen[1];
        ntimestep=matrixlen[2];
    }

    if(*nrow>0 && *nonz>0 && ntimestep>0){
        /* Allocate storage for nzval */
        *nzval=(double *) doubleMalloc_dist(*nonz);
        load_ddata_markders_vector_txt(*nzval,*nonz,ntimestep,1000);
        #if ( DEBUGlevel>=1 )
            if ( !iam ) printf("read nzval[%d] = %e\n", *nonz-1, (*nzval)[*nonz-1]);
        #endif

        *b_global=(double *) doubleMalloc_dist(*ncol);
        load_ddata_vector_txt(*b_global,*ncol,ntimestep,1003);
        #if ( DEBUGlevel>=1 )
            if ( !iam )	printf("read b_global[%ld] = %e\n", (*ncol)-1, b_global[(*ncol)-1]);
        #endif
        
    }

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(0, "Exit dreadhb_dist()");
#endif
}

#endif

/* Eat up the rest of the current line */
static int DumpLine(FILE *fp)
{
    register int c;
    while ((c = fgetc(fp)) != '\n') ;
    return 0;
}

static int ParseIntFormat(char *buf, int_t *num, int_t *size)
{
    char *tmp;

    tmp = buf;
    while (*tmp++ != '(') ;
    *num = atoi(tmp); 
    while (*tmp != 'I' && *tmp != 'i') ++tmp;
    ++tmp;
    *size = atoi(tmp); 
    return 0;
}

static int ParseFloatFormat(char *buf, int_t *num, int_t *size)
{
    char *tmp, *period;
    
    tmp = buf;
    while (*tmp++ != '(') ;
    *num = atoi(tmp); 
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
	   && *tmp != 'F' && *tmp != 'f') {
       /* May find kP before nE/nD/nF, like (1P6F13.6). In this case the
           num picked up refers to P, which should be skipped. */
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;
           *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/
        } else {
           ++tmp;
        }
    }
    ++tmp;
    period = tmp;
    while (*period != '.' && *period != ')') ++period ;
    *period = '\0';
    *size = atoi(tmp); 

    return 0;
}

static void
ReadVector(FILE *fp, int_t n, int_t *where, int_t perline, int_t persize)
{
    register int_t i, j, item;
    char tmp, buf[100];
    
    i = 0;
    while (i < n) {
	fgets(buf, 100, fp);    /* read a line at a time */
	for (j=0; j<perline && i<n; j++) {
	    tmp = buf[(j+1)*persize];     /* save the char at that place */
	    buf[(j+1)*persize] = 0;       /* null terminate */
	    item = atoi(&buf[j*persize]); 
	    buf[(j+1)*persize] = tmp;     /* recover the char at that place */
	    where[i++] = item - 1;
	}
    }
}

void
dReadValues(FILE *fp, int_t n, double *destination, 
             int_t perline, int_t persize)
{
    register int_t i, j, k, s;
    char tmp, buf[100];
    
    i = 0;
    while (i < n) {
	fgets(buf, 100, fp);    /* read a line at a time */
	for (j=0; j<perline && i<n; j++) {
	    tmp = buf[(j+1)*persize];     /* save the char at that place */
	    buf[(j+1)*persize] = 0;       /* null terminate */
	    s = j*persize;
	    for (k = 0; k < persize; ++k) /* No D_ format in C */
		if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
	    destination[i++] = atof(&buf[s]);
	    buf[(j+1)*persize] = tmp;     /* recover the char at that place */
	}
    }
}

/*! \brief
 *
 * <pre>
 * On input, nonz/nzval/rowind/colptr represents lower part of a symmetric
 * matrix. On exit, it represents the full matrix with lower and upper parts.
 * </pre>
 */
extern void
FormFullA(int_t n, int_t *nonz, double **nzval, int_t **rowind, int_t **colptr)
{
    register int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    double *t_val, *al_val, *a_val;

    al_rowind = *rowind;
    al_colptr = *colptr;
    al_val = *nzval;

    if ( !(marker =(int_t *) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
	ABORT("SUPERLU_MALLOC fails for marker[]");
    if ( !(t_colptr = (int_t *) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
	ABORT("SUPERLU_MALLOC t_colptr[]");
    if ( !(t_rowind = (int_t *) SUPERLU_MALLOC( *nonz * sizeof(int_t)) ) )
	ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    if ( !(t_val = (double*) SUPERLU_MALLOC( *nonz * sizeof(double)) ) )
	ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i) marker[i] = 0;
    for (j = 0; j < n; ++j) {
	for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)
	    ++marker[al_rowind[i]];
    }
    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
	t_colptr[i+1] = t_colptr[i] + marker[i];
	marker[i] = t_colptr[i];
    }

    /* Transpose matrix A to T */
    for (j = 0; j < n; ++j)
	for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
	    col = al_rowind[i];
	    t_rowind[marker[col]] = j;
	    t_val[marker[col]] = al_val[i];
	    ++marker[col];
	}

    new_nnz = *nonz * 2 - n;
    if ( !(a_colptr = (int_t *) SUPERLU_MALLOC( (n+1) * sizeof(int_t)) ) )
	ABORT("SUPERLU_MALLOC a_colptr[]");
    if ( !(a_rowind = (int_t *) SUPERLU_MALLOC( new_nnz * sizeof(int_t)) ) )
	ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    if ( !(a_val = (double*) SUPERLU_MALLOC( new_nnz * sizeof(double)) ) )
	ABORT("SUPERLU_MALLOC fails for a_val[]");
    
    a_colptr[0] = 0;
    k = 0;
    for (j = 0; j < n; ++j) {
      for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
	if ( t_rowind[i] != j ) { /* not diagonal */
	  a_rowind[k] = t_rowind[i];
	  a_val[k] = t_val[i];
#ifdef DEBUG
	  if ( fabs(a_val[k]) < 4.047e-300 )
	      printf("%5d: %e\n", k, a_val[k]);
#endif
	  ++k;
	}
      }

      for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
	a_rowind[k] = al_rowind[i];
	a_val[k] = al_val[i];
#ifdef DEBUG
	if ( fabs(a_val[k]) < 4.047e-300 )
	    printf("%5d: %e\n", k, a_val[k]);
#endif
	++k;
      }
      
      a_colptr[j+1] = k;
    }

    printf("FormFullA: new_nnz = %d, k = %d\n", new_nnz, k);

    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}

void load_ddata_markders_vector_txt(double *data,int_t n,int_t ntimestep,int_t index)
{
    int_t minstep=500000;
    int_t i,j,maxstep;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    #pragma omp parallel for private(i,maxstep,slen,filename,buf,fp)
    for(j=1;j<=n/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=n){
            maxstep=n;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif        

        if(access(filename,0)){
            printf("%s do not exist.\n",filename);
        }
        else{
            fp=fopen(filename,"r");
            
            for(i=(j-1)*minstep+1;i<=maxstep;i++)
            {
                fscanf(fp,"%s",buf);
                data[i-1]=atof(buf);
            }

            fclose(fp);
        }
        
    }

}

void load_ddata_markders_vector_binary(double *data,int_t n,int_t ntimestep,int_t index)
{
    int_t minstep=500000;
    int_t i,j,maxstep;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    #pragma omp parallel for private(i,maxstep,slen,filename,buf,fp)
    for(j=1;j<=n/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=n){
            maxstep=n;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif        

        if(access(filename,0)){
            printf("%s do not exist.\n",filename);
        }
        else{
            fp=fopen(filename,"r");

            fread(&data[(j-1)*minstep], sizeof(double), (maxstep - (j-1)*minstep), fp);

            fclose(fp);
        }
        
    }

}

void load_idata_markders_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index)
{
    int_t minstep=500000;
    int_t i,j,maxstep;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    #pragma omp parallel for private(i,maxstep,slen,filename,buf,fp)
    for(j=1;j<=n/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=n){
            maxstep=n;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif        

        if(access(filename,0)){
            printf("%s do not exist.\n",filename);
        }
        else{
            fp=fopen(filename,"r");
            
            for(i=(j-1)*minstep+1;i<=maxstep;i++)
            {
                fscanf(fp,"%s",buf);                
                data[i-1]=atoi(buf);
                
            }

            fclose(fp);
        }
        
    }

}

void load_idata_markders_vector_binary(int_t *data,int_t n,int_t ntimestep,int_t index)
{
    int_t minstep=500000;
    int_t i,j,maxstep;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    #pragma omp parallel for private(i,maxstep,slen,filename,buf,fp)
    for(j=1;j<=n/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=n){
            maxstep=n;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif        

        if(access(filename,0)){
            printf("%s do not exist.\n",filename);
        }
        else{
            fp=fopen(filename,"r");
            
            fread(&data[(j-1)*minstep], sizeof(int_t), (maxstep - (j-1)*minstep), fp);

            fclose(fp);
        }
        
    }

}

void load_idata_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
    }
    else{
        fp=fopen(filename,"r");

        for(i=0;i<n;i++)
        {
            fscanf(fp,"%s",buf);            
            data[i]=atoi(buf);            
        }

        fclose(fp);
    }

}

void delete_idata_vector_txt(int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
    }
    else{
        remove(filename);
    }

}

int isexist_idata_vector_txt(int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
        return 0;
    }
    else{
        return 1;
    }

}

void load_idata_vector_binary(int_t *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
    }
    else{
        fp=fopen(filename,"r");

        fread(data, sizeof(int_t), n, fp);

        fclose(fp);
    }

}

void load_idata_vector_binary_c(int_t *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
    }
    else{
        fp=fopen(filename,"r");

        fread(data, sizeof(int_t), n, fp);

        fclose(fp);
    }

}

void load_ddata_vector_txt(double *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
    }
    else{
        fp=fopen(filename,"r");

        for(i=0;i<n;i++)
        {
            fscanf(fp,"%s",buf);
            data[i]=atof(buf);
        }

        fclose(fp);
    }

}

void load_ddata_vector_binary(double *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char buf[100];
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("%s do not exist.\n",filename);
    }
    else{
        fp=fopen(filename,"r");
        fread(data, sizeof(double), n, fp);
        fclose(fp);
        
    }

}

void save_ddata_vector_txt(double *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    fp=fopen(filename,"w");

    for(i=0;i<n;i++)
    {
        fprintf(fp,"%.25e\n",data[i]);
    }

    fclose(fp);

}

void save_idata_vector_txt(int_t *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    fp=fopen(filename,"w");

    for(i=0;i<n;i++)
    {
        fprintf(fp,"%ld\n",data[i]);
    }

    fclose(fp);

}

void save_ddata_vector_binary(double *data,int_t n,int_t ntimestep,int_t index)
{
    int_t i;
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    fp=fopen(filename,"w");

    fwrite(data, sizeof(double), n, fp);

    fclose(fp);

}

void save_idata_vector_binary(int_t *data,int_t n,int_t ntimestep,int index)
{
    int_t i;
    char *filename;
    int slen;
    FILE *fp;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    fp=fopen(filename,"w");

    fwrite(data, sizeof(int_t), n, fp);

    fclose(fp);

}

#ifdef Torch

#ifdef Use_harddisk
// according to save_iam to save
void save_LUstruct_harddisk(int_t n, gridinfo3d_t *grid3d, dLUstruct_t *LUstruct)
{
    
    gridinfo_t *grid=&(grid3d->grid2d);
    int_t nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t ntimestep=0;

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    
    nsupers = Glu_persist->supno[n-1] + 1;

    nb = CEILING(nsupers, grid->npcol);
      
    int_t index=INDEX_Lnzval_bc_ptr+Llu->save_iam;  

    fpos_t *pos;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif

    fp=fopen(filename,"wb");
    Llu->Lnzval_bc_ptr_sumlen = 0;
    for(i=0;i<nb;i++)
    {            
        if ( Llu->Lrowind_bc_ptr[i] && Llu->Lnzval_bc_ptr_ilen[i]) {
            
            pos=(fpos_t*)malloc(sizeof(fpos_t)); 
            fgetpos(fp, pos);
            Llu->Lnzval_bc_ptr_fileposition[i]=pos;
            fwrite(Llu->Lnzval_bc_ptr[i], sizeof(double), Llu->Lnzval_bc_ptr_ilen[i], fp);
            Llu->Lnzval_bc_ptr_sumlen += Llu->Lnzval_bc_ptr_ilen[i];
        }
    }

    fclose(fp);

    nb = CEILING(nsupers, grid->nprow);
    
    index=INDEX_Unzval_br_ptr+Llu->save_iam;    

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif

    fp=fopen(filename,"wb");
    Llu->Unzval_br_ptr_sumlen = 0;
    for(i=0;i<nb;i++)
    {
        if ( Llu->Ufstnz_br_ptr[i] && Llu->Unzval_br_ptr_ilen[i] ) {
            pos=(fpos_t*)malloc(sizeof(fpos_t));
            fgetpos(fp, pos);
            Llu->Unzval_br_ptr_fileposition[i]=pos;            
            fwrite(Llu->Unzval_br_ptr[i], sizeof(double), Llu->Unzval_br_ptr_ilen[i], fp);
            Llu->Unzval_br_ptr_sumlen += Llu->Unzval_br_ptr_ilen[i];
        }
    }

    fclose(fp);    

}

void save_LUstruct_harddisk2(int_t n, gridinfo3d_t *grid3d, dLUstruct_t *LUstruct, int_t nsupers)
{
    
    gridinfo_t *grid=&(grid3d->grid2d);
    int_t nb;
    // Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t ntimestep=0;

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    
    // nsupers = Glu_persist->supno[n-1] + 1;
    nb = CEILING(nsupers, grid->npcol); 
    int_t index=INDEX_Lnzval_bc_ptr+Llu->save_iam;
    fpos_t *pos;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif   

    fp=fopen(filename,"wb");
    // Llu->Lnzval_bc_ptr_sumlen = 0;
    for(i=0;i<nb;i++)
    {            
        if ( Llu->Lrowind_bc_ptr[i] && Llu->Lnzval_bc_ptr_ilen[i]) {
            
            // pos=(fpos_t*)malloc(sizeof(fpos_t)); 
            // fgetpos(fp, pos);
            // Llu->Lnzval_bc_ptr_fileposition[i]=pos;
            fwrite(Llu->Lnzval_bc_ptr[i], sizeof(double), Llu->Lnzval_bc_ptr_ilen[i], fp);
            // Llu->Lnzval_bc_ptr_sumlen += Llu->Lnzval_bc_ptr_ilen[i];
        }
    }

    fclose(fp);

    nb = CEILING(nsupers, grid->nprow);
    
    index=INDEX_Unzval_br_ptr+Llu->save_iam;    

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif

    fp=fopen(filename,"wb");
    // Llu->Unzval_br_ptr_sumlen = 0;
    
    for(i=0;i<nb;i++)
    {        
        if ( Llu->Ufstnz_br_ptr[i] && Llu->Unzval_br_ptr_ilen[i] ) {
            // pos=(fpos_t*)malloc(sizeof(fpos_t));
            // fgetpos(fp, pos);
            // Llu->Unzval_br_ptr_fileposition[i]=pos;            
            fwrite(Llu->Unzval_br_ptr[i], sizeof(double), Llu->Unzval_br_ptr_ilen[i], fp);
            // Llu->Unzval_br_ptr_sumlen += Llu->Unzval_br_ptr_ilen[i];
        }
    }

    fclose(fp);    

}

// Load LUstruct from bin
int load_LUstruct_harddisk(int_t n, gridinfo3d_t *grid3d, dLUstruct_t *LUstruct)
{
    
    gridinfo_t *grid=&(grid3d->grid2d);
    int_t nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t ntimestep=0;
    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep,k;
    char *filename;
    int slen;
    FILE *fp;
    int_t errorflag = 0;
    nsupers = Glu_persist->supno[n-1] + 1;
    nb = CEILING(nsupers, grid->npcol);
      
    int_t index=INDEX_Lnzval_bc_ptr+Llu->save_iam; 
    double *iLnzval_bc_ptr;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif
    
    if(access(filename,0)){
        printf("load_Lnzval_bc_ptr error: %s do not exist.\n",filename);
        errorflag = -1;
    }
    else{
        fp=fopen(filename,"rb");

        double *temp_Lnzval_bc_ptr = doubleCalloc_dist(Llu->Lnzval_bc_ptr_sumlen);
        fread(temp_Lnzval_bc_ptr, sizeof(double), Llu->Lnzval_bc_ptr_sumlen, fp);
        double *pos = temp_Lnzval_bc_ptr;
       
        for(i=0;i<nb;i++)
        {
            if ( Llu->Lrowind_bc_ptr[i] && Llu->Lnzval_bc_ptr_ilen[i] ){
                // iLnzval_bc_ptr = load_Lnzval_bc_ptr_harddisk(i, Llu, Llu->save_iam);
                // Llu->Lnzval_bc_ptr[i] = iLnzval_bc_ptr;
                iLnzval_bc_ptr = doubleCalloc_dist(Llu->Lnzval_bc_ptr_ilen[i]);
                memcpy(iLnzval_bc_ptr, pos, Llu->Lnzval_bc_ptr_ilen[i] * sizeof(double));
                Llu->Lnzval_bc_ptr[i] = iLnzval_bc_ptr;
                pos += Llu->Lnzval_bc_ptr_ilen[i]; 
            }
        } 
        SUPERLU_FREE(temp_Lnzval_bc_ptr);
        fclose(fp);
        
    }  

    index=INDEX_Unzval_br_ptr+Llu->save_iam;

    nb = CEILING(nsupers, grid->nprow);
    double *iUnzval_br_ptr;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("load_Unzval_br_ptr error: %s do not exist.\n",filename);
        errorflag = -1;
    }
    else{
        fp=fopen(filename,"rb");

        // for(i=0;i<nb;i++)
        // {
        //     if ( Llu->Ufstnz_br_ptr[i] && Llu->Unzval_br_ptr_ilen[i] ) {
        //         iUnzval_br_ptr = load_Unzval_br_ptr_harddisk(i, Llu, Llu->save_iam);
        //         Llu->Unzval_br_ptr[i] = iUnzval_br_ptr;
        //     }
        // }    

        double *temp_Unzval_br_ptr = doubleCalloc_dist(Llu->Unzval_br_ptr_sumlen);
        fread(temp_Unzval_br_ptr, sizeof(double), Llu->Unzval_br_ptr_sumlen, fp);
        double *pos = temp_Unzval_br_ptr;
        for(i=0;i<nb;i++)
        {
            if ( Llu->Ufstnz_br_ptr[i] && Llu->Unzval_br_ptr_ilen[i] ){
                iUnzval_br_ptr = doubleCalloc_dist(Llu->Unzval_br_ptr_ilen[i]);
                memcpy(iUnzval_br_ptr, pos, Llu->Unzval_br_ptr_ilen[i] * sizeof(double));
                Llu->Unzval_br_ptr[i] = iUnzval_br_ptr;
                pos += Llu->Unzval_br_ptr_ilen[i];
            }
        } 
        SUPERLU_FREE(temp_Unzval_br_ptr);

        fclose(fp);        
    }

    if(errorflag == -1){
        return errorflag;
    }

}

double* load_Lnzval_bc_ptr_harddisk(int_t ljb, dLocalLU_t *Llu, int iam)
{
    // int iam;
    // MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    double *lsub;
    if(Llu->Lnzval_bc_ptr_ilen[ljb]){
        lsub=DOUBLE_ALLOC(Llu->Lnzval_bc_ptr_ilen[ljb]);
    }
    else{
        lsub=NULL;
        return lsub;
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Lnzval_bc_ptr+iam;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif      

    if(access(filename,0)){
        printf("load_Lnzval_bc_ptr[%ld]: %s do not exist.\n",ljb,filename);
        lsub=NULL;
        return lsub;
    }
    else{
        fp=fopen(filename,"rb");

        fpos_t *pos=Llu->Lnzval_bc_ptr_fileposition[ljb];
        fsetpos(fp,pos);
        fread(lsub, sizeof(double), Llu->Lnzval_bc_ptr_ilen[ljb], fp);

        fclose(fp);
        return lsub;
    }

}

double* load_Unzval_br_ptr_harddisk(int_t lb, dLocalLU_t *Llu, int iam)
{
    // int iam;
    // MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    double *uval;
    if(Llu->Unzval_br_ptr_ilen[lb]){
        uval=DOUBLE_ALLOC(Llu->Unzval_br_ptr_ilen[lb]);
    }
    else{
        uval=NULL;
        return uval;
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Unzval_br_ptr+iam;

    j=lb/minstep+1;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif        

    if(access(filename,0)){
        printf("load_Unzval_br_ptr[%ld] %s do not exist.\n",lb,filename);
        uval=NULL;
        return uval;
    }
    else{
        fp=fopen(filename,"rb");

        fpos_t *pos=Llu->Unzval_br_ptr_fileposition[lb];
        fsetpos(fp,pos);
        fread(uval, sizeof(double), Llu->Unzval_br_ptr_ilen[lb], fp);

        fclose(fp);
        return uval;
    }

}

int set_iLnzval_bc_ptr_harddisk(double *lsub, int_t ljb, int_t begin, int_t len, dLocalLU_t *Llu, int iam)
{
    if(!lsub && !Llu->Lnzval_bc_ptr_ilen[ljb]){
        return 0;
    }

    // int iam;
    // MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    if(begin+len>Llu->Lnzval_bc_ptr_ilen[ljb]){
        ABORT("exceed the length of Lnzval_bc_ptr[]\n");
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Lnzval_bc_ptr+iam;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif

    if(access(filename,0)){
        printf("set_iLnzval_bc_ptr_txt %s do not exist.\n",filename);
        return 0;
    }
    else{
        fp=fopen(filename,"rb+");

        fpos_t *pos=Llu->Lnzval_bc_ptr_fileposition[ljb]; 
        fsetpos(fp,pos);
        fseek(fp,begin*sizeof(double),SEEK_CUR);
        fwrite(lsub, sizeof(double), len, fp);
        fclose(fp);

        return 1;
    }

}

int set_iUnzval_br_ptr_harddisk(double *uval, int_t lb, int_t begin, int_t len, dLocalLU_t *Llu, int iam)
{
    if(!uval && !Llu->Unzval_br_ptr_ilen[lb]){
        return 0;
    }

    // int iam;
    // MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    if(begin+len>Llu->Unzval_br_ptr_ilen[lb]){
        ABORT("exceed the length of Unzval_br_ptr[]\n");
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int index=INDEX_Unzval_br_ptr+iam;    

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".bin\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.bin\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.bin\0",index,ntimestep);
    #endif     

    if(access(filename,0)){
        printf("set_iUnzval_br_ptr_txt %s do not exist.\n",filename);
        return 0;
    }
    else{
        fp=fopen(filename,"rb+");

        fpos_t *pos=Llu->Unzval_br_ptr_fileposition[lb]; 
        fsetpos(fp,pos);
        fseek(fp,begin*sizeof(double),SEEK_CUR);
        fwrite(uval, sizeof(double), len, fp);
        fclose(fp);
        return 1;
    }

}

#endif
#ifdef SuperLargeScale
// Save LUstruct to txt
void save_LUstruct_txt(int_t n, gridinfo3d_t *grid3d, dLUstruct_t *LUstruct)
{
    
    gridinfo_t *grid=&(grid3d->grid2d);
    int_t nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t ntimestep=0;

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    
    nsupers = Glu_persist->supno[n-1] + 1;

    nb = CEILING(nsupers, grid->npcol);
      
    int_t index=INDEX_Lnzval_bc_ptr+grid3d->iam;  

    fpos_t *pos;
    // #pragma omp parallel for private(i,maxstep,slen,filename,fp,pos) schedule(dynamic,4)
    for(j=1;j<=nb/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=nb){
            maxstep=nb;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif

        fp=fopen(filename,"w");
            
        for(i=(j-1)*minstep;i<maxstep;i++)
        {            
            if ( Llu->Lrowind_bc_ptr[i] && Llu->Lnzval_bc_ptr_ilen[i]) {
                pos=(fpos_t*)malloc(sizeof(fpos_t));
                fgetpos(fp, pos);
                Llu->Lnzval_bc_ptr_fileposition[i]=pos;
                fwrite(Llu->Lnzval_bc_ptr[i], sizeof(double), Llu->Lnzval_bc_ptr_ilen[i], fp);
            }
        }

        fclose(fp);
        
    }

    index=INDEX_Unzval_br_ptr+grid3d->iam;

    nb = CEILING(nsupers, grid->nprow);
    // #pragma omp parallel for private(i,maxstep,slen,filename,fp,pos) schedule(dynamic,4)
    for(j=1;j<=nb/minstep+1;j++)
    {        
        maxstep=j*minstep;
        if(maxstep>=nb){
            maxstep=nb;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif        

        fp=fopen(filename,"w");
            
        for(i=(j-1)*minstep;i<maxstep;i++)
        {
            if ( Llu->Ufstnz_br_ptr[i] && Llu->Unzval_br_ptr_ilen[i] ) {
                pos=(fpos_t*)malloc(sizeof(fpos_t));
                fgetpos(fp, pos);
                Llu->Unzval_br_ptr_fileposition[i]=pos;
                fwrite(Llu->Unzval_br_ptr[i], sizeof(double), Llu->Unzval_br_ptr_ilen[i], fp);
            }
        }

        fclose(fp);
        
    }

}

#ifdef SuperLargeScaleGPU
// Save RecordMatrix to hard disk
void save_RecordMatrix_txt(gridinfo3d_t *grid3d, dLUstruct_t *LUstruct)
{
    int_t i;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;
    int_t index=INDEX_Lnzval_RecordMatrix+grid3d->iam;  
    dLocalLU_t *Llu = LUstruct->Llu;   

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif    

    fp=fopen(filename,"w");
    for (int_t i = 0; i < 2; i++)
    {
        int_t *Lnzval_RecordVector = intCalloc_dist(LNZVAL_RECORDMATRIX(i,0,0,0));
        Lnzval_RecordVector[0] = LNZVAL_RECORDMATRIX(i,0,0,0);
        if(Lnzval_RecordVector[0] == 1){
            fwrite(Lnzval_RecordVector, sizeof(int_t), Lnzval_RecordVector[0], fp);
            // PrintInt10("LRecordVector", Lnzval_RecordVector[0], Lnzval_RecordVector);
            continue;
        }
        Lnzval_RecordVector[1] = LNZVAL_RECORDMATRIX(i,0,0,1);
        int_t maxTopoLevel = Lnzval_RecordVector[1];
        int_t j = 2;

        for (int_t topoLvl = 1; topoLvl < maxTopoLevel + 1 && j < Lnzval_RecordVector[0]; ++topoLvl)
        {
            Lnzval_RecordVector[j] = LNZVAL_RECORDMATRIX(i,topoLvl,0,0);
            int_t sumtopolev_lrecord = Lnzval_RecordVector[j++];
            for (int_t l = 1; l < sumtopolev_lrecord + 1; l++)
            {
                Lnzval_RecordVector[j++] = LNZVAL_RECORDMATRIX(i,topoLvl,0,l);
            }
            
            for (int_t l = 1; l < sumtopolev_lrecord + 1; l++)
            {                
                int_t k = LNZVAL_RECORDMATRIX(i,topoLvl,0,l);
                Lnzval_RecordVector[j] = LNZVAL_RECORDMATRIX(i,topoLvl,k+1,0);
                int_t sumUsed_Lnzval_bc_ptr_Record = Lnzval_RecordVector[j++];
                for (int_t m = 1; m < sumUsed_Lnzval_bc_ptr_Record + 1; m++)
                {
                    Lnzval_RecordVector[j++] = LNZVAL_RECORDMATRIX(i,topoLvl,k+1,m);
                }
                
            }            
            
        }
        fwrite(Lnzval_RecordVector, sizeof(int_t), Lnzval_RecordVector[0], fp);
        // PrintInt10("LRecordVector", Lnzval_RecordVector[0], Lnzval_RecordVector);
        SUPERLU_FREE(Lnzval_RecordVector);
    }

    fclose(fp);

    index=INDEX_Unzval_RecordMatrix+grid3d->iam;     

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif    

    fp=fopen(filename,"w");
    for (int_t i = 0; i < 2; i++)
    {
        int_t *Unzval_RecordVector = intCalloc_dist(UNZVAL_RECORDMATRIX(i,0,0,0));
        Unzval_RecordVector[0] = UNZVAL_RECORDMATRIX(i,0,0,0);
        if(Unzval_RecordVector[0] == 1){
            fwrite(Unzval_RecordVector, sizeof(int_t), Unzval_RecordVector[0], fp);
            // PrintInt10("URecordVector", Unzval_RecordVector[0], Unzval_RecordVector);
            continue;
        }
        Unzval_RecordVector[1] = UNZVAL_RECORDMATRIX(i,0,0,1);
        int_t maxTopoLevel = Unzval_RecordVector[1];
        int_t j = 2;
        
        for (int_t topoLvl = 1; topoLvl < maxTopoLevel + 1 && j < Unzval_RecordVector[0]; ++topoLvl)
        {
            Unzval_RecordVector[j] = UNZVAL_RECORDMATRIX(i,topoLvl,0,0);
            int_t sumtopolev_urecord = Unzval_RecordVector[j++];
            for (int_t l = 1; l < sumtopolev_urecord + 1; l++)
            {
                Unzval_RecordVector[j++] = UNZVAL_RECORDMATRIX(i,topoLvl,0,l);
            }
            
            for (int_t l = 1; l < sumtopolev_urecord + 1; l++)
            {                
                int_t k = UNZVAL_RECORDMATRIX(i,topoLvl,0,l);
                Unzval_RecordVector[j] = UNZVAL_RECORDMATRIX(i,topoLvl,k+1,0);
                int_t sumUsed_Unzval_br_ptr_Record = Unzval_RecordVector[j++];
                for (int_t m = 1; m < sumUsed_Unzval_br_ptr_Record + 1; m++)
                {
                    Unzval_RecordVector[j++] = UNZVAL_RECORDMATRIX(i,topoLvl,k+1,m);
                }
                
            }
        }

        fwrite(Unzval_RecordVector, sizeof(int_t), Unzval_RecordVector[0], fp);
        // PrintInt10("URecordVector", Unzval_RecordVector[0], Unzval_RecordVector);
        SUPERLU_FREE(Unzval_RecordVector);
        
    }

    fclose(fp);    

}

// Load RecordMatrix to hard disk
void load_RecordMatrix_txt(gridinfo3d_t *grid3d, dLUstruct_t *LUstruct, int_t *nexttopoLvl_Lnzval, int_t *nexttopoLvl_Unzval, int_t *nextk0_Lnzval, int_t *nextk0_Unzval, int_t *LnzvalPtr_host, int_t *UnzvalPtr_host, int_t *isGPUUsed_Lnzval_bc_ptr_host, int_t *isGPUUsed_Unzval_br_ptr_host, int_t *Lnzval_bc_ptr_len, int_t *Unzval_br_ptr_len, int_t **UsedOrder_Lnzval, int_t **UsedOrder_Unzval)
{
    int_t i;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;
    int_t index=INDEX_Lnzval_RecordMatrix+grid3d->iam;  
    dLocalLU_t *Llu = LUstruct->Llu;   

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    fp=fopen(filename,"r");
    
    for (int_t i = 0; i < 2; i++)
    {
        int_t len_RecordMatrix;
        fread(&len_RecordMatrix, sizeof(int_t), 1, fp);
        int_t *Lnzval_RecordVector = intMalloc_dist(len_RecordMatrix);
        Lnzval_RecordVector[0] = len_RecordMatrix;
        if(len_RecordMatrix == 1){
            Llu->Lnzval_RecordMatrix[i] = (int_t***)SUPERLU_MALLOC(sizeof(int_t**));
            int_t **LRecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
            int_t *iLnzval_RecordMatrix = intMalloc_dist(1);
            iLnzval_RecordMatrix[0] = len_RecordMatrix;
            
            LRecordMatrix[0] = iLnzval_RecordMatrix;
            Llu->Lnzval_RecordMatrix[i][0] = LRecordMatrix;
            continue;
        }

        fread(&Lnzval_RecordVector[1], sizeof(int_t), Lnzval_RecordVector[0] - 1, fp);

        // PrintInt10("LRecordVector", Lnzval_RecordVector[0], Lnzval_RecordVector);
        
        int_t maxTopoLevel = Lnzval_RecordVector[1];
        Llu->Lnzval_RecordMatrix[i] = (int_t***)SUPERLU_MALLOC((maxTopoLevel + 1) * sizeof(int_t**));
        
        int_t j = 2;
        int_t numOrder = 0;
        for (int_t topoLvl = 0; topoLvl < maxTopoLevel + 1 && j < len_RecordMatrix; ++topoLvl)
        {
            int_t **LRecordMatrix;
            if (topoLvl == 0)
            {
                int_t *iLnzval_RecordMatrix = intMalloc_dist(2);
                iLnzval_RecordMatrix[0] = len_RecordMatrix;
                iLnzval_RecordMatrix[1] = Lnzval_RecordVector[1];
                LRecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
                LRecordMatrix[0] = iLnzval_RecordMatrix;
            }
            else
            {
                
                int_t sumtopolev_lrecord = Lnzval_RecordVector[j++];                

                int_t *iLnzval_RecordMatrix = intMalloc_dist(sumtopolev_lrecord + 1);
                iLnzval_RecordMatrix[0] = sumtopolev_lrecord;                
                
                for (int_t l = 1; l < sumtopolev_lrecord + 1; l++)
                {
                    iLnzval_RecordMatrix[l] = Lnzval_RecordVector[j++];
                }

                LRecordMatrix = (int_t**)SUPERLU_MALLOC((iLnzval_RecordMatrix[sumtopolev_lrecord] + 2) * sizeof(int_t*));
                LRecordMatrix[0] = iLnzval_RecordMatrix;

                for (int_t l = 1; l < sumtopolev_lrecord + 1; l++)
                {                
                    int_t k = iLnzval_RecordMatrix[l];
                    int_t sumUsed_Lnzval_bc_ptr_Record = Lnzval_RecordVector[j++];
                    int_t *mLnzval_RecordMatrix = intMalloc_dist(sumUsed_Lnzval_bc_ptr_Record + 1);
                    mLnzval_RecordMatrix[0] = sumUsed_Lnzval_bc_ptr_Record;
                    int_t l_val_len = 0;
                    for (int_t m = 1; m < sumUsed_Lnzval_bc_ptr_Record + 1; m++)
                    {
                        mLnzval_RecordMatrix[m] = Lnzval_RecordVector[j++];
                        
                        if (l==1 && sumtopolev_lrecord && *nexttopoLvl_Lnzval == -1 && *nextk0_Lnzval == -1)
                        {
                            isGPUUsed_Lnzval_bc_ptr_host[mLnzval_RecordMatrix[m]] = GPUUsed;
                            LnzvalPtr_host[mLnzval_RecordMatrix[m]] = l_val_len;
                            l_val_len += Llu->Lnzval_bc_ptr_ilen[mLnzval_RecordMatrix[m]];
                        }
                        
                    }
                    
                    *(LRecordMatrix + k + 1) = mLnzval_RecordMatrix;

                    numOrder++;

                    if (l==1 && sumtopolev_lrecord && *nexttopoLvl_Lnzval == -1 && *nextk0_Lnzval == -1){
                        *nexttopoLvl_Lnzval = topoLvl - 1;
                        *nextk0_Lnzval = k;
                        *Lnzval_bc_ptr_len = l_val_len;
                    }                    
                    
                }

            }

            *(Llu->Lnzval_RecordMatrix[i] + topoLvl) = LRecordMatrix;
            
        }
        
        if (numOrder)
        {
            UsedOrder_Lnzval[i] = (int_t*)SUPERLU_MALLOC((numOrder * 2 + 1) * sizeof(int_t));
            int_t idx = 0;
            UsedOrder_Lnzval[i][idx++] = numOrder * 2 + 1;
            for (int_t topoLvl = 1; topoLvl < maxTopoLevel + 1; ++topoLvl)
            {
                int_t **LRecordMatrix = Llu->Lnzval_RecordMatrix[i][topoLvl];
                int_t *iLnzval_RecordMatrix = LRecordMatrix[0];
                int_t sumtopolev_lrecord = iLnzval_RecordMatrix[0];

                if (sumtopolev_lrecord)
                {
                    for (int_t l = 1; l < sumtopolev_lrecord + 1; l++)
                    {
                        int_t k = iLnzval_RecordMatrix[l];
                        UsedOrder_Lnzval[i][idx++] = topoLvl - 1;
                        UsedOrder_Lnzval[i][idx++] = k;
                    }
                    
                }                       
                
            }

            // PrintInt10("UsedOrder_Lnzval", numOrder * 2 + 1, UsedOrder_Lnzval[i]);
        }
          
        SUPERLU_FREE(Lnzval_RecordVector);
    }
    
    fclose(fp);    

    index=INDEX_Unzval_RecordMatrix+grid3d->iam;     

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")-3;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld.txt\0",index,ntimestep);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d.txt\0",index,ntimestep);
    #endif

    fp=fopen(filename,"r");
    for (int_t i = 0; i < 2; i++)
    {
        int_t len_RecordMatrix;
        fread(&len_RecordMatrix, sizeof(int_t), 1, fp);
        int_t *Unzval_RecordVector = intMalloc_dist(len_RecordMatrix);
        Unzval_RecordVector[0] = len_RecordMatrix;
        if(len_RecordMatrix == 1){
            Llu->Unzval_RecordMatrix[i] = (int_t***)SUPERLU_MALLOC(sizeof(int_t**));
            int_t **URecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
            int_t *iUnzval_RecordMatrix = intMalloc_dist(1);
            iUnzval_RecordMatrix[0] = len_RecordMatrix;
            
            URecordMatrix[0] = iUnzval_RecordMatrix;
            Llu->Unzval_RecordMatrix[i][0] = URecordMatrix;
            continue;
        }
        fread(&Unzval_RecordVector[1], sizeof(int_t), Unzval_RecordVector[0] - 1, fp);

        // PrintInt10("URecordVector", Unzval_RecordVector[0], Unzval_RecordVector);
        
        int_t maxTopoLevel = Unzval_RecordVector[1];
        Llu->Unzval_RecordMatrix[i] = (int_t***)SUPERLU_MALLOC((maxTopoLevel + 1) * sizeof(int_t**));

        int_t j = 2;
        int_t numOrder = 0;
        for (int_t topoLvl = 0; topoLvl < maxTopoLevel + 1 && j < len_RecordMatrix; ++topoLvl)
        {
            int_t **URecordMatrix;
            if (topoLvl == 0)
            {
                int_t *iUnzval_RecordMatrix = intMalloc_dist(2);
                iUnzval_RecordMatrix[0] = len_RecordMatrix;
                iUnzval_RecordMatrix[1] = Unzval_RecordVector[1];
                URecordMatrix = (int_t**)SUPERLU_MALLOC(sizeof(int_t*));
                URecordMatrix[0] = iUnzval_RecordMatrix;
            }
            else
            {
                
                int_t sumtopolev_urecord = Unzval_RecordVector[j++];

                int_t *iUnzval_RecordMatrix = intMalloc_dist(sumtopolev_urecord + 1);
                iUnzval_RecordMatrix[0] = sumtopolev_urecord;
                
                for (int_t l = 1; l < sumtopolev_urecord + 1; l++)
                {
                    iUnzval_RecordMatrix[l] = Unzval_RecordVector[j++];
                }

                URecordMatrix = (int_t**)SUPERLU_MALLOC((iUnzval_RecordMatrix[sumtopolev_urecord] + 2) * sizeof(int_t*));
                URecordMatrix[0] = iUnzval_RecordMatrix;

                for (int_t l = 1; l < sumtopolev_urecord + 1; l++)
                {                
                    int_t k = iUnzval_RecordMatrix[l];
                    int_t sumUsed_Unzval_br_ptr_Record = Unzval_RecordVector[j++];
                    int_t *mUnzval_RecordMatrix = intMalloc_dist(sumUsed_Unzval_br_ptr_Record + 1);
                    mUnzval_RecordMatrix[0] = sumUsed_Unzval_br_ptr_Record;
                    int_t u_val_len = 0;
                    for (int_t m = 1; m < sumUsed_Unzval_br_ptr_Record + 1; m++)
                    {
                        mUnzval_RecordMatrix[m] = Unzval_RecordVector[j++];

                        if (l==1 && sumtopolev_urecord && *nexttopoLvl_Unzval == -1 && *nextk0_Unzval == -1)
                        {
                            isGPUUsed_Unzval_br_ptr_host[mUnzval_RecordMatrix[m]] = GPUUsed;
                            UnzvalPtr_host[mUnzval_RecordMatrix[m]] = u_val_len;
                            u_val_len += Llu->Unzval_br_ptr_ilen[mUnzval_RecordMatrix[m]];
                        }
                    }
                    *(URecordMatrix + k + 1) = mUnzval_RecordMatrix;

                    numOrder++;

                    if (l==1 && sumtopolev_urecord && *nexttopoLvl_Unzval == -1 && *nextk0_Unzval == -1){
                        *nexttopoLvl_Unzval = topoLvl - 1;
                        *nextk0_Unzval = k;
                        *Unzval_br_ptr_len = u_val_len;
                    }
                }

            }

            *(Llu->Unzval_RecordMatrix[i] + topoLvl) = URecordMatrix;
            
        }

        if (numOrder)
        {
            UsedOrder_Unzval[i] = (int_t*)SUPERLU_MALLOC((numOrder * 2 + 1) * sizeof(int_t));
            int_t idx = 0;
            UsedOrder_Unzval[i][idx++] = numOrder * 2 + 1;
            for (int_t topoLvl = 1; topoLvl < maxTopoLevel + 1; ++topoLvl)
            {
                int_t **URecordMatrix = Llu->Unzval_RecordMatrix[i][topoLvl];
                int_t *iUnzval_RecordMatrix = URecordMatrix[0];
                int_t sumtopolev_urecord = iUnzval_RecordMatrix[0];

                if (sumtopolev_urecord)
                {
                    for (int_t l = 1; l < sumtopolev_urecord + 1; l++)
                    {
                        int_t k = iUnzval_RecordMatrix[l];
                        UsedOrder_Unzval[i][idx++] = topoLvl - 1;
                        UsedOrder_Unzval[i][idx++] = k;
                    }
                    
                }                       
                
            }

            PrintInt10("UsedOrder_Unzval", numOrder * 2 + 1, UsedOrder_Unzval[i]);
        }
        SUPERLU_FREE(Unzval_RecordVector);
    }
    fclose(fp);

}
#endif

void save_Changed_LUstruct_txt(int_t n, gridinfo3d_t *grid3d, dLUstruct_t *LUstruct)
{
    
    gridinfo_t *grid=&(grid3d->grid2d);
    int_t nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t ntimestep=0;

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    
    nsupers = Glu_persist->supno[n-1] + 1;

    nb = CEILING(nsupers, grid->npcol);
      
    int_t index=INDEX_Lnzval_bc_ptr+grid3d->iam;  

    fpos_t *pos;
    // #pragma omp parallel for private(i,maxstep,slen,filename,fp,pos) schedule(dynamic,4)
    for(j=1;j<=nb/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=nb){
            maxstep=nb;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif

        int_t iPart;        

        fp=fopen(filename,"r+");        
        
        for(i=(j-1)*minstep;i<maxstep;i++)
        {
            iPart=0;

            if(Llu->isSave && GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[i][iPart]) != Unused){
                if(GetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[i][iPart]) == Changed && Llu->Lnzval_bc_ptr_ilen[i]){
                    fsetpos(fp,Llu->Lnzval_bc_ptr_fileposition[i]);
                    fwrite(Llu->Lnzval_bc_ptr[i], sizeof(double), Llu->Lnzval_bc_ptr_ilen[i], fp);
                }
                if(Llu->Lnzval_bc_ptr_ilen[i]){
                    SUPERLU_FREE(Llu->Lnzval_bc_ptr[i]);
                }
                SetVectorStatus(Llu->isUsed_Lnzval_bc_ptr[i][iPart], Unused);
            }
        }

        fclose(fp);
        
    }

    index=INDEX_Unzval_br_ptr+grid3d->iam;

    nb = CEILING(nsupers, grid->nprow);
    // #pragma omp parallel for private(i,maxstep,slen,filename,fp,pos) schedule(dynamic,4)
    for(j=1;j<=nb/minstep+1;j++)
    {        
        maxstep=j*minstep;
        if(maxstep>=nb){
            maxstep=nb;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif   

        int_t iPart;        

        fp=fopen(filename,"r+");       
        
        for(i=(j-1)*minstep;i<maxstep;i++)
        {
            iPart=0;

            if(Llu->isSave && GetVectorStatus(Llu->isUsed_Unzval_br_ptr[i][iPart]) != Unused){
                if(GetVectorStatus(Llu->isUsed_Unzval_br_ptr[i][iPart]) == Changed && Llu->Unzval_br_ptr_ilen[i]){
                    fsetpos(fp,Llu->Unzval_br_ptr_fileposition[i]);
                    fwrite(Llu->Unzval_br_ptr[i], sizeof(double), Llu->Unzval_br_ptr_ilen[i], fp);
                }
                if(Llu->Unzval_br_ptr_ilen[i]){
                    SUPERLU_FREE(Llu->Unzval_br_ptr[i]);
                }
                SetVectorStatus(Llu->isUsed_Unzval_br_ptr[i][iPart], Unused);
            }
        }

        fclose(fp);
        
    }

}

// Load LUstruct from txt
double* load_Lnzval_bc_ptr(int_t ljb, dLocalLU_t *Llu)
{
    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    double *lsub;
    if(Llu->Lnzval_bc_ptr_ilen[ljb]){
        lsub=DOUBLE_ALLOC(Llu->Lnzval_bc_ptr_ilen[ljb]);
    }
    else{
        lsub=NULL;
        return lsub;
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Lnzval_bc_ptr+iam;

    j=ljb/minstep+1;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    slen=slen+(j==0?1:(int)log(j)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
    #endif        

    if(access(filename,0)){
        printf("load_Lnzval_bc_ptr[%ld]: %s do not exist.\n",ljb,filename);
        lsub=NULL;
        return lsub;
    }
    else{
        fp=fopen(filename,"r");

        fpos_t *pos=Llu->Lnzval_bc_ptr_fileposition[ljb];
        fsetpos(fp,pos);
        fread(lsub, sizeof(double), Llu->Lnzval_bc_ptr_ilen[ljb], fp);

        fclose(fp);
        return lsub;
    }

}

double* load_Unzval_br_ptr(int_t lb, dLocalLU_t *Llu)
{
    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    double *uval;
    if(Llu->Unzval_br_ptr_ilen[lb]){
        uval=DOUBLE_ALLOC(Llu->Unzval_br_ptr_ilen[lb]);
    }
    else{
        uval=NULL;
        return uval;
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Unzval_br_ptr+iam;

    j=lb/minstep+1;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    slen=slen+(j==0?1:(int)log(j)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
    #endif        

    if(access(filename,0)){
        printf("load_Unzval_br_ptr[%ld] %s do not exist.\n",lb,filename);
        uval=NULL;
        return uval;
    }
    else{
        fp=fopen(filename,"r");

        fpos_t *pos=Llu->Unzval_br_ptr_fileposition[lb];
        fsetpos(fp,pos);
        fread(uval, sizeof(double), Llu->Unzval_br_ptr_ilen[lb], fp);

        fclose(fp);
        return uval;
    }

}

// set LUstruct vector to txt
int set_iLnzval_bc_ptr_txt(double *lsub, int_t ljb, int_t begin, int_t len, dLocalLU_t *Llu)
{
    if(!lsub && !Llu->Lnzval_bc_ptr_ilen[ljb]){
        return 0;
    }

    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    if(begin+len>Llu->Lnzval_bc_ptr_ilen[ljb]){
        ABORT("exceed the length of Lnzval_bc_ptr[]\n");
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Lnzval_bc_ptr+iam;

    j=ljb/minstep+1;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    slen=slen+(j==0?1:(int)log(j)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
    #endif        

    if(access(filename,0)){
        printf("set_iLnzval_bc_ptr_txt %s do not exist.\n",filename);
        return 0;
    }
    else{
        fp=fopen(filename,"r+");

        fpos_t *pos=Llu->Lnzval_bc_ptr_fileposition[ljb]; 
        fsetpos(fp,pos);
        fseek(fp,begin*sizeof(double),SEEK_CUR);
        fwrite(lsub, sizeof(double), len, fp);
        fclose(fp);

        return 1;
    }

}

int set_iUnzval_br_ptr_txt(double *uval, int_t lb, int_t begin, int_t len, dLocalLU_t *Llu)
{
    if(!uval && !Llu->Unzval_br_ptr_ilen[lb]){
        return 0;
    }

    int iam;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam );
    
    if(begin+len>Llu->Unzval_br_ptr_ilen[lb]){
        ABORT("exceed the length of Unzval_br_ptr[]\n");
    }

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    int_t ntimestep=0;

    int_t index=INDEX_Unzval_br_ptr+iam;

    j=lb/minstep+1;

    slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
    slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
    slen=slen+(index==0?1:(int)log(index)+1);
    slen=slen+(j==0?1:(int)log(j)+1);
    filename=(char*)malloc(slen+1);
    #if defined (_LONGINT)
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
    #else /* Default */
    sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
    #endif        

    if(access(filename,0)){
        printf("set_iUnzval_br_ptr_txt %s do not exist.\n",filename);
        return 0;
    }
    else{
        fp=fopen(filename,"r+");

        fpos_t *pos=Llu->Unzval_br_ptr_fileposition[lb]; 
        fsetpos(fp,pos);
        fseek(fp,begin*sizeof(double),SEEK_CUR);
        fwrite(uval, sizeof(double), len, fp);
        fclose(fp);
        return 1;
    }

}

// every l vector to save for testing
void save_nLUstruct_txt(int_t n, gridinfo3d_t *grid3d, dLUstruct_t *LUstruct, int_t l)
{
    
    gridinfo_t *grid=&(grid3d->grid2d);
    int_t nb, nsupers;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    int_t ntimestep=0;

    int_t minstep=MINSAVESTEP;
    int_t i,j,maxstep;
    char *filename;
    int slen;
    FILE *fp;
    
    nsupers = Glu_persist->supno[n-1] + 1;

    nb = CEILING(nsupers, grid->npcol);
      
    int_t index=INDEX_Lnzval_bc_ptr+grid3d->iam;  

    #pragma omp parallel for private(i,maxstep,slen,filename,fp), schedule(dynamic,4)
    for(j=1;j<=nb/minstep+1;j++)
    {
        maxstep=j*minstep;
        if(maxstep>=nb){
            maxstep=nb;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif

        fp=fopen(filename,"w");
            
        for(i=(j-1)*minstep;i<maxstep;i++)
        {
            if ( Llu->Lrowind_bc_ptr[i] && Llu->Lnzval_bc_ptr_ilen[i] && i%l == 0) {
                fwrite(Llu->Lnzval_bc_ptr[i], sizeof(double), Llu->Lnzval_bc_ptr_ilen[i], fp);
            }
        }

        fclose(fp);
        
    }

    index=INDEX_Unzval_br_ptr+grid3d->iam;

    nb = CEILING(nsupers, grid->nprow);
    #pragma omp parallel for private(i,maxstep,slen,filename,fp), schedule(dynamic,4)
    for(j=1;j<=nb/minstep+1;j++)
    {        
        maxstep=j*minstep;
        if(maxstep>=nb){
            maxstep=nb;
        }
        slen=strlen("/home/412-23/test/superlu_test/cage15/data-\0")+strlen("_\0")+strlen(".txt\0")+strlen("-\0")-4;
        slen=slen+(ntimestep==0?1:(int)log(ntimestep)+1);
        slen=slen+(index==0?1:(int)log(index)+1);
        slen=slen+(j==0?1:(int)log(j)+1);
        filename=(char*)malloc(slen+1);
        #if defined (_LONGINT)
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%ld_%ld-%ld.txt\0",index,ntimestep,j);
        #else /* Default */
        sprintf(filename,"/home/412-23/test/superlu_test/cage15/data-%8d_%8d-%8d.txt\0",index,ntimestep,j);
        #endif        

        fp=fopen(filename,"w");
        for(i=(j-1)*minstep;i<maxstep;i++)
        {            
            if ( Llu->Ufstnz_br_ptr[i] && Llu->Unzval_br_ptr_ilen[i]  && i%l == 0 ) {
                fwrite(Llu->Unzval_br_ptr[i], sizeof(double), Llu->Unzval_br_ptr_ilen[i], fp);
            }            
        }

        fclose(fp);
        
    }

}

#endif
#endif
