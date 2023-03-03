############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   March 1, 2016	version 5.0.0
#
#  Modified:	    October 13, 2017    version 5.2.1
#		    February 20, 2021   version 7.0.0
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
SuperLUroot = /home/412-23/software/superlu_dist-master/program/local
#DSUPERLULIB = $(SuperLUroot)/SRC/libsuperlu_dist.a
DSUPERLULIB = $(SuperLUroot)//home/412-23/software/superlu_dist-master/program/lib64/libsuperlu_dist.a
INCLUDEDIR  = $(SuperLUroot)//home/412-23/software/superlu_dist-master/program/include

XSDK_INDEX_SIZE = 64
SLU_HAVE_LAPACK = FALSE
HAVE_PARMETIS   = TRUE
HAVE_COMBBLAS   = 
HAVE_CUDA       = TRUE

XSDK_ENABLE_Fortran = OFF
ifeq ($(XSDK_ENABLE_Fortran),ON)
  DFORTRANLIB = $(SuperLUroot)//home/412-23/software/superlu_dist-master/program/lib64/
  LIBS = $(DFORTRANLIB) $(DSUPERLULIB) /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64_lin/libmkl_gf_lp64.so /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64_lin/libmkl_gnu_thread.so /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64_lin/libmkl_core.so /usr/local/lib64/libgomp.so -lm -ldl -lm
  LIBS += 
else
  LIBS = $(DSUPERLULIB) /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64_lin/libmkl_gf_lp64.so /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64_lin/libmkl_gnu_thread.so /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/lib/intel64_lin/libmkl_core.so /usr/local/lib64/libgomp.so -lm -ldl -lm
endif

LIBS	 += 
LIBS	 += /home/412-23/software/backage/parmetis-4.0.3/build/Linux-x86_64/libparmetis/libparmetis.a /home/412-23/software/backage/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a
LIBS 	 += 
LIBS 	 += -lgcc_s -lgcc -lc -lpthread -lgcc_s -lgcc -lmpi 
# LIBS     += 

CUDALIBS = /public/software/cuda/10.2/lib64/libcudart.so /public/software/cuda/10.2/lib64/libcublas.so
LIBS     += $(CUDALIBS)

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /public/software/mpi/hpcx/2.8.1/ompi/bin/mpicc
CFLAGS 	     = -O3 -DNDEBUG -I/home/412-23/software/backage/parmetis-4.0.3/metis/include/ -I/home/412-23/software/backage/parmetis-4.0.3/metis/build/include/ -DUSE_VENDOR_BLAS -fopenmp -liomp5 -lpthread  -std=c99  -fopenmp -DDEBUGlevel=1 
##-fPIC
#CFLAGS      += -D
# CFLAGS     += 
CXX          = /public/software/mpi/hpcx/2.8.1/ompi/bin/mpicxx
CXXFLAGS     = -O3 -DNDEBUG -fopenmp -liomp5 -lpthread   -fopenmp -DDEBUGlevel=1
NVCC	     = /public/software/cuda/10.2/bin/nvcc
NVCCFLAGS    = -DUSE_VENDOR_BLAS --disable-warnings -DPRNTlevel=1 -DPROFlevel=0  -gencode arch=compute_70,code=sm_70

NOOPTS       = -O0
FORTRAN	     = /usr/bin/gfortran
FFLAGS	     =  

LOADER       = /public/software/mpi/hpcx/2.8.1/ompi/bin/mpicxx
LOADOPTS     =   -DDEBUGlevel=1 -fopenmp -liomp5 -lpthread  
