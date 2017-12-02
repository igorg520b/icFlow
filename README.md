# icFlow
Source code for fracture simulaiton, based on cohesive zone model and implicit finite element method. The formulation and results are described in the following papers (under review):
* Parallel Implementation of Implicit Finite Element Model with Cohesive Zones and Collision Response using CUDA
* Cohesive Zone Micromechanical Model for Compressive and Tensile Failure of Polycrystalline Ice

This code requires Intel Math Kernel Library

## Building instructions for Visual Studio in Windows

Required libraries: Intel MKL, OpenTK, ManagedCUDA, CUDA Toolkit 9.0. It is possible to build the library with other versions of CUDA Toolkit, but the supplied project files only support CUDA 9.0. Use 64 bit architecture consistently for all libraries and PTX.

Copy MKL libraries (64 bit) to the project folder or make them accessible via PATH variable:
mkl_avx.dll
mkl_avx2.dll
mkl_avx512.dll
mkl_avx512_mic.dll
mkl_core.dll
mkl_def.dll
mkl_intel_thread.dll
mkl_mc.dll
mkl_mc3.dll
mkl_rt.dll
mkl_sequential.dll
mkl_tbb_thread.dll
mkl_vml_avx.dll
mkl_vml_avx2.dll
mkl_vml_avx512.dll
mkl_vml_avx512_mic.dll
mkl_vml_cmpt.dll
mkl_vml_def.dll
mkl_vml_mc.dll
mkl_vml_mc2.dll
mkl_vml_mc3.dll

Build PardisoLoader2.dll, which is a wrapper for MKL's PARDISO solver.
Rebuild collision_kernels, cz_kernels and elem_kernels into corresponding .ptx files. Set compute capability to 3.0 or higher. 
Build OpenTK, ManagedCUDA, icFlowLibrary, and SimGUI. Load the simulation setup from one of the following:
https://goo.gl/AvZC4u
Additional geometries can be generated with Neper: http://neper.sourceforge.net/
