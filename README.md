# icFlow
Source code for fracture simulaiton, based on cohesive zone model and implicit finite element method. The formulation and results are described in the following papers (under review):
* Parallel Implementation of Implicit Finite Element Model with Cohesive Zones and Collision Response using CUDA
* Cohesive Zone Micromechanical Model for Compressive and Tensile Failure of Polycrystalline Ice

Sample simulation setups are available at: https://goo.gl/AvZC4u

Additional geometries can be generated with Never: http://neper.sourceforge.net/

This code requires Intel Math Kernel Library

## Building instructions for Windows

Required libraries: Intel MKL, OpenTK, ManagedCUDA, CUDA Toolkit 9.0
Hardware: CUDA device with compute capability 3.0 or above
Use 64 bit architecture consistently for all libraries and PTX.

First, build PardisoLoader.dll, which is a wrapper for PARDISO solver.
Copy the following libraries to the project folder (or make them accessible via PATH variable):
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

Rebuild collision_kernels, cz_kernels and elem_kernels into corresponding .ptx files. Set compute capability to 3.0 or higher. Build OpenTK, ManagedCUDA, icFlowLibrary, and SimGUI. 
