Building instructions for Windows

Required libraries: Intel MKL, OpenTK, ManagedCUDA, CUDA Toolkit
Hardware: CUDA device with compute capability 3.0 or above
Use 64 bit architecture consistently for all libraries and PTX.

First, build PardisoLoader.dll, which is a wrapper for PARDISO solver.
Copy the following libraries to the project folder:
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

For compute capability 3.0, rebuild collision_kernels, cz_kernels and elem_kernels into 
corresponding .ptx files.
If your CUDA device supports compute capability 5.2+, just use the provided PTX files.
Build OpenTK, ManagedCUDA, icFlowLibrary, and v3GUI. 
Simulation setups that reproduce the results from the paper will be posted online at this address:
BatchRun console app will run all simulaitons in the _sims folder.
BatchEdit can be used for analysis of batch resutls, easy edit of multiple simulations and for exporting .CSV files.


Building on Linux/MacOS

The project was not developed in Linux, but should work. Portions of the code were tested in Linux.
Install Intel MKL, CUDA Toolkit and Mono.
Compile ParidosoLoader proejct into .so library (minor change to source code required).
Build ManagedCUDA, icFlowLibrary, and BatchRun.
Create/copy appropriate DllMap file for DllImport, if needed.
v3GUI uses Windows Forms and OpenTK.
