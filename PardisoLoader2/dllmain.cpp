#include<iostream>
#include<sstream>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl.h"


BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
	)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}


// ja = rows; ia = cols; a = vals
extern "C" __declspec(dllexport) int SolveDouble3(int *ja, int *ia, double *a,
	const int n, double *b, double *x, int _mtype, int iparam4, int dim, int msglvl_, int check) {
	//	mkl_verbose(0);
	//	std::stringstream redirectStream;
	//	std::cout.rdbuf(redirectStream.rdbuf());

	MKL_INT mtype = _mtype;       // Real symmetric matrix: -2;  real unsymmetric: 11 
	MKL_INT nrhs = 1;     // Number of right hand sides. 
	void *pt[64];
	MKL_INT iparm[64] = {};
	MKL_INT maxfct, mnum, phase, error, msglvl;
	MKL_INT i;
	MKL_INT idum;
	iparm[0] = 1;         // No solver default 
	iparm[1] = 3;         // Fill-in reordering from METIS (was 2)
	iparm[3] = iparam4;         // No iterative-direct algorithm 
	iparm[4] = 0;         // No user fill-in reducing permutation 
	iparm[5] = 0;         // Write solution into x 
	iparm[6] = 0;         // Not in use 
	iparm[7] = 0;         // Max numbers of iterative refinement steps 
	iparm[8] = 0;
	iparm[9] = 8;        // Perturb the pivot elements with 1E-13 
	iparm[10] = 0;        // Use nonsymmetric permutation and scaling MPS 
	iparm[11] = 0;
	iparm[12] = 0;        // Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy 
	iparm[13] = 0;        // Output: Number of perturbed pivots 
	iparm[14] = 0;
	iparm[15] = 0;
	iparm[16] = 0;
	iparm[17] = 1;       // 1 - disable report; Output: Number of nonzeros in the factor LU 
	iparm[18] = 1;			// 1- disable report; output number of operations
	iparm[19] = 0;
	iparm[26] = 0; // check matrix structure for errors
	iparm[27] = 0; // 0 double; 1 single
	iparm[34] = 1;// zero-base index
	iparm[36] = dim;// BSR with block size 3
	maxfct = 1;
	mnum = 1;
	msglvl = msglvl_; // use 1 for verbose output
	error = 0;
	for (i = 0; i < 64; i++) pt[i] = 0;
	phase = 13;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);

	phase = -1;
	double ddum;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	return error;
}