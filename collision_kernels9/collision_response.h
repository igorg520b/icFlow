#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "point_triange_v2.h"
#include "atomic_add.h"

__device__ void AssembleBCSR12(const int *pcsr, double *A, double *b, const int stride,
	double(&LHS)[12][12], double(&rhs)[12]) {

	// BCSR format
	// distribute LHS into global matrix, and rhs into global rhs
	for (int i = 0; i < 4; i++)
	{
		// distribute K
		for (int j = 0; j < 4; j++)
		{
			int idx1 = pcsr[stride * (i * 4 + j)];
			if (idx1 >= 0)
			{
				// write into csr.vals
				for (int k = 0; k < 3; k++)
					for (int l = 0; l < 3; l++)
						atomicAdd2(&A[idx1 * 9 + 3 * k + l], LHS[i * 3 + k][j * 3 + l]);
			}
		}
		// distribute rhs
		int idx2 = pcsr[stride * (16 + i)];
		if (idx2 >= 0)
		{
			for (int k = 0; k < 3; k++)
				atomicAdd2(&b[idx2 * 3 + k], rhs[i * 3 + k]);
		}
	}
}


// ker_ElementElasticityForce compute and distributes the stiffness matrix and the rhs per element
// h: timestep
extern "C" __global__ void kCollisionResponseForce(
	const int *icr, double *dn, const double h,
	double *global_matrix, double *global_rhs,
	const int nImpacts, const int cr_stride, const int nd_stride,
	const double k, const double distanceEpsilon)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nImpacts) return;

	// filter out rigid-rigid collisions
	const int *pcsr = &icr[idx + cr_stride * 4];
	int ndp[4];
	for (int i = 0; i < 4; i++) ndp[i] = pcsr[cr_stride * (16 + i)];
	if (ndp[0] < 0 && ndp[1] < 0 && ndp[2] < 0 && ndp[3] < 0) return;

	// transfer values from buffer to local arrays
	double xc[12];
	// retrieve node coordinates from global memory

	int nid[4];
	for (int i = 0; i < 4; i++) {			// node
		int nn = icr[idx + cr_stride * i]; // node id
		nid[i] = nn;
		for (int j = 0; j < 3; j++)		// coordinate
		{
			int idx1 = j + i * 3;
			xc[idx1] = dn[nn + nd_stride*(X_CURRENT_OFFSET + j)];
		}
	}

	double w[3] = {};
//	double wd[3][12] = {};
//	double wdd[3][12][12] = {};
	double sqrdistd[12] = {};
	double sqrdistdd[12][12] = {};

	//pt(double(&x)[12], double(&fd)[12], double(&sd)[12][12], double &output_s, double &output_t, int &branch)

	/*
	double dsq = PT_Derivatives(
		xc[0], xc[1], xc[2],
		xc[3], xc[4], xc[5],
		xc[6], xc[7], xc[8],
		xc[9], xc[10], xc[11],
		w, wd, wdd, sqrdistd, sqrdistdd);
		*/
	double output_s, output_t;
	int branch;

	double dsq = pt(xc, sqrdistd, sqrdistdd, output_s, output_t, branch);

	w[1] = output_s; w[2] = output_t; w[0] = 1 - (output_s + output_t);

	if (dsq > distanceEpsilon*distanceEpsilon) {
		double fx, fy, fz;
		fx = k * 0.5 * sqrdistd[0];
		fy = k * 0.5 * sqrdistd[1];
		fz = k * 0.5 * sqrdistd[2];

		double fi[12];
		fi[0] = -fx;
		fi[1] = -fy;
		fi[2] = -fz;
		fi[3] = w[0] * fx;
		fi[4] = w[0] * fy;
		fi[5] = w[0] * fz;
		fi[6] = w[1] * fx;
		fi[7] = w[1] * fy;
		fi[8] = w[1] * fz;
		fi[9] = w[2] * fx;
		fi[10] = w[2] * fy;
		fi[11] = w[2] * fz;

		double dfij[12][12];
		for (int i = 0; i < 12; i++)
			for (int j = i; j < 12; j++)
				dfij[i][j] = dfij[j][i] = k * sqrdistdd[i][j] / 2;

		// distribute computed forces to fx fy fz (for analysis and visualization)
		for (int i = 0; i < 4; i++) {			// node
			int nn = nid[i]; // pre-fetched node id
			for (int j = 0; j < 3; j++)		// coordinate
				atomicAdd2(&dn[nn + nd_stride*(F_OFFSET + j)], fi[i * 3 + j]);
		}

		// distribute result
		AssembleBCSR12(pcsr, global_matrix, global_rhs, cr_stride, dfij, fi);
	}

}