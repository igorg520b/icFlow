// per-element integer data offsets
#define N0_OFFSET_ELEM 0
#define PCSR_OFFSET_ELEM 4 //16+4=20 entries   K / rhs
#define ROWSIZE_OFFSET_ELEM 24
#define INT_DATA_SIZE_ELEM 28

// offsets in the nodes array
#define X0_OFFSET 0
#define UN_OFFSET 3
#define VN_OFFSET 6
#define AN_OFFSET 9
#define F_OFFSET 12

// #include "mass_spring.h"


// the results of this function subsequently go into the equaiton of motion
// f[12] = elastic forces acting on nodes
// Df[12][12] = df/dx
// V = tetrahedron rest volume 
__device__ void F_and_Df_Corotational(
	const double(&x0)[12], const double(&un)[12],
	double(&f)[12], double(&Df)[12][12], double &V) {

	// Colorational formulation:
	// f = RK(Rt xc - x0)
	// Df = R K Rt

	double xc[12];
	for (int i = 0; i < 12; i++) xc[i] = x0[i] + un[i];

	// calculate K
	double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
	double x12, x13, x14, x23, x24, x34, x21, x31, x32, x42, x43, y12, y13, y14, y23, y24, y34;
	double y21, y31, y32, y42, y43, z12, z13, z14, z23, z24, z34, z21, z31, z32, z42, z43;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
	double Jdet;
	x1 = x0[0]; y1 = x0[1]; z1 = x0[2];
	x2 = x0[3]; y2 = x0[4]; z2 = x0[5];
	x3 = x0[6]; y3 = x0[7]; z3 = x0[8];
	x4 = x0[9]; y4 = x0[10]; z4 = x0[11];

	x12 = x1 - x2; x13 = x1 - x3; x14 = x1 - x4; x23 = x2 - x3; x24 = x2 - x4; x34 = x3 - x4;
	x21 = -x12; x31 = -x13; x32 = -x23; x42 = -x24; x43 = -x34;
	y12 = y1 - y2; y13 = y1 - y3; y14 = y1 - y4; y23 = y2 - y3; y24 = y2 - y4; y34 = y3 - y4;
	y21 = -y12; y31 = -y13; y32 = -y23; y42 = -y24; y43 = -y34;
	z12 = z1 - z2; z13 = z1 - z3; z14 = z1 - z4; z23 = z2 - z3; z24 = z2 - z4; z34 = z3 - z4;
	z21 = -z12; z31 = -z13; z32 = -z23; z42 = -z24; z43 = -z34;
	Jdet = x21 * (y23 * z34 - y34 * z23) + x32 * (y34 * z12 - y12 * z34) + x43 * (y12 * z23 - y23 * z12);
	V = Jdet / 6.;

	a1 = y42 * z32 - y32 * z42; b1 = x32 * z42 - x42 * z32; c1 = x42 * y32 - x32 * y42;
	a2 = y31 * z43 - y34 * z13; b2 = x43 * z31 - x13 * z34; c2 = x31 * y43 - x34 * y13;
	a3 = y24 * z14 - y14 * z24; b3 = x14 * z24 - x24 * z14; c3 = x24 * y14 - x14 * y24;
	a4 = y13 * z21 - y12 * z31; b4 = x21 * z13 - x31 * z12; c4 = x13 * y21 - x12 * y31;

	a1 /= Jdet; a2 /= Jdet; a3 /= Jdet; a4 /= Jdet;
	b1 /= Jdet; b2 /= Jdet; b3 /= Jdet; b4 /= Jdet;
	c1 /= Jdet; c2 /= Jdet; c3 /= Jdet; c4 /= Jdet;

	double B[6][12] = {
		{ a1, 0, 0, a2, 0, 0, a3, 0, 0, a4, 0, 0 },
		{ 0, b1, 0, 0, b2, 0, 0, b3, 0, 0, b4, 0 },
		{ 0, 0, c1, 0, 0, c2, 0, 0, c3, 0, 0, c4 },
		{ b1, a1, 0, b2, a2, 0, b3, a3, 0, b4, a4, 0 },
		{ 0, c1, b1, 0, c2, b2, 0, c3, b3, 0, c4, b4 },
		{ c1, 0, a1, c2, 0, a2, c3, 0, a3, c4, 0, a4 } };

	double BtE[12][6] = {}; // result of multiplication (Bt x E)
	for (int r = 0; r < 12; r++)
		for (int c = 0; c < 6; c++)
			for (int i = 0; i < 6; i++) BtE[r][c] += B[i][r] * E[i][c];

	// K = Bt x E x B x V
	double K[12][12] = {};
	for (int r = 0; r < 12; r++)
		for (int c = 0; c < 12; c++)
			for (int i = 0; i < 6; i++) K[r][c] += BtE[r][i] * B[i][c] * V;

	double R0[3][3], R1[3][3], R[3][3];
	fastRotationMatrix(
		x0[0], x0[1], x0[2],
		x0[3], x0[4], x0[5],
		x0[6], x0[7], x0[8],
		R0[0][0], R0[0][1], R0[0][2],
		R0[1][0], R0[1][1], R0[1][2],
		R0[2][0], R0[2][1], R0[2][2]);

	fastRotationMatrix(
		xc[0], xc[1], xc[2],
		xc[3], xc[4], xc[5],
		xc[6], xc[7], xc[8],
		R1[0][0], R1[0][1], R1[0][2],
		R1[1][0], R1[1][1], R1[1][2],
		R1[2][0], R1[2][1], R1[2][2]);

	multABd(
		R1[0][0], R1[0][1], R1[0][2],
		R1[1][0], R1[1][1], R1[1][2],
		R1[2][0], R1[2][1], R1[2][2],
		R0[0][0], R0[1][0], R0[2][0],
		R0[0][1], R0[1][1], R0[2][1],
		R0[0][2], R0[1][2], R0[2][2],
		R[0][0], R[0][1], R[0][2],
		R[1][0], R[1][1], R[1][2],
		R[2][0], R[2][1], R[2][2]);

	double RK[12][12] = {};
	double RKRt[12][12] = {};

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
		{
			// RK = R * K
			for (int k = 0; k < 3; k++)
				for (int l = 0; l < 3; l++) {
					for (int m = 0; m < 3; m++)
						RK[3 * i + k][3 * j + l] += R[k][m] * K[3 * i + m][3 * j + l];
				}

			// RKRT = RK * R^T
			for (int k = 0; k < 3; k++)
				for (int l = 0; l < 3; l++) {
					for (int m = 0; m < 3; m++)
						RKRt[3 * i + k][3 * j + l] += RK[3 * i + k][3 * j + m] * R[l][m];
				}
		}

	// xr = Rt xc
	double xr[12] = {};
	multAX(R[0][0], R[1][0], R[2][0],
		R[0][1], R[1][1], R[2][1],
		R[0][2], R[1][2], R[2][2],
		xc[0], xc[1], xc[2],
		xr[0], xr[1], xr[2]);
	multAX(R[0][0], R[1][0], R[2][0],
		R[0][1], R[1][1], R[2][1],
		R[0][2], R[1][2], R[2][2],
		xc[3], xc[4], xc[5],
		xr[3], xr[4], xr[5]);
	multAX(R[0][0], R[1][0], R[2][0],
		R[0][1], R[1][1], R[2][1],
		R[0][2], R[1][2], R[2][2],
		xc[6], xc[7], xc[8],
		xr[6], xr[7], xr[8]);
	multAX(R[0][0], R[1][0], R[2][0],
		R[0][1], R[1][1], R[2][1],
		R[0][2], R[1][2], R[2][2],
		xc[9], xc[10], xc[11],
		xr[9], xr[10], xr[11]);

	for (int i = 0; i < 12; i++) xr[i] -= x0[i];

	// f = RK(Rt pm - mx)
	// Df = RKRt
	for (int i = 0; i < 12; i++)
		for (int j = 0; j < 12; j++) {
			f[i] += RK[i][j] * xr[j];
			Df[i][j] = RKRt[i][j];
		}
}


__device__ void AssembleBCSR12(const int *pcsr, double *A, double *b, const int el_elastic_stride,
	double(&LHS)[12][12], double(&rhs)[12]) {

	// BCSR format
	// distribute LHS into global matrix, and rhs into global rhs
	for (int i = 0; i < 4; i++)
	{
		// distribute K
		for (int j = 0; j < 4; j++)
		{
			int idx1 = pcsr[el_elastic_stride * (i * 4 + j)];
			if (idx1 >= 0)
			{
				// write into csr.vals
				for (int k = 0; k < 3; k++)
					for (int l = 0; l < 3; l++)
						atomicAdd2(&A[idx1 * 9 + 3 * k + l], LHS[i * 3 + k][j * 3 + l]);
			}
		}
		// distribute rhs
		int idx2 = pcsr[el_elastic_stride * (16 + i)];
		if (idx2 >= 0)
		{
			for (int k = 0; k < 3; k++)
				atomicAdd2(&b[idx2 * 3 + k], rhs[i * 3 + k]);
		}
	}
}


__device__ void Assemble_CSR_Nonsymmetric_12(const int *pcsr, double *A, double *b, const int el_elastic_stride,
	double(&LHS)[12][12], double(&rhs)[12]) {

	// CSR format
	for (int i = 0; i < 4; i++)
	{
		int idx2 = pcsr[el_elastic_stride * (16 + i)];
		int rowSize = pcsr[el_elastic_stride * (20 + i)];

		// distribute rhs
		if (idx2 >= 0)
			for (int k = 0; k < 3; k++) atomicAdd2(&b[idx2 * 3 + k], rhs[i * 3 + k]);

		// distribute K
		for (int j = 0; j < 4; j++)
		{
			int idx1 = pcsr[el_elastic_stride * (i * 4 + j)];
			if (idx1 >= 0)
			{
				// write into csr.vals
				for (int k = 0; k < 3; k++)
					for (int l = 0; l < 3; l++)
						atomicAdd2(&A[idx1 + 3 * rowSize * k + l], LHS[i * 3 + k][j * 3 + l]);
			}
		}
	}
}

// ker_ElementElasticityForce compute and distributes the stiffness matrix and the rhs per element
// h: timestep
// assembly_type is 0 for BCSR and 1 for CSR
extern "C" __global__ void kelElementElasticityForce(
	const int *ie, double *dn, const double h,
	double *global_matrix, double *global_rhs,
	const int nElasticElems, const int el_elastic_stride, const int nd_stride, const int assembly_type)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nElasticElems) return;

	// transfer values from buffer to local arrays
	double x0[12], un[12], vn[12], an[12]; // material coords, displacement, velocity, acceleration
										   // retrieve node coordinates from global memory
	int nid[4];
	for (int i = 0; i < 4; i++) {			// node
		int nn = ie[idx + el_elastic_stride * (N0_OFFSET_ELEM + i)]; // node id
		nid[i] = nn;
		for (int j = 0; j < 3; j++)		// coordinate
		{
			int idx1 = j + i * 3;
			x0[idx1] = dn[nn + nd_stride*(X0_OFFSET + j)];
			un[idx1] = dn[nn + nd_stride*(UN_OFFSET + j)];
			vn[idx1] = dn[nn + nd_stride*(VN_OFFSET + j)];
			an[idx1] = dn[nn + nd_stride*(AN_OFFSET + j)];
		}
	}

	// compute K and f
	double Df[12][12] = {};
	double f[12] = {};
	double V; // rest volume

	F_and_Df_Corotational(x0, un, f, Df, V);
//	F_and_Df_Spring(x0, un, f, Df, V);

	// prepare left and right sides of the equation of motion
	double rhs[12] = {};		// rhs = -(f+g) - M an - D vn
	double LHS[12][12];			// LHS = M / (h^2 beta) + D gam / (h beta) - Df
	double gravityForcePerNode = gravity * rho * V / 4;

	rhs[2] += gravityForcePerNode;
	rhs[5] += gravityForcePerNode;
	rhs[8] += gravityForcePerNode;
	rhs[11] += gravityForcePerNode;
		
	// assemble the effective stiffness matrix Keff = M/(h^2 beta) + RKRt + D * gamma /(h beta) 
	// where D is the damping matrix D = a M + b K
	double massCoeff = V * (1.0 / (h*h) + dampingMass * NewmarkGamma / h) / NewmarkBeta;
	double stiffCoeff = 1.0 + dampingStiffness * NewmarkGamma / (h * NewmarkBeta);

	// add damping component to rhs
	// D = M[i][j] * V * dampingMass + RKRt[i][j] * dampingStiffness

	for (int i = 0; i < 12; i++) {
		rhs[i] -= f[i];
		for (int j = 0; j < 12; j++) {
			rhs[i] -= (M[i][j] * V * dampingMass + Df[i][j] * dampingStiffness) * vn[j] + (M[i][j] * V * an[j]);
			LHS[i][j] = Df[i][j] * stiffCoeff + M[i][j] * massCoeff;
		}
	}

	// Assembly stage

	// distribute computed forces to fx fy fz for visualization
	for (int i = 0; i < 4; i++) {			// node
		int nn = nid[i]; // pre-fetched node id
		for (int j = 0; j < 3; j++)		// coordinate
			atomicAdd2(&dn[nn + nd_stride*(F_OFFSET + j)], rhs[i * 3 + j]);
	}

	if (assembly_type == 0)  // BCSR format
		AssembleBCSR12(&ie[idx + el_elastic_stride*PCSR_OFFSET_ELEM], global_matrix, global_rhs, el_elastic_stride, LHS, rhs);
	else if (assembly_type == 1) // CSR format
		Assemble_CSR_Nonsymmetric_12(&ie[idx + el_elastic_stride*PCSR_OFFSET_ELEM], global_matrix, global_rhs, el_elastic_stride, LHS, rhs);
}