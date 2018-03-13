#include"PPR_math.h"

// node indices
#define X0_OFFSET 0
#define UN_OFFSET 3
#define VN_OFFSET 6
#define AN_OFFSET 9
#define F_OFFSET 12
#define X_CURRENT_OFFSET 15
#define FP_DATA_SIZE_NODE 18

// CZ fp data
#define CURRENT_PMAX_OFFSET_CZ 0
#define CURRENT_TMAX_OFFSET_CZ 3
#define TENTATIVE_PMAX_OFFSET_CZ 6
#define TENTATIVE_TMAX_OFFSET_CZ 9
#define FP_DATA_SIZE_CZ 12

// CZ integer data
#define CURRENT_FAILED_OFFSET_CZ 0
#define TENTATIVE_CONTACT_OFFSET_CZ 1
#define TENTATIVE_DAMAGED_OFFSET_CZ 2
#define TENTATIVE_FAILED_OFFSET_CZ 3
#define VRTS_OFFSET_CZ 4
#define PCSR_OFFSET_CZ 10 // 36+6=42 values
#define ROWSIZE_OFFSET_CZ 52 
#define INT_DATA_SIZE_CZ 58





/*
// smooth transition (for attenuation)
__device__ double N2(double x) {
if (x < 0) return 1;
else if (x < 0.5) return 1 - 2 * x*x;
else if (x < 1) return 2 * (x - 1)*(x - 1);
else return 0;
}

__device__ double N2Derivative(double x) {
if (x < 0) return 0;
else if (x < 0.5) return -4 * x;
else if (x < 1) return 4 * (x - 1);
else return 0;
}
__constant__ double k_coeff = 0.15; // special smoothing coefficient

// attenuation in the normal direction
__device__ double atn(double opn, double pmax) {
return N2((pmax - opn) / (k_coeff * deln * rn));
}

__device__ double atnDerivative(double opn, double pmax) {
return N2Derivative((pmax - opn) / (k_coeff * deln * rn)) / (-k_coeff * deln * rn);
}


// attenuation in the tangential direction
__device__ double att(double opt, double tmax) {
return N2((tmax - opt) / (k_coeff * delt * rt));
}

__device__ double attDerivative(double opt, double tmax) {
return N2Derivative((tmax - opt) / (k_coeff * delt * rt)) / (-k_coeff * delt * rt);
}

// with smoothing (1)
__device__ void cohesive_law_smoothed(bool &cz_contact, bool &cz_failed, double &pmax, double &tmax, double opn, double opt,
	double &Tn, double &Tt, double &Dnn, double &Dtt, double &Dnt, double &Dtn) {
	// Dnt = D[Tn,t]; Dtn = D[Tt,n]

	Tn = Tt = Dnn = Dtt = Dnt = Dtn = 0;
	if (opn > deln || opt > delt) {
		cz_failed = true; return;
	}
	cz_contact = (opn < 0);
	const double epsilon = -1e-9;

	if (cz_contact)
	{
		Dnn = Dnn_(0, opt);
		Tn = Dnn * opn;
		Tt = Tt_(0, opt);
		if (Tt >= epsilon)
		{
			if (opt >= tmax)
			{
				// tangential softening
				if (opt >= delt * rt) tmax = opt;
				Dtt = Dtt_(0, opt);
			}
			else
			{
				// unload/reload
				Tt = Tt_(0, tmax) * opt / tmax;
				Dtt = Tt_(0, tmax) / tmax;
			}
		}
		else
		{
			// cz failed in tangential direction while in contact
			Tt = Dtt = Dnt = 0;
			Tn = Dnn = 0;
			cz_failed = true;
		}
	}
	else
	{
		// not in contact
		Tt = Tt_(opn, opt);
		Tn = Tn_(opn, opt);
		if (Tt >= epsilon && Tn >= epsilon)
		{
			// tangential component
			bool tsoft = (opt >= tmax);
			bool nsoft = (opn >= pmax);
			if (tsoft && nsoft)
			{
				// tangential and normal softening
				if (opt >= delt * rt) tmax = opt;
				if (opn >= deln * rn) pmax = opn;
				Dnn = Dnn_(opn, opt);
				Dnt = Dnt_(opn, opt);
				Dtt = Dtt_(opn, opt);
			}
			else if (tsoft && !nsoft) {
				// normal unload/reload
				if (opt >= delt * rt) tmax = opt;
				Tn = (1.0 - atn(opn, pmax))*Tn_(pmax, opt) * opn / pmax + atn(opn, pmax)*Tn_(opn, opt);
				Dnn = (1.0 - atn(opn, pmax))*Tn_(pmax, opt) / pmax - atnDerivative(opn, pmax) * Tn_(pmax, opt) * opn / pmax +
					Tn_(opn, opt) * atnDerivative(opn, pmax) + atn(opn, pmax)*Dnn_(opn, opt);
				Dnt = (Dnt_(opn, opt) + (atn(opn, pmax) * Dnt_(opn, opt) + (1 - atn(opn, pmax))*(opn / pmax)*Dnt_(pmax, opt)))*0.5;
				Dtt = Dtt_(pmax, opt);
			}
			else if (!tsoft && nsoft)
			{
				// tangential unloading/reloading
				if (opn >= deln * rn) pmax = opn;
				Tt = (1.0 - att(opt, tmax))*Tt_(opn, tmax) * opt / tmax + att(opt, tmax)*Tt_(opn, opt);

				Dtt = (1.0 - att(opt, tmax))*Tt_(opn, tmax) / tmax - attDerivative(opt, tmax) * Tt_(opn, tmax) * opt / tmax +
					Tt_(opn, opt) * attDerivative(opt, tmax) + atn(opt, tmax)*Dtt_(opn, opt);
				Dnn = Dnn_(opn, tmax);
				Dnt = (Dnt_(opn, opt) + ((1.0 - att(opt, tmax))*Dnt_(opn, tmax) * opt / tmax + att(opt, tmax)*Dnt_(opn, opt))) * 0.5;
			}
			else
			{
				// reloading in both tangential and normal
				Tn = (1.0 - atn(opn, pmax))*Tn_(pmax, opt) * opn / pmax + atn(opn, pmax)*Tn_(opn, opt);
				Tt = (1.0 - att(opt, tmax))*Tt_(opn, tmax) * opt / tmax + att(opt, tmax)*Tt_(opn, opt);

				Dnn = (1.0 - atn(opn, pmax))*Tn_(pmax, opt) / pmax - atnDerivative(opn, pmax) * Tn_(pmax, opt) * opn / pmax +
					Tn_(opn, opt) * atnDerivative(opn, pmax) + atn(opn, pmax)*Dnn_(opn, opt);
				Dtt = (1.0 - att(opt, tmax))*Tt_(opn, tmax) / tmax - attDerivative(opt, tmax) * Tt_(opn, tmax) * opt / tmax +
					Tt_(opn, opt) * attDerivative(opt, tmax) + atn(opt, tmax)*Dtt_(opn, opt);

				double Dnt1 = (Dnt_(opn, opt) + (atn(opn, pmax) * Dnt_(opn, opt) + (1 - atn(opn, pmax))*(opn / pmax)*Dnt_(pmax, opt)))*0.5;
				double Dnt2 = (Dnt_(opn, opt) + ((1.0 - att(opt, tmax))*Dnt_(opn, tmax) * opt / tmax + att(opt, tmax)*Dnt_(opn, opt))) * 0.5;
				Dnt = (Dnt1 + Dnt2)*0.5;
			}
		}
		else
		{
			cz_failed = true;
			Tn = Tt = Dnn = Dtt = Dnt = 0;
		}
	}
	Dtn = Dnt;
}
*/

// one spring (2)
__device__ void cohesive_law_spring(bool &cz_contact, bool &cz_failed, double &pmax, double &tmax, double opn, double opt,
	double &Tn, double &Tt, double &Dnn, double &Dtt, double &Dnt, double &Dtn) {

	cz_failed = false;
	cz_contact = false;
	double opening_sq = opn*opn + opt*opt;
	double opening = sqrt(opening_sq);
	double opening_cubed = opening_sq * abs(opening);
	double k = Dnn_(0, 0);
	double traction = k * opening;

	if (opening > 1e-20) {
		// normal calculation
		Tn = traction * opn / opening;
		Tt = traction * opt / opening;
		Dnn = opt*opt*traction / opening_cubed + opn*opn*k / opening_sq;
		Dtt = opn*opn*traction / opening_cubed + opt*opt*k / opening_sq;
		Dnt = opn*opt*(k / opening_sq - traction / opening_cubed);
	}
	else {
		// Limit opening => 0
		Tn = traction;
		Tt = traction;
		Dnn = k;
		Dtt = k;
		Dnt = 0;
	}
	Dtn = Dnt;
}

// one spring that changes stiffness Tn(opening, 0);
// with unloading stage
__device__ void cohesive_law_nonlinear_spring(bool &cz_contact, bool &cz_failed, double &pmax, double &tmax, double opn, double opt,
	double &Tn, double &Tt, double &Dnn, double &Dtt, double &Dnt, double &Dtn) {

	cz_failed = false;
	double opening_sq = opn*opn + opt*opt;
	double opening = sqrt(opening_sq);
	double opening_cubed = opening_sq * abs(opening);

	Tn = Tt = Dnn = Dtt = Dnt = 0;
	if (opening > deln) { cz_failed = true; return; }

	if (opening >= pmax) {

		if (opening > deln*rn) pmax = opening;
		double k = Dnn_(opening, 0);
		double traction = Tn_(opening, 0);

		if (opening > 1e-20) {
			// normal calculation
			Tn = traction * opn / opening;
			Tt = traction * opt / opening;
			Dnn = opt*opt*traction / opening_cubed + opn*opn*k / opening_sq;
			Dtt = opn*opn*traction / opening_cubed + opt*opt*k / opening_sq;
			Dnt = opn*opt*(k / opening_sq - traction / opening_cubed);
		}
		else {
			// Limit opening => 0
			Tn = traction;
			Tt = traction;
			Dnn = k;
			Dtt = k;
			Dnt = 0;
		}
	}
	else {
		double k = Tn_(pmax, 0) / pmax;
		double traction = opening *k;

		if (opening > 1e-20) {
			// normal calculation
			Tn = traction * opn / opening;
			Tt = traction * opt / opening;
			Dnn = opt*opt*traction / opening_cubed + opn*opn*k / opening_sq;
			Dtt = opn*opn*traction / opening_cubed + opt*opt*k / opening_sq;
			Dnt = opn*opt*(k / opening_sq - traction / opening_cubed);
		}
		else {
			// Limit opening => 0
			Tn = traction;
			Tt = traction;
			Dnn = k;
			Dtt = k;
			Dnt = 0;
		}
	}
	Dtn = Dnt;
}

// original version (0)
__device__ void cohesive_law(bool &cz_contact, bool &cz_failed, double &pmax, double &tmax, double opn, double opt,
	double &Tn, double &Tt, double &Dnn, double &Dtt, double &Dnt, double &Dtn) {
	Tn = Tt = Dnn = Dtt = Dnt = 0;
	if (opn > deln || opt > delt) {
		cz_failed = true; return;
	}
	cz_contact = (opn < 0);
	const double epsilon = -1e-9;

	if (cz_contact)
	{
		Dnn = Dnn_(0, opt);
		Tn = Dnn * opn;
		Tt = Tt_(0, opt);
		if (Tt >= epsilon)
		{
			if (opt >= tmax)
			{
				// tangential softening
				if (opt >= delt * rt) tmax = opt;
				Dtt = Dtt_(0, opt);
			}
			else
			{
				// unload/reload
				Tt = Tt_(0, tmax) * opt / tmax;
				Dtt = Tt_(0, tmax) / tmax;
			}

		}
		else
		{
			// cz failed in tangential direction while in contact
			Tt = Dtt = Dnt = 0;
			Tn = Dnn = 0;
			//			printf("contact; Tt = %e; Tn = %e; opn = %e; opt = %e\n", Tt, Tn, opn, opt);
			cz_failed = true;
		}
	}
	else
	{
		// not in contact
		Tt = Tt_(opn, opt);
		Tn = Tn_(opn, opt);
		if (Tt >= epsilon && Tn >= epsilon)
		{
			// tangential component
			bool tsoft = (opt >= tmax);
			bool nsoft = (opn >= pmax);
			if (tsoft && nsoft)
			{
				// tangential and normal softening
				if (opt >= delt * rt) tmax = opt;
				if (opn >= deln * rn) pmax = opn;
				Dnn = Dnn_(opn, opt);
				Dnt = Dnt_(opn, opt);
				Dtt = Dtt_(opn, opt);
			}
			else if (tsoft && !nsoft) {
				// normal unload/reload
				if (opt >= delt * rt) tmax = opt;
				Tn = Tn_(pmax, opt) * opn / pmax;
				Dnn = Tn_(pmax, opt) / pmax;
				Dtt = Dtt_(pmax, opt);
				//				Dnt = Dnt_(pmax, opt) * opn / pmax;
				Dnt = (Dnt_(pmax, opt) * opn / pmax + Dnt_(opn, opt))*0.5;
			}
			else if (!tsoft && nsoft)
			{
				// tangential unloading/reloading
				if (opn >= deln * rn) pmax = opn;
				Tt = Tt_(opn, tmax) * opt / tmax;
				Dtt = Tt_(opn, tmax) / tmax;
				Dnn = Dnn_(opn, tmax);
				//				Dnt = Dnt_(opn, tmax) * opt / tmax;
				Dnt = (Dnt_(opn, tmax) * opt / tmax + Dnt_(opn, opt)) * 0.5;
			}
			else
			{
				// reloading in both tangential and normal
				Tn = Tn_(pmax, opt) * opn / pmax;
				Tt = Tt_(opn, tmax) * opt / tmax;
				Dnn = Tn_(pmax, opt) / pmax;
				Dtt = Tt_(opn, tmax) / tmax;
				Dnt = (Dnt_(pmax, opt) * opn / pmax + Dnt_(opn, tmax) * opt / tmax)*0.5;
			}

		}
		else
		{
			cz_failed = true;
			Tn = Tt = Dnn = Dtt = Dnt = 0;
		}
	}
	Dtn = Dnt;
}

// (5)
__device__ void cohesive_law_modified(bool &cz_contact, bool &cz_failed, double &pmax, double &tmax, double opn, double opt,
	double &Tn, double &Tt, double &Dnn, double &Dtt, double &Dnt, double &Dtn) {
	Tn = Tt = Dnn = Dtt = Dnt = 0;
	if (opn > deln || opt > delt) {
		cz_failed = true; return;
	}
	cz_contact = (opn < 0);
	const double epsilon = -1e-9;

	if (cz_contact)
	{
		Dnt = 0;
		if (pmax != 0) {
			double peakTn = Tn_(pmax, tmax);
			Tn = peakTn * opn / pmax;
			Dnn = peakTn / pmax;
		}
		else {
			Dnn = Dnn_(0, tmax);
			Tn = Dnn * opn;
		}

		Tt = Tt_(0, opt);
		if (Tt >= epsilon)
		{
			if (opt >= tmax)
			{
				// tangential softening
				if (opt >= delt * rt || tmax != 0) tmax = opt;
				Dtt = Dtt_(0, opt);
			}
			else
			{
				// unload/reload
				double peakTt = Tt_(0, tmax);
				Tt = peakTt * opt / tmax;
				Dtt = peakTt / tmax;
			}

		}
		else
		{
			// cz failed in tangential direction while in contact
			Tt = Dtt = Dnt = 0;
			Tn = Dnn = 0;
			cz_failed = true;
		}
	}
	else
	{
		// not in contact
		Tt = Tt_(opn, opt);
		Tn = Tn_(opn, opt);
		if (Tt >= epsilon && Tn >= epsilon)
		{
			// tangential component
			bool tsoft = (opt >= tmax);
			bool nsoft = (opn >= pmax);
			if (tsoft && nsoft)
			{
				// tangential and normal softening
				if (opt >= delt * rt) tmax = opt;
				if (opn >= deln * rn) pmax = opn;
				Dnn = Dnn_(opn, opt);
				Dnt = Dnt_(opn, opt);
				Dtt = Dtt_(opn, opt);
			}
			/*
			else {
				if (opn > pmax) pmax = opn;
				if (opt > tmax) tmax = opt;


				double peakTn = Tn_(pmax, tmax);
				double peakTt = Tt_(pmax, tmax);

				if (pmax != 0) {
					Tn = peakTn * opn / pmax;
					Dnn = peakTn / pmax;
				}
				else {
					Tn = 0; Dnn = Dnn_(0, tmax);
				}

				if (tmax != 0) {
					Tt = peakTt * opt / tmax;
					Dtt = peakTt / tmax;
				}
				else {
					Tt = 0; Dtt = Dtt_(pmax, 0);
				}

				Dnt = 0;


			}
			*/

			else if (tsoft && !nsoft) {
				Dnt = 0;
				if (pmax != 0) {
					double peakTn = Tn_(pmax, tmax);
					Tn = peakTn * opn / pmax;
					Dnn = peakTn / pmax;
				}
				else {
					Tn = 0; Dnn = Dnn_(0, tmax);
				}

				// normal unload/reload
				tmax = opt;
				Tt = Tt_(pmax, opt);
				Dtt = Dtt_(pmax, opt);
			}
			else if (!tsoft && nsoft)
			{
				Dnt = 0;
				if (tmax != 0) {
					double peakTt = Tt_(pmax, tmax);
					Tt = peakTt * opt / tmax;
					Dtt = peakTt / tmax;
				}
				else {
					Tt = 0; Dtt = Dtt_(pmax, 0);
				}

				pmax = opn;
				Tn = Tn_(pmax, tmax);
				Dnn = Dnn_(pmax, tmax);

			}
			else
			{
				Dnt = 0;
				// reloading in both tangential and normal
				double peakTn = Tn_(pmax, tmax);
				if (pmax != 0)
				{
					Tn = peakTn * opn / pmax;
					Dnn = peakTn / pmax;
				}
				else {
					Tn = 0; Dnn = Dnn_(0, tmax);
				}

				if (tmax != 0) {
					double peakTt = Tt_(pmax, tmax);
					Tt = peakTt * opt / tmax;
					Dtt = peakTt / tmax;
				}
				else {
					Tt = 0; Dtt = Dtt_(pmax, 0);
				}
			}

		}
		else
		{
			cz_failed = true;
			Tn = Tt = Dnn = Dtt = Dnt = 0;
		}
	}
	Dtn = Dnt;
}


// assembling routines
__device__ void AssembleBCSR18(const int *pcsr, double *A, double *b, const int stride,
	double(&LHS)[18][18], double(&rhs)[18]) {

	// BCSR format
	// distribute LHS into global matrix, and rhs into global rhs
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			int idx1 = pcsr[stride * (i * 6 + j)];
			if (idx1 >= 0)
			{
				// write into csr.vals
				for (int k = 0; k < 3; k++)
					for (int l = 0; l < 3; l++)
						atomicAdd2(&A[idx1 * 9 + 3 * k + l], LHS[i * 3 + k][j * 3 + l]);
			}
		}
		// distribute rhs
		int idx2 = pcsr[stride * (36 + i)];
		if (idx2 >= 0)
		{
			for (int k = 0; k < 3; k++)
				atomicAdd2(&b[idx2 * 3 + k], rhs[i * 3 + k]);
		}
	}
}

__device__ void Assemble_CSR_Nonsymmetric_18(const int *pcsr, double *A, double *b, const int stride,
	double(&LHS)[18][18], double(&rhs)[18]) {

	// CSR format
	for (int i = 0; i < 6; i++)
	{
		int idx2 = pcsr[stride * (36 + i)];
		int rowSize = pcsr[stride * (42 + i)];

		// distribute rhs
		if (idx2 >= 0)
			for (int k = 0; k < 3; k++) atomicAdd2(&b[idx2 * 3 + k], rhs[i * 3 + k]);

		// distribute K
		for (int j = 0; j < 6; j++)
		{
			int idx1 = pcsr[stride * (i * 6 + j)];
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

// dcz: double cz data
// icz: integer cz data (includes pcsr)
// dnd: double nodal data
// h: timestep
extern "C" __global__ void kczCZForce(
	double *dcz, int *icz,
	double *dnd,
	double *_global_matrix, double *_global_rhs,
	const double h,
	const int nCZs, const int cz_stride, const int nd_stride, const int formulation, const int assembly_type) 
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nCZs) return;
	if (icz[idx + cz_stride*(CURRENT_FAILED_OFFSET_CZ)] != 0) return; // cz failed

	double x0[18], un[18];
	double xc[18], xr[18]; // xc = x0 + un; xr = R xc;  *in this case R is the inverse rotation
	double pmax[3], tmax[3];

	// retrieve nodal data from the global memory
	for (int i = 0; i < 6; i++) {
		int vrtx = icz[idx + cz_stride * (VRTS_OFFSET_CZ + i)];
		for (int j = 0; j < 3; j++) {
			int idx1 = i * 3 + j;
			x0[idx1] = dnd[vrtx + nd_stride * (X0_OFFSET + j)];
			un[idx1] = dnd[vrtx + nd_stride * (UN_OFFSET + j)];
			xc[idx1] = x0[idx1] + un[idx1];
		}
	}

	// retrieve the cz damage state
	for (int i = 0; i < 3; i++) {
		pmax[i] = dcz[idx + cz_stride*(CURRENT_PMAX_OFFSET_CZ + i)];
		tmax[i] = dcz[idx + cz_stride*(CURRENT_TMAX_OFFSET_CZ + i)];
	}

	// find the midplane 
	double mpc[9]; // midplane coordinates
	for (int i = 0; i < 9; i++) mpc[i] = (xc[i] + xc[i + 9])*0.5;

	// find the rotation of the midplane
	double R[3][3];
	double a_Jacob;
	CZRotationMatrix(
		mpc[0], mpc[1], mpc[2],
		mpc[3], mpc[4], mpc[5],
		mpc[6], mpc[7], mpc[8],
		R[0][0], R[0][1], R[0][2],
		R[1][0], R[1][1], R[1][2],
		R[2][0], R[2][1], R[2][2],
		a_Jacob);

	// compute the coordinates xr in the local system
	for (int i = 0; i < 6; i++)
		multAX(
			R[0][0], R[0][1], R[0][2],
			R[1][0], R[1][1], R[1][2],
			R[2][0], R[2][1], R[2][2],
			xc[i * 3 + 0], xc[i * 3 + 1], xc[i * 3 + 2],
			xr[i * 3 + 0], xr[i * 3 + 1], xr[i * 3 + 2]
			);

	// total over all gauss points
	double Keff[18][18] = {};
	double rhs[18] = {};

	bool cz_contact_gp[3] = {};
	bool cz_failed_gp[3] = {};

	// loop over 3 Gauss points
	for (int gpt = 0; gpt < 3; gpt++)
	{
		// shear and normal local opening displacements
		double dt1, dt2, dn;
		dt1 = dt2 = dn = 0;
		for (int i = 0; i < 3; i++)
		{
			dt1 += (xr[i * 3 + 0] - xr[i * 3 + 9]) * sf[i][gpt];
			dt2 += (xr[i * 3 + 1] - xr[i * 3 + 10]) * sf[i][gpt];
			dn += (xr[i * 3 + 2] - xr[i * 3 + 11]) * sf[i][gpt];
		}
		double opn = dn;
		double opt = sqrt(dt1 * dt1 + dt2 * dt2);

		double Tn, Tt, Dnn, Dtt, Dnt, Dtn;
		if (formulation == 0) 
			cohesive_law(cz_contact_gp[gpt], cz_failed_gp[gpt], pmax[gpt], tmax[gpt], opn, opt, Tn, Tt, Dnn, Dtt, Dnt, Dtn);
		else if (formulation == 5)
			cohesive_law_modified(cz_contact_gp[gpt], cz_failed_gp[gpt], pmax[gpt], tmax[gpt], opn, opt, Tn, Tt, Dnn, Dtt, Dnt, Dtn);


		double T[3] = {};
		double T_d[3][3] = {};

		if (opt < 1e-20)
		{
			T[2] = Tn;
			T_d[0][0] = Dtt;
			T_d[1][1] = Dtt;
			T_d[2][2] = Dnn;

			T_d[1][0] = T_d[0][1] = 0;

			//T_d[2][0] = Dnt;
			//T_d[0][2] = Dtn;
			//T_d[2][1] = Dnt;
			//T_d[1][2] = Dtn;

			T_d[2][0] = Dtn;
			T_d[0][2] = Dnt;
			T_d[2][1] = Dtn;
			T_d[1][2] = Dnt;
		}
		else
		{
			T[0] = Tt * dt1 / opt;
			T[1] = Tt * dt2 / opt;
			T[2] = Tn;

			double opt_sq = opt * opt;
			double opt_cu = opt_sq * opt;
			double delu00 = dt1 * dt1;
			double delu10 = dt2 * dt1;
			double delu11 = dt2 * dt2;

			T_d[0][0] = Dtt * delu00 / opt_sq + Tt * delu11 / opt_cu;
			T_d[1][1] = Dtt * delu11 / opt_sq + Tt * delu00 / opt_cu;
			T_d[2][2] = Dnn;

			T_d[1][0] = T_d[0][1] = Dtt * delu10 / opt_sq - Tt * delu10 / opt_cu;

			//T_d[2][0] = Dnt * dt1 / opt;
			//T_d[0][2] = Dtn * dt1 / opt;
			//T_d[2][1] = Dnt * dt2 / opt;
			//T_d[1][2] = Dtn * dt2 / opt;

			T_d[2][0] = Dtn * dt1 / opt;
			T_d[0][2] = Dnt * dt1 / opt;
			T_d[2][1] = Dtn * dt2 / opt;
			T_d[1][2] = Dnt * dt2 / opt;
		}

		// RHS
		// BtT = Bt x T x (-GP_W)
		const double GP_W = 1.0 / 3.0; // Gauss point weight
		double BtT[18] = {};
		for (int i = 0; i < 18; i++) {
			for (int j = 0; j < 3; j++) {
				BtT[i] += B[gpt][j][i] * T[j];
			}
			BtT[i] *= -(GP_W*a_Jacob);
		}

		// rotate BtT
		double rhs_gp[18] = {};
		for (int i = 0; i < 6; i++) {
			multAX(R[0][0], R[1][0], R[2][0],
				R[0][1], R[1][1], R[2][1],
				R[0][2], R[1][2], R[2][2],
				BtT[i * 3 + 0], BtT[i * 3 + 1], BtT[i * 3 + 2],
				rhs_gp[i * 3 + 0], rhs_gp[i * 3 + 1], rhs_gp[i * 3 + 2]);
		}

		// add to rhs
		for (int i = 0; i < 18; i++) rhs[i] += rhs_gp[i];

		// STIFFNESS MATRIX
		// compute Bt x T_d x GP_W
		double BtTd[18][3] = {};
		for (int row = 0; row < 18; row++)
			for (int col = 0; col < 3; col++) {
				for (int k = 0; k < 3; k++) BtTd[row][col] += B[gpt][k][row] * T_d[k][col];
				BtTd[row][col] *= (GP_W*a_Jacob);
			}

		// BtTdB = BtTd x B
		double BtTdB[18][18] = {};
		for (int row = 0; row < 18; row++)
			for (int col = 0; col < 18; col++)
				for (int k = 0; k < 3; k++)
					BtTdB[row][col] += BtTd[row][k] * B[gpt][k][col];

		double TrMtBtTdB[18][18] = {};

		// Keff
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
			{
				// TrMtBtTdB = TrMt x BtTdB
				for (int k = 0; k < 3; k++)
					for (int l = 0; l < 3; l++) {
						for (int m = 0; m < 3; m++)
							TrMtBtTdB[3 * i + k][3 * j + l] += R[m][k] * BtTdB[3 * i + m][3 * j + l];
					}

				// Keff = TrMt x BtTdB x TrM
				for (int k = 0; k < 3; k++)
					for (int l = 0; l < 3; l++) {
						for (int m = 0; m < 3; m++)
							Keff[3 * i + k][3 * j + l] += TrMtBtTdB[3 * i + k][3 * j + m] * R[m][l];
					}
			}
	}

	// account for damping
	//	rhs -= Keff vn dampingStiffness

	//for (int row = 0; row < 18; row++)
	//	for (int col = 0; col < 18; col++)
	//		rhs[row] -= Keff[row][col] * vn[col] * dampingStiffness;

	//double Dcoeff = 1.0 +dampingStiffness * NewmarkGamma / (h * NewmarkBeta);
	//for (int row = 0; row < 18; row++)
	//	for (int col = 0; col < 18; col++)
	//		Keff[row][col] *= Dcoeff;

	// copy tentative values to global memory
	int damaged = 0;
	// ! EXPERIMENTAL CODE !
	double pmax_ = max(max(pmax[0], pmax[1]), pmax[2]);
	double tmax_ = max(max(tmax[0], tmax[1]), tmax[2]);

	for (int i = 0; i < 3; i++) {
		//		dcz[idx + cz_stride*(TENTATIVE_PMAX_OFFSET_CZ + i)] = pmax[i];
		//		dcz[idx + cz_stride*(TENTATIVE_TMAX_OFFSET_CZ + i)] = tmax[i];
		dcz[idx + cz_stride*(TENTATIVE_PMAX_OFFSET_CZ + i)] = pmax_;
		dcz[idx + cz_stride*(TENTATIVE_TMAX_OFFSET_CZ + i)] = tmax_;
		if (pmax[i] > 0 || tmax[i] > 0) damaged = 1;
	}
	// finally, cz_contact and _cz_failed
	bool cz_failed = cz_failed_gp[0] || cz_failed_gp[1] || cz_failed_gp[2];
	bool cz_contact = cz_contact_gp[0] || cz_contact_gp[1] || cz_contact_gp[2];
	icz[idx + cz_stride*TENTATIVE_CONTACT_OFFSET_CZ] = cz_contact ? 1 : 0;
	icz[idx + cz_stride*TENTATIVE_FAILED_OFFSET_CZ] = cz_failed ? 1 : 0;
	icz[idx + cz_stride*TENTATIVE_DAMAGED_OFFSET_CZ] = cz_failed ? 0 : damaged;

	if (cz_failed) return;

	// distribute K and rhs into CSR

	if (assembly_type == 0)  // BCSR format
		AssembleBCSR18(&icz[idx + cz_stride*PCSR_OFFSET_CZ], _global_matrix, _global_rhs, cz_stride, Keff, rhs);
	else if (assembly_type == 1) // CSR format
		Assemble_CSR_Nonsymmetric_18(&icz[idx + cz_stride*PCSR_OFFSET_CZ], _global_matrix, _global_rhs, cz_stride, Keff, rhs);


	/*
	// distribute computed forces to fx fy fz for visualization
	for (int i = 0; i < 6; i++) {			// node
		int vrtx = icz[idx + cz_stride * (VRTS_OFFSET_CZ + i)];
		for (int j = 0; j < 3; j++)		// coordinate
			atomicAdd2(&dnd[vrtx + nd_stride*(F_OFFSET + j)], rhs[i * 3 + j]);
	}
	*/
}
