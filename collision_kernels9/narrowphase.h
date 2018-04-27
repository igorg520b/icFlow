#define EPS 1E-10

__device__ void Bvalues(double x0, double y0, double z0,
	double x1, double y1, double z1,
	double x2, double y2, double z2,
	double x3, double y3, double z3,
	double &b11, double &b12, double &b13,
	double &b21, double &b22, double &b23,
	double &b31, double &b32, double &b33)
{
	double a11, a12, a13, a21, a22, a23, a31, a32, a33;
	a11 = x1 - x0;
	a12 = x2 - x0;
	a13 = x3 - x0;
	a21 = y1 - y0;
	a22 = y2 - y0;
	a23 = y3 - y0;
	a31 = z1 - z0;
	a32 = z2 - z0;
	a33 = z3 - z0;

	// inverse
	double det = a31 * (-a13 * a22 + a12 * a23) + a32 * (a13 * a21 - a11 * a23) + a33 * (-a12 * a21 + a11 * a22);
	b11 = (-a23 * a32 + a22 * a33) / det;
	b12 = (a13 * a32 - a12 * a33) / det;
	b13 = (-a13 * a22 + a12 * a23) / det;
	b21 = (a23 * a31 - a21 * a33) / det;
	b22 = (-a13 * a31 + a11 * a33) / det;
	b23 = (a13 * a21 - a11 * a23) / det;
	b31 = (-a22 * a31 + a21 * a32) / det;
	b32 = (a12 * a31 - a11 * a32) / det;
	b33 = (-a12 * a21 + a11 * a22) / det;
}

__device__ bool ctest(double b11, double b12, double b13,
	double b21, double b22, double b23,
	double b31, double b32, double b33,
	double x, double y, double z) {

	double y1 = x * b11 + y * b12 + z * b13;
	double y2 = x * b21 + y * b22 + z * b23;
	double y3 = x * b31 + y * b32 + z * b33;
	return (y1 > EPS && y2 > EPS && y3 > EPS && (y1 + y2 + y3) < (1 - EPS));
}

extern "C" __global__ void kNarrowPhase_new(int nPairs,
	const double* dn, const int *ie, int *narrowList,
	const int el_all_stride, const int nd_stride) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPairs) return;

	// fetch nodal coordinates
	double nds[24];
	int nd_idxs[8];
	for (int elem = 0; elem < 2; elem++) {
		int elemGlobalId = narrowList[idx*2 + elem];
		for (int i = 0; i < 4; i++) {
			int nd_idx = ie[elemGlobalId + el_all_stride * i];
			nd_idxs[i + elem * 4] = nd_idx;
			for (int j = 0; j < 3; j++)
				nds[j + i * 3 + elem * 12] = dn[nd_idx + nd_stride * (j + X_CURRENT_OFFSET)];
		}
	}

	// verify that elements are non-adjacent
	narrowList[idx * 2] = 0;
	for (int i = 0; i < 4; i++)
		for (int j = 4; j < 8;j++)
	if (nd_idxs[i] == nd_idxs[j]) return;

	// b-values for elements
	double bv0[9], bv1[9];

	Bvalues(nds[0], nds[1], nds[2],
		nds[3], nds[4], nds[5],
		nds[6], nds[7], nds[8],
		nds[9], nds[10], nds[11],
		bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8]);

	Bvalues(nds[12], nds[13], nds[14],
		nds[15], nds[16], nds[17],
		nds[18], nds[19], nds[20],
		nds[21], nds[22], nds[23],
		bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8]);

	int result = 0;
	// perform tests
	bool bres;
	double x0 = nds[0];
	double y0 = nds[1];
	double z0 = nds[2];

	// test if 1 element nodes are inside element 0

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[12] - x0, nds[13] - y0, nds[14] - z0);
	if (bres) result |= (1);
	
	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[15] - x0, nds[16] - y0, nds[17] - z0);
	if (bres) result |= (2);

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[18] - x0, nds[19] - y0, nds[20] - z0);
	if (bres) result |= (4);

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[21] - x0, nds[22] - y0, nds[23] - z0);
	if (bres) result |= (8);

	// test if 0 element nodes are inside element 1
	x0 = nds[12];
	y0 = nds[13];
	z0 = nds[14];

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[0] - x0, nds[1] - y0, nds[2] - z0);
	if (bres) result |= (16);

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[3] - x0, nds[4] - y0, nds[5] - z0);
	if (bres) result |= (32);

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[6] - x0, nds[7] - y0, nds[8] - z0);
	if (bres) result |= (64);

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[9] - x0, nds[10] - y0, nds[11] - z0);
	if (bres) result |= (128);
	
	// write result
	narrowList[idx * 2] = result;
}

// input: node-element tuples
// output: node-face tuples or -1
extern "C" __global__ void kFindClosestFace(int nPairs, int tet_stride, int *narrowList2, 
	const double* dn, const int *ie,
	const int *faces,
	const int nd_stride, const int el_all_stride, const int fc_stride) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPairs) return;

	int nodeId = narrowList2[idx];
	int elemId = narrowList2[idx+tet_stride];
	double nodeCoords[3];
	for (int i = 0; i < 3; i++) nodeCoords[i] = dn[nodeId + nd_stride * (X_CURRENT_OFFSET + i)];

	int nFaces = ie[elemId + el_all_stride * 4];
	int closestFace = -1;
	double closestDistance;
	for (int idx_face = 0; idx_face < nFaces; idx_face++) {
		int faceId = ie[elemId + el_all_stride * (5 + idx_face)];
		double faceCoords[9];
		for (int nd = 0; nd < 3; nd++) {
			int faceNdId = faces[faceId + fc_stride * nd];
			for (int idx_coord = 0; idx_coord < 3; idx_coord++) {
				faceCoords[idx_coord + nd * 3] = dn[faceNdId + nd_stride * (X_CURRENT_OFFSET + idx_coord)];
			}
		}

		double distance = dtn(faceCoords[0], faceCoords[1], faceCoords[2],
			faceCoords[3], faceCoords[4], faceCoords[5],
			faceCoords[6], faceCoords[7], faceCoords[8],
			nodeCoords[0], nodeCoords[1], nodeCoords[2]);

		if (idx_face == 0 || distance < closestDistance) {
			closestFace = faceId;
			closestDistance = distance;
		}
	}
//	if (closestDistance == 0) closestFace = -1;

	// write closest face vertices into results
	narrowList2[idx + tet_stride] = faces[closestFace + fc_stride * 0];
	narrowList2[idx  + tet_stride*2] = faces[closestFace + fc_stride * 1];
	narrowList2[idx  + tet_stride*3] = faces[closestFace + fc_stride * 2];
}


// temporary / testing
extern "C" __global__ void kNarrowPhase_testing1(int stride, int nPairs, const double *tCoord, int *result) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPairs) return;

	double nds[24];
	// extract individual coords
	for (int i = 0; i < 24;i++) nds[i] = tCoord[idx + i * stride];

	// b-values for elements
	double bv0[9], bv1[9];

	Bvalues(nds[0], nds[1], nds[2],
		nds[3], nds[4], nds[5],
		nds[6], nds[7], nds[8],
		nds[9], nds[10], nds[11],
		bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8]);

	Bvalues(nds[12], nds[13], nds[14],
		nds[15], nds[16], nds[17],
		nds[18], nds[19], nds[20],
		nds[21], nds[22], nds[23],
		bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8]);

	// perform tests
	bool bres;
	double x0 = nds[0];
	double y0 = nds[1];
	double z0 = nds[2];

	// test if 1 element nodes are inside element 0

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[12] - x0, nds[13] - y0, nds[14] - z0);
	if (bres) { result[idx] = 1;  return; }

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[15] - x0, nds[16] - y0, nds[17] - z0);
	if (bres) { result[idx] = 1;  return; }

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[18] - x0, nds[19] - y0, nds[20] - z0);
	if (bres) { result[idx] = 1;  return; }

	bres = ctest(bv0[0], bv0[1], bv0[2],
		bv0[3], bv0[4], bv0[5],
		bv0[6], bv0[7], bv0[8],
		nds[21] - x0, nds[22] - y0, nds[23] - z0);
	if (bres) { result[idx] = 1;  return; }

	// test if 0 element nodes are inside element 1
	x0 = nds[12];
	y0 = nds[13];
	z0 = nds[14];

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[0] - x0, nds[1] - y0, nds[2] - z0);
	if (bres) { result[idx] = 1;  return; }

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[3] - x0, nds[4] - y0, nds[5] - z0);
	if (bres) { result[idx] = 1;  return; }

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[6] - x0, nds[7] - y0, nds[8] - z0);
	if (bres) { result[idx] = 1;  return; }

	bres = ctest(bv1[0], bv1[1], bv1[2],
		bv1[3], bv1[4], bv1[5],
		bv1[6], bv1[7], bv1[8],
		nds[9] - x0, nds[10] - y0, nds[11] - z0);
	if (bres) { result[idx] = 1;  return; }
}


