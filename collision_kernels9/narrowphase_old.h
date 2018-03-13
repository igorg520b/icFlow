#include "triangle_overlap.h"

extern "C" __global__ void kNarrowPhase_old(int stride, int nPairs, const double *tCoord, int *result) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPairs) return;

	// extract individual coords
	double t11x = tCoord[idx];
	double t11y = tCoord[idx + stride];
	double t11z = tCoord[idx + 2 * stride];
	double t12x = tCoord[idx + 3 * stride];
	double t12y = tCoord[idx + 4 * stride];
	double t12z = tCoord[idx + 5 * stride];
	double t13x = tCoord[idx + 6 * stride];
	double t13y = tCoord[idx + 7 * stride];
	double t13z = tCoord[idx + 8 * stride];
	double t14x = tCoord[idx + 9 * stride];
	double t14y = tCoord[idx + 10 * stride];
	double t14z = tCoord[idx + 11 * stride];

	double t21x = tCoord[idx + 12 * stride];
	double t21y = tCoord[idx + 13 * stride];
	double t21z = tCoord[idx + 14 * stride];
	double t22x = tCoord[idx + 15 * stride];
	double t22y = tCoord[idx + 16 * stride];
	double t22z = tCoord[idx + 17 * stride];
	double t23x = tCoord[idx + 18 * stride];
	double t23y = tCoord[idx + 19 * stride];
	double t23z = tCoord[idx + 20 * stride];
	double t24x = tCoord[idx + 21 * stride];
	double t24y = tCoord[idx + 22 * stride];
	double t24z = tCoord[idx + 23 * stride];

	// run tests
	double p1[3], q1[3], r1[3], p2[3], q2[3], r2[3];

	//1
	p1[0] = t11x; p1[1] = t11y; p1[2] = t11z;
	q1[0] = t12x; q1[1] = t12y; q1[2] = t12z;
	r1[0] = t13x; r1[1] = t13y; r1[2] = t13z;

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t24x; r2[1] = t24y; r2[2] = t24z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t24x; p2[1] = t24y; p2[2] = t24z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t24x; q2[1] = t24y; q2[2] = t24z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	// 2
	p1[0] = t14x; p1[1] = t14y; p1[2] = t14z;
	q1[0] = t12x; q1[1] = t12y; q1[2] = t12z;
	r1[0] = t13x; r1[1] = t13y; r1[2] = t13z;

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t24x; r2[1] = t24y; r2[2] = t24z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t24x; p2[1] = t24y; p2[2] = t24z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t24x; q2[1] = t24y; q2[2] = t24z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	// 3
	p1[0] = t11x; p1[1] = t11y; p1[2] = t11z;
	q1[0] = t14x; q1[1] = t14y; q1[2] = t14z;
	r1[0] = t13x; r1[1] = t13y; r1[2] = t13z;

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t24x; r2[1] = t24y; r2[2] = t24z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t24x; p2[1] = t24y; p2[2] = t24z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t24x; q2[1] = t24y; q2[2] = t24z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	//4
	p1[0] = t11x; p1[1] = t11y; p1[2] = t11z;
	q1[0] = t12x; q1[1] = t12y; q1[2] = t12z;
	r1[0] = t14x; r1[1] = t14y; r1[2] = t14z;

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t24x; r2[1] = t24y; r2[2] = t24z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t24x; p2[1] = t24y; p2[2] = t24z;
	q2[0] = t22x; q2[1] = t22y; q2[2] = t22z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

	p2[0] = t21x; p2[1] = t21y; p2[2] = t21z;
	q2[0] = t24x; q2[1] = t24y; q2[2] = t24z;
	r2[0] = t23x; r2[1] = t23y; r2[2] = t23z;
	if (tri_tri_overlap_test_3d(p1, q1, r1, p2, q2, r2) != 0) { result[idx] = 1;  return; }

}