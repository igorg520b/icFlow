// second derivatives of a00
__constant__ double a00d2[12][12] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0 },
	{ 0, 0, 0, -2, 0, 0, 2, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, -2, 0, 0, 2, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, -2, 0, 0, 2, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

// second derivatives of a01
__constant__ double a01d2[12][12] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 2, 0, 0, -1, 0, 0, -1, 0, 0 },
	{ 0, 0, 0, 0, 2, 0, 0, -1, 0, 0, -1, 0 },
	{ 0, 0, 0, 0, 0, 2, 0, 0, -1, 0, 0, -1 },
	{ 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0 },
	{ 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0 },
	{ 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1 },
	{ 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0 } };

// second derivatives of a11
__constant__ double a11d2[12][12] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 2, 0, 0, 0, 0, 0, -2, 0, 0 },
	{ 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, -2, 0 },
	{ 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, -2 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, -2, 0, 0, 0, 0, 0, 2, 0, 0 },
	{ 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 2, 0 },
	{ 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 2 } };

// second derivatives of b0
__constant__ double b0d2[12][12] = {
	{ 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0 },
	{ 1, 0, 0, -2, 0, 0, 1, 0, 0, 0, 0, 0 },
	{ 0, 1, 0, 0, -2, 0, 0, 1, 0, 0, 0, 0 },
	{ 0, 0, 1, 0, 0, -2, 0, 0, 1, 0, 0, 0 },
	{ -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

// second derivatives of b1
__constant__ double b1d2[12][12] = {
	{ 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0 },
	{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0 },
	{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1 },
	{ 1, 0, 0, -2, 0, 0, 0, 0, 0, 1, 0, 0 },
	{ 0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 1, 0 },
	{ 0, 0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 1 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0 } };

// second derivatives of c
__constant__ double cd2[12][12] = {
	{ 2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 0, 0 },
	{ -2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, -2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, -2, 0, 0, 2, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };





__device__ void Case1(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// det = a00 * a11 - a01 * a01;
	double detsq = det * det;
	double detcu = detsq * det;
	// first derivatives of det
	for (int i = 0; i < 12; i++) detd1[i] = detSign * (a00d1[i] * a11 + a11d1[i] * a00 - 2 * a01d1[i] * a01);

	// second derivatives of det
	double detd2[12][12];
	for (int i = 0; i < 12; i++)
		for (int j = i; j < 12; j++)
			detd2[j][i] = detd2[i][j] = detSign * (a00d1[i] * a11d1[j] + a00d1[j] * a11d1[i] +
				a00d2[i][j] * a11 + a00 * a11d2[i][j] - 2 * a01d1[j] * a01d1[i] - 2 * a01 * a01d2[i][j]);

	//            double s = (a01 * b1 - a11 * b0)/det;
	//            double t = (a01 * b0 - a00 * b1)/det;

	// first derivatives of barycentric
	for (int i = 0; i < 12; i++)
	{
		wd[1][i] = (-(a11d1[i] * b0) - a11 * b0d1[i] + a01d1[i] * b1 + a01 * b1d1[i]) / det -
			((-(a11 * b0) + a01 * b1) * detd1[i]) / (detsq);
		wd[2][i] = (a01d1[i] * b0 + a01 * b0d1[i] - a00d1[i] * b1 - a00 * b1d1[i]) / det -
			((a01 * b0 - a00 * b1) * detd1[i]) / (detsq);
		wd[0][i] = -(wd[1][i] + wd[2][i]);

		sqrdistd[i] = s * s * a00d1[i] + 2 * s * t * a01d1[i] + t * t * a11d1[i] +
			2 * s * b0d1[i] + 2 * t * b1d1[i] + cd1[i] + 2 * b0 * wd[1][i] + 2 * a00 * s * wd[1][i] +
			2 * a01 * t * wd[1][i] + 2 * b1 * wd[2][i] + 2 * a01 * s * wd[2][i] + 2 * a11 * t * wd[2][i];
	}

	// second derivatives of barycentric
	for (int i = 0; i < 12; i++)
		for (int j = i; j < 12; j++)
		{
			wdd[1][i][j] = wdd[1][j][i] = -(((b1 * a01d1[j] - b0 * a11d1[j] - a11 * b0d1[j] +
				a01 * b1d1[j]) * detd1[i]) / detsq) - ((b1 * a01d1[i] - b0 * a11d1[i] -
					a11 * b0d1[i] + a01 * b1d1[i]) * detd1[j]) / detsq + (2 * (-(a11 * b0) +
						a01 * b1) * detd1[i] * detd1[j]) / detcu + (-(a11d1[j] * b0d1[i]) -
							a11d1[i] * b0d1[j] + a01d1[j] * b1d1[i] + a01d1[i] * b1d1[j] +
							b1 * a01d2[i][j] - b0 * a11d2[i][j] - a11 * b0d2[i][j] +
							a01 * b1d2[i][j]) / det - ((-(a11 * b0) + a01 * b1) * detd2[i][j]) / detsq;

			wdd[2][i][j] = wdd[2][j][i] = -(((-(b1 * a00d1[j]) + b0 * a01d1[j] + a01 * b0d1[j] -
				a00 * b1d1[j]) * detd1[i]) / detsq) - ((-(b1 * a00d1[i]) + b0 * a01d1[i] +
					a01 * b0d1[i] - a00 * b1d1[i]) * detd1[j]) / detsq + (2 * (a01 * b0 -
						a00 * b1) * detd1[i] * detd1[j]) / detcu + (a01d1[j] * b0d1[i] +
							a01d1[i] * b0d1[j] - a00d1[j] * b1d1[i] - a00d1[i] * b1d1[j] -
							b1 * a00d2[i][j] + b0 * a01d2[i][j] + a01 * b0d2[i][j] -
							a00 * b1d2[i][j]) / det - ((a01 * b0 - a00 * b1) * detd2[i][j]) / detsq;

			wdd[0][i][j] = wdd[0][j][i] = -(wdd[1][i][j] + wdd[2][i][j]);

			sqrdistdd[i][j] = sqrdistdd[j][i] = s * s * a00d2[i][j] + t * t * a11d2[i][j] + cd2[i][j] +
				2 * b0d1[j] * wd[1][i] + 2 * b0d1[i] * wd[1][j] + 2 * a00 * wd[1][i] * wd[1][j] +
				2 * b1d1[j] * wd[2][i] + 2 * a01 * wd[1][j] * wd[2][i] + 2 * b1d1[i] * wd[2][j] +
				2 * a01 * wd[1][i] * wd[2][j] + 2 * a11 * wd[2][i] * wd[2][j] + 2 * b0 * wdd[1][i][j] +
				2 * b1 * wdd[2][i][j] + 2 * s * (t * a01d2[i][j] + b0d2[i][j] +
					a00d1[j] * wd[1][i] + a00d1[i] * wd[1][j] + a01d1[j] * wd[2][i] +
					a01d1[i] * wd[2][j] + a00 * wdd[1][i][j] + a01 * wdd[2][i][j]) +
				2 * t * (b1d2[i][j] + a01d1[j] * wd[1][i] + a01d1[i] * wd[1][j] +
					a11d1[j] * wd[2][i] + a11d1[i] * wd[2][j] + a01 * wdd[1][i][j] +
					a11 * wdd[2][i][j]);
		}
}


__device__ void Case_v10(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// s=1; t=0
	// sqrDistance = a00 + (2) * b0 + c;
	for (int i = 0; i < 12; i++)
	{
		sqrdistd[i] = a00d1[i] + 2 * b0d1[i] + cd1[i];
		for (int j = i; j < 12; j++)
		{
			sqrdistdd[i][j] = sqrdistdd[j][i] = a00d2[i][j] + 2 * b0d2[i][j] + cd2[i][j];
		}
	}
}

__device__ void Case_v01(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// s=0; t=1
	// sqrDistance = a1 + (2) * b1 + c;
	for (int i = 0; i < 12; i++)
	{
		sqrdistd[i] = a11d1[i] + 2 * b1d1[i] + cd1[i];
		for (int j = i; j < 12; j++)
		{
			sqrdistdd[i][j] = sqrdistdd[j][i] = a11d2[i][j] + 2 * b1d2[i][j] + cd2[i][j];
		}
	}
}

__device__ void Case_v00(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// sqrDistance = c;
	for (int i = 0; i < 12; i++)
	{
		sqrdistd[i] = cd1[i];
		for (int j = i; j < 12; j++)
		{
			sqrdistdd[i][j] = sqrdistdd[j][i] = cd2[i][j];
		}
	}
}

__device__ void Case_e_s0(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{

	//            t = -b1 / a11;
	//            sqrDistance = b1 * t + c;
	double a11sq = a11 * a11;
	double a11cu = a11sq * a11;
	for (int i = 0; i < 12; i++)
	{
		wd[1][i] = 0;
		wd[2][i] = (b1 * a11d1[i]) / a11sq - b1d1[i] / a11;
		wd[0][i] = -(wd[1][i] + wd[2][i]);
		sqrdistd[i] = t * b1d1[i] + cd1[i] + b1 * wd[2][i];
		for (int j = i; j < 12; j++)
		{
			wdd[1][i][j] = wdd[1][j][i] = 0;
			wdd[2][i][j] = wdd[2][j][i] = (-2 * b1 * a11d1[i] * a11d1[j]) / a11cu +
				(a11d1[j] * b1d1[i]) / a11sq + (a11d1[i] * b1d1[j]) / a11sq +
				(b1 * a11d2[i][j]) / a11sq - b1d2[i][j] / a11;
			wdd[0][i][j] = wdd[0][j][i] = -(wdd[1][i][j] + wdd[2][i][j]);
			sqrdistdd[i][j] = sqrdistdd[j][i] = t * b1d2[i][j] + cd2[i][j] + b1d1[j] * wd[2][i] + b1d1[i] * wd[2][j] + b1 * wdd[2][i][j];
		}
	}
}

__device__ void Case_e_t0(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// s = -b0 / a00;
	// sqrDistance = b0 * s + c;
	double a00sq = a00 * a00;
	double a00cu = a00sq * a00;
	for (int i = 0; i < 12; i++)
	{
		wd[1][i] = (b0 * a00d1[i]) / a00sq - b0d1[i] / a00;
		wd[2][i] = 0;
		wd[0][i] = -(wd[1][i] + wd[2][i]);
		sqrdistd[i] = s * b0d1[i] + cd1[i] + b0 * wd[1][i];
		for (int j = i; j < 12; j++)
		{
			wdd[1][i][j] = wdd[1][j][i] = (-2 * b0 * a00d1[i] * a00d1[j]) / a00cu +
				(a00d1[j] * b0d1[i]) / a00sq + (a00d1[i] * b0d1[j]) / a00sq +
				(b0 * a00d2[i][j]) / a00sq - b0d2[i][j] / a00;

			wdd[2][i][j] = wdd[2][j][i] = 0;
			wdd[0][i][j] = wdd[0][j][i] = -(wdd[1][i][j] + wdd[2][i][j]);
			sqrdistdd[i][j] = sqrdistdd[j][i] = s * b0d2[i][j] + cd2[i][j] + b0d1[j] * wd[1][i] + b0d1[i] * wd[1][j] + b0 * wdd[1][i][j];
		}
	}
}

__device__ void Case_e_u0_s(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// s = (a11+b1-a01-b0)/(a00-(2)*a01+a11)
	// t = 1-s

	double u = a00 - 2 * a01 + a11;
	double usq = u * u;
	double ucu = usq * u;
	for (int i = 0; i < 12; i++)
	{
		wd[1][i] = ((a01 - a11 + b0 - b1) * (a00d1[i] - 2 * a01d1[i] + a11d1[i]) +
			u * (-a01d1[i] + a11d1[i] - b0d1[i] + b1d1[i])) / usq;
		wd[2][i] = -wd[1][i];
		wd[0][i] = 0;
		sqrdistd[i] = s * s * a00d1[i] + 2 * s * t * a01d1[i] + t * t * a11d1[i] +
			2 * s * b0d1[i] + 2 * t * b1d1[i] + cd1[i] + 2 * b0 * wd[1][i] + 2 * a00 * s * wd[1][i] +
			2 * a01 * t * wd[1][i] + 2 * b1 * wd[2][i] + 2 * a01 * s * wd[2][i] + 2 * a11 * t * wd[2][i];
		for (int j = i; j < 12; j++)
		{
			wdd[1][i][j] = wdd[1][j][i] = (2 * (-a01 + a11 - b0 + b1) * (a00d1[i] - 2 * a01d1[i] +
				a11d1[i]) * (a00d1[j] - 2 * a01d1[j] + a11d1[j]) +
				u * (a00d1[j] - 2 * a01d1[j] + a11d1[j]) * (a01d1[i] - a11d1[i] +
					b0d1[i] - b1d1[i]) + u * (a00d1[i] - 2 * a01d1[i] +
						a11d1[i]) * (a01d1[j] - a11d1[j] + b0d1[j] - b1d1[j]) +
				u * (a01 - a11 + b0 - b1) * (a00d2[i][j] - 2 * a01d2[i][j] + a11d2[i][j]) +
				usq * (-a01d2[i][j] + a11d2[i][j] - b0d2[i][j] + b1d2[i][j])) / ucu;
			wdd[2][i][j] = wdd[2][j][i] = -wdd[1][i][j];
			wdd[0][i][j] = wdd[0][j][i] = 0;
			sqrdistdd[i][j] = sqrdistdd[j][i] = s * s * a00d2[i][j] + t * t * a11d2[i][j] + cd2[i][j] +
				2 * b0d1[j] * wd[1][i] + 2 * b0d1[i] * wd[1][j] + 2 * a00 * wd[1][i] * wd[1][j] +
				2 * b1d1[j] * wd[2][i] + 2 * a01 * wd[1][j] * wd[2][i] + 2 * b1d1[i] * wd[2][j] +
				2 * a01 * wd[1][i] * wd[2][j] + 2 * a11 * wd[2][i] * wd[2][j] + 2 * b0 * wdd[1][i][j] +
				2 * b1 * wdd[2][i][j] + 2 * s * (t * a01d2[i][j] + b0d2[i][j] +
					a00d1[j] * wd[1][i] + a00d1[i] * wd[1][j] + a01d1[j] * wd[2][i] +
					a01d1[i] * wd[2][j] + a00 * wdd[1][i][j] + a01 * wdd[2][i][j]) +
				2 * t * (b1d2[i][j] + a01d1[j] * wd[1][i] + a01d1[i] * wd[1][j] +
					a11d1[j] * wd[2][i] + a11d1[i] * wd[2][j] + a01 * wdd[1][i][j] +
					a11 * wdd[2][i][j]);
		}
	}
}

__device__ void Case_e_u0_t(double(&x)[13],
	double det, double detSign,
	double a00, double a01, double a11, double b0, double b1,
	double s, double t,
	double(&detd1)[12],
	double(&a00d1)[12],
	double(&a01d1)[12],
	double(&a11d1)[12],
	double(&b0d1)[12],
	double(&b1d1)[12],
	double(&cd1)[12],
	/* output arrays */
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	// t = (a00+b0-a01-b1)/(a00-(2)*a01+a11)
	// s = 1-t
	double u = a00 - 2 * a01 + a11;
	double usq = u * u;
	double ucu = usq * u;

	for (int i = 0; i < 12; i++)
	{
		wd[2][i] = (-((a00 - a01 + b0 - b1) * (a00d1[i] - 2 * a01d1[i] + a11d1[i])) +
			(u)* (a00d1[i] - a01d1[i] + b0d1[i] - b1d1[i])) / usq;
		wd[1][i] = -wd[2][i];
		wd[0][i] = 0;
		sqrdistd[i] = s * s * a00d1[i] + 2 * s * t * a01d1[i] + t * t * a11d1[i] +
			2 * s * b0d1[i] + 2 * t * b1d1[i] + cd1[i] + 2 * b0 * wd[1][i] + 2 * a00 * s * wd[1][i] +
			2 * a01 * t * wd[1][i] + 2 * b1 * wd[2][i] + 2 * a01 * s * wd[2][i] + 2 * a11 * t * wd[2][i];
		for (int j = i; j < 12; j++)
		{
			wdd[2][i][j] = -((-2 * (a00 - a01 + b0 - b1) * (a00d1[i] - 2 * a01d1[i] + a11d1[i]) * (a00d1[j] - 2 * a01d1[j] + a11d1[j]) +
				(u)* (a00d1[j] - 2 * a01d1[j] + a11d1[j]) * (a00d1[i] - a01d1[i] + b0d1[i] - b1d1[i]) +
				(u)* (a00d1[i] - 2 * a01d1[i] + a11d1[i]) * (a00d1[j] - a01d1[j] + b0d1[j] - b1d1[j]) +
				(u)* (a00 - a01 + b0 - b1) * (a00d2[i][j] - 2 * a01d2[i][j] + a11d2[i][j]) -
				usq * (a00d2[i][j] - a01d2[i][j] + b0d2[i][j] - b1d2[i][j])) / ucu);
			wdd[1][i][j] = wdd[1][j][i] = -wdd[2][i][j];
			wdd[0][i][j] = wdd[0][j][i] = 0;
			sqrdistdd[i][j] = sqrdistdd[j][i] = s * s * a00d2[i][j] + t * t * a11d2[i][j] + cd2[i][j] +
				2 * b0d1[j] * wd[1][i] + 2 * b0d1[i] * wd[1][j] + 2 * a00 * wd[1][i] * wd[1][j] +
				2 * b1d1[j] * wd[2][i] + 2 * a01 * wd[1][j] * wd[2][i] + 2 * b1d1[i] * wd[2][j] +
				2 * a01 * wd[1][i] * wd[2][j] + 2 * a11 * wd[2][i] * wd[2][j] + 2 * b0 * wdd[1][i][j] +
				2 * b1 * wdd[2][i][j] + 2 * s * (t * a01d2[i][j] + b0d2[i][j] +
					a00d1[j] * wd[1][i] + a00d1[i] * wd[1][j] + a01d1[j] * wd[2][i] +
					a01d1[i] * wd[2][j] + a00 * wdd[1][i][j] + a01 * wdd[2][i][j]) +
				2 * t * (b1d2[i][j] + a01d1[j] * wd[1][i] + a01d1[i] * wd[1][j] +
					a11d1[j] * wd[2][i] + a11d1[i] * wd[2][j] + a01 * wdd[1][i][j] +
					a11 * wdd[2][i][j]);
		}
	}
}


// point-triangle derivatives; returns squared (!) distance
__device__ double PT_Derivatives(
	double ptx, double pty, double ptz,
	double v0x, double v0y, double v0z,
	double v1x, double v1y, double v1z,
	double v2x, double v2y, double v2z,
	/* output */
	double(&w)[3],
	double(&wd)[3][12],
	double(&wdd)[3][12][12],
	double(&sqrdistd)[12],
	double(&sqrdistdd)[12][12])
{
	double detd1[12];
	// 1-based index (because expressions were generated in Mathematica)
	double x[13] = { 0, ptx, pty, ptz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z };

	//            diff = triangle.V0 - point;
	//            edge0 = triangle.V1 - triangle.V0;
	//            edge1 = triangle.V2 - triangle.V0;
	//            double a00 = edge0.LengthSquared;

	double a00 = (-x[4] + x[7]) * (-x[4] + x[7]) + (-x[5] + x[8]) * (-x[5] + x[8]) + (-x[6] + x[9]) * (-x[6] + x[9]);
	// first derivatives of a00
	double a00d1[12] = { 0, 0, 0, -2 * (-x[4] + x[7]), -2 * (-x[5] + x[8]), -2 * (-x[6] + x[9]), 2 * (-x[4] + x[7]), 2 * (-x[5] + x[8]), 2 * (-x[6] + x[9]), 0, 0, 0 };

	//            double a01 = edge0.Dot(edge1);
	double a01 = (-x[4] + x[7]) * (-x[4] + x[10]) + (-x[5] + x[8]) * (-x[5] + x[11]) + (-x[6] + x[9]) * (-x[6] + x[12]);
	double a01d1[12] = { 0, 0, 0, 2 * x[4] - x[7] - x[10], 2 * x[5] - x[8] - x[11], 2 * x[6] - x[9] - x[12], -x[4] + x[10], -x[5] + x[11], -x[6] + x[12], -x[4] + x[7], -x[5] + x[8], -x[6] + x[9] };

	//            double a11 = edge1.LengthSquared;
	double a11 = (-x[4] + x[10]) * (-x[4] + x[10]) + (-x[5] + x[11]) * (-x[5] + x[11]) + (-x[6] + x[12]) * (-x[6] + x[12]);
	double a11d1[12] = { 0, 0, 0, -2 * (-x[4] + x[10]), -2 * (-x[5] + x[11]), -2 * (-x[6] + x[12]), 0, 0, 0, 2 * (-x[4] + x[10]), 2 * (-x[5] + x[11]), 2 * (-x[6] + x[12]) };

	//            double b0 = diff.Dot(edge0);
	double b0 = (-x[1] + x[4]) * (-x[4] + x[7]) + (-x[2] + x[5]) * (-x[5] + x[8]) + (-x[3] + x[6]) * (-x[6] + x[9]);
	double b0d1[12] = { x[4] - x[7], x[5] - x[8], x[6] - x[9], x[1] - 2 * x[4] + x[7], x[2] - 2 * x[5] + x[8], x[3] - 2 * x[6] + x[9], -x[1] + x[4], -x[2] + x[5], -x[3] + x[6], 0, 0, 0 };

	//            double b1 = diff.Dot(edge1);
	double b1 = (-x[1] + x[4]) * (-x[4] + x[10]) + (-x[2] + x[5]) * (-x[5] + x[11]) + (-x[3] + x[6]) * (-x[6] + x[12]);
	double b1d1[12] = { x[4] - x[10], x[5] - x[11], x[6] - x[12], x[1] - 2 * x[4] + x[10], x[2] - 2 * x[5] + x[11], x[3] - 2 * x[6] + x[12], 0, 0, 0, -x[1] + x[4], -x[2] + x[5], -x[3] + x[6] };

	//            double c = diff.LengthSquared;
	double c = (-x[1] + x[4]) * (-x[1] + x[4]) + (-x[2] + x[5]) * (-x[2] + x[5]) + (-x[3] + x[6]) * (-x[3] + x[6]);
	double cd1[12] = { -2 * (-x[1] + x[4]), -2 * (-x[2] + x[5]), -2 * (-x[3] + x[6]), 2 * (-x[1] + x[4]), 2 * (-x[2] + x[5]), 2 * (-x[3] + x[6]), 0, 0, 0, 0, 0, 0 };

	double det = a00 * a11 - a01 * a01;
	double detSign = det > 0 ? 1 : -1;
	det = fabs(det);

	double s = a01 * b1 - a11 * b0;
	double t = a01 * b0 - a00 * b1;
	double sqrDistance = 0;

	if (s + t <= det)
	{
		if (s < 0)
		{
			if (t < 0)  // region 4
			{
				if (b0 < 0)
				{
					t = 0;
					if (-b0 >= a00)
					{
						s = 1;
						sqrDistance = a00 + (2) * b0 + c;
						Case_v10(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
					}
					else {
						s = -b0 / a00;
						sqrDistance = b0 * s + c;
						Case_e_t0(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
					}
				}
				else {
					s = 0;
					if (b1 >= 0)
					{
						t = 0;
						sqrDistance = c;
						Case_v00(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
					}
					else if (-b1 >= a11)
					{
						t = 1;
						sqrDistance = a11 + (2) * b1 + c;
						Case_v01(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
					}
					else {
						t = -b1 / a11;
						sqrDistance = b1 * t + c;
						Case_e_s0(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
					}
				}
			}
			else  // region 3
			{
				s = 0;
				if (b1 >= 0)
				{
					t = 0;
					sqrDistance = c;
					Case_v00(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else if (-b1 >= a11)
				{
					t = 1;
					sqrDistance = a11 + (2) * b1 + c;
					Case_v01(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else {
					t = -b1 / a11;
					sqrDistance = b1 * t + c;
					Case_e_s0(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
			}
		}
		else if (t < 0)  // region 5
		{
			t = 0;
			if (b0 >= 0)
			{
				s = 0;
				sqrDistance = c;
				Case_v00(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
			}
			else if (-b0 >= a00)
			{
				s = 1;
				sqrDistance = a00 + (2) * b0 + c;
				Case_v10(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
			}
			else {
				s = -b0 / a00;
				sqrDistance = b0 * s + c;
				Case_e_t0(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
			}
		}
		else  // region 0
		{
			// minimum at interior point
			double invDet = (1) / det;
			s *= invDet;
			t *= invDet;
			sqrDistance = s * (a00 * s + a01 * t + (2) * b0) +
				t * (a01 * s + a11 * t + (2) * b1) + c;
			Case1(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
		}
	}
	else {
		double tmp0, tmp1, numer, denom;

		if (s < 0)  // region 2
		{
			tmp0 = a01 + b0;
			tmp1 = a11 + b1;
			if (tmp1 > tmp0)
			{
				numer = tmp1 - tmp0;
				denom = a00 - (2) * a01 + a11;
				if (numer >= denom)
				{
					s = 1;
					t = 0;
					sqrDistance = a00 + (2) * b0 + c;
					Case_v10(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else {
					s = numer / denom;
					t = 1 - s;
					sqrDistance = s * (a00 * s + a01 * t + (2) * b0) +
						t * (a01 * s + a11 * t + (2) * b1) + c;
					Case_e_u0_s(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
			}
			else {
				s = 0;
				if (tmp1 <= 0)
				{
					t = 1;
					sqrDistance = a11 + (2) * b1 + c;
					Case_v01(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else if (b1 >= 0)
				{
					t = 0;
					sqrDistance = c;
					Case_v00(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else {
					t = -b1 / a11;
					sqrDistance = b1 * t + c;
					Case_e_s0(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
			}
		}
		else if (t < 0)  // region 6
		{
			tmp0 = a01 + b1;
			tmp1 = a00 + b0;
			if (tmp1 > tmp0)
			{
				numer = tmp1 - tmp0;
				denom = a00 - (2) * a01 + a11;
				if (numer >= denom)
				{
					t = 1;
					s = 0;
					sqrDistance = a11 + (2) * b1 + c;
					Case_v01(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else {
					t = numer / denom;
					s = 1 - t;
					sqrDistance = s * (a00 * s + a01 * t + (2) * b0) +
						t * (a01 * s + a11 * t + (2) * b1) + c;
					Case_e_u0_t(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
			}
			else {
				t = 0;
				if (tmp1 <= 0)
				{
					s = 1;
					sqrDistance = a00 + (2) * b0 + c;
					Case_v10(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else if (b0 >= 0)
				{
					s = 0;
					sqrDistance = c;
					Case_v00(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else {
					s = -b0 / a00;
					sqrDistance = b0 * s + c;
					Case_e_t0(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
			}
		}
		else  // region 1
		{
			numer = a11 + b1 - a01 - b0;
			if (numer <= 0)
			{
				s = 0;
				t = 1;
				sqrDistance = a11 + (2) * b1 + c;
				Case_v01(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
			}
			else {
				denom = a00 - (2) * a01 + a11;
				if (numer >= denom)
				{
					s = 1;
					t = 0;
					sqrDistance = a00 + (2) * b0 + c;
					Case_v10(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
				else {
					s = numer / denom;
					t = 1 - s;
					sqrDistance = s * (a00 * s + a01 * t + (2) * b0) +
						t * (a01 * s + a11 * t + (2) * b1) + c;
					Case_e_u0_s(x, det, detSign, a00, a01, a11, b0, b1, s, t, detd1, a00d1, a01d1, a11d1, b0d1, b1d1, cd1, wd, wdd, sqrdistd, sqrdistdd);
				}
			}
		}
	}

	// Account for numerical round-off error. (This isn't necessary because of subsequent check)
	if (sqrDistance < 0) sqrDistance = 0;

	w[1] = s; w[2] = t; w[0] = 1 - (s + t);
	return sqrDistance;
}

