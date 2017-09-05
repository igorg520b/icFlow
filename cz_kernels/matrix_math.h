// multiply matrix A by column vector X and return Y = AX
__device__ __forceinline__
void multAX(double a11, double a12, double a13,
	double a21, double a22, double a23,
	double a31, double a32, double a33,
	double x1, double x2, double x3,
	double &y1, double &y2, double &y3
	)
{
	y1 = x1 * a11 + x2 * a12 + x3 * a13;
	y2 = x1 * a21 + x2 * a22 + x3 * a23;
	y3 = x1 * a31 + x2 * a32 + x3 * a33;
}

__device__ __forceinline__ void CZRotationMatrix(
	double x0, double y0, double z0,
	double x1, double y1, double z1,
	double x2, double y2, double z2,
	double &r00, double &r01, double &r02,
	double &r10, double &r11, double &r12,
	double &r20, double &r21, double &r22,
	double &a_Jacob) {

	double p1x, p1y, p1z, p2x, p2y, p2z;
	p1x = x1 - x0;
	p1y = y1 - y0;
	p1z = z1 - z0;

	p2x = x0 - x2;
	p2y = y0 - y2;
	p2z = z0 - z2;

	// normalized p1 goes into 1st row of R
	double p1mag = sqrt(p1x * p1x + p1y * p1y + p1z * p1z);
	r00 = p1x / p1mag;
	r01 = p1y / p1mag;
	r02 = p1z / p1mag;

	// normalized n = p1 x p2 goes into the 3rd row
	double nx, ny, nz;
	nx = -p1z * p2y + p1y * p2z;
	ny = p1z * p2x - p1x * p2z;
	nz = -p1y * p2x + p1x * p2y;
	double nmag = sqrt(nx * nx + ny * ny + nz * nz);
	a_Jacob = nmag / 2; // area of the cohesive element
	nx /= nmag;
	ny /= nmag;
	nz /= nmag;
	r20 = nx;
	r21 = ny;
	r22 = nz;

	// normalize p1
	p1x /= p1mag;
	p1y /= p1mag;
	p1z /= p1mag;

	// second row is: r2 = n x p1
	double r2x, r2y, r2z;
	r2x = -nz * p1y + ny * p1z;
	r2y = nz * p1x - nx * p1z;
	r2z = -ny * p1x + nx * p1y;

	nmag = sqrt(r2x*r2x + r2y*r2y + r2z*r2z);
	r10 = r2x / nmag;
	r11 = r2y / nmag;
	r12 = r2z / nmag;
}
