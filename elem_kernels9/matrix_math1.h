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

// matrix multiplication M = A * B
__device__ __forceinline__
void multABd(double a11, double a12, double a13,
	double a21, double a22, double a23,
	double a31, double a32, double a33,
	double b11, double b12, double b13,
	double b21, double b22, double b23,
	double b31, double b32, double b33,
	double &m11, double &m12, double &m13,
	double &m21, double &m22, double &m23,
	double &m31, double &m32, double &m33)
{
	m11 = a11*b11 + a12*b21 + a13*b31; m12 = a11*b12 + a12*b22 + a13*b32; m13 = a11*b13 + a12*b23 + a13*b33;
	m21 = a21*b11 + a22*b21 + a23*b31; m22 = a21*b12 + a22*b22 + a23*b32; m23 = a21*b13 + a22*b23 + a23*b33;
	m31 = a31*b11 + a32*b21 + a33*b31; m32 = a31*b12 + a32*b22 + a33*b32; m33 = a31*b13 + a32*b23 + a33*b33;
}

__device__ __forceinline__ void fastRotationMatrix(
	double p0x, double p0y, double p0z,
	double p1x, double p1y, double p1z,
	double p2x, double p2y, double p2z,
	double &r11, double &r12, double &r13,
	double &r21, double &r22, double &r23,
	double &r31, double &r32, double &r33
	) {
	double d10x = p1x - p0x;
	double d10y = p1y - p0y;
	double d10z = p1z - p0z;

	double mag = sqrt(d10x*d10x + d10y*d10y + d10z*d10z);
	r11 = d10x / mag;
	r21 = d10y / mag;
	r31 = d10z / mag;

	// p2-p0
	double wx = p2x - p0x;
	double wy = p2y - p0y;
	double wz = p2z - p0z;

	// cross product
	double cx = -d10z * wy + d10y * wz;
	double cy = d10z * wx - d10x * wz;
	double cz = -d10y * wx + d10x * wy;

	mag = sqrt(cx*cx + cy*cy + cz*cz);
	r12 = cx / mag;
	r22 = cy / mag;
	r32 = cz / mag;

	r13 = r22 * r31 - r21 * r32;
	r23 = -r12 * r31 + r11 * r32;
	r33 = r12 * r21 - r11 * r22;
	mag = sqrt(r13*r13 + r23*r23 + r33*r33);
	r13 /= mag;
	r23 /= mag;
	r33 /= mag;
}

