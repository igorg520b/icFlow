#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])



#define SUB(dest,v1,v2) dest[0]=v1[0]-v2[0]; \
                        dest[1]=v1[1]-v2[1]; \
                        dest[2]=v1[2]-v2[2]; 

__device__ double clamp(double n) {
	return n <= 0 ? 0 : n >= 1 ? 1 : n;
}

// distance from triangle to node
__device__ double dtn(
	double f1x, double f1y, double f1z,
	double f2x, double f2y, double f2z,
	double f3x, double f3y, double f3z,
	double ndx, double ndy, double ndz) {

	double t0[3], t1[3], t2[3], edge0[3], edge1[3], v0[3], sourcePosition[3];
	t0[0] = f1x; t0[1] = f1y; t0[2] = f1z;
	t1[0] = f2x; t1[1] = f2y; t1[2] = f2z;
	t2[0] = f3x; t2[1] = f3y; t2[2] = f3z;
	sourcePosition[0] = ndx; sourcePosition[1] = ndy; sourcePosition[2] = ndz;

	SUB(edge0, t1, t0);
	SUB(edge1, t2, t0);
	SUB(v0, t0, sourcePosition);

	double a = DOT(edge0, edge0);
	double b = DOT(edge0, edge1);
	double c = DOT(edge1, edge1);
	double d = DOT(edge0, v0);
	double e = DOT(edge1, v0);

	double det = a * c - b * b;
	double s = b * e - c * d;
	double t = b * d - a * e;

	if (s + t < det)
	{
		if (s < 0)
		{
			if (t < 0)
			{
				if (d < 0)
				{
					s = clamp(-d / a);
					t = 0;
				}
				else
				{
					s = 0;
					t = clamp(-e / c);
				}
			}
			else
			{
				s = 0;
				t = clamp(-e / c);
			}
		}
		else if (t < 0)
		{
			s = clamp(-d / a);
			t = 0;
		}
		else
		{
			double invDet = 1.0 / det;
			s *= invDet;
			t *= invDet;
		}
	}
	else
	{
		if (s < 0)
		{
			double tmp0 = b + d;
			double tmp1 = c + e;
			if (tmp1 > tmp0)
			{
				double numer = tmp1 - tmp0;
				double denom = a - 2 * b + c;
				s = clamp(numer / denom);
				t = 1 - s;
			}
			else
			{
				t = clamp(-e / c);
				s = 0;
			}
		}
		else if (t < 0)
		{
			if (a + d > b + e)
			{
				double numer = c + e - b - d;
				double denom = a - 2 * b + c;
				s = clamp(numer / denom);
				t = 1 - s;
			}
			else
			{
				s = clamp(-e / c);
				t = 0;
			}
		}
		else
		{
			double numer = c + e - b - d;
			double denom = a - 2 * b + c;
			s = clamp(numer / denom);
			t = 1 - s;
		}
	}

	double d1[3];

	d1[0] = t0[0] + s*edge0[0] + t*edge1[0] - sourcePosition[0];
	d1[1] = t0[1] + s*edge0[1] + t*edge1[1] - sourcePosition[1];
	d1[2] = t0[2] + s*edge0[2] + t*edge1[2] - sourcePosition[2];

	double sqdist = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];

	//	result[0] = s;
	//	result[1] = t;
	return sqdist; // squared
}

