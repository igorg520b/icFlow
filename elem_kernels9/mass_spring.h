
__device__ double Power(double x, double y) {
	if (y == 2) return x*x;
	else return pow(x,y);
}

__device__ double Sqrt(double x) { return sqrt(x); }

__device__ void OneSpringModel(
	const double(&p1)[3], const double(&p2)[3],
	const double(&u1)[3], const double(&u2)[3],
	double(&f_osm)[6], double(&Df_osm)[6][6]) {
	// computes force and df for interaciton of one spring
	double x1 = p1[0];
	double y1 = p1[1];
	double z1 = p1[2];
	double x2 = p2[0];
	double y2 = p2[1];
	double z2 = p2[2];
	double ux1 = u1[0];
	double uy1 = u1[1];
	double uz1 = u1[2];
	double ux2 = u2[0];
	double uy2 = u2[1];
	double uz2 = u2[2];
//	ux1 = ux2 = uy1 = uy2 = uz1 = uz2 = 0;
	x1 = x1 - x2;
	x2 = 0;
	y1 = y1 - y2;
	y2 = 0;
	z1 = z1 - z2;
	z2 = 0;
	ux1 = ux1 - ux2;
	ux2 = 0;
	uy1 = uy1 - uy2;
	uy2 = 0;
	uz1 = uz1 - uz2;
	uz2 = 0;

	double k = YoungsModulus;
	double d0 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

	f_osm[0] = (k*(ux1 - ux2 + x1 - x2)*(-d0 +
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) +
			Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) +
			Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2));

	f_osm[1] = (k*(uy1 - uy2 + y1 - y2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	f_osm[2] = (k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))*(uz1 - uz2 + z1 - z2)) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

		f_osm[3] = -f_osm[0];
	f_osm[4] = -f_osm[1];
	f_osm[5] = -f_osm[2];

	Df_osm[0][0] = Df_osm[3][3] = -((k*Power(ux1 - ux2 + x1 - x2, 2)*(-d0 +
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2), 1.5)) +
		(k*Power(ux1 - ux2 + x1 - x2, 2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2)) +
		(k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	Df_osm[1][1] = Df_osm[4][4] = -((k*Power(uy1 - uy2 + y1 - y2, 2)*(-d0 +
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2), 1.5)) +
		(k*Power(uy1 - uy2 + y1 - y2, 2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2)) +
		(k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	Df_osm[2][2] = Df_osm[5][5] = (k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2))\
		- (k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))*Power(uz1 - uz2 + z1 - z2, 2)) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) + (k*Power(uz1 - uz2 + z1 - z2, 2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// x1-y1, x2-y2
	Df_osm[0][1] = Df_osm[1][0] = Df_osm[3][4] = Df_osm[4][3] = -((k*(ux1 - ux2 + x1 - x2)*(uy1 - uy2 + y1 - y2)*
		(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2), 1.5)) +
		(k*(ux1 - ux2 + x1 - x2)*(uy1 - uy2 + y1 - y2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// x1-z1, x2-z2
	Df_osm[0][2] = Df_osm[2][0] = Df_osm[3][5] = Df_osm[5][3] = -((k*(ux1 - ux2 + x1 - x2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))*(uz1 - uz2 + z1 - z2)) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2), 1.5)) +
		(k*(ux1 - ux2 + x1 - x2)*(uz1 - uz2 + z1 - z2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// y1-z1, y2-z2
	Df_osm[1][2] = Df_osm[2][1] = Df_osm[4][5] = Df_osm[5][4] = -((k*(uy1 - uy2 + y1 - y2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))*(uz1 - uz2 + z1 - z2)) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2), 1.5)) +
		(k*(uy1 - uy2 + y1 - y2)*(uz1 - uz2 + z1 - z2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// ux1, ux2
	Df_osm[0][3] = Df_osm[3][0] = (k*Power(ux1 - ux2 + x1 - x2, 2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) +
		Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2)))) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) - (k*Power(ux1 - ux2 + x1 - x2, 2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2)) -
		(k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	Df_osm[1][4] = Df_osm[4][1] = (k*Power(uy1 - uy2 + y1 - y2, 2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) +
		Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2)))) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) - (k*Power(uy1 - uy2 + y1 - y2, 2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2)) -
		(k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// z1 z2
	Df_osm[2][5] = Df_osm[5][2] = -((k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))) /
		Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2))
		) + (k*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))*Power(uz1 - uz2 + z1 - z2, 2)) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) - (k*Power(uz1 - uz2 + z1 - z2, 2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// x1-y2, x2-y1
	Df_osm[0][4] = Df_osm[4][0] = Df_osm[1][3] = Df_osm[3][1] = (k*(ux1 - ux2 + x1 - x2)*(uy1 - uy2 + y1 - y2)*
		(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
			Power(uz1 - uz2 + z1 - z2, 2)))) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) - (k*(ux1 - ux2 + x1 - x2)*(uy1 - uy2 + y1 - y2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// x1-z2
	Df_osm[0][5] = Df_osm[5][0] = Df_osm[2][3] = Df_osm[3][2] = (k*(ux1 - ux2 + x1 - x2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))*(uz1 - uz2 + z1 - z2)) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) - (k*(ux1 - ux2 + x1 - x2)*(uz1 - uz2 + z1 - z2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));

	// y1-z2
	Df_osm[1][5] = Df_osm[5][1] = Df_osm[2][4] = Df_osm[4][2] = (k*(uy1 - uy2 + y1 - y2)*(-d0 + Sqrt(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) +
		Power(uz1 - uz2 + z1 - z2, 2)))*(uz1 - uz2 + z1 - z2)) /
		Power(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2),
			1.5) - (k*(uy1 - uy2 + y1 - y2)*(uz1 - uz2 + z1 - z2)) /
		(Power(ux1 - ux2 + x1 - x2, 2) + Power(uy1 - uy2 + y1 - y2, 2) + Power(uz1 - uz2 + z1 - z2, 2));


}

__device__ void F_and_Df_Spring(
	const double(&x0)[12], const double(&un)[12],
	double(&f)[12], double(&Df)[12][12], double &V) {

	double f_osm[6], Df_osm[6][6];
	double p1[3], p2[3], u1[3], u2[3];

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (i > j) {

				for (int k = 0; k < 3; k++) {
					p1[k] = x0[i * 3 + k];
					p2[k] = x0[j * 3 + k];
					u1[k] = un[i * 3 + k];
					u2[k] = un[j * 3 + k];
				}
				OneSpringModel(p1, p2, u1, u2, f_osm, Df_osm);
				for (int k = 0; k < 3; k++) {
					f[i * 3 + k] += f_osm[k];
					f[j * 3 + k] += f_osm[k + 3];
					for (int l = 0; l < 3; l++) {
						Df[i * 3 + k][i * 3 + l] += Df_osm[k][l];
						Df[j * 3 + k][j * 3 + l] += Df_osm[k + 3][l + 3];
						Df[i * 3 + k][j * 3 + l] += Df_osm[k][l + 3];
						Df[j * 3 + k][i * 3 + l] += Df_osm[k + 3][l];
					}
				}

			}
		}
	}
}
