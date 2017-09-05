
__device__ double Tn_(double Dn, double Dt)
{
	double Dndn = Dn / deln;
	double Dtdt = Dt / delt;
	double expr2 = p_m / alpha + Dndn;
	double pr1 = gam_n / deln;
	double pr2 = (p_m * pow(1 - Dndn, alpha) * pow(expr2, p_m - 1)) -
		(alpha * pow(1 - Dndn, alpha - 1) * pow(expr2, p_m));
	double pr3 = gam_t * pow(1 - Dtdt, beta) * pow(p_n / beta + Dtdt, p_n) + pMtn;
	return pr1 * pr2 * pr3;
}

__device__ double Tt_(double Dn, double Dt)
{
	double Dndn = Dn / deln;
	double Dtdt = Dt / delt;
	double expr1 = 1 - Dtdt;
	double expr2 = p_n / beta + Dtdt;
	double pr1 = gam_t / delt;
	double pr2 = p_n * pow(expr1, beta) * pow(expr2, p_n - 1) - beta * pow(expr1, beta - 1) * pow(expr2, p_n);
	double pr3 = gam_n * pow(1 - Dndn, alpha) * pow(p_m / alpha + Dndn, p_m) + pMnt;
	return pr1 * pr2 * pr3;
}

__device__ double Dnn_(double opn, double opt)
{
	double coeff = gam_n / (deln * deln);
	double expr1 = (p_m * p_m - p_m) * pow(1.0 - (opn / deln), alpha) * pow((p_m / alpha) + (opn / deln), p_m - 2.0);
	double expr2 = (alpha * alpha - alpha) * pow(1.0 - (opn / deln), alpha - 2.0) * pow((p_m / alpha) + (opn / deln), p_m);
	double expr3 = 2.0 * alpha * p_m * pow(1.0 - (opn / deln), alpha - 1.0) * pow((p_m / alpha) + (opn / deln), p_m - 1.0);
	double expr4 = gam_t * pow((1.0 - (opt / delt)), beta) * pow(((p_n / beta) + (opt / delt)), p_n) + pMtn;
	double result = coeff * (expr1 + expr2 - expr3) * expr4;
	return result;
}

__device__ double Dtt_(double opn, double opt)
{
	double coeff = gam_t / (delt * delt);
	double expr1 = (p_n * p_n - p_n) * pow(1.0 - (opt / delt), beta) * pow((p_n / beta) + (opt / delt), p_n - 2.0);
	double expr2 = (beta * beta - beta) * pow(1.0 - (opt / delt), beta - 2.0) * pow((p_n / beta) + (opt / delt), p_n);
	double expr3 = 2.0 * beta * p_n * pow(1.0 - (opt / delt), beta - 1.0) * pow((p_n / beta) + (opt / delt), p_n - 1.0);
	double expr4 = gam_n * pow(1.0 - (opn / deln), alpha) * pow((p_m / alpha) + (opn / deln), p_m) + pMnt;
	double result = coeff * (expr1 + expr2 - expr3) * expr4;
	return result;
}

__device__ double Dnt_(double opn, double opt)
{
	double coeff = gam_n * gam_t / (deln * delt);
	double expr1 = p_m * pow(1.0 - (opn / deln), alpha) * pow((p_m / alpha) + (opn / deln), p_m - 1.0);
	double expr2 = alpha * pow(1.0 - (opn / deln), alpha - 1.0) * pow((p_m / alpha) + (opn / deln), p_m);
	double expr3 = p_n * pow(1.0 - (opt / delt), beta) * pow((p_n / beta) + (opt / delt), p_n - 1.0);
	double expr4 = beta * pow(1.0 - (opt / delt), beta - 1.0) * pow((p_n / beta) + (opt / delt), p_n);
	double result = coeff * (expr1 - expr2) * (expr3 - expr4);
	return result;
}
