// general
__constant__ double NewmarkBeta, NewmarkGamma, dampingMass, dampingStiffness, rho, gravity;
__constant__ double M[12][12];		// mass matrix (nees to be multiplied by element volume)

									// cohesive zones
__constant__ double G_fn, G_ft;		// phi_n, phi_t (fracture energy)
__constant__ double f_tn, f_tt;		// sigma_max, tau_max
__constant__ double alpha, beta;
__constant__ double rn, rt;			// lambda_n, lambda_t

__constant__ double p_m, p_n;
__constant__ double deln, delt;
__constant__ double pMtn, pMnt;		// < phi_t - phi_n >, < phi_n - phi_t > Macaulay brackets
__constant__ double gam_n, gam_t;

__constant__ double B[3][3][18];	// [# gauss points][# coordinates][dof]
__constant__ double sf[3][3];


