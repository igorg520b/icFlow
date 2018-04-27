// parameters of the model
__constant__ double E[6][6];		// elasticity matrix
__constant__ double M[12][12];		// mass matrix (nees to be multiplied by element volume)
__constant__ double NewmarkBeta, NewmarkGamma, dampingMass, dampingStiffness, rho, gravity;
__constant__ double YoungsModulus;  // for spring implementation



