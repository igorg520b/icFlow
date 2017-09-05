// Copyright 2017 Igor Gribanov
// This file is part of icFlow library.
// icFlow is free software: you can redistribute it and/or modify it under the terms 
// of the GNU General Public License as published by the Free Software Foundation, either 
// version 3 of the License, or(at your option) any later version.
// icFlow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
// PURPOSE.See the GNU General Public License for more details.

using System;
using System.ComponentModel;
using System.Xml.Serialization;
using static System.Math;

namespace icFlow
{
    public class ModelPrms
    {
        public ModelPrms() { Recompute();}

        #region Simulation
        [Category("Simulation")]
        public string name { get; set; } = "1";

        [Category("Simulation")]
        public string Comments { get; set; } = "";

        [Category("Simulation")]
        public double InitialTimeStep { get; set; } = 0.025;

        [Category("Simulation")]
        [Description("Stop simulation at this step")]
        public int MaxSteps { get; set; } = 300;

        #endregion

        #region CZ
        double _alpha = 3, _beta = 3, _lambda_n = 0.02, _lambda_t = 0.02;
        double _phi_n = 30, _phi_t = 30;
        double _sigma_max = 4e5, _tau_max = 15e5;

        public enum CZFormulations { PPR_default = 0, PPR_Modified = 5}
        [Category("CZ")]
        public CZFormulations czFormulaiton { get; set; } = CZFormulations.PPR_Modified;

        [Category("CZ")]
        public double alpha { get { return _alpha; } set { _alpha = value; Recompute(); } }

        [Category("CZ")]
        public double beta { get { return _beta; } set { _beta = value; Recompute(); } }

        [Category("CZ")]
        public double lambda_n { get { return _lambda_n; } set { _lambda_n = value; Recompute(); } }

        [Category("CZ")]
        public double lambda_t { get { return _lambda_t; } set { _lambda_t = value; Recompute(); } }

        [Category("CZ")]
        [Description("Fracture energy in normal mode")]
        public double phi_n { get { return _phi_n; } set { _phi_n = value; Recompute(); } }

        [Category("CZ")]
        public double phi_t { get { return _phi_t; } set { _phi_t = value; Recompute(); } }

        [Category("CZ")]
        public double sigma_max { get { return _sigma_max; } set { _sigma_max = value; Recompute(); } }

        [Category("CZ")]
        public double tau_max { get { return _tau_max; } set { _tau_max = value; Recompute(); } }

        [Category("CZ")]
        public double _deln { get; set; }

        [Category("CZ")]
        public double _delt { get; set; }

        [Category("CZ")]
        public string _dnn00 { get; set; }


        void Recompute()
        {
            G_fn = phi_n; // fracture energy in normal mode 
            G_ft = phi_t;
            f_tn = sigma_max;
            f_tt = tau_max;
            rn = lambda_n;
            rt = lambda_t;
            pMnt = Macaulay(G_fn, G_ft);
            pMtn = Macaulay(G_ft, G_fn);

            double rn_sq = rn * rn;
            double rt_sq = rt * rt;
            p_m = (alpha * (alpha - 1.0) * rn_sq) / (1.0 - alpha * rn_sq);
            p_n = (beta * (beta - 1.0) * rt_sq) / (1.0 - beta * rt_sq);

            if (G_fn < G_ft)
            {
                gam_n = Pow(alpha / p_m, p_m);
                gam_t = -G_ft * Pow(beta / p_n, p_n);
            }
            else
            {
                gam_n = -G_fn * Pow(alpha / p_m, p_m);
                gam_t = Pow(beta / p_n, p_n);
            }

            deln = (G_fn / f_tn) * alpha * rn * Pow((1.0 - rn), (alpha - 1.0)) * ((alpha / p_m) + 1.0) * Pow(((alpha / p_m) * rn + 1.0), (p_m - 1.0));
            delt = (G_ft / f_tt) * beta * rt * Pow((1.0 - rt), (beta - 1.0)) * ((beta / p_n) + 1.0) * Pow(((beta / p_n) * rt + 1.0), (p_n - 1.0));

            _deln = deln; 
            _delt = delt; 

            _dnn00 = $"{Dnn_(0, 0):E2}";
        }

        double Dnn_(double opn, double opt)
        {
            double coeff = gam_n / (deln * deln);
            double expr1 = (p_m * p_m - p_m) * pow(1.0 - (opn / deln), alpha) * pow((p_m / alpha) + (opn / deln), p_m - 2.0);
            double expr2 = (alpha * alpha - alpha) * pow(1.0 - (opn / deln), alpha - 2.0) * pow((p_m / alpha) + (opn / deln), p_m);
            double expr3 = 2.0 * alpha * p_m * pow(1.0 - (opn / deln), alpha - 1.0) * pow((p_m / alpha) + (opn / deln), p_m - 1.0);
            double expr4 = gam_t * pow((1.0 - (opt / delt)), beta) * pow(((p_n / beta) + (opt / delt)), p_n) + pMtn;
            double result = coeff * (expr1 + expr2 - expr3) * expr4;
            return result;
        }

        static double pow(double a, double b) { return Math.Pow(a, b); }

        #endregion

        #region Material 

        [Category("Material")]
        [Description("Young's modulus")]
        public double Y { get; set; } = 1e10;

        [Category("Material")]
        [Description("Density")]
        public double rho { get; set; } = 916.2;

        [Category("Material")]
        public double dampingMass { get; set; } = 0.0005;

        [Category("Material")]
        public double dampingStiffness { get; set; } = 0.0005;

        [Category("Material")]
        [Description("Poisson ratio")]
        public double nu { get; set; } = 0.3;

        [Category("Material")]
        [Description("Computed when CZs are inserted; used to plot Hall-Petch relation. Do not change manually.")]
        public double GrainSize { get; set; }
        #endregion material

        #region Collisions

        public enum CollisionSchemes { None, Everything, RigidToDeformable, RTD_WithCreated}
        [Category("Collisions")]
        [Description("Model collisions")]
        public CollisionSchemes CollisionScheme { get; set; } = CollisionSchemes.Everything;

        [Category("Collisions")]
        public double penaltyK { get; set; } = 50;

        [Category("Collisions")]
        [Description("Smaller distance does not trigger collision reponse")]
        public double DistanceEpsilon { get; set; } = 1E-15;

        [Category("Collisions")]
        [Description("Reconstruct BVH every n steps; reduces costly BVH construction")]
        public int ReconstructBVH { get; set; }  = 10;
        #endregion

        #region Integration
        [Category("Integration")]
        public bool Symmetric { get; set; } = true;

        [Category("Integration")]
        public double NewmarkBeta { get; set; } = 0.25;

        [Category("Integration")]
        public double NewmarkGamma { get; set; } = 0.5;

        [Category("Integration")]
        [Description("Convergence criterion for Newton-Raphson")]
        public double ConvergenceEpsilon { get; set; } = 0.005;

        [Category("Integration")]
        [Description("Convergence absolute criterion, selected somewhat arbitrarily")]
        public double ConvergenceCutoff { get; set; } = 1E-8;

        [Category("Integration")]
        [Description("if the damage exceeds this value, frame is discarded and timestep is reduced")]
        public double maxDamagePerStep { get; set; } = 0.01;

        [Category("Integration")]
        [Description("if the damage exceeds this value, frame is discarded and timestep is reduced")]
        public double maxFailPerStep { get; set; } = 0.01;

        [Category("Integration")]
        [Description("Maximum number of iterations for fully implicit computation. Set to 1 for semi-implicit.")]
        public int maxIterations { get; set; } = 10;

        [Category("Integration")]
        public int minIterations { get; set; } = 3;

        [Category("Integration")]
        public double gravity { get; set; } = -9.8;

        #endregion

        #region computed variables
        [XmlIgnore]
        public double totalVolume;
        [XmlIgnore]
        public double[] E, M;
        [XmlIgnore]
        public double G_fn, G_ft; // fracture energy
        [XmlIgnore]
        public double f_tn, f_tt;
        [XmlIgnore]
        public double rn, rt;       // lambda_n, lambda_t
        [XmlIgnore]
        public double p_m, p_n;
        [XmlIgnore]
        public double deln, delt;
        [XmlIgnore]
        public double pMtn, pMnt; // < phi_t - phi_n >, < phi_n - phi_t >
        [XmlIgnore]
        public double gam_n, gam_t;
        [XmlIgnore]
        double[,] sf;
        [XmlIgnore]
        public double[] _B, _sf;

        public void SetComputedVariables()
        {
            M = new double[12 * 12];
            double coeff = rho / 20D;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    for (int m = 0; m < 3; m++)
                    {
                        int col = i * 3 + m;
                        int row = j * 3 + m;
                        M[col + 12 * row] = (col == row) ? 2 * coeff : coeff;
                    }

            double[,] E3 = new double[6, 6];
            double coeff1 = Y / ((1D + nu) * (1D - 2D * nu));
            E3[0, 0] = E3[1, 1] = E3[2, 2] = (1D - nu) * coeff1;
            E3[0, 1] = E3[0, 2] = E3[1, 2] = E3[1, 0] = E3[2, 0] = E3[2, 1] = nu * coeff1;
            E3[3, 3] = E3[4, 4] = E3[5, 5] = (0.5 - nu) * coeff1;
            E = new double[6 * 6];
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++) E[i + 6 * j] = E3[i, j];

            G_fn = phi_n; // fracture energy in normal mode 
            G_ft = phi_t;
            f_tn = sigma_max;
            f_tt = tau_max;
            rn = lambda_n;
            rt = lambda_t;
            pMnt = Macaulay(G_fn, G_ft);
            pMtn = Macaulay(G_ft, G_fn);

            double rn_sq = rn * rn;
            double rt_sq = rt * rt;
            p_m = (alpha * (alpha - 1.0) * rn_sq) / (1.0 - alpha * rn_sq);
            p_n = (beta * (beta - 1.0) * rt_sq) / (1.0 - beta * rt_sq);

            if (G_fn < G_ft)
            {
                gam_n = Pow(alpha / p_m, p_m);
                gam_t = -G_ft * Pow(beta / p_n, p_n);
            }
            else
            {
                gam_n = -G_fn * Pow(alpha / p_m, p_m);
                gam_t = Pow(beta / p_n, p_n);
            }

            deln = (G_fn / f_tn) * alpha * rn * Pow((1.0 - rn), (alpha - 1.0)) * ((alpha / p_m) + 1.0) * Pow(((alpha / p_m) * rn + 1.0), (p_m - 1.0));
            delt = (G_ft / f_tt) * beta * rt * Pow((1.0 - rt), (beta - 1.0)) * ((beta / p_n) + 1.0) * Pow(((beta / p_n) * rt + 1.0), (p_n - 1.0));


            int nGP = 3;
            sf = new double[3, nGP];

            double GP_coord_1 = 1.0 / 6.0;
            double GP_coord_2 = 2.0 / 3.0;
            sf[0, 0] = 1.0 - GP_coord_1 - GP_coord_2;
            sf[1, 0] = GP_coord_1;
            sf[2, 0] = GP_coord_2;

            GP_coord_1 = 2.0 / 3.0;
            GP_coord_2 = 1.0 / 6.0;
            sf[0, 1] = 1.0 - GP_coord_1 - GP_coord_2;
            sf[1, 1] = GP_coord_1;
            sf[2, 1] = GP_coord_2;

            GP_coord_1 = 1.0 / 6.0;
            GP_coord_2 = 1.0 / 6.0;
            sf[0, 2] = 1.0 - GP_coord_1 - GP_coord_2;
            sf[1, 2] = GP_coord_1;
            sf[2, 2] = GP_coord_2;

            double[][,] B = new double[3][,];
            for (int i = 0; i < 3; i++) B[i] = k_Bmatrix(i);

            _B = new double[3 * 3 * 18];
            _sf = new double[3 * 3];

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) _sf[i * 3 + j] = sf[i, j];

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 18; k++)
                        _B[i * 3 * 18 + j * 18 + k] = B[i][j, k];
        }

        double[,] k_Bmatrix(int i)
        {
            double[,] B = new double[3, 18];
            B[0, 0] = sf[0, i];
            B[1, 1] = sf[0, i];
            B[2, 2] = sf[0, i];
            B[0, 9] = -sf[0, i];
            B[1, 10] = -sf[0, i];
            B[2, 11] = -sf[0, i];

            B[0, 3] = sf[1, i];
            B[1, 4] = sf[1, i];
            B[2, 5] = sf[1, i];
            B[0, 12] = -sf[1, i];
            B[1, 13] = -sf[1, i];
            B[2, 14] = -sf[1, i];

            B[0, 6] = sf[2, i];
            B[1, 7] = sf[2, i];
            B[2, 8] = sf[2, i];

            B[0, 15] = -sf[2, i];
            B[1, 16] = -sf[2, i];
            B[2, 17] = -sf[2, i];
            return B;
        }

        static double Macaulay(double a, double b)
        {
            if (a > b) return a - b;
            else return 0;
        }
        #endregion

        public object Tag; // to attach any temporary data to simulation, used for Batch View/Edit
    }
}
