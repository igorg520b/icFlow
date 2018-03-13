using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace icFlow
{
    public partial class PPR_relations : Form
    {
        public ModelPrms prms; // this has to be initialized before the window is displayed


        public PPR_relations()
        {
            InitializeComponent();
        }

        private void PPR_relations_Load(object sender, EventArgs e)
        {
            Compute();
        }

        double G_fn, G_ft; // fracture energy
        double f_tn, f_tt;
        double alpha, beta;
        double rn, rt;       // lambda_n, lambda_t
        double p_m, p_n;
        double deln, delt;
        double pMtn, pMnt; // < phi_t - phi_n >, < phi_n - phi_t >
        double gam_n, gam_t;

        double Macaulay(double a, double b)
        {
            if (a > b) return a - b;
            else return 0;
        }


        double Tn_(double Dn, double Dt)
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

        double Tt_(double Dn, double Dt)
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

        double Dtt_(double opn, double opt)
        {
            double coeff = gam_t / (delt * delt);
            double expr1 = (p_n * p_n - p_n) * pow(1.0 - (opt / delt), beta) * pow((p_n / beta) + (opt / delt), p_n - 2.0);
            double expr2 = (beta * beta - beta) * pow(1.0 - (opt / delt), beta - 2.0) * pow((p_n / beta) + (opt / delt), p_n);
            double expr3 = 2.0 * beta * p_n * pow(1.0 - (opt / delt), beta - 1.0) * pow((p_n / beta) + (opt / delt), p_n - 1.0);
            double expr4 = gam_n * pow(1.0 - (opn / deln), alpha) * pow((p_m / alpha) + (opn / deln), p_m) + pMnt;
            double result = coeff * (expr1 + expr2 - expr3) * expr4;
            return result;
        }

        double Dnt_(double opn, double opt)
        {
            double coeff = gam_n * gam_t / (deln * delt);
            double expr1 = p_m * pow(1.0 - (opn / deln), alpha) * pow((p_m / alpha) + (opn / deln), p_m - 1.0);
            double expr2 = alpha * pow(1.0 - (opn / deln), alpha - 1.0) * pow((p_m / alpha) + (opn / deln), p_m);
            double expr3 = p_n * pow(1.0 - (opt / delt), beta) * pow((p_n / beta) + (opt / delt), p_n - 1.0);
            double expr4 = beta * pow(1.0 - (opt / delt), beta - 1.0) * pow((p_n / beta) + (opt / delt), p_n);
            double result = coeff * (expr1 - expr2) * (expr3 - expr4);
            return result;
        }

        double pow(double a, double b) { return Math.Pow(a, b); }

        void Compute()
        {
            chart1.Series["Tn"].Points.Clear();
            chart1.Series["Tt"].Points.Clear();
            chart1.Series["Dnn"].Points.Clear();
            chart1.Series["Dtt"].Points.Clear();
            chart1.Series["Dnt"].Points.Clear();

            G_fn = prms.phi_n; // fracture energy in normal mode 
            G_ft = prms.phi_t;
            f_tn = prms.sigma_max;
            f_tt = prms.tau_max;
            rn = prms.lambda_n;
            rt = prms.lambda_t;
            alpha = prms.alpha;
            beta = prms.beta;

            pMnt = Macaulay(G_fn, G_ft);
            pMtn = Macaulay(G_ft, G_fn);

            double rn_sq = rn * rn;
            double rt_sq = rt * rt;

            p_m = (alpha * (alpha - 1.0) * rn_sq) / (1.0 - alpha * rn_sq);
            p_n = (beta * (beta - 1.0) * rt_sq) / (1.0 - beta * rt_sq);

            if (G_fn < G_ft)
            {
                gam_n = pow(alpha / p_m, p_m);
                gam_t = -G_ft * pow(beta / p_n, p_n);
            }
            else
            {
                gam_n = -G_fn * pow(alpha / p_m, p_m);
                gam_t = pow(beta / p_n, p_n);
            }

            deln = (G_fn / f_tn) * alpha * rn * pow((1.0 - rn), (alpha - 1.0)) * ((alpha / p_m) + 1.0) * pow(((alpha / p_m) * rn + 1.0), (p_m - 1.0));
            delt = (G_ft / f_tt) * beta * rt * pow((1.0 - rt), (beta - 1.0)) * ((beta / p_n) + 1.0) * pow(((beta / p_n) * rt + 1.0), (p_n - 1.0));

            int N = 2000;

            for (int i = 0; i < N; i++)
            {
                double current_n = ((double)i / (double)N) * deln;
                double Tn = Tn_(current_n, 0);
                chart1.Series["Tn"].Points.AddXY(current_n, Tn);

                double Dnn = Dnn_(current_n, 0);
                chart1.Series["Dnn"].Points.AddXY(current_n, Dnn);

                double current_t = ((double)i / (double)N) * delt;
                double Tt = Tt_(0, current_t);
                chart1.Series["Tt"].Points.AddXY(current_t, Tt);

                double Dtt = Dtt_(0, current_t);
                chart1.Series["Dtt"].Points.AddXY(current_t, Dtt);

                double Dnt = Dnt_(current_t, current_t);
                chart1.Series["Dnt"].Points.AddXY(current_t, Dnt);
            }

            prms.deln = deln;
            prms.delt = delt;
        }
    }
}
