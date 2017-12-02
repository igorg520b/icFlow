using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;


namespace icFlow
{
    public class CollisionResponse
    {
        static Stopwatch sw = new Stopwatch();
        /*
        static void computeForces(Impact[] lst, FrameInfo tcf0, ModelPrms prms)
        {
            double timeStep = tcf0.TimeStep;
            double k = prms.penaltyK;
            double distanceEpsilon = prms.DistanceEpsilon;

            Parallel.ForEach(lst, im =>
//foreach(Impact im in lst)
            {
//                double[] n = new double[3];
                double ptx, pty, ptz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z;
                double[] w;
                double[,] wd;
                double[,,] wdd;
                double[] sqrdistd;
                double[,] g;

                ptx = im.nds[0].tx;
                pty = im.nds[0].ty;
                ptz = im.nds[0].tz;
                v0x = im.nds[1].tx;
                v0y = im.nds[1].ty;
                v0z = im.nds[1].tz;
                v1x = im.nds[2].tx;
                v1y = im.nds[2].ty;
                v1z = im.nds[2].tz;
                v2x = im.nds[3].tx;
                v2y = im.nds[3].ty;
                v2z = im.nds[3].tz;

//                double[] x = new double[12] { ptx, pty, ptz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z };

                double d = DistPoint3Triangle3.PT_Derivatives(ptx, pty, ptz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z,
                    out w, out wd, out wdd, out sqrdistd, out g);
                if (d > distanceEpsilon)
                {
                    
                                    //double[] wi = new double[4];
                                    //wi[0] = -1;
                                    //wi[1] = w[0];
                                    //wi[2] = w[1];
                                    //wi[3] = w[2];
                    
                    // force acting on the "main" node
                    double fx, fy, fz;
                    //                fx = k * d * 0.25 * sqrdistd[0];
                    //                fy = k * d * 0.25 * sqrdistd[1];
                    //                fz = k * d * 0.25 * sqrdistd[2];
                    fx = k * 0.5 * sqrdistd[0];
                    fy = k * 0.5 * sqrdistd[1];
                    fz = k * 0.5 * sqrdistd[2];

                    double[] fi = im.fi;
                    fi[0] = fx;
                    fi[1] = fy;
                    fi[2] = fz;
                    fi[3] = -w[0] * fx;
                    fi[4] = -w[0] * fy;
                    fi[5] = -w[0] * fz;
                    fi[6] = -w[1] * fx;
                    fi[7] = -w[1] * fy;
                    fi[8] = -w[1] * fz;
                    fi[9] = -w[2] * fx;
                    fi[10] = -w[2] * fy;
                    fi[11] = -w[2] * fz;

                    //                double[] fDamp = new double[12];
                    //                double[,] dfDamp = new double[12,12];
                    //                DampingTerm(x, im, wi, w, wd, d, sqrdistd, out fDamp, out dfDamp);

                    double[,] dfij = im.df;
                    for (int i = 0; i < 12; i++)
                        for (int j = i; j < 12; j++)
                            dfij[i, j] = dfij[j, i] = k * g[i, j] / 2;
                }
            }
);
        }

        static void distributeForces_alt(Impact[] lst, BCSR bcsr)
        {
            // done in sequence
            foreach (Impact im in lst)
            {
                double[] fi = im.fi;
                double[,] df = im.df;

                for (int i = 0; i < 4; i++)
                {
                    Node ni = im.nds[i];
                    ni.fx += fi[i * 3 + 0];
                    ni.fy += fi[i * 3 + 1];
                    ni.fz += fi[i * 3 + 2];

                    int altId1 = ni.altId;
                    if (altId1 < 0 || im.nds[i].anchored) continue;
                    for (int c1 = 0; c1 < 3; c1++)
                        bcsr.rhs[altId1 * 3 + c1] -= (fi[i * 3 + c1]);
                    for (int j = 0; j < 4; j++)
                    {
                        Node nj = im.nds[j];
                        int altId2 = nj.altId;
                        if (altId1 > altId2 || altId2 < 0 || im.nds[j].anchored) continue;
                        int pcsr_ij = ni.pcsr[nj.altId];
                        for (int c1 = 0; c1 < 3; c1++) for (int c2 = 0; c2 < 3; c2++)
                                bcsr.vals[pcsr_ij * 9 + 3 * c1 + c2] += df[i * 3 + c1, j * 3 + c2];
                    }
                }
            }
        }

        public static void collisionResponse(Impact[] lst, FrameInfo tcf0, ModelPrms prms, BCSR bcsr)
        {
            sw.Restart();
            tcf0.nCollisions = lst.Length;
            lock(lst)
            {
                computeForces(lst, tcf0, prms);
                distributeForces_alt(lst, bcsr);
            }
            sw.Stop();
            tcf0.CollForce += sw.ElapsedMilliseconds;
        }
*/

/*        
        static void DampingTerm(double[] x, Impact im, double[] wi, double[] w, double[,] wd, 
            double d, double[] sqrdistd, out double[] f, out double[,] df)
        {
            double vD = prms.NewmarkGamma / (tcf0.TimeStep * prms.NewmarkBeta);

            Node nd0 = im.nds[0];
            Node nd1 = im.nds[1];
            Node nd2 = im.nds[2];
            Node nd3 = im.nds[3];

            double[] v = new double[12] { nd0.vnx, nd0.vny, nd0.vnz,
                nd1.vnx, nd1.vny, nd1.vnz,
                nd2.vnx, nd2.vny, nd2.vnz,
                nd3.vnx, nd3.vny, nd3.vnz};

            double[,] dn = derivatives_of_n_not_normalized(x, w, wd);
            double[,] dv = derivatives_of_Vrel(v, w, wd, vD);

            // compute vrel and n (not normalized)
            double[] vrel = new double[3];
            double[] n = new double[3];
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < 4; j++)
                {
                    vrel[k] += wi[j] * v[k + j * 3];
                    n[k] += wi[j] * x[k + j * 3];
                }

            // compute tau = vrel.n
            double tau = vrel[0] * n[0] + vrel[1] * n[1] + vrel[2] * n[2];

            double[] dtau = new double[12];
            for (int i = 0; i < 12; i++)
                for (int k = 0; k < 3; k++)
                    dtau[i] += vrel[k] * dn[k, i] + dv[k, i] * n[k];

            double mult = prms.penaltyK * tcf0.TimeStep;

            // compute force
            f = new double[12];
            for (int k = 0; k < 3; k++)
                for (int j = 0; j < 4; j++)
                    f[j * 3 + k] = mult * wi[j] * tau * n[k] / (d* d);

            // get derivatives of force
            double[,] _df = new double[3, 12];
            for (int i = 0; i < 12; i++)
                for (int k = 0; k < 3; k++)
                    _df[k, i] = -(n[k]*tau* sqrdistd[i])/(2*d*d*d) + 
                        (tau * dn[k,i] / d) + (n[k] * dtau[i] / d);


            df = new double[12, 12];
            for (int i = 0; i < 12; i++)
                for (int ndi = 0; ndi < 4; ndi++)
                    for (int ci = 0; ci < 3; ci++)
                        {
                        if (ndi == 0) {
                            df[ndi * 3 + ci, i] = -_df[ci, i]*mult;
                        }
                        else
                        {
                            int idx = ndi * 3 + ci;
                            df[idx, i] = (_df[ci, i]*wi[ndi] + f[idx]*wd[ndi-1,i])*mult;
                        }
                    }
            // print out df
        }
    */

            /*
        static double[,] derivatives_of_Vrel(double[] v, double[] w, double[,] wd, double vD)
        {
            double[,] vreld = new double[3, 12];

            for (int k = 0; k < 3; k++)
                for (int i = 0; i < 12; i++)
                    vreld[k, i] = wd[0, i] * v[3 + k] + w[0] * (i == (3 + k) ? vD : 0) +
                                wd[1, i] * v[6 + k] + w[1] * (i == (6 + k) ? vD : 0) +
                                wd[2, i] * v[9 + k] + w[2] * (i == (9 + k) ? vD : 0) - (i==k ? vD : 0);
            return vreld;
        }

        static double[,] derivatives_of_n_not_normalized(double[] x, double[] w, double[,] wd)
        {
            double[,] xcd = new double[3, 12];

            for (int k = 0; k < 3; k++)
                for (int i = 0; i < 12; i++)
                    xcd[k, i] = wd[0, i] * x[3 + k] + w[0] * (i == (3 + k) ? 1 : 0) +
                                wd[1, i] * x[6 + k] + w[1] * (i == (6 + k) ? 1 : 0) +
                                wd[2, i] * x[9 + k] + w[2] * (i == (9 + k) ? 1 : 0) - (i==k ? 1 : 0);
            return xcd;
        }
        */

        // compute impact/friction forces and derivatives as in arcSim
        /*
        public static void collisionResponse(HashSet<Impact> lst, FrameInfo tcf0, ModelPrms prms, BCSR bcsr)
        {
            sw.Restart();
            Trace.WriteLine($"imacts: {lst.Count}");

            tcf0.nCollisions = lst.Count;
            double timeStep = tcf0.TimeStep;
            double k = prms.penaltyK;
            double PsiP, PsiPP, d;
            double[] n = new double[3];
            foreach (Impact im in lst)
            {
                if (im.type == Impact.Type.VF || im.type == Impact.Type.EE)
                {
                    n[0] = im.n.X; n[1] = im.n.Y; n[2] = im.n.Z;
                    d = im.d;
                    Debug.Assert(d > 0, $"im.d = {d}");
                    PsiP = k * d * d / 2;
                    PsiPP = k * d;

                    //               PsiP = k * d; PsiPP = k;

                    Vector3d v = im.nds[0].TVel * im.w[0] + im.nds[1].TVel * im.w[1] + im.nds[2].TVel * im.w[2] + im.nds[3].TVel * im.w[3];
                    double vn = Vector3d.Dot(v, im.n);
                    double[] va = new double[3];
                    va[0] = v.X; va[1] = v.Y; va[2] = v.Z;

                    for (int i = 0; i < 4; i++)
                    {
                        Node ni = im.nds[i];


                        int altId1 = ni.altId;
                        if (altId1 < 0 || im.nds[i].anchored) continue;
                        for (int c1 = 0; c1 < 3; c1++)
                            bcsr.rhs[altId1 * 3 + c1] -= im.w[i] * n[c1] * (PsiP + PsiPP * vn * timeStep);
                        for (int j = 0; j < 4; j++)
                        {
                            Node nj = im.nds[j];
                            int altId2 = nj.altId;
                            if (altId1 > altId2 || altId2 < 0 || im.nds[j].anchored) continue;
                            int pcsr_ij = ni.pcsr[nj.altId];
                            for (int c1 = 0; c1 < 3; c1++) for (int c2 = 0; c2 < 3; c2++)
                                    bcsr.vals[pcsr_ij * 9 + 3 * c1 + c2] += PsiPP * im.w[i] * n[c1] * im.w[j] * n[c2];
                        }
                    }

                    // friction force as in arcSim
                    double mu = prms.mu;
                    if (mu != 0)
                    {
                        // find inv_mass
                        double inv_mass = 0;
                        double rho = prms.rho;
                        for (int i = 0; i < 4; i++)
                        {
                            Node nd = im.nds[i];
                            if (!nd.anchored) inv_mass += im.w[i] * im.w[i] / (nd.volume * rho);
                        }

                        // tangential component of relative velocity
                        Vector3d vt = v - im.n * (Vector3d.Dot(im.n, v));
                        double[] vta = new double[3];
                        vta[0] = vt.X; vta[1] = vt.Y; vta[2] = vt.Z;
                        //                        double[,] T = new double[3, 3];
                        //                        T[0, 0] = T[1, 1] = T[2, 2] = 1;
                        //                        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) T[i, j] -= im.n[i] * im.n[j];

                        double f_by_v_1 = mu * PsiP / vt.Length;
                        double f_by_v_2 = 1.0 / (inv_mass * timeStep);
                        double f_by_v = Math.Min(f_by_v_1, f_by_v_2);
                        if (f_by_v < 0)
                        {
                            Trace.WriteLine("f_by_v is negative");
                        }
                        //                        Debug.Assert(f_by_v >= 0, "f_by_v is negative");

                        for (int i = 0; i < 4; i++)
                        {
                            Node ni = im.nds[i];

                            int altId1 = ni.altId;
                            if (altId1 < 0 || im.nds[i].anchored) continue;
                            for (int c1 = 0; c1 < 3; c1++)
                                bcsr.rhs[altId1 * 3 + c1] -= im.w[i] * f_by_v * vta[c1];
                            for (int j = 0; j < 4; j++)
                            {
                                Node nj = im.nds[j];
                                int altId2 = nj.altId;
                                if (altId1 > altId2 || altId2 < 0 || im.nds[j].anchored) continue;
                                int pcsr_ij = ni.pcsr[nj.altId];
                                //                                for (int c1 = 0; c1 < 3; c1++) for (int c2 = 0; c2 < 3; c2++)
                                //                                      bcsr.vals[pcsr_ij * 9 + 3 * c1 + c2] += im.w[i] * im.w[j] * f_by_v * ((c1==c2 ? 1 : 0) - im.n[c1]*im.n[c2]) ;
                            }
                        }

                    }

                }
            }
            sw.Stop();
            tcf0.CollForce += sw.ElapsedMilliseconds;
        }
        */

    }
}
