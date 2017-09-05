using System;
using System.Linq;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace icFlow
{
    public class BCSR
    {
        public int N, nnz;              // computed in CreateStructure
        public int[] rows, cols;        // structure arrays of the sparse matrix
        public double[] vals, rhs, dx;

        Stopwatch sw = new Stopwatch();
        const int mklCriterionExp = 6;
        const int mklPreconditioner = 1;
        public int dvalsSize { get { return nnz * 9; } }
        public int dxSize { get { return N * 3; } }
        public const float overAllocate = 1.5f;

        //        int param4 { get { return 10 * mklCriterionExp + (int)mklPreconditioner; } }

        // before this function runs, it is assumed that:
        // activeNodes have sequential .altId, .neighbors are filled
        public void CreateStructure(Node[] activeNodes, FrameInfo cf)
        {
            sw.Restart();
            nnz = activeNodes.Sum(el => el.allNeighbors.Count);
            N = activeNodes.Length;

            // structure arrays of the CSR 
            if (rows == null || rows.Length < N + 1) rows = new int[N + 1];
            if (cols == null || cols.Length < nnz) cols = new int[(int)(nnz*overAllocate)];
            rows[N] = nnz;

            // create CSR indices
            int count = 0;
            foreach (Node nd in activeNodes)
            {
                nd.CreateCSRIndices(count, cols);
                rows[nd.altId] = count;
                count += nd.allNeighbors.Count;
            }

            // allocate memory for values
            if (vals == null || vals.Length < dvalsSize) vals = new double[(int)(dvalsSize * overAllocate)];
            if (dx == null || dx.Length < dxSize)
            {
                rhs = new double[dxSize];
                dx = new double[dxSize];
            }
            cf.CSR_NNZ = nnz * 9;
            cf.CSR_N = N * 3;
            cf.CSR_Mb = $"{nnz * 9 * sizeof(double) / (1024 * 1024)} Mb";
            cf.CSR_alloc = $"{vals.Length * sizeof(double) / (1024 * 1024)} Mb";
            sw.Stop();
            cf.CSRStructure += sw.ElapsedMilliseconds;
        }

        [DllImport("PardisoLoader.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int SolveDouble3(int[] ja, int[] ia, double[] a, int n, double[] b, double[] x, int matrixType, int iparam4, int dim, int msglvl);

        public void Solve(FrameInfo cf, bool symmetric = false)
        {
            sw.Restart();
            int mklMatrixType = symmetric ? -2 : 11; // -2 for symmetric indefinite; 11 for nonsymmetric

            const int dim = 3;
            const int param4 = 0;
            Array.Clear(dx, 0, dx.Length);
            int mklResult = SolveDouble3(cols, rows, vals, N, rhs, dx, mklMatrixType, param4, dim,0);
            sw.Stop();
            if (mklResult != 0) throw new Exception("MKL solver error");
            cf.MKLSolve += sw.ElapsedMilliseconds;
        }

        // used to check convergence/divergence of the solution
        public double NormOfDx()
        {
            double result = 0;
            for (int i = 0; i < dxSize; i++) result += dx[i]*dx[i];
            return result;
        }

        public void Assert()
        {
            for(int i=0;i<N*3;i++) Debug.Assert(!double.IsNaN(rhs[i]),"rhs constains NaN");
            for(int i=0;i<nnz*9;i++) Debug.Assert(!double.IsNaN(vals[i]), "bcsr contains NaN");
        }
    }
}
