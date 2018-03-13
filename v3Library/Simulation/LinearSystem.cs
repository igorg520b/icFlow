using System;
using System.Linq;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace icFlow
{
    public class LinearSystem
    {
        // allocations for sparse matrix, RHS, and functionality for solving
        // matrix can be CSR/BCSR, can be symmetric/nonsymmetric
        // diffrent types of matrix require different assembly algorithms

        // initialized when creating structure
        public ModelPrms.MatrixTypes matrixType;
        public bool symmetric { get { return matrixType == ModelPrms.MatrixTypes.BCSR_SymmetricUpper; } }
        public bool blockFormat { get { return matrixType != ModelPrms.MatrixTypes.CSR_SymmetricExtended; } }
        public int N, nnz;              // computed in CreateStructure
        public int dvalsSize { get { return blockFormat ? nnz*9 : nnz; } }
        public int dxSize { get { return blockFormat ? N*3 : N; } }

        public int[] rows, cols;         // structure arrays of the sparse matrix
        public double[] vals, rhs, dx;   // value arrays
        public const float overAllocate = 1.5f; // initially allocate larger arrays to accomodate subsequent solves

        Stopwatch sw = new Stopwatch();

        // MKL specific
        const int mklCriterionExp = 6;
        const int mklPreconditioner = 1;

        // before this function runs, it is assumed that:
        // activeNodes have sequential .altId, .neighbors are filled
        public void CreateStructure(Node[] activeNodes, FrameInfo cf, ModelPrms prms)
        {
            sw.Restart();
            matrixType = prms.MatrixType;

            nnz = activeNodes.Sum(el => el.allNeighbors.Count);
            N = activeNodes.Length;

            if(!blockFormat) { nnz *= 9; N *= 3; } // using CSR format

            // allocate structure arrays
            if (rows == null || rows.Length < N + 1) rows = new int[N + 1];
            if (cols == null || cols.Length < nnz) cols = new int[(int)(nnz*overAllocate)];
            rows[N] = nnz;

            // allocate value arrays
            if (vals == null || vals.Length < dvalsSize) vals = new double[(int)(dvalsSize * overAllocate)];
            if (dx == null || dx.Length < dxSize)
            {
                rhs = new double[dxSize];
                dx = new double[dxSize];
            }

            // create CSR indices
            int count = 0;

            if (blockFormat)
            {
                foreach (Node nd in activeNodes)
                {
                    SortedSet<int> sortedNeighbors = new SortedSet<int>(nd.allNeighbors);
                    nd.pcsr.Clear();
                    rows[nd.altId] = count; // here we expect nd.altId to be sequential
                    foreach (int _altId in sortedNeighbors)
                    {
                        nd.pcsr.Add(_altId, count);
                        cols[count] = _altId;
                        count++;
                    }

//                    count += nd.allNeighbors.Count;
                }
                rows[N] = count;

                // record matrix size in CurrentFrame object for analysis
                cf.CSR_NNZ = nnz * 9;
                cf.CSR_N = N * 3;
                cf.CSR_Mb = $"{nnz * 9 * sizeof(double) / (1024 * 1024)} Mb";
            }
            else
            {
                // CSR format, non-symmetric structure (matrix can still be symmetric-extended)
                foreach (Node nd in activeNodes)
                {
                    SortedSet<int> sortedNeighbors = new SortedSet<int>(nd.allNeighbors);
                    int rowLength = sortedNeighbors.Count; 

                    // fill rows array
                    rows[nd.altId * 3 + 0] = count;
                    rows[nd.altId * 3 + 1] = count + rowLength * 3; // (3 entries per node on each row)
                    rows[nd.altId * 3 + 2] = count + rowLength * 6;

                    // fill nd.pcsr and cols array (3 rows per iteration)
                    nd.pcsr.Clear();
                    int sub_count = 0;
                    foreach (int _altId in sortedNeighbors)
                    {
                        nd.pcsr.Add(_altId, count+sub_count*3);
                        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
                            cols[count + sub_count*3 + rowLength*3*i + j] = _altId*3 + j;
                        sub_count++;
                    }

                    count += rowLength*9;
                }

                cf.CSR_NNZ = nnz;
                cf.CSR_N = N;
                cf.CSR_Mb = $"{nnz * sizeof(double) / (1024 * 1024)} Mb";
            }

            cf.CSR_alloc = $"{vals.Length * sizeof(double) / (1024 * 1024)} Mb";
            sw.Stop();
            cf.CSRStructure += sw.ElapsedMilliseconds;
        }

        [DllImport("PardisoLoader2.dll", CallingConvention = CallingConvention.Cdecl)]
        static extern int SolveDouble3(int[] ja, int[] ia, double[] a, int n, double[] b, double[] x, int matrixType, int iparam4, int dim, int msglvl, int check);

        public void Solve(FrameInfo cf, ModelPrms prms)
        {
            sw.Restart();
            const int check = 0;
            const int verbosity = 0;

            bool CSR_format = prms.MatrixType == ModelPrms.MatrixTypes.CSR_SymmetricExtended;
            if (prms.Solver == ModelPrms.Solvers.MKL) {

                int mklMatrixType = prms.SymmetricStructure ? -2 : 11; // -2 for symmetric indefinite; 11 for nonsymmetric
                int dim = CSR_format ? 0 : 3;
                const int param4 = 0;
                Array.Clear(dx, 0, dx.Length);
                int mklResult = SolveDouble3(cols, rows, vals, N, rhs, dx, mklMatrixType, param4, dim, verbosity, check);
                if (mklResult != 0) throw new Exception("MKL solver error");
            }

            sw.Stop();
            cf.MKLSolve += sw.ElapsedMilliseconds;
        }


        #region assertion and reduction
        // used to check convergence/divergence of the solution
        public double NormOfDx()
        {
            double result = 0;
            for (int i = 0; i < dxSize; i++) result += dx[i]*dx[i];
            return result;
        }

        public void Assert(ModelPrms prms)
        {
            for(int i=0;i<dxSize;i++) Debug.Assert(!double.IsNaN(rhs[i]),"rhs constains NaN");
            for(int i=0;i<dvalsSize;i++) Debug.Assert(!double.IsNaN(vals[i]), "bcsr contains NaN");

            // verify rows array
            Debug.Assert(rows[0] == 0, "rows[0] != 0");
            Debug.Assert(rows[N] == nnz, "rows[N] != nnz");
            for (int i=1;i< N+1;i++)
                if (rows[i] <= rows[i - 1]) throw new Exception("rows[i] is not increasing");

            if (prms.SymmetricStructure)
            {
                // verify columns array, upper triangular
                for (int i = 0; i < N; i++)
                {
                    if (cols[rows[i]] != i) throw new Exception("structure not UT");
                    for (int j = rows[i]; j < rows[i + 1] - 1; j++)
                        if (cols[j + 1] <= cols[j]) throw new Exception("cols in same row not increasing");
                }
            }

        }
        #endregion
    }
}
