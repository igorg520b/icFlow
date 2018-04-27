using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace icFlow
{
    /// <summary>
    /// The ValueTupe key represents a (row,column) index of the matrix element.
    /// The int value is the offset to the element in the array of values of the sparse matrix.
    /// This class does not store the double values themselves, but represents a mapping (i,j)->(offset).
    /// Row index is assumed to be dense (!), whereas the colum index is sparse.
    /// Another assumption is that j >= i (matrix is upper-triangular).
    /// </summary>
    public class CSRDictionary
    {
        public class Row
        {
            public readonly HashSet<int> staticNeighbors = new HashSet<int>();      // set of nodes that can interact with this node through CZ or Elems (adjacent nds)
            public readonly HashSet<int> allNeighbors = new HashSet<int>();      // Same as adjNeighbors, plus the set of nodes that can interact with this node through contact
        }
        public List<Row> rows = new List<Row>(100_000);
        int maxRowIndex = -1;                       // maximum row index, zero-based, equal to resulting matrix height minus 1
        public int[] csr_rows, csr_cols;            // structure arrays of the sparse matrix
        public int N, nnz;                             // number of non-zero entries
        Dictionary<ValueTuple<int, int>, int> _pcsr = new Dictionary<ValueTuple<int, int>, int>();

        // extends the list of rows to include rowIndex
        void updateMaxRowIndex(int rowIndex)
        {
            if (rowIndex <= maxRowIndex) return;
            maxRowIndex = rowIndex; // update the row count
            while (rows.Count <= rowIndex) rows.Add(new Row()); // if needed, initialize the list with empty placeholders
        }

        // complete clear
        public void ClearStatic()
        {
            maxRowIndex = -1;
            foreach (Row row in rows) row.staticNeighbors.Clear();
        }

        // clear dynamic entries (assume that matrix size does not change)
        public void ClearDynamic()
        {
            foreach (Row row in rows) row.allNeighbors.Clear();
        }

        public void AddDynamic(int row, int column)
        {
            rows[row].allNeighbors.Add(column);
        }

        public void AddStatic(int row, int column)
        {
            if (row > maxRowIndex) updateMaxRowIndex(row);
            rows[row].staticNeighbors.Add(column);
        }

        // updates N, nnz, csr_rows, csr_cols
        // user is reponsible for calling this before accessing elements by key through []
        public void CreateStructure()
        {
            N = maxRowIndex + 1;

            // parallel part (not sure if parallelism actually improves performance here)
            Parallel.For(0, N, row => {
                Row r = rows[row];
                r.allNeighbors.UnionWith(r.staticNeighbors); // combine together static and dynamic
            });

            _pcsr.Clear();

            // sequential part 
            // count non-zero entries
            nnz = rows.Sum(row => row.allNeighbors.Count);

            // allocate structure arrays
            if (csr_rows == null || csr_rows.Length < N + 1) csr_rows = new int[N + 1];
            if (csr_cols == null || csr_cols.Length < nnz) csr_cols = new int[nnz * 2]; // overallocate
            csr_rows[N] = nnz;

            // enumerate entries / initialize per-row dictionaries
            for(int i=0,count=0;i<N;i++)
            {
                csr_rows[i] = count;
                Row r = rows[i];
                foreach (int local_column in r.allNeighbors.OrderBy(x => x))
                {
                    _pcsr.Add((i, local_column), count);
                    csr_cols[count] = local_column;
                    count++;
                }
            }
        }

        public int this[int row, int column]
        {
            get
            {
                return _pcsr[(row,column)];
            }
        }

        public void Assert()
        {
            // verify rows array
            Debug.Assert(csr_rows[0] == 0, "rows[0] != 0");
            Debug.Assert(csr_rows[N] == nnz, "rows[N] != nnz");
            for (int i = 1; i < N + 1; i++)
                if (csr_rows[i] <= csr_rows[i - 1]) throw new Exception("rows[i] is not increasing");

            // verify columns array, upper triangular
            for (int i = 0; i < N; i++)
            {
                if (csr_cols[csr_rows[i]] != i) throw new Exception("structure not UT");
                for (int j = csr_rows[i]; j < csr_rows[i + 1] - 1; j++)
                    if (csr_cols[j + 1] <= csr_cols[j]) throw new Exception("cols in same row not increasing");
            }
        }
    }
}

