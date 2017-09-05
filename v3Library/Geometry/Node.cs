using System;
using System.Collections.Generic;

namespace icFlow
{
    public class Node
    {
        public int id, altId = -1, globalNodeId;
        public double x0, y0, z0;       // undisplaced position
        public double cx, cy, cz;       // current position

        public double ux, uy, uz;       // displacement
        public double vx, vy, vz;       // velocity 
        public double ax, ay, az;       // acceleration
        public double fx, fy, fz;       // force acting on node

        // tentative
        public double unx, uny, unz;       // displacement at timestep t+h, at iteration n
        public double vnx, vny, vnz;       // velocity at timestep t+h, at iteration n
        public double anx, any, anz;       // acceleration at timestep t+h, at iteration n
        public double tx, ty, tz;          // new position
        public double dux, duy, duz;       // un - u

        public bool anchored;           // anchored nodes do not contribute to the stiffness matrix
        public bool isSurface;

        #region matrix structure and collisions
        public readonly HashSet<int> adjNeighbors = new HashSet<int>();      // set of nodes that can interact with this node through CZ or Elems (adjacent nds)
        public readonly HashSet<int> allNeighbors = new HashSet<int>();      // Same as adjNeighbors, plus the set of nodes that can interact with this node through contact
        public readonly Dictionary<int, int> pcsr = new Dictionary<int, int>();   // Mapping of interacting nodes to offsets in the sparse matrix value array
        public readonly List<Face> faces = new List<Face>();     // Adjacent faces, if any
        #endregion

        #region consturctors
        public Node()
        {
        }

        public Node(double x, double y, double z, int id) 
        {
            this.id = id;
            cx = x0 = x;
            cy = y0 = y;
            cz = z0 = z;
            altId = -1;
        }

        public Node(Node other) 
        {
            id = other.id;
            cx = x0 = other.x0;
            cy = y0 = other.y0;
            cz = z0 = other.z0;
            isSurface = other.isSurface;
        }

        /*
        public Node(Node other, int id) : this(other)
        {
            this.id = id;
        }
        */

        #endregion

        #region simulation
        public void AcceptTentativeValues(double h)
        {
            vnx = (unx - ux) / h;
            vny = (uny - uy) / h;
            vnz = (unz - uz) / h;

            ax = (vnx - vx) / h;
            ay = (vny - vy) / h;
            az = (vnz - vz) / h;

            vx = vnx; vy = vny; vz = vnz;

            ux = unx; uy = uny; uz = unz;
            cx = x0 + ux; cy = y0 + uy; cz = z0 + uz;
            dux = duy = duz = 0;
        }

        static void InferTentativeUVA(double du, double u, double v, double a, double h,
            out double un, out double vn, out double an, double beta, double gamma)
        {
            un = u + du;
            an = a * (1.0 - 1.0 / (2*beta)) + du / (h*h*beta) + v * (-1.0/(h*beta));
            vn = v + h * ((1.0 - gamma) * a + gamma * an);
//            an = 2* du / (h * h) - 2* v  / h;
//            vn = 2 * du / h - v;
        }

        public void InferTentativeValues(double h, double beta = 0.25, double gamma = 0.5)
        {
            InferTentativeUVA(dux, ux, vx, ax, h, out unx, out vnx, out anx, beta, gamma);
            InferTentativeUVA(duy, uy, vy, ay, h, out uny, out vny, out any, beta, gamma);
            InferTentativeUVA(duz, uz, vz, az, h, out unz, out vnz, out anz, beta, gamma);
            tx = x0 + unx;
            ty = y0 + uny;
            tz = z0 + unz;
        }

        #endregion

        #region CSR structure computation

        // create a mapping of the form: (ni, nj) -> CSR_Index
        // only for nj >= ni (upper triangle of the matrix)
        public void CreateCSRIndices(int startIndex, int[] cols)
        {
            SortedSet<int> sortedNeighbors = new SortedSet<int>(allNeighbors);
            pcsr.Clear();
            foreach (int _altId in sortedNeighbors)
            {
                pcsr.Add(_altId, startIndex);
                cols[startIndex] = _altId;
                startIndex++;
            }
        }
        #endregion
    }
}
