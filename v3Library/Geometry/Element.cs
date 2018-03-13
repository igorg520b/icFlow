using System.Collections.Generic;
using System.Linq;

namespace icFlow
{
    public class Element
    {
        public Node[] vrts = new Node[4];   // References to element vertices
        public int granule;                 // Granule that the element belongs to (0 if not granular)
        public int globalElementId = -1;    // Sequential numbering of exterior (surface) elements for collision detection
        public int id;                      // sequential numbering within same mesh for saving/loading faces
        public bool isSurface;
        public HashSet<Face> adjFaces;      // Collection of adjacent faces for collision detection (only on surface elements)

        public void FindAdjFaces()
        {
            if (adjFaces == null) adjFaces = new HashSet<Face>();
            else adjFaces.Clear();
            foreach (Node v in vrts) foreach (Face f in v.faces) adjFaces.Add(f);
        }

        public double volume { get
            {
                double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
                double x12, x13, x14, x23, x24, x34, x21, x31, x32, x42, x43, y12, y13, y14, y23, y24, y34;
                double y21, y31, y32, y42, y43, z12, z13, z14, z23, z24, z34, z21, z31, z32, z42, z43;
                double Jdet;
                Node[] nds = vrts;
                x1 = nds[0].x0; y1 = nds[0].y0; z1 = nds[0].z0;
                x2 = nds[1].x0; y2 = nds[1].y0; z2 = nds[1].z0;
                x3 = nds[2].x0; y3 = nds[2].y0; z3 = nds[2].z0;
                x4 = nds[3].x0; y4 = nds[3].y0; z4 = nds[3].z0;

                x12 = x1 - x2; x13 = x1 - x3; x14 = x1 - x4; x23 = x2 - x3; x24 = x2 - x4; x34 = x3 - x4;
                x21 = -x12; x31 = -x13; x32 = -x23; x42 = -x24; x43 = -x34;
                y12 = y1 - y2; y13 = y1 - y3; y14 = y1 - y4; y23 = y2 - y3; y24 = y2 - y4; y34 = y3 - y4;
                y21 = -y12; y31 = -y13; y32 = -y23; y42 = -y24; y43 = -y34;
                z12 = z1 - z2; z13 = z1 - z3; z14 = z1 - z4; z23 = z2 - z3; z24 = z2 - z4; z34 = z3 - z4;
                z21 = -z12; z31 = -z13; z32 = -z23; z42 = -z24; z43 = -z34;
                Jdet = x21 * (y23 * z34 - y34 * z23) + x32 * (y34 * z12 - y12 * z34) + x43 * (y12 * z23 - y23 * z12);
                double V = Jdet / 6.0;
                return V; // supposed to be positive
            } }


/*
        #region collision detection
        public bool IsDisjoint(Element e2) { return !vrts.Any(e2.vrts.Contains); }
        public bool IsDisjointAlt(Element e2)
        {
            if (vrts[0] == e2.vrts[0]) return false;
            if (vrts[0] == e2.vrts[1]) return false;
            if (vrts[0] == e2.vrts[2]) return false;
            if (vrts[0] == e2.vrts[3]) return false;

            if (vrts[1] == e2.vrts[0]) return false;
            if (vrts[1] == e2.vrts[1]) return false;
            if (vrts[1] == e2.vrts[2]) return false;
            if (vrts[1] == e2.vrts[3]) return false;

            if (vrts[2] == e2.vrts[0]) return false;
            if (vrts[2] == e2.vrts[1]) return false;
            if (vrts[2] == e2.vrts[2]) return false;
            if (vrts[2] == e2.vrts[3]) return false;

            if (vrts[3] == e2.vrts[0]) return false;
            if (vrts[3] == e2.vrts[1]) return false;
            if (vrts[3] == e2.vrts[2]) return false;
            if (vrts[3] == e2.vrts[3]) return false;
            return true;
        }
        public double b11, b12, b13, b21, b22, b23, b31, b32, b33; // inverse of "deformed" matrix A; B(p-x0) = barycentric coord

        public void ComputeB()
        {
            Node n0 = vrts[0];
            Node n1 = vrts[1];
            Node n2 = vrts[2];
            Node n3 = vrts[3];

            double a11, a12, a13, a21, a22, a23, a31, a32, a33;
            a11 = n1.tx - n0.tx;
            a12 = n2.tx - n0.tx;
            a13 = n3.tx - n0.tx;
            a21 = n1.ty - n0.ty;
            a22 = n2.ty - n0.ty;
            a23 = n3.ty - n0.ty;
            a31 = n1.tz - n0.tz;
            a32 = n2.tz - n0.tz;
            a33 = n3.tz - n0.tz;

            inverse(a11, a12, a13, a21, a22, a23, a31, a32, a33,
                out b11, out b12, out b13, out b21, out b22, out b23, out b31, out b32, out b33);
        }

        static void inverse(double a11, double a12, double a13,
double a21, double a22, double a23,
double a31, double a32, double a33,
out double b11, out double b12, out double b13,
out double b21, out double b22, out double b23,
out double b31, out double b32, out double b33)
        {
            double det = a31 * (-a13 * a22 + a12 * a23) + a32 * (a13 * a21 - a11 * a23) + a33 * (-a12 * a21 + a11 * a22);
            b11 = (-a23 * a32 + a22 * a33) / det;
            b12 = (a13 * a32 - a12 * a33) / det;
            b13 = (-a13 * a22 + a12 * a23) / det;
            b21 = (a23 * a31 - a21 * a33) / det;
            b22 = (-a13 * a31 + a11 * a33) / det;
            b23 = (a13 * a21 - a11 * a23) / det;
            b31 = (-a22 * a31 + a21 * a32) / det;
            b32 = (a12 * a31 - a11 * a32) / det;
            b33 = (-a12 * a21 + a11 * a22) / det;
        }



        // needs b-coefficients to be pre-computed; 
        // answers the question if a node is inside this element
        public bool NodeInElement(Node nd, out double c1, out double c2, out double c3, out double c4, bool tentative = true)
        {
            double x0, y0, z0;
            Node n = vrts[0];
            x0 = n.tx; y0 = n.ty; z0 = n.tz;
            bool result = ctest(b11, b12, b13, b21, b22, b23, b31, b32, b33,
                    nd.tx - x0, nd.ty - y0, nd.tz - z0, out c1, out c2, out c3);
            c4 = 1 - (c1 + c2 + c3);
            return result;
        }

        const double eps = 1e-10;
        static bool ctest(double a11, double a12, double a13,
double a21, double a22, double a23,
double a31, double a32, double a33,
double x1, double x2, double x3,
out double y1, out double y2, out double y3)
        {
            y1 = x1 * a11 + x2 * a12 + x3 * a13;
            y2 = x1 * a21 + x2 * a22 + x3 * a23;
            y3 = x1 * a31 + x2 * a32 + x3 * a33;
            return (y1 > eps && y2 > eps && y3 > eps && (y1 + y2 + y3) < (1 - eps));
        }
        #endregion
        */
    }
}
