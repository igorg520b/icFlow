
namespace icFlow
{
        /*
          k=18 (Aabb + 12 diagonal planes that "cut off" some space of the edges):
      (-1,0,0) and (1,0,0)  -> indices 0 and 9
      (0,-1,0) and (0,1,0)  -> indices 1 and 10
      (0,0,-1) and (0,0,1)  -> indices 2 and 11
      (-1,-1,0) and (1,1,0) -> indices 3 and 12
      (-1,0,-1) and (1,0,1) -> indices 4 and 13
      (0,-1,-1) and (0,1,1) -> indices 5 and 14
      (-1,1,0) and (1,-1,0) -> indices 6 and 15
      (-1,0,1) and (1,0,-1) -> indices 7 and 16
      (0,-1,1) and (0,1,-1) -> indices 8 and 17
*/

    // alternative implementation
    public class kDOP24
    {
        public double d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11; // lower boundaries
        public double d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23; // higher boundaries

        public Element elem; // element that is enveloped by this kDOP, if exists

        #region constructors
        public kDOP24()
        {
            Reset();
        }

        public kDOP24(kDOP24 k)
        {
            d0 = k.d0;
            d1 = k.d1;
            d2 = k.d2;
            d3 = k.d3;
            d4 = k.d4;
            d5 = k.d5;
            d6 = k.d6;
            d7 = k.d7;
            d8 = k.d8;
            d9 = k.d9;

            d10 = k.d10;
            d11 = k.d11;
            d12 = k.d12;
            d13 = k.d13;
            d14 = k.d14;
            d15 = k.d15;
            d16 = k.d16;
            d17 = k.d17;
            d18 = k.d18;
            d19 = k.d19;

            d20 = k.d20;
            d21 = k.d21;
            d22 = k.d22;
            d23 = k.d23;
        }
        #endregion

        #region update 
        public void Reset()
        {
            d0 = d1 = d2 = d3 = d4 = d5 = d6 = d7 = d8 = d9 = d10 = d11 = double.MaxValue;
            d12 = d13 = d14 = d15 = d16 = d17 = d18 = d19 = d20 = d21 = d22 = d23 = double.MinValue;
        }

        public void UpdateTentative(Element e)
        {
            Reset();
            //            foreach (Node nd in e.vrts) Expand((double)nd.tx, (double)nd.ty, (double)nd.tz);

            for (int i = 0; i < e.vrts.Length; i++)
            {
                Node nd = e.vrts[i];
                Expand(nd.tx, nd.ty, nd.tz);
            }
        }

        #endregion

        #region core functions
        public bool Overlaps(kDOP24 b)
        {
            if (d0 > b.d12) return false;
            if (d1 > b.d13) return false;
            if (d2 > b.d14) return false;
            if (d3 > b.d15) return false;
            if (d4 > b.d16) return false;
            if (d5 > b.d17) return false;
            if (d6 > b.d18) return false;
            if (d7 > b.d19) return false;
            if (d8 > b.d20) return false;
            if (d9 > b.d21) return false;
            if (d10 > b.d22) return false;
            if (d11 > b.d23) return false;

            if (d12 < b.d0) return false;
            if (d13 < b.d1) return false;
            if (d14 < b.d2) return false;
            if (d15 < b.d3) return false;
            if (d16 < b.d4) return false;
            if (d17 < b.d5) return false;
            if (d18 < b.d6) return false;
            if (d19 < b.d7) return false;
            if (d20 < b.d8) return false;
            if (d21 < b.d9) return false;
            if (d22 < b.d10) return false;
            if (d23 < b.d11) return false;

            return true;
        }

        public void Expand(double x, double y, double z)
        {
            MinMax(x, ref d0, ref d12);
            MinMax(y, ref d1, ref d13);
            MinMax(z, ref d2, ref d14);

            double pd0 = x + y;
            double pd1 = x + z;
            double pd2 = y + z;
            double pd3 = x - y;
            double pd4 = x - z;
            double pd5 = y - z;
            double pd6 = x + y - z;
            double pd7 = x + z - y;
            double pd8 = y + z - x;

            MinMax(pd0, ref d3, ref d15);
            MinMax(pd1, ref d4, ref d16);
            MinMax(pd2, ref d5, ref d17);
            MinMax(pd3, ref d6, ref d18);
            MinMax(pd4, ref d7, ref d19);
            MinMax(pd5, ref d8, ref d20);
            MinMax(pd6, ref d9, ref d21);
            MinMax(pd7, ref d10, ref d22);
            MinMax(pd8, ref d11, ref d23);
        }

        public void Expand(kDOP24 b)
        {
            d0 = min(b.d0, d0);
            d1 = min(b.d1, d1);
            d2 = min(b.d2, d2);
            d3 = min(b.d3, d3);
            d4 = min(b.d4, d4);
            d5 = min(b.d5, d5);
            d6 = min(b.d6, d6);
            d7 = min(b.d7, d7);
            d8 = min(b.d8, d8);
            d9 = min(b.d9, d9);
            d10 = min(b.d10, d10);
            d11 = min(b.d11, d11);

            d12 = max(b.d12, d12);
            d13 = max(b.d13, d13);
            d14 = max(b.d14, d14);
            d15 = max(b.d15, d15);
            d16 = max(b.d16, d16);
            d17 = max(b.d17, d17);
            d18 = max(b.d18, d18);
            d19 = max(b.d19, d19);
            d20 = max(b.d20, d20);
            d21 = max(b.d21, d21);
            d22 = max(b.d22, d22);
            d23 = max(b.d23, d23);
        }

        public void Dimensions(out double dx, out double dy, out double dz)
        {
            dx = d12 - d0;
            dy = d13 - d1;
            dz = d14 - d2;
        }

        public double centerX { get { return (d0 + d12) / 2; } }
        public double centerY { get { return (d1 + d13) / 2; } }
        public double centerZ { get { return (d2 + d14) / 2; } }

        #endregion

        #region static
        static void MinMax(double p, ref double mi, ref double ma)
        {
            if (p > ma) ma = p;
            if (p < mi) mi = p;
        }

        static void MinMax(double a, double b, ref double mi, ref double ma)
        {
            if (a > b)
            {
                mi = b; ma = a;
            }
            else
            {
                mi = a; ma = b;
            }
        }

        static double min(double a, double b) { return a < b ? a : b; }
        static double max(double a, double b) { return a > b ? a : b; }

        #endregion
    }

}
