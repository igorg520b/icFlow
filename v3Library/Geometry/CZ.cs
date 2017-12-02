namespace icFlow
{
    public class CZ
    {
        public bool failed = false;                 // CZ has failed
        public Node[] vrts = new Node[6];
        public Face[] faces = new Face[2];          // each CZ connects two faces
        public double[] pmax = new double[3], tmax = new double[3];

        public bool damagedAtLevel(double nLevel, double tLevel)
        {
            return (pmax[0] > nLevel ||
                pmax[1] > nLevel ||
                pmax[2] > nLevel ||
                tmax[0] > tLevel ||
                tmax[1] > tLevel ||
                tmax[2] > tLevel);
        }

        public bool damaged
        {
            get
            {
                return (pmax[0] > 0 ||
    pmax[1] > 0 ||
    pmax[2] > 0 ||
    tmax[0] > 0 ||
    tmax[1] > 0 ||
    tmax[2] > 0);
            }
        }
    }
}


