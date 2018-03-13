namespace icFlow
{
    public class CZ
    {
        public int immutableID;         // preserve CZ identity to plot traction-separation relationships
        public bool failed = false;                 // CZ has failed
        public Node[] vrts = new Node[6];
        public Face[] faces = new Face[2];          // each CZ connects two faces
        public double[] pmax = new double[3], tmax = new double[3];
        public double avgDn, avgDt, avgTn, avgTt; // average traction-separations for subsequent analysis
        public double maxAvgDn, maxAvgDt;
        
        public bool damagedAtLevel(double nLevel, double tLevel)
        {
            return (pmax[0] > nLevel || tmax[0] > tLevel);
        }
        
        public enum Status { None, Softening, UnloadingReloading, Mixed }

        // only for non-failed CZs
        public Status status { get
            {
                Status result;

                if (maxAvgDn == 0 && maxAvgDt == 0) result = Status.None;
                if (maxAvgDn == avgDn && maxAvgDt == avgDt) result = Status.Softening;
                else if (maxAvgDn == avgDn || maxAvgDt == avgDt) result = Status.Mixed;
                else result = Status.UnloadingReloading;
                return result;
            } }

        /*
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
        */
    }
}


