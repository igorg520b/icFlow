using System;
using System.Diagnostics;

namespace icFlow
{
    public class Face
    {
        public Node[] vrts = new Node[3];
        public Element elem;                    // element that the face belongs to (null unless CZs are inserted)
        public int id;                          // sequential id
        public int globalFaceId;                // in global array
        public int granule;                     // to which granule the element belongs
        public int tag;                         // surface partition id
        public bool exposed = true;             // is this an outside surface
        public bool created = false;            // got exposed at simulation time due to fracture
        public double pnorm;                    // normal pressure on the face from collisions 


        public bool isAnchored
        {
            get
            {
                return (vrts[0].anchored && vrts[1].anchored && vrts[2].anchored);
            }
        }

        public double area { get
            {
                double tx0 = vrts[0].x0;
                double ty0 = vrts[0].y0;
                double tz0 = vrts[0].z0;
                double tx1 = vrts[1].x0;
                double ty1 = vrts[1].y0;
                double tz1 = vrts[1].z0;
                double tx2 = vrts[2].x0;
                double ty2 = vrts[2].y0;
                double tz2 = vrts[2].z0;
                double a = Math.Sqrt((tx1 - tx0) * (tx1 - tx0) + (ty1 - ty0) * (ty1 - ty0) + (tz1 - tz0) * (tz1 - tz0));
                double b = Math.Sqrt((tx2 - tx0) * (tx2 - tx0) + (ty2 - ty0) * (ty2 - ty0) + (tz2 - tz0) * (tz2 - tz0));
                double c = Math.Sqrt((tx1 - tx2) * (tx1 - tx2) + (ty1 - ty2) * (ty1 - ty2) + (tz1 - tz2) * (tz1 - tz2));
                double s = (a + b + c) / 2;
                double sres = s * (s - a) * (s - b) * (s - c);
                Debug.Assert(sres >= 0,"face area computation error");
                double result = Math.Sqrt(sres);
                return result;
            }
        }

    }
}
