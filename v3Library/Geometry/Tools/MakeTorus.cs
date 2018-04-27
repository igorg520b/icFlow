using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace icFlow
{
    public static class MeshTools
    {

        public static void MakeTorus(Mesh mg, double r1, double r2)
        {
            int nGranules = mg.elems.Max(elem => elem.granule)+1;
            List<Element>[] granules = new List<Element>[nGranules];
            for (int i = 0; i < nGranules; i++) granules[i] = new List<Element>();
            foreach (Element elem in mg.elems) granules[elem.granule].Add(elem);

            List<Element> remainingElems = new List<Element>();
            for(int i=0;i<nGranules;i++)
                if (IsInsideTorus(granules[i], r1, r2)) remainingElems.AddRange(granules[i]);
            mg.elems = remainingElems;
            
        }

        static bool IsInsideTorus(List<Element> granule, double r1, double r2)
        {
            bool result = true;
            foreach (Element elem in granule)
            {
                foreach (Node nd in elem.vrts)
                    if (!nd.IsInsideTorus(r1, r2)) result = false;
                if (result == false) break;
            }
            return result;
        }

        static bool IsInsideTorus(this Node nd, double r1, double r2)
        {
            double x = nd.x0, y = nd.y0, z = nd.z0;
            double r = Math.Sqrt(x * x + y * y);
            if (r < (r1 - r2) || r > (r1 + r2)) return false;
            if ((r - r1) * (r - r1) + (z - r2)*(z - r2) > r2 * r2) return false;
            return true;
        }
    }
}
