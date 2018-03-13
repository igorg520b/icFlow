using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace icFlow
{
    /*
    Alternative implementation of broad phase, where the tree is constructed with consideration to
    the constituent objects. Rigids objects and individual grains will not self-collide.
    (1) At simulation start, "level 2" nodes are created for each constituent object. 
        This structure will not change throughout the sim.
    (2) "Level 2" nodes are populated. The structure will not change.
    (3) Main tree is put together from level 2 nodes. Both "Create" and "Update" are available.
        */


    // Boundary Volume Hierarchy Tree - construction and traversal
    public class BVHT_Alt1
    {

        public BVHN[] Level2Nodes;

        public readonly List<kDOP24> b24;
        public readonly List<Element> broad_list = new List<Element>(3000000);
        readonly Stopwatch sw = new Stopwatch();
//        FrameInfo cf;
        public ModelPrms prms;
        public int treeConstructedStepsAgo = -1; // -1 means that reconstruciton is required

        public void ForceReconstruct() { treeConstructedStepsAgo = -1; }

        public BVHT_Alt1()
        {
            b24 = new List<kDOP24>();
            BVHN.broad_list = broad_list;
        }

        /*


        public void ConstructAndTraverse(FrameInfo cf)
        {
            if (prms.CollisionScheme == ModelPrms.CollisionSchemes.None) return;
            this.cf = cf;
            if (treeConstructedStepsAgo < 0 || treeConstructedStepsAgo >= prms.ReconstructBVH)
            {
                treeConstructedStepsAgo = 0;
                Construct();
            }
            else Update();
            Traverse();
        }

        public void Construct()
        {
            sw.Restart();
            // update KDOPs 
            //            foreach (kDOP24 k in b24) k.UpdateTentative(k.elem);
            Parallel.ForEach(b24, k => k.UpdateTentative(k.elem));

            // construct tree
            BVHN.maxLevel = 0;
            root = new BVHN(null, b24, 0);
            sw.Stop();
            cf.BVHConstructOrUpdate += sw.ElapsedMilliseconds;
            //            Trace.WriteLine($"BVH Construct {sw.ElapsedMilliseconds}");
        }

        public void Update()
        {
            // updating is a cheaper alternative to constructing the tree
            sw.Restart();
            // update KDOPs 
            Parallel.ForEach(b24, k => k.UpdateTentative(k.elem));

            // update
            root.Update();
            sw.Stop();
            cf.BVHConstructOrUpdate += sw.ElapsedMilliseconds;
            //            Trace.WriteLine($"BVH Update {sw.ElapsedMilliseconds}");
        }

        public void Traverse()
        {
            sw.Restart();
            broad_list.Clear();
            root.SelfCollide();
            sw.Stop();
            cf.BVHTraverse += sw.ElapsedMilliseconds;
        }
        */
    }
}
