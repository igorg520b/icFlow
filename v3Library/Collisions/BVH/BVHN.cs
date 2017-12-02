using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace icFlow
{
    // represents a node of a BV tree
    public class BVHN
    {
        public static int maxLevel;
        public static List<Element> broad_list; // list of pairs of elements, reuslt of broad phase

        public kDOP24 box;
        public BVHN child1, child2, parent;
        public int level;
        public bool isLeaf;
        List<kDOP24> bvs;

        #region initialization

        // create new node from a list of bounding volumes
        public BVHN(BVHN parent, List<kDOP24> bvs, int level)
        {
            // this way we can reuse already existing BVHN objects form the pool
            Initialize(parent, bvs, level);
        }

        public void Initialize(BVHN parent, List<kDOP24> bvs, int level)
        {
            Debug.Assert(bvs.Count > 0);
            this.parent = parent;
            this.bvs = bvs;
            this.level = level;
            if (maxLevel < level) maxLevel = level;

            FinalizeConstruction();
        }

        void FinalizeConstruction()
        {
            if (bvs.Count == 1)
            {
                // this node is leaf
                box = bvs[0];
                isLeaf = true;
                return;
            }

            // find bounding box for bvs
            box = new kDOP24();
            foreach (kDOP24 bv in bvs) box.Expand(bv);
            double dX, dY, dZ;
            box.Dimensions(out dX, out dY, out dZ);

            // arrays where left and right portions will be stored
            List<kDOP24> left = new List<kDOP24>(bvs.Count);
            List<kDOP24> right = new List<kDOP24>(bvs.Count);

            // identify splitting axis and split
            if (dX >= dY && dX >= dZ)
            {
                double center = box.centerX;
                foreach (kDOP24 bv in bvs)
                {
                    if (bv.centerX < center) left.Add(bv);
                    else right.Add(bv);
                }
                // make sure that there is at least one element on each side
                if (left.Count == 0)
                {
                    kDOP24 selected = null;
                    double min1 = double.MaxValue;
                    foreach (kDOP24 bv in right)
                        if (min1 >= bv.centerX) { min1 = bv.centerX; selected = bv; }
                    left.Add(selected);
                    right.Remove(selected);
                }
                else if (right.Count == 0)
                {
                    kDOP24 selected = null;
                    double min1 = double.MaxValue;
                    foreach (kDOP24 bv in left)
                        if (min1 >= bv.centerX) { min1 = bv.centerX; selected = bv; }
                    right.Add(selected);
                    left.Remove(selected);
                }
            }
            else if (dY >= dX && dY >= dZ)
            {
                double center = box.centerY;
                foreach (kDOP24 bv in bvs)
                {
                    if (bv.centerY < center) left.Add(bv);
                    else right.Add(bv);
                }
                if (left.Count == 0)
                {
                    kDOP24 selected = null;
                    double min1 = double.MaxValue;
                    foreach (kDOP24 bv in right)
                        if (min1 >= bv.centerY) { min1 = bv.centerY; selected = bv; }
                    left.Add(selected);
                    right.Remove(selected);
                }
                else if (right.Count == 0)
                {
                    kDOP24 selected = null;
                    double min1 = double.MaxValue;
                    foreach (kDOP24 bv in left)
                        if (min1 >= bv.centerY) { min1 = bv.centerY; selected = bv; }
                    right.Add(selected);
                    left.Remove(selected);
                }
            }
            else
            {
                double center = box.centerZ;
                foreach (kDOP24 bv in bvs)
                {
                    if (bv.centerZ < center) left.Add(bv);
                    else right.Add(bv);
                }
                if (left.Count == 0)
                {
                    kDOP24 selected = null;
                    double min1 = double.MaxValue;
                    foreach (kDOP24 bv in right)
                        if (min1 >= bv.centerZ) { min1 = bv.centerZ; selected = bv; }
                    left.Add(selected);
                    right.Remove(selected);
                }
                else if (right.Count == 0)
                {
                    kDOP24 selected = null;
                    double min1 = double.MaxValue;
                    foreach (kDOP24 bv in left)
                        if (min1 >= bv.centerZ) { min1 = bv.centerZ; selected = bv; }
                    right.Add(selected);
                    left.Remove(selected);
                }
            }
            left.TrimExcess();
            right.TrimExcess();

            child1 = new BVHN(this, left, level + 1);
            child2 = new BVHN(this, right, level + 1);
        }

        #endregion

        #region update
        public void Update()
        {
            // traverse the tree, but simply update the values bottom->up
            if (!child1.isLeaf) child1.Update();
            if (!child2.isLeaf) child2.Update();
            box.Reset();
            box.Expand(child1.box);
            box.Expand(child2.box);
            //            foreach (kDOP24 bv in bvs) box.Expand(bv);
        }
        #endregion

        #region traversal

        public void SelfCollide()
        {
            if (isLeaf) return;
            child1.SelfCollide();
            child2.SelfCollide();
            child1.Collide(child2);
        }

        public void Collide(BVHN b)
        {
            if (!box.Overlaps(b.box)) return;
            if (this.isLeaf && b.isLeaf)
            {
                Element e1 = box.elem;
                Element e2 = b.box.elem;
                broad_list.Add(e1);
                broad_list.Add(e2);
            }
            else if (this.isLeaf)
            {
                Collide(b.child1);
                Collide(b.child2);
            }
            else
            {
                b.Collide(child1);
                b.Collide(child2);
            }
        }

        #endregion

    }
}
