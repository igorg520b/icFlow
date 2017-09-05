using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace icFlow
{
    public static class CZInsertionTool
    {
        #region Extended geometry classes
        // used only for inserting cohesive zones after .msh file is loaded
        // during the simulaiton we don't need this functionality, so we use regular classes for optimal memory use
        class ExtendedNode : Node
        {
            public bool _belongs_to_cz;
            public List<int> granules = new List<int>();
            public LinkedListNode<Node> ll_node;
            public List<Element> elementsOfNode = new List<Element>(10); // connected elems

            public ExtendedNode(Node nd) : base(nd) { }

            public void SplitNode(LinkedList<Node> allNodes)
            {
                // first, count how many granules are involved
                int nGranules = granules.Count;

                // next, add (nGranules-1) new nodes
                // and update connected elements
                for (int i = 0; i < nGranules - 1; i++)
                {
                    int currentGranule = granules[i];
                    ExtendedNode newNode = new ExtendedNode(this);
                    newNode.ll_node = allNodes.AddAfter(ll_node, newNode);
                    foreach (ExtendedElement elem in elementsOfNode)
                        if (elem.granule == currentGranule) elem.SubstituteNode(this, newNode);
                }
            }
        }

        class ExtendedFace : Face
        {
            public List<Element> elementsOfFace = new List<Element>(2);
        }

        class ExtendedElement : Element
        {
            public int id;
            public List<int> edges = new List<int>();
            public List<int> faces = new List<int>();

            public void SubstituteNode(Node oldNode, Node newNode)
            {
                int idx = Array.IndexOf<Node>(vrts, oldNode);
                vrts[idx] = newNode;
            }

            #region edges
            bool ContainsEdge(GranuleEdge ge) { return ge.vrts.All(vrts.Contains); }

            public int WhichEdge(GranuleEdge ge)
            {
                FaceID fid = FaceID.None;
                foreach (Node n in ge.vrts)
                {
                    int idx = Array.IndexOf<Node>(this.vrts, n);
                    fid |= IdxToID(idx);
                }
                if (fid == FaceID.Edge0) return 0;
                else if (fid == FaceID.Edge1) return 1;
                else if (fid == FaceID.Edge2) return 2;
                else if (fid == FaceID.Edge3) return 3;
                else if (fid == FaceID.Edge4) return 4;
                else if (fid == FaceID.Edge5) return 5;
                else throw new Exception();
            }
            public void AddEdgeIfContains(GranuleEdge ge)
            {
                if (ContainsEdge(ge)) edges.Add(WhichEdge(ge));
            }

            public GranuleEdge GetEdge(int eidx, bool exp = false)
            {
                int n0 = myConventionForEdges[eidx, 0];
                int n1 = myConventionForEdges[eidx, 1];

                GranuleEdge resultingGE = new GranuleEdge();
                resultingGE.vrts[0] = vrts[n0];
                resultingGE.vrts[1] = vrts[n1];
                resultingGE.exposed = exp;
                return resultingGE;
            }

            #endregion

            #region facets and faces
            [Flags]
            enum FaceID
            {
                None = 0, N0 = 1, N1 = 2, N2 = 4, N3 = 8,
                Face3 = N1 | N0 | N2,
                Face2 = N0 | N1 | N3,
                Face1 = N0 | N3 | N2,
                Face0 = N3 | N1 | N2,

                Edge0 = N1 | N0,
                Edge1 = N1 | N2,
                Edge2 = N1 | N3,
                Edge3 = N2 | N3,
                Edge4 = N2 | N0,
                Edge5 = N3 | N0
            }

            static FaceID IdxToID(int idx)
            {
                if (idx == 0) return FaceID.N0;
                else if (idx == 1) return FaceID.N1;
                else if (idx == 2) return FaceID.N2;
                else if (idx == 3) return FaceID.N3;
                else throw new Exception();
            }

            public bool ContainsFace(Face t) { return t.vrts.All(vrts.Contains); }

            public int WhichFace(Face t)
            {
                FaceID fid = FaceID.None;
                foreach (Node n in t.vrts)
                {
                    int idx = Array.IndexOf<Node>(this.vrts, n);
                    fid |= IdxToID(idx);
                }
                if (fid == FaceID.Face0) return 0;
                else if (fid == FaceID.Face1) return 1;
                else if (fid == FaceID.Face2) return 2;
                else if (fid == FaceID.Face3) return 3;
                else throw new Exception();
            }

            public void AddFaceIfContains(Face f)
            {
                if (ContainsFace(f)) faces.Add(WhichFace(f));
            }

            public Face GetFace(int fidx)
            {
                int n0 = myConvention[fidx, 0];
                int n1 = myConvention[fidx, 1];
                int n2 = myConvention[fidx, 2];

                Face resultingFace = new Face();
                resultingFace.vrts[0] = vrts[n0];
                resultingFace.vrts[1] = vrts[n1];
                resultingFace.vrts[2] = vrts[n2];
                resultingFace.granule = granule;
                return resultingFace;
            }

            #endregion

            #region static
            // mapping {face, faceVertex} -> tetraVertex
            static int[,] myConvention = new int[4, 3] {
            {3,1,2},
            {0,3,2},
            {0,1,3},
            {1,0,2}};

            static int[,] myConventionForEdges = new int[6, 2] {
            {1,0},
            {1,2},
            {1,3},
            {2,3},
            {2,0},
            {3,0}};

            public static int TetraIdxOfFaceVertex(int fidx, int faceVertIdx)
            {
                return myConvention[fidx, faceVertIdx];
            }

            #endregion
        }

        class ExtendedCZ : CZ
        {
            public ExtendedElement e0, e1;
            public int fidx0, fidx1; // used when inserting CZ; at simulation time contains ids of connected "unexposed" faces
            int relativeOrientationIndex = -1; // used when creating CZ from Face

            public ExtendedCZ(Face f, ExtendedElement e0, ExtendedElement e1)
            {
                // create a CZ from a face that connects 2 elements
                this.e0 = e0; this.e1 = e1;
                Trace.Assert(e0 != e1, "CZ connects two identicla elements");
                Trace.Assert(e0.granule != e1.granule, $"CZ connects same granule, g0: {e0.granule} g1: {e1.granule}");
                fidx0 = e0.WhichFace(f);
                fidx1 = e1.WhichFace(f);

                Node firstNd = e0.vrts[ExtendedElement.TetraIdxOfFaceVertex(fidx0, 0)];
                relativeOrientationIndex = -1;
                for (int i = 0; i < 3; i++)
                {
                    Node testNd = e1.vrts[ExtendedElement.TetraIdxOfFaceVertex(fidx1, i)];
                    if (testNd == firstNd) { relativeOrientationIndex = i; break; }
                }
                Trace.Assert(relativeOrientationIndex != -1);
            }

            public void ReinitializeVerticeArrays()
            {
                for (int i = 0; i < 3; i++)
                {
                    vrts[i] = e0.vrts[ExtendedElement.TetraIdxOfFaceVertex(fidx0, i)];
                    vrts[i + 3] = e1.vrts[ExtendedElement.TetraIdxOfFaceVertex(fidx1, (i + relativeOrientationIndex) % 3)];
                }
                Node temp = vrts[5];
                vrts[5] = vrts[4];
                vrts[4] = temp;
            }
        }
        #endregion

        #region convert to extended classes and back
        // convert normal geometry to extended 
        public static void Extend(this Mesh mg)
        {
            // replace Nodes, Faces and Elements with Extended
            for (int i=0;i<mg.nodes.Count;i++)
                mg.nodes[i] = new ExtendedNode(mg.nodes[i]);

            for(int i=0;i<mg.faces.Count;i++)
            {
                Face fc = mg.faces[i];
                ExtendedFace exfc = new ExtendedFace();
                for(int j=0;j<3;j++) exfc.vrts[j] = mg.nodes[fc.vrts[j].id];
                exfc.granule = fc.granule;
                exfc.tag = fc.tag;
                mg.faces[i] = exfc;
            }

            for(int i=0;i<mg.elems.Count;i++)
            {
                Element elem = mg.elems[i];
                ExtendedElement exelem = new ExtendedElement();
                exelem.id = i;
                exelem.granule = elem.granule;
                for (int j = 0; j < 4; j++) exelem.vrts[j] = mg.nodes[elem.vrts[j].id];
                mg.elems[i] = exelem;
            }

            foreach(GranuleEdge ge in mg.edges)
            {
                ge.vrts[0] = mg.nodes[ge.vrts[0].id];
                ge.vrts[1] = mg.nodes[ge.vrts[1].id];
            }
            Trace.Assert(mg.czs.Count == 0, "mg.czs already initialized");
        }

        public static void ConvertBack(this Mesh mg)
        {
            // Nodes, Faces, Elements and CZs are converted from Extended to normal
            for(int i=0;i<mg.nodes.Count;i++)
                mg.nodes[i] = new Node(mg.nodes[i]);

            for (int i = 0; i < mg.elems.Count; i++)
            {
                Element old_elem = mg.elems[i];
                old_elem.id = i;
                Element new_elem = new Element();
                for (int j = 0; j < 4; j++) new_elem.vrts[j] = mg.nodes[old_elem.vrts[j].id];
                new_elem.granule = old_elem.granule;
                mg.elems[i] = new_elem;
            }

            for (int i=0;i<mg.faces.Count;i++)
            {
                Face old_fc = mg.faces[i];
                old_fc.id = i;
                Face new_fc = new Face();
                if(old_fc.elem != null) new_fc.elem = mg.elems[old_fc.elem.id];
                new_fc.id = i;
                new_fc.tag = old_fc.tag;
                new_fc.granule = old_fc.granule;
                new_fc.exposed = old_fc.exposed;
                for (int j = 0; j < 3; j++) new_fc.vrts[j] = mg.nodes[old_fc.vrts[j].id];
                mg.faces[i] = new_fc;
            }

            for(int i=0;i<mg.czs.Count;i++)
            {
                CZ old_cz = mg.czs[i];
                CZ new_cz = new CZ();
                for (int j = 0; j < 6; j++) new_cz.vrts[j] = mg.nodes[old_cz.vrts[j].id];
                for (int j = 0; j < 2; j++) new_cz.faces[j] = mg.faces[old_cz.faces[j].id];
                mg.czs[i] = new_cz;
            }
        }
        #endregion

        #region insert CZ, caps, separate granues
        // populate elementsOfFace arrays for each Face; works on extended Mesh
        public static void IdentifyParentsOfTriangles(this Mesh mg)
        {
            foreach (ExtendedNode nd in mg.nodes) nd.elementsOfNode.Clear();
            foreach (ExtendedFace f in mg.faces) f.elementsOfFace.Clear();

            foreach (Element elem in mg.elems)
                foreach (ExtendedNode nd in elem.vrts)
                {
                    Trace.Assert(!nd.elementsOfNode.Contains(elem), "nd.elementsOfNode.Contains");
                    nd.elementsOfNode.Add(elem);
                }

            foreach(ExtendedFace f in mg.faces)
            {
                ExtendedNode nd = (ExtendedNode)f.vrts[0];
                foreach (ExtendedElement e in nd.elementsOfNode)
                {
                    if(e.ContainsFace(f))
                    {
                        f.elementsOfFace.Add(e);
                        if (f.elementsOfFace.Count == 2)
                            break;
                    }
                }
            }
        }

        // make top and bottom caps as single granule, remove extra faces
        public static void FuseEndCaps(this Mesh mg)
        {
            mg.Extend();
            mg.IdentifyParentsOfTriangles();

            HashSet<int> fuseGranules = new HashSet<int>();
            List<Node> topAndBottom = mg.nodes.FindAll(nd => nd.z0 == mg.zmax || nd.z0 == mg.zmin);
            foreach (Element elem in mg.elems)
                if (elem.vrts.Any(topAndBottom.Contains)) fuseGranules.Add(elem.granule);

            int[] iFuseGranules = fuseGranules.ToArray();
            int firstGranule = iFuseGranules[0];

            List<Face> newFaces = new List<Face>();

            // only add those that are not fused
            foreach (ExtendedFace f in mg.faces)
            {
                bool fused = false;
                if (f.elementsOfFace.Count == 2)
                {
                    int g1 = f.elementsOfFace[0].granule;
                    int g2 = f.elementsOfFace[1].granule;
                    if (fuseGranules.Contains(g1) && fuseGranules.Contains(g2)) fused = true;
                }
                if (!fused) newFaces.Add(f);
            }
            mg.faces = newFaces;
            foreach (Element elem in mg.elems) if (fuseGranules.Contains(elem.granule)) elem.granule = firstGranule;

            mg.ConvertBack();
        }

        public static void InsertCohesiveElements(this Mesh mg)
        {
            Trace.Assert(mg.czs.Count == 0, "CZs can be inserted only once");
            mg.Extend();
            mg.IdentifyParentsOfTriangles();


            // connectivity information for nodes
            foreach (ExtendedNode nd in mg.nodes) nd.elementsOfNode.Clear();
            foreach (Element e in mg.elems)
                foreach (ExtendedNode n in e.vrts)
                {
                    n.elementsOfNode.Add(e);
                    if (!n.granules.Contains(e.granule)) n.granules.Add(e.granule);
                }

            // preserve the exposed faces
            List<ExtendedFace> surface = new List<ExtendedFace>();
            List<ExtendedFace> innerTris = new List<ExtendedFace>();

            foreach(ExtendedFace f in mg.faces)
            {
                if (f.elementsOfFace.Count == 1) surface.Add(f);
                else if (f.elementsOfFace.Count == 2) innerTris.Add(f);
            }

            foreach (Face f in surface) foreach (Node nd in f.vrts) nd.isSurface = true;
            List<Tuple<ExtendedElement, int>> surfaceFaces = new List<Tuple<ExtendedElement, int>>(); // (element, faceIdx) format
            foreach (ExtendedFace f in mg.faces)
                foreach (ExtendedElement e in f.elementsOfFace)
                {
                    Trace.Assert(e.ContainsFace(f), "error in .elementsOfFace");
                        int which = e.WhichFace(f);
                        // exposed faces are preserved in surfaceFaces
                        if (f.elementsOfFace.Count == 1) surfaceFaces.Add(new Tuple<ExtendedElement, int>(e, which));
                        e.faces.Add(which);
                }

            // store all edges within extended elements
            foreach (GranuleEdge ge in mg.edges)
            {
                ExtendedNode nd = (ExtendedNode)ge.vrts[0];
                foreach (ExtendedElement elem in nd.elementsOfNode) elem.AddEdgeIfContains(ge);
            }

            // convert inner triangles into cohesive elements
            foreach (ExtendedFace f in innerTris)
            {
                ExtendedElement e0 = (ExtendedElement)f.elementsOfFace[0];
                ExtendedElement e1 = (ExtendedElement)f.elementsOfFace[1];
                ExtendedCZ ecz = new ExtendedCZ(f, e0, e1);
                mg.czs.Add(ecz);
            }

            // list the nodes, which are connected to CZs
            foreach (Face t in innerTris) foreach (ExtendedNode n in t.vrts) n._belongs_to_cz = true;
            List<ExtendedNode> nczs = new List<ExtendedNode>();
            foreach (ExtendedNode nd in mg.nodes) if (nd._belongs_to_cz) nczs.Add(nd);

            // create linked list
            LinkedList<Node> ll = new LinkedList<Node>(mg.nodes);
            LinkedListNode<Node> lln = ll.First;
            do
            {
                ((ExtendedNode)lln.Value).ll_node = lln;
                lln = lln.Next;
            } while (lln != null);

            // split the nodes, which belong to cohesive elements
            foreach (ExtendedNode nd in nczs) nd.SplitNode(ll);

            // linked list becomes the new list of nodes; resequence
            mg.nodes = new List<Node>(ll);
            for (int i = 0; i < mg.nodes.Count; i++) mg.nodes[i].id = i;

            // infer cz.vrts[] from fidx
            foreach (ExtendedCZ cz in mg.czs) cz.ReinitializeVerticeArrays();

            // restore the list of faces and the list of edges
            mg.edges.Clear();
            mg.faces.Clear();
            foreach (ExtendedElement e in mg.elems)
                foreach (int i in e.edges) mg.edges.Add(e.GetEdge(i));
            foreach (GranuleEdge ge in mg.edges) ge.exposed = true;

            // first, create exposed faces from surfaceFaces array
            foreach (Tuple<ExtendedElement, int> tuple in surfaceFaces)
            {
                ExtendedElement elem = tuple.Item1;
                int which = tuple.Item2;

                Face fc = elem.GetFace(which);
                fc.exposed = true;
                fc.id = mg.faces.Count;
                mg.faces.Add(fc);
            }

            // create non-exposed faces from CZs, record their references into cz.faces
            foreach (ExtendedCZ cz in mg.czs)
            {
                Face fc = cz.e0.GetFace(cz.fidx0);
                fc.elem = cz.e0;
                cz.faces[0] = fc;
                fc.granule = cz.e0.granule;
                fc.exposed = false;
                fc.id = mg.faces.Count;
                mg.faces.Add(fc);

                fc = cz.e1.GetFace(cz.fidx1);
                fc.elem = cz.e1;
                cz.faces[1] = fc;
                fc.exposed = false;
                fc.id = mg.faces.Count;
                fc.granule = cz.e1.granule;
                mg.faces.Add(fc);
            }

            // convert extended nodes back to regular
            mg.ConvertBack();
            mg.DetectSurfacesAfterLoadingMSH();
        }

        // create separate granules for collision testing
        public static void SeparateGranules(this Mesh mg)
        {
            mg.InsertCohesiveElements();
            lock (mg.czs)
            {
                foreach (Face f in mg.faces) f.exposed = true;
                mg.czs.Clear();
                foreach (GranuleEdge ge in mg.edges) ge.MarkExposed();
                mg.ConnectFaces();
            }
        }
        #endregion

        #region other
        static double VolumeToSize(double volume) { return 2 * Math.Pow(volume * 3 / (4*Math.PI), 1.0 / 3.0); }


        public static double AverageGrainSize(this Mesh mg)
        {
            double result = 0;
            int nGrains = mg.elems.Max(f => f.granule) + 1;
            double[] grainVolumes = new double[nGrains];
            foreach (Element elem in mg.elems)
                grainVolumes[elem.granule] += elem.volume;

            int count = 0;
            double totSize = 0;
            foreach(double vol in grainVolumes)
            {
                if(vol > 0)
                {
                    count++;
                    totSize += VolumeToSize(vol);
                }
            }

            result = totSize / count;
            return result;
        }
        #endregion
    }
}
