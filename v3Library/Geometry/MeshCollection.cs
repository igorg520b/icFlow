using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Diagnostics;

namespace icFlow
{
    public class MeshCollection
    {
        #region fields
        // list of meshes in the collecion
        public readonly List<Mesh> mgs = new List<Mesh>();

        public Node[] allNodes, activeNodes;
        public Element[] surfaceElements, elasticElements;
        public CZ[] nonFailedCZs, allCZs, failedCZs;
        public Face[] allFaces;
        public List<Mesh> deformables, nonDeformables, indenters;
        public GranuleEdge[] exposedEdges;

        #endregion

        #region MC prepare and update

        public void Prepare()
        {
            deformables = mgs.FindAll(mg => mg.isDeformable);
            nonDeformables = mgs.FindAll(mg => !mg.isDeformable);
            indenters = mgs.FindAll(mg => mg.isIndenter);

            List<Node> allNodesList = new List<Node>(mgs.Sum(mg => mg.nodes.Count));
            foreach (Mesh mg in mgs) allNodesList.AddRange(mg.nodes);
            allNodes = allNodesList.ToArray();
            for (int i = 0; i < allNodes.Length; i++) allNodes[i].globalNodeId = i;

            List<Element> elasticElementList = new List<Element>(deformables.Sum(mg => mg.elems.Count));
            List<Face> allFacesList = new List<Face>(mgs.Sum(mg => mg.faces.Count));

            foreach (Mesh mg in mgs)
            {
                allFacesList.AddRange(mg.faces);
                mg.translationCollection.Sort();
            }
            allFaces = allFacesList.ToArray();
            for (int i = 0; i < allFaces.Length; i++) allFaces[i].globalFaceId = i;

            foreach (Mesh mg in deformables) elasticElementList.AddRange(mg.elems);
            elasticElements = elasticElementList.ToArray();

            foreach (Mesh mg in nonDeformables)
                foreach (Node nd in mg.nodes) nd.anchored = true;

            foreach (Node nd in allNodes) nd.isSurface = false;
            foreach (Face fc in allFaces) if (fc.exposed || fc.created) foreach (Node nd in fc.vrts) nd.isSurface = true;

            List<GranuleEdge> edges = new List<GranuleEdge>();
            foreach (Mesh mg in deformables)
                foreach (GranuleEdge ge in mg.edges)
                    if (ge.vrts[0].isSurface && ge.vrts[1].isSurface)
                        edges.Add(ge);
            exposedEdges = edges.ToArray();

            UpdateCZs();
        }

        public void UpdateCZs()
        {
            List<CZ> CZList = new List<CZ>(deformables.Sum(mg => mg.czs.Count));
            foreach (Mesh mg in deformables) CZList.AddRange(mg.czs.FindAll(cz => !cz.failed));
            nonFailedCZs = CZList.ToArray();

            CZList.Clear();
            foreach (Mesh mg in deformables) CZList.AddRange(mg.czs);
            allCZs = CZList.ToArray();

            CZList.Clear();
            foreach (Mesh mg in deformables) CZList.AddRange(mg.czs.FindAll(cz => cz.failed));
            failedCZs = CZList.ToArray();
        }

        public void PrepareSurfaceElements()
        {
            // this is done after surface elements are marked
            List<Element> surfaceElementList = new List<Element>(mgs.Sum(mg => mg.surfaceElements.Count));
            foreach (Mesh mg in mgs) surfaceElementList.AddRange(mg.surfaceElements);
            surfaceElements = surfaceElementList.ToArray();
            for (int i = 0; i < surfaceElements.Length; i++) surfaceElements[i].globalElementId = i;
        }

        public void UpdateStaticStructureData(bool symmetric)
        {
            // update node interaction information for creating CSR later
            // this is done (1) before simulation starts, (2) if number of CZs changes

            // populate adjNeighbors
            foreach (Node nd in activeNodes) nd.adjNeighbors.Clear();
            UpdateCZs();

            if (symmetric)
            {
                foreach (CZ cz in nonFailedCZs)
                {
                    if (cz.failed) continue;
                    foreach (Node nd1 in cz.vrts)
                        foreach (Node nd2 in cz.vrts)
                            if (!nd2.anchored && nd2.altId >= nd1.altId) nd1.adjNeighbors.Add(nd2.altId);
                }
                foreach (Element elem in elasticElements)
                {
                    foreach (Node nd1 in elem.vrts)
                        foreach (Node nd2 in elem.vrts)
                            if (!nd2.anchored && nd2.altId >= nd1.altId) nd1.adjNeighbors.Add(nd2.altId);
                }
            } else
            {
                // non-symmetric
                foreach (CZ cz in nonFailedCZs)
                {
                    if (cz.failed) continue;
                    foreach (Node nd1 in cz.vrts)
                        foreach (Node nd2 in cz.vrts)
                            if (!nd2.anchored) nd1.adjNeighbors.Add(nd2.altId);
                }
                foreach (Element elem in elasticElements)
                {
                    foreach (Node nd1 in elem.vrts)
                        foreach (Node nd2 in elem.vrts)
                            if (!nd2.anchored) nd1.adjNeighbors.Add(nd2.altId);
                }
            }
        }

        #endregion

        #region Load, Save, Update Reset
        public void Save(Stream str)
        {
            BinaryWriter bw = new BinaryWriter(str);
            bw.Write(mgs.Count);
            foreach (Mesh _mg in mgs) _mg.SaveFrame(bw);
        }

        public void Load(Stream str)
        {
            BinaryReader br = new BinaryReader(str);
            int nmg = br.ReadInt32();
            Trace.Assert(nmg > 0, "save file is corrupt");

            if (mgs.Count != nmg)
            {
                mgs.Clear();
                for (int i = 0; i < nmg; i++) mgs.Add(new Mesh());
            }

            foreach (Mesh _mg in mgs) _mg.LoadFrame(str);
        }

        public void Update(Stream str)
        {
            BinaryReader br = new BinaryReader(str);
            int nmg = br.ReadInt32();
            Trace.Assert(nmg > 0, "save file is corrupt");
            foreach (Mesh _mg in mgs) _mg.LoadFrame(str, true);
        }

        public void Reset()
        {
            foreach (CZ cz in allCZs)
            {
                cz.pmax[0] = cz.pmax[1] = cz.pmax[2] = 0;
                cz.tmax[0] = cz.tmax[1] = cz.tmax[2] = 0;
                cz.failed = false;
            }
            foreach (Node nd in allNodes)
            {
                nd.ux = nd.uy = nd.uz = 0;
                nd.vx = nd.vy = nd.vz = 0;
            }
        }
        #endregion

        #region collisions
       
        public int[] collisions; // saved in strided form as 4 integer ids of nodes
        public int nCollisions, collision_stride;

        byte[] buffer;
        public void WriteImpacts(Stream str)
        {
            BinaryWriter bw = new BinaryWriter(str);
            bw.Write(nCollisions);
            bw.Write(collision_stride);
            if (nCollisions != 0)
            {
                int buffer_size = collision_stride * 4 * sizeof(int);
                if (buffer == null || buffer.Length < buffer_size) buffer = new byte[buffer_size];
                Buffer.BlockCopy(collisions, 0, buffer, 0, buffer_size);
                bw.Write(buffer, 0, buffer_size);
            }
            bw.Flush();
        }

        void AllocateCollisionsArray()
        {
            if (collisions == null || collisions.Length < collision_stride * 4) collisions = new int[collision_stride * 8];
        }

        public void ReadImpacts(Stream str)
        {
            BinaryReader br = new BinaryReader(str);
            nCollisions = br.ReadInt32();
            collision_stride = br.ReadInt32();
            if (nCollisions != 0)
            {
                int buffer_size = collision_stride * 4 * sizeof(int);
                buffer = br.ReadBytes(buffer_size);
                AllocateCollisionsArray();
                Buffer.BlockCopy(buffer, 0, collisions, 0, buffer_size);
            }
        }

        public void TransferFromAnotherArray(int[] from, int n, int stride)
        {
            nCollisions = n;
            collision_stride = stride;
            AllocateCollisionsArray();
            Buffer.BlockCopy(from, 0, collisions, 0, sizeof(int) * 4 * stride);
        }

        #endregion
    }
}
