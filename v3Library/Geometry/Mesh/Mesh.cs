using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.ComponentModel;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;

namespace icFlow
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class Mesh
    {
        #region fields
        public List<Node> nodes = new List<Node>();
        public List<Element> elems = new List<Element>();
        public List<CZ> czs = new List<CZ>();
        public List<Face> faces = new List<Face>();
        public List<GranuleEdge> edges = new List<GranuleEdge>(); // for drawing
        public List<SurfaceFragment> surfaceFragments = new List<SurfaceFragment>();
        public List<Element> surfaceElements;           // elements that can potentially come in contact
        public TranslationCollection translationCollection = new TranslationCollection();
        #endregion

        #region properties
        public bool isDeformable { get; set; }
        public bool isIndenter { get; set; }
        public bool isFloor { get; set; }
        public bool hide { get; set; }
        public string name { get; set; }
        // bounding box
        public double xmin { get; set; }
        public double xmax { get; set; }
        public double ymin { get; set; }
        public double ymax { get; set; }
        public double zmin { get; set; }
        public double zmax { get; set; }
        public double scale { get; set; }
        public int nNodes { get { return nodes.Count; } }
        public int nElems { get { return elems.Count; } }
        public int nCZs { get { return czs.Count; } }
        public int nSurfaceElems { get { return surfaceElements == null ? 0 : surfaceElements.Count; } }
        public int nFaces { get { return faces.Count; } }
        public override string ToString() { return name; }
        double _volume;
        public double volume { get { return _volume; } }
        #endregion

        #region initialization

        public Mesh() { }

        public Mesh(Stream str, string name, string type) : this() {
            this.name = name;
            if (type == ".geo") LoadGeo(str);
            else if (type == ".msh") LoadMsh(str);
            else if (type == ".mg") LoadFrame(str);
            else throw new Exception("Load: incorrect mesh type");
        }

        public void BoundingBox()
        {
            xmin = nodes.Min(nd => nd.x0);
            xmax = nodes.Max(nd => nd.x0);
            ymin = nodes.Min(nd => nd.y0);
            ymax = nodes.Max(nd => nd.y0);
            zmin = nodes.Min(nd => nd.z0);
            zmax = nodes.Max(nd => nd.z0);
        }

        public double width { get { return xmax - xmin; } }
        public double height { get { return zmax - zmin; } }

        #endregion

        #region custom binary serialization

        int[] isn;
        double[] dsn;
        bool[] bsn;
        byte[] by_snapshot;

        // strides for double/integer/bool
        const int ndd_str = 12;                             // nodes
        const int eli_str = 5;                              // elements
        const int czi_str = 9, czd_str = 12, czb_str=1;      // czs
        const int fi_str = 5, fb_str = 2;                   // faces
        const int gei_str = 2, geb_str = 1;                 // edges

        int iSize, bSize, dSize, byteSize;
        void LoadSaveMemAlloc()
        {
            // allocate mem
            iSize = elems.Count * eli_str + czs.Count * czi_str + faces.Count * fi_str + edges.Count * gei_str;
            bSize = faces.Count * fb_str + edges.Count * geb_str + nCZs * czb_str;
            dSize = nodes.Count * ndd_str + czs.Count * czd_str;
            if (isn == null || isn.Length < iSize) isn = new int[iSize];
            if (dsn == null || dsn.Length < dSize) dsn = new double[dSize];
            if (bsn == null || bsn.Length < bSize) bsn = new bool[bSize];
            byteSize = iSize * sizeof(int) + bSize * sizeof(bool) + dSize * sizeof(double);
        }

        public void SaveFrame(BinaryWriter bw)
        {
            LoadSaveMemAlloc();

            // write nodes
            int i_off = 0, d_off = 0, b_off = 0;
            Parallel.For(0, nodes.Count, i => { 
                Node nd = nodes[i];
                Debug.Assert(nd.id == i,"Node ids are not sequential");
                dsn[i * ndd_str + 0] = nd.x0;
                dsn[i * ndd_str + 1] = nd.y0;
                dsn[i * ndd_str + 2] = nd.z0;
                dsn[i * ndd_str + 3] = nd.ux;
                dsn[i * ndd_str + 4] = nd.uy;
                dsn[i * ndd_str + 5] = nd.uz;
                dsn[i * ndd_str + 6] = nd.vx;
                dsn[i * ndd_str + 7] = nd.vy;
                dsn[i * ndd_str + 8] = nd.vz;
                dsn[i * ndd_str + 9] = nd.ax;
                dsn[i * ndd_str + 10] = nd.ay;
                dsn[i * ndd_str + 11] = nd.az;
            });
            d_off += ndd_str * nodes.Count;

            // elements
            Parallel.For(0, elems.Count, i => {
                Element elem = elems[i];
                elem.id = i; 
                isn[i * eli_str + 0] = elem.vrts[0].id;
                isn[i * eli_str + 1] = elem.vrts[1].id;
                isn[i * eli_str + 2] = elem.vrts[2].id;
                isn[i * eli_str + 3] = elem.vrts[3].id;
                isn[i * eli_str + 4] = elem.granule;
            });
            i_off += eli_str * elems.Count;

            // cohesive zones
            Parallel.For(0, czs.Count, i => {
                CZ cz = czs[i];
                isn[i_off + i * czi_str + 0] = cz.vrts[0].id;
                isn[i_off + i * czi_str + 1] = cz.vrts[1].id;
                isn[i_off + i * czi_str + 2] = cz.vrts[2].id;
                isn[i_off + i * czi_str + 3] = cz.vrts[3].id;
                isn[i_off + i * czi_str + 4] = cz.vrts[4].id;
                isn[i_off + i * czi_str + 5] = cz.vrts[5].id;
                isn[i_off + i * czi_str + 6] = cz.faces[0].id;
                isn[i_off + i * czi_str + 7] = cz.faces[1].id;
                isn[i_off + i * czi_str + 8] = cz.immutableID;

                dsn[d_off + i * czd_str + 0] = cz.pmax[0];
                dsn[d_off + i * czd_str + 1] = cz.pmax[1];
                dsn[d_off + i * czd_str + 2] = cz.pmax[2];
                dsn[d_off + i * czd_str + 3] = cz.tmax[0];
                dsn[d_off + i * czd_str + 4] = cz.tmax[1];
                dsn[d_off + i * czd_str + 5] = cz.tmax[2];
                dsn[d_off + i * czd_str + 6] = cz.avgDn;
                dsn[d_off + i * czd_str + 7] = cz.avgDt;
                dsn[d_off + i * czd_str + 8] = cz.avgTn;
                dsn[d_off + i * czd_str + 9] = cz.avgTt;
                dsn[d_off + i * czd_str + 10] = cz.maxAvgDn;
                dsn[d_off + i * czd_str + 11] = cz.maxAvgDt;

                bsn[b_off + i + 0] = cz.failed;
            });

            b_off += czb_str * czs.Count;
            i_off += czi_str * czs.Count;
            d_off += czd_str * czs.Count;

            // faces
            Parallel.For(0, faces.Count, i => {
                Face f = faces[i];
                isn[i_off + i * fi_str + 0] = f.vrts[0].id;
                isn[i_off + i * fi_str + 1] = f.vrts[1].id;
                isn[i_off + i * fi_str + 2] = f.vrts[2].id;
                isn[i_off + i * fi_str + 3] = f.granule;
                isn[i_off + i * fi_str + 4] = f.elem == null ? -1 : f.elem.id;
                bsn[b_off + i * fb_str + 0] = f.exposed;
                bsn[b_off + i * fb_str + 1] = f.created;
            });

            b_off += fb_str * faces.Count;
            i_off += fi_str * faces.Count;

            // edges
            Parallel.For(0, edges.Count, i => {
                GranuleEdge ge = edges[i];
                isn[i_off + i * gei_str + 0] = ge.vrts[0].id;
                isn[i_off + i * gei_str + 1] = ge.vrts[1].id;
                bsn[b_off + i] = ge.exposed;
            });

            // convert to byte array and write
            if (by_snapshot == null || by_snapshot.Length < byteSize) by_snapshot = new byte[byteSize];
            Buffer.BlockCopy(isn, 0, by_snapshot, 0, iSize * sizeof(int));
            int offset = iSize * sizeof(int);
            Buffer.BlockCopy(bsn, 0, by_snapshot, offset, bSize * sizeof(bool));
            offset += bSize * sizeof(bool);
            Buffer.BlockCopy(dsn, 0, by_snapshot, offset, dSize * sizeof(double));

            // save mesh properties
            bw.Write(isDeformable);
            bw.Write(isIndenter);
            bw.Write(isFloor);
            bw.Write(hide);
            bw.Write(name);

            bw.Write(nodes.Count);
            bw.Write(elems.Count);
            bw.Write(czs.Count);
            bw.Write(faces.Count);
            bw.Write(edges.Count);
            bw.Write(by_snapshot, 0, byteSize);
            bw.Flush();

            IFormatter bf = new BinaryFormatter();
            bf.Serialize(bw.BaseStream, surfaceFragments);
            bf.Serialize(bw.BaseStream, translationCollection);
            bw.Flush();
        }

        // loads from proprietary binary format
        public void LoadFrame(Stream str, bool update = false)
        {
            BinaryReader br = new BinaryReader(str);

            // read properties
            isDeformable = br.ReadBoolean();
            isIndenter = br.ReadBoolean();
            isFloor = br.ReadBoolean();
            bool _hide = br.ReadBoolean(); // discard
            name = br.ReadString();

            int nNodes = br.ReadInt32();
            int nElems = br.ReadInt32();
            int nCZs = br.ReadInt32();
            int nFaces = br.ReadInt32();
            int nEdges = br.ReadInt32();

            // make sure that List<> objects are filled with proper # of elements
            if (!update)
            {
                nodes.Clear();
                nodes.Capacity = nNodes;
                for (int i = 0; i < nNodes; i++) nodes.Add(new Node());

                elems.Clear();
                elems.Capacity = nElems;
                for (int i = 0; i < nElems; i++) elems.Add(new Element());

                czs.Clear();
                czs.Capacity = nCZs;
                for (int i = 0; i < nCZs; i++) czs.Add(new CZ());

                faces.Clear();
                faces.Capacity = nFaces;
                for (int i = 0; i < nFaces; i++) faces.Add(new Face());

                edges.Clear();
                edges.Capacity = nEdges;
                for (int i = 0; i < nEdges; i++) edges.Add(new GranuleEdge());
            } else
            {
//                Debug.Assert(nNodes == nodes.Count && nElems == elems.Count && nCZs == czs.Count && 
 //                   nFaces == faces.Count && nEdges == edges.Count, "mesh update: incorrect count");
                Debug.Assert(nNodes == nodes.Count, "mesh update: incorrect node count");
                Debug.Assert(nElems == elems.Count, "mesh update: incorrect element count");
                Debug.Assert(nFaces == faces.Count, "mesh update: incorrect face count");
                Debug.Assert(nEdges == edges.Count, "mesh update: incorrect edge count");
                Debug.Assert(nCZs == czs.Count, "mesh update: incorrect czs count");
            }

            LoadSaveMemAlloc();
            by_snapshot = br.ReadBytes(byteSize);

            // convert byte array to bool, int and double arrays
            Buffer.BlockCopy(by_snapshot, 0, isn, 0, iSize * sizeof(int));
            int offset = iSize * sizeof(int);
            Buffer.BlockCopy(by_snapshot, offset, bsn, 0, bSize * sizeof(bool));
            offset += bSize * sizeof(bool);
            Buffer.BlockCopy(by_snapshot, offset, dsn, 0, dSize * sizeof(double));

            // restore nodes
            int i_off = 0, d_off = 0, b_off = 0;

            Parallel.For(0, nNodes, i => {
                Node nd = nodes[i];
                nd.id = i;
                nd.x0 = dsn[i * ndd_str + 0];
                nd.y0 = dsn[i * ndd_str + 1];
                nd.z0 = dsn[i * ndd_str + 2];
                nd.ux = dsn[i * ndd_str + 3];
                nd.uy = dsn[i * ndd_str + 4];
                nd.uz = dsn[i * ndd_str + 5];
                nd.vx = dsn[i * ndd_str + 6];
                nd.vy = dsn[i * ndd_str + 7];
                nd.vz = dsn[i * ndd_str + 8];
                nd.ax = dsn[i * ndd_str + 9];
                nd.ay = dsn[i * ndd_str + 10];
                nd.az = dsn[i * ndd_str + 11];
                nd.fx = nd.fy = nd.fz = 0;
                nd.cx = nd.x0 + nd.ux;
                nd.cy = nd.y0 + nd.uy;
                nd.cz = nd.z0 + nd.uz;
                nd.tx = nd.ty = nd.tz = nd.unx = nd.uny = nd.unz = 0;
            });
            d_off += ndd_str * nodes.Count;

            // elements
            Parallel.For(0, nElems, i => {
                Element elem = elems[i];
                for (int j = 0; j < 4; j++) elem.vrts[j] = nodes[isn[i * eli_str + j]];
                elem.granule = isn[i * eli_str + 4];
            });
            i_off += eli_str * elems.Count;

            // cohesive zones
            Parallel.For(0, nCZs, i => {
                CZ cz = czs[i];
                for (int j = 0; j < 6; j++) cz.vrts[j] = nodes[isn[i_off + i * czi_str + j]];
                cz.faces[0] = faces[isn[i_off + i * czi_str + 6]];
                cz.faces[1] = faces[isn[i_off + i * czi_str + 7]];
                cz.immutableID = isn[i_off + i * czi_str + 8];

                cz.pmax[0] = dsn[d_off + i * czd_str + 0];
                cz.pmax[1] = dsn[d_off + i * czd_str + 1];
                cz.pmax[2] = dsn[d_off + i * czd_str + 2];
                cz.tmax[0] = dsn[d_off + i * czd_str + 3];
                cz.tmax[1] = dsn[d_off + i * czd_str + 4];
                cz.tmax[2] = dsn[d_off + i * czd_str + 5];
                cz.avgDn = dsn[d_off + i * czd_str + 6];
                cz.avgDt = dsn[d_off + i * czd_str + 7];
                cz.avgTn = dsn[d_off + i * czd_str + 8];
                cz.avgTt = dsn[d_off + i * czd_str + 9];
                cz.maxAvgDn = dsn[d_off + i * czd_str + 10];
                cz.maxAvgDt = dsn[d_off + i * czd_str + 11];

                cz.failed = bsn[b_off + i + 0];
            });
            b_off += czb_str * czs.Count;
            i_off += czi_str * czs.Count;
            d_off += czd_str * czs.Count;

            // faces
            Parallel.For(0, nFaces, i => {
                Face f = faces[i];
                f.id = i;
                for(int j=0;j<3;j++) f.vrts[j] = nodes[isn[i_off + i * fi_str + j]];
                f.granule = isn[i_off + i * fi_str + 3];
                f.exposed = bsn[b_off + i * fb_str + 0];
                f.created = bsn[b_off + i * fb_str + 1];
                int elem_id = isn[i_off + i * fi_str + 4];
                f.elem = elem_id < 0 ? null : elems[elem_id];
            });
            b_off += fb_str * faces.Count;
            i_off += fi_str * faces.Count;

            // edges
            Parallel.For(0, nEdges, i => {
                GranuleEdge ge = edges[i];
                ge.vrts[0] = nodes[isn[i_off + i * gei_str + 0]];
                ge.vrts[1] = nodes[isn[i_off + i * gei_str + 1]];
                ge.exposed = bsn[b_off + i];
            });

            if (!update)
            {
                BoundingBox();
                ComputeVolume();
            }
            IFormatter bf = new BinaryFormatter();
            surfaceFragments = (List<SurfaceFragment>)bf.Deserialize(str);
            object tcollObject = bf.Deserialize(str);
            translationCollection = (TranslationCollection)tcollObject;
            foreach (SurfaceFragment sf in surfaceFragments) { sf.allFaces = faces; sf.ComputeArea(); }
        }
        #endregion

        #region import MSH and GEO files

        void LoadGeo(Stream str)
        {
            Trace.WriteLine("LoadGeo");
            StreamReader sr = new StreamReader(str);
            String s;

            // zero node is origin
            int count = 0;

            // read node definitions of the form
            // ND, 74216,     0.012593053     0.010786273    -0.014337613
            // regular expression to match a decimal fraction
            string sn = @"(-?\d*\.?\d*E?-?\d*)";
            Regex el = new Regex(@"\sND,\s(\d+),\s+" + sn + @"\s+" + sn + @"\s+" + sn);
            Match match;
            // read header until encountering ND definitions
            Console.WriteLine("loading mesh nodes");
            do
            {
                s = sr.ReadLine();
                Console.WriteLine("reading header line: {0}", s);
            } while (!el.IsMatch(s));

            // parse nodes
            do
            {
                count++;
                match = el.Match(s);
                int idx = 0;
                try
                {
                    idx = Int32.Parse(match.Groups[1].Value);
                    double x, y, z;
                    x = Double.Parse(match.Groups[2].Value);
                    y = Double.Parse(match.Groups[3].Value);
                    z = Double.Parse(match.Groups[4].Value);
                    nodes.Add(new Node(x, -z, y, nodes.Count));
                }
                catch
                {
                    Console.WriteLine("idx {0}, x {1}, y {2}, z {3}", match.Groups[1].Value, match.Groups[2].Value, match.Groups[3].Value, match.Groups[4].Value);
                }
                if (count != idx) throw new Exception("element index is not in sequence");
                s = sr.ReadLine();
            } while (el.IsMatch(s));

            el = new Regex(@"\sEL,\s+(\d+),\s+VL\s+\d+,\s+4\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)");
            // read a few lines that separate ND and EL
            Console.WriteLine("read a few lines that separate ND and EL");
            do
            {
                s = sr.ReadLine();
                Console.WriteLine("reading header line: {0}", s);
            } while (!el.IsMatch(s));

            List<int> _surfaceInfo = new List<int>();
            // read EL
            // EL,     1,  VL      1,  10  3214  3215  3216  3217 19727 19728 19729 19730 19731 19732     0     0     0     0    -1     0
            //  EL, 150241,  VL      1,   4 97777 88611 82856 74225     0     0     0     0     0     0
            do
            {
                match = el.Match(s);
                Element elem = new Element();
                int n0 = Int32.Parse(match.Groups[2].Value) - 1;
                int n1 = Int32.Parse(match.Groups[3].Value) - 1;
                int n2 = Int32.Parse(match.Groups[4].Value) - 1;
                int n3 = Int32.Parse(match.Groups[5].Value) - 1;

                elem.vrts[0] = nodes[n0];
                elem.vrts[1] = nodes[n1];
                elem.vrts[2] = nodes[n2];
                elem.vrts[3] = nodes[n3];
                elems.Add(elem);

                for (int i = 6; i <= 9; i++) _surfaceInfo.Add(Int32.Parse(match.Groups[i].Value));
                s = sr.ReadLine();
            } while (el.IsMatch(s));

            str.Close();

            // parse _surfaceInfo to create faces
            for (int i = 0; i < elems.Count; i++)
            {
                Element elem = elems[i];
                int s0 = _surfaceInfo[i * 4 + 0];
                int s1 = _surfaceInfo[i * 4 + 1];
                int s2 = _surfaceInfo[i * 4 + 2];
                int s3 = _surfaceInfo[i * 4 + 3];

                if (s2 != 0)
                {
                    Face f = new Face();
                    f.vrts[0] = elem.vrts[1];
                    f.vrts[1] = elem.vrts[2];
                    f.vrts[2] = elem.vrts[3];
                    f.tag = s2;
                    faces.Add(f);
                }
                if (s3 != 0)
                {
                    Face f = new Face();
                    f.vrts[0] = elem.vrts[2];
                    f.vrts[1] = elem.vrts[0];
                    f.vrts[2] = elem.vrts[3];
                    f.tag = s3;
                    faces.Add(f);
                }
                if (s1 != 0)
                {
                    Face f = new Face();
                    f.vrts[0] = elem.vrts[3];
                    f.vrts[1] = elem.vrts[0];
                    f.vrts[2] = elem.vrts[1];
                    f.tag = s1;
                    faces.Add(f);
                }
                if (s0 != 0)
                {
                    Face f = new Face();
                    f.vrts[0] = elem.vrts[2];
                    f.vrts[1] = elem.vrts[1];
                    f.vrts[2] = elem.vrts[0];
                    f.tag = s0;
                    faces.Add(f);
                }
            }

            #region surfaceFragments
            // find all distinct face tags and number faces sequentially
            HashSet<int> distinctTags = new HashSet<int>();
            for (int i=0;i<faces.Count;i++)
            {
                faces[i].id = i;
                distinctTags.Add(faces[i].tag);
            }

            // map tags to sequential numbers, create fragments
            Dictionary<int, int> mapping = new Dictionary<int, int>();
            count = 0;
            foreach(int distinctTag in distinctTags)
            {
                mapping.Add(distinctTag, count);
                count++;
                surfaceFragments.Add(new SurfaceFragment() { id=count});
            }

            // add faces to corresponding fragments
            foreach (Face f in faces)
            {
                int fragmentId = mapping[f.tag];
                surfaceFragments[fragmentId].faces.Add(f.id);
            }
            foreach (SurfaceFragment sf in surfaceFragments) { sf.allFaces = faces; sf.ComputeArea(); }
            // 

            #endregion

            BoundingBox();
            ComputeVolume();
            IdentifySurfaceElements();
        }

        void LoadMsh(Stream str)
        {
            isDeformable = true;
            StreamReader sr = new StreamReader(str);
            string s;
            do { s = sr.ReadLine(); } while (s != "$Nodes");
            // read nodes
            int nNodes = Int32.Parse(sr.ReadLine());

            //12817 0.074644582223 0.087614145994 0.046097970400
            //$EndNodes
            s = sr.ReadLine();
            while (s != "$EndNodes")
            {
                string[] parts = s.Split(' ');
                if (parts.Length < 3)
                {
                    s = sr.ReadLine(); continue;
                }
                // int n = int.Parse(parts[0]);
                double x = double.Parse(parts[1]);
                double y = double.Parse(parts[2]);
                double z = double.Parse(parts[3]);
                nodes.Add(new Node(x, y, z, nodes.Count));
                s = sr.ReadLine();
            }

            //$Elements
            //89339
            if (sr.ReadLine() != "$Elements") throw new Exception("no $Elements tag");
            int nElemTotal = int.Parse(sr.ReadLine());

            //1 15 3 0 1 0 1
            //3718 1 3 0 1017 0 446 2702
            //16544 2 3 0 484 0 6283 1331 140
            //89339 4 3 0 100 0 6908 7749 6917 6902 
            int n0, n1, n2, n3, tag;
            s = sr.ReadLine();
            while (s != "$EndElements")
            {
                string[] parts = s.Split(' ');
                switch (int.Parse(parts[1]))
                {
                    case 2: // triangle
                        n0 = int.Parse(parts[6]) - 1;
                        n1 = int.Parse(parts[7]) - 1;
                        n2 = int.Parse(parts[8]) - 1;
                        Face f = new Face();
                        f.vrts[0] = nodes[n0];
                        f.vrts[1] = nodes[n1];
                        f.vrts[2] = nodes[n2];
                        faces.Add(f);
                        break;
                    case 4: // element
                        tag = int.Parse(parts[4]) - 1;
                        n0 = int.Parse(parts[6]) - 1;
                        n1 = int.Parse(parts[7]) - 1;
                        n2 = int.Parse(parts[8]) - 1;
                        n3 = int.Parse(parts[9]) - 1;
                        Element elem = new Element();
                        elem.vrts[0] = nodes[n0];
                        elem.vrts[1] = nodes[n1];
                        elem.vrts[2] = nodes[n2];
                        elem.vrts[3] = nodes[n3];
                        elem.granule = tag;
                        elems.Add(elem);
                        break;
                    case 1:
                        n0 = int.Parse(parts[6]) - 1;
                        n1 = int.Parse(parts[7]) - 1;
                        GranuleEdge ge = new GranuleEdge();
                        ge.vrts[0] = nodes[n0];
                        ge.vrts[1] = nodes[n1];
                        edges.Add(ge);
                        break;
                }
                s = sr.ReadLine();
            }
            str.Close();
            BoundingBox();
            ComputeVolume();
            DetectSurfacesAfterLoadingMSH();
        }

        public void DetectSurfacesAfterLoadingMSH()
        {
            surfaceFragments.Clear();
            // sequential ids for faces
            for (int i = 0; i < faces.Count; i++) faces[i].id = i;

            for (int i = 0; i < 6; i++) surfaceFragments.Add(new SurfaceFragment() { id = i });

            foreach(Face f in faces)
            {
                if (f.vrts.All(nd => nd.z0 == zmax)) surfaceFragments[0].faces.Add(f.id);
                else if (f.vrts.All(nd => nd.z0 == zmin)) surfaceFragments[1].faces.Add(f.id);
                else if (f.vrts.All(nd => nd.x0 == xmax)) surfaceFragments[2].faces.Add(f.id);
                else if (f.vrts.All(nd => nd.x0 == xmin)) surfaceFragments[3].faces.Add(f.id);
                else if (f.vrts.All(nd => nd.y0 == ymax)) surfaceFragments[4].faces.Add(f.id);
                else if (f.vrts.All(nd => nd.y0 == ymin)) surfaceFragments[5].faces.Add(f.id);
            }
            foreach (SurfaceFragment sf in surfaceFragments) { sf.allFaces = faces; sf.ComputeArea(); }
            surfaceFragments[0].sensor = true;
            surfaceFragments[0].role = SurfaceFragment.SurfaceRole.Anchored;
            surfaceFragments[0].dz = -0.23;
            surfaceFragments[1].role = SurfaceFragment.SurfaceRole.Anchored;
        }

        #endregion

        #region other
        void ComputeVolume()
        {
            _volume = 0;
            foreach (Element elem in elems) _volume += elem.volume;
        }

        public void IdentifySurfaceElements()
        {
            // Populate surfaceElements.
            foreach (Face f in faces) foreach (Node nd in f.vrts) nd.isSurface = true;
            foreach (Element elem in elems) if (elem.vrts.Any(nd => nd.isSurface)) elem.isSurface = true;
            surfaceElements = elems.FindAll(elem => elem.isSurface);
            Trace.WriteLine($"mesh {this.name}; surface element count {surfaceElements.Count}");
        }

        public void ConnectFaces()
        {
            foreach (Node nd in nodes) nd.faces.Clear();
            foreach (Face f in faces) foreach (Node nd in f.vrts) nd.faces.Add(f);
        }
        #endregion

        #region geometry transformation tools

        public void FlipYZ()
        {
            foreach (Node nd in nodes)
            {
                double y = nd.y0;
                nd.y0 = nd.cy = nd.z0;
                nd.z0 = nd.cz = -y;
            }
            BoundingBox();
        }

        public void CenterSample()
        {
            double x_center = (xmax + xmin) / 2D;
            double y_center = (ymax + ymin) / 2D;
            double scale = 1;// 0.1D / (xmax - xmin);
            foreach (Node nd in nodes)
            {
                nd.x0 = (nd.x0 - x_center) * scale;
                nd.y0 = (nd.y0 - y_center) * scale;
                nd.z0 *= scale;
                nd.cx = nd.x0;
                nd.cy = nd.y0;
                nd.cz = nd.z0;
            }
            BoundingBox();
        }

        public void TranslateZ(double dz)
        {
            foreach (Node nd in nodes)
            {
                nd.z0 += dz;
                nd.cz = nd.z0;
            }
            BoundingBox();
        }

        public void Resize(double scale)
        {
            foreach (Node nd in nodes)
            {
                nd.x0 *= scale;
                nd.y0 *= scale;
                nd.z0 *= scale;
                nd.cx = nd.x0;
                nd.cy = nd.y0;
                nd.cz = nd.z0;
            }
            BoundingBox();
        }

        public void Rotate90Deg()
        {
            foreach (Node nd in nodes)
            {
                double oldX = nd.x0;
                double oldY = nd.y0;
                nd.x0 = oldY;
                nd.y0 = -oldX;
                nd.cx = nd.x0;
                nd.cy = nd.y0;
                nd.cz = nd.z0;
            }
            BoundingBox();
        }

        #endregion


    }
}
