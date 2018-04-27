using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK;
using System.Drawing;
using System.Diagnostics;

namespace icFlow.Rendering
{
    // This class contains extensions for rendering Nodes, Faces, CZs, etc.
    // This functionality is defined in this assembly to separate platform-specific 
    //  functions that use OpenTK (so that v1Library could run in console mode without OpenTK).
    public static class RenderingExtensions
    {
        static RenderingExtensions()
        {
            InitializeColors();
        }


        #region Node
        public static void Render(this Node nd)
        {
            GL.Vertex3(nd.cx, nd.cy, nd.cz);
        }
        #endregion

        #region Face
        public static Color[] groupColors = { Color.Wheat, Color.Red, Color.Green, Color.Blue, Color.Violet, Color.Yellow, Color.Salmon, Color.Orange, Color.Orchid, Color.PaleTurquoise };

        public static void RenderWireFrame(this Face f)
        {
            Node n0 = f.vrts[0];
            Node n1 = f.vrts[1];
            Node n2 = f.vrts[2];
            Vector3d v0 = new Vector3d(n0.cx, n0.cy, n0.cz);
            Vector3d v1 = new Vector3d(n1.cx, n1.cy, n1.cz);
            Vector3d v2 = new Vector3d(n2.cx, n2.cy, n2.cz);
            GL.Vertex3(v0); GL.Vertex3(v1);
            GL.Vertex3(v0); GL.Vertex3(v2);
            GL.Vertex3(v2); GL.Vertex3(v1);
        }

        public static void RenderTriangle(this Face f)
        {
            Node n0 = f.vrts[0];
            Node n1 = f.vrts[1];
            Node n2 = f.vrts[2];
            Vector3d v0 = new Vector3d(n0.cx, n0.cy, n0.cz);
            Vector3d v1 = new Vector3d(n1.cx, n1.cy, n1.cz);
            Vector3d v2 = new Vector3d(n2.cx, n2.cy, n2.cz);
            GL.Normal3(Vector3d.Cross(v1 - v0, v2 - v0));
            GL.Vertex3(v2); GL.Vertex3(v0); GL.Vertex3(v1);
        }

        public static void RenderTriangleTentative(this Face f)
        {
            Vector3d v0 = new Vector3d(f.vrts[0].tx, f.vrts[0].ty, f.vrts[0].tz);
            Vector3d v1 = new Vector3d(f.vrts[1].tx, f.vrts[1].ty, f.vrts[1].tz);
            Vector3d v2 = new Vector3d(f.vrts[2].tx, f.vrts[2].ty, f.vrts[2].tz);
            GL.Normal3(Vector3d.Cross(v1 - v0, v2 - v0));
            GL.Vertex3(v2); GL.Vertex3(v0); GL.Vertex3(v1);
        }
        #endregion

        #region Mesh

        static Color[] colors;
        static void InitializeColors()
        {
            colors = new Color[17];
            //            colors[0] = Color.FromArgb(244, 154, 194);
            colors[0] = Color.White;
            colors[1] = Color.FromArgb(203, 153, 201);
            colors[2] = Color.FromArgb(194, 59, 34);
            colors[3] = Color.FromArgb(255, 209, 220);
            colors[4] = Color.FromArgb(222, 165, 164);
            colors[5] = Color.FromArgb(174, 198, 207);
            colors[6] = Color.FromArgb(119, 190, 119);
            colors[7] = Color.FromArgb(207, 207, 196);
            colors[8] = Color.FromArgb(179, 158, 181);
            colors[9] = Color.FromArgb(255, 179, 71);
            colors[10] = Color.FromArgb(100, 20, 100);
            colors[11] = Color.FromArgb(255, 105, 97);
            colors[12] = Color.FromArgb(3, 192, 60);
            colors[13] = Color.FromArgb(253, 253, 150);
            colors[14] = Color.FromArgb(130, 105, 83);
            colors[15] = Color.FromArgb(119, 158, 203);
            colors[16] = Color.FromArgb(150, 111, 214);
        }
        
        // render exposed, non-created surfaces with transparency
        public static void RenderSurface_exposed(this List<Face> faces, 
            float transparencyCoeff, 
            bool useGrainColor, Color surfaceColor)
        {
            byte transparency = (byte)(255f*transparencyCoeff);
            bool transparent = transparencyCoeff != 1;

            if (transparent)
            {
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
                GL.DepthMask(false);
                GL.Disable(EnableCap.Lighting);

                GL.Begin(PrimitiveType.Triangles);
                if (!useGrainColor) { GL.Color4(surfaceColor.R, surfaceColor.G, surfaceColor.B, transparency); }

                foreach (Face f in faces)
                {
                    if (f.exposed && !f.created)
                    {
                        if (useGrainColor)
                        {
                            Color col = colors[f.granule % 17];
                            GL.Color4(col.R,col.G,col.B,transparency);
                        }
                        f.RenderTriangle();
                    }
                }
                GL.End();

                GL.DepthMask(true);
                GL.Enable(EnableCap.DepthTest);
                GL.DepthFunc(DepthFunction.Lequal);
                GL.DrawBuffer(DrawBufferMode.None);
                GL.Begin(PrimitiveType.Triangles);
                foreach (Face f in faces)
                    if (f.exposed && !f.created) f.RenderTriangle();
                GL.End();
                GL.DrawBuffer(DrawBufferMode.Back);
            }
            else
            {
                GL.Enable(EnableCap.Lighting);
                GL.Disable(EnableCap.Blend);
                GL.Enable(EnableCap.DepthTest);
                GL.DepthFunc(DepthFunction.Less);
                GL.DepthMask(true);

                GL.Begin(PrimitiveType.Triangles);
                if(!useGrainColor) GL.Color3(surfaceColor);
                foreach (Face f in faces)
                {
                    if (f.exposed && !f.created)
                    {
                        if (useGrainColor) GL.Color3(colors[f.granule % 17]);
                        f.RenderTriangle();
                    }
                }
                GL.End();

            }

        }

        // render created surfaces, opaque
        public static void RenderSurface_created(this Face[] faces, Color createdColor)
        {
            GL.Enable(EnableCap.Lighting);
            GL.Disable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Less);
            GL.DepthMask(true);

            GL.Begin(PrimitiveType.Triangles);
            GL.Color3(createdColor);
            foreach (Face f in faces) if (f.created) f.RenderTriangle();
            GL.End();
        }

        public static void RenderEdges_created(this Face[] faces, Color edgeColor, float edgeWidth)
        {
            GL.Disable(EnableCap.Lighting);
            GL.Disable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.DepthMask(true);

            GL.LineWidth(edgeWidth);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(edgeColor);
            foreach (Face f in faces) if (f.created) f.RenderWireFrame();
            GL.End();
        }

        public static void RenderEdges_exposed(this List<Face> faces, Color edgeColor, float edgeWidth)
        {
            GL.Disable(EnableCap.Lighting);
            GL.Enable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.DepthMask(true);

            GL.LineWidth(edgeWidth);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(edgeColor);
            foreach (Face f in faces) if (f.exposed) f.RenderWireFrame();
            GL.End();
        }

        public static void RenderEdges_all(this List<Face> faces, Color edgeColor, float edgeWidth)
        {
            GL.Disable(EnableCap.Lighting);
            GL.Enable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.DepthMask(true);

            GL.LineWidth(edgeWidth);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(edgeColor);
            foreach (Face f in faces) f.RenderWireFrame();
            GL.End();
        }

        public static void RenderGrainEdges(this GranuleEdge[] edges,
            float width, Color color)
        {
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.Disable(EnableCap.Lighting);
            GL.LineWidth(width);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(color);
            foreach (GranuleEdge ge in edges) ge.Render();
            GL.End();
        }

        public static void RenderGrainEdgesAsCylinders(this GranuleEdge[] edges,
    Color color, int nSides, double radius)
        {
            GL.Enable(EnableCap.PolygonSmooth);
            GL.Hint(HintTarget.PolygonSmoothHint, HintMode.Nicest);

            Vector3d[] outline2d = new Vector3d[nSides];
            Vector3d[] outline2db = new Vector3d[nSides];
            for (int i = 0; i < nSides; i++) outline2d[i] = new Vector3d(radius * Math.Sin(Math.PI * 2 * i / nSides), radius * Math.Cos(Math.PI * 2 * i / nSides), 0);
            for (int i = 0; i < nSides; i++) outline2db[i] = new Vector3d(radius*0.5 * Math.Sin(Math.PI * 2 * (i+0.5f) / nSides), radius *0.5* Math.Cos(Math.PI * 2 * (i+0.5f) / nSides), 0);
            Vector3d[] edgeVertices = new Vector3d[nSides * 2];
            Vector3d[] edgeVertices2 = new Vector3d[nSides * 2];

            GL.Enable(EnableCap.DepthTest);
            GL.DepthMask(false);

            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.Disable(EnableCap.Lighting);

            GL.Disable(EnableCap.Blend);
            GL.Begin(PrimitiveType.Quads);
            GL.Color3(color);
            foreach (GranuleEdge ge in edges)
            {
                // initialize the values of edgeVertices
                Node fromNode = ge.vrts[0];
                Node toNode = ge.vrts[1];
                Vector3d fromVec = new Vector3d(fromNode.cx, fromNode.cy, fromNode.cz);
                Vector3d toVec = new Vector3d(toNode.cx, toNode.cy, toNode.cz);

                Vector3d x, y, z; // new coordinate frame, whith z pointing along the edge direction
                z = toVec - fromVec;
                z.Normalize();
                x = Vector3d.Cross(Vector3d.UnitX, z);
                if (x.LengthSquared == 0) x = Vector3d.Cross(Vector3d.UnitY, z);
                x.Normalize();
                y = Vector3d.Cross(z, x);
                y.Normalize();

                for (int i = 0; i < nSides; i++)
                {
                    Vector3d o = outline2db[i];
                    Vector3d rotated = x * o.X + y * o.Y;// + z * o.Z;
                    edgeVertices2[i * 2] = rotated + fromVec;
                    edgeVertices2[i * 2 + 1] = rotated + toVec;
                }

                // render grains as prisms that consist of quads
                for (int i = 0; i < nSides; i++)
                {
                    Vector3d s1n1, s1n2, s2n1, s2n2;
                    int idx1 = i;
                    int idx2 = (i + 1) % nSides;
                    s1n1 = edgeVertices2[idx1 * 2];
                    s1n2 = edgeVertices2[idx2 * 2];
                    s2n1 = edgeVertices2[idx1 * 2 + 1];
                    s2n2 = edgeVertices2[idx2 * 2 + 1];
                    GL.Vertex3(s1n1);
                    GL.Vertex3(s2n2);
                    GL.Vertex3(s2n1);
                    GL.Vertex3(s1n2);
                }
            }
            GL.End();


            GL.Enable(EnableCap.Blend);

            GL.Begin(PrimitiveType.Quads);
            GL.Color3(color);
            foreach (GranuleEdge ge in edges) {
                // initialize the values of edgeVertices
                Node fromNode = ge.vrts[0];
                Node toNode = ge.vrts[1];
                Vector3d fromVec = new Vector3d(fromNode.cx, fromNode.cy, fromNode.cz);
                Vector3d toVec = new Vector3d(toNode.cx, toNode.cy, toNode.cz);

                Vector3d x, y, z; // new coordinate frame, whith z pointing along the edge direction
                z = toVec - fromVec;
                z.Normalize();
                x = Vector3d.Cross(Vector3d.UnitX, z);
                if (x.LengthSquared == 0) x = Vector3d.Cross(Vector3d.UnitY, z);
                x.Normalize();
                y = Vector3d.Cross(z, x);
                y.Normalize();

                for (int i = 0; i < nSides; i++)
                {
                    Vector3d o = outline2d[i];
                    Vector3d rotated = x * o.X + y * o.Y;// + z * o.Z;
                    edgeVertices[i * 2] = rotated + fromVec;
                    edgeVertices[i * 2 + 1] = rotated + toVec;
                }

                // render grains as prisms that consist of quads
                for (int i = 0; i < nSides; i++)
                {
                    Vector3d s1n1, s1n2, s2n1, s2n2;
                    int idx1 = i;
                    int idx2 = (i + 1) % nSides;
                    s1n1 = edgeVertices[idx1 * 2];
                    s1n2 = edgeVertices[idx2 * 2];
                    s2n1 = edgeVertices[idx1 * 2 + 1];
                    s2n2 = edgeVertices[idx2 * 2 + 1];
                    GL.Vertex3(s1n1);
                    GL.Vertex3(s1n2);
                    GL.Vertex3(s2n2);
                    GL.Vertex3(s2n1);
                }
            }
            GL.End();
            
        }

        public static void RenderWireFrame(this Mesh mg)
        {
            if (mg.hide) return;
            // only draw tetrahedral elements on the surface of the mesh
            if (mg.surfaceElements == null) return;
            GL.LineWidth(1f);
            GL.DepthFunc(DepthFunction.Always);
            GL.Disable(EnableCap.Lighting);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.9f, 0.21f, 0.29f);
            foreach (Element elem in mg.surfaceElements) elem.RenderWireFrame();
            GL.End();
        }

        public static void RenderCZ_Fill(this CZ[] czs, Color colorFill, bool renderFailed,
            double nThreshold, double tThreshold)
        {
            GL.DepthMask(true);
            GL.Enable(EnableCap.Lighting);
            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.Blend);
            GL.Begin(PrimitiveType.Triangles);
            GL.Color3(colorFill);
            // render either damaged or failed CZs
            foreach (CZ cz in czs)
                if ((renderFailed && cz.failed) || (!renderFailed && !cz.failed && 
                    cz.damagedAtLevel(nThreshold, tThreshold)))
                    cz.Render();
            GL.End();
        }

        public static void RenderCZ_Edges(this CZ[] czs, Color colorEdge, 
            float edgesWidth, bool renderFailed, double nThreshold, double tThreshold)
        {
            GL.Disable(EnableCap.Blend);
            GL.Disable(EnableCap.Lighting);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.LineWidth(edgesWidth);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(colorEdge);

            foreach (CZ cz in czs)
                if ((renderFailed && cz.failed) || (!renderFailed && !cz.failed && cz.damagedAtLevel(nThreshold, tThreshold)))
                    cz.RenderWireframe(nThreshold, tThreshold);
            GL.End();
        }

        #endregion

        #region GranuleEdge
        public static void Render(this GranuleEdge ge)
        {
            Node n0 = ge.vrts[0];
            Node n1 = ge.vrts[1];
            Vector3d v0 = new Vector3d(n0.cx, n0.cy, n0.cz);
            Vector3d v1 = new Vector3d(n1.cx, n1.cy, n1.cz);
            GL.Vertex3(v0); GL.Vertex3(v1);
        }
        #endregion

        #region CZ

        public static void Render(this CZ cz)
        {

            double[] x = new double[3];
            double[] y = new double[3];
            double[] z = new double[3];

            for (int i = 0; i < 3; i++)
            {
                Node ndx1 = cz.vrts[i];
                Node ndx2 = cz.vrts[i + 3];
                x[i] = (ndx1.cx + ndx2.cx) * 0.5;
                y[i] = (ndx1.cy + ndx2.cy) * 0.5;
                z[i] = (ndx1.cz + ndx2.cz) * 0.5;
            }
            Vector3d v0 = new Vector3d(x[0], y[0], z[0]);
            Vector3d v1 = new Vector3d(x[1], y[1], z[1]);
            Vector3d v2 = new Vector3d(x[2], y[2], z[2]);
            GL.Normal3(Vector3d.Cross(v1 - v0, v2 - v0));
            GL.Vertex3(v2); GL.Vertex3(v0); GL.Vertex3(v1);
        }

        public static void RenderWireframe(this CZ cz, double nThreshold, double tThreshold)
        {
            if (!cz.failed && !cz.damagedAtLevel(nThreshold, tThreshold)) return;

            double[] x = new double[3];
            double[] y = new double[3];
            double[] z = new double[3];
            for (int i = 0; i < 3; i++)
            {
                Node ndx1 = cz.vrts[i];
                Node ndx2 = cz.vrts[i + 3];
                x[i] = (ndx1.cx + ndx2.cx) * 0.5;
                y[i] = (ndx1.cy + ndx2.cy) * 0.5;
                z[i] = (ndx1.cz + ndx2.cz) * 0.5;
            }
            Vector3d v0 = new Vector3d(x[0], y[0], z[0]);
            Vector3d v1 = new Vector3d(x[1], y[1], z[1]);
            Vector3d v2 = new Vector3d(x[2], y[2], z[2]);
            GL.Vertex3(v0); GL.Vertex3(v1);
            GL.Vertex3(v0); GL.Vertex3(v2);
            GL.Vertex3(v2); GL.Vertex3(v1);
        }
        #endregion

        #region Element

        public static void RenderWireFrame(this Element elem)
        {
            Node n0 = elem.vrts[0];
            Node n1 = elem.vrts[1];
            Node n2 = elem.vrts[2];
            Node n3 = elem.vrts[3];
            Vector3d v0 = new Vector3d(n0.cx, n0.cy, n0.cz);
            Vector3d v1 = new Vector3d(n1.cx, n1.cy, n1.cz);
            Vector3d v2 = new Vector3d(n2.cx, n2.cy, n2.cz);
            Vector3d v3 = new Vector3d(n3.cx, n3.cy, n3.cz);
            GL.Vertex3(v0); GL.Vertex3(v1);
            GL.Vertex3(v0); GL.Vertex3(v2);
            GL.Vertex3(v0); GL.Vertex3(v3);
            GL.Vertex3(v2); GL.Vertex3(v1);
            GL.Vertex3(v2); GL.Vertex3(v3);
            GL.Vertex3(v1); GL.Vertex3(v3);
        }


        #endregion

    }
}
