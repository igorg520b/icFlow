using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK;
using System.Drawing;

namespace icFlow
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

        public static void RenderSurface(this Face[] faces, bool transparent = true, bool showCreated = false, bool onlyExposed = false)
        {
            byte transparency = (byte)55;
            GL.Enable(EnableCap.DepthTest);
            if (!transparent)
            {
                GL.DepthFunc(DepthFunction.Lequal);
                GL.Disable(EnableCap.Blend);
            }
            else if (transparent || showCreated)
            {
                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
                GL.DepthMask(false);
            }

            GL.Begin(PrimitiveType.Triangles);

            foreach (Face f in faces)
            {
                if ((showCreated && !f.created) || (onlyExposed && !f.exposed)) continue;
                if (!transparent) GL.Color3(colors[f.granule % 17]);
                else
                {
                    Color cl = colors[f.granule % 17];
                    GL.Color4(cl.R, cl.G, cl.B, transparency);
                }
                f.RenderTriangle();
            }
            GL.End();

            if (transparent || showCreated)
            {
                GL.DepthMask(true);
                GL.Enable(EnableCap.DepthTest);
                GL.DepthFunc(DepthFunction.Lequal);
                GL.DrawBuffer(DrawBufferMode.None);
                GL.Begin(PrimitiveType.Triangles);
                foreach (Face f in faces)
                {
                    if ((showCreated && !f.created) || (onlyExposed && !f.exposed)) continue;
                    f.RenderTriangle();
                }
                GL.End();

                GL.DrawBuffer(DrawBufferMode.Back);
            }
        }

        public static void RenderFaceEdges(this Mesh mg, bool transparent = true)
        {
            if (mg.hide) return;
            GL.Disable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Lequal);
            /*            if (!transparent) {
                        else
                        {
                            GL.Disable(EnableCap.DepthTest);
                            GL.Enable(EnableCap.Blend);
                            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
                        }
                        */
            GL.LineWidth(1f);
            GL.Disable(EnableCap.Lighting);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.4f, 0.41f, 0.39f);
            foreach (Face f in mg.faces) f.RenderWireFrame();
            GL.End();
        }

        public static void RenderGranuleEdges(this GranuleEdge[] edges)
        {
            GL.Enable(EnableCap.LineSmooth);
            GL.Hint(HintTarget.LineSmoothHint, HintMode.Nicest);
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.Disable(EnableCap.Lighting);
            GL.LineWidth(2.0f);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.35f, 0.31f, 0.38f);

            foreach (GranuleEdge ge in edges) ge.Render();
            GL.End();

        }

        public static void RenderGranuleEdges(this Mesh mg, float width = 2f)
        {
            if (mg.hide) return;
            GL.Enable(EnableCap.LineSmooth);
            GL.Hint(HintTarget.LineSmoothHint, HintMode.Nicest);
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.Disable(EnableCap.Lighting);
            GL.LineWidth(width);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.35f, 0.31f, 0.38f);

            foreach (GranuleEdge ge in mg.edges) if (ge.exposed) ge.Render();
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

        public static void RenderCZ(this CZ[] czs)
        {
            GL.Enable(EnableCap.Lighting);
            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.Blend);

            GL.Begin(PrimitiveType.Triangles);

            foreach (CZ cz in czs) cz.Render();
            GL.End();

            GL.Disable(EnableCap.Lighting);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.11f, 0.14f, 0.09f);

            foreach (CZ cz in czs) cz.RenderWireframe();
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
            if (cz.failed) GL.Color3(0.9f, 0.1f, 0.1f);
            else if (cz.damaged) GL.Color3(0.8f, 0.8f, 0.8f);
            else return;

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

        public static void RenderWireframe(this CZ cz)
        {
            if (!cz.failed && !cz.damaged) return;

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

        /*
        public Vector3d cCenter
        {
            get
            {
                double x, y, z;
                x = y = z = 0;
                foreach (Node nd in vrts) { x += nd.cx; y += nd.cy; z += nd.cz; }
                x /= 4; y /= 4; z /= 4;
                return new Vector3d(x, y, z);
            }
        }
        */
        #endregion

        #region Collision

        public static void RenderCollisionTriangle(Node n0, Node n1, Node n2)
        {
            // similar to failed CZ; draw filled triangles with color
            Vector3d v0 = new Vector3d(n0.cx, n0.cy, n0.cz);
            Vector3d v1 = new Vector3d(n1.cx, n1.cy, n1.cz);
            Vector3d v2 = new Vector3d(n2.cx, n2.cy, n2.cz);
            GL.Normal3(Vector3d.Cross(v1 - v0, v2 - v0));
            GL.Vertex3(v2); GL.Vertex3(v0); GL.Vertex3(v1);
        }

        #endregion
    }
}
