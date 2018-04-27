using System;
using System.Drawing;
using System.Windows.Forms;
using OpenTK.Graphics.OpenGL;
using OpenTK;
using OpenTK.Graphics;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using icFlow.Rendering;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;

namespace icFlow
{
    public partial class Form1 : Form
    {
        int lastX, lastY;   // mouse last position
        double aspectRatio = 1;
        bool running = false; // computation is running in backgroundworker
        ImplicitModel3 model3 = new ImplicitModel3();
        TreeNode tnRoot, tnPrms, tnMeshCollection, tnCurrentFrame, tnRender;
        RenderPrms rprm = RenderPrms.Load();

        #region initialize

        public Form1()
        {
            model3.Initialize();
            InitializeComponent();
            glControl1.MouseWheel += GlControl1_MouseWheel;
            treeView1.NodeMouseClick += (sender, args) => treeView1.SelectedNode = args.Node;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            SetUpLight();
            trackBar1.Enabled = false;
            RebuildTree();
        }

        #endregion

        #region treeview
        private void treeView1_AfterSelect(object sender, TreeViewEventArgs e)
        {
            object tag = treeView1.SelectedNode == null ? null : treeView1.SelectedNode.Tag;
            pg.SelectedObject = tag;
            pg.Refresh();
            
        }

        void RebuildTree()
        {
            treeView1.Nodes.Clear();
            tnRoot = treeView1.Nodes.Add("sim");
            tnPrms = tnRoot.Nodes.Add("prms");
            tnRender = tnRoot.Nodes.Add("render");
            tnRender.Tag = rprm;
            tnPrms.ToolTipText = "Simulation parameters";
            tnPrms.Tag = model3.prms;
            tnMeshCollection = tnRoot.Nodes.Add("mc");
            tnMeshCollection.ToolTipText = "Mesh Collection";
            tnCurrentFrame = tnRoot.Nodes.Add("cf");
            tnCurrentFrame.ToolTipText = "Current Frame";

            foreach(Mesh msh in model3.mc.mgs)
            {
                TreeNode tn = tnMeshCollection.Nodes.Add(msh.ToString());
                tn.Tag = msh;
                tn.ContextMenuStrip = cmsMesh;
                TreeNode sfc = tn.Nodes.Add("sfc");
                sfc.ToolTipText = "Surfaces of the mesh";
                sfc.Tag = msh.surfaceFragments;
                foreach(SurfaceFragment sf in msh.surfaceFragments)
                {
                    TreeNode sfc_frag = sfc.Nodes.Add(sf.ToString());
                    sfc_frag.Tag = sf;
                }
                TreeNode transforms = tn.Nodes.Add("trs");
                transforms.Tag = msh.translationCollection;
                transforms.ContextMenuStrip = cmsTranslations;
                foreach(Translation trsl in msh.translationCollection)
                {
                    TreeNode trsl_node = transforms.Nodes.Add(trsl.ToString());
                    trsl_node.Tag = trsl;
                    trsl_node.ContextMenuStrip = cmsOneTranslation;
                }
            }

            tnRoot.Expand();
            tnMeshCollection.Expand();
        }



        #endregion

        #region glControl
        private void glControl1_Paint(object sender, PaintEventArgs e)
        {
            #region prepare
            GL.DepthMask(true);

            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);
            GL.Enable(EnableCap.LineSmooth);
            GL.Hint(HintTarget.LineSmoothHint, HintMode.Nicest);
            GL.Disable(EnableCap.PolygonSmooth);
            GL.Enable(EnableCap.Multisample);
            GL.Hint(HintTarget.MultisampleFilterHintNv, HintMode.Nicest);
            GL.Enable(EnableCap.PointSmooth);
            GL.Hint(HintTarget.PointSmoothHint, HintMode.Nicest);

            RenderBackground(rprm.WhiteBackground);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            if (rprm.UseFrustum) GL.Translate(0, 0, -rprm.zOffset);
            GL.Translate(rprm.dx / 1000, rprm.dy / 1000, 0);
            GL.Rotate(-rprm.phi, new Vector3(1, 0, 0));
            if (!tsbPhi.Checked) GL.Rotate(-rprm.theta, new Vector3(0, 0, 1));
            if (tsbRotate.Checked) GL.Rotate(90, new Vector3(0, 1, 0));

            GL.Enable(EnableCap.Lighting);
            #endregion

            SetUpLight();
            // draw czs
            if (model3.mc.failedCZs != null && rprm.FailedCZvisible && rprm.CZsFill)
                model3.mc.failedCZs.RenderCZ_Fill(rprm.FailedCZColor, true, model3.prms.nThreshold,model3.prms.tThreshold);

            if (model3.mc.nonFailedCZs != null && rprm.DamagedCZvisible && rprm.CZsFill)
                model3.mc.nonFailedCZs.RenderCZ_Fill(rprm.DamagedCZColor, false, model3.prms.nThreshold, model3.prms.tThreshold);
                
            // draw created surfaces
            if (rprm.ShowSurface && model3.mc.allFaces != null)
                model3.mc.allFaces.RenderSurface_created(rprm.CreatedSurfaceColor);

            // draw edges of creates surface
            if(rprm.ShowTetraEdgesCreated && model3.mc.allFaces != null)
                model3.mc.allFaces.RenderEdges_created(rprm.TetraEdgesColor, rprm.TetraEdgesWidth);

            // draw cz edges
            if (model3.mc.failedCZs != null && rprm.FailedCZvisible && rprm.CZsEdgesVisible)
                model3.mc.failedCZs.RenderCZ_Edges(rprm.CZsEdgesColor, rprm.CZsEdgeWidth, true, model3.prms.nThreshold, model3.prms.tThreshold);

            if (model3.mc.nonFailedCZs != null && rprm.DamagedCZvisible && rprm.CZsEdgesVisible)
                model3.mc.nonFailedCZs.RenderCZ_Edges(rprm.CZsEdgesColor, rprm.CZsEdgeWidth, false, model3.prms.nThreshold, model3.prms.tThreshold);

            // draw rigid objects
            if (model3.mc.nonDeformables != null)
                foreach (Mesh msh in model3.mc.nonDeformables)
                    if (!msh.hide)
                        msh.faces.RenderSurface_exposed(1f, false, rprm.RigidObjectColor);

            // draw translucent shell
            if (model3.mc.deformables != null)
                foreach (Mesh msh in model3.mc.deformables)
                    msh.faces.RenderSurface_exposed(rprm.TransparencyCoeff,
                                        rprm.UseGrainColor, rprm.SurfaceColor);

            // draw edges on rigid objects
            if (model3.mc.nonDeformables != null && rprm.ShowTetraEdgesRigid)
                foreach (Mesh msh in model3.mc.nonDeformables)
                    if (!msh.hide)
                        msh.faces.RenderEdges_exposed(rprm.TetraEdgesColor, rprm.TetraEdgesWidth);

            // draw all edges (for figures in paper)
            if (model3.mc.deformables != null && rprm.ShowAllTetraEdges)
                foreach (Mesh msh in model3.mc.deformables)
                    if (!msh.hide)
                        msh.faces.RenderEdges_all(rprm.TetraEdgesColor, rprm.TetraEdgesWidth);

            // draw grain edges (double-check coordinates used)
            if (rprm.ShowGrainBoundaries && model3.mc.exposedEdges != null)
            {
                model3.mc.exposedEdges.RenderGrainEdgesAsCylinders(
                    rprm.GrainBoundaryColor, rprm.CylinderSides, rprm.CylinderRadius);
                model3.mc.exposedEdges.RenderGrainEdges(rprm.GrainBoundaryWidth, rprm.GrainBoundaryColor);
            }

            if (tsbAxes.Checked) RenderAxes();
//            if (rprm.RenderText) RenderTextAndPlots();

            glControl1.SwapBuffers();
        }




        void RenderBackground(bool white = false)
        {
            float k = 0.95f;
            float k2 = 0.9f;

            if (white)
            {
                k = 1; k2 = 1;
            }

            GL.Disable(EnableCap.Lighting);
            GL.ClearColor(k, k, k, 1f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            GL.MatrixMode(MatrixMode.Projection);
            GL.PushMatrix();
            GL.LoadIdentity();
            GL.Begin(PrimitiveType.Quads);
            double s = 1.0;
            GL.Color3(k2, k2, k2);
            GL.Vertex3(-s, -s, 0);
            GL.Color3(k, k, k);
            GL.Vertex3(-s, s, 0);
            GL.Vertex3(s, s, 0);
            GL.Color3(k2, k2, k2);
            GL.Vertex3(s, -s, 0);
            GL.End();
            GL.PopMatrix();
            GL.Clear(ClearBufferMask.DepthBufferBit);
        }

        void RenderAxes()
        {
            GL.Disable(EnableCap.Lighting);
            GL.Disable(EnableCap.DepthTest);
            double d = 0.1;
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(Color.Red);
            GL.Vertex3(0.0, 0.0, 0.0);
            GL.Vertex3(d, 0.0, 0.0);
            GL.Color3(Color.Green);
            GL.Vertex3(0.0, 0.0, 0.0);
            GL.Vertex3(0.0, d, 0.0);
            GL.Color3(Color.Blue);
            GL.Vertex3(0.0, 0.0, 0.0);
            GL.Vertex3(0.0, 0.0, d);
            GL.End();
        }
        
        void SetUpLight()
        {
            GL.Enable(EnableCap.ColorMaterial);
            GL.Enable(EnableCap.Normalize);
            GL.Enable(EnableCap.Lighting);
            GL.ShadeModel(ShadingModel.Smooth);

            if (rprm.Light0)
            {
                GL.Enable(EnableCap.Light0);
                GL.Light(LightName.Light0, LightParameter.Diffuse, new Color4(rprm.L0intensity, rprm.L0intensity, rprm.L0intensity,1f));
                GL.Light(LightName.Light0, LightParameter.Position, new Color4(rprm.L0x, rprm.L0y, rprm.L0z, 0));
            }
            else
                GL.Disable(EnableCap.Light0);

            if (rprm.Light1)
            {
                GL.Enable(EnableCap.Light1);
                GL.Light(LightName.Light1, LightParameter.Diffuse, new Color4(rprm.L1intensity, rprm.L1intensity, rprm.L1intensity, 1f));
                GL.Light(LightName.Light1, LightParameter.Position, new Color4(rprm.L1x, rprm.L1y, rprm.L1z, 0));
            }
            else
                GL.Disable(EnableCap.Light1);

            if (rprm.Light2)
            {
                GL.Enable(EnableCap.Light2);
                GL.Light(LightName.Light2, LightParameter.Diffuse, new Color4(rprm.L2intensity, rprm.L2intensity, rprm.L2intensity, 1f));
                GL.Light(LightName.Light2, LightParameter.Position, new Color4(rprm.L2x, rprm.L2y, rprm.L2z, 0));
            }
            else
                GL.Disable(EnableCap.Light2);
        }

        private void glControl1_MouseDown(object sender, MouseEventArgs e)
        {
            lastX = e.X; lastY = e.Y;
        }

        private void glControl1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == System.Windows.Forms.MouseButtons.Left)
            {
                rprm.theta += (e.X - lastX);
                rprm.phi += (e.Y - lastY);
                lastX = e.X;
                lastY = e.Y;
            }
            else if (e.Button == MouseButtons.Middle)
            {
                rprm.dx += (e.X - lastX);
                rprm.dy -= (e.Y - lastY);
                lastX = e.X;
                lastY = e.Y;
                reshape();
            }
            glControl1.Invalidate();
            //            Trace.WriteLine($"phi {phi} theta {theta} dx {dx} dy {dy} scale {scale}");
        }

        private void glControl1_Resize(object sender, EventArgs e)
        {
            reshape();
        }

        private void GlControl1_MouseWheel(object sender, MouseEventArgs e)
        {
            if(rprm.UseFrustum)
                rprm.zOffset += 0.001 * rprm.zOffset * e.Delta;
            else 
                rprm.scale += 0.001 * rprm.scale * e.Delta;
            reshape();
            glControl1.Invalidate();
        }

        void reshape()
        {
            aspectRatio = (double)glControl1.Width / glControl1.Height;
            if (rprm.UseFrustum)
            {
                GL.Viewport(0, 0, glControl1.Width, glControl1.Height);
                GL.MatrixMode(MatrixMode.Projection);
                GL.LoadIdentity();
                perspectiveGL(rprm.fovY, aspectRatio, rprm.zNear, rprm.zFar);
                GL.MatrixMode(MatrixMode.Modelview);
                GL.LoadIdentity();
            }
            else
            {
                GL.Viewport(0, 0, glControl1.Width, glControl1.Height);
                GL.MatrixMode(MatrixMode.Projection);
                GL.LoadIdentity();

                GL.Ortho(-rprm.scale * aspectRatio,
                        rprm.scale * aspectRatio,
                        -rprm.scale,
                        rprm.scale, -1000, 1000);
                GL.MatrixMode(MatrixMode.Modelview);
                GL.LoadIdentity();

            }
        }

        // Replaces gluPerspective. Sets the frustum to perspective mode.
        // fovY     - Field of vision in degrees in the y direction
        // aspect   - Aspect ratio of the viewport
        // zNear    - The near clipping distance
        // zFar     - The far clipping distance
        void perspectiveGL(double fovY, double aspect, double zNear, double zFar)
        {
            double fW, fH;
            fH = System.Math.Tan((fovY / 2) / 180 * Math.PI) * zNear;
            fH = System.Math.Tan(fovY / 360 * Math.PI) * zNear;
            fW = fH * aspect;
            GL.Frustum(-fW, fW, -fH, fH, zNear, zFar);
        }
        #endregion

        #region background worker
        bool timeToPause;
        private void backgroundWorker1_DoWork(object sender, System.ComponentModel.DoWorkEventArgs e)
        {
            running = true;
            bool simulationFinished = false;
            do
            {
                model3.Step();
                if (model3.prms.MaxSteps > 0 && model3.cf.StepNumber >= model3.prms.MaxSteps) timeToPause = true;
                backgroundWorker1.ReportProgress(0);
            } while (!timeToPause && !simulationFinished);
        }

        private void backgroundWorker1_ProgressChanged(object sender, System.ComponentModel.ProgressChangedEventArgs e)
        {
            glControl1.Invalidate();
            tnCurrentFrame.Tag = model3.cf;
            if (treeView1.SelectedNode == tnCurrentFrame)
            {
                pg.SelectedObject = model3.cf;
                pg.Refresh();
            }
            if (model3.cf != null) tssCurrentFrame.Text = $"{model3.cf.StepNumber}-{model3.cf.IterationsPerformed}-{model3.cf.TimeScaleFactor}";
            else tssCurrentFrame.Text = "reduce time step";
        }

        private void backgroundWorker1_RunWorkerCompleted(object sender, System.ComponentModel.RunWorkerCompletedEventArgs e)
        {
            running = false;
            tssStatus.Text = "Done";
            tssCurrentFrame.Text = "";
            glControl1.Invalidate();
            pg.Refresh();
            UpdateTrackbar();
        }
        #endregion

        #region trackbar
        void UpdateTrackbar()
        {
            trackBar1.ValueChanged -= trackBar1_ValueChanged;
            if (tsbPreviewMode.Checked)
            {
                if (model3.prms.MaxSteps > 0)
                {
                    trackBar1.Enabled = true;
                    trackBar1.Maximum = model3.prms.MaxSteps;
                    trackBar1.Value = 0;
                }
                else
                {
                    trackBar1.Enabled = false;
                }
            }
            else
            {
                trackBar1.Enabled = true;

                if (model3.allFrames.Count >= 2)
                {
                    trackBar1.Maximum = model3.allFrames.Count - 1;
                    trackBar1.Enabled = true;
                    trackBar1.Value = model3.cf == null ? 0 : model3.cf.StepNumber;
                }
                else
                {
                    trackBar1.Value = 0;
                    trackBar1.Enabled = false;
                }
            }

            trackBar1.ValueChanged += trackBar1_ValueChanged;
        }

        private void trackBar1_ValueChanged(object sender, EventArgs e)
        {
            if (tsbPreviewMode.Checked)
            {
                // move 
                double time = trackBar1.Value * model3.prms.InitialTimeStep;
                model3._positionNonDeformables(time);
            }
            else
            {
                int frame = trackBar1.Value;
                model3.GoToFrame(frame);
                tnCurrentFrame.Tag = model3.cf;
                if (treeView1.SelectedNode == tnCurrentFrame)
                {
                    pg.SelectedObject = model3.cf;
                    pg.Refresh();
                }
                foreach (TreeNode tn in tnMeshCollection.Nodes)
                {
                    RefreshSfcNodes(tn);
                }
                tssCurrentFrame.Text = frame.ToString();

            }
            glControl1.Invalidate();
        }

        private void tsbPreviewMode_Click(object sender, EventArgs e)
        {
            // enable preview of rigid body translation
            UpdateTrackbar();
            if (tsbPreviewMode.Checked && trackBar1.Enabled)
            {
                // move rigid objects
                trackBar1_ValueChanged(sender, e);
            }
            else
            {
                // move to the state from cf
                if(model3.cf != null) model3._positionNonDeformables(model3.cf.SimulationTime);
                else model3._positionNonDeformables(0);
            }
        }
        #endregion

        #region mesh context menu
        private void removeToolStripMenuItem2_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            if (mg == null) return;
            model3.mc.mgs.Remove(mg);
            treeView1.SelectedNode.Remove();
            model3.isReady = false;
            model3.mc.Prepare();
            glControl1.Invalidate();
        }

        private void resizeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            mg.Resize(mg.scale);
            glControl1.Invalidate();
            pg.Refresh();
        }

        private void makeTorusToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            MeshTools.MakeTorus(mg, 0.35, 0.15);
            cZsToolStripMenuItem1_Click(sender, e);
        }

        private void splitToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            if (!mg.isDeformable)
            {
                MessageBox.Show("This operation is only applicable to deformable mesh");
                return;
            }
            mg.SeparateGranules();
            model3.mc.Prepare();

            glControl1.Invalidate();
            pg.Refresh();
            RefreshSfcNodes(treeView1.SelectedNode);
        }

        private void cZsToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            // insert cohesive zones, mark surface elements, create faces
            object selection = treeView1.SelectedNode.Tag; // this should come from UI selection
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            if (mg == null) return;
            
            if (!mg.isDeformable)
            {
                MessageBox.Show("This operation is only applicable to deformable mesh");
                return;
            }
            mg.InsertCohesiveElements();
            model3.mc.Prepare();
            glControl1.Invalidate();
            pg.Refresh();
            model3.isReady = false;
            RefreshSfcNodes(treeView1.SelectedNode);
        }

        void RefreshSfcNodes(TreeNode meshNode)
        {
            Mesh msh = (Mesh)meshNode.Tag;
            TreeNode sfc = null;

            foreach (TreeNode tn in meshNode.Nodes) { if(tn.Text == "sfc") { sfc = tn; break; } }
            sfc.Nodes.Clear();
            foreach (SurfaceFragment sf in msh.surfaceFragments)
            {
                TreeNode sfc_frag = sfc.Nodes.Add(sf.ToString());
                sfc_frag.Tag = sf;
            }
            TreeNode trs = null;
            foreach (TreeNode tn in meshNode.Nodes) { if (tn.Text == "trs") { trs = tn; break; } }
            trs.Nodes.Clear();
            trs.Tag = msh.translationCollection;
            foreach (Translation trsl in msh.translationCollection)
            {
                TreeNode trsl_node = trs.Nodes.Add(trsl.ToString());
                trsl_node.Tag = trsl;
                trsl_node.ContextMenuStrip = cmsOneTranslation;
            }

        }

        private void cZsAndCapsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // insert cohesive zones, mark surface elements, create faces
            object selection = treeView1.SelectedNode.Tag;
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            if (!mg.isDeformable)
            {
                MessageBox.Show("This operation is only applicable to deformable mesh");
                return;
            }
            mg.FuseEndCaps();
            mg.InsertCohesiveElements();
            model3.mc.Prepare();
            glControl1.Invalidate();
            pg.Refresh();
            RefreshSfcNodes(treeView1.SelectedNode);
        }
        #endregion

        #region other context menus
        private void addToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            Translation trsl = new Translation();
            TranslationCollection tc = (TranslationCollection)selection;
            tc.Add(trsl);
            TreeNode tn = treeView1.SelectedNode.Nodes.Add(trsl.ToString());
            tn.ContextMenuStrip = cmsOneTranslation;
            tn.Tag = trsl;
            treeView1.SelectedNode = tn;

        }

        private void refreshToolStripMenuItem_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            TranslationCollection tc = (TranslationCollection)selection;
            tc.Sort();
            treeView1.SelectedNode.Nodes.Clear();
            foreach(Translation trsl in tc)
            {
                TreeNode tn = treeView1.SelectedNode.Nodes.Add(trsl.ToString());
                tn.Tag = trsl;
                tn.ContextMenuStrip = cmsOneTranslation;
            }
            model3.mc.Prepare();
            glControl1.Invalidate();
        }

        private void removeToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            Translation trsl = (Translation)selection;
            TranslationCollection tc = (TranslationCollection)treeView1.SelectedNode.Parent.Tag;
            tc.Remove(trsl);
            treeView1.SelectedNode.Remove();
        }

        #endregion

        #region interface

        private void oneStepToolStripMenuItem_Click(object sender, EventArgs e)
        {
            model3.Step();
            glControl1.Invalidate();
            pg.SelectedObject = model3.cf;
            pg.Refresh();
        }

        private void tsbInvalidate_Click(object sender, EventArgs e)
        {
            glControl1.Invalidate();
        }

        private void tsbFaceDisplay_Click(object sender, EventArgs e)
        {
            glControl1.Invalidate();
        }

        // Mesh >> Add
        private void addToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) return;
            // load mesh from file and add to ImplicitModel.mgs
            openFileDialogMeshes.Filter = "Mesh|*.geo;*.msh;*.mg";
            if(openFileDialogMeshes.ShowDialog() == DialogResult.OK)
            {
                string ext = Path.GetExtension(openFileDialogMeshes.FileName);
                string name = Path.GetFileNameWithoutExtension(openFileDialogMeshes.FileName);
                Stream str = openFileDialogMeshes.OpenFile();
                Mesh mg = new Mesh(str, name, ext);
                Trace.Assert(mg != null);
                if (model3.mc.mgs.Count == 0) mg.isDeformable = true;
                else if (model3.mc.mgs.Count == 1) mg.isFloor = true;
                model3.mc.mgs.Add(mg);
                mg.CenterSample();

                // name the simulation
                if(model3.mc.mgs.Count == 1) model3.prms.name = name;
                RebuildTree();
                if(mg.isDeformable) model3.prms.GrainSize = mg.AverageGrainSize();
            }
            glControl1.Invalidate();
            model3.mc.Prepare();
            model3.isReady = false;
        }

        private void openSimulationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) return;
            // load frame history
            openFileDialog1.Filter = "Serialized Frames|params";
            openFileDialog1.InitialDirectory = AppDomain.CurrentDomain.BaseDirectory + "_sims";

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                model3.saveFolder = Path.GetDirectoryName(openFileDialog1.FileName);
                model3.LoadSimulation();
                UpdateTrackbar();
                reshape();
                glControl1.Invalidate();
                model3.mc.Prepare();
                rprm = RenderPrms.Load(model3.saveFolder);
                RebuildTree();
                reshape();
            }
        }

        private void openClearSimToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) return;
            // load frame history
            openFileDialog1.Filter = "Serialized Frames|params";
            openFileDialog1.InitialDirectory = AppDomain.CurrentDomain.BaseDirectory + "_sims";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                model3.saveFolder = Path.GetDirectoryName(openFileDialog1.FileName);
                model3.LoadSimulation(clear: true);
                UpdateTrackbar();
                reshape();
                glControl1.Invalidate();
                model3.mc.Prepare();
                rprm = RenderPrms.Load(model3.saveFolder);
                RebuildTree();
                reshape();
            }
        }


        private void saveInitialStateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) return;
            model3.SaveSimulationInitialState();
            tssStatus.Text = "Saved";
        }

        private void resizeX01ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) return;
            double f = 0.1;
            foreach(Node nd in model3.mc.allNodes)
            {
                nd.x0 *= f; nd.y0 *= f; nd.z0 *= f;
                nd.cx = nd.x0;nd.cy = nd.y0;nd.cz = nd.z0;
            }
            foreach(Mesh msh in model3.mc.mgs)
            {
                msh.BoundingBox();
            }
            glControl1.Invalidate();
        }

        private void rotate90DegToolStripMenuItem_Click(object sender, EventArgs e)
        {
            object selection = treeView1.SelectedNode.Tag;
            if (selection == null || running) return;
            Mesh mg = (Mesh)selection;
            mg.Rotate90Deg();
            glControl1.Invalidate();
            pg.Refresh();
        }

        private void writeCSVToolStripMenuItem_Click(object sender, EventArgs e)
        {
            string CSVFolder = model3.saveFolder + "\\CSV";
            if (!Directory.Exists(CSVFolder)) Directory.CreateDirectory(CSVFolder);
            string CSVPath = CSVFolder + "\\data.csv";
            FrameInfo.WriteCSV(model3.allFrames, CSVPath);
            FrameInfo.FrameSummary smr = new FrameInfo.FrameSummary(model3.allFrames, model3.prms.name, model3.prms.GrainSize);
            smr.WriteCSV(CSVFolder + "\\summary.csv");
        }

        private void pgRendering_SelectedObjectsChanged(object sender, EventArgs e)
        {
            glControl1.Invalidate();
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            rprm.Save(model3.saveFolder);
        }

        private void analysisToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // open a new analysis window
            FormAnalysis fa2 = new FormAnalysis();
            fa2.smr = new FrameInfo.FrameSummary(model3.allFrames, model3.prms.name, model3.prms.GrainSize);
            fa2.Text = $"Analysis - {model3.prms.name}";
            fa2.Show();
        }

        private void takeScreenshotToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (model3.cf == null) return;
            string savePath = model3.saveFolder + "\\screenshots\\";
            if (!Directory.Exists(savePath)) Directory.CreateDirectory(savePath);

            string filename = $"{savePath}{model3.cf.StepNumber:0000}_{model3.cf.SimulationTime:F5}.jpg";
            TakeScreenshot(filename,0,model3.cf.SimulationTime,model3.allFrames[model3.allFrames.Count-1].SimulationTime);
        }

        private void continueFromLastToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) { timeToPause = true; tssStatus.Text = "Stopping"; return; }

            if(trackBar1.Enabled == false || trackBar1.Value == 0) model3.cf = null;
            tsbPreviewMode.Checked = false;
            backgroundWorker1.RunWorkerAsync();
            tssStatus.Text = "Running";
            trackBar1.Enabled = false;
            timeToPause = false;
        }

        #endregion

        #region animation rendering

        
        private void reshapeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            reshape();
            glControl1.Invalidate();
        }

        private void eraseSubsequentStepsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int currentStep = model3.cf.StepNumber;
            int removeCount = model3.allFrames.Count - 1 - currentStep;
            if (removeCount == 0) return;
            model3.allFrames.RemoveRange(currentStep + 1, removeCount);
            UpdateTrackbar();
            model3.SaveFrameData();
        }

        private void pPRRelationsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            PPR_relations frm = new PPR_relations();
            frm.prms = model3.prms;
            frm.Show();
        }

        // saves an image file of what is currently shown in glControl1 in PNG format
        // Create string formatting options (used for alignment)
        StringFormat format = new StringFormat()
        {
            Alignment = StringAlignment.Far,
            LineAlignment = StringAlignment.Near
        };



        FormAnalysis fa = new FormAnalysis();
        SolidBrush shadowBrush = new SolidBrush(Color.FromArgb(0, Color.Red));
        Font myFont = new Font("Tahoma", 40);
        void TakeScreenshot(string filename, double fromTime, double toTime, double maxTime)
        {
            if (model3.cf == null) return;
            int w = glControl1.ClientSize.Width;
            int h = glControl1.ClientSize.Height;
            using (Bitmap bmp = new Bitmap(w, h))
            {
                BitmapData data = bmp.LockBits(glControl1.ClientRectangle, ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

                glControl1.SwapBuffers();
                GL.ReadPixels(0, 0, w, h, OpenTK.Graphics.OpenGL.PixelFormat.Bgr, PixelType.UnsignedByte, data.Scan0);
                glControl1.SwapBuffers();

                bmp.UnlockBits(data);
                bmp.RotateFlip(RotateFlipType.RotateNoneFlipY);

                // draw text and plots
                if(rprm.RenderText)
                {
                    using(Graphics g = Graphics.FromImage(bmp))
                    {
                        g.CompositingMode = CompositingMode.SourceOver;
                        g.SmoothingMode = SmoothingMode.AntiAlias;
                        g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                        g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAliasGridFit;

                        RectangleF rectf = new RectangleF(0, 0, w, 600);
                        if (model3.cf != null) g.DrawString($"{rprm.Comment}\ntime: {toTime:0.000} s", myFont, Brushes.Black, rectf, format);

                        // render plots
                        fa.Width = w * 5/8;
                        fa.Height = h * 7 / 9;
                        int w2 = fa.chart1.Width;
                        int h2 = fa.chart1.Height;
                        fa.forRendering = true;
                        fa.fromTime = fromTime;
                        fa.toTime = toTime;
                        fa.maxTime = maxTime;
                        fa.smr = new FrameInfo.FrameSummary(model3.allFrames, model3.prms.name, model3.prms.GrainSize);
                        fa.ShowData();

                        Rectangle bounds = new Rectangle(w-w2, h-h2, w, h);
                        fa.chart1.DrawToBitmap(bmp, bounds);

                        g.Flush();
                    }
                }

                bmp.Save(filename, ImageFormat.Jpeg);
            }
        }

        //        bool renderingAnimation = false;
        string renderSavePath;
        int _ToStep;
        Stream strRenderingReport;
        StreamWriter swReport;
        double mixingCoeff, frameTime, firstFrameTime, lastFrameTime, timeSpan;
        int renderingFrame;
        private void renderSimulationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running || timer1.Enabled) return;

            // clear save directory
            renderSavePath = model3.saveFolder + $"\\{rprm.renderFolder}\\";
            if (Directory.Exists(renderSavePath)) Directory.Delete(renderSavePath, true);
            Directory.CreateDirectory(renderSavePath);


            if (rprm.ToStep == -1) _ToStep = model3.allFrames.Count - 1;
            else _ToStep = rprm.ToStep;
            strRenderingReport = File.Create(renderSavePath + "report.csv");
            swReport = new StreamWriter(strRenderingReport);
            swReport.WriteLine($"renderingFrame, frameTime, correspondingStep");
            firstFrameTime = model3.allFrames[rprm.FromStep].SimulationTime;
            lastFrameTime = model3.allFrames[_ToStep].SimulationTime;
            timeSpan = lastFrameTime - firstFrameTime;


            renderingFrame = 0;

            timer1.Interval = rprm.Delay;
            timer1.Enabled = true;
            // subsequent processing takes place in timer1_Tick
            // in order to not freeze UI


        }


        private void timer1_Tick(object sender, EventArgs e)
        {
            tssStatus.Text = $"Rendering frame {renderingFrame}";
            statusStrip1.Refresh();

            // render frame
            mixingCoeff = (double)renderingFrame / (double)rprm.nFrames;
            frameTime = (1f - mixingCoeff) * firstFrameTime + mixingCoeff * lastFrameTime;

            // find the simulation step that corresponds to current animation frame
            int stepIdx = rprm.FromStep;
            while (model3.allFrames[stepIdx].SimulationTime < frameTime && stepIdx < model3.allFrames.Count)
                stepIdx++;

            // linearly interpolate, note that the value is negative
            double simFrameTime = model3.allFrames[stepIdx].SimulationTime;
            double goBackTime = (simFrameTime - frameTime);
            Trace.WriteLine($"{renderingFrame}; {frameTime} s; last: {lastFrameTime}");

            // load if needed
            model3.GoToFrame(stepIdx);

            // fine-tune current position to correspond to frame timing
            foreach (Node nd in model3.mc.allNodes)
            {
                nd.cx -= goBackTime * nd.vx;
                nd.cy -= goBackTime * nd.vy;
                nd.cz -= goBackTime * nd.vz;
            }

            // ask glControl to render the setup
            glControl1.Invalidate();
            glControl1.Update();
            glControl1.Refresh();
            swReport.WriteLine($"{renderingFrame}, {frameTime}, {stepIdx}");

            // save image
            TakeScreenshot($"{renderSavePath}{renderingFrame:0000}.jpg", firstFrameTime, frameTime, lastFrameTime);

            renderingFrame++;

            if(renderingFrame >= rprm.nFrames)
            {
                timer1.Enabled = false;
                int lastRenderingFrame = renderingFrame - 1;
                // write 50 copies of the last frame (for pause)
                for (int i = renderingFrame; i < (renderingFrame + 50); i++)
                    File.Copy($"{renderSavePath}{lastRenderingFrame:0000}.jpg", $"{renderSavePath}{i:0000}.jpg");

                tssStatus.Text = "Finished rendering";
                swReport.Close();
            }
        }


        #endregion
    }
}
