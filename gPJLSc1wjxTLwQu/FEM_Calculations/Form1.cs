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


namespace icFlow
{
    public partial class Form1 : Form
    {
        class DisplayPrms
        {
            public float theta = 0, phi = 0;           // view angle
            public float dx = 0, dy = 0;           // rendering offset
            public double scale = 0.17;
        }

        int lastX, lastY;   // mouse last position
        double aspectRatio = 1;
        bool running = false; // computation is running in backgroundworker
        DisplayPrms dparam;
        ImplicitModel3 model3 = new ImplicitModel3();
        TreeNode tnRoot, tnPrms, tnMeshCollection, tnCurrentFrame;

        #region initialize

        public Form1()
        {
            model3.Initialize();

            dparam = new DisplayPrms();
            InitializeComponent();
            glControl1.MouseWheel += GlControl1_MouseWheel;
            treeView1.NodeMouseClick += (sender, args) => treeView1.SelectedNode = args.Node;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            SetUpLight();
            // glControl1.Dock = DockStyle.Fill;
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
            RenderBackground(tsbWhite.Checked);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            GL.Translate(dparam.dx / 1000, dparam.dy / 1000, 0);
            GL.Rotate(-dparam.phi, new Vector3(1, 0, 0));
            if (!tsbPhi.Checked) GL.Rotate(-dparam.theta, new Vector3(0, 0, 1));
            if (tsbRotate.Checked) GL.Rotate(90, new Vector3(0, 1, 0));

            if (tsbLight.Checked) GL.Enable(EnableCap.Lighting); else GL.Disable(EnableCap.Lighting);
            if (model3.mc.failedCZs != null && tsbIncludeFailedCZs.Checked) model3.mc.failedCZs.RenderCZ();
            if (model3.mc.nonFailedCZs != null && tsbCZs.Checked) model3.mc.nonFailedCZs.RenderCZ();

            if (tsbLight.Checked) GL.Enable(EnableCap.Lighting); else GL.Disable(EnableCap.Lighting);

            if (tsbSurface.Checked && model3.mc.allFaces != null)
                model3.mc.allFaces.RenderSurface(tsbTransparent.Checked, tsbShowCreated.Checked, tsbExposed.Checked);

            if (tsbTEdges.Checked) foreach (Mesh mg in model3.mc.mgs) mg.RenderFaceEdges();
            //            if (tsbGEdges.Checked) foreach (Mesh mg in model3.mc.mgs) mg.RenderGranuleEdges();
            if (tsbGEdges.Checked && model3.mc.exposedEdges != null)
                model3.mc.exposedEdges.RenderGranuleEdges();
            if (tsbContacts.Checked)
            {
                GL.Enable(EnableCap.Lighting);
                GL.Enable(EnableCap.DepthTest);
                GL.Disable(EnableCap.Blend);
                GL.Begin(PrimitiveType.Triangles);
                GL.Color3(0.9f, 0.8f, 0.1f);

                if(model3.gf != null)
                    for(int impact = 0;impact<model3.gf.nImpacts;impact++)
                    {
                    }
                GL.End();
                GL.Disable(EnableCap.Lighting);
            }
            if (tsbAxes.Checked) RenderAxes();

            /*
            Mesh mg1 = lbMeshes.SelectedItem as Mesh;
            SurfaceFragment sf = lbSurfaceFragments.SelectedItem as SurfaceFragment;
            if(sf != null && mg1 != null)
            {
                GL.DepthFunc(DepthFunction.Lequal);
                GL.Disable(EnableCap.Blend);

                GL.Begin(PrimitiveType.Triangles);
                GL.Color3(Color.Red);
                foreach (int idx in sf.faces)
                {
                    Face f = mg1.faces[idx];
                    f.RenderTriangle();
                }
                GL.End();
            }
            */
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
            GL.Enable(EnableCap.DepthTest);
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

        /*
        void RenderContacts()
        {
            GL.Disable(EnableCap.Lighting);
            GL.Disable(EnableCap.DepthTest);
            GL.LineWidth(2f);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.9f, 0.91f, 0.29f);
            foreach (Element.EV_Collision col in ImplicitModel3.mg.bvht.clst) col.elem.RenderWireFrame();
            GL.End();

            GL.PointSize(5f);
            GL.Begin(PrimitiveType.Points);
            GL.Color3(0.1f, 0.91f, 0.29f);
            foreach (Element.EV_Collision col in ImplicitModel3.mg.bvht.clst) col.node.Render();
            GL.End();

            GL.LineWidth(1f);
            GL.Begin(PrimitiveType.Lines);
            GL.Color3(0.05f, 0.05f, 0.05f);
            foreach (Element.EV_Collision col in ImplicitModel3.mg.bvht.clst)
            {
                GL.Vertex3(col.elem.cCenter);
                GL.Vertex3(col.node.cx, col.node.cy, col.node.cz);
            }

            GL.End();

        }
        */

        void SetUpLight()
        {
            GL.Enable(EnableCap.ColorMaterial);
            GL.Enable(EnableCap.Normalize);
            GL.Enable(EnableCap.Lighting);
            GL.Enable(EnableCap.Light0);
            GL.ShadeModel(ShadingModel.Smooth);
            GL.Light(LightName.Light0, LightParameter.Ambient, Color4.Black);
            GL.Light(LightName.Light0, LightParameter.Diffuse, Color4.White);
            GL.Light(LightName.Light0, LightParameter.Position, new Color4(3, -3, -3, 0));
            GL.Enable(EnableCap.Light1);
            GL.Light(LightName.Light1, LightParameter.Ambient, Color4.Black);
            GL.Light(LightName.Light1, LightParameter.Diffuse, Color4.White);
            GL.Light(LightName.Light1, LightParameter.Position, new Color4(-1, -1, 3, 0));
            GL.Enable(EnableCap.Light2);
            GL.Light(LightName.Light2, LightParameter.Ambient, Color4.Black);
            GL.Light(LightName.Light2, LightParameter.Diffuse, Color4.White);
            GL.Light(LightName.Light2, LightParameter.Position, new Color4(-1, 1, 0, 0));
        }

        private void glControl1_MouseDown(object sender, MouseEventArgs e)
        {
            lastX = e.X; lastY = e.Y;
        }

        private void glControl1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == System.Windows.Forms.MouseButtons.Left)
            {
                dparam.theta += (e.X - lastX);
                dparam.phi += (e.Y - lastY);
                lastX = e.X;
                lastY = e.Y;
            }
            else if (e.Button == MouseButtons.Middle)
            {
                dparam.dx += (e.X - lastX);
                dparam.dy -= (e.Y - lastY);
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
            dparam.scale += 0.001 * dparam.scale * e.Delta;
            reshape();
            glControl1.Invalidate();
        }

        void reshape()
        {
            aspectRatio = (double)glControl1.Width / glControl1.Height;
            GL.Viewport(0, 0, glControl1.Width, glControl1.Height);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();

            GL.Ortho(-dparam.scale * aspectRatio,
                    dparam.scale * aspectRatio,
                    -dparam.scale,
                    dparam.scale, -1000, 1000);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();

            GL.Enable(EnableCap.PointSmooth);
            GL.Hint(HintTarget.PointSmoothHint, HintMode.Nicest);
            GL.Enable(EnableCap.LineSmooth);
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
                RebuildTree();
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
                RebuildTree();
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

        private void analysisToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // open a new analysis window
            FormAnalysis fa = new FormAnalysis();
            fa.smr = new FrameInfo.FrameSummary(model3.allFrames, model3.prms.name, model3.prms.GrainSize);
            fa.Text = $"Analysis - {model3.prms.name}";
            fa.Show();
        }

        private void takeScreenshotToolStripMenuItem_Click(object sender, EventArgs e)
        {
            string savePath = model3.saveFolder + "screenshots\\";
            if (!Directory.Exists(savePath)) Directory.CreateDirectory(savePath);

            int w = glControl1.ClientSize.Width;
            int h = glControl1.ClientSize.Height;
            Bitmap bmp = new Bitmap(w, h);
            System.Drawing.Imaging.BitmapData data =
                bmp.LockBits(glControl1.ClientRectangle, System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            GL.ReadPixels(0, 0, w, h, PixelFormat.Bgr, PixelType.UnsignedByte, data.Scan0);
            bmp.UnlockBits(data);

            bmp.RotateFlip(RotateFlipType.RotateNoneFlipY);
            string filename = savePath + "screenshot.png";
            bmp.Save(filename, System.Drawing.Imaging.ImageFormat.Png);
        }

        private void renderSimulationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (running) return;
            string savePath = model3.saveFolder + "screenshots\\";
            if (!Directory.Exists(savePath)) Directory.CreateDirectory(savePath);
            int w = glControl1.ClientSize.Width;
            int h = glControl1.ClientSize.Height;
            Bitmap bmp = new Bitmap(w, h);

            for (int i = 0; i < model3.allFrames.Count; i++)
            {
                model3.GoToFrame(i);
                glControl1.Invalidate();
                glControl1.Refresh();
                System.Drawing.Imaging.BitmapData data =
                    bmp.LockBits(glControl1.ClientRectangle, System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                GL.ReadPixels(0, 0, w, h, PixelFormat.Bgr, PixelType.UnsignedByte, data.Scan0);
                bmp.UnlockBits(data);

                bmp.RotateFlip(RotateFlipType.RotateNoneFlipY);
                string filename = $"{savePath}{model3.prms.name}_{i:000}.png";
                bmp.Save(filename, System.Drawing.Imaging.ImageFormat.Png);
            }
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

    }
}
