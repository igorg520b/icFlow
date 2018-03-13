namespace icFlow
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openSimulationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openClearSimToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.addMeshToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem2 = new System.Windows.Forms.ToolStripSeparator();
            this.saveInitialStateToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.takeScreenshotToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.renderSimulationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.writeCSVToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.simulationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.continueFromLastToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.oneStepToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.analysisToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.resizeX01ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.reshapeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.eraseSubsequentStepsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem3 = new System.Windows.Forms.ToolStripSeparator();
            this.pPRRelationsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.tssStatus = new System.Windows.Forms.ToolStripStatusLabel();
            this.tssCurrentFrame = new System.Windows.Forms.ToolStripStatusLabel();
            this.glControl1 = new OpenTK.GLControl();
            this.backgroundWorker1 = new System.ComponentModel.BackgroundWorker();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.trackBar1 = new System.Windows.Forms.TrackBar();
            this.pg = new System.Windows.Forms.PropertyGrid();
            this.tsDisplayOptions = new System.Windows.Forms.ToolStrip();
            this.tsbRenderPanel = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator5 = new System.Windows.Forms.ToolStripSeparator();
            this.tsbPhi = new System.Windows.Forms.ToolStripButton();
            this.tsbAxes = new System.Windows.Forms.ToolStripButton();
            this.tsbRotate = new System.Windows.Forms.ToolStripButton();
            this.tsbPreviewMode = new System.Windows.Forms.ToolStripButton();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.panel2 = new System.Windows.Forms.Panel();
            this.treeView1 = new System.Windows.Forms.TreeView();
            this.cmsTranslations = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.addToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.refreshToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.cmsOneTranslation = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.removeToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.cmsMesh = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.cZsAndCapsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.cZsToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.splitToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.makeTorusToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem6 = new System.Windows.Forms.ToolStripSeparator();
            this.resizeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.rotate90DegToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem7 = new System.Windows.Forms.ToolStripSeparator();
            this.removeToolStripMenuItem2 = new System.Windows.Forms.ToolStripMenuItem();
            this.openFileDialogMeshes = new System.Windows.Forms.OpenFileDialog();
            this.menuStrip1.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBar1)).BeginInit();
            this.tsDisplayOptions.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.cmsTranslations.SuspendLayout();
            this.cmsOneTranslation.SuspendLayout();
            this.cmsMesh.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Font = new System.Drawing.Font("Segoe UI", 9F);
            this.menuStrip1.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.simulationToolStripMenuItem,
            this.toolsToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1012, 40);
            this.menuStrip1.TabIndex = 1;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.openSimulationToolStripMenuItem,
            this.openClearSimToolStripMenuItem,
            this.addMeshToolStripMenuItem,
            this.toolStripMenuItem2,
            this.saveInitialStateToolStripMenuItem,
            this.toolStripMenuItem1,
            this.takeScreenshotToolStripMenuItem,
            this.renderSimulationToolStripMenuItem,
            this.writeCSVToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(64, 36);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // openSimulationToolStripMenuItem
            // 
            this.openSimulationToolStripMenuItem.Name = "openSimulationToolStripMenuItem";
            this.openSimulationToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)(((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Shift) 
            | System.Windows.Forms.Keys.O)));
            this.openSimulationToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.openSimulationToolStripMenuItem.Text = "Open Simulation";
            this.openSimulationToolStripMenuItem.Click += new System.EventHandler(this.openSimulationToolStripMenuItem_Click);
            // 
            // openClearSimToolStripMenuItem
            // 
            this.openClearSimToolStripMenuItem.Name = "openClearSimToolStripMenuItem";
            this.openClearSimToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.O)));
            this.openClearSimToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.openClearSimToolStripMenuItem.Text = "Open Clear Sim";
            this.openClearSimToolStripMenuItem.Click += new System.EventHandler(this.openClearSimToolStripMenuItem_Click);
            // 
            // addMeshToolStripMenuItem
            // 
            this.addMeshToolStripMenuItem.Name = "addMeshToolStripMenuItem";
            this.addMeshToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.F2;
            this.addMeshToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.addMeshToolStripMenuItem.Text = "Add Mesh";
            this.addMeshToolStripMenuItem.Click += new System.EventHandler(this.addToolStripMenuItem_Click);
            // 
            // toolStripMenuItem2
            // 
            this.toolStripMenuItem2.Name = "toolStripMenuItem2";
            this.toolStripMenuItem2.Size = new System.Drawing.Size(443, 6);
            // 
            // saveInitialStateToolStripMenuItem
            // 
            this.saveInitialStateToolStripMenuItem.Name = "saveInitialStateToolStripMenuItem";
            this.saveInitialStateToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.S)));
            this.saveInitialStateToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.saveInitialStateToolStripMenuItem.Text = "Save Initial State";
            this.saveInitialStateToolStripMenuItem.Click += new System.EventHandler(this.saveInitialStateToolStripMenuItem_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(443, 6);
            // 
            // takeScreenshotToolStripMenuItem
            // 
            this.takeScreenshotToolStripMenuItem.Name = "takeScreenshotToolStripMenuItem";
            this.takeScreenshotToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.F12;
            this.takeScreenshotToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.takeScreenshotToolStripMenuItem.Text = "Take Screenshot";
            this.takeScreenshotToolStripMenuItem.Click += new System.EventHandler(this.takeScreenshotToolStripMenuItem_Click);
            // 
            // renderSimulationToolStripMenuItem
            // 
            this.renderSimulationToolStripMenuItem.Name = "renderSimulationToolStripMenuItem";
            this.renderSimulationToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.F12)));
            this.renderSimulationToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.renderSimulationToolStripMenuItem.Text = "Render Simulation";
            this.renderSimulationToolStripMenuItem.Click += new System.EventHandler(this.renderSimulationToolStripMenuItem_Click);
            // 
            // writeCSVToolStripMenuItem
            // 
            this.writeCSVToolStripMenuItem.Name = "writeCSVToolStripMenuItem";
            this.writeCSVToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.F11;
            this.writeCSVToolStripMenuItem.Size = new System.Drawing.Size(446, 38);
            this.writeCSVToolStripMenuItem.Text = "Write CSV";
            this.writeCSVToolStripMenuItem.Click += new System.EventHandler(this.writeCSVToolStripMenuItem_Click);
            // 
            // simulationToolStripMenuItem
            // 
            this.simulationToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.continueFromLastToolStripMenuItem,
            this.oneStepToolStripMenuItem});
            this.simulationToolStripMenuItem.Name = "simulationToolStripMenuItem";
            this.simulationToolStripMenuItem.Size = new System.Drawing.Size(141, 36);
            this.simulationToolStripMenuItem.Text = "Simulation";
            // 
            // continueFromLastToolStripMenuItem
            // 
            this.continueFromLastToolStripMenuItem.Name = "continueFromLastToolStripMenuItem";
            this.continueFromLastToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.F10;
            this.continueFromLastToolStripMenuItem.Size = new System.Drawing.Size(336, 38);
            this.continueFromLastToolStripMenuItem.Text = "Continue/Pause";
            this.continueFromLastToolStripMenuItem.Click += new System.EventHandler(this.continueFromLastToolStripMenuItem_Click);
            // 
            // oneStepToolStripMenuItem
            // 
            this.oneStepToolStripMenuItem.Name = "oneStepToolStripMenuItem";
            this.oneStepToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.F11;
            this.oneStepToolStripMenuItem.Size = new System.Drawing.Size(336, 38);
            this.oneStepToolStripMenuItem.Text = "One step";
            this.oneStepToolStripMenuItem.Click += new System.EventHandler(this.oneStepToolStripMenuItem_Click);
            // 
            // toolsToolStripMenuItem
            // 
            this.toolsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.analysisToolStripMenuItem,
            this.resizeX01ToolStripMenuItem,
            this.reshapeToolStripMenuItem,
            this.eraseSubsequentStepsToolStripMenuItem,
            this.toolStripMenuItem3,
            this.pPRRelationsToolStripMenuItem});
            this.toolsToolStripMenuItem.Name = "toolsToolStripMenuItem";
            this.toolsToolStripMenuItem.Size = new System.Drawing.Size(82, 36);
            this.toolsToolStripMenuItem.Text = "Tools";
            // 
            // analysisToolStripMenuItem
            // 
            this.analysisToolStripMenuItem.Name = "analysisToolStripMenuItem";
            this.analysisToolStripMenuItem.ShortcutKeys = System.Windows.Forms.Keys.F3;
            this.analysisToolStripMenuItem.Size = new System.Drawing.Size(368, 38);
            this.analysisToolStripMenuItem.Text = "Analysis";
            this.analysisToolStripMenuItem.Click += new System.EventHandler(this.analysisToolStripMenuItem_Click);
            // 
            // resizeX01ToolStripMenuItem
            // 
            this.resizeX01ToolStripMenuItem.Name = "resizeX01ToolStripMenuItem";
            this.resizeX01ToolStripMenuItem.Size = new System.Drawing.Size(368, 38);
            this.resizeX01ToolStripMenuItem.Text = "Resize x0.1";
            this.resizeX01ToolStripMenuItem.Click += new System.EventHandler(this.resizeX01ToolStripMenuItem_Click);
            // 
            // reshapeToolStripMenuItem
            // 
            this.reshapeToolStripMenuItem.Name = "reshapeToolStripMenuItem";
            this.reshapeToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.R)));
            this.reshapeToolStripMenuItem.Size = new System.Drawing.Size(368, 38);
            this.reshapeToolStripMenuItem.Text = "Reshape";
            this.reshapeToolStripMenuItem.Click += new System.EventHandler(this.reshapeToolStripMenuItem_Click);
            // 
            // eraseSubsequentStepsToolStripMenuItem
            // 
            this.eraseSubsequentStepsToolStripMenuItem.Name = "eraseSubsequentStepsToolStripMenuItem";
            this.eraseSubsequentStepsToolStripMenuItem.Size = new System.Drawing.Size(368, 38);
            this.eraseSubsequentStepsToolStripMenuItem.Text = "Erase Subsequent Steps";
            this.eraseSubsequentStepsToolStripMenuItem.Click += new System.EventHandler(this.eraseSubsequentStepsToolStripMenuItem_Click);
            // 
            // toolStripMenuItem3
            // 
            this.toolStripMenuItem3.Name = "toolStripMenuItem3";
            this.toolStripMenuItem3.Size = new System.Drawing.Size(365, 6);
            // 
            // pPRRelationsToolStripMenuItem
            // 
            this.pPRRelationsToolStripMenuItem.Name = "pPRRelationsToolStripMenuItem";
            this.pPRRelationsToolStripMenuItem.Size = new System.Drawing.Size(368, 38);
            this.pPRRelationsToolStripMenuItem.Text = "PPR Relations...";
            this.pPRRelationsToolStripMenuItem.Click += new System.EventHandler(this.pPRRelationsToolStripMenuItem_Click);
            // 
            // statusStrip1
            // 
            this.statusStrip1.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.tssStatus,
            this.tssCurrentFrame});
            this.statusStrip1.Location = new System.Drawing.Point(0, 485);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(1012, 50);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // tssStatus
            // 
            this.tssStatus.Name = "tssStatus";
            this.tssStatus.Size = new System.Drawing.Size(33, 45);
            this.tssStatus.Text = "-";
            // 
            // tssCurrentFrame
            // 
            this.tssCurrentFrame.Name = "tssCurrentFrame";
            this.tssCurrentFrame.Size = new System.Drawing.Size(33, 45);
            this.tssCurrentFrame.Text = "-";
            // 
            // glControl1
            // 
            this.glControl1.BackColor = System.Drawing.Color.Black;
            this.glControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.glControl1.Location = new System.Drawing.Point(0, 0);
            this.glControl1.Margin = new System.Windows.Forms.Padding(7, 6, 7, 6);
            this.glControl1.Name = "glControl1";
            this.glControl1.Size = new System.Drawing.Size(830, 312);
            this.glControl1.TabIndex = 0;
            this.glControl1.VSync = false;
            this.glControl1.Paint += new System.Windows.Forms.PaintEventHandler(this.glControl1_Paint);
            this.glControl1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.glControl1_MouseDown);
            this.glControl1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.glControl1_MouseMove);
            this.glControl1.Resize += new System.EventHandler(this.glControl1_Resize);
            // 
            // backgroundWorker1
            // 
            this.backgroundWorker1.WorkerReportsProgress = true;
            this.backgroundWorker1.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorker1_DoWork);
            this.backgroundWorker1.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.backgroundWorker1_ProgressChanged);
            this.backgroundWorker1.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.backgroundWorker1_RunWorkerCompleted);
            // 
            // trackBar1
            // 
            this.trackBar1.Dock = System.Windows.Forms.DockStyle.Top;
            this.trackBar1.Enabled = false;
            this.trackBar1.LargeChange = 1;
            this.trackBar1.Location = new System.Drawing.Point(0, 39);
            this.trackBar1.Margin = new System.Windows.Forms.Padding(2);
            this.trackBar1.Name = "trackBar1";
            this.trackBar1.Size = new System.Drawing.Size(830, 90);
            this.trackBar1.TabIndex = 0;
            this.trackBar1.ValueChanged += new System.EventHandler(this.trackBar1_ValueChanged);
            // 
            // pg
            // 
            this.pg.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pg.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pg.LineColor = System.Drawing.SystemColors.ControlDark;
            this.pg.Location = new System.Drawing.Point(3, 118);
            this.pg.Name = "pg";
            this.pg.Size = new System.Drawing.Size(172, 324);
            this.pg.TabIndex = 9;
            this.pg.ToolbarVisible = false;
            // 
            // tsDisplayOptions
            // 
            this.tsDisplayOptions.Font = new System.Drawing.Font("Segoe UI", 9F);
            this.tsDisplayOptions.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.tsDisplayOptions.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.tsbRenderPanel,
            this.toolStripSeparator5,
            this.tsbPhi,
            this.tsbAxes,
            this.tsbRotate,
            this.tsbPreviewMode});
            this.tsDisplayOptions.Location = new System.Drawing.Point(0, 0);
            this.tsDisplayOptions.Name = "tsDisplayOptions";
            this.tsDisplayOptions.Size = new System.Drawing.Size(830, 39);
            this.tsDisplayOptions.TabIndex = 10;
            this.tsDisplayOptions.Text = "toolStrip3";
            // 
            // tsbRenderPanel
            // 
            this.tsbRenderPanel.Checked = true;
            this.tsbRenderPanel.CheckOnClick = true;
            this.tsbRenderPanel.CheckState = System.Windows.Forms.CheckState.Checked;
            this.tsbRenderPanel.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbRenderPanel.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbRenderPanel.Name = "tsbRenderPanel";
            this.tsbRenderPanel.Size = new System.Drawing.Size(128, 36);
            this.tsbRenderPanel.Text = "Rendering";
            // 
            // toolStripSeparator5
            // 
            this.toolStripSeparator5.Name = "toolStripSeparator5";
            this.toolStripSeparator5.Size = new System.Drawing.Size(6, 39);
            // 
            // tsbPhi
            // 
            this.tsbPhi.CheckOnClick = true;
            this.tsbPhi.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbPhi.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbPhi.Name = "tsbPhi";
            this.tsbPhi.Size = new System.Drawing.Size(52, 36);
            this.tsbPhi.Text = "Phi";
            this.tsbPhi.Click += new System.EventHandler(this.tsbInvalidate_Click);
            // 
            // tsbAxes
            // 
            this.tsbAxes.CheckOnClick = true;
            this.tsbAxes.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbAxes.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbAxes.Name = "tsbAxes";
            this.tsbAxes.Size = new System.Drawing.Size(68, 36);
            this.tsbAxes.Text = "Axes";
            // 
            // tsbRotate
            // 
            this.tsbRotate.CheckOnClick = true;
            this.tsbRotate.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbRotate.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbRotate.Name = "tsbRotate";
            this.tsbRotate.Size = new System.Drawing.Size(87, 36);
            this.tsbRotate.Text = "Rotate";
            this.tsbRotate.Click += new System.EventHandler(this.tsbInvalidate_Click);
            // 
            // tsbPreviewMode
            // 
            this.tsbPreviewMode.CheckOnClick = true;
            this.tsbPreviewMode.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbPreviewMode.Image = ((System.Drawing.Image)(resources.GetObject("tsbPreviewMode.Image")));
            this.tsbPreviewMode.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbPreviewMode.Name = "tsbPreviewMode";
            this.tsbPreviewMode.Size = new System.Drawing.Size(101, 36);
            this.tsbPreviewMode.Text = "Preview";
            this.tsbPreviewMode.Click += new System.EventHandler(this.tsbPreviewMode_Click);
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 2;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 17.64706F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 82.35294F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel1.Controls.Add(this.panel1, 1, 0);
            this.tableLayoutPanel1.Controls.Add(this.pg, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.treeView1, 0, 0);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 40);
            this.tableLayoutPanel1.Margin = new System.Windows.Forms.Padding(2);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 25.88997F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 74.11003F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1012, 445);
            this.tableLayoutPanel1.TabIndex = 11;
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.panel2);
            this.panel1.Controls.Add(this.trackBar1);
            this.panel1.Controls.Add(this.tsDisplayOptions);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(180, 2);
            this.panel1.Margin = new System.Windows.Forms.Padding(2);
            this.panel1.Name = "panel1";
            this.tableLayoutPanel1.SetRowSpan(this.panel1, 2);
            this.panel1.Size = new System.Drawing.Size(830, 441);
            this.panel1.TabIndex = 12;
            // 
            // panel2
            // 
            this.panel2.AutoScroll = true;
            this.panel2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.panel2.Controls.Add(this.glControl1);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel2.Location = new System.Drawing.Point(0, 129);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(830, 312);
            this.panel2.TabIndex = 12;
            // 
            // treeView1
            // 
            this.treeView1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.treeView1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F);
            this.treeView1.Location = new System.Drawing.Point(2, 2);
            this.treeView1.Margin = new System.Windows.Forms.Padding(2);
            this.treeView1.Name = "treeView1";
            this.treeView1.ShowNodeToolTips = true;
            this.treeView1.Size = new System.Drawing.Size(174, 111);
            this.treeView1.TabIndex = 10;
            this.treeView1.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.treeView1_AfterSelect);
            // 
            // cmsTranslations
            // 
            this.cmsTranslations.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.cmsTranslations.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addToolStripMenuItem1,
            this.refreshToolStripMenuItem});
            this.cmsTranslations.Name = "cmsTranslations";
            this.cmsTranslations.Size = new System.Drawing.Size(195, 80);
            // 
            // addToolStripMenuItem1
            // 
            this.addToolStripMenuItem1.Name = "addToolStripMenuItem1";
            this.addToolStripMenuItem1.Size = new System.Drawing.Size(194, 38);
            this.addToolStripMenuItem1.Text = "Add";
            this.addToolStripMenuItem1.Click += new System.EventHandler(this.addToolStripMenuItem1_Click);
            // 
            // refreshToolStripMenuItem
            // 
            this.refreshToolStripMenuItem.Name = "refreshToolStripMenuItem";
            this.refreshToolStripMenuItem.Size = new System.Drawing.Size(194, 38);
            this.refreshToolStripMenuItem.Text = "Refresh";
            this.refreshToolStripMenuItem.Click += new System.EventHandler(this.refreshToolStripMenuItem_Click);
            // 
            // cmsOneTranslation
            // 
            this.cmsOneTranslation.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.cmsOneTranslation.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.removeToolStripMenuItem1});
            this.cmsOneTranslation.Name = "cmsTranslations";
            this.cmsOneTranslation.Size = new System.Drawing.Size(202, 42);
            // 
            // removeToolStripMenuItem1
            // 
            this.removeToolStripMenuItem1.Name = "removeToolStripMenuItem1";
            this.removeToolStripMenuItem1.Size = new System.Drawing.Size(201, 38);
            this.removeToolStripMenuItem1.Text = "Remove";
            this.removeToolStripMenuItem1.Click += new System.EventHandler(this.removeToolStripMenuItem1_Click);
            // 
            // cmsMesh
            // 
            this.cmsMesh.ImageScalingSize = new System.Drawing.Size(32, 32);
            this.cmsMesh.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.cZsAndCapsToolStripMenuItem,
            this.cZsToolStripMenuItem1,
            this.splitToolStripMenuItem1,
            this.makeTorusToolStripMenuItem1,
            this.toolStripMenuItem6,
            this.resizeToolStripMenuItem,
            this.rotate90DegToolStripMenuItem,
            this.toolStripMenuItem7,
            this.removeToolStripMenuItem2});
            this.cmsMesh.Name = "cmsMesh";
            this.cmsMesh.Size = new System.Drawing.Size(260, 282);
            // 
            // cZsAndCapsToolStripMenuItem
            // 
            this.cZsAndCapsToolStripMenuItem.Name = "cZsAndCapsToolStripMenuItem";
            this.cZsAndCapsToolStripMenuItem.Size = new System.Drawing.Size(259, 38);
            this.cZsAndCapsToolStripMenuItem.Text = "CZs and Caps";
            this.cZsAndCapsToolStripMenuItem.Click += new System.EventHandler(this.cZsAndCapsToolStripMenuItem_Click);
            // 
            // cZsToolStripMenuItem1
            // 
            this.cZsToolStripMenuItem1.Name = "cZsToolStripMenuItem1";
            this.cZsToolStripMenuItem1.Size = new System.Drawing.Size(259, 38);
            this.cZsToolStripMenuItem1.Text = "CZs";
            this.cZsToolStripMenuItem1.Click += new System.EventHandler(this.cZsToolStripMenuItem1_Click);
            // 
            // splitToolStripMenuItem1
            // 
            this.splitToolStripMenuItem1.Name = "splitToolStripMenuItem1";
            this.splitToolStripMenuItem1.Size = new System.Drawing.Size(259, 38);
            this.splitToolStripMenuItem1.Text = "Split";
            this.splitToolStripMenuItem1.Click += new System.EventHandler(this.splitToolStripMenuItem1_Click);
            // 
            // makeTorusToolStripMenuItem1
            // 
            this.makeTorusToolStripMenuItem1.Name = "makeTorusToolStripMenuItem1";
            this.makeTorusToolStripMenuItem1.Size = new System.Drawing.Size(259, 38);
            this.makeTorusToolStripMenuItem1.Text = "Make Torus";
            this.makeTorusToolStripMenuItem1.Click += new System.EventHandler(this.makeTorusToolStripMenuItem1_Click);
            // 
            // toolStripMenuItem6
            // 
            this.toolStripMenuItem6.Name = "toolStripMenuItem6";
            this.toolStripMenuItem6.Size = new System.Drawing.Size(256, 6);
            // 
            // resizeToolStripMenuItem
            // 
            this.resizeToolStripMenuItem.Name = "resizeToolStripMenuItem";
            this.resizeToolStripMenuItem.Size = new System.Drawing.Size(259, 38);
            this.resizeToolStripMenuItem.Text = "Resize";
            this.resizeToolStripMenuItem.Click += new System.EventHandler(this.resizeToolStripMenuItem_Click);
            // 
            // rotate90DegToolStripMenuItem
            // 
            this.rotate90DegToolStripMenuItem.Name = "rotate90DegToolStripMenuItem";
            this.rotate90DegToolStripMenuItem.Size = new System.Drawing.Size(259, 38);
            this.rotate90DegToolStripMenuItem.Text = "Rotate90Deg";
            this.rotate90DegToolStripMenuItem.Click += new System.EventHandler(this.rotate90DegToolStripMenuItem_Click);
            // 
            // toolStripMenuItem7
            // 
            this.toolStripMenuItem7.Name = "toolStripMenuItem7";
            this.toolStripMenuItem7.Size = new System.Drawing.Size(256, 6);
            // 
            // removeToolStripMenuItem2
            // 
            this.removeToolStripMenuItem2.Name = "removeToolStripMenuItem2";
            this.removeToolStripMenuItem2.Size = new System.Drawing.Size(259, 38);
            this.removeToolStripMenuItem2.Text = "Remove";
            this.removeToolStripMenuItem2.Click += new System.EventHandler(this.removeToolStripMenuItem2_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(13F, 26F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1012, 535);
            this.Controls.Add(this.tableLayoutPanel1);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.menuStrip1);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Form1";
            this.ShowIcon = false;
            this.Text = "icFlow";
            this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBar1)).EndInit();
            this.tsDisplayOptions.ResumeLayout(false);
            this.tsDisplayOptions.PerformLayout();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.cmsTranslations.ResumeLayout(false);
            this.cmsOneTranslation.ResumeLayout(false);
            this.cmsMesh.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel tssStatus;
        private System.ComponentModel.BackgroundWorker backgroundWorker1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private OpenTK.GLControl glControl1;
        private System.Windows.Forms.ToolStripMenuItem simulationToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem continueFromLastToolStripMenuItem;
        private System.Windows.Forms.ToolStripStatusLabel tssCurrentFrame;
        private System.Windows.Forms.TrackBar trackBar1;
        private System.Windows.Forms.ToolStripMenuItem oneStepToolStripMenuItem;
        private System.Windows.Forms.PropertyGrid pg;
        private System.Windows.Forms.ToolStripMenuItem toolsToolStripMenuItem;
        private System.Windows.Forms.ToolStrip tsDisplayOptions;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem openSimulationToolStripMenuItem;
        private System.Windows.Forms.ToolStripButton tsbRotate;
        private System.Windows.Forms.ToolStripButton tsbPhi;
        private System.Windows.Forms.ToolStripButton tsbAxes;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator5;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem2;
        private System.Windows.Forms.ToolStripMenuItem saveInitialStateToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem analysisToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem takeScreenshotToolStripMenuItem;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.TreeView treeView1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.ContextMenuStrip cmsTranslations;
        private System.Windows.Forms.ToolStripMenuItem addToolStripMenuItem1;
        private System.Windows.Forms.ContextMenuStrip cmsOneTranslation;
        private System.Windows.Forms.ToolStripMenuItem removeToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem addMeshToolStripMenuItem;
        private System.Windows.Forms.ContextMenuStrip cmsMesh;
        private System.Windows.Forms.ToolStripMenuItem cZsAndCapsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem cZsToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem splitToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem makeTorusToolStripMenuItem1;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem6;
        private System.Windows.Forms.ToolStripMenuItem resizeToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem7;
        private System.Windows.Forms.ToolStripMenuItem removeToolStripMenuItem2;
        private System.Windows.Forms.ToolStripMenuItem refreshToolStripMenuItem;
        private System.Windows.Forms.ToolStripButton tsbPreviewMode;
        private System.Windows.Forms.ToolStripMenuItem openClearSimToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem renderSimulationToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem resizeX01ToolStripMenuItem;
        private System.Windows.Forms.OpenFileDialog openFileDialogMeshes;
        private System.Windows.Forms.ToolStripMenuItem rotate90DegToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem writeCSVToolStripMenuItem;
        private System.Windows.Forms.ToolStripButton tsbRenderPanel;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.ToolStripMenuItem reshapeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem eraseSubsequentStepsToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem3;
        private System.Windows.Forms.ToolStripMenuItem pPRRelationsToolStripMenuItem;
    }
}

