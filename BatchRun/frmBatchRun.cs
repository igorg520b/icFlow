using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using v1Library;
using System.Diagnostics;
using System.Xml.Serialization;
using MoreLinq;

namespace BatchRun
{
    public partial class frmBatchRun : Form
    {
        List<Simulation> sims = new List<Simulation>();
        Queue<Simulation> queue = new Queue<Simulation>();
        bool running = false;
        bool askToStop = false;
        string path;

        public frmBatchRun()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            ImplicitModel3.Initialize();
            string[] result = File.ReadAllLines(@"path.txt");
            ImplicitModel3.appPath = result[0];
            path = $"{ImplicitModel3.appPath}{ImplicitModel3.savePath}";
            CreateSimList();
        }

        void CreateSimList()
        {
            string[] simList = Directory.GetDirectories(path);
            sims.Clear();
            foreach(string str in simList)
            {
                Simulation sim = new Simulation();
                sim.Name = Path.GetFileName(str);
                sims.Add(sim);
            }
            AnalyzeSimList();
            // assign queue numbers
            int qNumber = 1;
            foreach (Simulation sim in sims)
            {
                if (sim.status == Simulation.Status.Completed) sim.QNumber = -1;
                else sim.QNumber = qNumber++;
            }
            dataGridView1.DataSource = sims;
        }

        void AnalyzeSim(Simulation sim)
        {
            // see if frame file exists
            string lfiPath = $"{path}\\{sim.Name}\\{sim.Name}.lfi";
            if(File.Exists(lfiPath))
            {
                // read lfi
                StreamReader str = new StreamReader(lfiPath);
                XmlSerializer xs = new XmlSerializer(typeof(List<FrameInfo>));
                List<FrameInfo> allFrames = (List<FrameInfo>)xs.Deserialize(str);
                str.Close();
                if(allFrames.Count == 0) { sim.status = Simulation.Status.Clean; return; }

                // determine status
                List<FrameInfo> fractureFrames = allFrames.FindAll(x => x.FractureDetected).OrderBy(fi => fi.StepNumber).ToList();
                if(fractureFrames.Count == 0)
                {
                    // not completed yet
                    sim.status = Simulation.Status.Paused;
                } else
                {
                    // completed
                    sim.status = Simulation.Status.Completed;
                    FrameInfo fi = fractureFrames[0]; // first frame where fracture detected
                    sim.FrStep = fi.StepNumber;
                    sim.FrTime = fi.SimulationTime;
                    sim.FrDamage = (double)fi.nCZDamaged / fi.nCZ_Initial;

                    fi = allFrames.MaxBy(x => Math.Abs(x.stress));
                    sim.MaxStress = fi.stress;
                    sim.Strain = fi.strain;
                    sim.Enqueue = false;
                }
                FrameInfo lastFrame = allFrames[allFrames.Count - 1];
                sim.Steps = lastFrame.StepNumber;
                sim.SimTime = lastFrame.SimulationTime;
                long compTimeMilliseconds = allFrames.Sum(x => x.TotalTotal);
                sim.CompTime = new TimeSpan(compTimeMilliseconds * 10000);
            }
            else
            {
                sim.status = Simulation.Status.Clean;
            }
        }

        void AnalyzeSimList()
        {
            foreach (Simulation sim in sims) AnalyzeSim(sim);
        }

        void CreateQueue()
        {
            // only put checkmarked items on the queue
            IOrderedEnumerable<Simulation> queueSims = sims.FindAll(s => (s.Enqueue && s.status != Simulation.Status.Completed)).OrderBy(s => s.QNumber);
            queue.Clear();
            foreach (Simulation sim in queueSims) queue.Enqueue(sim);
            dataGridView1.DataSource = null;
            dataGridView1.DataSource = sims;
        }

        Simulation current = null;
        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            running = true;
            bool simulationFinished;
            // run all items from the queue one by one
            do
            {
                // load sim
                current = queue.Dequeue();
                Stream str = File.OpenRead($"{path}\\{current.Name}\\{current.Name}.params");
                ImplicitModel3.LoadSimulation(str);
                if(current.status == Simulation.Status.Paused) 
                    ImplicitModel3.GoToFrame(ImplicitModel3.allFrames.Count - 1); // resume from last available frame
                do
                {
                    ImplicitModel3.Step();
                    backgroundWorker1.ReportProgress(0);
                    simulationFinished = (ImplicitModel3.cf.StepsRemaining == 0) || ImplicitModel3.cf.StepNumber > 1600;
                } while (!askToStop && !simulationFinished);
                AnalyzeSimList();
                backgroundWorker1.ReportProgress(1);
            } while (queue.Count > 0 && !askToStop);
        }

        private void backgroundWorker1_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            if(e.ProgressPercentage == 0)
            {
                // running steps
                toolStripStatusLabel1.Text = $"Running {current.Name} step {ImplicitModel3.cf.StepNumber}";
            }
            else if(e.ProgressPercentage == 1)
            {
                // switching simulations
                // update details of current sim
                // update datagridview
                dataGridView1.DataSource = null;
                dataGridView1.DataSource = sims;
            }
        }

        private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            askToStop = false;
            running = false;
            toolStripStatusLabel1.Text = "Completed";
        }

        private void updateToolStripMenuItem_Click(object sender, EventArgs e)
        {
            AnalyzeSimList();
            dataGridView1.DataSource = null;
            dataGridView1.DataSource = sims;
        }

        private void runPauseToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if(running)
            {
                askToStop = true;
                toolStripStatusLabel1.Text = "Stopping";
            } else
            {
                CreateQueue();
                askToStop = false;
                backgroundWorker1.RunWorkerAsync();
                toolStripStatusLabel1.Text = "Starting";
            }
        }
    }
}
