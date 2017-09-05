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
using icFlow;

namespace BatchEdit
{
    public partial class Form1 : Form
    {
        string simsFolder;
        ImplicitModel3 model;
        ModelPrms[] simParams;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            model = new ImplicitModel3();
            Populate();
        }

        void DisplaySelection(ModelPrms prms)
        {
            SimData sd = (SimData)prms.Tag;
            FrameInfo.FrameSummary smr = sd.summary;

            if(smr.stress != null)
            {
                propertyGrid1.SelectedObject = smr;
                if (tsbStep.Checked)
                {
                    chart1.Series["StressStrain"].Points.DataBindXY(smr.timeStep, smr.stress);
                    chart1.Series["DamageAndFail"].Points.DataBindXY(smr.timeStep, smr.damageAndFail);
                    chart1.Series["Fail"].Points.DataBindXY(smr.timeStep, smr.fail);
                    chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.timeStep, smr.logOfTimeScaleFactor);
                    chart1.Series["Iterations"].Points.DataBindXY(smr.timeStep, smr.iterations);
                    chart1.Series["IndenterForce"].Points.DataBindXY(smr.timeStep, smr.indenterForce);
                }
                else {
                    chart1.Series["StressStrain"].Points.DataBindXY(smr.strain, smr.stress);
                    chart1.Series["DamageAndFail"].Points.DataBindXY(smr.strain, smr.damageAndFail);
                    chart1.Series["Fail"].Points.DataBindXY(smr.strain, smr.fail);
                    chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.strain, smr.logOfTimeScaleFactor);
                    chart1.Series["Iterations"].Points.DataBindXY(smr.strain, smr.iterations);
                    chart1.Series["IndenterForce"].Points.DataBindXY(smr.strain, smr.indenterForce);
                }

                double[] pie_chart_values = new double[5];
                string[] pie_chart_titles = new string[5];
                pie_chart_values[0] = smr.MKLSolve;
                pie_chart_values[1] = smr.Collisions;
                pie_chart_values[2] = smr.InternalForces;
                pie_chart_values[3] = smr.Other;
                pie_chart_values[4] = smr.Discarded;

                pie_chart_titles[0] = $"Solve {smr.MKLSolve * 100:##.0}%";
                pie_chart_titles[1] = $"Coll {smr.Collisions * 100:##.0}%";
                pie_chart_titles[2] = $"Force {smr.InternalForces * 100:##.0}%";
                pie_chart_titles[3] = $"Other {smr.Other * 100:##.0}%";
                pie_chart_titles[4] = $"Discard {smr.Discarded * 100:##.0}%";

                chart2.Series["Benchmarking"].Points.DataBindXY(pie_chart_titles, pie_chart_values);
            } else
            {
                chart1.Series["StressStrain"].Points.Clear();
                chart1.Series["DamageAndFail"].Points.Clear();
                chart1.Series["Fail"].Points.Clear();
                chart1.Series["LogOfTimeScaleFactor"].Points.Clear();
                chart1.Series["Iterations"].Points.Clear();
                chart1.Series["IndenterForce"].Points.Clear();
                chart2.Series["Benchmarking"].Points.Clear();
                propertyGrid1.SelectedObject = null;
            }
        }

        void Populate()
        {
            simsFolder = AppDomain.CurrentDomain.BaseDirectory + "_sims\\";
            string[] sims = Directory.GetDirectories(simsFolder);
            simParams = new ModelPrms[sims.Length];
            for(int i=0;i<sims.Length;i++)
            {
                model.saveFolder = sims[i];
                string newName = Path.GetFileName(model.saveFolder);
                Console.WriteLine(newName);
                model.LoadSimulation();
                simParams[i] = model.prms;
                SimData sd = new SimData();
                sd.path = model.saveFolder;
                sd.summary = new FrameInfo.FrameSummary(model.allFrames, model.prms.name, model.prms.GrainSize);
                simParams[i].Tag = sd;
            }
            dataGridView1.DataSource = simParams;
        }

        private void dataGridView1_SelectionChanged(object sender, EventArgs e)
        {
            if (dataGridView1.CurrentCell == null) return;
            int rowidx = dataGridView1.CurrentCell.RowIndex;
            ModelPrms prms = simParams[rowidx];
            DisplaySelection(prms);
        }
    }
}
