using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Windows.Forms;

namespace icFlow
{
    public partial class FormAnalysis : Form
    {
        public FrameInfo.FrameSummary smr;

        public FormAnalysis()
        {
            InitializeComponent();
        }

        enum ShowStates { Step = 0, Time = 1, Strain = 2 }
        ShowStates currentState = ShowStates.Step;
        private void tsbStep_Click(object sender, EventArgs e)
        {
            currentState = (ShowStates)(((int)currentState + 1) % 3);
            tsbStep.Text = currentState.ToString();
            ShowData();
        }

        private void FormAnalysis_Load(object sender, EventArgs e)
        {
            ShowData();
            propertyGrid1.SelectedObject = smr;
        }

        void ShowData()
        {
            if (smr == null) return;

            if (currentState == ShowStates.Step)
            {
                chart1.Series["StressStrain"].Points.DataBindXY(smr.timeStep, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.timeStep, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.timeStep, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.timeStep, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.timeStep, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.timeStep, smr.indenterForce);
            }
            else if (currentState == ShowStates.Strain)
            {
                chart1.Series["StressStrain"].Points.DataBindXY(smr.strain, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.strain, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.strain, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.strain, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.strain, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.strain, smr.indenterForce);
            } else if(currentState == ShowStates.Time)
            {
                chart1.Series["StressStrain"].Points.DataBindXY(smr.time, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.time, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.time, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.time, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.time, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.time, smr.indenterForce);
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
        }

        private void tsbWriteCSV_Click(object sender, EventArgs e)
        {

        }
    }
}
