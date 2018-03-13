using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Windows.Forms;
using System.Drawing;
using System.Diagnostics;
using System.Linq;


namespace icFlow
{
    public partial class FormAnalysis : Form
    {
        public FrameInfo.FrameSummary smr;
        public FormAnalysis()
        {
            InitializeComponent();
        }

        public bool forRendering = false; // set to true if the plots are made for animation rendering
        public double fromTime = 0, toTime = 0, maxTime = 0; // these should be set if forRendering is set

        enum ShowStates { Step = 0, Time = 1, Strain = 2 }
        ShowStates currentState = ShowStates.Time;
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

        public void ShowData()
        {
            if (smr == null) return;

            if(forRendering)
            {
                double maxStress = smr.stress.Max();
                int trim_idx = 0;
                while (trim_idx < smr.time.Length && smr.time[trim_idx] <= toTime) trim_idx++;

                if(trim_idx < smr.time.Length-1 && trim_idx != 0)
                {
                    Array.Resize(ref smr.time, trim_idx);
                    Array.Resize(ref smr.stress, trim_idx);
                    Array.Resize(ref smr.damageAndFail, trim_idx);
                    Array.Resize(ref smr.fail, trim_idx);
                    Array.Resize(ref smr.logOfTimeScaleFactor, trim_idx);
                    Array.Resize(ref smr.time, trim_idx);
                    Array.Resize(ref smr.iterations, trim_idx);
                    Array.Resize(ref smr.indenterForce, trim_idx);
                    Array.Resize(ref smr.CZ_Softening, trim_idx);
                    Array.Resize(ref smr.CZ_Mixed, trim_idx);
                    Array.Resize(ref smr.CZ_UnloadingReloading, trim_idx);
                    Array.Resize(ref smr.CZ_Fail, trim_idx);
                }
                

                // trim down the summary arrays
                chart1.Series["StressStrain"].Points.DataBindXY(smr.time, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.time, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.time, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.time, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.time, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.time, smr.indenterForce);

                chart1.Series["CZ_Softening"].Points.DataBindXY(smr.time, smr.CZ_Softening);
                chart1.Series["CZ_Mixed"].Points.DataBindXY(smr.time, smr.CZ_Mixed);
                chart1.Series["CZ_UnloadingReloading"].Points.DataBindXY(smr.time, smr.CZ_UnloadingReloading);
                chart1.Series["CZ_Failed"].Points.DataBindXY(smr.time, smr.CZ_Fail);

                chart1.ChartAreas["ChartArea1"].Axes[0].Minimum = fromTime;
                chart1.ChartAreas["ChartArea2"].Axes[0].Minimum = fromTime;
                chart1.ChartAreas["ChartArea3"].Axes[0].Minimum = fromTime;
                chart1.ChartAreas["ChartArea4"].Axes[0].Minimum = fromTime;

                chart1.ChartAreas["ChartArea1"].Axes[0].Maximum = maxTime;
                chart1.ChartAreas["ChartArea2"].Axes[0].Maximum = maxTime;
                chart1.ChartAreas["ChartArea3"].Axes[0].Maximum = maxTime;
                chart1.ChartAreas["ChartArea4"].Axes[0].Maximum = maxTime;

                int exp = (int)Math.Ceiling(Math.Log10(maxStress)-1);
                int oneDigit = (int)Math.Ceiling(maxStress / Math.Pow(10, exp));

                chart1.ChartAreas["ChartArea1"].Axes[1].Maximum = (double)oneDigit*Math.Pow(10,exp);


                chart1.ChartAreas["ChartArea2"].Visible = false;
                chart1.ChartAreas["ChartArea3"].Visible = false;
                chart1.Series["DamageAndFail"].Enabled = false;
                chart1.Series["Fail"].Enabled = false;
                chart1.Series["LogOfTimeScaleFactor"].Enabled = false;
                chart1.Series["Iterations"].Enabled = false;
                chart1.Series["IndenterForce"].Enabled = false;

            }

            else if (currentState == ShowStates.Step)
            {
                chart1.Series["StressStrain"].Points.DataBindXY(smr.timeStep, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.timeStep, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.timeStep, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.timeStep, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.timeStep, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.timeStep, smr.indenterForce);

                chart1.Series["CZ_Softening"].Points.DataBindXY(smr.timeStep, smr.CZ_Softening);
                chart1.Series["CZ_Mixed"].Points.DataBindXY(smr.timeStep, smr.CZ_Mixed);
                chart1.Series["CZ_UnloadingReloading"].Points.DataBindXY(smr.timeStep, smr.CZ_UnloadingReloading);
                chart1.Series["CZ_Failed"].Points.DataBindXY(smr.timeStep, smr.CZ_Fail);
            }
            else if (currentState == ShowStates.Strain)
            {
                chart1.Series["StressStrain"].Points.DataBindXY(smr.strain, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.strain, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.strain, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.strain, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.strain, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.strain, smr.indenterForce);

                chart1.Series["CZ_Softening"].Points.DataBindXY(smr.strain, smr.CZ_Softening);
                chart1.Series["CZ_Mixed"].Points.DataBindXY(smr.strain, smr.CZ_Mixed);
                chart1.Series["CZ_UnloadingReloading"].Points.DataBindXY(smr.strain, smr.CZ_UnloadingReloading);
                chart1.Series["CZ_Failed"].Points.DataBindXY(smr.strain, smr.CZ_Fail);

            }
            else if(currentState == ShowStates.Time)
            {
                chart1.Series["StressStrain"].Points.DataBindXY(smr.time, smr.stress);
                chart1.Series["DamageAndFail"].Points.DataBindXY(smr.time, smr.damageAndFail);
                chart1.Series["Fail"].Points.DataBindXY(smr.time, smr.fail);
                chart1.Series["LogOfTimeScaleFactor"].Points.DataBindXY(smr.time, smr.logOfTimeScaleFactor);
                chart1.Series["Iterations"].Points.DataBindXY(smr.time, smr.iterations);
                chart1.Series["IndenterForce"].Points.DataBindXY(smr.time, smr.indenterForce);

                chart1.Series["CZ_Softening"].Points.DataBindXY(smr.time, smr.CZ_Softening);
                chart1.Series["CZ_Mixed"].Points.DataBindXY(smr.time, smr.CZ_Mixed);
                chart1.Series["CZ_UnloadingReloading"].Points.DataBindXY(smr.time, smr.CZ_UnloadingReloading);
                chart1.Series["CZ_Failed"].Points.DataBindXY(smr.time, smr.CZ_Fail);
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
