using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Collections.Generic;

namespace icFlow
{
    [Serializable]
    public class FrameInfo
    {
        public const int Parts = 1024*32; // initial time step is split into 1024 intervals
        public const int Factor2 = Parts;
        public int nCZFailedThisStep;

        #region Essential
        [Category("_Essential")]
        public int StepNumber { get; set; }
        [Category("_Essential")]
        public double SimulationTime { get; set; }   // what time this frame represents in the simulation
        [Category("_Essential")]
        [Description("Time scale that will be used on next (!) step")]
        public int TimeScaleFactor { get; set; } = 1;     // timestep = initialTimestep * TSF / Parts

        public enum CapReasons { NotCapped, NoConvergence, Divergence, Damage, Fail, None}
        [Category("_Essential")]
        [Description("Explains why the time step is reduced")]
        public CapReasons CapReason { get; set; } = CapReasons.NotCapped;

        [Category("_Essential")]
        public int nCZFailed { get; set; }
        [Category("_Essential")]
        public int nCZDamaged { get; set; }

        [Category("_Essential")]
        public int nCollisions { get; set; }
        #endregion

        #region Revisions for Paper 2
        public int nCZStatusNone { get; set; }
        public int nCZStatusSoftening { get; set; }
        public int nCZStatusUnloadingReloading { get; set; }
        public int nCZStatusMixed { get; set; }

        #endregion

        #region Time
        [Category("Time")]
        public int SimulationIntegerTime { get; set; }   // measures time in 1/1024 intervals of InitialTimeStep
        [Category("Time")]
        public double TimeStep { get; set; } // time difference between current frame and last frame
        [Category("Time")]
        public int StepsWithCurrentFactor { get; set; } // time steps left with current factor (if TSF > 1)

        [Category("Time")]
        [Description("Time scale used for this step")]
        public int TimeScaleFactorThisStep { get; set; } = 1; 

        #endregion

        #region Stats
        [Category("Stats")]
        public int nElems { get; set; }
        [Category("Stats")]
        public int nCZ { get; set; }
        [Category("Stats")]
        public int nCZ_Initial { get; set; }
        [Category("Stats")]
        [Description("Depth of bounding volume hierarchy")]
        public int BVHT_depth { get; set; }
        [Category("Stats")]
        public int ActiveNodes { get; set; }
        #endregion

        #region Benchmarking
        // timing (milliseconds)
        [Category("Benchmarking")]
        public long BVHConstructOrUpdate { get; set; }
        [Category("Benchmarking")]
        public long BVHTraverse { get; set; }
        [Category("Benchmarking")]
        [Description("Elementary tests for collision detection on GPU")]
        public long ElT_GPU { get; set; }
        [Category("Benchmarking")]
        [Description("Elementary tests for collision detection on CPU")]
        public long ElT_CPU { get; set; }
        [Category("Benchmarking")]
        public long CSRStructure { get; set; }
        [Category("Benchmarking")]
        public long KForcePrepare { get; set; }
        [Category("Benchmarking")]
        public long KerElemForce { get; set; }
        [Category("Benchmarking")]
        public long KerCZForce { get; set; }
        [Category("Benchmarking")]
        public long CollForce { get; set; }
        [Category("Benchmarking")]
        public long MKLSolve { get; set; }
        [Category("Benchmarking")]
        [Description("Computational effort on all tasks")]
        public long Total { get; set; } // per frame
        [Category("Benchmarking")]
        [Description("Discarded subsequent frames for all iterations")]
        public long Discarded { get; set; }
        #endregion

        #region CSR
        [Category("CSR")]
        public int CSR_NNZ { get; set; }
        [Category("CSR")]
        public int CSR_N { get; set; }
        [Category("CSR")]
        public string CSR_Mb { get; set; }
        [Category("CSR")]
        public string CSR_alloc { get; set; }
        #endregion

        #region Stress-Strain
        [Category("StressStrain")]
        [Description("This is only applicable to vertically oriented sample")]
        public double stress { get; set; }
        [Category("StressStrain")]
        public double strain { get; set; }
        [Category("StressStrain")]
        [Description("Magnitude of force recorded on all rigid objects that are marked as indenter")]
        public double IndenterForce { get; set; }
        #endregion

        #region Analysis of computation step
        [Category("Analysis")]
        public bool ConvergenceReached { get; set; } // if not, it is a bad sign
        [Category("Analysis")]
        public int IterationsPerformed { get; set; }          // number of iteration it took to produce this frame (excluding discarded attempts)
        [Category("Analysis")]
        public int AttemptsTaken { get; set; }       // 1 if solution converged while Iterations <= maxIterations
        [Category("Analysis")]
        public double RelativeError { get; set; }    // convergence measure
        [Category("Analysis")]
        public double Error0;
        #endregion

        #region Methods

        public FrameInfo SpecialCopy()
        {
            FrameInfo copy = new FrameInfo();
            copy.StepNumber = StepNumber;
            copy.SimulationTime = SimulationTime;
            copy.SimulationIntegerTime = SimulationIntegerTime;
            copy.StepsWithCurrentFactor = StepsWithCurrentFactor;
            copy.TimeScaleFactor = TimeScaleFactor;

            copy.nCZ_Initial = nCZ_Initial;
            copy.nCZFailed = nCZFailed;
            return copy;
        }

        public int MaxIntTimestep() { return Parts - SimulationIntegerTime % Parts; }

        public void IncrementTime(double initialTimestep, int temporaryTimeScale = -1)
        {
            if (temporaryTimeScale == -1)
            {
                int ticks = Parts / TimeScaleFactor;
                SimulationIntegerTime += ticks;
                TimeStep = initialTimestep * (double)ticks / Parts;
                SimulationTime = initialTimestep * (double)SimulationIntegerTime / Parts;
            } else
            {
                int ticks = Parts / temporaryTimeScale;
                SimulationIntegerTime += ticks;
                TimeStep = initialTimestep * (double)ticks / Parts;
                SimulationTime = initialTimestep * (double)SimulationIntegerTime / Parts;
            }
        }

        #endregion

        #region FrameSummary

        public static void WriteCSV(List<FrameInfo> frames, string path)
        {
            StreamWriter sw = new StreamWriter(File.Create(path));
            sw.WriteLine($"Step,Stress,Strain,Force,DamageAndFail,Fail,TimeScaleFactorLog,Iterations, Time, CumulativeComputeTime, nCZSoftening, nCZMixed, nCZUnloadingReloading, nCZFailed");
            long ComputeTime = 0;
            for (int i = 0; i < frames.Count; i++)
            {
                FrameInfo f = frames[i];
                ComputeTime += (f.Total + f.Discarded);
                double damageAndFail = (f.nCZDamaged + f.nCZFailed) / (double)f.nCZ_Initial;
                double fail = (f.nCZFailed) / (double)f.nCZ_Initial;
                sw.WriteLine($"{i},{f.stress},{f.strain},{f.IndenterForce},{damageAndFail},{fail},{Math.Log((double)f.TimeScaleFactor, 2) + 1},{f.IterationsPerformed}, {f.SimulationTime}, {ComputeTime},{f.nCZStatusSoftening},{f.nCZStatusMixed},{f.nCZStatusUnloadingReloading},{f.nCZFailed}");
            }
            sw.Close();
        }

        public class FrameSummary
        {
            #region arrays for plots
            public double[] stress, strain, damageAndFail, fail, timeStep, indenterForce, time;
            public double[] logOfTimeScaleFactor, iterations;
            public double[] CZ_Softening, CZ_Mixed, CZ_UnloadingReloading, CZ_Fail; // counts of CZ status
            #endregion

            #region Essential
            [Category("_Essential")]
            public string Name { get; set; }

            [Category("_Essential")]
            public int TotalSteps { get; set; }

            [Category("_Essential")]
            [Description("Modeled time span in seconds")]
            public double Duration { get; set; }

            [Category("_Essential")]
            [Description("Time spent on computing")]
            public TimeSpan TotCompute { get; set; }

            [Category("_Essential")]
            [Description("Time spent on computing per step")]
            public TimeSpan StepCompute { get; set; }

            [Category("_Essential")]
            [Description("Average number of iterations per successful step")]
            public double AvgIterations { get; set; }
            #endregion

            #region Matrix
            [Category("Matrix")]
            [Description("Minimum number of non-zero values in the matrix")]
            public double MinNNZ { get; set; }

            [Category("Matrix")]
            [Description("Maximum number of non-zero values in the matrix")]
            public double MaxNNZ { get; set; }

            #endregion

            #region Physics
            [Category("Physics")]
            [Description("Max Stress (if recorded)")]
            public double MaxStress { get; set; }

            [Category("Physics")]
            [Description("Max indenter force (if recorded)")]
            public double MaxForce { get; set; }

            [Category("Physics")]
            [Description("Average grain size of deformable object")]
            public double GrainSize { get; set; }
            #endregion

            #region Benchmarking
            [Category("Benchmarking")]
            [Description("Proportion of computational effort")]
            public double MKLSolve { get; set; }

            [Category("Benchmarking")]
            [Description("Proportion of computational effort")]
            public double Collisions { get; set; }

            [Category("Benchmarking")]
            [Description("Proportion of computational effort on CZ and Element forces and force derivatives")]
            public double InternalForces { get; set; }

            [Category("Benchmarking")]
            public double Discarded { get; set; }

            [Category("Benchmarking")]
            [Description("Reductions, file saving, other")]
            public double Other { get; set; }
            #endregion

            public long TotalComputationTime { get; set; } = 0; // in ms

            public FrameSummary(List<FrameInfo> frames, string Name, double GrainSize)
            {
                this.Name = Name; this.GrainSize = GrainSize;
                if (frames == null || frames.Count == 0) return;

                int TotalIterations = 0;
                MaxStress = MaxForce = 0;
                MaxNNZ = int.MinValue;
                MinNNZ = int.MaxValue;

                long lMKLSolve, lCollisions, lInternalForces, lDiscarded, lOther;
                lMKLSolve = lCollisions= lInternalForces= lDiscarded= lOther = 0;

                foreach (FrameInfo f in frames)
                {
                    TotalIterations += f.IterationsPerformed;
                    TotalComputationTime += f.Total;

                    lMKLSolve += f.MKLSolve;
                    lCollisions += (f.CollForce + f.ElT_CPU + f.ElT_GPU + f.BVHConstructOrUpdate + f.BVHTraverse);
                    lInternalForces += (f.KForcePrepare + f.KerElemForce + f.KerCZForce);
                    lDiscarded += f.Discarded;

                    if (f.stress > MaxStress) MaxStress = f.stress;
                    if (f.IndenterForce > MaxForce) MaxForce = f.IndenterForce;
                    if (f.CSR_NNZ > MaxNNZ) MaxNNZ = f.CSR_NNZ;
                    if (f.CSR_NNZ < MinNNZ) MinNNZ = f.CSR_NNZ;
                }
                TotalComputationTime += lDiscarded;
                long subtotal = lMKLSolve + lCollisions + lInternalForces + lDiscarded;
                lOther = TotalComputationTime - subtotal;  // f.CSRStructure goes into "Other"

                // Essential
                TotalSteps = frames.Count;
                Duration = frames[frames.Count - 1].SimulationTime;
                TotCompute = new TimeSpan(10000 * TotalComputationTime);
                StepCompute = new TimeSpan(10000 * TotalComputationTime / TotalSteps);
                AvgIterations = (double)TotalIterations / TotalSteps;

                // Benchmarking

                MKLSolve = (double)lMKLSolve / TotalComputationTime;
                Collisions = (double) (lCollisions) / TotalComputationTime;
                InternalForces = (double)lInternalForces / TotalComputationTime;
                Discarded = (double)lDiscarded / TotalComputationTime;
                Other = (double)lOther / TotalComputationTime;

                // arrays
                int n = frames.Count;
                stress = new double[n];
                strain = new double[n];
                damageAndFail = new double[n];
                fail = new double[n];
                logOfTimeScaleFactor = new double[n];
                iterations = new double[n];
                timeStep = new double[n];
                indenterForce = new double[n];
                time = new double[n];
                double totalCZ = frames[0].nCZ_Initial;

                CZ_Softening = new double[n];
                CZ_Mixed = new double[n];
                CZ_UnloadingReloading = new double[n];
                CZ_Fail = new double[n];

                for (int i = 0; i < n; i++)
                {
                    FrameInfo fi = frames[i];

                    stress[i] = Math.Abs(fi.stress);
                    strain[i] = Math.Abs(fi.strain);
                    damageAndFail[i] = (fi.nCZDamaged + fi.nCZFailed) / totalCZ;
                    fail[i] = fi.nCZFailed / totalCZ;
                    logOfTimeScaleFactor[i] = Math.Log((double)fi.TimeScaleFactor, 2);
                    iterations[i] = fi.IterationsPerformed;
                    timeStep[i] = (double)i;
                    indenterForce[i] = fi.IndenterForce;
                    time[i] = fi.SimulationTime;

                    CZ_Softening[i] = (double)fi.nCZStatusSoftening;
                    CZ_Mixed[i] = (double)fi.nCZStatusMixed;
                    CZ_UnloadingReloading[i] = (double)fi.nCZStatusUnloadingReloading;
                    CZ_Fail[i] = fi.nCZFailed;
                }
            }
    
            string ToCSVString()
            {
                 return $"{Name},{TotalSteps},{Duration},{TotalComputationTime},{MaxStress},{MaxForce},{MKLSolve},{Collisions},{InternalForces},{Discarded},{Other},{GrainSize},{MinNNZ},{MaxNNZ}";
            }

            static string CSVHeader()
            {
                return "Name, TotalSteps, Duration, TotalComputationTime,MaxStress,MaxForce,MKLSolve,Collisions,InternalForces,Discarded,Other,GrainSize, MinNNZ, MaxNNZ";
            }

            public void WriteCSV(string path)
            {
                // write only one entry - this summary
                StreamWriter sw = new StreamWriter(File.Create(path));
                sw.WriteLine(CSVHeader());
                sw.WriteLine(ToCSVString());
                sw.Close();
            }

            public static void WriteCSV(FrameSummary[] summary, string path)
            {
                // write a list of entries from the array
                StreamWriter sw = new StreamWriter(File.Create(path));
                sw.WriteLine(CSVHeader());
                foreach (FrameSummary f in summary) sw.WriteLine(f.ToCSVString());
                sw.Close();
            }
        }

        #endregion

    }
}