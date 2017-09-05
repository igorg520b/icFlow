using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BatchRun
{
    class Simulation
    {
        public enum Status { None, Clean, Paused, Completed }
        public bool Enqueue { get; set; } = true;
        public int QNumber { get; set; }
        public string Name { get; set; }
        public int GrainCount { get; set; }
        public Status status { get; set; } 
        public int Steps { get; set; }
        public double SimTime { get; set; }
        public double MaxStress { get; set; }
        public double Strain { get; set; }
        public double FrDamage { get; set; }  // damage at time of fracture
        public double FrTime { get; set; }
        public double FrStep { get; set; }
        public TimeSpan CompTime { get; set; }
    }
}
