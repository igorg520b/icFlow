using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using icFlow;

namespace BatchRun
{
    class Program
    {
        static void Main(string[] args)
        {
            // results folder
            string resultsFolder = AppDomain.CurrentDomain.BaseDirectory + "_batchResults";
            if (!Directory.Exists(resultsFolder)) Directory.CreateDirectory(resultsFolder);

            // find simulations in _sims
            string simsFolder = AppDomain.CurrentDomain.BaseDirectory + "_sims\\";
            string[] sims = Directory.GetDirectories(simsFolder);

            var exitEvent = new ManualResetEvent(false);

            Console.CancelKeyPress += (sender, eventArgs) =>
            {
                eventArgs.Cancel = true;
                exitEvent.Set();
            };

            var server = new MyServer();     // example
            server.Run(sims, exitEvent);

            exitEvent.WaitOne();
            server.Stop();
            MyServer.t1.Wait();
            Console.WriteLine("Finished");
            Console.ReadKey();
        }
    }
    class MyServer
    {
        static bool requestToStop = false;
        static ManualResetEvent finished;
        static string[] simList;
        public static Task t1;


        public static void DoWork()
        {
            string simsFolder = AppDomain.CurrentDomain.BaseDirectory + "_sims\\";
            ImplicitModel3 model3 = new ImplicitModel3();
            model3.Initialize();

            foreach (string simName in simList)
            {
                Console.WriteLine($"Starting {simName}");
                model3.saveFolder = simName;
                model3.LoadSimulation(true);
                do
                {
                    model3.Step();
                    Console.WriteLine($"{Path.GetFileName(simName)}: {model3.cf.StepNumber}/{model3.prms.MaxSteps}");
                } while (model3.cf.StepNumber < model3.prms.MaxSteps && requestToStop == false);
                if (requestToStop) break;
                Console.WriteLine($"Finished {simName}");
            }
            Console.WriteLine("Computation finished");
            finished.Set();
        }

        public void Run(string[] simList, ManualResetEvent finished)
        {
            MyServer.finished = finished;
            MyServer.simList = simList;
            // run loop
            t1 = new Task(new Action(DoWork));
            t1.Start();
        }
        public void Stop()
        {
            Console.WriteLine("Stopping");
            requestToStop = true;
        }
    }
}


