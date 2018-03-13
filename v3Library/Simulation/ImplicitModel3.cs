using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.IO.Compression;
using System.Diagnostics;
using System.Xml.Serialization;

namespace icFlow
{
    public class ImplicitModel3
    {
        #region fields
        public MeshCollection mc = new MeshCollection();
        public ModelPrms prms = new ModelPrms();
        public FrameInfo cf, tcf0;
        public List<FrameInfo> allFrames = new List<FrameInfo>();

        public BVHT bvh = new BVHT(); // bounding voulume hierarchy tree
        public LinearSystem linearSystem = new LinearSystem();
        public bool isReady = false;

        public GPU_Functionality gf;
        Stopwatch sw = new Stopwatch();
        #endregion

        #region initialization

        public void Initialize()
        {
            gf = new GPU_Functionality();
        }
        #endregion

        #region preparation for computation

        public void PrepareToCompute()
        {
            // this is called once before simulation starts
            mc.Prepare();

            if (cf == null)
            {
                mc.Reset();
                // initialize current frame
                cf = new FrameInfo();
                cf.TimeScaleFactor = 16;
                allFrames.Clear();
                allFrames.Add(cf);
                cf.nCZ_Initial = mc.nonFailedCZs.Length;
                SaveSimulationInitialState();
                SaveStep();
                gf.prms = prms;
                gf.SetConstants();
            }

            // identify and mark anchored nodes
            foreach (Mesh deformableMesh in mc.deformables)
                foreach (SurfaceFragment sf in deformableMesh.surfaceFragments)
                    if (sf.role == SurfaceFragment.SurfaceRole.Anchored)
                        foreach (int idx in sf.faces)
                        {
                            Face f = deformableMesh.faces[idx];
                            foreach (Node nd in f.vrts) { nd.anchored = true; nd.altId = -1; }
                        }

            mc.activeNodes = Array.FindAll(mc.allNodes, nd => { return nd.anchored == false; });
            for (int i = 0; i < mc.activeNodes.Length; i++) mc.activeNodes[i].altId = i;

            if (allFrames[allFrames.Count - 1] != cf)
            {
                // trim frame list
                int from = cf.StepNumber + 1;
                int qty = allFrames.Count - cf.StepNumber - 1;
                allFrames.RemoveRange(cf.StepNumber + 1, allFrames.Count - cf.StepNumber - 1);
            }

            // find surface elements in all meshes, create leaf KDOP list
            bvh.b24.Clear();
            bvh.prms = prms;
            bvh.ForceReconstruct();
            foreach (Mesh mg in mc.mgs)
            {
                mg.IdentifySurfaceElements();
                mg.ConnectFaces();

                foreach (Element elem in mg.surfaceElements)
                {
                    kDOP24 k = new kDOP24();
                    k.elem = elem;
                    bvh.b24.Add(k);
                    elem.FindAdjFaces();
                }
            }
            mc.PrepareSurfaceElements();
            mc.UpdateStaticStructureData(prms.SymmetricStructure);

            gf.linearSystem = linearSystem;
            gf.prms = prms;
            gf.mc = mc;
            gf.ctx.SetCurrent();
            gf.cf = cf;
            gf.TransferStaticDataToDevice();

            isReady = true;
        }
        
        #endregion

        #region computation helpers

        void _beginStep()
        {
            // create tentative frame, position anchored surfaces
            // prms.SetComputedVariables();
            gf.cf = tcf0 = cf.SpecialCopy();

//            if(tcf0.TimeStep == 0) tcf0.IncrementTime(prms.InitialTimeStep,8);
//            else tcf0.IncrementTime(prms.InitialTimeStep);
            tcf0.IncrementTime(prms.InitialTimeStep);
            tcf0.ActiveNodes = mc.activeNodes.Length;

            // position nodes of anchored surfaces
            foreach (Mesh deformable in mc.deformables)
                foreach (SurfaceFragment sf in deformable.surfaceFragments)
                    if (sf.role == SurfaceFragment.SurfaceRole.Anchored)
                    {
                        // compute tentative location of this anchored surface
                        double attenuation = sf.applicationTime < tcf0.SimulationTime ? 1 : tcf0.SimulationTime / sf.applicationTime;
                        double dx = attenuation * sf.dx;
                        double dy = attenuation * sf.dy;
                        double dz = attenuation * sf.dz;
                        // apply to all nodes in this surface
                        foreach (int idx in sf.faces)
                        {
                            Face f = deformable.faces[idx];
                            foreach (Node nd in f.vrts)
                            {
                                Debug.Assert(nd.anchored && nd.altId == -1, "_beginStep assertion failed");
                                nd.dux = (nd.x0 + dx) - nd.cx;
                                nd.duy = (nd.y0 + dy) - nd.cy;
                                nd.duz = (nd.z0 + dz) - nd.cz;
                            }
                        }
                    }
            _positionNonDeformables(tcf0.SimulationTime);

            // initial displacement guess for active nodes
            double h = tcf0.TimeStep;
            double hsq2 = h * h / 2;
            Parallel.ForEach(mc.activeNodes, nd => {
                nd.dux = nd.vx * h;
                nd.duy = nd.vy * h;
                nd.duz = nd.vz * h;
            });
        }

        public void _positionNonDeformables(double time)
        {
            // position nonDeformables according to their translation list
            foreach (Mesh msh in mc.nonDeformables)
            {
                Translation tr = msh.translationCollection.GetTranslation(time);
                foreach(Node nd in msh.nodes)
                {
                    nd.unx = nd.ux = tr.dx;
                    nd.uny = nd.uy = tr.dy;
                    nd.unz = nd.uz = tr.dz;
                    nd.tx = nd.cx = nd.x0 + nd.ux;
                    nd.ty = nd.cy = nd.y0 + nd.uy;
                    nd.tz = nd.cz = nd.z0 + nd.uz;
                }
            }
        }

        void _addCollidingNodesToStructure()
        {
            Parallel.ForEach(mc.activeNodes, nd => { nd.allNeighbors.Clear(); });

            if (prms.CollisionScheme != ModelPrms.CollisionSchemes.None)
            {
                if (prms.SymmetricStructure)
                {
                    for (int i_im = 0; i_im < gf.nImpacts; i_im++)
                    {
                        for (int n1idx = 0; n1idx < 4; n1idx++)
                        {
                            Node n1 = mc.allNodes[gf.c_itet[i_im + n1idx * gf.tet_stride]];
                            for (int n2idx = 0; n2idx < 4; n2idx++)
                            {
                                Node n2 = mc.allNodes[gf.c_itet[i_im + n2idx * gf.tet_stride]];
                                if (!n2.anchored && (n2.altId > n1.altId)) n1.allNeighbors.Add(n2.altId);
                            }
                        }
                    }
                }
                else 
                {
                    // not symmetric
                    for (int i_im = 0; i_im < gf.nImpacts; i_im++)
                    {
                        for (int n1idx = 0; n1idx < 4; n1idx++)
                        {
                            Node n1 = mc.allNodes[gf.c_itet[i_im + n1idx * gf.tet_stride]];
                            for (int n2idx = 0; n2idx < 4; n2idx++)
                            {
                                Node n2 = mc.allNodes[gf.c_itet[i_im + n2idx * gf.tet_stride]];
                                if (!n2.anchored) n1.allNeighbors.Add(n2.altId);
                            }
                        }
                    }
                }
            }
            Parallel.ForEach(mc.activeNodes, nd => { nd.allNeighbors.UnionWith(nd.adjNeighbors); });
        }

        bool _checkDamage()
        {
            // return true if proportion of damaged or failed CZs at current step exceeds prms.maxDamagePerStep
            if (tcf0.TimeScaleFactor == FrameInfo.Parts) return false; // can't reduce time step anyway

            FrameInfo pf = allFrames[allFrames.Count - 1];
            double dn_damaged = (double)(tcf0.nCZDamaged - pf.nCZDamaged) / cf.nCZ_Initial;
            double dn_failed = (double)(tcf0.nCZFailedThisStep) / tcf0.nCZ_Initial;

            // return true if tentative damage exceeds threshold
            Trace.WriteLineIf(dn_damaged > prms.maxDamagePerStep, $"explodes due to damage: .nCZDamaged {tcf0.nCZDamaged}, pf.nCZDamaged {pf.nCZDamaged}, dn_damaged {dn_damaged}");
            Trace.WriteLineIf(dn_failed > prms.maxFailPerStep, $"explodes due to fail: .nCZFailedThisStep {tcf0.nCZFailedThisStep}, .nCZ_Initial {tcf0.nCZ_Initial}, dn_failed {dn_failed}");
            bool result = (dn_damaged > prms.maxDamagePerStep || dn_failed > prms.maxFailPerStep);
            return result;
        }

        public bool _checkDivergence()
        {
            bool divergence;
            double cutoff = prms.ConvergenceCutoff; // somewhat arbitrary constant
            double norm = linearSystem.NormOfDx();
            if (tcf0.IterationsPerformed == 0)
            {
                tcf0.Error0 = norm;
                if (tcf0.Error0 == 0) tcf0.Error0 = prms.ConvergenceEpsilon;
                tcf0.ConvergenceReached = false;
                divergence = false;
            }
            else if (norm < cutoff)
            {
                tcf0.ConvergenceReached = true;
                tcf0.RelativeError = -1;
                divergence = false;
            }
            else
            {
                Debug.Assert(tcf0.Error0 != 0, $"tcf0.Error0 = {tcf0.Error0}");
                tcf0.RelativeError = Math.Sqrt(norm / tcf0.Error0);
                if (tcf0.RelativeError <= prms.ConvergenceEpsilon) tcf0.ConvergenceReached = true;
                divergence = (tcf0.RelativeError > 1.01); // return true diverges
            }
            string s1 = $"st {tcf0.StepNumber}-{tcf0.IterationsPerformed}; sf {tcf0.TimeScaleFactor}; {(divergence ? "diverges; " : "")}{(norm < cutoff ? "cutoff-" : "")}{(tcf0.ConvergenceReached ? "convergence; " : "")}";
            string s2 = $"nrm {norm:0.###E-00}; rel {tcf0.RelativeError:0.###E-00}; E {tcf0.Error0:0.###E-00};";
            //            string sres = $"{s1:0,-40}{s2}";
            string sres = String.Format("{0,-40}", s1) + String.Format("{0,-16}", $"nrm {norm:0.###E-00};") + $"rel {tcf0.RelativeError:0.###E-00}; E {tcf0.Error0:0.###E-00};";
            Trace.WriteLine(sres);
            return divergence;
        }

        void _XtoDU()
        {
            // add linear system solution to displacements of active nodes
            double[] x = linearSystem.dx;
            Parallel.ForEach(mc.activeNodes, nd => {
                nd.dux += x[nd.altId * 3 + 0];
                nd.duy += x[nd.altId * 3 + 1];
                nd.duz += x[nd.altId * 3 + 2];
            });
        }

        const int holdFactorDelay = 4;
        void _adjustTimeStep()
        {
            cf.AttemptsTaken++;
            if (explodes)
            {
                cf.TimeScaleFactor *= 2;
                if (cf.TimeScaleFactor > FrameInfo.Parts) cf.TimeScaleFactor = FrameInfo.Parts;
                Debug.WriteLine($"explodes, new sf: {cf.TimeScaleFactor}");
                cf.StepsWithCurrentFactor = holdFactorDelay;
            }
            else if ((diverges || !tcf0.ConvergenceReached) && tcf0.TimeScaleFactor < FrameInfo.Factor2 && prms.maxIterations > 1)
            {
                // does not converge
                cf.TimeScaleFactor *= 2;
                cf.StepsWithCurrentFactor = holdFactorDelay;
                Debug.WriteLine($"double ts: diverges {diverges}; convergenceReached {tcf0.ConvergenceReached}; factor {cf.TimeScaleFactor}");
            }
            else if (tcf0.ConvergenceReached && cf.TimeScaleFactor > 1)
            {
                // converges
                if (tcf0.StepsWithCurrentFactor > 0) tcf0.StepsWithCurrentFactor--;
                else
                {
                    tcf0.TimeScaleFactor /= 2;
                    if (tcf0.TimeScaleFactor < 1) tcf0.TimeScaleFactor = 1;
                    tcf0.StepsWithCurrentFactor = 2;
                }
            }
        }

        void _acceptFrame()
        {
            // remove CZs that failed
            int nFailedCZ = mc.nonFailedCZs.Count(x => x.failed);
            if (nFailedCZ > 0)
            {
                tcf0.nCZFailed += nFailedCZ;
                CZ[] fczs = Array.FindAll(mc.nonFailedCZs, cz => cz.failed);
                // mark corresponding faces are "exposed"
                foreach (CZ cz in fczs) foreach (Face fc in cz.faces) fc.created = fc.exposed = true;
                mc.UpdateStaticStructureData(prms.SymmetricStructure);    // active CZ count has changed
                bvh.ForceReconstruct(); // this is not necessary if between-grain collisions are enabled
            }
            //                Trace.WriteLine($"Created surfaces: {mc.allFaces.Sum(face => face.created ? 1 :0)}");

            Trace.WriteLine("");
            cf = tcf0;
            cf.StepNumber++;
            if (cf.StepNumber == 1) cf.TimeScaleFactor = 1;

            cf.BVHT_depth = BVHN.maxLevel;
            if (prms.CollisionScheme != ModelPrms.CollisionSchemes.None) bvh.treeConstructedStepsAgo++;
            allFrames.Add(cf);

            // not sure if this is the right place for this fragment of code
            // record stress-strain data
            // pressure on anchored surfaces is computed from elastic forces on the nodes
            if(mc.deformables.Count > 0)
            {
                Mesh msh = mc.deformables[0]; // assuming there is just one cylinder
                SurfaceFragment top = msh.surfaceFragments[0];
                SurfaceFragment bottom = msh.surfaceFragments[1];
                if(top != null && bottom != null && 
                    top.role == SurfaceFragment.SurfaceRole.Anchored && 
                    bottom.role == SurfaceFragment.SurfaceRole.Anchored)
                {
                    top.ComputeTotalForce();
                    bottom.ComputeTotalForce();
                    top.ComputeArea();
                    bottom.ComputeArea();
                    double tz = top.AverageVerticalDisplacement();
                    double bz = bottom.AverageVerticalDisplacement();
                    double height = msh.height; // undeformed
                    cf.strain = (tz - bz) / height;
                    cf.stress = top.fz / top.area;
                }
            }

            double indFrcX, indFrcY, indFrcZ;
            indFrcX = indFrcY = indFrcZ = 0;
            foreach (Mesh msh in mc.indenters)
                foreach(Node nd in msh.nodes)
                {
                    indFrcX += nd.fx;
                    indFrcY += nd.fy;
                    indFrcZ += nd.fz;
                }
            cf.IndenterForce = Math.Sqrt(indFrcX * indFrcX + indFrcY * indFrcY + indFrcZ * indFrcZ);
        }

        #endregion

        #region one simulation step

        bool explodes, diverges;
        public void Step()
        {
            sw.Restart();
            if (cf == null || !isReady || cf.StepNumber == 0) PrepareToCompute(); 
            
            _beginStep(); // timestep, indenter position and parameters of GPU

            // Perform Newton Raphson iterations
            do
            {
                // infer tentative UVA from dU
                Parallel.ForEach(mc.allNodes, nd => nd.InferTentativeValues(tcf0.TimeStep, prms.NewmarkBeta, prms.NewmarkGamma)); 
                
                // detect collisions; add colliding nodes to Node.collisionNodes
                gf.TransferNodesToGPU();
                bvh.ConstructAndTraverse(tcf0); // CHANGE THIS TO ACCOUNT FOR DIFFERENT COLLISION SCHEMES
                gf.NarrowPhaseCollisionDetection(bvh.broad_list);
                _addCollidingNodesToStructure(); // account for impacts in matrix structure

                // create CSR, accounting for collision Nodes
                linearSystem.CreateStructure(mc.activeNodes, tcf0, prms);

                // transfer all values to GPU, including tentative UVA
                gf.TransferPCSR();
                gf.AssembleElemsAndCZs();
                explodes = _checkDamage(); // discard frame if threshold is exceeded

                // add collision response to the matrix
                gf.collisionResponse();
                gf.TransferLinearSystemToHost();

                // solve with MKL
//                linearSystem.Assert(prms);
                linearSystem.Solve(tcf0, prms);

                // convergence analysis
                diverges = _checkDivergence();

                // distribute the solution from ga.c_dx to mg
                _XtoDU();

                tcf0.IterationsPerformed++;
            }
            while (tcf0.IterationsPerformed < prms.minIterations || 
            (!explodes && !diverges && !tcf0.ConvergenceReached && tcf0.IterationsPerformed < prms.maxIterations));

            cf.TimeScaleFactorThisStep = tcf0.TimeScaleFactor; // record what TSF was used for this step
            _adjustTimeStep();
            // compute stats and save frame
            if (prms.maxIterations == 1 ||
                tcf0.TimeScaleFactor == FrameInfo.Parts ||
                (!tcf0.ConvergenceReached && tcf0.TimeScaleFactor >= FrameInfo.Factor2) ||
                (!explodes && !diverges && tcf0.ConvergenceReached))
            {
                // tentative values -> current values
                Parallel.ForEach(mc.allNodes, nd => nd.AcceptTentativeValues(tcf0.TimeStep));

                // tentative CZ states (from GPU) -> current CZ states
                gf.TransferUpdatedStateToHost();

                tcf0.Total = sw.ElapsedMilliseconds;
                _acceptFrame();
                SaveStep();
            }
            else cf.Discarded += sw.ElapsedMilliseconds; // keep discarded time
            sw.Stop();
        }

        #endregion

        #region save / load
        public string saveFolder; // this has to be initialized externally
        public const string meshSubfolder = "meshes\\";

        #region save/resume simulation step
        public void SaveStep()
        {
            if (prms.name == "") return;
            if (!Directory.Exists(saveFolder)) Directory.CreateDirectory(saveFolder);
            if (!Directory.Exists($"{saveFolder}\\{meshSubfolder}")) Directory.CreateDirectory($"{saveFolder}\\{meshSubfolder}");

            SaveParams();
            SaveFrameData();

            // save snapshot
            Stream str = File.OpenWrite($"{saveFolder}\\{meshSubfolder}\\{cf.StepNumber}.mgs");
            mc.Save(str);
            mc.WriteImpacts(str);
            str.Close();
        }

        public void GoToFrame(int frame)
        {
            isReady = false;
            Stream str = File.OpenRead($"{saveFolder}\\{meshSubfolder}\\{frame}.mgs");
            mc.Update(str);
            mc.ReadImpacts(str);
            str.Close();
            cf = allFrames[frame];
        }

        #endregion

        #region params

        // save params in xml format
        public void SaveParams()
        {
            StreamWriter sw = new StreamWriter($"{saveFolder}\\params");
            XmlSerializer xs = new XmlSerializer(typeof(ModelPrms));
            xs.Serialize(sw, prms);
            sw.Close();
        }

        // open params from a given stream
        public void LoadParams()
        {
            Stream str = File.OpenRead($"{saveFolder}\\params");
            XmlSerializer xs = new XmlSerializer(typeof(ModelPrms));
            prms = (ModelPrms)xs.Deserialize(str);
            str.Close();
        }
        #endregion

        #region frame data

        public void SaveFrameData()
        {
            // save list of frames
            StreamWriter str = new StreamWriter($"{saveFolder}\\frames");
            XmlSerializer xs = new XmlSerializer(typeof(List<FrameInfo>));
            xs.Serialize(str, allFrames);
            str.Close();
        }

        void LoadFrameData()
        {
            StreamReader str = new StreamReader($"{saveFolder}\\frames");
            XmlSerializer xs = new XmlSerializer(typeof(List<FrameInfo>));
            allFrames = (List<FrameInfo>)xs.Deserialize(str);
            str.Close();
        }
        #endregion

        #region save and load initial setup

        // save undeformed geometry and parameters
        public void SaveSimulationInitialState()
        {
            if(saveFolder == null) saveFolder = AppDomain.CurrentDomain.BaseDirectory + "_sims\\" + prms.name + "\\";
            if (!Directory.Exists(saveFolder)) Directory.CreateDirectory(saveFolder);
            SaveParams();
            Stream str;
            str = File.Create($"{saveFolder}\\geometry");
            GZipStream str2 = new GZipStream(str, CompressionLevel.Optimal);
            mc.Save(str2);
            str2.Close();
        }

        public void LoadSimulation(bool clear = false)
        {
            // saveFolder must be initialized
            isReady = false;
            LoadParams();
            allFrames.Clear();
            cf = null;
            string filePath = $"{saveFolder}\\geometry";
            Stream str = File.OpenRead(filePath);
            GZipStream str2 = new GZipStream(str, CompressionMode.Decompress);
            mc.Load(str2);
            str2.Close();

            foreach (Mesh msh in mc.mgs)
            {
                foreach (Node nd in msh.nodes)
                {
                    nd.ux = nd.uy = nd.uz = nd.vx = nd.vy = nd.vz = 0;
                    nd.ax = nd.ay = nd.az = 0;
                    nd.cx = nd.x0;
                    nd.cy = nd.y0;
                    nd.cz = nd.z0;
                }
                foreach(CZ cz in msh.czs)
                {
                    cz.pmax[0] = cz.pmax[1] = cz.pmax[2] = 0;
                    cz.tmax[0] = cz.tmax[1] = cz.tmax[2] = 0;
                    cz.failed = false;
                }
            }

            string frames_data_file = $"{saveFolder}\\frames";
            if (!clear && File.Exists(frames_data_file))
            {
                LoadFrameData();
                GoToFrame(allFrames.Count - 1);
            }

            if (gf != null)
            {
                // gf == null if this is called from renderer
                gf.prms = prms;
                gf.SetConstants();
            }
        }

        #endregion

        #endregion
    }
}
