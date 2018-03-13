using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System.IO;

namespace icFlow
{
    public class GPU_Functionality
    {
        #region fields
        public CudaContext ctx;

        public MeshCollection mc;
        public LinearSystem linearSystem;
        public ModelPrms prms;
        public FrameInfo cf;
        Stopwatch sw = new Stopwatch();

        CudaKernel kelElementElasticityForce; // elem_kernels.ptx
        CudaKernel kczCZForce;                // cz_kernels.ptx
        CudaKernel kNarrowPhase, kFindClosestFace, kCollisionResponseForce; 

        // model data (nodes, elems, czs)
        CudaDeviceVariable<double> g_dn, g_dcz;
        CudaDeviceVariable<int> g_ie, g_ie_pcsr, g_icz, g_ifc;
        double[] c_dn, c_dcz;
        int[] c_ie, c_icz, c_ie_pcsr, c_ifc;

        // bcsr 
        CudaDeviceVariable<double> g_dvals, g_drhs;

        // reductions
        CudaDeviceVariable<int> nres_total;
        CudaKernel n_kerSum;
        const int block_size = 128;

        // integers
        int nd_stride, el_all_stride, el_elastic_stride, cz_stride, fc_stride;

        // narrow phase data
        public int nImpacts; // size of array may be larger
        int maxAdjacentFaces;
        #endregion

        #region offsets
        // element counts for g_dn, g_ie, g_dcz, g_icz
        const int FP_DATA_SIZE_NODE = 18;
        const int INT_DATA_SIZE_ELEM = 28;
        const int FP_DATA_SIZE_CZ = 12;
        const int INT_DATA_SIZE_CZ = 58;

        // g_dn
        const int X0_OFFSET = 0;
        const int UN_OFFSET = 3;
        const int VN_OFFSET = 6;
        const int AN_OFFSET = 9;
        const int F_OFFSET = 12;
        const int X_CURRENT_OFFSET = 15;

        // g_ie
        const int N0_OFFSET_ELEM = 0;
        const int PCSR_OFFSET_ELEM = 4;
        const int ROWSIZE_OFFSET_ELEM = 24;

        // g_icz
        const int CURRENT_FAILED_OFFSET_CZ = 0;
        const int TENTATIVE_CONTACT_OFFSET_CZ = 1;
        const int TENTATIVE_DAMAGED_OFFSET_CZ = 2;
        const int TENTATIVE_FAILED_OFFSET_CZ = 3;
        const int VRTS_OFFSET_CZ = 4;
        const int PCSR_OFFSET_CZ = 10;
        const int ROWSIZE_OFFSET_CZ = 52;

        // g_dcz
        const int CURRENT_PMAX_OFFSET_CZ = 0;
        const int CURRENT_TMAX_OFFSET_CZ = 3;
        const int TENTATIVE_PMAX_OFFSET_CZ = 6;
        const int TENTATIVE_TMAX_OFFSET_CZ = 9;
        #endregion

        #region reductions

        static int grid(int val, int block_size) { return ((val + block_size - 1) / block_size); }

        int Sum(CUdeviceptr array_ptr, int size)
        {
            if (size == 0) return 0;
            nres_total.CopyToDevice(0);

            int block_size = 1024;
            int max_grid_size = 1024;
            int grid_size = grid(size, block_size);
            if (grid_size > max_grid_size) grid_size = max_grid_size;
            n_kerSum.BlockDimensions = new dim3(block_size, 1, 1);
            n_kerSum.GridDimensions = new dim3(grid_size, 1, 1);

            n_kerSum.Run(array_ptr, size);
            int nresult = 0;
            nres_total.CopyToHost(ref nresult);
            return nresult;
        }

        #endregion

        #region initialization
        public GPU_Functionality(int deviceID = 0)
        {
            ctx = new CudaContext(deviceID);
            CUmodule collision_module = ctx.LoadModulePTX("collision_kernels.ptx");

            kNarrowPhase = new CudaKernel("kNarrowPhase_new", collision_module, ctx);
            kFindClosestFace = new CudaKernel("kFindClosestFace", collision_module, ctx);
            kCollisionResponseForce = new CudaKernel("kCollisionResponseForce", collision_module, ctx);
            dim3 block = new dim3(block_size, 1, 1);
            kNarrowPhase.BlockDimensions = block;
            kFindClosestFace.BlockDimensions = block;
            kCollisionResponseForce.BlockDimensions = block;

            // cz
            CUmodule module_cz_kernels = ctx.LoadModulePTX("cz_kernels.ptx");
            kczCZForce = new CudaKernel("kczCZForce", module_cz_kernels, ctx);
            kczCZForce.BlockDimensions = block;

            // elem
            CUmodule module_elem_kernels = ctx.LoadModulePTX("elem_kernels.ptx");
            kelElementElasticityForce = new CudaKernel("kelElementElasticityForce", module_elem_kernels, ctx);
            kelElementElasticityForce.BlockDimensions = block;

            // reduction kernels
            CUmodule reduction_kernels = ctx.LoadModulePTX("reduction_kernels.ptx");
            n_kerSum = new CudaKernel("n_kerSum", reduction_kernels, ctx);
            nres_total = new CudaDeviceVariable<int>(reduction_kernels, "nresult");
        }

        int grid(int val) { return ((val + block_size - 1) / block_size); }

        void ComputeStrides()
        {
            // strides may change throughout simulation
            nd_stride = grid(mc.allNodes.Length) * block_size;
            el_all_stride = grid(mc.surfaceElements.Length) * block_size;
            el_elastic_stride = grid(mc.elasticElements.Length) * block_size;
            fc_stride = grid(mc.allFaces.Length) * block_size;
            cz_stride = grid(mc.nonFailedCZs.Length) * block_size;
        }

        public void SetConstants()
        {
            ctx.SetCurrent();
            prms.SetComputedVariables();
            CudaKernel ker_elasticity = kelElementElasticityForce;
            //ker_elasticity.
            ker_elasticity.SetConstantVariable("E", prms.E);
            ker_elasticity.SetConstantVariable("M", prms.M);
            ker_elasticity.SetConstantVariable("NewmarkBeta", prms.NewmarkBeta);
            ker_elasticity.SetConstantVariable("NewmarkGamma", prms.NewmarkGamma);
            ker_elasticity.SetConstantVariable("dampingMass", prms.dampingMass);
            ker_elasticity.SetConstantVariable("dampingStiffness", prms.dampingStiffness);
            ker_elasticity.SetConstantVariable("rho", prms.rho);
            ker_elasticity.SetConstantVariable("gravity", prms.gravity);
            ker_elasticity.SetConstantVariable("YoungsModulus", prms.Y);

            CudaKernel ker = kczCZForce;
            // parameters
            ker.SetConstantVariable("NewmarkBeta", prms.NewmarkBeta);
            ker.SetConstantVariable("NewmarkGamma", prms.NewmarkGamma);
            ker.SetConstantVariable("dampingMass", prms.dampingMass);
            ker.SetConstantVariable("dampingStiffness", prms.dampingStiffness);
            ker.SetConstantVariable("rho", prms.rho);
            ker.SetConstantVariable("M", prms.M);
            ker.SetConstantVariable("G_fn", prms.G_fn);
            ker.SetConstantVariable("G_ft", prms.G_ft);
            ker.SetConstantVariable("f_tn", prms.f_tn);
            ker.SetConstantVariable("f_tt", prms.f_tt);
            ker.SetConstantVariable("alpha", prms.alpha);
            ker.SetConstantVariable("beta", prms.beta);
            ker.SetConstantVariable("rn", prms.rn);
            ker.SetConstantVariable("rt", prms.rt);
            ker.SetConstantVariable("p_m", prms.p_m);
            ker.SetConstantVariable("p_n", prms.p_n);
            ker.SetConstantVariable("deln", prms.deln);
            ker.SetConstantVariable("delt", prms.delt);
            ker.SetConstantVariable("pMtn", prms.pMtn);
            ker.SetConstantVariable("pMnt", prms.pMnt);
            ker.SetConstantVariable("gam_n", prms.gam_n);
            ker.SetConstantVariable("gam_t", prms.gam_t);
            ker.SetConstantVariable("B", prms._B);
            ker.SetConstantVariable("sf", prms._sf);
            ker.SetConstantVariable("gravity", prms.gravity);
        }

        void AllocateMemoryForMesh()
        {
            // max number of adjacent faces per element
            maxAdjacentFaces = mc.surfaceElements.Max(elem => elem.adjFaces.Count);
            // Trace.WriteLine($"maxAdjacentFaces {maxAdjacentFaces}");

            // size of GPU arrays
            int size_ie = el_all_stride * (4 + 1 + maxAdjacentFaces);
            int size_dn = nd_stride * FP_DATA_SIZE_NODE;
            int size_ifc = fc_stride * 3;
            int size_ie_pcsr = el_elastic_stride * INT_DATA_SIZE_ELEM;
            int size_dcz = cz_stride * FP_DATA_SIZE_CZ;
            int size_icz = cz_stride * INT_DATA_SIZE_CZ;

            // verify that sufficient memory space is allocated
            if (g_dn == null || g_dn.Size < size_dn)
            {
                if (g_dn != null) g_dn.Dispose();
                g_dn = new CudaDeviceVariable<double>(size_dn);
                c_dn = new double[size_dn];
            }

            if (g_ie == null || g_ie.Size < size_ie)
            {
                if (g_ie != null) g_ie.Dispose();
                g_ie = new CudaDeviceVariable<int>(size_ie);
                c_ie = new int[size_ie];
            }

            if (g_ie_pcsr == null || g_ie_pcsr.Size < size_ie_pcsr)
            {
                if (g_ie_pcsr != null) g_ie_pcsr.Dispose();
                g_ie_pcsr = new CudaDeviceVariable<int>(size_ie_pcsr);
                c_ie_pcsr = new int[size_ie_pcsr];
            }

            if (cz_stride != 0 && (g_dcz == null || g_dcz.Size < size_dcz))
            {
                if (g_dcz != null) g_dcz.Dispose();
                if (g_icz != null) g_icz.Dispose();
                g_dcz = new CudaDeviceVariable<double>(size_dcz);
                c_dcz = new double[size_dcz];
                g_icz = new CudaDeviceVariable<int>(size_icz);
                c_icz = new int[size_icz];

            }

            if (g_ifc == null || g_ifc.Size < size_ifc)
            {
                if (g_ifc != null) g_ifc.Dispose();
                g_ifc = new CudaDeviceVariable<int>(size_ifc);
                c_ifc = new int[size_ifc];
            }
        }

        #endregion

        #region initialize static data 

        public void TransferStaticDataToDevice()
        {
            // All elements and faces are transferred
            // Memory is also allocated for elastic elements, nodes and CZs
            // Performed if (1) initialized or (2) mesh topology changes
            ComputeStrides();
            AllocateMemoryForMesh();

            // transfer all elements + their adjacent faces
            Parallel.For(0, mc.surfaceElements.Length, i_elem =>
            {
                Element elem = mc.surfaceElements[i_elem];
                c_ie[i_elem + el_all_stride * 0] = elem.vrts[0].globalNodeId;
                c_ie[i_elem + el_all_stride * 1] = elem.vrts[1].globalNodeId;
                c_ie[i_elem + el_all_stride * 2] = elem.vrts[2].globalNodeId;
                c_ie[i_elem + el_all_stride * 3] = elem.vrts[3].globalNodeId;

                c_ie[i_elem + el_all_stride * 4] = elem.adjFaces.Count;
                int count = 0;
                foreach (Face fc in elem.adjFaces)
                {
                    c_ie[i_elem + el_all_stride * (5 + count)] = fc.globalFaceId;
                    count++;
                }
            });
            g_ie.CopyToDevice(c_ie);

            Trace.Assert(fc_stride == grid(mc.allFaces.Length) * block_size, "fc_stride is incorrect");
            Parallel.For(0, mc.allFaces.Length, i_fc => {
                Face fc = mc.allFaces[i_fc];
                Trace.Assert(fc.globalFaceId == i_fc, "globalFaceId error");
                c_ifc[i_fc + fc_stride * 0] = fc.vrts[0].globalNodeId;
                c_ifc[i_fc + fc_stride * 1] = fc.vrts[1].globalNodeId;
                c_ifc[i_fc + fc_stride * 2] = fc.vrts[2].globalNodeId;
            });

            // verify
            for(int i= 0;i< mc.allFaces.Length;i++)
            {
                Face fc = mc.allFaces[i];
                int i_fc = fc.globalFaceId;
                Trace.Assert(i == i_fc, "index assertion");
                Trace.Assert(c_ifc[i_fc + fc_stride * 0] == fc.vrts[0].globalNodeId, "face assertion 0");
                Trace.Assert(c_ifc[i_fc + fc_stride * 1] == fc.vrts[1].globalNodeId, "face assertion 1");
                Trace.Assert(c_ifc[i_fc + fc_stride * 2] == fc.vrts[2].globalNodeId, "face assertion 2");
            }
            g_ifc.CopyToDevice(c_ifc);
        }

        #endregion

        #region element force computation
        public void TransferNodesToGPU()
        {
            ctx.SetCurrent();
            sw.Restart();

            Parallel.For(0, mc.allNodes.Length, i =>
            {
                Node nd = mc.allNodes[i];
                Trace.Assert(i == nd.globalNodeId, "i == nd.globalNodeId");
                c_dn[i + nd_stride * (F_OFFSET + 0)] = 0;
                c_dn[i + nd_stride * (F_OFFSET + 1)] = 0;
                c_dn[i + nd_stride * (F_OFFSET + 2)] = 0;

                c_dn[i + nd_stride * (UN_OFFSET + 0)] = nd.unx;
                c_dn[i + nd_stride * (UN_OFFSET + 1)] = nd.uny;
                c_dn[i + nd_stride * (UN_OFFSET + 2)] = nd.unz;

                c_dn[i + nd_stride * (VN_OFFSET + 0)] = nd.vnx;
                c_dn[i + nd_stride * (VN_OFFSET + 1)] = nd.vny;
                c_dn[i + nd_stride * (VN_OFFSET + 2)] = nd.vnz;

                c_dn[i + nd_stride * (AN_OFFSET + 0)] = nd.anx;
                c_dn[i + nd_stride * (AN_OFFSET + 1)] = nd.any;
                c_dn[i + nd_stride * (AN_OFFSET + 2)] = nd.anz;

                c_dn[i + nd_stride * (X0_OFFSET + 0)] = nd.x0;
                c_dn[i + nd_stride * (X0_OFFSET + 1)] = nd.y0;
                c_dn[i + nd_stride * (X0_OFFSET + 2)] = nd.z0;

                c_dn[i + nd_stride * (X_CURRENT_OFFSET + 0)] = nd.tx;
                c_dn[i + nd_stride * (X_CURRENT_OFFSET + 1)] = nd.ty;
                c_dn[i + nd_stride * (X_CURRENT_OFFSET + 2)] = nd.tz;
            });
            g_dn.CopyToDevice(c_dn, 0,0,sizeof(double)*nd_stride * FP_DATA_SIZE_NODE);
            sw.Stop();
            cf.KForcePrepare += sw.ElapsedMilliseconds;
        }

        public void TransferPCSR()
        {
            // tranfer elastic elems and CZs along with offsets in sparse matrix (where to write the result)
            sw.Restart();
            bool nonSymmetric = !prms.SymmetricStructure;
            // elastic elements
            Parallel.For(0, mc.elasticElements.Length, i_elem =>
            {
                Element elem = mc.elasticElements[i_elem];

                for (int i = 0; i < 4; i++)
                {
                    Node ni = elem.vrts[i];
                    for (int j = 0; j < 4; j++)
                    {
                        int pcsr_ij;
                        Node nj = elem.vrts[j];
                        if (!ni.anchored && !nj.anchored && (nonSymmetric || nj.altId >= ni.altId))
                            pcsr_ij = ni.pcsr[nj.altId];
                        else pcsr_ij = -1; // ni is anchored => ignore
                        c_ie_pcsr[i_elem + el_elastic_stride * (PCSR_OFFSET_ELEM + (i * 4 + j))] = pcsr_ij;
                    }
                    c_ie_pcsr[i_elem + el_elastic_stride * (PCSR_OFFSET_ELEM + 16 + i)] = ni.anchored ? -1 : ni.altId;
                    c_ie_pcsr[i_elem + el_elastic_stride * (N0_OFFSET_ELEM + i)] = ni.globalNodeId;
                    c_ie_pcsr[i_elem + el_elastic_stride * (ROWSIZE_OFFSET_ELEM + i)] = ni.allNeighbors.Count;
                }
            });
            g_ie_pcsr.CopyToDevice(c_ie_pcsr, 0, 0, el_elastic_stride * INT_DATA_SIZE_ELEM * sizeof(int));

            // initialize indices in cohesive zones
            cz_stride = grid(mc.nonFailedCZs.Length) * block_size;

            if (cz_stride != 0) {
                Parallel.For(0, mc.nonFailedCZs.Length, i_cz =>
                {
                    CZ cz = mc.nonFailedCZs[i_cz];
                    Trace.Assert(!cz.failed, "failed CZ in GPU array");

                    c_icz[i_cz + cz_stride * (CURRENT_FAILED_OFFSET_CZ)] = 0;
                    c_icz[i_cz + cz_stride * (TENTATIVE_FAILED_OFFSET_CZ)] = 0;
                    c_icz[i_cz + cz_stride * (TENTATIVE_DAMAGED_OFFSET_CZ)] = 0;

                    for (int i = 0; i < 6; i++)
                    {
                        Node ni = cz.vrts[i];
                        for (int j = 0; j < 6; j++)
                        {
                            Node nj = cz.vrts[j];
                            int pcsr_ij;
                            if (!ni.anchored && !nj.anchored && (nonSymmetric || nj.altId >= ni.altId))
                                pcsr_ij = ni.pcsr[nj.altId];
                            else pcsr_ij = -2; // ni is anchored => ignore
                            c_icz[i_cz + cz_stride * (PCSR_OFFSET_CZ + (i * 6 + j))] = pcsr_ij;
                        }
                        c_icz[i_cz + cz_stride * (PCSR_OFFSET_CZ + 36 + i)] = ni.anchored ? -1 : ni.altId;
                        c_icz[i_cz + cz_stride * (VRTS_OFFSET_CZ + i)] = ni.globalNodeId;
                        c_icz[i_cz + cz_stride * (ROWSIZE_OFFSET_CZ + i)] = ni.allNeighbors.Count;
                    }

                    for (int j = 0; j < 3; j++)
                    {
                        c_dcz[i_cz + cz_stride * (CURRENT_PMAX_OFFSET_CZ + j)] = cz.pmax[j];
                        c_dcz[i_cz + cz_stride * (CURRENT_TMAX_OFFSET_CZ + j)] = cz.tmax[j];
                    }
                });
                g_icz.CopyToDevice(c_icz, 0, 0, cz_stride * sizeof(int) * INT_DATA_SIZE_CZ);
                g_dcz.CopyToDevice(c_dcz, 0, 0, cz_stride * sizeof(double) * FP_DATA_SIZE_CZ);
            }
            sw.Stop();
            cf.KForcePrepare += sw.ElapsedMilliseconds;
        }

        // this is called right before assembly
        void AllocateMemoryForLinearSystem()
        {
            sw.Restart();

            if (g_dvals == null || g_dvals.Size < linearSystem.dvalsSize)
            {
                if (g_dvals != null) g_dvals.Dispose();
                g_dvals = new CudaDeviceVariable<double>((int)(linearSystem.dvalsSize * LinearSystem.overAllocate));
            }
            if (g_drhs == null || g_drhs.Size < linearSystem.dxSize)
            {
                if (g_drhs != null) g_drhs.Dispose();
                g_drhs = new CudaDeviceVariable<double>(linearSystem.dxSize);
            }
            g_dvals.MemsetAsync(0, CUstream.NullStream);
            g_drhs.MemsetAsync(0, CUstream.NullStream);
            sw.Stop();
            cf.KForcePrepare += sw.ElapsedMilliseconds;
        }

        public void AssembleElemsAndCZs()
        {
            AllocateMemoryForLinearSystem();

            sw.Restart();
            cf.nElems = mc.elasticElements.Length;
            cf.nCZ = mc.nonFailedCZs.Length;

            // set kernel configurations
            kelElementElasticityForce.GridDimensions = new dim3(grid(mc.elasticElements.Length), 1, 1);

            // run kernels
            kelElementElasticityForce.RunAsync(CUstream.NullStream, 
                g_ie_pcsr.DevicePointer, g_dn.DevicePointer, cf.TimeStep, 
                g_dvals.DevicePointer, g_drhs.DevicePointer,
                mc.elasticElements.Length, el_elastic_stride, nd_stride, prms.AssemblyType);

            ctx.Synchronize(); // this is for benchmarkng only - does not affect functionality
            sw.Stop();
            cf.KerElemForce += sw.ElapsedMilliseconds;

            if (cz_stride != 0)
            {
                sw.Restart();
                kczCZForce.GridDimensions = new dim3(grid(mc.nonFailedCZs.Length), 1, 1);
                kczCZForce.RunAsync(CUstream.NullStream, g_dcz.DevicePointer, g_icz.DevicePointer, 
                    g_dn.DevicePointer,
                    g_dvals.DevicePointer, g_drhs.DevicePointer, 
                    cf.TimeStep,
                    mc.nonFailedCZs.Length, cz_stride, nd_stride, (int)prms.czFormulaiton, prms.AssemblyType);

                // pointers to specific portions of the arrays
                CUdeviceptr ptr_failed = g_icz.DevicePointer + sizeof(int) * cz_stride * TENTATIVE_FAILED_OFFSET_CZ;
                CUdeviceptr ptr_damaged = g_icz.DevicePointer + sizeof(int) * cz_stride * TENTATIVE_DAMAGED_OFFSET_CZ;

                cf.nCZDamaged = Sum(ptr_damaged, mc.nonFailedCZs.Length);
                cf.nCZFailedThisStep = Sum(ptr_failed, mc.nonFailedCZs.Length);
                sw.Stop();
                cf.KerCZForce += sw.ElapsedMilliseconds;
            }
        }

        public void TransferLinearSystemToHost()
        {
            g_dvals.CopyToHost(linearSystem.vals, 0, 0, linearSystem.dvalsSize * sizeof(double));
            g_drhs.CopyToHost(linearSystem.rhs, 0, 0, linearSystem.dxSize * sizeof(double));
        }

        public void TransferUpdatedStateToHost()
        {
            if (cz_stride != 0)
            {
                // transfer to host dcz and icz
                g_dcz.CopyToHost(c_dcz, 0, 0, sizeof(double) * cz_stride * 12); // pmax, tmax
                g_icz.CopyToHost(c_icz, 0, 0, sizeof(int) * cz_stride * 4); // cz_failed

                // infer failed state, pmax[] and tmax[]
                Parallel.For(0, mc.nonFailedCZs.Length, i => 
                {
                    CZ cz = mc.nonFailedCZs[i];
                    if (!cz.failed)
                    {
                        bool tentative_fail = c_icz[i + cz_stride * (TENTATIVE_FAILED_OFFSET_CZ)] == 0 ? false : true;
                        if (tentative_fail) cz.failed = true;
                        for (int j = 0; j < 3; j++)
                        {
                            cz.pmax[j] = c_dcz[i + cz_stride * (TENTATIVE_PMAX_OFFSET_CZ + j)];
                            cz.tmax[j] = c_dcz[i + cz_stride * (TENTATIVE_TMAX_OFFSET_CZ + j)];
                        }
                    }
                });
            }

            // copy back elastic forces (should be zero on rigid objects)
            g_dn.CopyToHost(c_dn, sizeof(double)*nd_stride*F_OFFSET, sizeof(double) * nd_stride * F_OFFSET, sizeof(double) * nd_stride * 3); 
            Parallel.For(0, mc.allNodes.Length, i =>
            {
                Node nd = mc.allNodes[i];
                nd.fx = c_dn[i + nd_stride * (F_OFFSET + 0)];
                nd.fy = c_dn[i + nd_stride * (F_OFFSET + 1)];
                nd.fz = c_dn[i + nd_stride * (F_OFFSET + 2)];
            });
        }

        #endregion

        #region NarrowPhase

        public int[] c_itet;
        CudaDeviceVariable<int> g_itet;
        public int tet_stride;
        HashSet<Tuple<int, int>> NL2set = new HashSet<Tuple<int, int>>();

        public void NarrowPhaseCollisionDetection(List<Element> narrowList)
        {
            if (prms.CollisionScheme == ModelPrms.CollisionSchemes.None || narrowList.Count == 0) { cf.nCollisions = nImpacts = 0; return; }
//            Trace.WriteLine($"narrowList: {narrowList.Count}");
            int nTetra = narrowList.Count;
            Debug.Assert(nTetra % 2 == 0, "narrowList size is not even");
            sw.Restart();
            ctx.SetCurrent();

            // detemine the memory needed
            int nPairs = nTetra / 2;
            if (c_itet == null || c_itet.Length < nTetra) c_itet = new int[nTetra * 2];
            if(g_itet == null) g_itet = new CudaDeviceVariable<int>(nTetra * 2);
            else if(g_itet.Size < nTetra) { g_itet.Dispose(); g_itet = new CudaDeviceVariable<int>(nTetra * 2); }

            for (int i = 0; i < nTetra; i++) c_itet[i] = narrowList[i].globalElementId;
            g_itet.CopyToDevice(c_itet, 0, 0, nTetra*sizeof(int));

            kNarrowPhase.GridDimensions = new dim3(grid(nPairs), 1, 1);
            kNarrowPhase.Run(nPairs, g_dn.DevicePointer, g_ie.DevicePointer, g_itet.DevicePointer,
                el_all_stride, nd_stride);
            g_itet.CopyToHost(c_itet, 0, 0, nTetra * sizeof(int));
            sw.Stop();
            cf.ElT_GPU += sw.ElapsedMilliseconds;
            sw.Restart();

            // Tuple ( node# inside element, which element)
            NL2set.Clear();
            for (int i = 0; i < nPairs; i++) 
                if (c_itet[i * 2] != 0)
                {
                    int bits = c_itet[i * 2];
                    if ((bits & 1) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i*2+1].vrts[0].globalNodeId, narrowList[i*2].globalElementId));
                    if ((bits & 2) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 + 1].vrts[1].globalNodeId, narrowList[i * 2].globalElementId));
                    if ((bits & 4) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 + 1].vrts[2].globalNodeId, narrowList[i * 2].globalElementId));
                    if ((bits & 8) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 + 1].vrts[3].globalNodeId, narrowList[i * 2].globalElementId));

                    if ((bits & 16) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 ].vrts[0].globalNodeId, narrowList[i * 2+1].globalElementId));
                    if ((bits & 32) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 ].vrts[1].globalNodeId, narrowList[i * 2+1].globalElementId));
                    if ((bits & 64) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 ].vrts[2].globalNodeId, narrowList[i * 2+1].globalElementId));
                    if ((bits & 128) != 0) NL2set.Add(new Tuple<int, int>(narrowList[i * 2 ].vrts[3].globalNodeId, narrowList[i * 2+1].globalElementId));
                }

            sw.Stop();
            cf.ElT_CPU += sw.ElapsedMilliseconds;
            sw.Restart();

            // identify closest face
            // detemine the memory needed
            nPairs = NL2set.Count;
            if(nPairs == 0) { nImpacts = 0; return; }
            tet_stride = grid(nPairs) * block_size;
            if (c_itet.Length < tet_stride*4) c_itet = new int[tet_stride * 6];
            if (g_itet.Size < tet_stride * 4) { g_itet.Dispose(); g_itet = new CudaDeviceVariable<int>(tet_stride * 6); }

            int count = 0;
            foreach(Tuple<int,int> nodeElemPair in NL2set)
            {
                c_itet[count] = nodeElemPair.Item1;
                c_itet[count + tet_stride] = nodeElemPair.Item2;
                count++;
            }
            g_itet.CopyToDevice(c_itet, 0, 0, sizeof(int) * tet_stride * 2);
            kFindClosestFace.GridDimensions = new dim3(grid(nPairs), 1, 1);

            // result written to g_itet
            kFindClosestFace.Run(nPairs, tet_stride, g_itet.DevicePointer,
                g_dn.DevicePointer, g_ie.DevicePointer, g_ifc.DevicePointer, 
                nd_stride, el_all_stride, fc_stride);

            g_itet.CopyToHost(c_itet, 0, 0, sizeof(int) * tet_stride * 4);
            cf.nCollisions = nImpacts = count;
            // at this point c_itet[] contains strided (p_nd - f_nd1 -  f_nd2 -  f_nd3) sequences (global node id)
            mc.TransferFromAnotherArray(c_itet, nImpacts, tet_stride);

            sw.Stop();
            cf.ElT_GPU += sw.ElapsedMilliseconds;
        }



        #endregion

        #region Collision Response

        int[] c_icr; // integer data for impacts, i.e. pcsr 
        CudaDeviceVariable<int> g_icr;

        public void collisionResponse()
        {
            if (nImpacts == 0) return;
            sw.Restart();
            
            // allocate memory
            int cr_stride = grid(nImpacts) * block_size;
            int size_icr = cr_stride * 28;
            if (c_icr == null || c_icr.Length < size_icr)
            {
                if (g_icr != null) g_icr.Dispose();
                c_icr = new int[size_icr*2];
                g_icr = new CudaDeviceVariable<int>(size_icr*2);
            }

            const int pcsr_offset = 4;
            // populate and transfer to GPU
            Parallel.For(0, nImpacts, i_im => {

                for (int i = 0; i < 4; i++)
                {
                    Node ni = mc.allNodes[c_itet[i_im + i*tet_stride]];
                    c_icr[i_im + cr_stride * i] = ni.globalNodeId;
                    c_icr[i_im + cr_stride * (pcsr_offset + 16 + i)] = ni.anchored ? -1 : ni.altId;
                    c_icz[i_im + cz_stride * (pcsr_offset + 20 + i)] = ni.allNeighbors.Count;

                    for (int j = 0; j < 4; j++)
                    {
                        int pcsr_ij;
                        Node nj = mc.allNodes[c_itet[i_im + j * tet_stride]];
                        if (!ni.anchored && !nj.anchored && (!prms.SymmetricStructure || nj.altId >= ni.altId)) pcsr_ij = ni.pcsr[nj.altId];
                        else pcsr_ij = -1; // ni is anchored => ignore
                        c_icr[i_im + cr_stride * (pcsr_offset + (i * 4 + j))] = pcsr_ij;
                    }
                }
            });
            g_icr.CopyToDevice(c_icr, 0, 0, sizeof(int) * cr_stride * 28);

            // set grid size, execute kernel
            kCollisionResponseForce.GridDimensions = new dim3(grid(nImpacts), 1, 1);

            kCollisionResponseForce.Run(g_icr.DevicePointer, g_dn.DevicePointer, cf.TimeStep,
                g_dvals.DevicePointer, g_drhs.DevicePointer, nImpacts, cr_stride, nd_stride, prms.penaltyK, 
                prms.DistanceEpsilon, prms.AssemblyType);
            
            sw.Stop();
            cf.CollForce += sw.ElapsedMilliseconds;
        }
        #endregion
    }

}
