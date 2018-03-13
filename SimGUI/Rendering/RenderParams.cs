using System;
using System.Text;
using System.Xml.Serialization;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Diagnostics;

namespace icFlow.Rendering
{
    public class RenderPrms
    {
        #region animation
        [Category("Animation")]
        public int nFrames { get; set; } = 50;

        [Category("Animation")]
        public int FromStep { get; set; } = 1;

        [Category("Animation")]
        public int ToStep { get; set; } = -1;

        [Category("Animation")]
        [Description("Timer period in ms for rendering process")]
        public int Delay { get; set; } = 50;
        #endregion

        #region cohesive zones
        [Category("CZs")]
        [XmlIgnore]
        public Color DamagedCZColor { get; set; } = Color.Lime;
        public string DamagedCZColorHtml;

        [Category("CZs")]
        public bool DamagedCZvisible { get; set; } = true;

        [Category("CZs")]
        [XmlIgnore]
        public Color FailedCZColor { get; set; } = Color.DarkRed;
        public string FailedCZColorHtml;

        [Category("CZs")]
        [Description("Shown as midplane")]
        public bool FailedCZvisible { get; set; } = false;

        [Category("CZs")]
        public bool CZsEdgesVisible { get; set; } = true;

        [Category("CZs")]
        [XmlIgnore]
        public Color CZsEdgesColor { get; set; } = Color.Black;
        public string CZsEdgesColorHtml;

        [Category("CZs")]
        public float CZsEdgeWidth { get; set; } = 1f;

        [Category("CZs")]
        public bool CZsFill { get; set; } = true;


        #endregion

        #region grain
        [Category("Grains")]
        public bool ShowGrainBoundaries { get; set; } = true;

        [Category("Grains")]
        public float GrainBoundaryWidth { get; set; } = 1.9f;

        [Category("Grains")]
        [XmlIgnore]
        public Color GrainBoundaryColor { get; set; } = Color.Black;
        public string GrainBoundaryColorHtml;

        [Category("Grains")]
        public bool RenderBoundariesAsCylinders { get; set; } = true;

        [Category("Grains")]
        [Description("If the boundaries are rendered as cylinders, this sets the number of sides")]
        public int CylinderSides { get; set; } = 5;

        [Category("Grains")]
        public double CylinderRadius { get; set; } = 0.00025;

        #endregion

        #region surface
        [Category("Surface")]
        [Description("Only apples to outer shell")]
        public float TransparencyCoeff { get; set; } = 0.4f;

        [Category("Surface")]
        public bool ShowSurface { get; set; } = true;

        [Category("Surface")]
        public bool UseGrainColor { get; set; } = false;

        [Category("Surface")]
        [XmlIgnore]
        public Color SurfaceColor { get; set; } = Color.Silver;
        public string SurfaceColorHtml;

        [Category("Surface")]
        [XmlIgnore]
        public Color CreatedSurfaceColor { get; set; } = Color.FromArgb(186,47,40);
        public string CreatedSurfaceColorHtml;

        [Category("Surface")]
        public bool ShowTetraEdgesCreated { get; set; } = true;

        [Category("Surface")]
        [Description("For rendering ")]
        public bool ShowAllTetraEdges { get; set; } = true;

        [Category("Surface")]
        public float TetraEdgesWidth { get; set; } = 1f;

        [Category("Surface")]
        [XmlIgnore]
        public Color TetraEdgesColor { get; set; } = Color.Black;
        public string TetraEdgesColorHtml;

        #endregion

        #region rigid objects
        [Category("Rigid")]
        [Description("Show edges of tetrahedra on non-deformable objects")]
        public bool ShowTetraEdgesRigid { get; set; } = true;

        [Category("Rigid")]
        [XmlIgnore]
        public Color RigidObjectColor { get; set; } = Color.FromArgb(19,24,43);
        public string RigidObjectColorHtml;

        #endregion

        #region misc
        public bool WhiteBackground { get; set; } = true;
        public bool ShowSimulationTime { get; set; } = true;
        public bool ShowSimulationFrame { get; set; } = true;
        public string renderFolder { get; set; } = "render";
        #endregion

        #region view angle, offset and scale

        public float theta = 0;
        public float phi = 0;           // view angle
        public double scale = 0.17;

        public float dx = 0;
        public float dy = 0;           // rendering offset
        #endregion

        #region perspective

        [Category("Perspective")]
        public bool UseFrustum { get; set; } = true;

        [Category("Perspective")]
        [Description("In degrees")]
        public double fovY { get; set; } = 20;

        [Category("Perspective")]
        public double zNear { get; set; } = 0.1;

        [Category("Perspective")]
        public double zFar { get; set; } = 10000;

        [Category("Perspective")]
        public double zOffset { get; set; } = 0.5;

        #endregion

        #region light

        [Category("Light")]
        public bool Light0 { get; set; } = true;
        [Category("Light")]
        public bool Light1 { get; set; } = true;
        [Category("Light")]
        public bool Light2 { get; set; } = true;


        [Category("Light")]
        public float L0x { get; set; } = -0.5f;
        [Category("Light")]
        public float L0y { get; set; } = 0.5f;
        [Category("Light")]
        public float L0z { get; set; } = 0.7f;
        [Category("Light")]
        public float L0intensity { get; set; } = 0.7f;

        [Category("Light")]
        public float L1x { get; set; } = -0.5f;
        [Category("Light")]
        public float L1y { get; set; } = 0;
        [Category("Light")]
        public float L1z { get; set; } = -0.4f;
        [Category("Light")]
        public float L1intensity { get; set; } = 0.7f;

        [Category("Light")]
        public float L2x { get; set; } = 1;
        [Category("Light")]
        public float L2y { get; set; } = 0;
        [Category("Light")]
        public float L2z { get; set; } = -1;
        [Category("Light")]
        public float L2intensity { get; set; } = 0.5f;
        #endregion

        #region load/save

        const string fileName = "RenderParams";
        public void Save(string Path)
        {
            string fullFileName = Path == null ? fileName : $"{Path}\\{fileName}";

            DamagedCZColorHtml = ColorTranslator.ToHtml(DamagedCZColor);
            FailedCZColorHtml = ColorTranslator.ToHtml(FailedCZColor);
            GrainBoundaryColorHtml = ColorTranslator.ToHtml(GrainBoundaryColor);
            SurfaceColorHtml = ColorTranslator.ToHtml(SurfaceColor);
            CreatedSurfaceColorHtml = ColorTranslator.ToHtml(CreatedSurfaceColor);
            TetraEdgesColorHtml = ColorTranslator.ToHtml(TetraEdgesColor);
            RigidObjectColorHtml = ColorTranslator.ToHtml(RigidObjectColor);

            Stream str;
            try {
                str = File.Create(fullFileName);
            } catch
            {
                Trace.WriteLine("could not write RenderParams");
                return;
            }
            StreamWriter sw = new StreamWriter(str);
            XmlSerializer xs = new XmlSerializer(typeof(RenderPrms));
            xs.Serialize(sw, this);
            sw.Close();
            Trace.WriteLine("saved RenderParams");
        }

        public static RenderPrms Load(string Path = null)
        {
            string fullFileName = Path == null? fileName :  $"{Path}\\{fileName}";
            if (!File.Exists(fullFileName))
            {
                Trace.WriteLine("using default RenderPrms");
                return new RenderPrms(); // use defaults
            }

            Stream str = File.OpenRead(fullFileName);
            XmlSerializer xs = new XmlSerializer(typeof(RenderPrms));
            try {
                Trace.WriteLine("attempting to load RenderPrms");

                RenderPrms prms = (RenderPrms)xs.Deserialize(str);
                str.Close();

                prms.DamagedCZColor = ColorTranslator.FromHtml(prms.DamagedCZColorHtml);
                prms.FailedCZColor = ColorTranslator.FromHtml(prms.FailedCZColorHtml);
                prms.GrainBoundaryColor = ColorTranslator.FromHtml(prms.GrainBoundaryColorHtml);
                prms.SurfaceColor = ColorTranslator.FromHtml(prms.SurfaceColorHtml);
                prms.CreatedSurfaceColor = ColorTranslator.FromHtml(prms.CreatedSurfaceColorHtml);
                prms.TetraEdgesColor = ColorTranslator.FromHtml(prms.TetraEdgesColorHtml);
                prms.RigidObjectColor = ColorTranslator.FromHtml(prms.RigidObjectColorHtml);
                return prms;
            } catch (Exception e)
            {
                Trace.WriteLine("using default RenderPrms because exception occurred:" + e.Message);
                str.Close();
                File.Delete(fullFileName);
                return new RenderPrms();
            }
        }

        #endregion

        #region text and plots overlay
        [Category("Overlay")]
        public bool RenderText { get; set; } = true;

        [Category("Overlay")]
        public string Comment { get; set; } = "-";

        #endregion
    }
}
