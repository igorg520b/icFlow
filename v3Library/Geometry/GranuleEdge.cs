
namespace icFlow
{
    /// <summary>
    ///  represents portion of the granule edge 
    ///  for drawing purpose mainly 
    /// </summary>
    public class GranuleEdge
    {
        public Node[] vrts = new Node[2]; // pointers to vertices
        public bool exposed;

        // non-serialized
        public int granule;

        public void MarkExposed()
        {
            exposed = (vrts[0].isSurface && vrts[1].isSurface);
        }
    }
}
