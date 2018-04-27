using System;
using System.Collections.Generic;

namespace icFlow
{
    [Serializable]
    public class Translation : IComparable<Translation>
    {
        public double dx { get; set; }
        public double dy { get; set; }
        public double dz { get; set; }
        public double t { get; set; }

        public static Translation Interpolate(Translation t1, Translation t2, double r)
        {
            Translation result = new Translation();
            result.dx = r * t1.dx + (1 - r) * t2.dx;
            result.dy = r * t1.dy + (1 - r) * t2.dy;
            result.dz = r * t1.dz + (1 - r) * t2.dz;
            return result;
        }

        public int CompareTo(Translation other)
        {
            if (t > other.t) return 1;
            else if (t == other.t) return 0;
            else return -1;
        }

        public override string ToString() { return $"{t:0.000}; ({dx:0.00},{dy:0.00},{dz:0.00})"; }
    }

    [Serializable]
    public class TranslationCollection : List<Translation>
    {
        public Translation GetTranslation(double time)
        {
            // array must be sorted
            Translation result;
            if (this.Count == 1) result = this[0];
            else if (this.Count == 0) result = new Translation();
            else
            {
                result = new Translation();
                if (time >= this[Count - 1].t) result = this[Count - 1];
                else if (time < this[0].t) result = new Translation();
                else
                {
                    Translation lower = null, higher = null;
                    int i = 0;
                    while (!(this[i].t <= time && time < this[i + 1].t)) i++;
                    lower = this[i];
                    higher = this[i + 1];
                    result = Translation.Interpolate(lower, higher, (higher.t - time) / (higher.t - lower.t));
                }


            }

            return result;
        }
    }
}
