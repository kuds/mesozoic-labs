/* Orthographic, opaque triangle rasterizer. Pixels retain the nearest world
   depth. Vertex lighting is interpolated within each visible triangle. */
#include <math.h>
#include <stdint.h>

void raster(int n, int w, int h, const float *points, const float *colors,
            float *depth, uint8_t *rgba) {
    for (int t = 0; t < n; t++) {
        const float *v = points + 9*t;
        const float *c = colors + 9*t;
        int minx = fmaxf(0, floorf(fminf(v[0], fminf(v[3], v[6]))));
        int maxx = fminf(w-1, ceilf(fmaxf(v[0], fmaxf(v[3], v[6]))));
        int miny = fmaxf(0, floorf(fminf(v[1], fminf(v[4], v[7]))));
        int maxy = fminf(h-1, ceilf(fmaxf(v[1], fmaxf(v[4], v[7]))));
        float den = (v[4]-v[7])*(v[0]-v[6]) + (v[6]-v[3])*(v[1]-v[7]);
        if (fabsf(den)<1e-8f) continue;
        for (int y=miny; y<=maxy; y++) {
            for (int x=minx; x<=maxx; x++) {
                float px = x+0.5f-v[6], py = y+0.5f-v[7];
                float a = ((v[4]-v[7])*px + (v[6]-v[3])*py)/den;
                float b = ((v[7]-v[1])*px + (v[0]-v[6])*py)/den;
                float d = 1.f-a-b;
                if (a < -1e-6f || b < -1e-6f || d < -1e-6f) continue;
                int i = y*w+x;
                float z = a*v[2]+b*v[5]+d*v[8];
                if (z <= depth[i]) continue;
                depth[i] = z;
                for (int k=0; k<3; k++) rgba[4*i+k] = (uint8_t)(a*c[k]+b*c[k+3]+d*c[k+6]);
                rgba[4*i+3]=255;
            }
        }
    }
}
