// The intro's bison logo: a wild bison charging head-on at the viewer, as a black-metal statue.
//
// Model: a signed-distance field of ellipsoids and round cones blended smoothly, in metres (y up, the bison faces
// +z, toward the camera; ground at y = 0). The pose and proportions follow a reference photo of a charging bison
// (used for proportions only): a peaked hump, a long head lowered at us under a big curly mop, thick horns swept
// out and up, eyes on the sides of the face, a broad nose, a beard, shaggy forearms, one leg folded up and one
// reaching for the ground. Fur is displacement close to the surface: cell-noise curls on the head, woolly locks
// with grooves on the hump and shoulders (a lighter metal, like the photo's two-tone coat), long hanging strands
// on the mane, beard and forearms.
// Render: sphere tracing through a long lens, normals from the field, ambient occlusion, soft shadows, shaded as
// dark metal: almost no body colour, light from reflections of a cool softbox upper left, the red sign above and
// two cool strips behind (energy-conserving, so rough fur stays dark), plus glints in the eyes.
//
// Build: g++ -O2 -fopenmp tools/bison_logo.cpp -o bison_logo   (tools/bison_logo.py builds and runs it)
// Usage: bison_logo OUT.pam [WIDTH HEIGHT SUPERSAMPLE]     writes RGBA (PAM), straight alpha
//        bison_logo probe X Y [X Y ...]                     where the bare face is, for placing eyes and such
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct V { float x, y, z; };
static inline V v3(float x, float y, float z) { return {x, y, z}; }
static inline V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V operator*(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline V operator*(V a, V b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline V operator/(V a, V b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
static inline float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline float len(V a) { return std::sqrt(dot(a, a)); }
static inline V norm(V a) { float l = len(a); return a * (1.0f / (l > 1e-9f ? l : 1e-9f)); }
static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
static inline V mixv(V a, V b, float t) { return a + (b - a) * t; }
static inline float smoothstep(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

// ---------------------------------------------------------------------------------------------- primitives
static inline float sdEllipsoid(V p, V c, V r) {
    V q = (p - c) / r; float k0 = len(q), k1 = len(q / r);
    return k0 * (k0 - 1.0f) / std::max(k1, 1e-6f);
}
static inline float sdSphere(V p, V c, float r) { return len(p - c) - r; }
static float sdEllipsoidR(V p, V c, V r, float roll, float pitch, float yaw) {
    V d = p - c;
    float cy = std::cos(-yaw), sy = std::sin(-yaw); d = v3(cy * d.x + sy * d.z, d.y, -sy * d.x + cy * d.z);
    float cp = std::cos(-pitch), sp = std::sin(-pitch); d = v3(d.x, cp * d.y - sp * d.z, sp * d.y + cp * d.z);
    float cr = std::cos(-roll), sr = std::sin(-roll); d = v3(cr * d.x - sr * d.y, sr * d.x + cr * d.y, d.z);
    return sdEllipsoid(d, v3(0, 0, 0), r);
}
static float sdRoundCone(V p, V a, V b, float r1, float r2) {     // iq's round cone
    V ba = b - a; float l2 = dot(ba, ba), rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1.0f / l2;
    V pa = p - a; float y = dot(pa, ba), z = y - l2;
    V xv = pa * l2 - ba * y; float x2 = dot(xv, xv), y2 = y * y * l2, z2 = z * z * l2;
    float k = (rr > 0 ? 1.f : -1.f) * rr * rr * x2;
    if ((z > 0 ? 1.f : -1.f) * a2 * z2 > k) return std::sqrt(x2 + z2) * il2 - r2;
    if ((y > 0 ? 1.f : -1.f) * a2 * y2 < k) return std::sqrt(x2 + y2) * il2 - r1;
    return (std::sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
static inline float smin(float a, float b, float k) {
    float h = clampf(0.5f + 0.5f * (b - a) / k, 0, 1); return mixf(b, a, h) - k * h * (1 - h);
}
static inline float smax(float a, float b, float k) { return -smin(-a, -b, k); }
static inline V mirrorX(V p) { return {std::fabs(p.x), p.y, p.z}; }

// ---------------------------------------------------------------------------------------------- noise
static inline uint32_t hash3(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)z * 83492791u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16; return h;
}
static inline float h01(int x, int y, int z) { return (hash3(x, y, z) & 0xFFFFFF) * (1.0f / 16777215.0f); }
static float vnoise(V p) {                  // value noise in [0, 1]
    float fx = std::floor(p.x), fy = std::floor(p.y), fz = std::floor(p.z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float ux = p.x - fx, uy = p.y - fy, uz = p.z - fz;
    ux = ux * ux * (3 - 2 * ux); uy = uy * uy * (3 - 2 * uy); uz = uz * uz * (3 - 2 * uz);
    float a = mixf(h01(ix, iy, iz), h01(ix + 1, iy, iz), ux), b = mixf(h01(ix, iy + 1, iz), h01(ix + 1, iy + 1, iz), ux);
    float c = mixf(h01(ix, iy, iz + 1), h01(ix + 1, iy, iz + 1), ux), d = mixf(h01(ix, iy + 1, iz + 1), h01(ix + 1, iy + 1, iz + 1), ux);
    return mixf(mixf(a, b, uy), mixf(c, d, uy), uz);
}
static float fbm(V p, int oct) {
    float s = 0, a = 0.5f, n = 0;
    for (int i = 0; i < oct; i++) { s += a * vnoise(p); n += a; p = p * 2.03f + v3(17.1f, 5.3f, 9.7f); a *= 0.5f; }
    return s / n;
}
static float worley(V p) {                  // distance to the nearest feature point (F1), cells of size 1
    int ix = (int)std::floor(p.x), iy = (int)std::floor(p.y), iz = (int)std::floor(p.z);
    float best = 9;
    for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
        uint32_t h = hash3(ix + i, iy + j, iz + k);
        V f = v3(ix + i + (h & 1023) / 1023.f, iy + j + ((h >> 10) & 1023) / 1023.f, iz + k + ((h >> 20) & 1023) / 1023.f);
        V d = f - p; best = std::min(best, dot(d, d));
    }
    return std::sqrt(best);
}

// ---------------------------------------------------------------------------------------------- the bison
enum Mat { FUR_CURL, FUR_SHAG, FUR_HUMP, FUR_SHORT, NOSE, HORN, HOOF, EYE, NMAT };

static V EYE_C = {0.26f, 1.27f, 0.698f};   // the eye's centre (mirrored)
static bool NO_EYES = false;                 // probe the bare face
struct Parts {   // distance to each part group, before fur
    float curl, shag, hump, shortfur, nose, horn, hoof, eye;
};

// Leg as a chain of round cones
static float leg(V p, const V* pts, const float* r, int n, float k) {
    float d = 1e9f;
    for (int i = 0; i + 1 < n; i++) d = smin(d, sdRoundCone(p, pts[i], pts[i + 1], r[i], r[i + 1]), k);
    return d;
}

static Parts parts(V p) {
    // Proportions from the reference photo (a long lens, so close to orthographic): hump top 2.15 m, mop top 1.85,
    // horn tips 1.88, eyes 1.27 (0.52 m apart), nose 0.90, beard to 0.45, knees 0.43, the lifted hoof 0.2, the
    // planted hoof 0; body 1.28 m wide, head 0.85 m across the eyes, horns 1.08 m tip to tip.
    Parts P;
    V q = mirrorX(p);
    // hump and body (coarse fur): the peaked dome above the shoulders, the barrel behind
    float hump = sdEllipsoid(p, v3(0, 1.52f, -0.45f), v3(0.69f, 0.68f, 0.80f));
    hump = smin(hump, sdEllipsoid(p, v3(0, 1.98f, -0.62f), v3(0.32f, 0.18f, 0.50f)), 0.24f);
    hump = smin(hump, sdEllipsoid(p, v3(0, 1.16f, -0.24f), v3(0.68f, 0.50f, 0.56f)), 0.22f);   // the shoulders, down the sides
    hump = smin(hump, sdEllipsoid(p, v3(0, 1.25f, -1.40f), v3(0.58f, 0.60f, 1.00f)), 0.30f);
    P.hump = hump;
    // shaggy: the neck mane and shoulders, chest hair, the beard, and the long hair on the forearms
    float shag = sdEllipsoid(p, v3(0, 1.36f, 0.14f), v3(0.50f, 0.52f, 0.44f));
    shag = smin(shag, sdEllipsoid(p, v3(0, 1.10f, 0.10f), v3(0.56f, 0.40f, 0.42f)), 0.12f);
    shag = smin(shag, sdEllipsoid(p, v3(0, 0.90f, 0.44f), v3(0.19f, 0.30f, 0.20f)), 0.06f);    // chest hair under the head
    shag = smin(shag, sdEllipsoid(p, v3(0, 0.64f, 0.70f), v3(0.14f, 0.24f, 0.13f)), 0.07f);    // beard
    {
        V a[2] = {v3(-0.37f, 1.02f, 0.22f), v3(-0.28f, 0.50f, 0.53f)}; float r[2] = {0.18f, 0.13f};
        V b[2] = {v3(0.37f, 1.02f, 0.22f), v3(0.22f, 0.50f, 0.47f)};
        shag = smin(shag, leg(p, a, r, 2, 0.05f), 0.05f);
        shag = smin(shag, leg(p, b, r, 2, 0.05f), 0.05f);
    }
    P.shag = shag;
    // curls: the head is a wedge pointing down at us, broad brow to muzzle, under a big mop, with side whiskers
    float curl = sdRoundCone(p, v3(0, 1.52f, 0.45f), v3(0, 0.96f, 0.80f), 0.34f, 0.19f);
    curl = smin(curl, sdEllipsoid(p, v3(0, 1.68f, 0.42f), v3(0.36f, 0.20f, 0.28f)), 0.09f);              // the mop
    curl = smin(curl, sdEllipsoid(q, v3(0.30f, 1.28f, 0.44f), v3(0.13f, 0.33f, 0.20f)), 0.08f);          // side whiskers
    curl = smin(curl, sdEllipsoidR(q, EYE_C + v3(-0.012f, 0.062f, -0.005f), v3(0.12f, 0.045f, 0.065f), -0.25f, 0.0f, 0.35f), 0.04f);   // brows
    P.curl = curl;
    // short fur: the muzzle, around the eyes, the lower legs (knees, cannons, pasterns), the hind legs
    float sf = sdEllipsoid(p, v3(0, 0.95f, 0.81f), v3(0.18f, 0.15f, 0.15f));
    if (!NO_EYES) sf = smin(sf, sdEllipsoidR(q, EYE_C + v3(0, -0.004f, -0.03f), v3(0.075f, 0.048f, 0.045f), -0.2f, 0.0f, 0.45f), 0.03f);
    {
        V L[5] = {v3(-0.28f, 0.50f, 0.54f), v3(-0.272f, 0.43f, 0.57f), v3(-0.248f, 0.28f, 0.50f), v3(-0.240f, 0.225f, 0.535f), v3(-0.238f, 0.215f, 0.542f)};
        V R[5] = {v3(0.22f, 0.50f, 0.48f), v3(0.215f, 0.43f, 0.51f), v3(0.075f, 0.12f, 0.71f), v3(0.058f, 0.065f, 0.765f), v3(0.056f, 0.055f, 0.772f)};
        float Lr[5] = {0.135f, 0.110f, 0.088f, 0.074f, 0.068f};
        sf = std::min(sf, leg(p, L, Lr, 5, 0.03f));
        sf = std::min(sf, leg(p, R, Lr, 5, 0.03f));
        V H[3] = {v3(0.28f, 1.05f, -1.80f), v3(0.25f, 0.50f, -1.65f), v3(0.23f, 0.05f, -1.55f)};
        float Hr[3] = {0.17f, 0.09f, 0.065f};
        sf = std::min(sf, leg(q, H, Hr, 3, 0.03f));
    }
    P.shortfur = sf;
    // nose pad (leather)
    P.nose = sdEllipsoid(p, v3(0, 0.90f, 0.905f), v3(0.17f, 0.10f, 0.06f));
    // horns: thick, out of the top corners of the head, out, up and in, sharp tips
    {
        V H[6] = {v3(0.31f, 1.50f, 0.52f), v3(0.44f, 1.53f, 0.50f), v3(0.525f, 1.62f, 0.49f), v3(0.552f, 1.76f, 0.48f),
                  v3(0.525f, 1.875f, 0.49f), v3(0.468f, 1.945f, 0.51f)};
        float Hr[6] = {0.102f, 0.087f, 0.068f, 0.050f, 0.031f, 0.006f};
        P.horn = leg(q, H, Hr, 6, 0.02f);
    }
    // hooves: two toes each; the lifted one points down, the planted one reaches the ground in front
    {
        float d = 1e9f;
        d = std::min(d, sdRoundCone(p, v3(-0.267f, 0.215f, 0.54f), v3(-0.273f, 0.138f, 0.575f), 0.050f, 0.035f));
        d = std::min(d, sdRoundCone(p, v3(-0.209f, 0.215f, 0.54f), v3(-0.203f, 0.138f, 0.575f), 0.050f, 0.035f));
        d = std::min(d, sdRoundCone(p, v3(0.028f, 0.055f, 0.775f), v3(0.022f, 0.008f, 0.850f), 0.052f, 0.036f));
        d = std::min(d, sdRoundCone(p, v3(0.086f, 0.055f, 0.775f), v3(0.092f, 0.008f, 0.850f), 0.052f, 0.036f));
        d = std::min(d, sdRoundCone(q, v3(0.23f, 0.07f, -1.53f), v3(0.23f, 0.02f, -1.48f), 0.055f, 0.038f));
        P.hoof = d;
    }
    P.eye = NO_EYES ? 1e9f : sdEllipsoidR(q, EYE_C, v3(0.040f, 0.027f, 0.030f), -0.22f, 0.0f, 0.45f);
    return P;
}

// fur displacement for a material at p (metres outward)
static float furDisp(int m, V p) {
    switch (m) {
    case FUR_CURL: {       // tight curls: rounded bumps from cellular noise, two sizes
        V w = v3(fbm(p * 9.0f, 2), fbm(p * 9.0f + v3(3.1f, 7.7f, 1.3f), 2), 0) * 0.02f;
        float f1 = worley((p + w) * 28.0f) / 0.62f, f2 = worley((p + w) * 55.0f + v3(5, 1, 3)) / 0.62f;
        float c1 = std::sqrt(std::max(0.0f, 1.0f - f1 * f1)), c2 = std::sqrt(std::max(0.0f, 1.0f - f2 * f2));
        float clump = fbm(p * 7.0f + v3(2, 9, 4), 2);
        return 0.016f * c1 + 0.006f * c2 + 0.022f * (clump - 0.45f);
    }
    case FUR_SHAG: {       // long shaggy hair: noise stretched along the hang of the hair (down)
        V s = v3(p.x * 12.0f, p.y * 2.2f, p.z * 12.0f);
        float n = fbm(s, 4);
        float strands = 0.5f + 0.5f * std::sin(p.x * 80.0f + p.z * 63.0f + n * 11.0f);
        float fine = vnoise(v3(p.x * 90.0f, p.y * 12.0f, p.z * 90.0f));
        return 0.060f * (n - 0.35f) + 0.014f * strands + 0.008f * fine;
    }
    case FUR_HUMP: {       // the hump's coarse wool: locks in vertical clumps
        V w = v3(fbm(p * 6.0f, 2), fbm(p * 6.0f + v3(4.2f, 1.7f, 8.1f), 2), 0) * 0.05f;
        float f1 = worley(v3((p.x + w.x) * 36.0f, (p.y + w.y) * 14.0f, p.z * 36.0f)) / 0.62f;
        float lock = std::sqrt(std::max(0.0f, 1.0f - f1 * f1));
        float n = fbm(v3(p.x * 12.0f, p.y * 3.5f, p.z * 12.0f), 4);
        float grooves = 1.0f - std::fabs(2.0f * vnoise(v3(p.x * 40.0f, p.y * 8.0f, p.z * 40.0f)) - 1.0f);
        return 0.010f * lock + 0.030f * (n - 0.45f) + 0.008f * grooves;
    }
    case FUR_SHORT: return 0.006f * (fbm(p * 55.0f, 2) - 0.5f);
    case NOSE: return 0.0025f * (vnoise(p * 140.0f) - 0.5f);
    default: return 0.0f;
    }
}

struct Hit { float d; int mat; };
static const float SHELL = 0.09f;   // fur only matters this close to the surface

static Hit scene(V p, bool fur = true) {
    Parts P = parts(p);
    float body = smin(P.hump, P.shag, 0.12f);
    body = smin(body, P.curl, 0.05f);
    body = smin(body, P.shortfur, 0.05f);
    float base = smin(body, P.nose, 0.05f);
    base = smin(base, P.horn, 0.015f);
    base = std::min(base, P.hoof);
    // eye sockets, nostrils, mouth
    V q = mirrorX(p);
    if (!NO_EYES) base = smax(base, -sdEllipsoidR(q, EYE_C + v3(0, 0.002f, 0.02f), v3(0.054f, 0.032f, 0.040f), -0.22f, 0.0f, 0.45f), 0.010f);
    base = std::min(base, P.eye);
    base = smax(base, -sdEllipsoidR(q, v3(0.082f, 0.893f, 0.955f), v3(0.048f, 0.027f, 0.05f), -0.55f, 0.0f, 0.3f), 0.008f);
    base = smax(base, -sdEllipsoid(p, v3(0, 0.80f, 0.86f), v3(0.11f, 0.008f, 0.07f)), 0.006f);
    // which material: the nearest group
    float ds[NMAT] = {P.curl, P.shag, P.hump, P.shortfur, P.nose, P.horn, P.hoof, P.eye};
    int m = 0;
    for (int i = 1; i < NMAT; i++) if (ds[i] < ds[m]) m = i;
    if (!fur || base > SHELL) return {base, m};
    float eyeKeep = NO_EYES ? 1.0f : smoothstep(0.045f, 0.08f, len(q - EYE_C));   // keep the hair out of the eyes
    // blend the fur of nearby groups so seams between them don't show
    float wsum = 0, disp = 0;
    for (int i = 0; i < NMAT; i++) {
        float w = std::exp(-std::max(0.0f, ds[i] - ds[m]) * 70.0f);
        if (w < 0.02f) continue;
        disp += w * furDisp(i, p); wsum += w;
    }
    return {base - disp / wsum * eyeKeep, m};
}

// ---------------------------------------------------------------------------------------------- rendering
static V calcNormal(V p) {
    const float h = 0.0018f;
    V k1 = v3(1, -1, -1), k2 = v3(-1, -1, 1), k3 = v3(-1, 1, -1), k4 = v3(1, 1, 1);
    return norm(k1 * scene(p + k1 * h).d + k2 * scene(p + k2 * h).d + k3 * scene(p + k3 * h).d + k4 * scene(p + k4 * h).d);
}
static float ao(V p, V n) {
    float occ = 0, sca = 1;
    for (int i = 1; i <= 5; i++) {
        float h = 0.015f + 0.06f * i;
        float d = scene(p + n * h, false).d;
        occ += (h - d) * sca; sca *= 0.75f;
    }
    return clampf(1.0f - 2.2f * occ, 0.0f, 1.0f);
}
static float softShadow(V ro, V rd, float k) {
    float res = 1, t = 0.02f;
    for (int i = 0; i < 48 && t < 3.0f; i++) {
        float h = scene(ro + rd * t, false).d;
        res = std::min(res, k * h / t);
        if (res < 0.002f) return 0;
        t += clampf(h, 0.01f, 0.25f);
    }
    return clampf(res, 0, 1);
}

static float lobe(V r, V dir, float w, float spread) {   // a soft light seen in reflection; rough surfaces spread it thinner
    float ws = w * spread;
    float t = smoothstep(1.0f - ws, 1.0f - ws * 0.25f, dot(r, norm(dir)));
    return t * (w / ws) * (w / ws) * 1.0f;
}
static V env(V r, float rough) {      // what black metal reflects: darkness, the red sign above, a softbox, two rims
    float spread = 1.0f + rough * 4.0f;
    V c = v3(0.003f, 0.003f, 0.004f);
    float red = lobe(r, v3(0, 0.92f, 0.38f), 0.09f, spread) * smoothstep(0.9f, 0.3f, std::fabs(r.x));
    c = c + v3(1.0f, 0.05f, 0.03f) * (0.7f * red);
    c = c + v3(0.80f, 0.86f, 0.96f) * (2.6f * lobe(r, v3(-0.55f, 0.55f, 0.63f), 0.06f, spread));
    c = c + v3(0.66f, 0.72f, 0.84f) * (2.4f * (lobe(r, v3(-0.75f, 0.25f, -0.62f), 0.10f, spread) + lobe(r, v3(0.75f, 0.25f, -0.62f), 0.10f, spread)));
    return c;
}

struct Mtl { float rough, f0, base; };
static Mtl mtlOf(int m) {
    switch (m) {
    case HORN: return {0.18f, 0.50f, 0.010f};
    case HOOF: return {0.24f, 0.40f, 0.010f};
    case EYE: return {0.04f, 0.60f, 0.004f};
    case NOSE: return {0.26f, 0.36f, 0.008f};
    case FUR_HUMP: return {0.42f, 0.42f, 0.055f};     // the hump: a lighter, brushed gunmetal (the photo's tan coat)
    case FUR_SHORT: return {0.44f, 0.24f, 0.010f};
    default: return {0.50f, 0.20f, 0.008f};           // the dark curls and shaggy hair
    }
}

static V shade(V p, V rd, int m) {
    V n = calcNormal(p);
    V v = rd * -1.0f;
    Mtl M = mtlOf(m);
    float nv = clampf(dot(n, v), 0, 1);
    V r = rd - n * (2.0f * dot(rd, n));
    float occ = ao(p, n);
    float fres = M.f0 + (1 - M.f0) * std::pow(1 - nv, 5.0f);
    V col = env(r, M.rough) * fres;
    struct L { V d; V c; float i; bool sh; };
    L lights[4] = {{norm(v3(0, 0.94f, 0.34f)), v3(1.0f, 0.06f, 0.035f), 0.45f, true},      // the red sign above
                   {norm(v3(-0.55f, 0.55f, 0.63f)), v3(0.80f, 0.86f, 0.96f), 1.9f, true},    // cool key, upper left
                   {norm(v3(-0.70f, 0.25f, -0.67f)), v3(0.62f, 0.68f, 0.80f), 2.2f, false},  // rims behind
                   {norm(v3(0.70f, 0.25f, -0.67f)), v3(0.62f, 0.68f, 0.80f), 2.2f, false}};
    for (auto& l : lights) {
        float nl = dot(n, l.d); if (nl <= 0) continue;
        V h = norm(l.d + v);
        float nh = clampf(dot(n, h), 0, 1), a = M.rough * M.rough, a2 = a * a;
        float dd = nh * nh * (a2 - 1) + 1, D = a2 / (3.14159f * dd * dd);
        float spec = D * fres * 0.25f / std::max(nv, 0.25f);
        float s = l.sh ? softShadow(p + n * 0.01f, l.d, 10.0f) : 1.0f;
        col = col + l.c * ((spec + M.base) * nl * l.i * s);
    }
    col = col * (0.25f + 0.75f * occ);
    col = col + v3(0.40f, 0.44f, 0.52f) * (0.06f * std::pow(1 - nv, 3.0f) * occ);   // a faint sheen on the silhouette
    if (m == EYE) {   // glints: the red sign above, the cool key
        float g = std::pow(clampf(dot(r, norm(v3(0.05f, 0.80f, 0.60f))), 0, 1), 300.0f);
        float k = std::pow(clampf(dot(r, norm(v3(-0.55f, 0.55f, 0.63f))), 0, 1), 500.0f);
        col = col + v3(1.0f, 0.12f, 0.06f) * (2.5f * g) + v3(0.9f, 0.95f, 1.0f) * (2.0f * k);
    }
    return col;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "probe") {      // surface z at (x, y), marching from the front, no fur
        NO_EYES = true;
        for (int k = 2; k + 1 < argc; k += 2) {
            float x = atof(argv[k]), y = atof(argv[k + 1]), z = 2.0f;
            while (z > -2.5f && scene(v3(x, y, z), false).d > 0.0005f) z -= std::max(0.0005f, scene(v3(x, y, z), false).d * 0.5f);
            Parts P = parts(v3(x, y, z));
            printf("x %.3f y %.3f -> z %.3f  (curl %.3f shag %.3f short %.3f nose %.3f eye %.3f)\n", x, y, z, P.curl, P.shag, P.shortfur, P.nose, P.eye);
        }
        return 0;
    }
    const char* out = argc > 1 ? argv[1] : "bison.pam";
    int W = argc > 2 ? atoi(argv[2]) : 1100, H = argc > 3 ? atoi(argv[3]) : 1100, SS = argc > 4 ? atoi(argv[4]) : 3;
    float camY = argc > 5 ? atof(argv[5]) : 0.9f, camZ = argc > 6 ? atof(argv[6]) : 10.0f, fov = argc > 7 ? atof(argv[7]) : 14.0f;   // a long lens, like the photo
    V ro = v3(0, camY, camZ), ta = v3(0, 1.08f, 0);
    V cw = norm(ta - ro), cu = norm(cross(cw, v3(0, 1, 0))), cv = cross(cu, cw);
    float fl = 1.0f / std::tan(fov * 0.5f * 3.14159265f / 180.0f);
    V bc = v3(0, 1.08f, -0.35f); float br = 2.0f;     // bounding sphere
    std::vector<float> img(W * H * 4, 0.0f);
#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            V acc = v3(0, 0, 0); float cov = 0;
            for (int sy = 0; sy < SS; sy++) for (int sx = 0; sx < SS; sx++) {
                float px = (2.0f * (i + (sx + 0.5f) / SS) - W) / H, py = -(2.0f * (j + (sy + 0.5f) / SS) - H) / H;
                V rd = norm(cu * px + cv * py + cw * fl);
                V oc = ro - bc; float b = dot(oc, rd), c = dot(oc, oc) - br * br, disc = b * b - c;
                if (disc < 0) continue;
                float t = std::max(0.0f, -b - std::sqrt(disc)), tmax = -b + std::sqrt(disc);
                bool hit = false; int mat = 0;
                for (int s = 0; s < 400 && t < tmax; s++) {
                    V p = ro + rd * t;
                    Hit h = scene(p);
                    if (h.d < 0.0004f * t) { hit = true; mat = h.mat; break; }
                    t += h.d * (h.d < SHELL + 0.02f ? 0.45f : 0.9f);
                }
                if (!hit) continue;
                V col = shade(ro + rd * t, rd, mat);
                acc = acc + col; cov += 1;
            }
            float n = (float)(SS * SS);
            float* o = &img[(j * W + i) * 4];
            if (cov > 0) { o[0] = acc.x / cov; o[1] = acc.y / cov; o[2] = acc.z / cov; }
            o[3] = cov / n;
        }
    }
    FILE* f = fopen(out, "wb");
    fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", W, H);
    std::vector<uint8_t> px(W * H * 4);
    for (int k = 0; k < W * H; k++) {
        for (int c = 0; c < 3; c++) {
            float v = img[k * 4 + c];
            v = v / (1.0f + v * 0.15f);                        // a gentle shoulder for the hottest glints
            px[k * 4 + c] = (uint8_t)std::lround(255.0f * std::pow(clampf(v, 0, 1), 1.0f / 2.2f));
        }
        px[k * 4 + 3] = (uint8_t)std::lround(255.0f * img[k * 4 + 3]);
    }
    fwrite(px.data(), 1, px.size(), f); fclose(f);
    return 0;
}
