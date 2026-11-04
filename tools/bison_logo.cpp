// The intro's bison: a wild bison charging three-quarters toward the viewer, as a muscular black-chrome sculpture.
//
// Model: a signed-distance field in metres (body frame: x forward, y up, z the bison's left; ground at y = 0) built
// from ellipsoids and round cones blended smoothly: a deep chest, the high hump over the forelegs, the barrel and
// leaner hindquarters, with muscle masses laid over them (shoulder, chest, neck, flank, haunch, hip) and on the
// legs (triceps, forearm, thigh, gaskin), which meet in creases. The pose is a charge: the near foreleg reaching,
// the far one folded, the near hind leg driving back and the far one coming forward, the whole animal leaning in
// with the head down and turned toward the viewer. Hair is sculpted, not furred: wool over the head down to a bare
// nose, long locks falling from the hump over the cape, beard and forelegs.
// Render: sphere tracing, normals from the field, ambient occlusion, a soft shadow from the key, shaded as black
// chrome: almost no colour of its own, only reflections of a dark studio (a big soft box up left with a hard key in
// it, a strip overhead, two tall strips behind for the silhouette, a dim fill, and the off-white sign above).
//
// Build: g++ -O2 -fopenmp tools/bison_logo.cpp -o bison_logo   (tools/bison_logo.py builds and runs it)
// Usage: bison_logo OUT.pam [SIZE SUPERSAMPLE]      writes RGBA (PAM), straight alpha
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
static inline V operator/(V a, V b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
static inline float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline float len(V a) { return std::sqrt(dot(a, a)); }
static inline V norm(V a) { float l = len(a); return a * (1.0f / (l > 1e-9f ? l : 1e-9f)); }
static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
static inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
static inline float smoothstep(float a, float b, float x) { float t = clampf((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
static inline V lerp(V a, V b, float t) { return a + (b - a) * t; }

// ------------------------------------------------------------------------------------------- primitives
static inline float sdEllipsoid(V q, V r) {               // q relative to the centre, axis-aligned
    V a = q / r; float k0 = len(a), k1 = len(a / r);
    return k0 * (k0 - 1.0f) / std::max(k1, 1e-6f);
}
static inline float sdEll(V p, V c, V r) { return sdEllipsoid(p - c, r); }
// an ellipsoid along a direction: ra along the axis, rl to the side (lateral, z when the axis is upright), rd the third
static float sdEllA(V p, V c, V axis, float ra, float rl, float rd) {
    V a = norm(axis);
    V up = std::fabs(a.y) < 0.97f ? v3(0, 1, 0) : v3(1, 0, 0);
    V l = norm(cross(up, a)), d = cross(a, l);
    V q = p - c;
    return sdEllipsoid(v3(dot(q, a), dot(q, l), dot(q, d)), v3(ra, rl, rd));
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
static inline float sdSphere(V p, V c, float r) { return len(p - c) - r; }
static inline float smin(float a, float b, float k) {
    float h = clampf(0.5f + 0.5f * (b - a) / k, 0, 1); return mixf(b, a, h) - k * h * (1 - h);
}
static inline float smax(float a, float b, float k) { return -smin(-a, -b, k); }
static float chain(V p, const V* pts, const float* r, int n, float k) {
    float d = 1e9f;
    for (int i = 0; i + 1 < n; i++) d = smin(d, sdRoundCone(p, pts[i], pts[i + 1], r[i], r[i + 1]), k);
    return d;
}
static inline V side(V p, float s) { return v3(p.x, p.y, p.z * s); }     // mirror to the other side

// ------------------------------------------------------------------------------------------- noise
static inline uint32_t hash3(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)z * 83492791u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16; return h;
}
static inline float h01(int x, int y, int z) { return (hash3(x, y, z) & 0xFFFFFF) * (1.0f / 16777215.0f); }
static float vnoise(V p) {
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
// ------------------------------------------------------------------------------------------- the pose
// joints of the four legs: shoulder/hip, elbow/stifle, knee/hock, fetlock, hoof (z < 0 is the bison's right side)
static const V FR[5] = {{0.58f, 1.00f, -0.26f}, {0.74f, 0.64f, -0.27f}, {1.02f, 0.38f, -0.28f}, {1.22f, 0.16f, -0.28f}, {1.30f, 0.065f, -0.28f}};  // right fore, reaching
static const V FL[5] = {{0.58f, 1.00f, 0.26f}, {0.58f, 0.64f, 0.27f}, {0.80f, 0.42f, 0.28f}, {0.58f, 0.32f, 0.28f}, {0.46f, 0.31f, 0.28f}};       // left fore, folded
static const V HR[5] = {{-0.80f, 1.06f, -0.24f}, {-0.64f, 0.74f, -0.27f}, {-1.06f, 0.50f, -0.26f}, {-1.36f, 0.20f, -0.26f}, {-1.46f, 0.11f, -0.26f}};  // right hind, driving
static const V HL[5] = {{-0.80f, 1.06f, 0.24f}, {-0.52f, 0.72f, 0.27f}, {-0.76f, 0.44f, 0.27f}, {-0.52f, 0.17f, 0.27f}, {-0.42f, 0.08f, 0.27f}};    // left hind, forward
static const float FRAD[5] = {0.23f, 0.175f, 0.122f, 0.106f, 0.104f};
static const float HRAD[5] = {0.24f, 0.18f, 0.118f, 0.100f, 0.100f};

enum Mat { BODY, HAIR, HORN, HOOF, EYE, NMAT };
struct Hit { float d; int m; };

static float foreleg(V p, const V* J, float s) {
    float d = chain(p, J, FRAD, 4, 0.03f);
    V u = J[1] - J[0], f = J[2] - J[1], fd = norm(f);
    V fwd = norm(cross(v3(0, 0, 1), fd)) * -1.0f;                       // the front of the leg, in its plane
    d = smin(d, sdEllA(p, lerp(J[0], J[1], 0.50f) + v3(-0.07f, 0, 0.02f * s), u, len(u) * 0.62f, 0.15f, 0.21f), 0.03f);   // triceps
    d = smin(d, sdEllA(p, lerp(J[1], J[2], 0.30f) + fwd * 0.035f + v3(0, 0, 0.01f * s), f, len(f) * 0.44f, 0.135f, 0.115f), 0.025f);   // extensor
    d = smin(d, sdEllA(p, lerp(J[1], J[2], 0.28f) - fwd * 0.04f, f, len(f) * 0.42f, 0.125f, 0.10f), 0.025f);                 // flexor
    d = smin(d, sdSphere(p, J[2], 0.122f), 0.02f);                                                                           // knee
    d = smin(d, sdSphere(p, J[3], 0.108f), 0.02f);                                                                           // fetlock
    return d;
}
static float hindleg(V p, const V* J, float s) {
    float d = chain(p, J, HRAD, 4, 0.03f);
    V t = J[1] - J[0], g = J[2] - J[1];
    d = smin(d, sdEllA(p, lerp(J[0], J[1], 0.40f) + v3(-0.08f, 0, 0.04f * s), t, len(t) * 0.80f, 0.18f, 0.32f), 0.03f);  // biceps femoris
    d = smin(d, sdEllA(p, lerp(J[0], J[1], 0.58f) + v3(0.11f, 0, 0.03f * s), t, len(t) * 0.55f, 0.155f, 0.18f), 0.025f); // quadriceps
    d = smin(d, sdEllA(p, lerp(J[1], J[2], 0.32f) + v3(0, 0, 0.01f * s), g, len(g) * 0.50f, 0.14f, 0.16f), 0.02f);       // gaskin
    d = smin(d, sdRoundCone(p, lerp(J[1], J[2], 0.25f) + v3(-0.08f, 0, 0), J[2] + v3(-0.06f, 0, 0), 0.05f, 0.03f), 0.03f);   // the tendon to the hock
    d = smin(d, sdSphere(p, J[2], 0.118f), 0.02f);                                                                         // hock
    d = smin(d, sdSphere(p, J[3], 0.104f), 0.02f);
    return d;
}
static float hoof(V p, const V* J) {
    V dir = norm(J[4] - J[3]);
    float d = sdEllA(p, J[4], dir, 0.098f, 0.104f, 0.108f);
    return smax(d, -sdEllA(p, J[4] + dir * 0.06f, dir, 0.08f, 0.012f, 0.2f), 0.006f);     // the cleft between the toes
}

static float HEAD_TURN = 0.30f;                    // radians, toward the bison's right (the viewer)
static V turnHead(V p) {                             // the head's own frame: undo the turn about the poll
    V piv = v3(1.05f, 1.05f, 0);
    float w = smoothstep(0.85f, 1.15f, p.x);            // blend: the neck twists, the head turns fully
    float a = HEAD_TURN * w, c = std::cos(a), sn = std::sin(a);
    V d = p - piv;
    return piv + v3(c * d.x - sn * d.z, d.y, sn * d.x + c * d.z);
}
static Hit scene(V p0, bool detail = true) {
    V p = p0;
    V ph = turnHead(p0);                             // head parts use ph
    V qh = side(ph, ph.z < 0 ? -1.0f : 1.0f);
    V q = side(p, p.z < 0 ? -1.0f : 1.0f);          // q: folded onto the left side, for symmetric parts
    // the trunk: a deep chest, the high hump over the forelegs, the barrel, smaller hindquarters
    float body = sdEll(p, v3(-0.08f, 1.09f, 0), v3(0.66f, 0.48f, 0.44f));                         // barrel
    body = smin(body, sdEll(p, v3(0.64f, 0.98f, 0), v3(0.46f, 0.40f, 0.41f)), 0.12f);             // chest
    body = smin(body, sdEll(p, v3(0.30f, 1.50f, 0), v3(0.62f, 0.46f, 0.37f)), 0.16f);             // hump
    body = smin(body, sdEll(p, v3(0.30f, 1.84f, 0), v3(0.36f, 0.19f, 0.23f)), 0.14f);             // its crest
    body = smin(body, sdEll(p, v3(-0.80f, 1.14f, 0), v3(0.36f, 0.31f, 0.34f)), 0.16f);            // hindquarters
    body = smin(body, sdRoundCone(p, v3(0.84f, 1.33f, 0), v3(1.16f, 1.02f, 0), 0.42f, 0.33f), 0.10f);   // neck
    // muscles, both sides
    float mus = sdEllA(q, v3(0.47f, 1.30f, 0.30f), v3(0.30f, -1, 0), 0.40f, 0.12f, 0.28f);        // over the shoulder blade
    mus = std::min(mus, sdEllA(q, v3(0.72f, 1.04f, 0.31f), v3(0.5f, -1, 0), 0.24f, 0.12f, 0.17f));   // point of the shoulder
    mus = std::min(mus, sdEllA(q, v3(0.90f, 0.84f, 0.19f), v3(0.25f, -1, 0), 0.26f, 0.13f, 0.15f)); // chest (pectoral)
    mus = std::min(mus, sdEllA(q, v3(0.97f, 1.13f, 0.23f), v3(0.6f, -0.55f, 0), 0.36f, 0.13f, 0.18f));   // neck muscle
    mus = std::min(mus, sdEllA(q, v3(-0.40f, 1.02f, 0.33f), v3(1, 0.12f, 0), 0.34f, 0.10f, 0.24f));  // flank
    mus = std::min(mus, sdEllA(q, v3(-0.86f, 1.24f, 0.26f), v3(0.25f, -1, 0), 0.30f, 0.13f, 0.28f)); // haunch
    mus = std::min(mus, sdEllA(q, v3(-0.74f, 1.36f, 0.17f), v3(1, 0.25f, 0), 0.30f, 0.12f, 0.12f));  // gluteal, over the hip
    mus = std::min(mus, sdEll(q, v3(-0.62f, 1.33f, 0.27f), v3(0.09f, 0.08f, 0.08f)));                // the point of the hip
    mus = std::min(mus, sdEll(q, v3(-1.08f, 1.36f, 0.11f), v3(0.08f, 0.07f, 0.07f)));                // a pin bone
    body = smin(body, mus, 0.022f);
    // the legs
    body = smin(body, foreleg(p, FR, -1), 0.05f);
    body = smin(body, foreleg(p, FL, 1), 0.05f);
    body = smin(body, hindleg(p, HR, -1), 0.05f);
    body = smin(body, hindleg(p, HL, 1), 0.05f);
    // the head: big, lowered, forehead toward the charge
    float head = sdEll(ph, v3(1.27f, 0.90f, 0), v3(0.34f, 0.36f, 0.33f));                       // skull
    head = smin(head, sdEll(ph, v3(1.48f, 0.62f, 0), v3(0.16f, 0.16f, 0.195f)), 0.07f);         // a short, broad muzzle
    head = smin(head, sdEll(qh, v3(1.35f, 0.72f, 0.18f), v3(0.18f, 0.18f, 0.13f)), 0.04f);        // jowls
    head = smin(head, sdEll(qh, v3(1.49f, 0.92f, 0.225f), v3(0.08f, 0.036f, 0.07f)), 0.025f);    // brow over the eye
    float nose = sdEll(ph, v3(1.605f, 0.585f, 0), v3(0.055f, 0.085f, 0.135f));                   // the bare, wet nose pad
    body = smin(body, nose, 0.02f);
    // hair, sculpted: the mop, the beard, the long hair behind the forearms, the tail
    float hair = sdEll(ph, v3(1.33f, 1.07f, 0), v3(0.30f, 0.31f, 0.39f));                         // mop
    float cape = sdEll(p, v3(0.62f, 1.22f, 0), v3(0.52f, 0.62f, 0.47f));                           // the cape over the shoulders
    cape = smin(cape, sdRoundCone(p, v3(0.90f, 1.30f, 0), v3(1.16f, 0.98f, 0), 0.46f, 0.38f), 0.12f);   // and the neck
    cape = smin(cape, sdEll(p, v3(0.95f, 0.78f, 0), v3(0.30f, 0.36f, 0.33f)), 0.10f);              // the throat and chest
    hair = smin(hair, cape, 0.10f);
    hair = smin(hair, head, 0.08f);                                                                   // the head, all wool
    hair = smin(hair, sdRoundCone(ph, v3(1.48f, 0.47f, 0), v3(1.30f, 0.08f, 0), 0.17f, 0.04f), 0.05f);  // beard
    hair = smin(hair, sdEllA(p, v3(0.60f, 0.62f, 0.28f), v3(0.15f, -1, 0), 0.30f, 0.15f, 0.19f), 0.06f);   // chaps, left
    hair = smin(hair, sdEllA(p, v3(0.80f, 0.60f, -0.29f), v3(0.45f, -1, 0), 0.30f, 0.15f, 0.19f), 0.06f);  // chaps, right
    float tail = sdRoundCone(p, v3(-1.16f, 1.32f, 0), v3(-1.36f, 1.42f, 0), 0.055f, 0.045f);
    tail = std::min(tail, sdEll(p, v3(-1.44f, 1.40f, 0), v3(0.11f, 0.065f, 0.065f)));
    hair = std::min(hair, tail);
    // horns: out of the sides of the mop, out, up and in
    static const V HN[6] = {{1.27f, 1.10f, 0.28f}, {1.27f, 1.12f, 0.45f}, {1.30f, 1.23f, 0.56f}, {1.35f, 1.38f, 0.57f}, {1.41f, 1.51f, 0.51f}, {1.46f, 1.57f, 0.44f}};
    static const float HNR[6] = {0.105f, 0.092f, 0.072f, 0.050f, 0.025f, 0.004f};
    float horn = chain(qh, HN, HNR, 6, 0.02f);
    float hooves = std::min(std::min(hoof(p, FR), hoof(p, FL)), std::min(hoof(p, HR), hoof(p, HL)));
    V EYEC = v3(1.50f, 0.86f, 0.235f);
    float eye = sdSphere(qh, EYEC, 0.036f);

    if (detail && hair < 0.06f) {
        float w_mop = smoothstep(1.02f, 1.16f, ph.x)                                                   // the head,
                    * smoothstep(0.015f, 0.07f, sdEll(ph, v3(1.605f, 0.585f, 0), v3(0.055f, 0.085f, 0.135f)));   // not the nose
        w_mop *= 1.0f - smoothstep(0.10f, 0.02f, sdRoundCone(ph, v3(1.48f, 0.47f, 0), v3(1.30f, 0.08f, 0), 0.17f, 0.04f));   // not the beard
        float wool = fbm(ph * 12.0f, 3);
        wool = 1.0f - std::fabs(2.0f * wool - 1.0f);                                   // ridged: tight woolly ridges
        float th = std::atan2(p.z, p.x - 0.62f) + 0.9f * (1.85f - p.y);                // around the hump, twisting as it falls
        float lk = 0.5f + 0.5f * std::sin(th * 13.0f + 1.5f * fbm(p * 2.5f, 2));
        float lock = std::pow(lk, 0.6f);
        float w_beard = smoothstep(0.12f, 0.04f, sdRoundCone(ph, v3(1.48f, 0.47f, 0), v3(1.30f, 0.08f, 0), 0.17f, 0.04f));
        float bl = std::pow(0.5f + 0.5f * std::sin(std::atan2(ph.z, ph.x - 1.40f) * 9.0f + ph.y * 4.0f), 0.7f);
        lock = mixf(lock, bl, w_beard);
        hair -= w_mop * 0.026f * wool + (1 - w_mop) * 0.030f * lock;
    }
    float d = smin(body, hair, 0.04f);
    d = smin(d, horn, 0.02f);
    d = std::min(d, hooves);
    d = smax(d, -sdEll(qh, v3(1.655f, 0.575f, 0.072f), v3(0.030f, 0.036f, 0.030f)), 0.008f);      // nostrils
    d = smax(d, -sdSphere(qh, EYEC + v3(0, 0.003f, 0.012f), 0.046f), 0.008f);                      // eye socket
    d = std::min(d, eye);
    int m = BODY;
    float best = body;
    if (hair < best) { best = hair; m = HAIR; }
    if (horn < best) { best = horn; m = HORN; }
    if (hooves < best) { best = hooves; m = HOOF; }
    if (eye < best + 0.002f) m = EYE;
    return {d, m};
}

// ------------------------------------------------------------------------------------------- render
struct Cam { V ro, cu, cv, cw; float fl; V yawF, yawL; };
static Cam CAM;
static float PITCH = -0.07f;                         // radians; negative tips the head down
static inline V toBody(V w) {
    V b = v3(dot(w, CAM.yawF), w.y, dot(w, CAM.yawL));
    V piv = v3(0, 0.9f, 0), d = b - piv;
    float c = std::cos(-PITCH), s = std::sin(-PITCH);
    return piv + v3(c * d.x - s * d.y, s * d.x + c * d.y, d.z);
}
static Hit sceneW(V w, bool detail = true) { return scene(toBody(w), detail); }

static V normalAt(V w) {
    const float h = 0.0015f;
    V k1 = v3(1, -1, -1), k2 = v3(-1, -1, 1), k3 = v3(-1, 1, -1), k4 = v3(1, 1, 1);
    return norm(k1 * sceneW(w + k1 * h).d + k2 * sceneW(w + k2 * h).d + k3 * sceneW(w + k3 * h).d + k4 * sceneW(w + k4 * h).d);
}
static float ao(V p, V n) {
    float occ = 0, sca = 1;
    for (int i = 1; i <= 6; i++) {
        float h = 0.01f + 0.035f * i;
        occ += (h - sceneW(p + n * h, false).d) * sca; sca *= 0.8f;
    }
    return clampf(1.0f - 2.4f * occ, 0.0f, 1.0f);
}
static float softShadow(V ro, V rd, float k) {
    float res = 1, t = 0.02f;
    for (int i = 0; i < 64 && t < 4.0f; i++) {
        float h = sceneW(ro + rd * t, false).d;
        res = std::min(res, k * h / t);
        if (res < 0.002f) return 0;
        t += clampf(h, 0.01f, 0.2f);
    }
    return clampf(res, 0, 1);
}
static inline float lobe(V r, V d, float w) { float c = (dot(r, norm(d)) - (1 - w)) / w; c = clampf(c, 0, 1); return c * c; }
// the studio, in camera space: R right, U up, B back toward the camera
static V studio(V r, float spread) {
    float U = r.y, R = r.x, B = r.z, w = spread;
    float white = 0.003f + 0.03f * smoothstep(-0.30f, 0.90f, U) * smoothstep(-0.9f, 0.3f, B);        // a dark room
    // a big softbox up and to the left: a soft gradient that models the big forms
    float big = smoothstep(0.55f - 0.10f * w, 0.95f, dot(r, norm(v3(-0.55f, 0.65f, 0.52f))));
    white += 1.0f * big * big;
    // a small hard key inside it, and a strip overhead: crisp lines along the tops of muscles
    white += 2.2f * lobe(r, v3(-0.62f, 0.45f, 0.64f), 0.05f * w);
    float band1 = smoothstep(0.86f - 0.05f * w, 0.88f, U) * (1 - smoothstep(0.94f, 0.96f + 0.03f * w, U)) * smoothstep(-0.2f, 0.3f, B);
    white += 1.0f * band1;
    // two tall strips behind for the silhouette
    float rim = smoothstep(0.84f - 0.05f * w, 0.88f, std::fabs(R)) * smoothstep(-0.1f, 0.35f, -B) * smoothstep(-0.3f, 0.1f, U);
    white += 2.6f * rim;
    white += 0.15f * lobe(r, v3(0.60f, -0.05f, 0.80f), 0.20f * w);                                  // a dim fill, low right
    // the sign above and in front (BRUBUS in off-white): a thin warm strip
    float sign = smoothstep(0.48f - 0.04f * w, 0.50f, U) * (1 - smoothstep(0.55f, 0.57f + 0.03f * w, U)) * smoothstep(0.4f, 0.75f, B)
               * (1 - smoothstep(0.55f, 0.75f, std::fabs(R)));
    return v3(0.84f, 0.87f, 0.95f) * white + v3(1.0f, 0.93f, 0.8f) * (0.9f * sign);
}
static V camSpace(V w) { return v3(dot(w, CAM.cu), dot(w, CAM.cv), -dot(w, CAM.cw)); }
static V worldFromCam(V c) { return CAM.cu * c.x + CAM.cv * c.y - CAM.cw * c.z; }

static V shade(V p, V rd, int m) {
    V n = normalAt(p), v = rd * -1.0f;
    float nv = clampf(dot(n, v), 0, 1);
    V r = rd - n * (2.0f * dot(rd, n));
    float occ = ao(p, n);
    float rough = m == HAIR ? 2.0f : (m == HORN || m == HOOF || m == EYE ? 0.7f : 1.0f);
    float f0 = m == HAIR ? 0.34f : (m == EYE ? 0.55f : 0.50f);
    float fres = f0 + (1 - f0) * std::pow(1 - nv, 5.0f);
    V col = studio(camSpace(r), rough) * fres;
    // the key light also shadows: dim the reflection of the key where the body blocks it
    V keyW = norm(worldFromCam(v3(-0.62f, 0.45f, 0.64f)));   // (the key's shadow)
    float sh = softShadow(p + n * 0.01f, keyW, 12.0f);
    col = col * (0.55f + 0.45f * sh);
    col = col * (0.18f + 0.82f * occ);
    return col;
}

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "bison.pam";
    int W = argc > 2 ? atoi(argv[2]) : 1400, H = W, SS = argc > 3 ? atoi(argv[3]) : 2;
    const float yaw = 42.0f, camY = 1.2f, dist = 11.0f, fov = 20.0f;      // a long lens, a little below the hump
    // the bison faces the viewer's left, turned `yaw` degrees toward the camera; its left side (z) is away from us
    CAM.yawF = norm(v3(-std::cos(yaw * 3.14159265f / 180.0f), 0, std::sin(yaw * 3.14159265f / 180.0f)));
    CAM.yawL = cross(CAM.yawF, v3(0, 1, 0));
    V target = v3(0, 0.95f, 0);
    CAM.ro = v3(0, camY, dist);
    CAM.cw = norm(target - CAM.ro); CAM.cu = norm(cross(CAM.cw, v3(0, 1, 0))); CAM.cv = cross(CAM.cu, CAM.cw);
    CAM.fl = 1.0f / std::tan(fov * 0.5f * 3.14159265f / 180.0f);
    V bc = v3(0, 0.95f, 0); float br = 2.3f;
    std::vector<float> img(W * H * 4, 0.0f);
#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) {
            V acc = v3(0, 0, 0); float cov = 0;
            for (int sy = 0; sy < SS; sy++) for (int sx = 0; sx < SS; sx++) {
                float px = (2.0f * (i + (sx + 0.5f) / SS) - W) / H, py = -(2.0f * (j + (sy + 0.5f) / SS) - H) / H;
                V rd = norm(CAM.cu * px + CAM.cv * py + CAM.cw * CAM.fl);
                V oc = CAM.ro - bc; float b = dot(oc, rd), c = dot(oc, oc) - br * br, disc = b * b - c;
                if (disc < 0) continue;
                float t = std::max(0.0f, -b - std::sqrt(disc)), tmax = -b + std::sqrt(disc);
                bool hit = false; int mat = 0;
                for (int s = 0; s < 300 && t < tmax; s++) {
                    Hit h = sceneW(CAM.ro + rd * t);
                    if (h.d < 0.0003f * t) { hit = true; mat = h.m; break; }
                    t += h.d * 0.8f;
                }
                if (!hit) continue;
                acc = acc + shade(CAM.ro + rd * t, rd, mat); cov += 1;
            }
            float* o = &img[(j * W + i) * 4];
            if (cov > 0) { o[0] = acc.x / cov; o[1] = acc.y / cov; o[2] = acc.z / cov; }
            o[3] = cov / (float)(SS * SS);
        }
    }
    FILE* f = fopen(out, "wb");
    fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", W, H);
    std::vector<uint8_t> px(W * H * 4);
    for (int k = 0; k < W * H; k++) {
        for (int c = 0; c < 3; c++) {
            float v = img[k * 4 + c]; v = v / (1.0f + v * 0.12f);
            px[k * 4 + c] = (uint8_t)std::lround(255.0f * std::pow(clampf(v, 0, 1), 1.0f / 2.2f));
        }
        px[k * 4 + 3] = (uint8_t)std::lround(255.0f * img[k * 4 + 3]);
    }
    fwrite(px.data(), 1, px.size(), f); fclose(f);
    return 0;
}
