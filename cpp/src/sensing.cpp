#include "robotnav/sensing.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

#include "robotnav/planners.hpp"
#include "robotnav/thread_pool.hpp"

namespace robotnav {

namespace {

const double kSnr50 = std::log(SENSE.pfa) / std::log(0.5) - 1.0;

// Is the line from a tower's eye (over cell tr, tc, at height E) to height a over cell (r, c) clear of the ground and
// the trees? One sample per cell crossed, as make_airspace takes them.
bool clear_line(const Airspace& air, int tr, int tc, double E, int r, int c, double a) {
    const int dr = r - tr, dc = c - tc, n = std::max(std::abs(dr), std::abs(dc));
    for (int j = 1; j < n; ++j) {
        const double f = static_cast<double>(j) / n;
        const double pr = tr + f * (r - tr), pc = tc + f * (c - tc), line = E + f * (a - E);
        const int sr = static_cast<int>(std::floor(pr + 0.5)), sc = static_cast<int>(std::floor(pc + 0.5));
        if (air.top[sr * air.C + sc] > line) return false;
    }
    return true;
}

// Tower n's chance, per look, of detecting a mover at height a over cell (r, c).
double pd_from(const Airspace& air, int n, int r, int c, double a) {
    const int tr = air.towers[n].first, tc = air.towers[n].second, dr = r - tr, dc = c - tc;
    const double E = air.eyes[n], dy = a - E, d2 = (dr * dr + dc * dc) + dy * dy;
    if (!(dr || dc) || d2 > AIR.sight * AIR.sight || !clear_line(air, tr, tc, E, r, c, a)) return 0.0;
    return pd_at(std::sqrt(d2));
}

void sum_up(const Field& f, RiskRoute& rt) {         // time, hazard and risk from the steps
    const std::vector<float>& haz = rt.air ? f.air_hazard : f.ground_hazard;
    double time = 0.0, sum = 0.0;
    for (std::size_t k = 1; k < rt.nodes.size(); ++k) {
        time += rt.step[k];
        sum += rt.step[k] * haz[rt.nodes[k]];
    }
    rt.time = time;
    rt.hazard = sum;
    rt.risk = 1.0 - std::exp(-sum);
}

// The terrain with hazard priced in: a move costs its drive time, plus w seconds per unit of hazard where it ends.
struct RiskGraph {
    const Map& m;
    const std::vector<float>& haz;
    double w;
    int avoid;
    template <class F>
    void neighbours(int u, F&& f) const {
        m.neighbours(u, [&](int v, double c) {
            if (v == avoid) return;
            f(v, (w != 0.0 && haz[v] != 0.0f) ? c + w * c * haz[v] : c);
        });
    }
    double heuristic(int a, int b) const { return m.heuristic(a, b); }
    double path_cost(const std::vector<int>& p) const { return m.path_cost(p); }
};

// mulberry32, as the viewer's: a small generator with a 32-bit state. next() is uniform in [0, 1).
struct Rng {
    std::uint32_t a;
    double next() {
        a += 0x6D2B79F5u;
        std::uint32_t t = (a ^ (a >> 15)) * (1u | a);
        t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
        return static_cast<double>(t ^ (t >> 14)) / 4294967296.0;
    }
};

// What a search needs to remember, kept by each thread from one search to the next: nothing is allocated or cleared
// per search. A cell's numbers count only if its stamp is this search's.
struct Workspace {
    std::vector<double> g;
    std::vector<int> parent;
    std::vector<std::uint32_t> seen, closed;
    std::vector<detail::Entry> heap;
    std::uint32_t stamp = 0;
    void begin(std::size_t n) {
        if (g.size() != n || ++stamp == 0) {         // a new map size, or the stamps have wrapped round
            g.assign(n, 0.0);
            parent.assign(n, -1);
            seen.assign(n, 0);
            closed.assign(n, 0);
            stamp = 1;
        }
        heap.clear();
    }
};

void add(McResult& to, const McResult& x) {
    to.n += x.n; to.block += x.block; to.stuck += x.stuck; to.detected += x.detected; to.tracked += x.tracked;
    to.late += x.late; to.clean += x.clean; to.clean_detected += x.clean_detected; to.ok += x.ok;
    to.time_sum += x.time_sum;
}

}  // namespace

double pd_at(double r) {
    if (r > AIR.sight) return 0.0;
    const double snr = kSnr50 * std::pow(SENSE.r50 / std::max(r, 0.5), 4);
    return std::min(0.999, std::pow(SENSE.pfa, 1.0 / (1.0 + snr)));
}

double hazard_of(double pd) { return -std::log(1.0 - pd) / SENSE.look; }

std::pair<double, double> wilson(long k, long n) {
    const double z = 1.959964, q = static_cast<double>(k) / n, den = 1.0 + z * z / n;
    const double mid = (q + z * z / (2.0 * n)) / den;
    const double half = z * std::sqrt(q * (1.0 - q) / n + z * z / (4.0 * n * n)) / den;
    return {std::max(0.0, mid - half), std::min(1.0, mid + half)};
}

Field make_field(const Map& m, const Airspace& air) {
    Field f;
    f.T = static_cast<int>(air.towers.size());
    f.N = m.size();
    f.L = air.L;
    const int T = f.T, N = f.N, L = f.L, C = m.cols;
    f.ground_pd.assign(static_cast<std::size_t>(N) * T, 0.0f);
    f.ground_hazard.assign(static_cast<std::size_t>(N), 0.0f);
    f.air_pd.assign(static_cast<std::size_t>(L) * N * T, 0.0f);
    f.air_hazard.assign(static_cast<std::size_t>(L) * N, 0.0f);
    for (int n = 0; n < T; ++n)
        for (int i = 0; i < N; ++i) {
            const int r = i / C, c = i % C;
            if (!m.blocked(i)) {
                const double pd = pd_from(air, n, r, c, m.height[i] + 1.0);
                if (pd > 0.0) {
                    f.ground_pd[static_cast<std::size_t>(i) * T + n] = static_cast<float>(pd);
                    f.ground_hazard[i] = static_cast<float>(f.ground_hazard[i] + hazard_of(pd));
                }
            }
            for (int k = 0; k < L; ++k) {
                const std::size_t v = static_cast<std::size_t>(k) * N + i;
                if (!air.seen[v]) continue;
                const double pd = pd_from(air, n, r, c, air.alt[k]);
                if (pd > 0.0) {
                    f.air_pd[v * T + n] = static_cast<float>(pd);
                    f.air_hazard[v] = static_cast<float>(f.air_hazard[v] + hazard_of(pd));
                }
            }
        }
    f.edge.assign(static_cast<std::size_t>(N) * 8, -1.0);        // each cell's legal moves and their times, in the planners' order
    for (int a = 0; a < N; ++a) {
        const int r = a / C, c = a % C;
        for (int k = 0; k < 8; ++k) {
            const int nr = r + NB_R[k], nc = c + NB_C[k];
            if (!m.inside(nr, nc)) continue;
            const int b = nr * C + nc;
            if (m.blocked(b)) continue;
            if (NB_R[k] && NB_C[k] && (m.blocked((r + NB_R[k]) * C + c) || m.blocked(r * C + c + NB_C[k]))) continue;
            f.edge[static_cast<std::size_t>(a) * 8 + k] = m.edge_cost(a, b);
        }
    }
    return f;
}

RiskRoute risk_drive_reference(const Map& m, const Field& f, int start, int goal, double w, int avoid) {
    const RiskGraph g{m, f.ground_hazard, w, avoid};
    const SearchResult res = best_first(g, m.size(), start, goal, "risk", [](double gg, double h) { return gg + h; });
    RiskRoute rt;
    rt.searched = res.nodes_expanded;
    rt.found = res.found();
    if (!rt.found) return rt;
    rt.nodes = res.path;
    rt.step.assign(1, 0.0);
    for (std::size_t k = 1; k < rt.nodes.size(); ++k) rt.step.push_back(m.edge_cost(rt.nodes[k - 1], rt.nodes[k]));
    sum_up(f, rt);
    return rt;
}

RiskRoute risk_drive(const Map& m, const Field& f, int start, int goal, double w, int avoid) {
    // A* as best_first runs it (the same order off the frontier, the same sums), with each move's time read from the
    // field's table instead of worked out again.
#ifdef ROBOTNAV_NO_THREADS
    static Workspace ws;
#else
    thread_local Workspace ws;                       // one per worker thread
#endif
    ws.begin(static_cast<std::size_t>(m.size()));
    const int C = m.cols, gr = goal / C, gc = goal % C;
    const std::vector<float>& haz = f.ground_hazard;
    const double* edge = f.edge.data();
    auto h = [&](int v) {
        const int dr = std::abs(v / C - gr), dc = std::abs(v % C - gc);
        return ((dr + dc) + (kSqrt2 - 2.0) * std::min(dr, dc)) * m.cell;
    };
    const detail::Later later;
    long tie = 0, searched = 0;
    const std::uint32_t now = ws.stamp;
    ws.g[start] = 0.0;
    ws.seen[start] = now;
    ws.parent[start] = -1;
    ws.heap.push_back({h(start), tie++, start});
    bool found = false;
    while (!ws.heap.empty()) {
        std::pop_heap(ws.heap.begin(), ws.heap.end(), later);
        const int cur = ws.heap.back().v;
        ws.heap.pop_back();
        if (ws.closed[cur] == now) continue;
        ws.closed[cur] = now;
        ++searched;
        if (cur == goal) {
            found = true;
            break;
        }
        const double* e = edge + static_cast<std::size_t>(cur) * 8;
        const double g_cur = ws.g[cur];
        for (int k = 0; k < 8; ++k) {
            const double c = e[k];
            if (c < 0.0) continue;
            const int nxt = cur + NB_R[k] * C + NB_C[k];
            if (nxt == avoid) continue;
            const double step = (w != 0.0 && haz[nxt] != 0.0f) ? c + w * c * haz[nxt] : c;
            const double new_g = g_cur + step;
            if (ws.seen[nxt] != now || new_g < ws.g[nxt]) {
                ws.g[nxt] = new_g;
                ws.seen[nxt] = now;
                ws.parent[nxt] = cur;
                ws.heap.push_back({new_g + h(nxt), tie++, nxt});
                std::push_heap(ws.heap.begin(), ws.heap.end(), later);
            }
        }
    }
    RiskRoute rt;
    rt.searched = searched;
    rt.found = found;
    if (!found) return rt;
    for (int v = goal; v != -1; v = ws.parent[v]) rt.nodes.push_back(v);
    std::reverse(rt.nodes.begin(), rt.nodes.end());
    rt.step.assign(1, 0.0);
    for (std::size_t k = 1; k < rt.nodes.size(); ++k) rt.step.push_back(m.edge_cost(rt.nodes[k - 1], rt.nodes[k]));
    sum_up(f, rt);
    return rt;
}

RiskRoute risk_flight(const Airspace& air, const Field& f, int start, int goal, double w) {
    const Flight fl = fly(air, start, goal, w, f.air_hazard);
    RiskRoute rt;
    rt.air = true;
    rt.searched = fl.nodes;
    rt.found = fl.found;
    if (!rt.found) return rt;
    rt.nodes = fl.voxels;
    const int C = air.C, RC = air.R * air.C;
    rt.step.assign(1, 0.0);
    for (std::size_t i = 1; i < rt.nodes.size(); ++i) {
        const int a = rt.nodes[i - 1], b = rt.nodes[i];
        rt.step.push_back(move_time((b % RC) / C - (a % RC) / C, b % C - a % C, b / RC - a / RC));
    }
    sum_up(f, rt);
    return rt;
}

RiskRoute route_from(const Map& m, const Airspace& air, const Field& f, bool is_air, std::vector<int> nodes) {
    RiskRoute rt;
    rt.air = is_air;
    rt.found = !nodes.empty();
    rt.nodes = std::move(nodes);
    const int C = air.C, RC = air.R * air.C;
    rt.step.assign(rt.found ? 1 : 0, 0.0);
    for (std::size_t i = 1; i < rt.nodes.size(); ++i) {
        const int a = rt.nodes[i - 1], b = rt.nodes[i];
        rt.step.push_back(is_air ? move_time((b % RC) / C - (a % RC) / C, b % C - a % C, b / RC - a / RC) : m.edge_cost(a, b));
    }
    if (rt.found) sum_up(f, rt);
    return rt;
}

Trial run_trial(const Map& m, const Field& f, const RiskRoute& rt, double w, std::uint32_t seed, long number) {
    Rng rng{seed + static_cast<std::uint32_t>(number + 1) * 0x9E3779B9u};
    const int T = f.T, C = m.cols, len = static_cast<int>(rt.nodes.size());
    struct Leg {
        const RiskRoute* rt;
        int to;
    };
    Leg legs[2] = {{&rt, len - 1}, {nullptr, 0}};
    int n_legs = 1;
    RiskRoute next;
    Trial out;
    if (!rt.air && len > 14 && rng.next() < 0.5) {   // a roadblock somewhere along the route, not near its ends
        const int j = 6 + static_cast<int>(std::floor(rng.next() * (len - 12))), at = rt.nodes[j];
        int from = j - 1;                            // the tank notices it from `notice` metres away
        while (from > 0 && std::hypot(static_cast<double>(rt.nodes[from - 1] / C - at / C),
                                      static_cast<double>(rt.nodes[from - 1] % C - at % C)) <= SENSE.notice)
            --from;
        next = risk_drive(m, f, rt.nodes[from], rt.nodes[len - 1], w, at);
        out.block = true;
        if (!next.found) {
            out.stuck = true;
            out.time = kInf;
            return out;
        }
        legs[0].to = from;
        legs[1] = {&next, static_cast<int>(next.nodes.size()) - 1};
        n_legs = 2;
    }
    double look[16];
    const int towers = std::min(T, 16);
    for (int n = 0; n < towers; ++n) look[n] = rng.next() * SENSE.look;
    double t = 0.0, last_hit = -kInf;
    long hits = 0;
    for (int l = 0; l < n_legs; ++l) {
        const RiskRoute& leg = *legs[l].rt;
        const std::vector<float>& pds = leg.air ? f.air_pd : f.ground_pd;
        for (int k = 1; k <= legs[l].to; ++k) {
            const double t1 = t + leg.step[k];
            const std::size_t at = static_cast<std::size_t>(leg.nodes[k]) * T;
            for (int n = 0; n < towers; ++n)
                for (; look[n] < t1; look[n] += SENSE.look) {
                    const double pd = pds[at + n];
                    if (pd > 0.0 && rng.next() < pd) {
                        ++hits;
                        if (look[n] - last_hit <= SENSE.drop) out.tracked = true;
                        last_hit = look[n];
                    }
                }
            t = t1;
        }
    }
    out.detected = hits > 0;
    out.time = t;
    return out;
}

McResult monte_carlo(const Map& m, const Field& f, const RiskRoute& rt, double w, long trials, std::uint32_t seed,
                     double arrive_by, int threads) {
    const long kChunk = 256;                         // trials per job: the same split whatever the thread count
    const long chunks = (trials + kChunk - 1) / kChunk;
    std::vector<McResult> part(static_cast<std::size_t>(chunks));
    auto job = [&](long c) {
        McResult& r = part[static_cast<std::size_t>(c)];
        for (long i = c * kChunk, end = std::min(trials, (c + 1) * kChunk); i < end; ++i) {
            const Trial tr = run_trial(m, f, rt, w, seed, i);
            const bool late = arrive_by >= 0.0 && tr.time > arrive_by + 1e-9;
            ++r.n;
            r.block += tr.block; r.stuck += tr.stuck; r.detected += tr.detected; r.tracked += tr.tracked; r.late += late;
            if (!tr.block) { ++r.clean; r.clean_detected += tr.detected; }
            if (!tr.stuck) r.time_sum += tr.time;
            if (!tr.stuck && !tr.detected && !late) ++r.ok;
        }
    };
    if (threads < 1) threads = ThreadPool::hardware();
    if (threads == 1 || chunks == 1) {
        for (long c = 0; c < chunks; ++c) job(c);
    } else {
        ThreadPool pool(static_cast<int>(std::min<long>(threads, chunks)));
        for (long c = 0; c < chunks; ++c) pool.run([&job, c] { job(c); });
        pool.wait();
    }
    McResult total;
    for (const McResult& r : part) add(total, r);    // in the chunks' order: the same sum whatever ran first
    return total;
}

}  // namespace robotnav
