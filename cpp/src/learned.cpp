#include "robotnav/learned.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>
#include <utility>

namespace robotnav {

namespace {

// floor(a / b) for b > 0, as Python's // gives it.
long floordiv(long a, long b) {
    long q = a / b;
    if (a % b != 0 && a < 0) --q;
    return q;
}

int blocked_around(const Map& m, int cell) {             // blocked cells in the 5 x 5 square (off the map counts)
    const int r = cell / m.cols, c = cell % m.cols;
    int count = 0;
    for (int dr = -2; dr <= 2; ++dr)
        for (int dc = -2; dc <= 2; ++dc)
            if (!m.inside(r + dr, c + dc) || m.blocked(m.at(r + dr, c + dc))) ++count;
    return count;
}

int moves_into(const Map& m, int goal) {                 // legal moves into the goal from its neighbours
    const int ur = goal / m.cols, uc = goal % m.cols;
    int count = 0;
    for (int k = 0; k < 8; ++k) {
        const int pr = ur - NB_R[k], pc = uc - NB_C[k];
        if (!m.inside(pr, pc)) continue;
        const int p = m.at(pr, pc);
        if (m.blocked(p)) continue;
        if (NB_R[k] && NB_C[k] && (m.blocked(m.at(ur, pc)) || m.blocked(m.at(pr, uc)))) continue;
        if (m.edge_cost(p, goal) >= 0.0) ++count;
    }
    return count;
}

double parse_number(const std::string& tok) {
    char* end = nullptr;
    const double v = std::strtod(tok.c_str(), &end);    // correctly rounded, so the file's digits give the same double
    if (end == tok.c_str() || *end != '\0') fail("heuristic file: expected a number, got '" + tok + "'");
    return v;
}

}  // namespace

Features features(const Map& m, int cell, int goal) {
    Features f;
    if (cell == goal || m.blocked(cell)) return f;
    const int r0 = cell / m.cols, c0 = cell % m.cols, dr = goal / m.cols - r0, dc = goal % m.cols - c0;
    const int adr = std::abs(dr), adc = std::abs(dc), n = std::max(adr, adc);
    const double oct = ((adr + adc) + (kSqrt2 - 2.0) * std::min(adr, adc)) * m.cell;
    const double dh = m.height[goal] - m.height[cell];
    f.climb = oct + m.uphill * std::max(dh, 0.0);

    // Along the straight line: n steps of one cell (the cells at round(start + (goal - start) * k / n)).
    double line_time = 0.0, rise_up = 0.0, rise_down = 0.0;
    int bad = 0, steep = 0, blocked = 0, water = 0, run = 0, longest = 0;
    int pr = r0, pc = c0;
    for (int k = 1; k <= n; ++k) {
        const int r = r0 + static_cast<int>(floordiv(2L * dr * k + n, 2L * n));
        const int c = c0 + static_cast<int>(floordiv(2L * dc * k + n, 2L * n));
        const int a = m.at(pr, pc), b = m.at(r, c);
        if (!m.blocked(a) && !m.blocked(b)) {
            const double rise = m.height[b] - m.height[a];
            const double step = m.edge_cost(a, b);
            if (step < 0.0) {
                ++steep;
                ++bad;
            } else {
                line_time += step;
            }
            rise_up += std::max(rise, 0.0);
            rise_down += std::max(-rise, 0.0);
        } else {
            ++bad;
        }
        if (k < n) {                                         // the cells in between (not the goal)
            if (m.blocked(b)) {
                ++blocked;
                if (m.obstacle[b] == WATER) ++water;
                ++run;
                longest = std::max(longest, run);
            } else {
                run = 0;
            }
        }
        pr = r;
        pc = c;
    }
    int out = 0;
    m.neighbours(cell, [&](int, double) { ++out; });
    const double nd = n;
    f.x = {oct,
           dh,
           line_time / f.climb,
           line_time / oct,
           bad / nd,
           steep / nd,
           blocked / nd,
           water / nd,
           longest / nd,
           static_cast<double>(longest),
           rise_up / oct,
           rise_down / oct,
           (8 - out) / 8.0,
           (8 - moves_into(m, goal)) / 8.0,
           blocked_around(m, cell) / 25.0,
           blocked_around(m, goal) / 25.0};
    return f;
}

double Heuristic::output(const std::array<double, kFeatures>& raw) const {
    std::vector<double> a(raw.begin(), raw.end());
    for (int i = 0; i < kFeatures; ++i) a[i] = (a[i] - mean[i]) / scale[i];
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const Dense& L = layers[l];
        std::vector<double> z(L.b);
        for (int i = 0; i < L.in; ++i)                       // inputs in order, the same sums as the NumPy reference
            for (int j = 0; j < L.out; ++j) z[j] = z[j] + a[i] * L.w[static_cast<std::size_t>(i) * L.out + j];
        if (l + 1 < layers.size())
            for (double& v : z) v = std::max(v, 0.0);
        a = std::move(z);
    }
    return a[0];
}

double Heuristic::estimate(const Map& m, int cell, int goal) const {
    const Features f = features(m, cell, goal);
    if (f.climb == 0.0) return 0.0;
    return f.climb * (1.0 + std::max(output(f.x), 0.0));
}

Heuristic parse_heuristic(const std::string& text) {
    Heuristic h;
    std::istringstream lines(text);
    std::string line, all;
    while (std::getline(lines, line)) {
        if (!line.empty() && line[0] == '#') {
            h.note += line.substr(line.find_first_not_of("# ") == std::string::npos ? line.size() : line.find_first_not_of("# "));
            h.note += '\n';
        } else {
            all += line;
            all += ' ';
        }
    }
    std::istringstream in(all);
    std::string tok;
    auto next = [&]() -> std::string {
        if (!(in >> tok)) fail("heuristic file: it ends too soon");
        return tok;
    };
    auto expect = [&](const char* word) {
        if (next() != word) fail(std::string("heuristic file: expected '") + word + "', got '" + tok + "'");
    };
    expect("brubus-heuristic");
    if (next() != "1") fail("heuristic file: unknown version " + tok);
    expect("inputs");
    if (parse_number(next()) != kFeatures) fail("heuristic file: the network must take " + std::to_string(kFeatures) + " inputs");
    expect("mean");
    for (int i = 0; i < kFeatures; ++i) h.mean.push_back(parse_number(next()));
    expect("scale");
    for (int i = 0; i < kFeatures; ++i) h.scale.push_back(parse_number(next()));
    int width = kFeatures;
    while (next() != "end") {
        if (tok != "layer") fail("heuristic file: expected 'layer' or 'end', got '" + tok + "'");
        Dense L;
        L.in = static_cast<int>(parse_number(next()));
        L.out = static_cast<int>(parse_number(next()));
        if (L.in != width || L.out < 1) fail("heuristic file: the layers don't fit together");
        for (int k = 0; k < L.in * L.out; ++k) L.w.push_back(parse_number(next()));
        for (int k = 0; k < L.out; ++k) L.b.push_back(parse_number(next()));
        width = L.out;
        h.layers.push_back(std::move(L));
    }
    if (h.layers.empty() || width != 1) fail("heuristic file: the network must end in one output");
    return h;
}

Heuristic load_heuristic(const std::string& path) {
    std::ifstream f(path);
    if (!f) fail("can't open " + path);
    std::stringstream s;
    s << f.rdbuf();
    return parse_heuristic(s.str());
}

std::vector<double> learned_table(const Heuristic& h, const Map& m, int goal, double w) {
    std::vector<double> out(static_cast<std::size_t>(m.size()), 0.0);
    for (int v = 0; v < m.size(); ++v) {
        const Features f = features(m, v, goal);
        if (f.climb == 0.0) continue;
        const double est = f.climb * (1.0 + std::max(h.output(f.x), 0.0));
        out[v] = std::min(est, w * f.climb);
    }
    return out;
}

namespace {

struct Open {
    double f;
    long tie;
    int v;
    double g;                                                // the cost it was found at (later ones make it stale)
};
struct OpenLater {
    bool operator()(const Open& a, const Open& b) const { return a.f > b.f || (a.f == b.f && a.tie > b.tie); }
};

template <class Estimate>
SearchResult reopen_search(const Map& m, int start, int goal, std::string name, Estimate&& est) {
    const auto t0 = std::chrono::steady_clock::now();
    const int n = m.size();
    std::vector<double> g(static_cast<std::size_t>(n), kInf);
    std::vector<int> parent(static_cast<std::size_t>(n), -1);
    std::priority_queue<Open, std::vector<Open>, OpenLater> open;
    long tie = 0;
    SearchResult out;
    out.algorithm = std::move(name);
    out.max_frontier = 1;
    g[start] = 0.0;
    open.push({est(start), tie++, start, 0.0});
    bool reached = false;
    while (!open.empty()) {
        const Open top = open.top();
        open.pop();
        if (top.g != g[top.v]) continue;                    // stale: a quicker way here was found since
        out.explored.push_back(top.v);
        if (top.v == goal) {
            reached = true;
            break;
        }
        m.neighbours(top.v, [&](int nxt, double step) {
            const double ng = top.g + step;
            if (ng < g[nxt]) {
                g[nxt] = ng;
                parent[nxt] = top.v;
                open.push({ng + est(nxt), tie++, nxt, ng});
            }
        });
        out.max_frontier = std::max(out.max_frontier, static_cast<long>(open.size()));
    }
    out.nodes_expanded = static_cast<long>(out.explored.size());
    if (reached) {
        out.path = detail::walk_back(parent, goal);
        out.cost = m.path_cost(out.path);
    }
    out.runtime_ms = detail::ms_since(t0);
    return out;
}

}  // namespace

SearchResult astar_reopen(const Map& m, int start, int goal, const std::vector<double>& h, std::string name) {
    if (static_cast<int>(h.size()) != m.size()) fail("astar_reopen: the table needs one estimate per cell");
    if (start < 0 || start >= m.size() || goal < 0 || goal >= m.size()) fail("astar_reopen: start and goal must be on the map");
    return reopen_search(m, start, goal, std::move(name), [&](int v) { return h[v]; });
}

SearchResult astar_learned(const Map& m, int start, int goal, const Heuristic& h, double w, std::string name) {
    if (start < 0 || start >= m.size() || goal < 0 || goal >= m.size()) fail("astar_learned: start and goal must be on the map");
    std::vector<double> cache(static_cast<std::size_t>(m.size()), -1.0);
    return reopen_search(m, start, goal, std::move(name), [&](int v) {
        if (cache[v] < 0.0) {
            const Features f = features(m, v, goal);
            cache[v] = f.climb == 0.0 ? 0.0 : std::min(f.climb * (1.0 + std::max(h.output(f.x), 0.0)), w * f.climb);
        }
        return cache[v];
    });
}

}  // namespace robotnav
