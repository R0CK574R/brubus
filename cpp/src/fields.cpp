#include "robotnav/fields.hpp"

#include <cmath>
#include <queue>
#include <utility>

namespace robotnav {

std::vector<double> cost_to_go(const Map& m, int goal) {
    const int n = m.size();
    if (goal < 0 || goal >= n) fail("cost_to_go: the goal is off the map");
    if (m.blocked(goal)) fail("cost_to_go: the goal must be an open cell");
    std::vector<double> dist(static_cast<std::size_t>(n), kInf);
    std::vector<char> done(static_cast<std::size_t>(n), 0);
    std::priority_queue<detail::Entry, std::vector<detail::Entry>, detail::Later> frontier;
    long tie = 0;
    dist[goal] = 0.0;
    frontier.push({0.0, tie++, goal});
    while (!frontier.empty()) {
        const int u = frontier.top().v;
        frontier.pop();
        if (done[u]) continue;                                      // stale duplicate entry
        done[u] = 1;
        const int ur = u / m.cols, uc = u % m.cols;
        for (int k = 0; k < 8; ++k) {                               // p -> u is the move (NB_R[k], NB_C[k])
            const int pr = ur - NB_R[k], pc = uc - NB_C[k];
            if (!m.inside(pr, pc)) continue;
            const int p = pr * m.cols + pc;
            if (done[p] || m.blocked(p)) continue;
            if (NB_R[k] && NB_C[k] && (m.blocked(ur * m.cols + pc) || m.blocked(pr * m.cols + uc))) continue;
            const double step = m.edge_cost(p, u);
            if (step < 0.0) continue;                               // steeper than the grade limit
            const double d = dist[u] + step;
            if (d < dist[p]) {
                dist[p] = d;
                frontier.push({d, tie++, p});
            }
        }
    }
    return dist;
}

namespace {

// The terrain with its estimate read from a table (the goal is fixed by the table).
struct TableGraph {
    const Map& m;
    const std::vector<double>& h;
    template <class F>
    void neighbours(int u, F&& f) const { m.neighbours(u, f); }
    double heuristic(int a, int) const { return h[a]; }
    double path_cost(const std::vector<int>& p) const { return m.path_cost(p); }
};

}  // namespace

SearchResult astar_table(const Map& m, int start, int goal, const std::vector<double>& h, double w, std::string name) {
    if (static_cast<int>(h.size()) != m.size()) fail("astar_table: the table needs one estimate per cell");
    for (double v : h)
        if (std::isnan(v) || v < 0.0) fail("astar_table: every estimate must be a number, zero or more");
    if (start < 0 || start >= m.size() || goal < 0 || goal >= m.size()) fail("astar_table: start and goal must be on the map");
    const TableGraph graph{m, h};
    if (w == 1.0)
        return best_first(graph, m.size(), start, goal, std::move(name), [](double g, double est) { return g + est; });
    return best_first(graph, m.size(), start, goal, std::move(name), [w](double g, double est) { return g + w * est; });
}

std::vector<double> octile_table(const Map& m, int goal) {
    std::vector<double> h(static_cast<std::size_t>(m.size()));
    for (int i = 0; i < m.size(); ++i) h[i] = m.heuristic(i, goal);
    return h;
}

}  // namespace robotnav
