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

namespace {

int move_index(const Map& m, int a, int b) {                    // which of the eight moves goes from a to b
    const int dr = b / m.cols - a / m.cols, dc = b % m.cols - a % m.cols;
    for (int k = 0; k < 8; ++k)
        if (NB_R[k] == dr && NB_C[k] == dc) return k;
    fail("route_time: two cells of the route aren't neighbours");
}

struct CostsGraph {
    const Map& m;
    const std::vector<double>& costs;
    template <class F>
    void neighbours(int u, F&& f) const {
        const int r = u / m.cols, c = u % m.cols;
        for (int k = 0; k < 8; ++k) {
            const double t = costs[static_cast<std::size_t>(u) * 8 + k];
            if (t < kInf) f((r + NB_R[k]) * m.cols + c + NB_C[k], t);
        }
    }
    double heuristic(int a, int b) const { return m.heuristic(a, b); }
    double path_cost(const std::vector<int>& p) const { return route_time(m, costs, p); }
};

void check_costs(const Map& m, const std::vector<double>& costs) {
    if (costs.size() != static_cast<std::size_t>(m.size()) * 8) fail("move times: the table needs 8 per cell");
    for (int u = 0; u < m.size(); ++u)
        for (int k = 0; k < 8; ++k) {
            const double t = costs[static_cast<std::size_t>(u) * 8 + k];
            if (std::isnan(t) || t < 0.0) fail("move times: every time must be a number, zero or more");
            if (t < kInf && !m.inside(u / m.cols + NB_R[k], u % m.cols + NB_C[k])) fail("move times: a move off the map");
        }
}

}  // namespace

double route_time(const Map& m, const std::vector<double>& costs, const std::vector<int>& path) {
    double total = 0.0;
    for (std::size_t k = 1; k < path.size(); ++k)
        total += costs[static_cast<std::size_t>(path[k - 1]) * 8 + move_index(m, path[k - 1], path[k])];
    return total;
}

SearchResult astar_costs(const Map& m, const std::vector<double>& costs, int start, int goal, std::string name) {
    check_costs(m, costs);
    if (start < 0 || start >= m.size() || goal < 0 || goal >= m.size()) fail("astar_costs: start and goal must be on the map");
    const CostsGraph graph{m, costs};
    return best_first(graph, m.size(), start, goal, std::move(name), [](double g, double h) { return g + h; });
}

std::vector<double> cost_to_go_costs(const Map& m, const std::vector<double>& costs, int goal) {
    check_costs(m, costs);
    const int n = m.size();
    if (goal < 0 || goal >= n) fail("cost_to_go_costs: the goal is off the map");
    std::vector<double> dist(static_cast<std::size_t>(n), kInf);
    std::vector<char> done(static_cast<std::size_t>(n), 0);
    std::priority_queue<detail::Entry, std::vector<detail::Entry>, detail::Later> frontier;
    long tie = 0;
    dist[goal] = 0.0;
    frontier.push({0.0, tie++, goal});
    while (!frontier.empty()) {
        const int u = frontier.top().v;
        frontier.pop();
        if (done[u]) continue;
        done[u] = 1;
        const int ur = u / m.cols, uc = u % m.cols;
        for (int k = 0; k < 8; ++k) {                               // p -> u is the move k out of p
            const int pr = ur - NB_R[k], pc = uc - NB_C[k];
            if (!m.inside(pr, pc)) continue;
            const int p = pr * m.cols + pc;
            if (done[p]) continue;
            const double step = costs[static_cast<std::size_t>(p) * 8 + k];
            if (!(step < kInf)) continue;
            const double d = dist[u] + step;
            if (d < dist[p]) {
                dist[p] = d;
                frontier.push({d, tie++, p});
            }
        }
    }
    return dist;
}

std::vector<double> octile_table(const Map& m, int goal) {
    std::vector<double> h(static_cast<std::size_t>(m.size()));
    for (int i = 0; i < m.size(); ++i) h[i] = m.heuristic(i, goal);
    return h;
}

}  // namespace robotnav
