#include "robotnav/landmarks.hpp"

#include <chrono>
#include <cstdlib>
#include <deque>
#include <queue>

#include "robotnav/fields.hpp"

namespace robotnav {

std::vector<double> cost_from(const Map& m, int source) {
    const int n = m.size();
    if (source < 0 || source >= n) fail("cost_from: the source is off the map");
    if (m.blocked(source)) fail("cost_from: the source must be an open cell");
    std::vector<double> dist(static_cast<std::size_t>(n), kInf);
    std::vector<char> done(static_cast<std::size_t>(n), 0);
    std::priority_queue<detail::Entry, std::vector<detail::Entry>, detail::Later> frontier;
    long tie = 0;
    dist[source] = 0.0;
    frontier.push({0.0, tie++, source});
    while (!frontier.empty()) {
        const int u = frontier.top().v;
        frontier.pop();
        if (done[u]) continue;
        done[u] = 1;
        const double du = dist[u];
        m.neighbours(u, [&](int v, double step) {
            if (done[v]) return;
            const double d = du + step;
            if (d < dist[v]) {
                dist[v] = d;
                frontier.push({d, tie++, v});
            }
        });
    }
    return dist;
}

std::vector<char> largest_region(const Map& m) {
    const int n = m.size();
    std::vector<int> label(static_cast<std::size_t>(n), -1);
    int best = -1, best_size = 0, count = 0;
    for (int s = 0; s < n; ++s) {
        if (m.blocked(s) || label[s] >= 0) continue;
        label[s] = count;
        int size = 1;
        std::deque<int> queue{s};
        while (!queue.empty()) {
            const int u = queue.front();
            queue.pop_front();
            m.neighbours(u, [&](int v, double) {
                if (label[v] < 0) {
                    label[v] = count;
                    ++size;
                    queue.push_back(v);
                }
            });
        }
        if (size > best_size) {
            best = count;
            best_size = size;
        }
        ++count;
    }
    std::vector<char> region(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) region[i] = label[i] == best && best >= 0;
    return region;
}

Landmarks pick_landmarks(const Map& m, int k) {
    const auto t0 = std::chrono::steady_clock::now();
    if (k < 1) fail("pick_landmarks: at least one landmark");
    const std::vector<char> region = largest_region(m);
    Landmarks lm;
    int first = -1;
    long best_score = -1;
    for (int i = 0; i < m.size(); ++i) {                        // twice the distance from the centre; first wins ties
        if (!region[i]) continue;
        const long score = std::labs(2L * (i / m.cols) - m.rows) + std::labs(2L * (i % m.cols) - m.cols);
        if (score > best_score) {
            best_score = score;
            first = i;
        }
    }
    if (first < 0) fail("pick_landmarks: the map has no open cells");
    lm.cells.push_back(first);
    std::vector<double> nearest(static_cast<std::size_t>(m.size()), kInf);
    while (true) {
        lm.from.push_back(cost_from(m, lm.cells.back()));
        if (static_cast<int>(lm.cells.size()) == k) break;
        const std::vector<double>& d = lm.from.back();
        int next = 0;
        double far = -kInf;
        for (int i = 0; i < m.size(); ++i) {
            nearest[i] = std::min(nearest[i], d[i]);
            const double v = region[i] ? nearest[i] : -1.0;
            if (v > far) {
                far = v;
                next = i;
            }
        }
        if (far <= 0.0) break;                                  // fewer region cells than landmarks asked for
        lm.cells.push_back(next);
    }
    for (int c : lm.cells) lm.to.push_back(cost_to_go(m, c));
    lm.setup_ms = detail::ms_since(t0);
    return lm;
}

double landmark_bound(const Map& m, const Landmarks& lm, int cell, int goal) {
    const int dr = std::abs(cell / m.cols - goal / m.cols), dc = std::abs(cell % m.cols - goal % m.cols);
    const double oct = ((dr + dc) + (kSqrt2 - 2.0) * std::min(dr, dc)) * m.cell;
    double best = oct + m.uphill * std::max(m.height[goal] - m.height[cell], 0.0);
    for (std::size_t k = 0; k < lm.cells.size(); ++k) {
        const double tc = lm.to[k][cell], tg = lm.to[k][goal], fc = lm.from[k][cell], fg = lm.from[k][goal];
        if (tc < kInf && tg < kInf) best = std::max(best, tc - tg);
        if (fc < kInf && fg < kInf) best = std::max(best, fg - fc);
    }
    return best;
}

std::vector<double> landmark_table(const Map& m, const Landmarks& lm, int goal) {
    std::vector<double> out(static_cast<std::size_t>(m.size()));
    for (int i = 0; i < m.size(); ++i) out[i] = landmark_bound(m, lm, i, goal);
    return out;
}

namespace {

struct LandmarkGraph {
    const Map& m;
    const Landmarks& lm;
    template <class F>
    void neighbours(int u, F&& f) const { m.neighbours(u, f); }
    double heuristic(int a, int b) const { return landmark_bound(m, lm, a, b); }
    double path_cost(const std::vector<int>& p) const { return m.path_cost(p); }
};

}  // namespace

SearchResult astar_landmarks(const Map& m, const Landmarks& lm, int start, int goal) {
    if (start < 0 || start >= m.size() || goal < 0 || goal >= m.size()) fail("astar_landmarks: start and goal must be on the map");
    if (lm.from.empty() || static_cast<int>(lm.from[0].size()) != m.size()) fail("astar_landmarks: landmarks of another map");
    const LandmarkGraph graph{m, lm};
    return best_first(graph, m.size(), start, goal, "A* + landmarks", [](double g, double h) { return g + h; });
}

}  // namespace robotnav
