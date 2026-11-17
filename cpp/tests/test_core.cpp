// Unit tests for the C++ core on small hand-made maps (no test framework needed: ctest runs this executable).
// The number-for-number comparison with the Python reference on the real maps is tests/test_cpp.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "robotnav/airspace.hpp"
#include "robotnav/fields.hpp"
#include "robotnav/landmarks.hpp"
#include "robotnav/learned.hpp"
#include "robotnav/mapio.hpp"
#include "robotnav/planners.hpp"
#include "robotnav/replan.hpp"
#include "robotnav/sensing.hpp"
#include "robotnav/thread_pool.hpp"

using namespace robotnav;

namespace {

int g_checks = 0, g_failures = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failures;                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

// A flat, open map: every move costs its length (1 m/s on the only surface).
Map flat_map(int rows, int cols) {
    Map m;
    m.theme = "test";
    m.rows = rows;
    m.cols = cols;
    m.height.assign(static_cast<std::size_t>(rows) * cols, 0.0);
    m.surface.assign(static_cast<std::size_t>(rows) * cols, 0);
    m.obstacle.assign(static_cast<std::size_t>(rows) * cols, OPEN);
    m.surface_cost = {1.0, 3.0};
    m.surface_names = {"Grass", "Mud"};
    m.start = 0;
    m.goal = rows * cols - 1;
    return m;
}

bool legal_route(const Map& m, const std::vector<int>& path) {
    for (std::size_t k = 1; k < path.size(); ++k) {
        bool ok = false;
        m.neighbours(path[k - 1], [&](int v, double) { ok = ok || v == path[k]; });
        if (!ok) return false;
    }
    return true;
}

void test_open_ground() {
    const Map m = flat_map(6, 6);
    const int s = m.at(5, 0), g = m.at(0, 5);
    const SearchResult a = astar(m, s, g), d = dijkstra(m, s, g), b = bfs(m, s, g);
    CHECK(a.found() && d.found() && b.found());
    CHECK(a.path.size() == 6 && b.path.size() == 6);          // five diagonal moves
    CHECK_NEAR(a.cost, 5 * kSqrt2, 1e-12);                      // each step is sqrt(2) m at 1 m/s
    CHECK(a.cost == d.cost);
    CHECK(a.nodes_expanded <= d.nodes_expanded);                // the estimate saves work
    CHECK(a.path.front() == s && a.path.back() == g);
    CHECK_NEAR(m.heuristic(s, g), a.cost, 1e-12);               // octile distance is exact on open flat ground
}

void test_straight_up() {
    // From the bottom-left corner straight up the side: the quickest route, and the first cell expanded is the start.
    const Map m = flat_map(3, 3);
    const SearchResult a = astar(m, m.at(2, 0), m.at(0, 0));
    CHECK((a.path == std::vector<int>{m.at(2, 0), m.at(1, 0), m.at(0, 0)}));
    CHECK(a.explored.front() == m.at(2, 0));
}

void test_wall_with_a_gap() {
    Map m = flat_map(7, 7);
    for (int r = 0; r < 6; ++r) m.obstacle[m.at(r, 3)] = ROCK;    // a wall down column 3, open only at the bottom
    const SearchResult a = astar(m, m.at(0, 0), m.at(0, 6)), d = dijkstra(m, m.at(0, 0), m.at(0, 6));
    CHECK(a.found());
    CHECK(std::find(a.path.begin(), a.path.end(), m.at(6, 3)) != a.path.end());
    CHECK(a.cost == d.cost);
    CHECK(legal_route(m, a.path));
    for (const SearchResult& r : plan_pair(m, m.at(0, 0), m.at(0, 6))) {
        CHECK(r.found());
        CHECK(legal_route(m, r.path));
        CHECK(r.cost >= a.cost - 1e-9);                             // nothing beats A*
    }
}

void test_no_squeezing_between_obstacles() {
    Map m = flat_map(3, 3);
    m.obstacle[m.at(0, 1)] = ROCK;
    m.obstacle[m.at(1, 0)] = TREE;
    CHECK(!astar(m, m.at(0, 0), m.at(1, 1)).found());               // the only way out is the diagonal squeeze
    CHECK(!bfs(m, m.at(0, 0), m.at(2, 2)).found());
}

void test_grade_limit_and_uphill_time() {
    Map m = flat_map(1, 3);
    m.height = {0.0, 1.0, 2.0};                                     // 100% grade: too steep
    CHECK(m.edge_cost(0, 1) < 0);
    CHECK(!astar(m, 0, 2).found());
    m.height = {0.0, 0.5, 1.0};                                     // 50% grade: drivable, slow uphill
    const double up = std::sqrt(1.0 * 1.0 + 0.5 * 0.5) * 1.0 * (1.0 + 3.0 * 0.5);
    CHECK(m.edge_cost(0, 1) == up);
    CHECK(m.edge_cost(1, 0) == std::sqrt(1.0 * 1.0 + 0.5 * 0.5));  // downhill: no penalty
    const SearchResult a = astar(m, 0, 2);
    CHECK(a.found() && a.cost == up + up);
}

void test_hills_and_mud_shape_the_quickest_route() {
    // A mud strip across the map with one firm crossing just beside the straight line: A* steps across on the firm
    // ground, the hill-blind route (shortest distance) wades straight through the mud, and takes longer.
    Map m = flat_map(9, 9);
    for (int c = 0; c < 9; ++c) m.surface[m.at(4, c)] = 1;
    m.surface[m.at(4, 1)] = 0;
    const int s = m.at(8, 0), g = m.at(0, 0);
    const SearchResult quick = astar(m, s, g), blind = shortest_route(m, s, g);
    CHECK(quick.found() && blind.found());
    CHECK(blind.path.size() == 9);                                  // straight up through the mud
    CHECK(quick.cost < blind.cost);
    CHECK_NEAR(quick.cost, 6 + 2 * kSqrt2, 1e-12);
    CHECK(blind.cost == 7.0 + 3.0);
    CHECK(blind.cost == m.path_cost(blind.path));                   // its cost is the time it really takes
    const PathStats st = path_stats(m, blind.path);
    CHECK(st.distance_m == 8.0 && st.rough_m == 1.0 && st.climb_m == 0.0);
}

void test_cost_to_go_and_table_astar() {
    // Hills, mud and a wall with a gap: the field from the goal must give, for every cell, the time of the quickest
    // route from that cell (Dijkstra forward), and A* steered by a table must behave like A* with the same estimate.
    Map m = flat_map(12, 12);
    for (int r = 0; r < 12; ++r)
        for (int c = 0; c < 12; ++c) m.height[m.at(r, c)] = 0.35 * std::sin(0.7 * r) * std::cos(0.5 * c) + 0.04 * r;
    for (int c = 0; c < 12; ++c)
        if (c != 9) m.obstacle[m.at(6, c)] = ROCK;
    for (int r = 0; r < 4; ++r) m.surface[m.at(r, 5)] = 1;
    m.obstacle[m.at(10, 10)] = ROCK;                                   // and a lone rock
    const int goal = m.at(1, 2);
    const std::vector<double> field = cost_to_go(m, goal);
    CHECK(field[goal] == 0.0);
    CHECK(field[m.at(6, 0)] == kInf);                                  // a rock: nothing to drive from
    int compared = 0;
    for (int s = 0; s < m.size(); ++s) {
        if (m.blocked(s)) continue;
        const SearchResult d = dijkstra(m, s, goal);
        CHECK(d.found() == (field[s] < kInf));
        if (d.found()) {
            CHECK_NEAR(field[s], d.cost, 1e-12 * (1.0 + d.cost));      // the same sums, added the other way round
            ++compared;
        }
    }
    CHECK(compared > 100);
    const int s = m.at(11, 0);
    const SearchResult a = astar(m, s, goal), t = astar_table(m, s, goal, octile_table(m, goal));
    CHECK(t.path == a.path && t.cost == a.cost && t.nodes_expanded == a.nodes_expanded && t.explored == a.explored);
    const SearchResult best = astar_table(m, s, goal, field);
    CHECK(best.found() && legal_route(m, best.path));
    CHECK_NEAR(best.cost, a.cost, 1e-12 * a.cost);                     // the true field is a perfect estimate ...
    CHECK(best.nodes_expanded < a.nodes_expanded);                     // ... and searches less
    CHECK(best.nodes_expanded <= static_cast<long>(best.path.size()) + 2);
    std::vector<double> zero(static_cast<std::size_t>(m.size()), 0.0);
    const SearchResult dj = dijkstra(m, s, goal), z = astar_table(m, s, goal, zero);
    CHECK(z.cost == dj.cost && z.nodes_expanded == dj.nodes_expanded);  // no estimate: Dijkstra
    bool threw = false;
    try {
        cost_to_go(m, m.at(6, 0));
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

Map hilly_map() {
    Map m = flat_map(14, 14);
    for (int r = 0; r < 14; ++r)
        for (int c = 0; c < 14; ++c) m.height[m.at(r, c)] = 0.4 * std::sin(0.6 * r) * std::cos(0.45 * c) + 0.03 * c;
    for (int c = 2; c < 14; ++c) m.obstacle[m.at(7, c)] = ROCK;      // a wall with a gap on the left
    for (int r = 0; r < 5; ++r) m.surface[m.at(r, 6)] = 1;            // and a strip of mud
    return m;
}

void test_landmarks() {
    const Map m = hilly_map();
    const Landmarks lm = pick_landmarks(m, 5);
    CHECK(lm.cells.size() == 5 && lm.from.size() == 5 && lm.to.size() == 5);
    const int goal = m.at(1, 12);
    const std::vector<double> truth = cost_to_go(m, goal), h = landmark_table(m, lm, goal);
    bool below = true, consistent = true;
    for (int v = 0; v < m.size(); ++v) {
        if (m.blocked(v)) continue;
        if (truth[v] < kInf && h[v] > truth[v] * (1 + 1e-12)) below = false;
        m.neighbours(v, [&](int n, double step) { consistent = consistent && h[v] <= step + h[n] + 1e-9; });
    }
    CHECK(below && consistent);                                         // never too high, and consistent
    for (int s : {m.at(13, 13), m.at(12, 0), m.at(8, 9)}) {
        const SearchResult a = astar(m, s, goal), b = astar_landmarks(m, lm, s, goal);
        CHECK(b.found() && legal_route(m, b.path));
        CHECK_NEAR(b.cost, a.cost, 1e-12 * a.cost);                    // still the quickest route ...
        CHECK(b.nodes_expanded <= a.nodes_expanded);                    // ... searching no more cells
    }
    const std::vector<double> d = cost_from(m, lm.cells[0]);
    CHECK(d[lm.cells[0]] == 0.0);
    CHECK_NEAR(d[goal], dijkstra(m, lm.cells[0], goal).cost, 1e-12 * d[goal]);
}

// Every move's time as a table, filled from the map's own drive-time model: planning on it must be the plain A*.
std::vector<double> move_table(const Map& m) {
    std::vector<double> costs(static_cast<std::size_t>(m.size()) * 8, kInf);
    for (int u = 0; u < m.size(); ++u) {
        if (m.blocked(u)) continue;                                     // nothing drives out of a rock
        m.neighbours(u, [&](int v, double step) {
            for (int k = 0; k < 8; ++k)
                if (v == (u / m.cols + NB_R[k]) * m.cols + u % m.cols + NB_C[k]) costs[static_cast<std::size_t>(u) * 8 + k] = step;
        });
    }
    return costs;
}

void test_move_time_tables() {
    const Map m = hilly_map();
    std::vector<double> costs = move_table(m);
    const int goal = m.at(1, 12);
    CHECK(cost_to_go_costs(m, costs, goal) == cost_to_go(m, goal));     // the same field, bit for bit
    for (int s : {m.at(13, 13), m.at(12, 0), m.at(8, 9)}) {
        const SearchResult a = astar(m, s, goal), t = astar_costs(m, costs, s, goal);
        CHECK(t.path == a.path && t.cost == a.cost && t.nodes_expanded == a.nodes_expanded && t.explored == a.explored);
        CHECK(route_time(m, costs, a.path) == m.path_cost(a.path));
    }
    for (int r = 0; r < 5; ++r)                                         // now the mud can't be driven into at all
        for (int k = 0; k < 8; ++k) {
            const int pr = r - NB_R[k], pc = 6 - NB_C[k];
            if (m.inside(pr, pc)) costs[static_cast<std::size_t>(pr * m.cols + pc) * 8 + k] = kInf;
        }
    const int s = m.at(0, 0);
    const SearchResult t = astar_costs(m, costs, s, goal);
    CHECK(t.found() && legal_route(m, t.path));
    for (int v : t.path) CHECK(m.surface[v] == 0);
    CHECK_NEAR(t.cost, cost_to_go_costs(m, costs, goal)[s], 1e-12 * t.cost);
    CHECK(t.cost > astar(m, s, goal).cost);
    int refused = 0;
    std::vector<double> bad = costs;
    bad.pop_back();
    try { astar_costs(m, bad, s, goal); } catch (const std::exception&) { ++refused; }
    bad = costs;
    bad[8] = -1.0;
    try { astar_costs(m, bad, s, goal); } catch (const std::exception&) { ++refused; }
    bad = costs;
    bad[0] = 1.0;                                                       // move 0 out of the corner leaves the map
    try { cost_to_go_costs(m, bad, goal); } catch (const std::exception&) { ++refused; }
    CHECK(refused == 3);
}

void test_learned_estimate() {
    // A network that outputs 0.25 everywhere (all weights 0, last bias 0.25): the estimate is 1.25 x climb-aware.
    std::string text = "# a test network\nbrubus-heuristic 1\ninputs 16\nmean";
    for (int i = 0; i < 16; ++i) text += " 0";
    text += "\nscale";
    for (int i = 0; i < 16; ++i) text += " 1";
    text += "\nlayer 16 2\n";
    for (int i = 0; i < 32; ++i) text += "0 ";
    text += "0 0\nlayer 2 1\n0 0 0.25\nend\n";
    const Heuristic h = parse_heuristic(text);
    CHECK(h.layers.size() == 2 && h.note == "a test network\n");
    const Map m = hilly_map();
    const int goal = m.at(1, 12), s = m.at(12, 1);
    const Features f = features(m, s, goal);
    CHECK(f.climb > 0.0 && h.estimate(m, s, goal) == f.climb * 1.25);
    CHECK(features(m, goal, goal).climb == 0.0 && h.estimate(m, goal, goal) == 0.0);
    const std::vector<double> capped = learned_table(h, m, goal, 1.1);
    CHECK(capped[s] == 1.1 * f.climb);                                  // the cap holds it to 1.1 x climb-aware
    const SearchResult a = astar_reopen(m, s, goal, octile_table(m, goal)), b = astar(m, s, goal);
    CHECK(a.path == b.path && a.nodes_expanded == b.nodes_expanded);    // consistent estimate: plain A*
    const SearchResult l = astar_learned(m, s, goal, h, 1.1);
    CHECK(l.found() && legal_route(m, l.path) && l.cost <= 1.1 * b.cost * (1 + 1e-12));
    bool threw = false;
    try {
        parse_heuristic("brubus-heuristic 1\ninputs 3\n");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_python_rounding() {
    CHECK(py_round(2.25, 1) == 2.2);                                // an exact tie goes to the even digit
    CHECK(py_round(0.125, 2) == 0.12);
    CHECK(py_round(2.675, 2) == 2.67);                              // 2.675 is really 2.67499999...
    CHECK(py_round(76.04, 1) == 76.0);
}

void test_sensor_disk() {
    CHECK(disk(1.5).size() == 9);
    CHECK(disk(2.0).size() == 13);
    CHECK(disk(5.0).size() == 81);
}

void test_dstar_lite_drives() {
    Map m = flat_map(15, 15);
    for (int r = 2; r < 15; ++r) m.obstacle[m.at(r, 7)] = ROCK;    // a wall the rover only finds when it gets close
    for (int k = 0; k < m.size(); ++k) m.height[k] = 0.05 * (k % 15);
    const int s = m.at(14, 0), g = m.at(14, 14);
    const SensorRun known = sensor_drive(m, s, g, 3.0, Prior::Map);
    CHECK(known.found && known.replans == 0);                   // a full map: the quickest route, no surprises
    CHECK_NEAR(known.time, known.optimal, 1e-9);
    for (Prior p : {Prior::None, Prior::Elevation}) {
        const SensorRun run = sensor_drive(m, s, g, 3.0, p, {}, true);
        CHECK(run.found);
        CHECK(run.driven.front() == s && run.driven.back() == g);
        CHECK(legal_route(m, run.driven));
        CHECK(run.time >= run.optimal - 1e-9);
        CHECK(run.replans > 0);
        CHECK(run.max_check_error < 1e-9);                          // every repaired plan was as quick as A*
        CHECK(run.steps.size() == run.driven.size());
    }
    // A roadblock on the quickest route, known to nobody: the rover finds it and drives round.
    const int block = known.driven[known.driven.size() / 2];
    const SensorRun around = sensor_drive(m, s, g, 3.0, Prior::Map, {{block, 0}}, true);
    CHECK(around.found && around.replans >= 1 && around.time >= known.time - 1e-9);
    CHECK(std::find(around.driven.begin(), around.driven.end(), block) == around.driven.end());
    // Walled off: no way through.
    Map shut = flat_map(5, 5);
    for (int r = 0; r < 5; ++r) shut.obstacle[shut.at(r, 2)] = WATER;
    CHECK(!sensor_drive(shut, shut.at(0, 0), shut.at(0, 4), 3.0).found);
}

void test_airspace_and_terrain_masking() {
    Map m = flat_map(40, 40);
    for (int r = 0; r < 40; ++r) m.height[m.at(r, 20)] = m.height[m.at(r, 21)] = 9.0;   // a ridge down the middle
    const std::vector<std::pair<int, int>> towers{{20, 10}};
    const Airspace air = make_airspace(m, towers);
    CHECK(air.L == AIR.levels && air.alt.front() == 0.0 + 1 * AIR.dz);
    CHECK(air.eyes.front() == 0.0 + AIR.eye);
    const auto voxel = [&](int k, int r, int c) { return (static_cast<std::size_t>(k) * 40 + r) * 40 + c; };
    CHECK(air.seen[voxel(0, 20, 15)]);                              // open ground in front of the tower
    CHECK(!air.seen[voxel(0, 20, 24)]);                             // low behind the ridge: hidden
    CHECK(!air.free[voxel(0, 20, 20)]);                             // the ridge itself is solid at the lowest level
    const Flight st = fly(air, m.at(5, 2), m.at(35, 2), true), di = fly(air, m.at(5, 2), m.at(35, 2), false);
    CHECK(st.found && di.found);
    CHECK(st.seen <= di.seen + 1e-9 && st.time >= di.time - 1e-9);
    CHECK(di.seen > 0.0);                                           // the direct route flies past the tower
    const Flight dj = fly(air, m.at(5, 2), m.at(35, 2), true, false);
    CHECK_NEAR(dj.cost, st.cost, 1e-9);                             // A* agrees with Dijkstra, with less work
    CHECK(dj.nodes >= st.nodes);
    const auto picked = place_towers(m, 2, 30.0);
    CHECK(picked.size() == 2);
    CHECK((picked[0].first - picked[1].first) * (picked[0].first - picked[1].first) +
              (picked[0].second - picked[1].second) * (picked[0].second - picked[1].second) >= 900);
}

void test_detection_model() {
    CHECK_NEAR(pd_at(SENSE.r50), 0.5, 1e-12);                       // one half at r50, by construction
    CHECK(pd_at(2.0) > 0.99 && pd_at(AIR.sight + 0.1) == 0.0);
    for (double r = 4.0; r + 0.5 <= AIR.sight; r += 0.5) CHECK(pd_at(r + 0.5) < pd_at(r));   // falls with range
    CHECK_NEAR(hazard_of(0.5), std::log(2.0), 1e-15);
    const auto [lo, hi] = wilson(50, 100);
    CHECK_NEAR(lo, 0.4038, 5e-4);
    CHECK_NEAR(hi, 0.5962, 5e-4);
    CHECK(wilson(0, 10).first < 1e-12 && wilson(10, 10).second > 1.0 - 1e-12);
}

// A 40 x 40 field with a ridge of rocks to drive round and one tower in the middle.
Map tower_field() {
    Map m = flat_map(40, 40);
    for (int r = 8; r < 32; ++r) m.obstacle[m.at(r, 14)] = ROCK;
    for (int c = 22; c < 30; ++c) m.surface[m.at(30, c)] = 1;
    return m;
}

void test_risk_routes() {
    const Map m = tower_field();
    const Airspace air = make_airspace(m, {{20, 22}});
    const Field f = make_field(m, air);
    CHECK(f.T == 1 && f.ground_hazard[m.at(20, 25)] > 0.0f && f.ground_hazard[m.at(2, 2)] == 0.0f);
    CHECK(f.ground_pd[m.at(20, 25)] > f.ground_pd[m.at(20, 35)]);   // nearer the tower, likelier seen
    const int trips[][2] = {{m.at(2, 2), m.at(37, 37)}, {m.at(20, 2), m.at(20, 38)}, {m.at(38, 20), m.at(1, 24)}, {m.at(5, 30), m.at(35, 5)}};
    for (const auto& t : trips) {
        const SearchResult plain = astar(m, t[0], t[1]);
        double last_hazard = kInf, last_time = 0.0;
        for (const double w : {0.0, 2.0, 4.0, 10.0, 40.0}) {
            const RiskRoute a = risk_drive(m, f, t[0], t[1], w), b = risk_drive_reference(m, f, t[0], t[1], w);
            CHECK(a.found && a.nodes == b.nodes && a.time == b.time && a.hazard == b.hazard && a.searched == b.searched);
            CHECK(legal_route(m, a.nodes));
            if (w == 0.0) CHECK(a.nodes == plain.path && a.time == plain.cost);       // no weight: plain A*
            CHECK(a.hazard <= last_hazard + 1e-12 && a.time >= last_time - 1e-12);    // more weight: safer, never quicker
            CHECK_NEAR(a.risk, 1.0 - std::exp(-a.hazard), 1e-15);
            last_hazard = a.hazard;
            last_time = a.time;
            for (std::size_t k = 3; k + 3 < a.nodes.size(); k += 5) {                 // and the same with a cell closed
                const RiskRoute c = risk_drive(m, f, a.nodes[1], t[1], w, a.nodes[k]), d = risk_drive_reference(m, f, a.nodes[1], t[1], w, a.nodes[k]);
                CHECK(c.found == d.found && c.nodes == d.nodes && c.time == d.time);
                CHECK(std::find(c.nodes.begin(), c.nodes.end(), a.nodes[k]) == c.nodes.end());
            }
        }
    }
    const RiskRoute fl = risk_flight(air, f, m.at(20, 2), m.at(20, 38), 0.0), hid = risk_flight(air, f, m.at(20, 2), m.at(20, 38), 40.0);
    const Flight direct = fly(air, m.at(20, 2), m.at(20, 38), false);
    CHECK(fl.found && fl.nodes == direct.voxels && fl.time == direct.time);           // no weight: the direct flight
    CHECK(hid.found && hid.hazard < fl.hazard && hid.time > fl.time);
    const RiskRoute again = route_from(m, air, f, true, hid.nodes);
    CHECK(again.time == hid.time && again.hazard == hid.hazard);
}

void test_thread_pool() {
    std::vector<int> done(500, 0);
    {
        ThreadPool pool(4);
        CHECK(pool.size() == 4 && ThreadPool::hardware() >= 1);
        for (int k = 0; k < 500; ++k) pool.run([&done, k] { done[k] = k + 1; });   // each job writes its own slot
        pool.wait();
        long sum = 0;
        for (int v : done) sum += v;
        CHECK(sum == 500L * 501 / 2);
        pool.run([&done] { done[0] = -7; });                                        // and it takes more work after a wait
        pool.wait();
        CHECK(done[0] == -7);
    }
}

void test_monte_carlo() {
    const Map m = tower_field();
    const Airspace air = make_airspace(m, {{20, 22}});
    const Field f = make_field(m, air);
    const RiskRoute rt = risk_drive(m, f, m.at(34, 16), m.at(34, 38), 2.0);
    CHECK(rt.found && rt.risk > 0.05 && rt.risk < 0.95);
    const Trial a = run_trial(m, f, rt, 2.0, 7u, 12), b = run_trial(m, f, rt, 2.0, 7u, 12);
    CHECK(a.detected == b.detected && a.tracked == b.tracked && a.time == b.time && a.block == b.block);   // a trial is its seed and number
    const long n = 20000;
    const McResult one = monte_carlo(m, f, rt, 2.0, n, 7u, -1.0, 1);
    for (const int threads : {2, 3, 8}) {                                          // the same answer on any number of threads
        const McResult r = monte_carlo(m, f, rt, 2.0, n, 7u, -1.0, threads);
        CHECK(r.n == n && r.ok == one.ok && r.detected == one.detected && r.tracked == one.tracked && r.block == one.block &&
              r.stuck == one.stuck && r.clean == one.clean && r.clean_detected == one.clean_detected && r.time_sum == one.time_sum);
    }
    CHECK(monte_carlo(m, f, rt, 2.0, n, 8u, -1.0, 2).detected != one.detected);    // another seed, other luck
    const auto [lo, hi] = wilson(one.clean_detected, one.clean);                   // the plan's risk is what the trials find
    CHECK(one.clean > n / 3 && lo - 0.01 <= rt.risk && rt.risk <= hi + 0.01);
    CHECK(one.block > n / 3 && one.tracked <= one.detected && one.ok <= n - one.detected);
    const McResult late = monte_carlo(m, f, rt, 2.0, 2000, 7u, 5.0, 2);            // a deadline no drive can keep: all late
    CHECK(late.late == 2000 && late.ok == 0);
    const RiskRoute fl = risk_flight(air, f, m.at(20, 2), m.at(20, 38), 0.0);
    const McResult up = monte_carlo(m, f, fl, 0.0, n, 7u, -1.0, 2);
    CHECK(up.block == 0 && up.clean == n);                                         // nothing blocks the drone
    const auto [flo, fhi] = wilson(up.detected, n);
    CHECK(flo - 0.01 <= fl.risk && fl.risk <= fhi + 0.01);
}

void test_map_file_round_trip() {
    Map m = flat_map(4, 5);
    m.theme = "island";
    m.seed = 7;
    m.height = {0.1, 0.2, 1.0 / 3.0, 2.5, -1.25, 6.02e23, 1e-300, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    m.obstacle[m.at(3, 4)] = WATER;
    m.sea = 0.6;
    m.ponds.push_back({0.25, {m.at(2, 2), m.at(2, 3)}});
    m.presets.push_back({m.at(0, 0), m.at(3, 3), "NW → SE"});
    const std::string path = "robotnav_test_roundtrip.map";
    save_map(m, path);
    const Map r = load_map(path);
    std::remove(path.c_str());
    CHECK(r.theme == m.theme && r.seed == m.seed && r.rows == m.rows && r.cols == m.cols);
    CHECK(r.height == m.height);                                    // every double back exactly
    CHECK(r.surface == m.surface && r.obstacle == m.obstacle);
    CHECK(r.surface_cost == m.surface_cost && r.surface_names == m.surface_names);
    CHECK(r.sea && *r.sea == 0.6);
    CHECK(r.ponds.size() == 1 && r.ponds[0].level == 0.25 && r.ponds[0].cells == m.ponds[0].cells);
    CHECK(r.presets.size() == 1 && r.presets[0].name == "NW → SE" && r.presets[0].goal == m.at(3, 3));
    CHECK(r.start == m.start && r.goal == m.goal);
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"open ground", test_open_ground},
        {"straight up", test_straight_up},
        {"wall with a gap", test_wall_with_a_gap},
        {"no squeezing between obstacles", test_no_squeezing_between_obstacles},
        {"grade limit and uphill time", test_grade_limit_and_uphill_time},
        {"hills and mud shape the quickest route", test_hills_and_mud_shape_the_quickest_route},
        {"cost to go and table A*", test_cost_to_go_and_table_astar},
        {"landmarks", test_landmarks},
        {"move-time tables", test_move_time_tables},
        {"learned estimate", test_learned_estimate},
        {"Python rounding", test_python_rounding},
        {"sensor disk", test_sensor_disk},
        {"D* Lite drives", test_dstar_lite_drives},
        {"airspace and terrain masking", test_airspace_and_terrain_masking},
        {"detection model", test_detection_model},
        {"risk routes", test_risk_routes},
        {"thread pool", test_thread_pool},
        {"Monte Carlo", test_monte_carlo},
        {"map file round trip", test_map_file_round_trip},
    };
    for (const auto& [name, run] : tests) {
        const int before = g_failures;
        run();
        std::printf("%-42s %s\n", name, g_failures == before ? "ok" : "FAILED");
    }
    std::printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
