// A learned estimate of the drive time left to the goal, and A* guided by it (stage 2 of the learning work, see
// robotnav/learn and the README's "Learning from the planner").
//
// The estimate is a small neural network (dense layers with ReLU, a linear output) over 16 features of a cell and
// the goal: how far it is, how much climbing, and what lies on the straight line between them (its drive time,
// obstacles, water, cliffs). It predicts how much longer than the climb-aware estimate the drive really is:
//     estimate = climb x (1 + max(z, 0)),    climb = octile distance + uphill x net climb to the goal
// so it is never below the climb-aware estimate. Everything here is +, -, x, /, sqrt, min and max on doubles, in a
// fixed order: robotnav.learn.features and robotnav.learn.model compute the same numbers to the last bit (and so
// does the browser), which is why the searches below agree cell for cell across the three.
//
// A learned estimate can be too high, and then A* can miss the quickest route. astar_learned caps it at
// w x the climb-aware estimate, which is never too high, so the estimate is never more than w times the truth;
// with re-expansions (a cell found again by a quicker way is searched again) that guarantees a route at most
// w times as long as the quickest. w = infinity drops the cap (no guarantee; measured instead).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "robotnav/map.hpp"
#include "robotnav/planners.hpp"

namespace robotnav {

inline constexpr int kFeatures = 16;

// The network's inputs for one cell and the goal, and the climb-aware estimate they are relative to. For the goal
// itself (or a blocked cell) every input is 0 and so is the estimate.
struct Features {
    double climb = 0.0;
    std::array<double, kFeatures> x{};
};
Features features(const Map& m, int cell, int goal);

struct Dense {
    int in = 0, out = 0;
    std::vector<double> w;                  // in x out, row-major: w[i * out + j] links input i to output j
    std::vector<double> b;                  // out
};

// The trained network as read from its file (robotnav/assets/heuristic.txt, written by robotnav.learn.model).
struct Heuristic {
    std::vector<double> mean, scale;        // inputs are standardised as (x - mean) / scale
    std::vector<Dense> layers;              // ReLU after every layer but the last
    std::string note;                       // the file's comment lines (where it came from)
    // The estimate of the seconds left from `cell` to `goal`.
    double estimate(const Map& m, int cell, int goal) const;
    // The network's raw output for standardised-or-not inputs (exposed for the tests).
    double output(const std::array<double, kFeatures>& x) const;
};
Heuristic parse_heuristic(const std::string& text);
Heuristic load_heuristic(const std::string& path);

// The estimate for every cell (0 at blocked cells and the goal), capped at w x the climb-aware estimate.
std::vector<double> learned_table(const Heuristic& h, const Map& m, int goal, double w = kInf);

// A* with h[cell] as the estimate, searching a cell again when a quicker way to it turns up (needed for the
// guarantee when the estimate isn't consistent). With a consistent estimate it is exactly astar_table.
SearchResult astar_reopen(const Map& m, int start, int goal, const std::vector<double>& h,
                          std::string name = "A* (re-expanding)");

// astar_reopen with the learned estimate, capped at w x the climb-aware one, computed only for the cells the
// search touches (so it costs per cell reached, not per cell on the map).
SearchResult astar_learned(const Map& m, int start, int goal, const Heuristic& h, double w = kInf,
                           std::string name = "A* (learned)");

}  // namespace robotnav
