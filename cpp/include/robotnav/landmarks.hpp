// A* with landmarks (ALT: A*, Landmarks, Triangle inequality): a C++ port of robotnav/landmarks.py.
//
// Once per map: k landmarks spread over the largest drivable region (farthest-point in drive time), and for each
// the quickest times from it to every cell and from every cell to it. Then for a cell c and goal g,
//     time(c -> g) >= time(c -> L) - time(g -> L)   and   time(c -> g) >= time(L -> g) - time(L -> c),
// and the largest of these (and the climb-aware estimate) is a lower bound on the time left that is consistent:
// A* guided by it still finds the quickest route, searching far fewer cells. Same work in the same order as the
// Python reference, so the two agree to the last bit.
#pragma once

#include <string>
#include <vector>

#include "robotnav/map.hpp"
#include "robotnav/planners.hpp"

namespace robotnav {

// Fewest seconds from `source` to every cell (Dijkstra forwards); kInf where it can't be reached.
std::vector<double> cost_from(const Map& m, int source);

// The largest set of open cells the rover can drive between (on a tie, the one holding the lowest cell).
std::vector<char> largest_region(const Map& m);

struct Landmarks {
    std::vector<int> cells;                     // flat indices
    std::vector<std::vector<double>> from, to;  // per landmark: seconds from it to every cell, from every cell to it
    double setup_ms = 0.0;                      // the time it took to pick them and fill the fields
};
Landmarks pick_landmarks(const Map& m, int k = 8);

// The landmark bound for one cell (never below the climb-aware estimate), and for every cell.
double landmark_bound(const Map& m, const Landmarks& lm, int cell, int goal);
std::vector<double> landmark_table(const Map& m, const Landmarks& lm, int goal);

// A* guided by the landmark bound (worked out for each cell the search reaches).
SearchResult astar_landmarks(const Map& m, const Landmarks& lm, int start, int goal);

}  // namespace robotnav
