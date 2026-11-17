// Whole-map fields for the learning work (robotnav/learn): the true drive time from every cell to a goal, which is
// what a learned estimate is trained to predict, and A* guided by a table of estimates, which is how such an
// estimate is used and judged.
#pragma once

#include <string>
#include <vector>

#include "robotnav/map.hpp"
#include "robotnav/planners.hpp"

namespace robotnav {

// The fewest seconds of drive time from every cell to `goal`, or kInf where the goal can't be reached: Dijkstra
// run backwards from the goal, over the moves into each cell (moves are one-way in cost: uphill is slower). Same
// order of work as robotnav.learn.labels.cost_to_go, so the two agree to the last bit. A route's time added up
// from its start (path_cost) may differ from this in the last bits: the same numbers, added the other way round.
std::vector<double> cost_to_go(const Map& m, int goal);

// A* from start to goal with h[cell] as the estimate of the time left, multiplied by w (w = 1: plain A*). With the
// octile distance in the table this is the ordinary A*; with the true cost_to_go field it searches little more than
// the route itself, the best any estimate can do. Like the other planners, a cell is expanded at most once.
SearchResult astar_table(const Map& m, int start, int goal, const std::vector<double>& h, double w = 1.0,
                         std::string name = "A* (table)");

// The octile estimate as a table, the one A* uses: one value per cell, for this goal.
std::vector<double> octile_table(const Map& m, int goal);

// Every move's time read from a table instead of the drive-time model: costs[cell * 8 + k] is the time of move k
// (the planners' order) out of the cell, kInf where it can't be made. Stage 3 plans with Bruce's learned speeds and
// scores routes with the hidden true ones this way. Every time must be at least the move's flat length (1 s per
// metre at best), so the octile estimate stays a lower bound and A* stays optimal.
SearchResult astar_costs(const Map& m, const std::vector<double>& costs, int start, int goal,
                         std::string name = "A* (move times)");
// The fewest seconds from every cell to `goal` with those move times (Dijkstra backwards).
std::vector<double> cost_to_go_costs(const Map& m, const std::vector<double>& costs, int goal);
// A route's time with those move times, added up from its start.
double route_time(const Map& m, const std::vector<double>& costs, const std::vector<int>& path);

}  // namespace robotnav
