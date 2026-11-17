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

}  // namespace robotnav
