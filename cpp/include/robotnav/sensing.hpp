// Sensing: what the watchtowers make of a mover, and what a route risks. A C++ port of robotnav.sensing and of the
// console's planning and trials in the viewer.
//
// A tower looks once a second. A mover in clear view is detected with a probability that falls with range: one look
// at a fluctuating (Swerling 1) target, Pd = Pfa^(1 / (1 + SNR)), with SNR falling as range^4 and Pd one half at
// r50. A place's hazard is -ln(1 - Pd) per look, summed over the towers, so the chance of crossing a route
// undetected is exp(-hazard added up along it). A plan is the quickest route that pays w seconds per unit of hazard.
//
// A trial runs a plan once with the looks left to chance: each tower's first look falls at a random moment, every
// look in view detects with its probability, and two detections within `drop` seconds make a track. For the tank,
// half the trials put a roadblock on the route; it notices from `notice` metres away and plans again from there.
// monte_carlo runs many trials on a thread pool. Every trial draws from its own random stream, set by the seed and
// the trial's number, and results are added up in a fixed order, so the answer is the same on one thread or sixty-four.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "robotnav/airspace.hpp"
#include "robotnav/map.hpp"

namespace robotnav {

// The towers' sensor (robotnav.sensing.SENSE; the viewer gets the same numbers).
struct SenseParams {
    double pfa = 1e-4;         // false-alarm probability a look is tuned to
    double r50 = 10.0;         // range (m) at which a look detects half the time
    double look = 1.0;         // seconds between one tower's looks
    double drop = 4.0;         // a track with no detection for this long is dropped (s)
    double notice = 5.0;       // how far off the tank notices a roadblock (m)
};
inline const SenseParams SENSE{};

double pd_at(double r);                              // one look at a mover r metres away in clear view
double hazard_of(double pd);                         // the detection rate (per second) that matches a per-look probability
std::pair<double, double> wilson(long k, long n);    // the 95% interval for a share of k in n (Wilson's score interval)

// Every place as the towers see it. Kept as 32-bit floats, as the viewer keeps them.
struct Field {
    int T = 0, N = 0, L = 0;                         // towers, cells, flight levels
    std::vector<float> ground_pd;                    // [cell * T + tower]: a look's chance at a mover a metre above the ground
    std::vector<float> ground_hazard;                // [cell]
    std::vector<float> air_pd;                       // [voxel * T + tower]
    std::vector<float> air_hazard;                   // [voxel]
    std::vector<double> edge;                        // [cell * 8 + move]: seconds for each of a cell's eight moves, -1 where it can't go
};
Field make_field(const Map& m, const Airspace& air);

struct RiskRoute {
    bool found = false;
    bool air = false;
    std::vector<int> nodes;                          // cells (ground) or voxels (air)
    std::vector<double> step;                        // seconds of each step; step[0] = 0
    double time = kInf;                              // seconds, start to finish
    double hazard = 0.0;                             // added up along the route
    double risk = 0.0;                               // 1 - exp(-hazard): the chance of being detected at least once
    long searched = 0;
};
// The quickest route that pays w seconds per unit of hazard: for the tank (never through `avoid`, a cell, if given),
// and for the drone. `start` and `goal` are cells.
RiskRoute risk_drive(const Map& m, const Field& f, int start, int goal, double w, int avoid = -1);
// The same route from the planners' shared search, without risk_drive's shortcuts (the moves' times looked up in a
// table, the search's arrays kept from one call to the next): what the tests hold risk_drive to.
RiskRoute risk_drive_reference(const Map& m, const Field& f, int start, int goal, double w, int avoid = -1);
RiskRoute risk_flight(const Airspace& air, const Field& f, int start, int goal, double w);
// A route given as its cells or voxels (the viewer hands its own over): the steps' times and the risk worked out.
RiskRoute route_from(const Map& m, const Airspace& air, const Field& f, bool is_air, std::vector<int> nodes);

struct Trial {
    bool stuck = false, block = false, detected = false, tracked = false;
    double time = 0.0;
};
Trial run_trial(const Map& m, const Field& f, const RiskRoute& rt, double w, std::uint32_t seed, long number);

struct McResult {
    long n = 0, block = 0, stuck = 0, detected = 0, tracked = 0, late = 0;
    long clean = 0, clean_detected = 0;              // trials with no roadblock, and those of them detected
    long ok = 0;                                     // arrived, in time, never detected
    double time_sum = 0.0;                           // of the trials that arrived
};
// `trials` runs of a plan. arrive_by < 0: no deadline. threads < 1: as many as the machine runs at once.
McResult monte_carlo(const Map& m, const Field& f, const RiskRoute& rt, double w, long trials, std::uint32_t seed,
                     double arrive_by = -1.0, int threads = 1);

}  // namespace robotnav
