// The C++ core for the browser: the same sources compiled to WebAssembly, behind a small C interface. The page (and
// cpp/wasm/test.mjs under Node) hands a map over as plain arrays, then asks for routes, the digest, or Monte Carlo
// trials. One map is loaded at a time. Numbers cross as doubles and 32-bit ints; arrays as offsets into the module's
// memory, which the caller gets from rn_alloc.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "robotnav/airspace.hpp"
#include "robotnav/digest.hpp"
#include "robotnav/planners.hpp"
#include "robotnav/sensing.hpp"

using namespace robotnav;

#define RN_EXPORT(name) extern "C" __attribute__((export_name(#name)))

namespace {

struct World {
    Map map;
    std::vector<std::pair<int, int>> towers;         // empty: the towers stand where place_towers puts them
    std::unique_ptr<Airspace> air;
    std::unique_ptr<Field> field;
    std::string text;                                // the last digest
    double time = 0.0, hazard = 0.0, risk = 0.0;     // of the last route asked for
    long searched = 0;
};
World* W = nullptr;

int give(const std::vector<int>& nodes, int* out, int cap) {   // copy a route out; its length, or -1 if it won't fit
    const int n = static_cast<int>(nodes.size());
    if (n > cap) return -1;
    if (n) std::memcpy(out, nodes.data(), static_cast<std::size_t>(n) * sizeof(int));
    return n;
}

}  // namespace

RN_EXPORT(rn_alloc) void* rn_alloc(int bytes) { return std::malloc(static_cast<std::size_t>(bytes)); }
RN_EXPORT(rn_free) void rn_free(void* p) { std::free(p); }

// A new map: its size and drive-time model, then its grids, surfaces, water and preset trips, then rn_map_done.
RN_EXPORT(rn_map_begin) void rn_map_begin(int rows, int cols, double cell, double max_grade, double uphill, double short_tie) {
    delete W;
    W = new World;
    Map& m = W->map;
    m.theme = "map";
    m.rows = rows;
    m.cols = cols;
    m.cell = cell;
    m.max_grade = max_grade;
    m.uphill = uphill;
    m.short_tie = short_tie;
}
RN_EXPORT(rn_map_theme) void rn_map_theme(const char* name, long seed) {
    W->map.theme = name;
    W->map.seed = seed;
}
RN_EXPORT(rn_map_grids) void rn_map_grids(const double* height, const int* surface, const int* obstacle) {
    const std::size_t n = static_cast<std::size_t>(W->map.size());
    W->map.height.assign(height, height + n);
    W->map.surface.assign(surface, surface + n);
    W->map.obstacle.assign(obstacle, obstacle + n);
}
RN_EXPORT(rn_map_surfaces) void rn_map_surfaces(const double* cost, int n) { W->map.surface_cost.assign(cost, cost + n); }
RN_EXPORT(rn_map_sea) void rn_map_sea(double level) { W->map.sea = level; }
RN_EXPORT(rn_map_pond) void rn_map_pond(double level, const int* cells, int n) { W->map.ponds.push_back({level, {cells, cells + n}}); }
RN_EXPORT(rn_map_preset) void rn_map_preset(int start, int goal) { W->map.presets.push_back({start, goal, ""}); }
RN_EXPORT(rn_map_towers) void rn_map_towers(const int* rc, int n) {
    W->towers.clear();
    for (int k = 0; k < n; ++k) W->towers.emplace_back(rc[2 * k], rc[2 * k + 1]);
}
RN_EXPORT(rn_map_done) int rn_map_done() {           // builds the airspace and the towers' field; how many towers
    W->air = std::make_unique<Airspace>(W->towers.empty() ? make_airspace(W->map) : make_airspace(W->map, W->towers));
    W->field = std::make_unique<Field>(make_field(W->map, *W->air));
    return static_cast<int>(W->air->towers.size());
}

// Every result on every preset trip (robotnav/digest.hpp): the text's address; rn_text_length its length.
RN_EXPORT(rn_digest) const char* rn_digest() {
    W->text = digest_text(W->map);
    return W->text.c_str();
}
RN_EXPORT(rn_text_length) int rn_text_length() { return static_cast<int>(W->text.size()); }

// The quickest drive between two cells (A*): the route's cells into `out`, its length returned (0: no route).
RN_EXPORT(rn_astar) int rn_astar(int start, int goal, int* out, int cap) {
    const SearchResult r = astar(W->map, start, goal);
    W->time = r.cost;
    W->searched = r.nodes_expanded;
    return give(r.path, out, cap);
}
// The plan that pays w seconds per unit of hazard, for the tank (air = 0: cells) or the drone (air = 1: voxels).
RN_EXPORT(rn_plan) int rn_plan(int air, int start, int goal, double w, int avoid, int* out, int cap) {
    const RiskRoute r = air ? risk_flight(*W->air, *W->field, start, goal, w) : risk_drive(W->map, *W->field, start, goal, w, avoid);
    W->time = r.time;
    W->hazard = r.hazard;
    W->risk = r.risk;
    W->searched = r.searched;
    return give(r.nodes, out, cap);
}
RN_EXPORT(rn_last_time) double rn_last_time() { return W->time; }
RN_EXPORT(rn_last_hazard) double rn_last_hazard() { return W->hazard; }
RN_EXPORT(rn_last_risk) double rn_last_risk() { return W->risk; }
RN_EXPORT(rn_last_searched) double rn_last_searched() { return static_cast<double>(W->searched); }

// Monte Carlo: `trials` runs, numbered from `first`, of a route the caller hands over (its cells or voxels). Ten
// numbers into `out`: runs, met a roadblock, no way past, detected, tracked, late, without a roadblock, of those
// detected, succeeded, and the arrivals' seconds added up.
RN_EXPORT(rn_trials) void rn_trials(int air, const int* nodes, int n, double w, int first, int trials, unsigned seed, double arrive_by, double* out) {
    const RiskRoute rt = route_from(W->map, *W->air, *W->field, air != 0, std::vector<int>(nodes, nodes + n));
    McResult r;
    for (int i = first; i < first + trials; ++i) {
        const Trial t = run_trial(W->map, *W->field, rt, w, seed, i);
        const bool late = arrive_by >= 0.0 && t.time > arrive_by + 1e-9;
        ++r.n;
        r.block += t.block; r.stuck += t.stuck; r.detected += t.detected; r.tracked += t.tracked; r.late += late;
        if (!t.block) { ++r.clean; r.clean_detected += t.detected; }
        if (!t.stuck) r.time_sum += t.time;
        if (!t.stuck && !t.detected && !late) ++r.ok;
    }
    const double v[10] = {double(r.n), double(r.block), double(r.stuck), double(r.detected), double(r.tracked), double(r.late),
                          double(r.clean), double(r.clean_detected), double(r.ok), r.time_sum};
    std::memcpy(out, v, sizeof v);
}
// A place's hazard as the field has it (a cell, or a voxel), for the caller to compare its own with.
RN_EXPORT(rn_hazard) double rn_hazard(int air, int at) { return air ? W->field->air_hazard[at] : W->field->ground_hazard[at]; }
