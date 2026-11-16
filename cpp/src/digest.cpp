#include "robotnav/digest.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "robotnav/airspace.hpp"
#include "robotnav/planners.hpp"
#include "robotnav/replan.hpp"

namespace robotnav {

namespace {

template <class... A>
void line(std::string& out, const char* f, A... a) {
    char buf[512];
    const int n = std::snprintf(buf, sizeof buf, f, a...);
    out.append(buf, n < 0 ? 0 : (n < static_cast<int>(sizeof buf) ? n : static_cast<int>(sizeof buf) - 1));
}

std::string bits(double x) {                                 // a double's 64 bits
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(u));
    return buf;
}

std::string cells_hash(const std::vector<int>& cells) {      // FNV-1a over the cells' bytes, low byte first
    std::uint64_t h = 1469598103934665603ull;
    for (const int c : cells)
        for (int k = 0; k < 4; ++k) {
            h ^= (static_cast<std::uint32_t>(c) >> (8 * k)) & 0xffu;
            h *= 1099511628211ull;
        }
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

}  // namespace

std::string digest_text(const Map& m) {
    const Airspace air = make_airspace(m);
    std::string out;
    line(out, "map %s %dx%d trips %zu\n", m.theme.c_str(), m.rows, m.cols, m.presets.size());
    int k = 0;
    for (const Preset& p : m.presets) {
        ++k;
        const int s = p.start, g = p.goal;
        const std::pair<const char*, SearchResult> searches[] = {
            {"bfs", bfs(m, s, g)}, {"dijkstra", dijkstra(m, s, g)}, {"astar", astar(m, s, g)},
            {"weighted", weighted_astar(m, s, g)}, {"greedy", greedy(m, s, g)}, {"shortest", shortest_route(m, s, g)}};
        for (const auto& [name, r] : searches)
            line(out, "P%d %s time %s searched %ld peak %ld route %s order %s\n", k, name, bits(r.cost).c_str(),
                 r.nodes_expanded, r.max_frontier, cells_hash(r.path).c_str(), cells_hash(r.explored).c_str());
        for (const double range : {5.0, 20.0}) {
            const SensorRun run = sensor_drive(m, s, g, range);
            line(out, "P%d dstar-%g time %s work %ld again %ld replans %d sensed %d driven %s\n", k, range,
                 bits(run.time).c_str(), run.dstar_work, run.astar_work, run.replans, run.sensed,
                 cells_hash(run.driven).c_str());
        }
        for (const bool stealth : {true, false}) {
            const Flight f = fly(air, s, g, stealth);
            line(out, "P%d %s time %s seen %s searched %ld voxels %s\n", k, stealth ? "stealth" : "direct",
                 bits(f.time).c_str(), bits(f.seen).c_str(), f.nodes, cells_hash(f.voxels).c_str());
        }
    }
    return out;
}

}  // namespace robotnav
