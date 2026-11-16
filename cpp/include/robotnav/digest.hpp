// The digest: every result on every preset trip of a map as text, times as their exact 64 bits and routes and search
// orders as hashes. Nothing is rounded, so two builds that print the same digest computed the same thing.
// cpp/tests/golden.digest holds it for the three maps; every build (native and WebAssembly) is checked against it.
#pragma once

#include <string>

#include "robotnav/map.hpp"

namespace robotnav {

std::string digest_text(const Map& m);

}  // namespace robotnav
