#ifndef PYBIND_HELPERS_MAGNETIZATION_H
#define PYBIND_HELPERS_MAGNETIZATION_H

#include "k_point/k_point_set.hpp"
#include "density/density.hpp"

namespace sirius {

std::vector<double> magnetization(Density& density);
std::string sprint_magnetization(K_point_set& kset, const Density& density);

} // namespace sirius

#endif /* PYBIND_HELPERS_MAGNETIZATION_H */
