#ifndef PYBIND_HELPERS_H
#define PYBIND_HELPERS_H

#include "pybind11/pybind11.h"
#include "utils/json.hpp"

namespace sirius {

py::oject pj_convert(nlohmann::json&);

std::string show_mat(const matrix3d<double>& mat);

}  // sirius

#endif /* PYBIND_HELPERS_H */
