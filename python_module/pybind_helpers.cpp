#include "utils/json.hpp"
#include "pybind11/pybind11.h"

using json = nlohmann::json;
using nlohmann::basic_json;
namespace py = pybind11;

// inspired by: https://github.com/mdcb/python-jsoncpp11/blob/master/extension.cpp
py::object pj_convert(json& node)
{
    switch (node.type()) {
        case json::value_t::null: {
            return py::reinterpret_borrow<py::object>(Py_None);
        }
        case json::value_t::boolean: {
            return py::bool_(node.get<bool>());
        }
        case json::value_t::string: {
            return py::str(node.get<std::string>());
        }
        case json::value_t::number_integer: {
            return py::int_(node.get<int>());
        }
        case json::value_t::number_unsigned: {
            return py::int_(node.get<unsigned int>());
        }
        case json::value_t::number_float: {
            return py::float_(node.get<double>());
        }
        case json::value_t::object: {
            py::dict result;
            for (auto it = node.begin(); it != node.end(); ++it) {
                json my_key(it.key());
                result[pj_convert(my_key)] = pj_convert(*it);
            }
            return result;
        }
        case json::value_t::array: {
            py::list result;
            for (auto it = node.begin(); it != node.end(); ++it) {
                result.append(pj_convert(*it));
            }
            return result;
        }
        default: {
            throw std::runtime_error("undefined json value");
            /* make compiler happy */
            return py::reinterpret_borrow<py::object>(Py_None);
        }
    }
}
