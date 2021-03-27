#include "python_module_includes.hpp"

void
init_kpoint(py::module& m)
{
    py::class_<K_point>(m, "K_point")
        .def("band_energy", py::overload_cast<int, int>(&K_point::band_energy, py::const_))
        .def_property_readonly("vk", &K_point::vk, py::return_value_policy::copy)
        .def("generate_fv_states", &K_point::generate_fv_states)
        .def("set_band_energy", [](K_point& kpoint, int j, int ispn, double val) { kpoint.band_energy(j, ispn, val); })
        .def(
            "band_energies",
            [](K_point const& kpoint, int ispn) {
                std::vector<double> energies(kpoint.ctx().num_bands());
                for (int i = 0; i < kpoint.ctx().num_bands(); ++i) {
                    energies[i] = kpoint.band_energy(i, ispn);
                }
                return energies;
            },
            py::return_value_policy::copy)

        .def("band_occupancy",
             [](K_point const& kpoint, int ispn) {
                 std::vector<double> occ(kpoint.ctx().num_bands());
                 for (int i = 0; i < kpoint.ctx().num_bands(); ++i) {
                     occ[i] = kpoint.band_occupancy(i, ispn);
                 }
                 return occ;
             })
        .def(
            "set_band_occupancy",
            [](K_point& kpoint, int ispn, const std::vector<double>& fn) {
                assert(static_cast<int>(fn.size()) == kpoint.ctx().num_bands());
                for (size_t i = 0; i < fn.size(); ++i) {
                    kpoint.band_occupancy(i, ispn, fn[i]);
                }
            },
            "ispn"_a, "fn"_a)
        .def("gkvec_partition", &K_point::gkvec_partition, py::return_value_policy::reference_internal)
        .def("beta_projectors", py::overload_cast<>(&K_point::beta_projectors),
             py::return_value_policy::reference_internal)
        .def("gkvec", &K_point::gkvec, py::return_value_policy::reference_internal)
        .def("fv_states", &K_point::fv_states, py::return_value_policy::reference_internal)
        .def("ctx", &K_point::ctx, py::return_value_policy::reference_internal)
        .def("weight", &K_point::weight)
        .def("spinor_wave_functions", &K_point::spinor_wave_functions, py::return_value_policy::reference_internal);

    py::class_<K_point_set>(m, "K_point_set")
        .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>())
        .def(py::init<Simulation_context&, std::vector<std::array<double, 3>>>(), py::keep_alive<1, 2>())
        .def(py::init<Simulation_context&, std::initializer_list<std::array<double, 3>>>(), py::keep_alive<1, 2>())
        .def(py::init<Simulation_context&, vector3d<int>, vector3d<int>, bool>(), py::keep_alive<1, 2>())
        .def(py::init<Simulation_context&, std::vector<int>, std::vector<int>, bool>(), py::keep_alive<1, 2>())
        .def("initialize", &K_point_set::initialize, py::arg("counts") = std::vector<int>{})
        .def("ctx", &K_point_set::ctx, py::return_value_policy::reference_internal)
        .def("unit_cell", &K_point_set::unit_cell, py::return_value_policy::reference_internal)
        .def("_num_kpoints", &K_point_set::num_kpoints)
        .def("size", [](K_point_set& ks) -> int { return ks.spl_num_kpoints().local_size(); })
        .def("energy_fermi", &K_point_set::energy_fermi)
        .def("get_band_energies", &K_point_set::get_band_energies)
        .def("find_band_occupancies", &K_point_set::find_band_occupancies)
        .def("band_gap", &K_point_set::band_gap)
        .def("sync_band_energy", &K_point_set::sync_band<sync_band_t::energy>)
        .def("sync_band_occupancy", &K_point_set::sync_band<sync_band_t::occupancy>)
        .def("valence_eval_sum", &K_point_set::valence_eval_sum)
        .def("__contains__", [](K_point_set& ks, int i) { return (i >= 0 && i < ks.spl_num_kpoints().local_size()); })
        .def(
            "__getitem__",
            [](K_point_set& ks, int i) -> K_point& {
                if (i >= ks.spl_num_kpoints().local_size())
                    throw pybind11::index_error("out of bounds");
                return *ks[ks.spl_num_kpoints(i)];
            },
            py::return_value_policy::reference_internal)
        .def("__len__", [](K_point_set const& ks) { return ks.spl_num_kpoints().local_size(); })
        .def("add_kpoint",
             [](K_point_set& ks, std::vector<double> v, double weight) { ks.add_kpoint(v.data(), weight); })
        .def("add_kpoint", [](K_point_set& ks, vector3d<double>& v, double weight) { ks.add_kpoint(&v[0], weight); });
}
