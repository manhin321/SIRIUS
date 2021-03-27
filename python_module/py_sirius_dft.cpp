#include "python_module_includes.hpp"

py::object pj_convert(json& node);

void init_dft(py::module& m)
{
    py::class_<Band>(m, "Band")
        .def(py::init<Simulation_context&>())
        .def("initialize_subspace", (void (Band::*)(K_point_set&, Hamiltonian0&) const) & Band::initialize_subspace)
        .def("solve", &Band::solve, "kset"_a, "hamiltonian"_a, py::arg("precompute")=true);

    py::class_<DFT_ground_state>(m, "DFT_ground_state")
        .def(py::init<K_point_set&>(), py::keep_alive<1, 2>())
        .def("print_info", &DFT_ground_state::print_info)
        .def("initial_state", &DFT_ground_state::initial_state)
        //.def("print_magnetic_moment", &DFT_ground_state::print_magnetic_moment)
        .def("total_energy", &DFT_ground_state::total_energy)
        .def("ks_energy", &DFT_ground_state::ks_energy)
        .def("serialize",
             [](DFT_ground_state& dft) {
                 auto json = dft.serialize();
                 return pj_convert(json);
             })
        .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
        .def(
            "find",
            [](DFT_ground_state& dft, double potential_tol, double energy_tol, double initial_tol, int num_dft_iter,
               bool write_state) {
                json js = dft.find(potential_tol, energy_tol, initial_tol, num_dft_iter, write_state);
                return pj_convert(js);
            },
            "potential_tol"_a, "energy_tol"_a, "initial_tol"_a, "num_dft_iter"_a, "write_state"_a)
        .def("check_scf_density",
             [](DFT_ground_state& dft) {
                 json js = dft.check_scf_density();
                 return pj_convert(js);
             })
        .def("k_point_set", &DFT_ground_state::k_point_set, py::return_value_policy::reference_internal)
        .def("potential", &DFT_ground_state::potential, py::return_value_policy::reference_internal)
        .def("forces", &DFT_ground_state::forces, py::return_value_policy::reference_internal)
        .def("stress", &DFT_ground_state::stress, py::return_value_policy::reference_internal)
        .def("update", &DFT_ground_state::update)
        .def("energy_kin_sum_pw", &DFT_ground_state::energy_kin_sum_pw);

    py::class_<Free_atom>(m, "Free_atom")
        .def(py::init<std::string>())
        .def(py::init<int>())
        .def("ground_state",
             [](Free_atom& atom, double energy_tol, double charge_tol, bool rel) {
                 json js = atom.ground_state(energy_tol, charge_tol, rel);
                 return pj_convert(js);
             })
        .def("radial_grid_points", &Free_atom::radial_grid_points)
        .def("num_atomic_levels", &Free_atom::num_atomic_levels)
        .def("atomic_level",
             [](Free_atom& atom, int idx) {
                 auto level = atom.atomic_level(idx);
                 json js;
                 js["n"]         = level.n;
                 js["l"]         = level.l;
                 js["k"]         = level.k;
                 js["occupancy"] = level.occupancy;
                 js["energy"]    = atom.atomic_level_energy(idx);
                 return pj_convert(js);
             })
        .def("free_atom_electronic_potential", [](Free_atom& atom) { return atom.free_atom_electronic_potential(); })
        .def("free_atom_wave_function", [](Free_atom& atom, int idx) { return atom.free_atom_wave_function(idx); })
        .def("free_atom_wave_function_x", [](Free_atom& atom, int idx) { return atom.free_atom_wave_function_x(idx); })
        .def("free_atom_wave_function_x_deriv",
             [](Free_atom& atom, int idx) { return atom.free_atom_wave_function_x_deriv(idx); })
        .def("free_atom_wave_function_residual",
             [](Free_atom& atom, int idx) { return atom.free_atom_wave_function_residual(idx); });


}
