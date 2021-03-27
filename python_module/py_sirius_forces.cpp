#include "python_module_includes.hpp"

void
init_forces(py::module& m)
{
    py::class_<Force>(m, "Force")
        .def(py::init<Simulation_context&, Density&, Potential&, K_point_set&>())
        .def("calc_forces_total", &Force::calc_forces_total, py::return_value_policy::reference_internal)
        .def_property_readonly("ewald", &Force::forces_ewald)
        .def_property_readonly("hubbard", &Force::forces_hubbard)
        .def_property_readonly("vloc", &Force::forces_vloc)
        .def_property_readonly("nonloc", &Force::forces_nonloc)
        .def_property_readonly("core", &Force::forces_core)
        .def_property_readonly("scf_corr", &Force::forces_scf_corr)
        .def_property_readonly("us", &Force::forces_us)
        .def_property_readonly("total", &Force::forces_total)
        .def("print_info", &Force::print_info);

    py::class_<Stress>(m, "Stress")
        .def(py::init<Simulation_context&, Density&, Potential&, K_point_set&>())
        .def("calc_stress_total", &Stress::calc_stress_total, py::return_value_policy::reference_internal)
        .def("calc_stress_har", &Stress::calc_stress_har, py::return_value_policy::reference_internal)
        .def("calc_stress_ewald", &Stress::calc_stress_ewald, py::return_value_policy::reference_internal)
        .def("calc_stress_xc", &Stress::calc_stress_xc, py::return_value_policy::reference_internal)
        .def("calc_stress_kin", &Stress::calc_stress_kin, py::return_value_policy::reference_internal)
        .def("calc_stress_vloc", &Stress::calc_stress_vloc, py::return_value_policy::reference_internal)
        .def("print_info", &Stress::print_info);
}
