#include "python_module_includes.hpp"
#include "utils/rt_graph.hpp"

void
init_ex1(py::module& m)
{
    py::class_<Communicator>(m, "Communicator");

    py::class_<Simulation_parameters>(m, "Simulation_parameters")
        .def_property_readonly("cfg", py::overload_cast<>(&Simulation_parameters::cfg, py::const_));

    py::class_<config_t>(m, "config_t").def_property_readonly("parameters", py::overload_cast<>(&config_t::parameters));
    py::class_<Config, config_t>(m, "Config");

    py::class_<config_t::settings_t>(m, "settings_t");
    py::class_<config_t::parameters_t>(m, "parameters_t")
        .def_property_readonly("density_tol", py::overload_cast<>(&config_t::parameters_t::density_tol, py::const_))
        .def_property_readonly("energy_tol", py::overload_cast<>(&config_t::parameters_t::energy_tol, py::const_))
        .def_property_readonly("use_symmetry", py::overload_cast<>(&config_t::parameters_t::use_symmetry, py::const_))
        .def_property_readonly("shiftk", py::overload_cast<>(&config_t::parameters_t::shiftk, py::const_))
        .def_property_readonly("ngridk", py::overload_cast<>(&config_t::parameters_t::ngridk, py::const_));

    py::class_<Simulation_context, Simulation_parameters>(m, "Simulation_context")
        .def(py::init<std::string const&>())
        .def(py::init<std::string const&, Communicator const&>(), py::keep_alive<1, 3>())
        .def("initialize", &Simulation_context::initialize)
        .def("num_bands", py::overload_cast<>(&Simulation_context::num_bands, py::const_))
        .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands))
        .def("max_occupancy", &Simulation_context::max_occupancy)
        .def("num_fv_states", py::overload_cast<>(&Simulation_context::num_fv_states, py::const_))
        .def("num_spins", &Simulation_context::num_spins)
        .def("verbosity", py::overload_cast<>(&Simulation_context::verbosity, py::const_))
        .def("create_storage_file", &Simulation_context::create_storage_file)
        .def("processing_unit", py::overload_cast<>(&Simulation_context::processing_unit, py::const_))
        .def("processing_unit", py::overload_cast<std::string>(&Simulation_context::processing_unit))
        .def("gvec", &Simulation_context::gvec, py::return_value_policy::reference_internal)
        .def("full_potential", &Simulation_context::full_potential)
        .def("hubbard_correction", &Simulation_context::hubbard_correction)
        //.def("fft", &Simulation_context::fft, py::return_value_policy::reference_internal)
        //.def("fft_coarse", &Simulation_context::fft_coarse, py::return_value_policy::reference_internal)
        .def("unit_cell", py::overload_cast<>(&Simulation_context::unit_cell, py::const_),
             py::return_value_policy::reference)
        .def("pw_cutoff", py::overload_cast<>(&Simulation_context::pw_cutoff, py::const_))
        .def("pw_cutoff", py::overload_cast<double>(&Simulation_context::pw_cutoff))
        .def("gk_cutoff", py::overload_cast<>(&Simulation_context::gk_cutoff, py::const_))
        .def("gk_cutoff", py::overload_cast<double>(&Simulation_context::gk_cutoff))
        .def("aw_cutoff", py::overload_cast<>(&Simulation_context::aw_cutoff, py::const_))
        .def("aw_cutoff", py::overload_cast<double>(&Simulation_context::aw_cutoff))
        //.def("parameters_input", py::overload_cast<>(&Simulation_context::parameters_input, py::const_),
        //     py::return_value_policy::reference)
        .def("num_spinors", &Simulation_context::num_spinors)
        .def("num_mag_dims", &Simulation_context::num_mag_dims)
        .def("gamma_point", py::overload_cast<bool>(&Simulation_context::gamma_point))
        .def("update", &Simulation_context::update)
        .def("use_symmetry", py::overload_cast<>(&Simulation_context::use_symmetry, py::const_))
        .def("preferred_memory_t", &Simulation_context::preferred_memory_t)
        //.def("mixer_input", &Simulation_context::mixer_input)
        .def(
            "comm", [](Simulation_context& obj) { return make_pycomm(obj.comm()); },
            py::return_value_policy::reference_internal)
        .def(
            "comm_k", [](Simulation_context& obj) { return make_pycomm(obj.comm_k()); },
            py::return_value_policy::reference_internal)
        .def(
            "comm_fft", [](Simulation_context& obj) { return make_pycomm(obj.comm_fft()); },
            py::return_value_policy::reference_internal)
        .def("iterative_solver_tolerance", py::overload_cast<double>(&Simulation_context::iterative_solver_tolerance))
        .def("iterative_solver_tolerance",
             py::overload_cast<>(&Simulation_context::iterative_solver_tolerance, py::const_));

    py::class_<rt_graph::TimingResult>(m, "TimningResult").def("json", &rt_graph::TimingResult::json);
    m.def("_timings", []() {
        // returns a json as string
        return ::utils::global_rtgraph_timer.process().json();
    });

}
