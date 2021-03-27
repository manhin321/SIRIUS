#include "python_module_includes.hpp"

using complex_double = std::complex<double>;

void
init_density_potential(py::module& m)
{
    py::class_<Field4D>(m, "Field4D")
        .def(
            "f_pw_local",
            [](py::object& obj, int i) -> py::array_t<complex_double> {
                Field4D& field       = obj.cast<Field4D&>();
                auto& matrix_storage = field.component_raise(i).f_pw_local();
                int nrows            = matrix_storage.size(0);
                /* return underlying data as numpy.ndarray view */
                return py::array_t<complex_double>({nrows}, {1 * sizeof(complex_double)},
                                                   matrix_storage.at(memory_t::host), obj);
            },
            py::keep_alive<0, 1>())
        .def("f_rg",
             [](py::object& obj, int i) -> py::array_t<double> {
                 Field4D& field       = obj.cast<Field4D&>();
                 auto& matrix_storage = field.component_raise(i).f_rg();
                 int nrows            = matrix_storage.size(0);
                 /* return underlying data as numpy.ndarray view */
                 return py::array_t<double>({nrows}, {1 * sizeof(double)}, matrix_storage.at(memory_t::host), obj);
             })
        .def("component", py::overload_cast<int>(&Field4D::component), py::return_value_policy::reference_internal)
        .def(py::init<Simulation_context&, int>())
        .def("symmetrize", py::overload_cast<>(&Field4D::symmetrize));

    py::class_<Potential, Field4D>(m, "Potential")
        .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
        .def("generate", &Potential::generate)
        .def("scalar", py::overload_cast<>(&Potential::scalar, py::const_), py::return_value_policy::reference_internal)
        .def("symmetrize", py::overload_cast<>(&Potential::symmetrize))
        .def("fft_transform", &Potential::fft_transform)
        .def("save", &Potential::save)
        .def("load", &Potential::load)
        .def_property("vxc", py::overload_cast<>(&Potential::xc_potential),
                      py::overload_cast<>(&Potential::xc_potential), py::return_value_policy::reference_internal)
        .def_property("exc", py::overload_cast<>(&Potential::xc_energy_density),
                      py::overload_cast<>(&Potential::xc_energy_density), py::return_value_policy::reference_internal)
        .def_property("vha", py::overload_cast<>(&Potential::hartree_potential),
                      py::overload_cast<>(&Potential::hartree_potential), py::return_value_policy::reference_internal)
        .def_property("vloc", py::overload_cast<>(&Potential::local_potential),
                      py::overload_cast<>(&Potential::local_potential), py::return_value_policy::reference_internal)
        .def("energy_vha", &Potential::energy_vha)
        .def("energy_vxc", &Potential::energy_vxc)
        .def("energy_exc", &Potential::energy_exc)
        .def("PAW_total_energy", &Potential::PAW_total_energy)
        .def("PAW_one_elec_energy", &Potential::PAW_one_elec_energy);

    py::class_<Density, Field4D>(m, "Density")
        .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
        .def("initial_density", &Density::initial_density)
        .def("mixer_init", &Density::mixer_init)
        .def("check_num_electrons", &Density::check_num_electrons)
        .def("fft_transform", &Density::fft_transform)
        .def("mix", &Density::mix)
        .def("get_rho", py::overload_cast<>(&Density::rho, py::const_), py::return_value_policy::reference_internal)
        .def("symmetrize", py::overload_cast<>(&Density::symmetrize))
        .def("symmetrize_density_matrix", &Density::symmetrize_density_matrix)
        .def("generate", py::overload_cast<K_point_set const&, bool, bool>(&Density::generate), "kpointset"_a,
             "add_core"_a = true, "transform_to_rg"_a = false)
        .def("generate_paw_loc_density", &Density::generate_paw_loc_density)
        .def("compute_atomic_mag_mom", &Density::compute_atomic_mag_mom)
        .def("save", &Density::save)
        .def("check_num_electrons", &Density::check_num_electrons)
        .def("get_magnetisation", &Density::get_magnetisation)
        .def_property(
            "density_matrix",
            [](py::object& obj) -> py::array_t<complex_double> {
                Density& density = obj.cast<Density&>();
                auto& dm         = density.density_matrix();
                if (dm.at(memory_t::host) == nullptr) {
                    throw std::runtime_error("trying to access null pointer");
                }
                return py::array_t<complex_double, py::array::f_style>({dm.size(0), dm.size(1), dm.size(2), dm.size(3)},
                                                                       dm.at(memory_t::host), obj);
            },
            [](py::object& obj) -> py::array_t<complex_double> {
                Density& density = obj.cast<Density&>();
                auto& dm         = density.density_matrix();
                if (dm.at(memory_t::host) == nullptr) {
                    throw std::runtime_error("trying to access null pointer");
                }
                return py::array_t<complex_double, py::array::f_style>({dm.size(0), dm.size(1), dm.size(2), dm.size(3)},
                                                                       dm.at(memory_t::host), obj);
            },
            py::return_value_policy::reference_internal)
        .def("load", &Density::load);
}
