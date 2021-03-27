#include "python_module_includes.hpp"

using complex_double = std::complex<double>;

void
init_fft(py::module& m)
{
    py::class_<FFT3D_grid>(m, "FFT3D_grid")
        .def_property_readonly("num_points", py::overload_cast<>(&FFT3D_grid::num_points, py::const_))
        .def_property_readonly("shape",
                               [](const FFT3D_grid& obj) -> std::array<int, 3> {
                                   return {obj[0], obj[1], obj[2]};
                               })
        //.def_property_readonly("grid_size", &FFT3D_grid::grid_size) // TODO: is this needed?
        ;

    py::class_<Smooth_periodic_function<complex_double>>(m, "CSmooth_periodic_function")
        .def("fft", [](Smooth_periodic_function<complex_double>& obj) { return obj.fft_transform(-1); })
        .def("ifft", [](Smooth_periodic_function<complex_double>& obj) { return obj.fft_transform(1); })
        .def_property("pw", py::overload_cast<>(&Smooth_periodic_function<complex_double>::f_pw_local),
                      py::overload_cast<>(&Smooth_periodic_function<complex_double>::f_pw_local),
                      py::return_value_policy::reference_internal)
        .def_property("rg", py::overload_cast<>(&Smooth_periodic_function<complex_double>::f_rg),
                      py::overload_cast<>(&Smooth_periodic_function<complex_double>::f_rg),
                      py::return_value_policy::reference_internal)
        .def_property_readonly("gvec_partition", &Smooth_periodic_function<complex_double>::gvec_partition,
                               py::return_value_policy::reference_internal);

    py::class_<Smooth_periodic_function<double>>(m, "Smooth_periodic_function")
        .def("fft", [](Smooth_periodic_function<double>& obj) { return obj.fft_transform(-1); })
        .def("ifft", [](Smooth_periodic_function<double>& obj) { return obj.fft_transform(1); })
        .def("checksum_rg", &Smooth_periodic_function<double>::checksum_rg)
        .def("checksum_pw", &Smooth_periodic_function<double>::checksum_pw)
        .def_property("pw", py::overload_cast<>(&Smooth_periodic_function<double>::f_pw_local),
                      py::overload_cast<>(&Smooth_periodic_function<double>::f_pw_local),
                      py::return_value_policy::reference_internal)
        .def_property("rg", py::overload_cast<>(&Smooth_periodic_function<double>::f_rg),
                      py::overload_cast<>(&Smooth_periodic_function<double>::f_rg),
                      py::return_value_policy::reference_internal)
        .def_property_readonly("gvec_partition", &Smooth_periodic_function<double>::gvec_partition,
                               py::return_value_policy::reference_internal);

    py::class_<Periodic_function<double>, Smooth_periodic_function<double>>(m, "RPeriodic_function");
}
