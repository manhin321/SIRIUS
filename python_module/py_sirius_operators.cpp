#include "python_module_includes.hpp"

using complex_double = std::complex<double>;

void init_operators(py::module& m)
{
    py::class_<Non_local_operator>(m, "Non_local_operator")
        .def("get_matrix", &Non_local_operator::get_matrix<std::complex<double>>);
    py::class_<D_operator, Non_local_operator>(m, "D_operator");
    py::class_<Q_operator, Non_local_operator>(m, "Q_operator");

    py::class_<Hamiltonian0>(m, "Hamiltonian0")
        .def(py::init<Potential&>(), py::keep_alive<1, 2>())
        .def("Q", &Hamiltonian0::Q, py::return_value_policy::reference_internal)
        .def("D", &Hamiltonian0::D, py::return_value_policy::reference_internal)
        .def("potential", &Hamiltonian0::potential, py::return_value_policy::reference_internal);

    py::class_<Hamiltonian_k>(m, "Hamiltonian_k")
        .def(py::init<Hamiltonian0&, K_point&>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>());

    py::class_<S_k<complex_double>>(m, "S_k")
        .def(py::init<const Simulation_context&, const Q_operator&, const Beta_projectors_base&, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def_property_readonly("size", &S_k<complex_double>::size)
        .def("apply", [](py::object& obj, py::array_t<complex_double>& X) {
            using class_t = S_k<complex_double>;
            class_t& sk = obj.cast<class_t&>();

            if (X.strides(0) != sizeof(complex_double)) {
                char msg[256];
                std::sprintf(msg, "invalid stride: [%ld, %ld] in %s:%d", X.strides(0), X.strides(1), __FILE__, __LINE__);
                throw std::runtime_error(msg);
            }
            if (X.ndim() != 2) {
                char msg[256];
                std::sprintf(msg, "wrong dimension: %lu in %s:%d", X.ndim(), __FILE__, __LINE__);
                throw std::runtime_error(msg);
            }
            auto ptr = X.request().ptr;
            int rows = X.shape(0);
            int cols = X.shape(1);
            const sddk::mdarray<complex_double, 2> array(static_cast<complex_double*>(ptr), rows, cols);
            return sk.apply(array, memory_t::host);
        });

    py::class_<InverseS_k<complex_double>>(m, "InverseS_k")
        .def(py::init<const Simulation_context&, const Q_operator&, const Beta_projectors_base&, int>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def_property_readonly("size", &InverseS_k<complex_double>::size)
        .def("apply", [](py::object& obj, py::array_t<complex_double>& X) {
            using class_t = InverseS_k<complex_double>;
            class_t& inverse_sk = obj.cast<class_t&>();

            if (X.strides(0) != sizeof(complex_double)) {
                char msg[256];
                std::sprintf(msg, "invalid stride: [%ld, %ld] in %s:%d", X.strides(0), X.strides(1), __FILE__,
                             __LINE__);
                throw std::runtime_error(msg);
            }
            if (X.ndim() != 2) {
                char msg[256];
                std::sprintf(msg, "wrong dimension: %lu in %s:%d", X.ndim(), __FILE__, __LINE__);
                throw std::runtime_error(msg);
            }
            auto ptr = X.request().ptr;
            int rows = X.shape(0);
            int cols = X.shape(1);
            const sddk::mdarray<complex_double, 2> array(reinterpret_cast<complex_double*>(ptr), rows, cols);
            return inverse_sk.apply(array, memory_t::host);
        });

    py::class_<Ultrasoft_preconditioner<complex_double>>(m, "Precond_us")
        .def(py::init<Simulation_context&, const Q_operator&, int, const Beta_projectors_base&, const Gvec&>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 5>(), py::keep_alive<1, 6>())
        .def_property_readonly("size", &Ultrasoft_preconditioner<complex_double>::size)
        .def("apply", [](py::object& obj, py::array_t<complex_double>& X) {
            using class_t    = Ultrasoft_preconditioner<complex_double>;
            class_t& precond = obj.cast<class_t&>();
            if (precond.ctx().preferred_memory_t() != memory_t::host) {
                char msg[256];
                std::sprintf(msg, "only implemented for host memory (%s:%d)", __FILE__, __LINE__);
                throw std::runtime_error(msg);
            }
            if (X.strides(0) != sizeof(complex_double)) {
                char msg[256];
                std::sprintf(msg, "invalid stride: [%ld, %ld] in %s:%d", X.strides(0), X.strides(1), __FILE__,
                             __LINE__);
                throw std::runtime_error(msg);
            }
            if (X.ndim() != 2) {
                char msg[256];
                std::sprintf(msg, "wrong dimension: %lu in %s:%d", X.ndim(), __FILE__, __LINE__);
                throw std::runtime_error(msg);
            }
            auto ptr = X.request().ptr;
            int rows = X.shape(0);
            int cols = X.shape(1);
            const sddk::mdarray<complex_double, 2> array(static_cast<complex_double*>(ptr), rows, cols);
            return precond.apply(array);
        });

    py::class_<beta_chunk_t>(m, "beta_chunk")
        .def_readonly("num_beta", &beta_chunk_t::num_beta_)
        .def_readonly("num_atoms", &beta_chunk_t::num_atoms_)
        .def_readonly("desc", &beta_chunk_t::desc_, py::return_value_policy::reference_internal)
        .def("__repr__",
             [](const beta_chunk_t& obj) {
                 std::stringstream buffer;
                 for (int ia = 0; ia < obj.num_atoms_; ++ia) {
                     buffer << "\t atom ia = " << ia << " ";
                     buffer << "\t nbf     : " << std::setw(10) << obj.desc_(static_cast<int>(beta_desc_idx::nbf), ia)
                            << " ";
                     buffer << "\t offset  : " << std::setw(10)
                            << obj.desc_(static_cast<int>(beta_desc_idx::offset), ia) << " ";
                     buffer << "\t offset_t: " << std::setw(10)
                            << obj.desc_(static_cast<int>(beta_desc_idx::offset_t), ia) << " ";
                     buffer << "\t ja      : " << std::setw(10) << obj.desc_(static_cast<int>(beta_desc_idx::ia), ia)
                            << " ";
                     buffer << "\n";
                 }

                 return "num_beta: " + std::to_string(obj.num_beta_) + "\n" + "offset: " + std::to_string(obj.offset_) +
                        "\n"
                        "num_atoms: " +
                        std::to_string(obj.num_atoms_) + "\n" + "desc:\n" + buffer.str();
             })
        .def_readonly("offset", &beta_chunk_t::offset_);

    py::class_<beta_projectors_coeffs_t>(m, "beta_projector_coeffs")
        .def_readonly("a", &beta_projectors_coeffs_t::pw_coeffs_a, py::return_value_policy::reference_internal)
        .def_readonly("chunk", &beta_projectors_coeffs_t::beta_chunk, py::return_value_policy::reference_internal);

    py::class_<Beta_projector_generator>(m, "Beta_projector_generator")
        .def("generate", py::overload_cast<beta_projectors_coeffs_t&, int>(&Beta_projector_generator::generate, py::const_))
        .def("generate_j", py::overload_cast<beta_projectors_coeffs_t&, int, int>(&Beta_projector_generator::generate, py::const_));

    py::class_<Beta_projectors_base>(m, "Beta_projectors_base")
        .def("prepare", &Beta_projectors_base::prepare, py::keep_alive<1, 0>())
        .def_property_readonly("num_chunks", &Beta_projectors_base::num_chunks)
        .def("make_generator", py::overload_cast<>(&Beta_projectors_base::make_generator, py::const_), py::keep_alive<1, 0>());

    py::class_<Beta_projectors, Beta_projectors_base>(m, "Beta_projectors");



}
