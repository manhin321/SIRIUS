#include "python_module_includes.hpp"

using complex_double = std::complex<double>;

void init_ex2(py::module& m)
{
        py::class_<Atom>(m, "Atom")
        .def("position", &Atom::position)
        .def("type_id", &Atom::type_id)
        .def("type", &Atom::type, py::return_value_policy::reference)
        .def_property_readonly("label", [](const Atom& obj) { return obj.type().label(); })
        .def_property_readonly("mass", [](const Atom& obj) { return obj.type().mass(); })
        .def("set_position", [](Atom& obj, const std::vector<double>& pos) {
            if (pos.size() != 3)
                throw std::runtime_error("wrong input");
            obj.set_position({pos[0], pos[1], pos[2]});
        });

    py::class_<Atom_type>(m, "Atom_type")
        .def_property_readonly("augment", [](const Atom_type& atype) { return atype.augment(); })
        .def_property_readonly("mass", &Atom_type::mass)
        .def_property_readonly("num_atoms", [](const Atom_type& atype) { return atype.num_atoms(); });

    py::class_<Unit_cell>(m, "Unit_cell")
        .def("add_atom_type", &Unit_cell::add_atom_type, py::return_value_policy::reference)
        .def("add_atom", py::overload_cast<const std::string, std::vector<double>>(&Unit_cell::add_atom))
        .def("atom", py::overload_cast<int>(&Unit_cell::atom), py::return_value_policy::reference)
        .def("atom_type", py::overload_cast<int>(&Unit_cell::atom_type), py::return_value_policy::reference)
        .def_property_readonly("atom_types", [](const Unit_cell& uc){
            // note: this might create memory issues, but Atom_type does not have a copy operators
            std::vector<const Atom_type*> types(uc.num_atom_types());
            for (int i = 0; i < uc.num_atom_types(); ++i) {
                types[i] = &uc.atom_type(i);
            }
            return types;
        }, py::return_value_policy::reference_internal)
        .def("lattice_vectors", &Unit_cell::lattice_vectors)
        .def(
            "set_lattice_vectors",
            [](Unit_cell& obj, py::buffer l1, py::buffer l2, py::buffer l3) { set_lattice_vectors(obj, l1, l2, l3); },
            "l1"_a, "l2"_a, "l3"_a)
        .def(
            "set_lattice_vectors",
            [](Unit_cell& obj, std::vector<double> l1, std::vector<double> l2, std::vector<double> l3) {
                obj.set_lattice_vectors(vector3d<double>(l1), vector3d<double>(l2), vector3d<double>(l3));
            },
            "l1"_a, "l2"_a, "l3"_a)
        .def("get_symmetry", &Unit_cell::get_symmetry)
        .def_property_readonly("num_electrons", &Unit_cell::num_electrons)
        .def_property_readonly("num_atoms", &Unit_cell::num_atoms)
        .def_property_readonly("num_valence_electrons", &Unit_cell::num_valence_electrons)
        .def_property_readonly("reciprocal_lattice_vectors", &Unit_cell::reciprocal_lattice_vectors)
        .def("generate_radial_functions", &Unit_cell::generate_radial_functions)
        .def_property_readonly("min_mt_radius", &Unit_cell::min_mt_radius)
        .def_property_readonly("max_mt_radius", &Unit_cell::max_mt_radius)
        .def_property_readonly("omega", &Unit_cell::omega)
        .def("print_info", &Unit_cell::print_info);

    py::class_<z_column_descriptor>(m, "z_column_descriptor")
        .def_readwrite("x", &z_column_descriptor::x)
        .def_readwrite("y", &z_column_descriptor::y)
        .def_readwrite("z", &z_column_descriptor::z)
        .def(py::init<int, int, std::vector<int>>());

    py::class_<Gvec>(m, "Gvec")
        .def(py::init<matrix3d<double>, double, bool>())
        .def("num_gvec", &sddk::Gvec::num_gvec)
        .def("count", &sddk::Gvec::count)
        .def("offset", &sddk::Gvec::offset)
        .def("gvec", &sddk::Gvec::gvec)
        .def("gkvec", &sddk::Gvec::gkvec)
        .def("gkvec_cart", &sddk::Gvec::gkvec_cart<index_domain_t::global>)
        .def("num_zcol", &sddk::Gvec::num_zcol)
        .def("gvec_alt",
             [](Gvec& obj, int idx) {
                 vector3d<int> vec(obj.gvec(idx));
                 std::vector<int> retr = {vec[0], vec[1], vec[2]};
                 return retr;
             })
        .def("index_by_gvec",
             [](Gvec& obj, std::vector<int> vec) {
                 vector3d<int> vec3d(vec);
                 return obj.index_by_gvec(vec3d);
             })
        .def("zcol",
             [](Gvec& gvec, int idx) {
                 z_column_descriptor obj(gvec.zcol(idx));
                 py::dict dict("x"_a = obj.x, "y"_a = obj.y, "z"_a = obj.z);
                 return dict;
             })
        .def("index_by_gvec", &Gvec::index_by_gvec);

    py::class_<Gvec_partition>(m, "Gvec_partition")
        .def_property_readonly("gvec", &Gvec_partition::gvec)
        .def_property_readonly("gvec_array", &Gvec_partition::get_gvec);

    // use std::shared_ptr as holder type, this required by Hamiltonian.apply_ref, apply_ref_inner
    py::class_<Wave_functions, std::shared_ptr<Wave_functions>>(m, "Wave_functions")
        .def(py::init<Gvec_partition const&, int, memory_t, int>(), "gvecp"_a, "num_wf"_a, "mem"_a, "num_sc"_a)
        .def("num_sc", &Wave_functions::num_sc)
        .def("num_wf", &Wave_functions::num_wf)
        .def("has_mt", &Wave_functions::has_mt)
        .def("zero_pw", &Wave_functions::zero_pw)
        .def("preferred_memory_t", py::overload_cast<>(&Wave_functions::preferred_memory_t, py::const_))
        .def("pw_coeffs",
             [](py::object& obj, int i) -> py::array_t<complex_double> {
                 Wave_functions& wf   = obj.cast<Wave_functions&>();
                 auto& matrix_storage = wf.pw_coeffs(i);
                 int nrows            = matrix_storage.prime().size(0);
                 int ncols            = matrix_storage.prime().size(1);
                 /* return underlying data as numpy.ndarray view */
                 return py::array_t<complex_double>({nrows, ncols},
                                                    {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                                    matrix_storage.prime().at(memory_t::host), obj);
             },
             py::keep_alive<0, 1>())
        .def("copy_to_gpu",
             [](Wave_functions& wf) {
                 /* is_on_device -> true if all internal storage is allocated on device */
                 bool is_on_device = true;
                 for (int i = 0; i < wf.num_sc(); ++i) {
                     is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
                 }
                 if (!is_on_device) {
                     for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                         wf.pw_coeffs(ispn).prime().allocate(memory_t::device);
                     }
                 }
                 for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                     wf.copy_to(spin_range(ispn), memory_t::device, 0, wf.num_wf());
                 }
             })
        .def("copy_to_cpu",
             [](Wave_functions& wf) {
                 /* is_on_device -> true if all internal storage is allocated on device */
                 bool is_on_device = true;
                 for (int i = 0; i < wf.num_sc(); ++i) {
                     is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
                 }
                 if (!is_on_device) {
                 } else {
                     for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                         wf.copy_to(spin_range(ispn), memory_t::host, 0, wf.num_wf());
                     }
                 }
             })
        .def("allocated_on_device",
             [](Wave_functions& wf) {
                 bool is_on_device = true;
                 for (int i = 0; i < wf.num_sc(); ++i) {
                     is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
                 }
                 return is_on_device;
             })
        .def("pw_coeffs_obj", py::overload_cast<int>(&Wave_functions::pw_coeffs, py::const_),
             py::return_value_policy::reference_internal);



}
