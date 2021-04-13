#include "python_module_includes.hpp"

template <typename T>
using matrix_storage_slab = sddk::matrix_storage<T, sddk::matrix_storage_t::slab>;
using complex_double    = std::complex<double>;

void init_sddk(py::module& m)
{
    py::class_<matrix_storage_slab<complex_double>>(m, "MatrixStorageSlabC")
        .def("is_remapped", &matrix_storage_slab<complex_double>::is_remapped)
        .def("prime", py::overload_cast<>(&matrix_storage_slab<complex_double>::prime),
             py::return_value_policy::reference_internal);

    py::class_<mdarray<complex_double, 1>>(m, "mdarray1c")
        .def("on_device", &mdarray<complex_double, 1>::on_device)
        .def("copy_to_host", [](mdarray<complex_double, 1>& mdarray) { mdarray.copy_to(memory_t::host); })
        .def("__array__", [](py::object& obj) {
                            mdarray<complex_double, 1>& arr = obj.cast<mdarray<complex_double, 1>&>();
                            int nrows                       = arr.size(0);
                            return py::array_t<complex_double>({nrows},
                                                               {1 * sizeof(complex_double)},
                                                               arr.at(memory_t::host), obj);
                          });

    py::class_<mdarray<double, 1>>(m, "mdarray1r")
        .def("on_device", &mdarray<double, 1>::on_device)
        .def("copy_to_host", [](mdarray<double, 1>& mdarray) { mdarray.copy_to(memory_t::host); })
        .def("__array__", [](py::object& obj) {
                            mdarray<double, 1>& arr = obj.cast<mdarray<double, 1>&>();
                            int nrows                       = arr.size(0);
                            return py::array_t<double>({nrows},
                              {1 * sizeof(double)},
                                                               arr.at(memory_t::host), obj);
                          });

    py::class_<mdarray<complex_double, 2>>(m, "mdarray2c")
        .def("on_device", &mdarray<complex_double, 2>::on_device)
        .def("copy_to_host", [](mdarray<complex_double, 2>& mdarray) { mdarray.copy_to(memory_t::host); })
        .def("__array__", [](py::object& obj) {
            mdarray<complex_double, 2>& arr = obj.cast<mdarray<complex_double, 2>&>();
            int nrows                       = arr.size(0);
            int ncols                       = arr.size(1);
            return py::array_t<complex_double>({nrows, ncols},
                                               {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                               arr.at(memory_t::host), obj);
        });

    py::class_<dmatrix<complex_double>, mdarray<complex_double, 2>>(m, "dmatrix");

    py::class_<mdarray<double, 2>>(m, "mdarray2")
        .def("on_device", &mdarray<double, 2>::on_device)
        .def("copy_to_host", [](mdarray<double, 2>& mdarray) { mdarray.copy_to(memory_t::host, 0, mdarray.size(1)); })
        .def("__array__", [](py::object& obj) {
            mdarray<double, 2>& arr = obj.cast<mdarray<double, 2>&>();
            int nrows               = arr.size(0);
            int ncols               = arr.size(1);
            return py::array_t<double>({nrows, ncols}, {1 * sizeof(double), nrows * sizeof(double)},
                                       arr.at(memory_t::host), obj);
        });

    py::class_<mdarray<int, 2>>(m, "mdarray2i")
        .def("on_device", &mdarray<int, 2>::on_device)
        .def("copy_to_host", [](mdarray<int, 2>& mdarray) { mdarray.copy_to(memory_t::host, 0, mdarray.size(1)); })
        .def("__array__", [](py::object& obj) {
            mdarray<int, 2>& arr = obj.cast<mdarray<int, 2>&>();
            int nrows               = arr.size(0);
            int ncols               = arr.size(1);
            return py::array_t<int>({nrows, ncols}, {1 * sizeof(int), nrows * sizeof(int)},
                                    arr.at(memory_t::host), obj);
        });
    py::class_<vector3d<int>>(m, "vector3d_int")
        .def(py::init<std::vector<int>>())
        .def("__call__", [](const vector3d<int>& obj, int x) { return obj[x]; })
        // .def("__repr__", [](const vector3d<int>& vec) { return show_vec(vec); })
        .def("__len__", &vector3d<int>::length)
        .def("__array__", [](vector3d<int>& v3d) {
            py::array_t<int> x(3);
            auto r = x.mutable_unchecked<1>();
            r(0)   = v3d[0];
            r(1)   = v3d[1];
            r(2)   = v3d[2];
            return x;
        });

    py::class_<vector3d<double>>(m, "vector3d_double")
        .def(py::init<std::vector<double>>())
        .def("__call__", [](const vector3d<double>& obj, int x) { return obj[x]; })
        // .def("__repr__", [](const vector3d<double>& vec) { return show_vec(vec); })
        .def("__array__",
             [](vector3d<double>& v3d) {
                 py::array_t<double> x(3);
                 auto r = x.mutable_unchecked<1>();
                 r(0)   = v3d[0];
                 r(1)   = v3d[1];
                 r(2)   = v3d[2];
                 return x;
             })
        .def("__len__", &vector3d<double>::length)
        .def(py::self - py::self)
        .def(py::self * float())
        .def(py::self + py::self)
        .def(py::init<vector3d<double>>());

    py::class_<matrix3d<double>>(m, "matrix3d")
        .def(py::init<std::vector<std::vector<double>>>())
        .def(py::init<>())
        .def("__call__", [](const matrix3d<double>& obj, int x, int y) { return obj(x, y); })
        .def("__array__",
             [](const matrix3d<double>& mat) {
                 return py::array_t<double>({3, 3}, {3 * sizeof(double), sizeof(double)}, &mat(0, 0));
             },
             py::return_value_policy::reference_internal)
        .def(py::self * py::self)
        .def("__getitem__", [](const matrix3d<double>& obj, int x, int y) { return obj(x, y); })
        .def("__mul__",
             [](const matrix3d<double>& obj, vector3d<double> const& b) {
                 vector3d<double> res = obj * b;
                 return res;
             })
        // .def("__repr__", [](const matrix3d<double>& mat) { return show_mat(mat); })
        .def(py::init<matrix3d<double>>())
        .def("det", &matrix3d<double>::det);

    py::class_<matrix3d<int>>(m, "matrix3di")
        .def(py::init<std::vector<std::vector<int>>>())
        .def(py::init<>())
        .def("__call__", [](const matrix3d<int>& obj, int x, int y) { return obj(x, y); })
        .def(
            "__array__",
            [](const matrix3d<int>& mat) {
                return py::array_t<int>({3, 3}, {3 * sizeof(int), sizeof(int)}, &mat(0, 0));
            },
            py::return_value_policy::reference_internal)
        .def(py::self * py::self)
        .def("__getitem__", [](const matrix3d<int>& obj, int x, int y) { return obj(x, y); })
        .def("__mul__",
             [](const matrix3d<int>& obj, vector3d<int> const& b) {
                 vector3d<int> res = obj * b;
                 return res;
             })
        // .def("__repr__", [](const matrix3d<int>& mat) { return show_mat(mat); })
        .def(py::init<matrix3d<int>>())
        .def("det", &matrix3d<int>::det);
}
