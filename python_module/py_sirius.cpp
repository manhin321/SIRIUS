#include "python_module_includes.hpp"

// inspired by: https://github.com/mdcb/python-jsoncpp11/blob/master/extension.cpp
py::object pj_convert(json& node);

// forward declaration
void initialize_subspace(DFT_ground_state&, Simulation_context&);
void apply_hamiltonian(Hamiltonian0& H0, K_point& kp, Wave_functions& wf_out, Wave_functions& wf, std::shared_ptr<Wave_functions>& swf);

/* typedefs */
using complex_double = std::complex<double>;

void init_ex1(py::module&);
void init_ex2(py::module&);
void init_density_potential(py::module&);
void init_operators(py::module&);
void init_sddk(py::module&);
void init_fft(py::module&);
void init_dft(py::module&);
void init_kpoint(py::module&);
void init_forces(py::module&);

PYBIND11_MODULE(py_sirius, m)
{
    // this is needed to be able to pass MPI_Comm from Python->C++
    if (import_mpi4py() < 0)
        return;
    // MPI_Init/Finalize
    int mpi_init_flag;
    MPI_Initialized(&mpi_init_flag);
    if (mpi_init_flag == true) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cout << "loading SIRIUS python module, MPI already initialized"
                      << "\n";
        sirius::initialize(false);
    } else {
        sirius::initialize(true);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cout << "loading SIRIUS python module, initialize MPI"
                      << "\n";
    }
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        int mpi_finalized_flag;
        MPI_Finalized(&mpi_finalized_flag);
        if (mpi_finalized_flag == true) {
            sirius::finalize(false);
        } else {
            sirius::finalize(
                /* call MPI_Finalize */
                true,
                /* reset device */
                false,
                /* fftw cleanup */
                false);
        }
    }));

    try {
        py::module::import("numpy");
    } catch (...) {
        return;
    }

    //m.def("timer_print", &utils::timer::print);
    m.def("num_devices", &acc::num_devices);

    init_ex1(m);
    init_ex2(m);
    init_density_potential(m);
    init_operators(m);
    init_sddk(m);
    init_fft(m);
    init_dft(m);
    init_kpoint(m);
    init_forces(m);

    py::class_<sddk::memory_pool>(m, "memory_pool");
    py::enum_<sddk::device_t>(m, "DeviceEnum").value("CPU", sddk::device_t::CPU).value("GPU", sddk::device_t::GPU);
    py::enum_<sddk::memory_t>(m, "MemoryEnum").value("device", memory_t::device).value("host", memory_t::host);

    m.def("ewald_energy", &ewald_energy);
    m.def("total_energy_components", &total_energy_components);
    m.def("set_atom_positions", &set_atom_positions);
    m.def("atom_positions", &atom_positions);
    m.def("energy_bxc", &energy_bxc);
#ifdef _OPENMP
    m.def("omp_set_num_threads", &omp_set_num_threads);
    m.def("omp_get_num_threads", &omp_get_num_threads);
#endif
    m.def("make_sirius_comm", &make_sirius_comm);
    m.def("make_pycomm", &make_pycomm);
    m.def("magnetization", &magnetization);
    m.def("sprint_magnetization", &sprint_magnetization);
    m.def("apply_hamiltonian", &apply_hamiltonian, "Hamiltonian0"_a, "kpoint"_a, "wf_out"_a,
          "wf_in"_a, py::arg("swf_out") = nullptr);
    m.def("initialize_subspace", &initialize_subspace);
    // m.def("inner_beta", &inner_beta(const Beta_projectors_base&, const Simulation_context&)); // TODO change name
    m.def("inner_beta", static_cast<sddk::matrix<double_complex>(*)(const Beta_projectors_base&, const Simulation_context&)>(inner_beta)); // TODO change name
}

void apply_hamiltonian(Hamiltonian0& H0, K_point& kp, Wave_functions& wf_out, Wave_functions& wf,
                       std::shared_ptr<Wave_functions>& swf)
{
    /////////////////////////////////////////////////////////////
    // // TODO: Hubbard needs manual call to copy to device // //
    /////////////////////////////////////////////////////////////
    int num_wf = wf.num_wf();
    int num_sc = wf.num_sc();
    if (num_wf != wf_out.num_wf() || wf_out.num_sc() != num_sc) {
        throw std::runtime_error("Hamiltonian::apply_ref (python bindings): num_sc or num_wf do not match");
    }
    auto H    = H0(kp);
    auto& ctx = H0.ctx();
#ifdef SIRIUS_GPU
    if (is_device_memory(ctx.preferred_memory_t())) {
        auto& mpd = ctx.mem_pool(memory_t::device);
        for (int ispn = 0; ispn < num_sc; ++ispn) {
            wf_out.pw_coeffs(ispn).allocate(mpd);
            wf.pw_coeffs(ispn).allocate(mpd);
            wf.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_wf);
        }
    }
#endif
    /* apply H to all wave functions */
    int N = 0;
    int n = num_wf;
    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        // sping_range: 2 for non-colinear magnetism, otherwise ispn_step
        auto spin_range = sddk::spin_range((ctx.num_mag_dims() == 3) ? 2 : ispn_step);
        H.apply_h_s<complex_double>(spin_range, N, n, wf, &wf_out, swf.get());
    }
#ifdef SIRIUS_GPU
    if (is_device_memory(ctx.preferred_memory_t())) {
        for (int ispn = 0; ispn < num_sc; ++ispn) {
            wf_out.pw_coeffs(ispn).copy_to(memory_t::host, 0, n);
            if (swf) {
                swf->pw_coeffs(ispn).copy_to(memory_t::host, 0, n);
            }
        }
    }
#endif // SIRIUS_GPU
}


void initialize_subspace(DFT_ground_state& dft_gs, Simulation_context& ctx)
{
    auto& kset = dft_gs.k_point_set();
    Hamiltonian0 H0(dft_gs.potential());
    Band(ctx).initialize_subspace(kset, H0);
}
