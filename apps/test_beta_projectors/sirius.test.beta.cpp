#include "utils/profiler.hpp"
#include <sirius.hpp>
#include "filesystem.hpp"
#include <utils/json.hpp>
#include <cfenv>
#include <fenv.h>

#include "gpu/acc_lapack.hpp"
#include "hamiltonian/inverse_overlap.hpp"
#include "preconditioner/ultrasoft_precond.hpp"
#include "nlcglib/overlap.hpp"

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t : int
{
    ground_state_new     = 0,
    ground_state_restart = 1,
    k_point_path         = 2
};

void json_output_common(json& dict__)
{
    dict__["git_hash"] = sirius::git_hash();
    //dict__["build_date"] = build_date;
    dict__["comm_world_size"] = Communicator::world().size();
    dict__["threads_per_rank"] = omp_get_max_threads();
}

void rewrite_relative_paths(json& dict__, fs::path const &working_directory = fs::current_path())
{
    // the json.unit_cell.atom_files[] dict might contain relative paths,
    // which should be relative to the json file. So better make them
    // absolute such that the simulation context does not have to be
    // aware of paths.
    if (!dict__.count("unit_cell"))
        return;

    auto &section = dict__["unit_cell"];

    if (!section.count("atom_files"))
        return;

    auto &atom_files = section["atom_files"];

    for (auto& label : atom_files.items()) {
        label.value() = working_directory / std::string(label.value());
    }
}

void wfct_alloc_copy_to_device(sddk::Wave_functions& wfct, int nspin)
{
    for (int ispn = 0; ispn < nspin; ++ispn) {
        int nbnd = wfct.pw_coeffs(ispn).prime().size(1);
        wfct.allocate(spin_range(ispn), memory_t::device);
        wfct.copy_to(spin_range(ispn),memory_t::device, 0, nbnd);
    }
}

void
wfct_copy_to_host(sddk::Wave_functions& wfct, int nspin)
{
    for (int ispn = 0; ispn < nspin; ++ispn) {
        int nbnd = wfct.pw_coeffs(ispn).prime().size(1);
        wfct.copy_to(spin_range(ispn), memory_t::host, 0, nbnd);
    }
}

nlohmann::json preprocess_json_input(std::string fname__)
{
    if (fname__.find("{") == std::string::npos) {
        // If it's a file, set the working directory to that file.
        auto json = utils::read_json_from_file(fname__);
        rewrite_relative_paths(json, fs::path{fname__}.parent_path());
        return json;
    } else {
        // Raw JSON input
        auto json = utils::read_json_from_string(fname__);
        rewrite_relative_paths(json);
        return json;
    }
}

std::unique_ptr<Simulation_context> create_sim_ctx(std::string fname__,
                                                   cmd_args const& args__)
{

    auto json = preprocess_json_input(fname__);

    auto ctx_ptr = std::make_unique<Simulation_context>(json.dump(), Communicator::world());
    Simulation_context& ctx = *ctx_ptr;

    auto& inp = ctx.parameters_input();
    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    ctx.import(args__);

    return ctx_ptr;
}

// sddk::matrix<double_complex> inner_beta(const Beta_projectors& beta, Simulation_context& ctx)
// {
//     // beta.make_generator()
//     auto generator = beta.make_generator();
//     int num_beta_chunks = beta.num_chunks();
//     auto bcoeffs_row = beta.prepare();
//     auto bcoeffs_col = beta.prepare();
//     auto linalg_t = ctx.blas_linalg_t();
//     auto preferred_memory = ctx.preferred_memory_t();

//     int size{beta.num_total_beta()};
//     // for (int ichunk = 0; ichunk < beta.num_chunks(); ++ichunk) {
//     //     size += beta.chunk(ichunk).num_beta_;
//     // }
//     std::cout << "beta total size: " << beta.num_total_beta() << "\n";

//     sddk::matrix<double_complex> out(size, size, preferred_memory);

//     for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
//         generator.generate(bcoeffs_row, ichunk);
//         for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {

//             generator.generate(bcoeffs_col, jchunk);

//             int m = bcoeffs_row.pw_coeffs_a.size(1);
//             int n = bcoeffs_col.pw_coeffs_a.size(1);
//             int k = bcoeffs_col.pw_coeffs_a.size(0);
//             int dest_row = bcoeffs_row.beta_chunk.offset_;
//             int dest_col = bcoeffs_col.beta_chunk.offset_;
//             double_complex calpha = double_complex(1);
//             double_complex cbeta  = double_complex(0);
//             const double_complex* A = bcoeffs_row.pw_coeffs_a.at(preferred_memory);
//             const double_complex* B = bcoeffs_col.pw_coeffs_a.at(preferred_memory);
//             double_complex* C = out.at(preferred_memory, dest_row, dest_col);
//             linalg(linalg_t).gemm('T', 'N', m, n, k, &calpha, A, bcoeffs_row.pw_coeffs_a.ld(), B, bcoeffs_col.pw_coeffs_a.ld(), &cbeta, C, out.ld());
//         }
//     }
//     return out;
// }
template <class T, int N>
std::enable_if_t<std::is_same<T, double_complex>::value>
dump_matrix(const mdarray<T, N>& arr, std::ostream& fout)
{
    static_assert(N == 2);

    for (int i = 0; i < static_cast<int>(arr.size(0)); ++i) {
        for (int j = 0; j < static_cast<int>(arr.size(1)); ++j) {
            if (arr(i, j).imag() > 0)
                fout << std::setprecision(10) << arr(i, j).real() << "+" << arr(i, j).imag() << "j\t ";
            else
                fout << std::setprecision(10) << arr(i, j).real() << "" << arr(i, j).imag() << "j\t ";
        }
        fout << "\n";
    }
}

template <class T, int N>
std::enable_if_t<std::is_same<T, double_complex>::value>
dump_matrix(const mdarray<T, N>& arr, const std::string& fname)
{
    static_assert(N == 2);

    std::ofstream fout;
    fout.open(fname.c_str());

    dump_matrix(arr, fout);
}


void check_overlap(Simulation_context& ctx,
                           task_t              task,
                           cmd_args const&     args,
                           int                 write_output)
{
    ctx.print_memory_usage(__FILE__, __LINE__);

    std::string ref_file = args.value<std::string>("test_against", "");
    /* don't write output if we compare against the reference calculation */

    K_point_set kset(ctx, ctx.parameters_input().ngridk_, ctx.parameters_input().shiftk_, ctx.use_symmetry());
    DFT_ground_state dft(kset);
    ctx.print_memory_usage(__FILE__, __LINE__);

    dft.initial_state();

    Hamiltonian0 H0(dft.potential());
    // todo hamiltonian_k

    std::cout << "------------------------------------------------------" << "\n";
    std::cout << "------------------------------------------------------" << "\n";
    std::cout << "------------------------------------------------------" << "\n";
    int nk = kset.spl_num_kpoints().local_size();
    // auto linalg_t = ctx.blas_linalg_t();
    // auto preferred_memory = ctx.preferred_memory_t();

    for (int ik_loc = 0; ik_loc < nk; ++ik_loc) {
        auto& kp   = *kset[kset.spl_num_kpoints(ik_loc)];
        auto Hk = H0(kp);
        // auto& q_op = H0.D();
        auto& beta_projectors = kp.beta_projectors();
        auto mat              = inner_beta(beta_projectors, ctx);

        for (int ispn = 0; ispn < ctx.num_spins(); ++ispn) {
            InverseS_k<double_complex> inversek(ctx, H0.Q(), beta_projectors, ispn);
            S_k<double_complex> sk(ctx, H0.Q(), beta_projectors, ispn);
            // usP.apply(kp.spinor_wave_functions().pw_coeffs(ispn).prime(), ispn);
            auto X = empty_like(kp.spinor_wave_functions().pw_coeffs(ispn).prime());
            sddk::copy(X, kp.spinor_wave_functions().pw_coeffs(ispn).prime());
            std::cout << "X.checksum: " << X.checksum() << "\n";
            if (is_device_memory(ctx.preferred_memory_t())) {
                X.allocate(memory_t::device);
                X.copy_to(memory_t::device);
            }
            // auto Y = inversek.apply(X);
            // auto X2 = inversek.apply_overlap(Y);
            std::cout << "-- sk.apply(X)" << "\n";
            // auto SX = inversek.apply_overlap(X);
            auto SX = sk.apply(X);

            if(is_device_memory(ctx.preferred_memory_t())) {
                X.copy_to(memory_t::host);
            }

            std::cout << "SX.checksum: " << SX.checksum() << "\n";
            {
                char fname[256];
                std::sprintf(fname, "SX_%d_%d.csv", ik_loc, ispn);
                dump_matrix(SX, fname);

                std::sprintf(fname, "X_%d_%d.csv", ik_loc, ispn);
                dump_matrix(X, fname);

            }

            std::cout << "X.dim: " << X.size(0) << " " << X.size(1) << "\n";

            // Hk.apply_h_s<double_complex>(sddk::spin_range(ispn), int N__, int n__, sddk::Wave_functions &phi__, sddk::Wave_functions *hphi__, sddk::Wave_functions *sphi__)
            {
                auto& mp = ctx.mem_pool(ctx.host_memory_t());
                int num_bands = ctx.num_bands();
                /* true if this is a non-collinear case */
                const bool nc_mag = (ctx.num_mag_dims() == 3);
                /* number of spin components, treated simultaneously
                 *   1 - in case of non-magnetic or collinear calculation
                 *   2 - in case of non-collinear calculation
                 */
                const int num_sc = nc_mag ? 2 : 1;
                /* S operator, applied to new Psi wave-functions */
                Wave_functions spsi(mp, kp.gkvec_partition(), num_bands, ctx.preferred_memory_t(), num_sc);
                /* Hamiltonian, applied to new Psi wave-functions, plus some extra space */
                Wave_functions hpsi(mp, kp.gkvec_partition(), num_bands, ctx.preferred_memory_t(), num_sc);

                // wfct allocate on device
                wfct_alloc_copy_to_device(spsi, num_sc);
                wfct_alloc_copy_to_device(hpsi, num_sc);

                spsi.pw_coeffs(ispn).remap_forward(num_bands, 0, &ctx.mem_pool(ctx.preferred_memory_t()));
                hpsi.pw_coeffs(ispn).remap_forward(num_bands, 0, &ctx.mem_pool(ctx.preferred_memory_t()));

                Hk.apply_h_s<double_complex>(sddk::spin_range(ispn), 0, num_bands, kp.spinor_wave_functions(), &hpsi, &spsi);

                // copy from device
                wfct_copy_to_host(spsi, num_sc);
                wfct_copy_to_host(hpsi, num_sc);

                // TODO: copy result for spsi
                auto SX_ref = empty_like(spsi.pw_coeffs(ispn).prime());
                sddk::copy(SX_ref, spsi.pw_coeffs(ispn).prime());
                std::cout << "SX_ref.checksum: " << SX_ref.checksum() << "\n";
                /// compare SX and SX_ref
                double l2err = 0;
                for (size_t i = 0; i < SX_ref.size(0); ++i) {
                    for (size_t j = 0; j < SX_ref.size(1); ++j) {
                        double loc = std::abs(SX_ref(i, j) - SX(i, j));
                        l2err += loc * loc;
                    }
                }
                std::printf("|SX-SX_ref|^2 at (k-point: (%d, %d)) => l2err = %.5g\n", ik_loc, ispn, l2err);
                {
                    char fname[256];
                    std::sprintf(fname, "SX_ref_%d_%d.csv", ik_loc, ispn);
                    dump_matrix(SX_ref, fname);

                    std::sprintf(fname, "X_ref_%d_%d.csv", ik_loc, ispn);
                    dump_matrix(kp.spinor_wave_functions().pw_coeffs(ispn).prime(), fname);
                }
            }
            // check that X2 and X are the same
            // double l2err = 0;
            // for (int i = 0; i < X.size(0); ++i) {
            //     for (int j = 0; j < X.size(1); ++j) {
            //         double loc = std::abs(X(i, j) - X2(i, j));
            //         l2err += loc*loc;
            //     }
            // }
            // std::printf("kp: (%d, %d), l2err = %.5g\n", ik_loc, ispn, l2err);
            {
                /* dump matrix */
                char fname[256];
                std::sprintf(fname, "Q_op_%d_%d.csv", ik_loc, ispn);
                auto Q_mat = H0.Q().get_matrix<double_complex>(ispn, memory_t::host);
                dump_matrix(Q_mat, fname);
            }
        }
    }
}


void check_inverse_overlap(Simulation_context& ctx,
                           task_t              task,
                           cmd_args const&     args,
                           int                 write_output)
{
    ctx.print_memory_usage(__FILE__, __LINE__);

    std::string ref_file = args.value<std::string>("test_against", "");
    /* don't write output if we compare against the reference calculation */

    K_point_set kset(ctx, ctx.parameters_input().ngridk_, ctx.parameters_input().shiftk_, ctx.use_symmetry());
    DFT_ground_state dft(kset);
    ctx.print_memory_usage(__FILE__, __LINE__);

    dft.initial_state();

    Hamiltonian0 H0(dft.potential());
    // todo hamiltonian_k

    std::cout << "------------------------------------------------------" << "\n";
    std::cout << "------------------------------------------------------" << "\n";
    std::cout << "------------------------------------------------------" << "\n";
    int nk = kset.spl_num_kpoints().local_size();
    // auto linalg_t = ctx.blas_linalg_t();
    // auto preferred_memory = ctx.preferred_memory_t();

    for (int ik_loc = 0; ik_loc < nk; ++ik_loc) {
        auto& kp   = *kset[kset.spl_num_kpoints(ik_loc)];
        auto Hk = H0(kp);
        // auto& q_op = H0.D();
        auto& beta_projectors = kp.beta_projectors();
        auto mat              = inner_beta(beta_projectors, ctx);

        for (int ispn = 0; ispn < ctx.num_spins(); ++ispn) {
            InverseS_k<double_complex> inversek(ctx, H0.Q(), beta_projectors, ispn);
            S_k<double_complex> sk(ctx, H0.Q(), beta_projectors, ispn);
            // usP.apply(kp.spinor_wave_functions().pw_coeffs(ispn).prime(), ispn);
            auto X = empty_like(kp.spinor_wave_functions().pw_coeffs(ispn).prime());
            sddk::copy(X, kp.spinor_wave_functions().pw_coeffs(ispn).prime());

            if(is_device_memory(ctx.preferred_memory_t())) {
                X.allocate(memory_t::device);
                X.copy_to(memory_t::device);
            }

            {
                auto SX = sk.apply(X);
                auto X2 = inversek.apply(SX);

                if (is_device_memory(ctx.preferred_memory_t())) {
                    X2.allocate(memory_t::host);
                    X2.copy_to(memory_t::host);
                }

                double l2err = 0;
                for (int i = 0; i < X.size(0); ++i) {
                    for (int j = 0; j < X.size(1); ++j) {
                        double loc = std::abs(X(i, j) - X2(i, j));
                        l2err += loc * loc;
                    }
                }
                std::printf("(%d,%d): |X-S^-1SX|, l2err = %.5g\n", ik_loc, ispn, l2err);
            }
            {
                auto Y = inversek.apply(X);
                auto X2 = sk.apply(Y);

                if (is_device_memory(ctx.preferred_memory_t())) {
                    X2.allocate(memory_t::host);
                    X2.copy_to(memory_t::host);
                }

                double l2err = 0;
                for (int i = 0; i < X.size(0); ++i) {
                    for (int j = 0; j < X.size(1); ++j) {
                        double loc = std::abs(X(i, j) - X2(i, j));
                        l2err += loc * loc;
                    }
                }
                std::printf("(%d,%d): |X-SS^-1X|, l2err = %.5g\n", ik_loc, ispn, l2err);
            }
        }
    }
}


void check_preconditioner(Simulation_context& ctx)
{
    K_point_set kset(ctx, ctx.parameters_input().ngridk_, ctx.parameters_input().shiftk_, ctx.use_symmetry());
    DFT_ground_state dft(kset);
    ctx.print_memory_usage(__FILE__, __LINE__);

    dft.initial_state();

    Hamiltonian0 H0(dft.potential());
    // todo hamiltonian_k

    std::cout << "------------------------------------------------------" << "\n";
    std::cout << "------------------------------------------------------" << "\n";
    std::cout << "------------------------------------------------------" << "\n";
    int nk = kset.spl_num_kpoints().local_size();
    // auto linalg_t = ctx.blas_linalg_t();
    // auto preferred_memory = ctx.preferred_memory_t();

    for (int ik_loc = 0; ik_loc < nk; ++ik_loc) {
        auto& kp   = *kset[kset.spl_num_kpoints(ik_loc)];
        auto Hk = H0(kp);
        // auto& q_op = H0.D();
        auto& beta_projectors = kp.beta_projectors();
        auto mat              = inner_beta(beta_projectors, ctx);

        for (int ispn = 0; ispn < ctx.num_spins(); ++ispn) {
            // usP.apply(kp.spinor_wave_functions().pw_coeffs(ispn).prime(), ispn);
            Ultrasoft_preconditioner<double_complex> precond(ctx, H0.Q(), ispn, kp.beta_projectors(), kp.gkvec());
            auto X = empty_like(kp.spinor_wave_functions().pw_coeffs(ispn).prime());
            sddk::copy(X, kp.spinor_wave_functions().pw_coeffs(ispn).prime());

            if (is_device_memory(ctx.preferred_memory_t())) {
                X.allocate(memory_t::device);
                X.copy_to(memory_t::device);
            }

            auto Y = precond.apply(X);
        }
    }

}

void check_non_local_operator(Simulation_context& ctx, cmd_args const& args)
{
    ctx.print_memory_usage(__FILE__, __LINE__);

    std::string ref_file = args.value<std::string>("test_against", "");
    /* don't write output if we compare against the reference calculation */

    K_point_set kset(ctx, ctx.parameters_input().ngridk_, ctx.parameters_input().shiftk_, ctx.use_symmetry());
    DFT_ground_state dft(kset);
    ctx.print_memory_usage(__FILE__, __LINE__);

    dft.initial_state();

    Hamiltonian0 H0(dft.potential());
    // todo hamiltonian_k
}

/// Run a task based on a command line input.
void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));

    /* get the input file name */
    auto fpath = args.value<fs::path>("input", "sirius.json");

    if (fs::is_directory(fpath)) {
        fpath /= "sirius.json";
    }

    if (!fs::exists(fpath)) {
        if (Communicator::world().rank() == 0) {
            std::printf("input file does not exist\n");
        }
        return;
    }

    auto fname = fpath.string();

    auto ctx = create_sim_ctx(fname, args);
    ctx->initialize();
    //if (ctx->full_potential()) {
    //    ctx->gk_cutoff(ctx->aw_cutoff() / ctx->unit_cell().min_mt_radius());
    //}
    // do_compute(*ctx, task, args, 1);
    std::cout << "--- check_overlap ---\n";
    check_overlap(*ctx, task, args, 1);
    std::cout << "--- check_inverse_overlap ---\n";
    check_inverse_overlap(*ctx, task, args, 1);
    std::cout << "--- check preconditioner ---\n";
    check_preconditioner(*ctx);
}

int main(int argn, char** argv)
{
    std::feclearexcept(FE_ALL_EXCEPT);
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--output=", "{string} output file name");
    args.register_key("--aiida_output", "write output for AiiDA");
    args.register_key("--test_against=", "{string} json file with reference values");
    args.register_key("--repeat_update=", "{int} number of times to repeat update()");
    args.register_key("--fpe", "enable check of floating-point exceptions using GNUC library");
    args.register_key("--control.processing_unit=", "");
    args.register_key("--control.verbosity=", "");
    args.register_key("--control.verification=", "");
    args.register_key("--control.mpi_grid_dims=","");
    args.register_key("--control.std_evp_solver_name=", "");
    args.register_key("--control.gen_evp_solver_name=", "");
    args.register_key("--control.fft_mode=", "");
    args.register_key("--control.memory_usage=", "");
    args.register_key("--parameters.ngridk=", "");
    args.register_key("--parameters.gamma_point=", "");
    args.register_key("--parameters.pw_cutoff=", "");
    args.register_key("--iterative_solver.orthogonalize=", "");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

#if defined(_GNU_SOURCE)
    if (args.exist("fpe")) {
        feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
    }
#endif

    sirius::initialize(1);

    run_tasks(args);

    sirius::finalize(1);

    // if (my_rank == 0)  {
    //     auto timing_result = ::utils::global_rtgraph_timer.process();
    //     std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
    //                                       rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median, rt_graph::Stat::Min,
    //                                       rt_graph::Stat::Max});
    //     std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
    //     ofs << timing_result.json();
    // }
    // if (std::fetestexcept(FE_DIVBYZERO)) {
    //     std::cout << "FE_DIVBYZERO exception\n";
    // }
    // if (std::fetestexcept(FE_INVALID)) {
    //     std::cout << "FE_INVALID exception\n";
    // }
    // if (std::fetestexcept(FE_UNDERFLOW)) {
    //     std::cout << "FE_UNDERFLOW exception\n";
    // }
    // if (std::fetestexcept(FE_OVERFLOW)) {
    //     std::cout << "FE_OVERFLOW exception\n";
    // }

    return 0;
}
