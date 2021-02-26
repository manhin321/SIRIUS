// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file beta_projectors_base.cpp
 *
 *  \brief Contains implementation of sirius::Beta_projectors_base class.
 */

#include "utils/env.hpp"
#include "beta_projectors_base.hpp"
#include "utils/profiler.hpp"

namespace sirius {

// template <typename T>
// std::enable_if_t<std::is_same<T, double_complex>::value>
// local_inner_aux_array(linalg_t linalg_t__, memory_t preferred_memory_t, const T* beta_pw_coeffs_a_ptr__, int nbeta__, int n__,
//                       int num_gkvec_loc__, const T* phi_ptr__, int phi_ld__, matrix<T>& beta_phi__, const T* _, const Communicator& comm)
// {
//     auto pp = utils::get_env<int>("SIRIUS_PRINT_PERFORMANCE");
//     if (pp && comm.rank() == 0) {
//         PROFILE_START("sirius::Beta_projectors_base::local_inner_aux");
//     }

//     const auto t1 = std::chrono::high_resolution_clock::now();
//     linalg(linalg_t__)
//         .gemm('C', 'N', nbeta__, n__, num_gkvec_loc__, &linalg_const<double_complex>::one(), beta_pw_coeffs_a_ptr__,
//               num_gkvec_loc__, phi_ptr__, phi_ld__, &linalg_const<double_complex>::zero(),
//               beta_phi__.at(preferred_memory_t), beta_phi__.ld());

//     if (pp && comm.rank() == 0) {
// #ifdef __GPU
//         if (linalg_t__ == linalg_t::gpublas) {
//             acc::sync_stream(stream_id(-1));
//         }
// #endif
//         std::chrono::duration<double> t = std::chrono::high_resolution_clock::now() - t1;
//         PROFILE_STOP("sirius::Beta_projectors_base::local_inner_aux");

//         std::printf("Beta_projectors_base::local_inner performance: %12.6f GFlops [m,n,k=%i %i %i, time=%f (sec)]\n",
//                     8e-9 * nbeta__ * n__ * num_gkvec_loc__ / t.count(), nbeta__, n__, num_gkvec_loc__, t.count());
//     }
// }


template <typename T>
std::enable_if_t<std::is_same<T, double_complex>::value>
local_inner_aux(linalg_t linalg_t__, memory_t preferred_memory_t, const T* beta_pw_coeffs_a_ptr__, int nbeta__, int n__,
                int num_gkvec_loc__, const T* phi_ptr__, int phi_ld__, matrix<T>& beta_phi__, const T* beta_pw_coeffs_a_g0_ptr__, const Communicator& comm)
{
    auto pp = utils::get_env<int>("SIRIUS_PRINT_PERFORMANCE");
    if (pp && comm.rank() == 0) {
        PROFILE_START("sirius::Beta_projectors_base::local_inner_aux");
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    linalg(linalg_t__)
        .gemm('C', 'N', nbeta__, n__, num_gkvec_loc__, &linalg_const<double_complex>::one(), beta_pw_coeffs_a_ptr__,
              num_gkvec_loc__, phi_ptr__, phi_ld__, &linalg_const<double_complex>::zero(),
              beta_phi__.at(preferred_memory_t), beta_phi__.ld());

    if (pp && comm.rank() == 0) {
#ifdef __GPU
        if (linalg_t__ == linalg_t::gpublas) {
            acc::sync_stream(stream_id(-1));
        }
#endif
        std::chrono::duration<double> t = std::chrono::high_resolution_clock::now() - t1;
        PROFILE_STOP("sirius::Beta_projectors_base::local_inner_aux");

        std::printf("Beta_projectors_base::local_inner performance: %12.6f GFlops [m,n,k=%i %i %i, time=%f (sec)]\n",
                    8e-9 * nbeta__ * n__ * num_gkvec_loc__ / t.count(), nbeta__, n__, num_gkvec_loc__, t.count());
    }
}

template <typename T>
std::enable_if_t<std::is_same<T, double>::value>
local_inner_aux(linalg_t linalg_t__, memory_t preferred_memory_t, const T* beta_pw_coeffs_a_ptr__, int nbeta__, int n__,
                int num_gkvec_loc__, const T* phi_ptr__, int phi_ld__, matrix<T>& beta_phi__, const T* beta_pw_coeffs_a_g0_ptr__, const Communicator& comm)
{
    linalg(linalg_t__).gemm('C', 'N', nbeta__, n__, 2 * num_gkvec_loc__,
                            &linalg_const<double>::two(),
                            beta_pw_coeffs_a_ptr__,
                            2 * num_gkvec_loc__,
                            phi_ptr__,
                            2 * phi_ld__,
                            &linalg_const<double>::zero(),
                            beta_phi__.at(preferred_memory_t), beta_phi__.ld());

    /* rank 0 has to do some extra work for Gamma-point case */
    if (comm.rank() == 0) {
        int incx{2 * num_gkvec_loc__};
        linalg_t la{linalg_t::none};
        /* both wave-functions and beta-projectors are on GPU */
        if (is_device_memory(preferred_memory_t)) {
            la = linalg_t::gpublas;
        } else { /* wave-functions are on CPU but the beta-projectors are in the memory of main device */
            la = linalg_t::blas;
            // TODO: what are the conditions makeing this a nullptr?
            if(beta_pw_coeffs_a_g0_ptr__ != nullptr) {
                beta_pw_coeffs_a_ptr__ = reinterpret_cast<T*>(&beta_pw_coeffs_a_g0_ptr__);
                incx                   = 2;
            }
        }
        linalg(la).ger(nbeta__, n__,
                       &linalg_const<double>::m_one(),
                       beta_pw_coeffs_a_ptr__, incx,
                       phi_ptr__,
                       2 * phi_ld__,
                       beta_phi__.at(preferred_memory_t), beta_phi__.ld());
    }
}

// template<typename T>
// __attribute_deprecated__ matrix<T>
// inner(Beta_projectors_base& beta__, int chunk__, Wave_functions& phi__, int ispn__, int idx0__, int n__)
// {
//     // PROFILE("sirius::Beta_projectors_base::inner");
//     // auto& ctx = beta__.ctx();

//     // assert(beta__.num_gkvec_loc() == phi__.pw_coeffs(ispn__).num_rows_loc());

//     // int nbeta = beta__.chunk(chunk__).num_beta_;

//     // matrix<T> beta_phi(nbeta, n__, ctx.mem_pool(ctx.host_memory_t()));

//     // /* location of the beta-projectors is always on the memory of the processing unit being used */
//     // T* pw_coeffs_a_ptr{nullptr};
//     // switch (ctx.processing_unit()) {
//     //     case device_t::CPU: {
//     //         pw_coeffs_a_ptr = reinterpret_cast<T*>(beta__.pw_coeffs_a().at(memory_t::host));
//     //         break;
//     //     }
//     //     case device_t::GPU: {
//     //         beta_phi.allocate(ctx.mem_pool(memory_t::device));
//     //         pw_coeffs_a_ptr = reinterpret_cast<T*>(beta__.pw_coeffs_a().at(memory_t::device));
//     //         break;
//     //     }
//     //     default:
//     //         assert(false);
//     // }
//     // auto& gkvec = beta__.gkvec();

//     // T* pw_coeffs_a0_ptr = nullptr;
//     // if (ctx.processing_unit() == device_t::GPU) {
//     //     auto& pw_coeffs_a0 = beta__.pw_coeffs_a0();
//     //     pw_coeffs_a0_ptr = reinterpret_cast<T*>(&pw_coeffs_a0(0));
//     // }

//     // local_inner_aux(ctx.blas_linalg_t(), phi__.preferred_memory_t(), reinterpret_cast<const T*>(pw_coeffs_a_ptr), nbeta, n__,
//     //                 beta__.num_gkvec_loc(), reinterpret_cast<T*>(phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__)),
//     //                 phi__.pw_coeffs(ispn__).prime().ld(), beta_phi, pw_coeffs_a0_ptr, gkvec.comm());

//     // /* copy to host in MPI sequential or parallel case */
//     // if (is_device_memory(ctx.preferred_memory_t())) {
//     //     beta_phi.copy_to(memory_t::host);
//     // }

//     // /* in parallel case do a reduction */
//     // if (gkvec.comm().size() > 1) {
//     //     PROFILE("sirius::Beta_projectors_base::inner|comm");
//     //     /* MPI reduction on the host */
//     //     gkvec.comm().allreduce(beta_phi.at(memory_t::host), static_cast<int>(beta_phi.size()));
//     // }

//     // switch (ctx.processing_unit()) {
//     //     case device_t::GPU: {
//     //         /* copy back to device */
//     //         if (gkvec.comm().size() > 1 || is_host_memory(ctx.preferred_memory_t())) {
//     //             beta_phi.copy_to(memory_t::device);
//     //         }
//     //         break;
//     //     }
//     //     case device_t::CPU: break;
//     // }

//     // return beta_phi;
// }

template <typename T>
matrix<T>
inner(linalg_t linalg, device_t processing_unit, memory_t preferred_memory, memory_pool& mempool,
      const beta_projectors_coeffs_t& beta_projector_coeffs, Wave_functions& phi__, int ispn__, int idx0__, int n__)
{
    PROFILE("sirius::Beta_projectors_base::inner");
    // auto& ctx = beta__.ctx();
    assert(mempool.memory_type() == memory_t::host);

    // int nbeta = beta__.chunk(chunk__).num_beta_;
    int nbeta         = beta_projector_coeffs.beta_chunk.num_beta_;
    int num_gkvec_loc = beta_projector_coeffs.pw_coeffs_a.size(0);
    assert(num_gkvec_loc == phi__.pw_coeffs(ispn__).num_rows_loc());

    auto& comm        = beta_projector_coeffs.comm;
    auto& pw_coeffs_a = beta_projector_coeffs.pw_coeffs_a;

    matrix<T> beta_phi(nbeta, n__, mempool);

    /* location of the beta-projectors is always on the memory of the processing unit being used */
    const T* pw_coeffs_a_ptr{nullptr};
    switch (processing_unit) {
        case device_t::CPU: {
            pw_coeffs_a_ptr = reinterpret_cast<const T*>(pw_coeffs_a.at(memory_t::host));
            break;
        }
        case device_t::GPU: {
            beta_phi.allocate(memory_t::device); // TODO use gpu memory pool
            pw_coeffs_a_ptr = reinterpret_cast<const T*>(pw_coeffs_a.at(memory_t::device));
            break;
        }
        default:
            assert(false);
    }

    const T* pw_coeffs_a_g0_ptr = nullptr;
    if (processing_unit == device_t::GPU) {
        auto& pw_coeffs_a_g0 = beta_projector_coeffs.pw_coeffs_a_g0;
        pw_coeffs_a_g0_ptr   = reinterpret_cast<const T*>(&pw_coeffs_a_g0(0));
    }

    local_inner_aux(linalg, phi__.preferred_memory_t(), reinterpret_cast<const T*>(pw_coeffs_a_ptr), nbeta, n__,
                    num_gkvec_loc,
                    reinterpret_cast<T*>(phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__)),
                    phi__.pw_coeffs(ispn__).prime().ld(), beta_phi, pw_coeffs_a_g0_ptr, comm);

    /* copy to host in MPI sequential or parallel case */
    if (is_device_memory(preferred_memory)) {
        beta_phi.copy_to(memory_t::host);
    }

    /* in parallel case do a reduction */
    if (comm.size() > 1) {
        PROFILE("sirius::Beta_projectors_base::inner|comm");
        /* MPI reduction on the host */
        comm.allreduce(beta_phi.at(memory_t::host), static_cast<int>(beta_phi.size()));
    }

    switch (processing_unit) {
        case device_t::GPU: {
            /* copy back to device */
            if (comm.size() > 1 || is_host_memory(preferred_memory)) {
                beta_phi.copy_to(memory_t::device);
            }
            break;
        }
        case device_t::CPU:
            break;
    }

    return beta_phi;
}

template <typename T>
matrix<T>
inner(linalg_t linalg, device_t processing_unit, memory_t preferred_memory, memory_pool& mempool,
      const beta_projectors_coeffs_t& beta_projector_coeffs, const matrix<double_complex>& other, int idx0__, int n__, memory_t target_memory)
{
    PROFILE("sirius::Beta_projectors_base::inner");
    // auto& ctx = beta__.ctx();
    assert(mempool.memory_type() == memory_t::host);

    // int nbeta = beta__.chunk(chunk__).num_beta_;
    int nbeta         = beta_projector_coeffs.beta_chunk.num_beta_;
    int num_gkvec_loc = beta_projector_coeffs.pw_coeffs_a.size(0);
    assert(num_gkvec_loc == other.size(0));

    auto& comm        = beta_projector_coeffs.comm;
    auto& pw_coeffs_a = beta_projector_coeffs.pw_coeffs_a;

    matrix<T> beta_phi(nbeta, n__, mempool);

    /* location of the beta-projectors is always on the memory of the processing unit being used */
    const T* pw_coeffs_a_ptr{nullptr};
    switch (processing_unit) {
        case device_t::CPU: {
            pw_coeffs_a_ptr = reinterpret_cast<const T*>(pw_coeffs_a.at(memory_t::host));
            break;
        }
        case device_t::GPU: {
            beta_phi.allocate(memory_t::device); // TODO: use gpu memory pool
            pw_coeffs_a_ptr = reinterpret_cast<const T*>(pw_coeffs_a.at(memory_t::device));
            break;
        }
        default:
            assert(false);
    }

    const T* pw_coeffs_a_g0_ptr = nullptr;
    if (processing_unit == device_t::GPU) {
        auto& pw_coeffs_a_g0 = beta_projector_coeffs.pw_coeffs_a_g0;
        pw_coeffs_a_g0_ptr   = reinterpret_cast<const T*>(&pw_coeffs_a_g0(0));
    }

    local_inner_aux(linalg, preferred_memory, reinterpret_cast<const T*>(pw_coeffs_a_ptr), nbeta, n__, num_gkvec_loc,
                    reinterpret_cast<const T*>(other.at(preferred_memory, 0, idx0__)), other.ld(), beta_phi,
                    pw_coeffs_a_g0_ptr, comm);

    /* copy to host in MPI sequential or parallel case */
    if (is_device_memory(preferred_memory) && (target_memory == memory_t::none || is_host_memory(target_memory))) {
        beta_phi.copy_to(memory_t::host);
    }

    /* in parallel case do a reduction */
    if (comm.size() > 1) {
        PROFILE("sirius::Beta_projectors_base::inner|comm");
        /* MPI reduction on the host */
        comm.allreduce(beta_phi.at(memory_t::host), static_cast<int>(beta_phi.size()));
    }

    switch (processing_unit) {
        case device_t::GPU: {
            /* copy back to device */
            if (comm.size() > 1 || is_host_memory(preferred_memory)) {
                beta_phi.copy_to(memory_t::device);
            }
            break;
        }
        case device_t::CPU:
            break;
    }

    return beta_phi;
}



void Beta_projectors_base::split_in_chunks()
{
    auto& uc = ctx_.unit_cell();

    if (uc.mt_lo_basis_size() == 0) {
        /* no beta projectors at all */
        beta_chunks_ = std::vector<beta_chunk_t>(0);
        num_beta_t_ = 0;
        max_num_beta_ = 0;
        return;
    }

    /* initial chunk size */
    int chunk_size = std::min(uc.num_atoms(), ctx_.cfg().control().beta_chunk_size());
    /* maximum number of chunks */
    int num_chunks = uc.num_atoms() / chunk_size + std::min(1, uc.num_atoms() % chunk_size);
    /* final maximum chunk size */
    chunk_size = uc.num_atoms() / num_chunks + std::min(1, uc.num_atoms() % num_chunks);

    int offset_in_beta_gk{0};
    beta_chunks_ = std::vector<beta_chunk_t>(num_chunks);

    for (int ib = 0; ib < num_chunks; ib++) {
        /* number of atoms in this chunk */
        int na = std::min(uc.num_atoms(), (ib + 1) * chunk_size) - ib * chunk_size;
        beta_chunks_[ib].num_atoms_ = na;
        beta_chunks_[ib].desc_      = mdarray<int, 2>(4, na);
        beta_chunks_[ib].atom_pos_  = mdarray<double, 2>(3, na);

        int num_beta{0};
        for (int i = 0; i < na; i++) {
            /* global index of atom by local index and chunk */
            int ia     = ib * chunk_size + i;
            auto pos   = uc.atom(ia).position();
            auto& type = uc.atom(ia).type();
            /* atom fractional coordinates */
            for (int x: {0, 1, 2}) {
                beta_chunks_[ib].atom_pos_(x, i) = pos[x];
            }
            /* number of beta functions for atom */
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::nbf), i) = type.mt_basis_size();
            /* offset in beta_gk*/
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::offset), i) = num_beta;
            /* offset in beta_gk_t */
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::offset_t), i) = type.offset_lo();
            /* global index of atom */
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::ia), i) = ia;

            num_beta += type.mt_basis_size();
        }
        /* number of beta-projectors in this chunk */
        beta_chunks_[ib].num_beta_ = num_beta;
        beta_chunks_[ib].offset_ = offset_in_beta_gk;
        offset_in_beta_gk += num_beta;

        if (ctx_.processing_unit() == device_t::GPU) {
            beta_chunks_[ib].desc_.allocate(memory_t::device).copy_to(memory_t::device);
            beta_chunks_[ib].atom_pos_.allocate(memory_t::device).copy_to(memory_t::device);
        }
    }
    num_total_beta_ = offset_in_beta_gk;

    max_num_beta_ = 0;
    for (auto& e: beta_chunks_) {
        max_num_beta_ = std::max(max_num_beta_, e.num_beta_);
    }

    num_beta_t_ = 0;
    for (int iat = 0; iat < uc.num_atom_types(); iat++) {
        num_beta_t_ += uc.atom_type(iat).mt_lo_basis_size();
    }
}


Beta_projectors_base::Beta_projectors_base(Simulation_context& ctx__, Gvec const& gkvec__,
                                           std::vector<int> const& igk__, int N__)
    : ctx_(ctx__)
    , gkvec_(gkvec__)
    , igk_(igk__)
    , N_(N__)
{
    split_in_chunks();

    if (!num_beta_t()) {
        return;
    }

    /* allocate memory */
    pw_coeffs_t_ = mdarray<double_complex, 3>(num_gkvec_loc(), num_beta_t(), N__, memory_t::host, "pw_coeffs_t_");

    if (ctx_.processing_unit() == device_t::GPU) {
        gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc());
        gkvec_coord_.allocate(memory_t::device);
        /* copy G+k vectors */
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
            auto vgk = gkvec_.gkvec(igk_[igk_loc]);
            for (auto x: {0, 1, 2}) {
                gkvec_coord_(x, igk_loc) = vgk[x];
            }
        }
        gkvec_coord_.copy_to(memory_t::device);
    }
}


// template matrix<double> inner<double>(Beta_projectors_base&, int, Wave_functions&, int, int, int);
// template matrix<double_complex> inner<double_complex>(Beta_projectors_base&, int, Wave_functions&, int, int, int);

// template <typename T>
// __attribute_deprecated__ matrix<T>
// Beta_projectors_base::inner(int chunk__, Wave_functions& phi__, int ispn__, int idx0__, int n__)
// {
//     PROFILE("sirius::Beta_projectors_base::inner");

//     assert(num_gkvec_loc() == phi__.pw_coeffs(ispn__).num_rows_loc());

//     int nbeta = chunk(chunk__).num_beta_;

//     matrix<T> beta_phi(nbeta, n__, ctx_.mem_pool(ctx_.host_memory_t()));

//     /* location of the beta-projectors is always on the memory of the processing unit being used */
//     T* pw_coeffs_a_ptr{nullptr};
//     switch (ctx_.processing_unit()) {
//         case device_t::CPU: {
//             pw_coeffs_a_ptr = reinterpret_cast<T*>(pw_coeffs_a().at(memory_t::host));
//             break;
//         }
//         case device_t::GPU: {
//             beta_phi.allocate(ctx_.mem_pool(memory_t::device));
//             pw_coeffs_a_ptr = reinterpret_cast<T*>(pw_coeffs_a().at(memory_t::device));
//             break;
//         }
//     }

//     local_inner_aux(pw_coeffs_a_ptr, nbeta, phi__, ispn__, idx0__, n__, beta_phi);

//     /* copy to host in MPI sequential or parallel case */
//     if (is_device_memory(ctx_.preferred_memory_t())) {
//         beta_phi.copy_to(memory_t::host);
//     }

//     /* in parallel case do a reduction */
//     if (gkvec_.comm().size() > 1) {
//         PROFILE("sirius::Beta_projectors_base::inner|comm");
//         /* MPI reduction on the host */
//         gkvec_.comm().allreduce(beta_phi.at(memory_t::host), static_cast<int>(beta_phi.size()));
//     }

//     switch (ctx_.processing_unit()) {
//         case device_t::GPU: {
//             /* copy back to device */
//             if (gkvec_.comm().size() > 1 || is_host_memory(ctx_.preferred_memory_t())) {
//                 beta_phi.copy_to(memory_t::device);
//             }
//             break;
//         }
//         case device_t::CPU: break;
//     }

//     return beta_phi;
// }

__attribute_deprecated__
void Beta_projectors_base::generate(beta_projectors_coeffs_t& out, int ichunk__, int j__) const
{

    PROFILE("sirius::Beta_projectors_base::generate");

    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            #pragma omp parallel for
            for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
                int ia = chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

                double phase = twopi * dot(gkvec_.vk(), ctx_.unit_cell().atom(ia).position());
                double_complex phase_k = std::exp(double_complex(0.0, phase));

                std::vector<double_complex> phase_gk(num_gkvec_loc());
                for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                    auto G = gkvec_.gvec(igk_[igk_loc]);
                    /* total phase e^{-i(G+k)r_{\alpha}} */
                    phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
                }
                for (int xi = 0; xi < chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
                    for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                        out.pw_coeffs_a(igk_loc, chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
                            pw_coeffs_t_(igk_loc, chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi, j__) * phase_gk[igk_loc];
                    }
                }
            }
            break;
        }
        case device_t::GPU: {
#if defined(SIRIUS_GPU)
            auto& desc = chunk(ichunk__).desc_;
            create_beta_gk_gpu(chunk(ichunk__).num_atoms_,
                               num_gkvec_loc(),
                               desc.at(memory_t::device),
                               pw_coeffs_t_.at(memory_t::device, 0, 0, j__),
                               gkvec_coord_.at(memory_t::device),
                               chunk(ichunk__).atom_pos_.at(memory_t::device),
                               out.pw_coeffs_a.at(memory_t::device));
#endif
            /* wave-functions are on CPU but the beta-projectors are on GPU */
            if (gkvec_.comm().rank() == 0 && is_host_memory(ctx_.preferred_memory_t())) {
                /* make beta-projectors for G=0 on the CPU */
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
                    for (int xi = 0; xi < chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
                        out.pw_coeffs_a_g0(chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
                            pw_coeffs_t_(0, chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi, j__);
                    }
                }
            }
            break;
        }
    }
}

beta_projectors_coeffs_t Beta_projectors_base::prepare(memory_t pm) const
{
    PROFILE("sirius::Beta_projectors_base::prepare");

    if (max_num_beta() == 0) {
        // return an empty struct
        return beta_projectors_coeffs_t{};
    }

    beta_projectors_coeffs_t beta_storage;

    beta_storage.comm = gkvec_.comm().duplicate();

    device_t pu;
    switch (pm) {
        case memory_t::none: {
            pu = ctx_.processing_unit();
            break;
        }
        case memory_t::host: {
            pu = device_t::CPU;
            break;
        }
        case memory_t::device: {
            pu = device_t::GPU;
            break;
        }
        default:
            throw std::runtime_error("invalid memory type in Beta_projectors_base::prepare");
            break;
    }

    if (pu == device_t::GPU) {
        beta_storage.__pw_coeffs_a_buffer =
            matrix<double_complex>(num_gkvec_loc(), max_num_beta(), ctx_.mem_pool(memory_t::device));
        beta_storage.__pw_coeffs_a_g0_buffer = mdarray<double_complex, 1>(max_num_beta(), ctx_.mem_pool(memory_t::host));
        beta_storage.__pw_coeffs_a_g0_buffer.allocate(ctx_.mem_pool(memory_t::device));
    }

    return beta_storage;
}

__attribute_deprecated__
void Beta_projectors_base::dismiss()
{
    PROFILE("sirius::Beta_projectors_base::dismiss");

    if (ctx_.processing_unit() == device_t::GPU && reallocate_pw_coeffs_t_on_gpu_) {
        pw_coeffs_t_.deallocate(memory_t::device);
    }
    pw_coeffs_a_.deallocate(memory_t::device);
    pw_coeffs_a_g0_.deallocate(memory_t::device);
}

template <>
__attribute_deprecated__
void Beta_projectors_base::local_inner_aux<double_complex>(double_complex* beta_pw_coeffs_a_ptr__, int nbeta__,
                                                      Wave_functions& phi__, int ispn__, int idx0__, int n__,
                                                      matrix<double_complex>& beta_phi__) const
{
    auto pp = utils::get_env<int>("SIRIUS_PRINT_PERFORMANCE");
    if (pp && gkvec_.comm().rank() == 0) {
        PROFILE_START("sirius::Beta_projectors_base::local_inner_aux");
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    linalg(ctx_.blas_linalg_t()).gemm('C', 'N', nbeta__, n__, num_gkvec_loc(),
                                      &linalg_const<double_complex>::one(),
                                      beta_pw_coeffs_a_ptr__,
                                      num_gkvec_loc(),
                                      phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__),
                                      phi__.pw_coeffs(ispn__).prime().ld(),
                                      &linalg_const<double_complex>::zero(),
                                      beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());

    if (pp && gkvec_.comm().rank() == 0) {
#ifdef SIRIUS_GPU
        if (ctx_.blas_linalg_t() == linalg_t::gpublas) {
            acc::sync_stream(stream_id(-1));
        }
#endif
        std::chrono::duration<double> t = std::chrono::high_resolution_clock::now() - t1;
        PROFILE_STOP("sirius::Beta_projectors_base::local_inner_aux");
        std::printf("Beta_projectors_base::local_inner performance: %12.6f GFlops [m,n,k=%i %i %i, time=%f (sec)]\n",
               8e-9 * nbeta__ * n__ * num_gkvec_loc() / t.count(), nbeta__, n__, num_gkvec_loc(), t.count());
    }
}

template <>
__attribute_deprecated__
void Beta_projectors_base::local_inner_aux<double>(double* beta_pw_coeffs_a_ptr__, int nbeta__, Wave_functions& phi__,
                                              int ispn__, int idx0__, int n__, matrix<double>& beta_phi__) const
{
    linalg(ctx_.blas_linalg_t()).gemm('C', 'N', nbeta__, n__, 2 * num_gkvec_loc(),
                                       &linalg_const<double>::two(),
                                       beta_pw_coeffs_a_ptr__,
                                       2 * num_gkvec_loc(),
                                       reinterpret_cast<double const*>(phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__)),
                                       2 * phi__.pw_coeffs(ispn__).prime().ld(),
                                       &linalg_const<double>::zero(),
                                       beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());

    /* rank 0 has to do some extra work for Gamma-point case */
    if (gkvec_.comm().rank() == 0) {
        int incx{2 * num_gkvec_loc()};
        linalg_t la{linalg_t::none};
        /* both wave-functions and beta-projectors are on GPU */
        if (is_device_memory(ctx_.preferred_memory_t())) {
            la = linalg_t::gpublas;
        } else { /* wave-functions are on CPU but the beta-projectors are in the memory of main device */
            la = linalg_t::blas;
            switch (ctx_.processing_unit()) {
                case device_t::GPU: {
                    beta_pw_coeffs_a_ptr__ = reinterpret_cast<double*>(const_cast<double_complex*>(&pw_coeffs_a_g0_(0)));
                    incx = 2;
                    break;
                }
                case device_t::CPU: break;
            }
        }
        linalg(la).ger(nbeta__, n__,
                       &linalg_const<double>::m_one(),
                       beta_pw_coeffs_a_ptr__, incx,
                       reinterpret_cast<double*>(phi__.pw_coeffs(ispn__).prime().at(phi__.preferred_memory_t(), 0, idx0__)),
                       2 * phi__.pw_coeffs(ispn__).prime().ld(),
                       beta_phi__.at(ctx_.preferred_memory_t()), beta_phi__.ld());
    }
}


void
beta_projectors_generate_cpu(matrix<double_complex>& pw_coeffs_a, const mdarray<double_complex, 3>& pw_coeffs_t,
                             int ichunk__, int j__, const beta_chunk_t& beta_chunk, const Simulation_context& ctx,
                             const Gvec& gkvec, const std::vector<int>& igk__)
{
    PROFILE("sirius::Beta_projectors_base::generate");

    int num_gkvec_loc = igk__.size();
    auto& unit_cell = ctx.unit_cell();

#pragma omp parallel for
    for (int i = 0; i < beta_chunk.num_atoms_; i++) {
        int ia = beta_chunk.desc_(static_cast<int>(beta_desc_idx::ia), i);

        double phase           = twopi * dot(gkvec.vk(), unit_cell.atom(ia).position());
        double_complex phase_k = std::exp(double_complex(0.0, phase));

        std::vector<double_complex> phase_gk(num_gkvec_loc);
        for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
            auto G = gkvec.gvec(igk__[igk_loc]);
            /* total phase e^{-i(G+k)r_{\alpha}} */
            phase_gk[igk_loc] = std::conj(ctx.gvec_phase_factor(G, ia) * phase_k);
        }
        for (int xi = 0; xi < beta_chunk.desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
            for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                pw_coeffs_a(igk_loc, beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
                    pw_coeffs_t(igk_loc, beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi, j__) *
                    phase_gk[igk_loc];
            }
        }
    }
}

void
beta_projectors_generate_gpu(beta_projectors_coeffs_t& out, const mdarray<double_complex, 3>& pw_coeffs_t_device,
                             const mdarray<double_complex, 3>& pw_coeffs_t_host,
                             const Simulation_context& ctx,
                             const Gvec& gkvec,
                             const mdarray<double, 2>& gkvec_coord_, const beta_chunk_t& beta_chunk,
                             const std::vector<int>& igk__, int j__)
{
    int num_gkvec_loc = igk__.size();
    PROFILE("sirius::Beta_projectors_base::generate");
#if defined(__GPU)
    auto& desc = beta_chunk.desc_;
    create_beta_gk_gpu(beta_chunk.num_atoms_, num_gkvec_loc, desc.at(memory_t::device),
                       pw_coeffs_t_device.at(memory_t::device, 0, 0, j__), gkvec_coord_.at(memory_t::device),
                       beta_chunk.atom_pos_.at(memory_t::device), out.pw_coeffs_a.at(memory_t::device));
#endif
    /* wave-functions are on CPU but the beta-projectors are on GPU */
    if (gkvec.comm().rank() == 0 && is_host_memory(ctx.preferred_memory_t())) {
/* make beta-projectors for G=0 on the CPU */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < beta_chunk.num_atoms_; i++) {
            for (int xi = 0; xi < beta_chunk.desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
                out.pw_coeffs_a_g0(beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
                    pw_coeffs_t_host(0, beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi, j__);
            }
        }
    }
}

void Beta_projector_generator::generate(beta_projectors_coeffs_t &out, int ichunk__) const
{
    PROFILE("sirius::Beta_projectors_base::generate");

    int j__        = 0;
    out.beta_chunk = beta_chunks_.at(ichunk__);

    auto num_beta = out.beta_chunk.num_beta_;
    auto gk_size  = igk_.size();

    switch (processing_unit_) {
        case device_t::CPU: {
            out.pw_coeffs_a = matrix<double_complex>(const_cast<double_complex*>(&beta_pw_all_atoms_(0, beta_chunks_[ichunk__].offset_)), igk_.size(),
                                                     beta_chunks_[ichunk__].num_beta_);

            break;
        }
        case device_t::GPU: {
            out.pw_coeffs_a =
                sddk::matrix<double_complex>(nullptr, out.__pw_coeffs_a_buffer.device_data(), gk_size, num_beta);
            out.pw_coeffs_a_g0 =
                sddk::mdarray<double_complex, 1>(nullptr, out.__pw_coeffs_a_g0_buffer.device_data(), num_beta);

            beta_projectors_generate_gpu(out, pw_coeffs_t_device_, pw_coeffs_t_host_, ctx_, gkvec_, gkvec_coord_, beta_chunks_[ichunk__], igk_, j__);
            break;
        }
    }
}

void
Beta_projector_generator::generate(beta_projectors_coeffs_t& out, int ichunk__, int j__) const
{
    PROFILE("sirius::Beta_projectors_base::generate");

    out.beta_chunk = beta_chunks_.at(ichunk__);

    auto num_beta = out.beta_chunk.num_beta_;
    auto gk_size = igk_.size();

    switch (processing_unit_) {
        case device_t::CPU: {
            //allocate pw_coeffs_a
            out.pw_coeffs_a = sddk::matrix<double_complex>(gk_size, num_beta, ctx_.mem_pool(memory_t::host));
            out.pw_coeffs_a_g0 = sddk::mdarray<double_complex, 1>(num_beta, ctx_.mem_pool(memory_t::host));
            beta_projectors_generate_cpu(out.pw_coeffs_a, pw_coeffs_t_host_, ichunk__, j__, beta_chunks_[ichunk__], ctx_, gkvec_, igk_);
            break;
        }
        case device_t::GPU: {
            // view of internal buffer with correct number of cols (= num_beta)
            out.pw_coeffs_a =
                sddk::matrix<double_complex>(nullptr, out.__pw_coeffs_a_buffer.device_data(), gk_size, num_beta);
            out.pw_coeffs_a_g0 = sddk::mdarray<double_complex, 1>(nullptr, out.__pw_coeffs_a_g0_buffer.device_data(), num_beta);

            beta_projectors_generate_gpu(out, pw_coeffs_t_device_, pw_coeffs_t_host_, ctx_, gkvec_, gkvec_coord_,
                                         beta_chunks_[ichunk__], igk_, j__);
            break;
        }
    }
}

/// inner product <beta|beta>, result in preferred_memory
sddk::matrix<double_complex> inner_beta(const Beta_projectors_base& beta, const Simulation_context& ctx)
{
    if (beta.comm().size() == 1) {
        auto generator        = beta.make_generator();
        int num_beta_chunks   = beta.num_chunks();
        auto bcoeffs_row      = beta.prepare();
        auto bcoeffs_col      = beta.prepare();
        auto linalg_t         = ctx.blas_linalg_t();
        auto preferred_memory = ctx.preferred_memory_t();

        int size{beta.num_total_beta()};

        sddk::matrix<double_complex> out(size, size, preferred_memory);

        std::cout << "num_beta_chunks: " << num_beta_chunks << "\n";

        double_complex one  = double_complex(1);
        double_complex zero = double_complex(0);

        for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
            generator.generate(bcoeffs_row, ichunk);
            for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {

                generator.generate(bcoeffs_col, jchunk);

                int m                   = bcoeffs_row.beta_chunk.num_beta_;
                int n                   = bcoeffs_col.beta_chunk.num_beta_;
                int k                   = bcoeffs_col.pw_coeffs_a.size(0);
                int dest_row            = bcoeffs_row.beta_chunk.offset_;
                int dest_col            = bcoeffs_col.beta_chunk.offset_;
                const double_complex* A = bcoeffs_row.pw_coeffs_a.at(preferred_memory);
                const double_complex* B = bcoeffs_col.pw_coeffs_a.at(preferred_memory);
                double_complex* C       = out.at(preferred_memory, dest_row, dest_col);
                linalg(linalg_t).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B,
                                      bcoeffs_col.pw_coeffs_a.ld(), &zero, C, out.ld());
            }
        }
        return out;
    } else {
        throw std::runtime_error("distributed case not yet implemented: " + std::string(__func__) + " in " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

// template matrix<double> Beta_projectors_base::inner<double>(int chunk__, Wave_functions& phi__, int ispn__, int idx0__,
//                                                             int n__);

// template matrix<double_complex> Beta_projectors_base::inner<double_complex>(int chunk__, Wave_functions& phi__,
//                                                                             int ispn__, int idx0__, int n__);

template matrix<double>
inner<double>(linalg_t, device_t, memory_t, memory_pool&,
              const beta_projectors_coeffs_t&, Wave_functions&, int,
              int, int);

template matrix<double_complex> inner<double_complex>(linalg_t, device_t, memory_t, memory_pool&,
                                                      const beta_projectors_coeffs_t&, Wave_functions&,
                                                      int, int, int);

template matrix<double_complex>
inner(linalg_t, device_t, memory_t, memory_pool&,
      const beta_projectors_coeffs_t&, const matrix<double_complex>&, int, int, memory_t);

template matrix<double> inner(linalg_t, device_t, memory_t, memory_pool&, const beta_projectors_coeffs_t&,
                              const matrix<double_complex>&, int, int, memory_t);

} // namespace sirius
