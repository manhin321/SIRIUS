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

/** \file non_local_operator.hpp
 *
 *  \brief Contains declaration of sirius::Non_local_operator class.
 */

#ifndef __NON_LOCAL_OPERATOR_HPP__
#define __NON_LOCAL_OPERATOR_HPP__

#include "SDDK/omp.hpp"
#include "SDDK/memory.hpp"
#include "SDDK/type_definition.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "context/simulation_context.hpp"
#include "hubbard/hubbard_matrix.hpp"

namespace sddk {
template <typename T>
class Wave_functions;
class spin_range;
};

namespace sirius {
/* forward declaration */
template <typename T>
class Beta_projectors;
template <typename T>
class Beta_projectors_base;

/// Non-local part of the Hamiltonian and S-operator in the pseudopotential method.
template <typename T>
class Non_local_operator
{
  protected:
    Simulation_context const& ctx_;

    sddk::device_t pu_;

    int packed_mtrx_size_;

    sddk::mdarray<int, 1> packed_mtrx_offset_;

    /// Non-local operator matrix.
    sddk::mdarray<T, 3> op_;

    bool is_null_{false};

    /// True if the operator is diagonal in spin.
    bool is_diag_{true};

    /* copy assignment operrator is forbidden */
    Non_local_operator<T>& operator=(Non_local_operator<T> const& src) = delete;
    /* copy constructor is forbidden */
    Non_local_operator(Non_local_operator<T> const& src) = delete;

  public:
    /// Constructor.
    Non_local_operator(Simulation_context const& ctx__);

    /// Apply chunk of beta-projectors to all wave functions.
    template <typename F, std::enable_if_t<std::is_same<T, F>::value, bool> = true>
    void apply(int chunk__, int ispn_block__, Wave_functions<T>& op_phi__, int idx0__, int n__,
               Beta_projectors_base<T>& beta__, matrix<F>& beta_phi__)
    {
        PROFILE("sirius::Non_local_operator::apply");

        if (is_null_) {
            return;
        }

        auto& beta_gk     = beta__.pw_coeffs_a();
        int num_gkvec_loc = beta__.num_gkvec_loc();
        int nbeta         = beta__.chunk(chunk__).num_beta_;

        /* setup linear algebra parameters */
        memory_t mem{memory_t::none};
        linalg_t la{linalg_t::none};
        switch (pu_) {
            case device_t::CPU: {
                mem = memory_t::host;
                la  = linalg_t::blas;
                break;
            }
            case device_t::GPU: {
                mem = memory_t::device;
                la  = linalg_t::gpublas;
                break;
            }
        }

        auto work = mdarray<T, 1>(nbeta * n__, ctx_.mem_pool(mem));

        /* compute O * <beta|phi> for atoms in a chunk */
#pragma omp parallel for
        for (int i = 0; i < beta__.chunk(chunk__).num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i);
            int offs = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::offset), i);
            int ia   = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

            if (nbf == 0) {
                continue;
            }
            linalg(la).gemm('N', 'N', nbf, n__, nbf, &linalg_const<T>::one(),
                            op_.at(mem, 0, packed_mtrx_offset_(ia), ispn_block__), nbf, beta_phi__.at(mem, offs, 0),
                            beta_phi__.ld(), &linalg_const<T>::zero(), work.at(mem, offs), nbeta,
                            stream_id(omp_get_thread_num()));
        }
        switch (pu_) {
            case device_t::GPU: {
                /* wait for previous zgemms */
#pragma omp parallel
                acc::sync_stream(stream_id(omp_get_thread_num()));
                break;
            }
            case device_t::CPU: {
                break;
            }
        }

        int jspn = ispn_block__ & 1;

        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg(ctx_.blas_linalg_t())
            .gemm('N', 'N', 2 * num_gkvec_loc, n__, nbeta, &linalg_const<T>::one(),
                  reinterpret_cast<T*>(beta_gk.at(mem)), 2 * num_gkvec_loc, work.at(mem), nbeta,
                  &linalg_const<T>::one(),
                  reinterpret_cast<T*>(op_phi__.pw_coeffs(jspn).prime().at(op_phi__.preferred_memory_t(), 0, idx0__)),
                  2 * op_phi__.pw_coeffs(jspn).prime().ld());

        switch (pu_) {
            case device_t::GPU: {
                acc::sync_stream(stream_id(-1));
                break;
            }
            case device_t::CPU: {
                break;
            }
        }
    }

    template <typename F, std::enable_if_t<std::is_same<std::complex<T>, F>::value, bool> = true>
    void apply(int chunk__, int ispn_block__, Wave_functions<T>& op_phi__, int idx0__, int n__,
               Beta_projectors_base<T>& beta__, matrix<F>& beta_phi__)
    {
        PROFILE("sirius::Non_local_operator::apply");

        if (is_null_) {
            return;
        }

        auto& beta_gk     = beta__.pw_coeffs_a();
        int num_gkvec_loc = beta__.num_gkvec_loc();
        int nbeta         = beta__.chunk(chunk__).num_beta_;

        /* setup linear algebra parameters */
        memory_t mem{memory_t::none};
        linalg_t la{linalg_t::none};
        switch (pu_) {
            case device_t::CPU: {
                mem = memory_t::host;
                la  = linalg_t::blas;
                break;
            }
            case device_t::GPU: {
                mem = memory_t::device;
                la  = linalg_t::gpublas;
                break;
            }
        }

        auto work = mdarray<std::complex<T>, 1>(nbeta * n__, ctx_.mem_pool(mem));

/* compute O * <beta|phi> for atoms in a chunk */
#pragma omp parallel
        {
            acc::set_device_id(sddk::get_device_id(acc::num_devices())); // avoid cuda mth bugs

#pragma omp for
            for (int i = 0; i < beta__.chunk(chunk__).num_atoms_; i++) {
                /* number of beta functions for a given atom */
                int nbf  = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i);
                int offs = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::offset), i);
                int ia   = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

                if (nbf) {
                    linalg(la).gemm(
                        'N', 'N', nbf, n__, nbf, &linalg_const<std::complex<T>>::one(),
                        reinterpret_cast<std::complex<T>*>(op_.at(mem, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                        beta_phi__.at(mem, offs, 0), beta_phi__.ld(), &linalg_const<std::complex<T>>::zero(),
                        work.at(mem, offs), nbeta, stream_id(omp_get_thread_num()));
                }
            }
        }
        switch (pu_) {
            case device_t::GPU: {
/* wait for previous zgemms */
#pragma omp parallel
                acc::sync_stream(stream_id(omp_get_thread_num()));
                break;
            }
            case device_t::CPU: {
                break;
            }
        }

        int jspn = ispn_block__ & 1;

        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        linalg(ctx_.blas_linalg_t())
            .gemm('N', 'N', num_gkvec_loc, n__, nbeta, &linalg_const<std::complex<T>>::one(), beta_gk.at(mem),
                  num_gkvec_loc, work.at(mem), nbeta, &linalg_const<std::complex<T>>::one(),
                  op_phi__.pw_coeffs(jspn).prime().at(op_phi__.preferred_memory_t(), 0, idx0__),
                  op_phi__.pw_coeffs(jspn).prime().ld());

        switch (pu_) {
            case device_t::GPU: {
                acc::sync_stream(stream_id(-1));
                break;
            }
            case device_t::CPU: {
                break;
            }
        }
    }

    /// Apply beta projectors from one atom in a chunk of beta projectors to all wave-functions.
    template <typename F, std::enable_if_t<std::is_same<std::complex<T>, F>::value, bool> = true>
    void apply(int chunk__, int ia__, int ispn_block__, Wave_functions<T>& op_phi__, int idx0__, int n__,
               Beta_projectors_base<T>& beta__, matrix<F>& beta_phi__)
    {
        if (is_null_) {
            return;
        }

        auto& beta_gk     = beta__.pw_coeffs_a();
        int num_gkvec_loc = beta__.num_gkvec_loc();

        int nbf  = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::nbf), ia__);
        int offs = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::offset), ia__);
        int ia   = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), ia__);

        if (nbf == 0) {
            return;
        }

        /* setup linear algebra parameters */
        memory_t mem{memory_t::none};
        linalg_t la{linalg_t::none};

        switch (pu_) {
            case device_t::CPU: {
                mem = memory_t::host;
                la  = linalg_t::blas;
                break;
            }
            case device_t::GPU: {
                mem = memory_t::device;
                la  = linalg_t::gpublas;
                break;
            }
        }

        auto work = mdarray<std::complex<T>, 1>(nbf * n__, ctx_.mem_pool(mem));

        linalg(la).gemm('N', 'N', nbf, n__, nbf, &linalg_const<std::complex<T>>::one(),
                        reinterpret_cast<std::complex<T>*>(op_.at(mem, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                        beta_phi__.at(mem, offs, 0), beta_phi__.ld(), &linalg_const<std::complex<T>>::zero(), work.at(mem),
                        nbf);

        int jspn = ispn_block__ & 1;

        linalg(ctx_.blas_linalg_t())
            .gemm('N', 'N', num_gkvec_loc, n__, nbf, &linalg_const<std::complex<T>>::one(), beta_gk.at(mem, 0, offs),
                  num_gkvec_loc, work.at(mem), nbf, &linalg_const<std::complex<T>>::one(),
                  op_phi__.pw_coeffs(jspn).prime().at(op_phi__.preferred_memory_t(), 0, idx0__),
                  op_phi__.pw_coeffs(jspn).prime().ld());
        switch (pu_) {
            case device_t::CPU: {
                break;
            }
            case device_t::GPU: {
#ifdef SIRIUS_GPU
                acc::sync_stream(stream_id(-1));
#endif
                break;
            }
        }
    }

    template <typename F, typename = std::enable_if_t<std::is_same<T, real_type<F>>::value>>
    inline F value(int xi1__, int xi2__, int ia__)
    {
        return this->value<F>(xi1__, xi2__, 0, ia__);
    }

    template <typename F, std::enable_if_t<std::is_same<T, F>::value, bool> = true>
    F value(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
        return this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__);
    }

    template <typename F, std::enable_if_t<std::is_same<std::complex<T>, F>::value, bool> = true>
    F value(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
        return std::complex<T>(this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__),
                               this->op_(1, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__));
    }

    inline bool is_diag() const
    {
        return is_diag_;
    }

};


template <typename T>
class D_operator : public Non_local_operator<T>
{
  private:
    void initialize();

  public:
    D_operator(Simulation_context const& ctx_);
};

template <typename T>
class Q_operator : public Non_local_operator<T>
{
  private:
    void initialize();

  public:
    Q_operator(Simulation_context const& ctx__);
};

template <typename T>
class U_operator
{
  private:
    Simulation_context const& ctx_;
    sddk::mdarray<std::complex<T>, 3> um_;
    std::vector<int> offset_;
    int nhwf_;

  public:
    U_operator(Simulation_context const& ctx__, Hubbard_matrix const& um1__, std::array<double, 3> vk__)
        : ctx_(ctx__)
    {
        /* a pair of "total number, offests" for the Hubbard orbitals idexing */
        auto r = ctx_.unit_cell().num_hubbard_wf();
        this->nhwf_ = r.first;
        this->offset_ = r.second;
        um_ = sddk::mdarray<std::complex<T>, 3>(this->nhwf_, this->nhwf_, ctx_.num_mag_dims() + 1);
        um_.zero();

        /* copy only local blocks */
        // TODO: implement Fourier-transfomation of the T-dependent occupancy matrix
        // to get the generic k-dependent matrix
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            if (ctx_.unit_cell().atom(ia).type().hubbard_correction()) {
                int nb = ctx_.unit_cell().atom(ia).type().indexb_hub().size();
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    for (int m1 = 0; m1 < nb; m1++) {
                        for (int m2 = 0; m2 < nb; m2++) {
                            um_(this->offset_[ia] + m1, this->offset_[ia] + m2, j) = um1__.local(ia)(m1, m2, j);
                        }
                    }
                }
            }
        }
    }

    ~U_operator()
    {
    }

    inline auto nhwf() const
    {
        return nhwf_;
    }

    inline auto offset(int ia__) const
    {
        return offset_[ia__];
    }

    std::complex<T> operator()(int m1, int m2, int j)
    {
        return um_(m1, m2, j);
    }
};

//template <typename T>
//class P_operator : public Non_local_operator<T>
//{
//  public:
//    P_operator(Simulation_context const& ctx_, mdarray<double_complex, 3>& p_mtrx__)
//        : Non_local_operator<T>(ctx_)
//    {
//        /* Q-operator is independent of spin */
//        this->op_ = mdarray<T, 2>(this->packed_mtrx_size_, 1);
//        this->op_.zero();
//
//        auto& uc = ctx_.unit_cell();
//        for (int ia = 0; ia < uc.num_atoms(); ia++) {
//            int iat = uc.atom(ia).type().id();
//            if (!uc.atom_type(iat).augment()) {
//                continue;
//            }
//            int nbf = uc.atom(ia).mt_basis_size();
//            for (int xi2 = 0; xi2 < nbf; xi2++) {
//                for (int xi1 = 0; xi1 < nbf; xi1++) {
//                    this->op_(this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0) = -p_mtrx__(xi1, xi2, iat).real();
//                }
//            }
//        }
//        if (this->pu_ == device_t::GPU) {
//            this->op_.allocate(memory_t::device);
//            this->op_.copy_to(memory_t::device);
//        }
//    }
//};

/// Apply non-local part of the Hamiltonian and S operators.
/** This operations must be combined because of the expensive inner product between wave-functions and beta
 *  projectors, which is computed only once.
 *
 *  \param [in]  spins   Range of the spin index.
 *  \param [in]  N       Starting index of the wave-functions.
 *  \param [in]  n       Number of wave-functions to which D and Q are applied.
 *  \param [in]  beta    Beta-projectors.
 *  \param [in]  phi     Wave-functions.
 *  \param [in]  d_op    Pointer to D-operator.
 *  \param [out] hphi    Resulting |beta>D<beta|phi>
 *  \param [in]  q_op    Pointer to Q-operator.
 *  \param [out] sphi    Resulting |beta>Q<beta|phi>
 */
template <typename T>
void
apply_non_local_d_q(sddk::spin_range spins__, int N__, int n__, Beta_projectors<real_type<T>>& beta__,
                    sddk::Wave_functions<real_type<T>>& phi__, D_operator<real_type<T>>* d_op__, sddk::Wave_functions<real_type<T>>* hphi__,
                    Q_operator<real_type<T>>* q_op__, sddk::Wave_functions<real_type<T>>* sphi__);

template <typename T>
void
apply_S_operator(sddk::device_t pu__, sddk::spin_range spins__, int N__, int n__, Beta_projectors<real_type<T>>& beta__,
                 sddk::Wave_functions<real_type<T>>& phi__, Q_operator<real_type<T>>* q_op__, sddk::Wave_functions<real_type<T>>& sphi__);

template <typename T>
void
apply_U_operator(Simulation_context& ctx__, spin_range spins__, int N__, int n__, Wave_functions<T>& hub_wf__,
    Wave_functions<T>& phi__, U_operator<T>& um__, Wave_functions<T>& hphi__);

} // namespace sirius

#endif
