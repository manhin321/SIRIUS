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

/** \file beta_projectors_base.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_base class.
 */

#ifndef __BETA_PROJECTORS_BASE_HPP__
#define __BETA_PROJECTORS_BASE_HPP__

#include "context/simulation_context.hpp"
#include "SDDK/wave_functions.hpp"

namespace sirius {

#if defined(SIRIUS_GPU)
extern "C" void create_beta_gk_gpu(int                   num_atoms,
                                   int                   num_gkvec,
                                   int const*            beta_desc,
                                   double_complex const* beta_gk_t,
                                   double const*         gkvec,
                                   double const*         atom_pos,
                                   double_complex*       beta_gk);
#endif

/// Named index of a descriptor of beta-projectors. The same order is used by the GPU kernel.
enum class beta_desc_idx
{
    /// Number of beta-projector functions for this atom.
    nbf      = 0,
    /// Offset of beta-projectors in this chunk.
    offset   = 1,
    /// Offset of beta-projectors in the array for atom types.
    offset_t = 2,
    /// Global index of atom.
    ia       = 3
};

struct beta_chunk_t
{
    /// Number of beta-projectors in the current chunk.
    int num_beta_;
    /// Number of atoms in the current chunk.
    int num_atoms_;
    /// Offset in the global index of beta projectors.
    int offset_;
    /// Descriptor of block of beta-projectors for an atom.
    mdarray<int, 2> desc_;
    /// Positions of atoms.
    mdarray<double, 2> atom_pos_;

    beta_chunk_t() = default;

    beta_chunk_t(const beta_chunk_t& other)
        : desc_{empty_like(other.desc_)}
        , atom_pos_{empty_like(other.atom_pos_)}
    {
        // pass
        num_beta_  = other.num_beta_;
        num_atoms_ = other.num_atoms_;
        offset_    = other.offset_;

        copy(desc_, other.desc_);
        copy(atom_pos_, other.atom_pos_);
    }

    beta_chunk_t& operator=(const beta_chunk_t& other)
    {
        num_beta_ = other.num_beta_;
        num_atoms_ = other.num_atoms_;
        offset_ = other.offset_;

        desc_ = empty_like(other.desc_);
        copy(desc_, other.desc_);

        atom_pos_ = empty_like(other.atom_pos_);
        copy(atom_pos_, other.atom_pos_);
        return *this;
    }
};


struct beta_projectors_coeffs_t
{

    matrix<double_complex> pw_coeffs_a;
    mdarray<double_complex, 1> pw_coeffs_a_g0;
    Communicator comm;
    beta_chunk_t beta_chunk;

    /// buffer (num_max_beta) for pw_coeffs_a_g0
    matrix<double_complex> __pw_coeffs_a_buffer;
    /// buffer (num_max_beta) for pw_coeffs_a_g0
    mdarray<double_complex, 1> __pw_coeffs_a_g0_buffer;
    /// communicator of the G+k vector distribution
    };

/// Generates beta projector PW coefficients and holds GPU memory phase-factor
/// independent coefficients of |> functions for atom types.
class Beta_projector_generator
{
  public:
    typedef mdarray<double_complex, 3> array_t;

  public:
    Beta_projector_generator(Simulation_context& ctx, const array_t& pw_coeffs_t_host, const matrix<double_complex>& beta_pw_all, device_t processing_unit,
                             const std::vector<beta_chunk_t>& beta_chunks, const Gvec& gkvec,
                             const mdarray<double, 2>& gkvec_coord, const std::vector<int>& igk, int num_gkvec_loc)
        : ctx_(ctx)
        , pw_coeffs_t_host_(pw_coeffs_t_host)
        , beta_pw_all_atoms_(beta_pw_all)
        , processing_unit_(processing_unit)
        , beta_chunks_(beta_chunks)
        , gkvec_(gkvec)
        , gkvec_coord_(gkvec_coord)
        , igk_(igk)
        , num_gkvec_loc_(num_gkvec_loc)
    {

        if (processing_unit == device_t::GPU) {
            // copy to GPU if needed
            pw_coeffs_t_device_ = array_t(pw_coeffs_t_host.size(0), pw_coeffs_t_host.size(1),
                                          pw_coeffs_t_host.size(2), ctx_.mem_pool(device_t::GPU));
            // copy to device
            acc::copyin(pw_coeffs_t_device_.device_data(), pw_coeffs_t_host.host_data(), pw_coeffs_t_host.size());
        }
    }


    void generate(beta_projectors_coeffs_t& coeffs, int ichunk, int j) const;
    void generate(beta_projectors_coeffs_t& coeffs, int ichunk) const;

  private:
    // from Beta_projectors_base
    Simulation_context& ctx_;
    const array_t& pw_coeffs_t_host_;
    /// precomputed beta coefficients on CPU
    const matrix<double_complex>& beta_pw_all_atoms_;
    device_t processing_unit_;
    const std::vector<beta_chunk_t>& beta_chunks_;
    const Gvec& gkvec_;
    const mdarray<double, 2>& gkvec_coord_;
    const std::vector<int>& igk_;
    int num_gkvec_loc_;
    // own
    array_t pw_coeffs_t_device_;

};


/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
class Beta_projectors_base
{
  protected:
    Simulation_context& ctx_;

    /// List of G+k vectors.
    Gvec const& gkvec_;

    /// Mapping between local and global G+k vector index.
    std::vector<int> const& igk_;

    /// Coordinates of G+k vectors used by GPU kernel.
    mdarray<double, 2> gkvec_coord_;

    /// Number of different components: 1 for beta-projectors, 3 for gradient, 9 for strain derivatives.
    int N_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    mdarray<double_complex, 3> pw_coeffs_t_;

    bool reallocate_pw_coeffs_t_on_gpu_{true};

    /// Set of beta PW coefficients for a chunk of atoms.
    matrix<double_complex> pw_coeffs_a_;

    /// Set of beta PW coefficients for all atoms
    matrix<double_complex> beta_pw_all_atoms_;

    mdarray<double_complex, 1> pw_coeffs_a_g0_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// total number of beta-projectors
    int num_total_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;

    /// Split beta-projectors into chunks.
    void split_in_chunks();

    template <typename T>
    void local_inner_aux(T* beta_pw_coeffs_a_ptr__, int nbeta__, Wave_functions& phi__, int ispn__, int idx0__, int n__,
                         matrix<T>& beta_phi__) const;

  public:
    Beta_projectors_base(Simulation_context& ctx__, Gvec const& gkvec__, std::vector<int> const& igk__, int N__);

    // /// Calculate inner product between beta-projectors and wave-functions.
    // /** The following is computed: <beta|phi> */
    // template <typename T>
    // matrix<T> inner(int chunk__, Wave_functions& phi__, int ispn__, int idx0__, int n__);

    Beta_projector_generator make_generator() const
    {
        return make_generator(ctx_.processing_unit());
    }

    Beta_projector_generator make_generator(device_t pu) const
    {
        return Beta_projector_generator{ctx_,
                                        pw_coeffs_t_,
                                        beta_pw_all_atoms_,
                                        pu,
                                        beta_chunks_,
                                        gkvec_,
                                        gkvec_coord_,
                                        igk_,
                                        num_gkvec_loc()};
    }

    Simulation_context& ctx()
    {
        return ctx_;
    }

    const Gvec& gkvec() const
    {
        return gkvec_;
    }


    __attribute__((deprecated)) void generate(int ichunk__, int j__)
    {
        throw std::runtime_error("generate(chunk,j) is obsolete");
//         PROFILE("sirius::Beta_projectors_base::generate");

//         switch (ctx_.processing_unit()) {
//             case device_t::CPU: {
// #pragma omp parallel for
//                 for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
//                     int ia = chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

//                     double phase           = twopi * dot(gkvec_.vk(), ctx_.unit_cell().atom(ia).position());
//                     double_complex phase_k = std::exp(double_complex(0.0, phase));

//                     std::vector<double_complex> phase_gk(num_gkvec_loc());
//                     for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
//                         auto G = gkvec_.gvec(igk_[igk_loc]);
//                         /* total phase e^{-i(G+k)r_{\alpha}} */
//                         phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
//                     }
//                     for (int xi = 0; xi < chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
//                         for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
//                             pw_coeffs_a_(igk_loc,
//                                          chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
//                                 pw_coeffs_t_(igk_loc,
//                                              chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi,
//                                              j__) *
//                                 phase_gk[igk_loc];
//                         }
//                     }
//                 }
//                 break;
//             }
//             case device_t::GPU: {
// #if defined(__GPU)
//                 auto& desc = chunk(ichunk__).desc_;
//                 create_beta_gk_gpu(chunk(ichunk__).num_atoms_, num_gkvec_loc(), desc.at(memory_t::device),
//                                    pw_coeffs_t_.at(memory_t::device, 0, 0, j__), gkvec_coord_.at(memory_t::device),
//                                    chunk(ichunk__).atom_pos_.at(memory_t::device), pw_coeffs_a().at(memory_t::device));
// #endif
//                 /* wave-functions are on CPU but the beta-projectors are on GPU */
//                 if (gkvec_.comm().rank() == 0 && is_host_memory(ctx_.preferred_memory_t())) {
//                     /* make beta-projectors for G=0 on the CPU */
// #pragma omp parallel for schedule(static)
//                     for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
//                         for (int xi = 0; xi < chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
//                             pw_coeffs_a_g0_(chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
//                                 pw_coeffs_t_(
//                                     0, chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi, j__);
//                         }
//                     }
//                 }
//                 break;
//             }
//         }
    }

    /// Generate beta-projectors for a chunk of atoms.
    /** Beta-projectors are always generated and stored in the memory of a processing unit.
     *
     *  \param [in] ichunk Index of a chunk of atoms for which beta-projectors are generated.
     *  \param [in] j index of the component (up to 9 components are used for the strain derivative)
     */
    __attribute__((deprecated)) void generate(beta_projectors_coeffs_t& out, int ichunk__, int j__) const;

    beta_projectors_coeffs_t prepare(memory_t pm = memory_t::none) const;

    __attribute_deprecated__ void dismiss();

    inline int num_gkvec_loc() const
    {
        return static_cast<int>(igk_.size());
    }

    inline int num_comp() const
    {
        return N_;
    }

    int num_total_beta() const
    {
        return num_total_beta_;
    }

    inline Unit_cell const& unit_cell() const
    {
        return ctx_.unit_cell();
    }

    double_complex& pw_coeffs_t(int ig__, int n__, int j__)
    {
        return pw_coeffs_t_(ig__, n__, j__);
    }

    matrix<double_complex> pw_coeffs_t(int j__)
    {
        return matrix<double_complex>(&pw_coeffs_t_(0, 0, j__), num_gkvec_loc(), num_beta_t());
    }

    /// Plane wave coefficients of |beta> projectors for a chunk of atoms.
    __attribute_deprecated__ matrix<double_complex>& pw_coeffs_a()
    {
        return pw_coeffs_a_;
    }

    __attribute_deprecated__ mdarray<double_complex, 1>& pw_coeffs_a0()
    {
        return pw_coeffs_a_g0_;
    }

    inline int num_beta_t() const
    {
        return num_beta_t_;
    }

    inline int num_chunks() const
    {
        return static_cast<int>(beta_chunks_.size());
    }

    inline beta_chunk_t const& chunk(int idx__) const
    {
        return beta_chunks_[idx__];
    }

    inline int max_num_beta() const
    {
        return max_num_beta_;
    }

    const Communicator& comm() const
    {
        return gkvec_.comm();
    }
};


/// inner product <beta|Op|beta>
template<class Op>
sddk::matrix<double_complex> inner_beta(const Beta_projectors_base& beta, const Simulation_context& ctx, Op&& op)
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
                // const double_complex* B = bcoeffs_col.pw_coeffs_a.at(preferred_memory);
                // apply Op on |b>  (in-place operation)
                auto G = op(bcoeffs_col.pw_coeffs_a);

                const double_complex* B2 = G.at(preferred_memory);
                double_complex* C       = out.at(preferred_memory, dest_row, dest_col);
                // linalg(linalg_t).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B,
                //                       bcoeffs_col.pw_coeffs_a.ld(), &zero, C, out.ld());
                linalg(linalg_t).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B2,
                                      G.ld(), &zero, C, out.ld());
            }
        }
        return out;
    } else {
        throw std::runtime_error("distributed case not yet implemented: " + std::string(__func__) + " in " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}


template <typename T>
__attribute_deprecated__ matrix<T> inner(Beta_projectors_base& beta__, int chunk__, Wave_functions& phi__, int ispn__,
                                         int idx0__, int n__);

void beta_projectors_generate_cpu(matrix<double_complex>& pw_coeffs_a, const mdarray<double_complex, 3>& pw_coeffs_t,
                                  int ichunk__, int j__, const beta_chunk_t& beta_chunk, const Simulation_context& ctx,
                                  const Gvec& gkvec, const std::vector<int>& igk__);

void beta_projectors_generate_gpu(beta_projectors_coeffs_t& out, const mdarray<double_complex, 3>& pw_coeffs_t_device,
                                  const mdarray<double_complex, 3>& pw_coeffs_t_host, const Simulation_context& ctx,
                                  const Gvec& gkvec, const mdarray<double, 2>& gkvec_coord_,
                                  const beta_chunk_t& beta_chunk, const std::vector<int>& igk__, int j__);

// /// TODO: add docstring, standalone inner product, Wave_functions carry communicator
// template <typename T>
// matrix<T>
// inner(const beta_projectors_coeffs_t& , int chunk__, Wave_functions& phi__, int ispn, int idx0__, int n__);
template <typename T>
matrix<T> inner(linalg_t linalg, device_t processing_unit, memory_t preferred_memory, memory_pool& mempool,
                const beta_projectors_coeffs_t& beta_projector_coeffs, Wave_functions& phi__, int ispn__,
                int idx0__, int n__);

/// inner product of beta projectors, mdarray
template <typename T>
matrix<T> inner(linalg_t linalg, device_t processing_unit, memory_t preferred_memory, memory_pool& mempool,
                const beta_projectors_coeffs_t& beta_projector_coeffs, const matrix<double_complex>& other, int idx0__, int n__,
                memory_t target_memory = memory_t::none);

/// inner product <ϐ|ϐ>
sddk::matrix<double_complex> inner_beta(const Beta_projectors_base& beta, const Simulation_context& ctx);


} // namespace sirius



#endif
