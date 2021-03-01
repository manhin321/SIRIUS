/** \file ultrasoft_precond.hpp
    \brief Provides preconditioner for ultrasoft case.
 */

#include "context/simulation_context.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "SDDK/memory.hpp"
#include "SDDK/gvec.hpp"
#include "diag_mm.hpp"

namespace sirius {

template<class numeric_t>
class DiagonalPreconditioner
{
  public:
    DiagonalPreconditioner(Simulation_context& ctx) : ctx_(ctx) {}
    sddk::mdarray<numeric_t, 2> apply(const sddk::mdarray<numeric_t, 2>& X, device_t processing_unit);
    void apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X, device_t processing_unit);

  protected:
    sddk::mdarray<numeric_t, 1> d_;
    Simulation_context& ctx_;
};

template <class numeric_t>
sddk::mdarray<numeric_t, 2>
DiagonalPreconditioner<numeric_t>::apply(const sddk::mdarray<numeric_t, 2> &X, device_t processing_unit)
{
    auto Y = empty_like(X, ctx_.mem_pool(processing_unit));
    this->apply(Y, X, processing_unit);
    return Y;
}

/// computes Y <- P*X
template <class numeric_t>
inline void
DiagonalPreconditioner<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X,
                                         device_t processing_unit)
{
    // copy d_ to gpu
    switch (processing_unit) {
        case device_t::CPU: {
            int n = X.size(0);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                numeric_t d = d_(i);
                for (int j = 0; j < static_cast<int>(X.size(1)); ++j) {
                    Y(i, j) = d * X(i, j);
                }
            }
            break;
        }
#ifdef __GPU
        case device_t::GPU: {
            d_.allocate(memory_t::device);
            d_.copy_to(memory_t::device);
            int n = d_.size(0);
            zdiagmm(d_.at(memory_t::device), n, X.at(memory_t::device), X.ld(), X.size(1), Y.at(memory_t::device),
                    Y.ld(), std::complex<double>{1});
            break;
        }
#endif
        default:
            std::cout << "processing_unit: " << int(processing_unit) << "\n";
            throw std::runtime_error("unknown processing unit in DiagonalPreconditioner");
            break;
    }
}

/** Payne, M. C., Teter, M. P., Allan, D. C., Arias, T. A., & Joannopoulos, J.
 *  D., Iterative minimization techniques for ab initio total-energy
 *  calculations: molecular dynamics and conjugate gradients.
 *  https://dx.doi.org/10.1103/RevModPhys.64.1045
 */
template <class numeric_t>
class Teter : DiagonalPreconditioner<numeric_t>
{
  public:
    Teter(Simulation_context& ctx, const Gvec& gkvec) : DiagonalPreconditioner<numeric_t>(ctx)
    {
        this->d_ = mdarray<numeric_t, 1>(gkvec.count());
        for (int i = 0; i < gkvec.count(); ++i) {
            // teter formula
            double T    = gkvec.gkvec_cart<index_domain_t::global>(i).length2();
            double T2   = T * T;
            double T3   = T2 * T;
            double T4   = T2 * T2;
            double Tp   = 16 * T4 / (27 + 18 * T + 12 * T2 + 8 * T3);
            // Eq. (5.16) in Payne et. al
            this->d_(i) = 1 / (1 + Tp);
        }
    }

    using DiagonalPreconditioner<numeric_t>::apply;
};

/** Ultrasoft preconditioner for direct minimization.
 *
 *  (1+T)⁻¹ + G R G⊹
 *  where R = -Q (1 + C Q)⁻¹
 *  and G  are the "preconditioned" beta projectors, C = B⊹ K B
 *  TODO: what is K?
 *
 * Hasnip, P. J., & Pickard, C. J. (). Electronic energy minimisation with
 * ultrasoft pseudopotentials. , 174(1), 24–29.
 * http://dx.doi.org/10.1016/j.cpc.2005.07.011
 */
template <class numeric_t>
class Ultrasoft_preconditioner
{
  public:
    Ultrasoft_preconditioner(Simulation_context& simulation_context, const Q_operator& q_op,
                             int ispn,
                             const Beta_projectors_base& bp, const Gvec& gkvec);

    sddk::mdarray<numeric_t, 2> apply(const sddk::mdarray<numeric_t, 2>& X);
    void apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X);

    const Simulation_context& ctx() const { return ctx_; }

  private:
    // cannot be const, because memory pool is used
    Simulation_context& ctx_;
    Teter<numeric_t> P;
    const Q_operator& q_op;
    int ispn;
    const Beta_projectors_base& bp;
    sddk::mdarray<int, 1> ipiv;
    sddk::mdarray<numeric_t, 2> LU;
};

template <class numeric_t>
Ultrasoft_preconditioner<numeric_t>::Ultrasoft_preconditioner(Simulation_context& simulation_context,
                                                              const Q_operator& q_op, int ispn, const Beta_projectors_base& bp, const Gvec& gkvec)
    : ctx_(simulation_context)
    , P(simulation_context, gkvec)
    , q_op(q_op)
    , ispn(ispn)
    , bp(bp)
{
    /* compute C <- <ϐ|P|ϐ> */
    auto C = inner_beta(bp, simulation_context, [&simulation_context = this->ctx_, &P = this->P](auto& Y) {
        // P.apply(Y, Y, simulation_context.processing_unit());
        return P.apply(Y, simulation_context.processing_unit());
    });
    // auto Cref = inner_beta(bp, simulation_context);

    sddk::matrix<numeric_t> CQ(C.size(0), q_op.size(1), memory_t::host);
    if (is_device_memory(ctx_.preferred_memory_t())) {
        C.allocate(memory_t::host);
        C.copy_to(memory_t::host);
    }
    /* compute C <- C@Q */
    this->q_op.lmatmul(CQ, C, this->ispn, memory_t::host);
    /* compute C <- 1 + C */
    int n = CQ.size(0);
    // add identiy matrix
    std::vector<double_complex> ones(n, 1);
    // add identity matrix
    linalg(linalg_t::blas).axpy(n, &linalg_const<double_complex>::one(), ones.data(), 1, CQ.at(memory_t::host), n + 1);
    // compute LU factorization
    this->LU = sddk::empty_like(CQ);
    sddk::copy(this->LU, CQ);
    this->ipiv = mdarray<int, 1>(n, memory_t::host);
    // compute LU factorization
    linalg(linalg_t::lapack).getrf(n, n, this->LU.at(memory_t::host), this->LU.ld(), this->ipiv.at(memory_t::host));
    // copy LU factorization to device if needed
    auto mem = ctx_.preferred_memory_t();
    if (is_device_memory(mem)) {
        ipiv.allocate(mem);
        ipiv.copy_to(mem);

        LU.allocate(mem);
        LU.copy_to(mem);
    }
}

template <class numeric_t>
sddk::mdarray<numeric_t,2> Ultrasoft_preconditioner<numeric_t>::apply(const sddk::mdarray<numeric_t, 2> &X)
{
    auto Y = empty_like(X, ctx_.mem_pool(ctx_.preferred_memory_t()));
    this->apply(Y, X);
    return Y;
}

template <class numeric_t>
void
Ultrasoft_preconditioner<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X)
{
    int num_beta = bp.num_total_beta();
    int nbnd     = X.size(1);

    auto bp_gen      = bp.make_generator();
    auto beta_coeffs = bp.prepare();
    sddk::mdarray<numeric_t, 2> bphi(num_beta, nbnd, ctx_.mem_pool(ctx_.preferred_memory_t()));
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        // apply preconditioner to beta projectors
        auto G = P.apply(beta_coeffs.pw_coeffs_a, ctx_.processing_unit());
        int row_offset = beta_coeffs.beta_chunk.offset_;
        // TODO: at memory_t dependent ptr / blas
        // linalg(linalg_t::blas)
        //     .gemm('C', 'N', G.size(1), nbnd, G.size(0), &linalg_const<numeric_t>::one, G.at(memory_t::host), G.ld(),
        //           X.at(memory_t::host), X.ld(), &linalg_const<numeric_t>::zero(),
        //           bphi.at(memory_t::host, row_offset, 0), bphi.ld());
        linalg_t la{linalg_t::blas};
        memory_t mem{memory_t::host};
        if (ctx_.processing_unit() == device_t::GPU) {
            la = linalg_t::gpublas;
            mem = memory_t::device;
        }
        linalg(la)
            .gemm('C', 'N', G.size(1), nbnd, G.size(0), &linalg_const<numeric_t>::one(), G.at(mem), G.ld(),
                  X.at(mem), X.ld(), &linalg_const<numeric_t>::zero(),
                  bphi.at(mem, row_offset, 0), bphi.ld());
        // linalg(linalg_t::blas).gemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, const T *alpha, const
        // T *A, ftn_int lda, const T *B, ftn_int ldb, const T *beta, T *C, ftn_int ldc)
        //         auto bphi_loc = inner<numeric_t>(ctx_.blas_linalg_t(), ctx_.processing_unit(),
        //         ctx_.preferred_memory_t(),
        //                                          ctx_.mem_pool(memory_t::host), G, X, 0, nbnd);
        //         // copy submatrix to bphi
        //         int beta_offset = beta_coeffs.beta_chunk.offset_;
        // // std::cout << "beta_offset: " << beta_offset << "\n";
        // #pragma omp parallel for
        //         for (int lbnd = 0; lbnd < nbnd; ++lbnd) {
        //             // issue copy operation
        //             sddk::copy(memory_t::host, bphi_loc.at(memory_t::host, 0, lbnd), memory_t::host,
        //                        bphi.at(memory_t::host, beta_offset, lbnd), beta_coeffs.beta_chunk.num_beta_);
        //         }
    }
    assert(num_beta == static_cast<int>(bphi.size(0)) && nbnd == static_cast<int>(bphi.size(1)));
    linalg_t la{linalg_t::lapack};
    memory_t mem{memory_t::host};
    if(ctx_.processing_unit() == device_t::GPU) {
        la = linalg_t::gpublas;
        mem = memory_t::device;
    }
    linalg(la)
        .getrs('N', num_beta, nbnd, LU.at(mem), LU.ld(), ipiv.at(mem), bphi.at(mem),
               bphi.ld());

    auto R = empty_like(bphi, ctx_.mem_pool(ctx_.preferred_memory_t()));
    q_op.rmatmul(R, bphi, ispn, ctx_.preferred_memory_t(), -1);

//     if (ctx_.processing_unit() == device_t::GPU) {
// #ifdef __GPU
//         // allocate bphi on gpu if needed
//         bphi.allocate(ctx_.mem_pool(ctx_.processing_unit()));
//         Y.allocate(ctx_.mem_pool(ctx_.processing_unit()));
// #endif
//     }

    // compute Y <- (1+T')^(-1) X
    this->P.apply(Y, X, ctx_.processing_unit());

    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        // apply preconditioner to beta projectors in place
        auto G = P.apply(beta_coeffs.pw_coeffs_a, ctx_.processing_unit());
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a.size(1);

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                linalg(linalg_t::blas)
                    .gemm('N', 'N', m, n, k, &linalg_const<numeric_t>::one(),
                          G.at(memory_t::host), G.ld(),
                          R.at(memory_t::host, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                          &linalg_const<numeric_t>::one(), Y.at(memory_t::host), Y.ld());
                break;
            }
#ifdef __GPU
            case device_t::GPU:
                linalg(linalg_t::gpublas)
                    .gemm('N', 'N', m, n, k, &linalg_const<numeric_t>::one(),
                          G.at(memory_t::device), G.ld(),
                          R.at(memory_t::device, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                          &linalg_const<numeric_t>::one(), Y.at(memory_t::device), Y.ld());

                break;
#endif
            default:
                throw std::runtime_error("invalid processing unit");
                break;
        }
    }
}
} // namespace sirius
