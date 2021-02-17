#include "beta_projectors.hpp"

namespace sirius {

Beta_projectors::Beta_projectors(Simulation_context& ctx__, Gvec const& gkvec__, std::vector<int>& igk__)
    : Beta_projectors_base(ctx__, gkvec__, igk__, 1)
{
    PROFILE("sirius::Beta_projectors");
    /* generate phase-factor independent projectors for atom types */
    generate_pw_coefs_t(igk__);
    /* special treatment for beta-projectors as they are mostly often used */

    switch (ctx_.processing_unit()) {
        /* beta projectors for atom types will be stored on GPU for the entire run */
        case device_t::GPU: {
            reallocate_pw_coeffs_t_on_gpu_ = false;
            // TODO remove is done in `Beta_projector_generator`
            pw_coeffs_t_.allocate(memory_t::device).copy_to(memory_t::device);
            break;
        }
        /* generate beta projectors for all atoms */
        case device_t::CPU: {
            // allocate beta_pw_all_atoms
            beta_pw_all_atoms_ = matrix<double_complex>(num_gkvec_loc(), ctx_.unit_cell().mt_lo_basis_size());
            for (int ichunk = 0; ichunk < num_chunks(); ++ichunk) {
                pw_coeffs_a_ = matrix<double_complex>(&beta_pw_all_atoms_(0, chunk(ichunk).offset_), num_gkvec_loc(),
                                                      chunk(ichunk).num_beta_);
                beta_projectors_generate_cpu(pw_coeffs_a_, pw_coeffs_t_, ichunk, /*j*/ 0, chunk(ichunk), ctx__, gkvec__,
                                             igk__);
            }

            break;
        }
    }
}

void
Beta_projectors::generate_pw_coefs_t(std::vector<int>& igk__)
{
    PROFILE("sirius::Beta_projectors::generate_pw_coefs_t");
    if (!num_beta_t()) {
        return;
    }

    auto& comm = gkvec_.comm();

    auto& beta_radial_integrals = ctx_.beta_ri();

    std::vector<double_complex> z(ctx_.unit_cell().lmax() + 1);
    for (int l = 0; l <= ctx_.unit_cell().lmax(); l++) {
        z[l] = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());
    }

/* compute <G+k|beta> */
#pragma omp parallel for
    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
        int igk = igk__[igkloc];
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart<index_domain_t::global>(igk));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> gkvec_rlm(utils::lmmax(ctx_.unit_cell().lmax()));
        sf::spherical_harmonics(ctx_.unit_cell().lmax(), vs[1], vs[2], &gkvec_rlm[0]);
        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
            auto& atom_type = ctx_.unit_cell().atom_type(iat);
            /* get all values of radial integrals */
            auto ri_val = beta_radial_integrals.values(iat, vs[0]);
            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                int l     = atom_type.indexb(xi).l;
                int lm    = atom_type.indexb(xi).lm;
                int idxrf = atom_type.indexb(xi).idxrf;

                pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, 0) = z[l] * gkvec_rlm[lm] * ri_val(idxrf);
            }
        }
    }

    if (ctx_.control().print_checksum_) {
        auto c1 = pw_coeffs_t_.checksum();
        comm.allreduce(&c1, 1);
        if (comm.rank() == 0) {
            utils::print_checksum("beta_pw_coeffs_t", c1);
        }
    }
}


__attribute_deprecated__
void
Beta_projectors::dismiss()
{
    // if (!prepared_) {
    //     TERMINATE("beta projectors are already dismissed");
    // }
    // switch (ctx_.processing_unit()) {
    //     case device_t::GPU: {
    //         Beta_projectors_base::dismiss();
    //         break;
    //     }
    //     case device_t::CPU:
    //         break;
    // }
    // prepared_ = false;
}


__attribute_deprecated__
void
Beta_projectors::generate(int chunk__)
{
    // switch (ctx_.processing_unit()) {
    //     case device_t::CPU: {
    //         pw_coeffs_a_ = matrix<double_complex>(&beta_pw_all_atoms_(0, chunk(chunk__).offset_), num_gkvec_loc(),
    //                                               chunk(chunk__).num_beta_);
    //         break;
    //     }
    //     case device_t::GPU: {
    //         Beta_projectors_base::generate(chunk__, 0);
    //         break;
    //     }
    // }
    throw std::runtime_error("deprecated");
}

} // namespace sirius
