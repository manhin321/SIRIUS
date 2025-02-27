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

/** \file xc_mt.cpp
 *
 *  \brief Generate XC potential in the muffin-tins.
 */

#include <vector>

#include "potential.hpp"
#include "typedefs.hpp"
#include "utils/profiler.hpp"
#include "SDDK/omp.hpp"
#include "xc_functional.hpp"

namespace sirius {

void xc_mt_nonmagnetic(Radial_grid<double> const& rgrid__, SHT const& sht__, std::vector<XC_functional> const& xc_func__,
                       Flm const& rho_lm__, Ftp& rho_tp__, Flm& vxc_lm__, Flm& exc_lm__)
{
    bool is_gga{false};
    for (auto& ixc : xc_func__) {
        if (ixc.is_gga() || ixc.is_vdw()) {
            is_gga = true;
        }
    }

    Ftp exc_tp(sht__.num_points(), rgrid__);
    Ftp vxc_tp(sht__.num_points(), rgrid__);

    assert(rho_tp__.size() == vxc_tp.size());
    assert(rho_tp__.size() == exc_tp.size());

    Ftp grad_rho_grad_rho_tp;
    Ftp vsigma_tp;
    Ftp lapl_rho_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_tp;

    /* use Laplacian (true) or divergence of gradient (false) */
    bool use_lapl{false};

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_lm = gradient(rho_lm__);
        grad_rho_tp = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            transform(sht__, grad_rho_lm[x], grad_rho_tp[x]);
        }
        /* compute density gradient product */
        grad_rho_grad_rho_tp = grad_rho_tp * grad_rho_tp;
        assert(rho_tp__.size() == grad_rho_grad_rho_tp.size());

        vsigma_tp = Ftp(sht__.num_points(), rgrid__);
        assert(rho_tp__.size() == vsigma_tp.size());
        if (use_lapl) {
            /* backward transform Laplacian from Rlm to (theta, phi) */
            lapl_rho_tp = transform(sht__, laplacian(rho_lm__));
            assert(lapl_rho_tp.size() == rho_tp__.size());
        }
    }

    for (auto& ixc: xc_func__) {
        /* if this is an LDA functional */
        if (ixc.is_lda()) {
            ixc.get_lda(sht__.num_points() * rgrid__.num_points(), rho_tp__.at(memory_t::host),
                vxc_tp.at(memory_t::host), exc_tp.at(memory_t::host));
        }
        /* if this is a GGA functional */
        if (ixc.is_gga()) {

            /* compute vrho and vsigma */
            ixc.get_gga(sht__.num_points() * rgrid__.num_points(), rho_tp__.at(memory_t::host),
                grad_rho_grad_rho_tp.at(memory_t::host), vxc_tp.at(memory_t::host), vsigma_tp.at(memory_t::host),
                exc_tp.at(memory_t::host));

            if (use_lapl) {
                vxc_tp -= 2.0 * vsigma_tp * lapl_rho_tp;

                /* compute gradient of vsgima in spherical harmonics */
                auto grad_vsigma_lm = gradient(transform(sht__, vsigma_tp));

                /* backward transform gradient from Rlm to (theta, phi) */
                Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_tp(sht__.num_points(), rgrid__);
                for (int x = 0; x < 3; x++) {
                    transform(sht__, grad_vsigma_lm[x], grad_vsigma_tp[x]);
                }

                /* compute scalar product of two gradients */
                auto grad_vsigma_grad_rho_tp = grad_vsigma_tp * grad_rho_tp;
                /* add remaining term to Vxc */
                vxc_tp -= 2.0 * grad_vsigma_grad_rho_tp;
            } else {
                Spheric_vector_function<function_domain_t::spectral, double> vsigma_grad_rho_lm(sht__.lmmax(), rgrid__);
                for (int x: {0, 1, 2}) {
                    auto vsigma_grad_rho_tp = vsigma_tp * grad_rho_tp[x];
                    transform(sht__, vsigma_grad_rho_tp, vsigma_grad_rho_lm[x]);
                }
                auto div_vsigma_grad_rho_tp = transform(sht__, divergence(vsigma_grad_rho_lm));
                /* add remaining term to Vxc */
                vxc_tp -= 2.0 * div_vsigma_grad_rho_tp;
            }
        }
        exc_lm__ += transform(sht__, exc_tp);
        vxc_lm__ += transform(sht__, vxc_tp);
    } //ixc
}

void xc_mt_magnetic(Radial_grid<double> const& rgrid__, SHT const& sht__, int num_mag_dims__,
                    std::vector<XC_functional> const& xc_func__, std::vector<Ftp> const& rho_tp__,
                    std::vector<Flm*> vxc__, Flm& exc__)
{
    bool is_gga{false};
    for (auto& ixc : xc_func__) {
        if (ixc.is_gga() || ixc.is_vdw()) {
            is_gga = true;
        }
    }

    Ftp exc_tp(sht__.num_points(), rgrid__);
    Ftp vxc_tp(sht__.num_points(), rgrid__);

    /* convert to rho_up, rho_dn */
    Ftp rho_dn_tp(sht__.num_points(), rgrid__);
    Ftp rho_up_tp(sht__.num_points(), rgrid__);
    /* loop over radial grid points */
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        /* loop over points on the sphere */
        for (int itp = 0; itp < sht__.num_points(); itp++) {
            vector3d<double> m;
            for (int j = 0; j < num_mag_dims__; j++) {
                m[j] = rho_tp__[1 + j](itp, ir);
            }
            auto rud = get_rho_up_dn(num_mag_dims__, rho_tp__[0](itp, ir), m);

            /* compute "up" and "dn" components */
            rho_up_tp(itp, ir) = rud.first;
            rho_dn_tp(itp, ir) = rud.second;
        }
    }
    /* transform from (theta, phi) to Rlm */
    auto rho_up_lm = transform(sht__, rho_up_tp);
    auto rho_dn_lm = transform(sht__, rho_dn_tp);

    std::vector<Ftp> bxc_tp(num_mag_dims__);

    Ftp vxc_up_tp(sht__.num_points(), rgrid__);
    Ftp vxc_dn_tp(sht__.num_points(), rgrid__);
    for (int j = 0; j < num_mag_dims__; j++) {
        bxc_tp[j] = Ftp(sht__.num_points(), rgrid__);
    }

    Ftp grad_rho_up_grad_rho_up_tp;
    Ftp grad_rho_up_grad_rho_dn_tp;
    Ftp grad_rho_dn_grad_rho_dn_tp;
    Ftp vsigma_uu_tp;
    Ftp vsigma_ud_tp;
    Ftp vsigma_dd_tp;
    Ftp lapl_rho_up_tp;
    Ftp lapl_rho_dn_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_up_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_dn_tp;

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_up_lm = gradient(rho_up_lm);
        auto grad_rho_dn_lm = gradient(rho_dn_lm);
        grad_rho_up_tp = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        grad_rho_dn_tp = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            transform(sht__, grad_rho_up_lm[x], grad_rho_up_tp[x]);
            transform(sht__, grad_rho_dn_lm[x], grad_rho_dn_tp[x]);
        }
        /* compute density gradient products */
        grad_rho_up_grad_rho_up_tp = grad_rho_up_tp * grad_rho_up_tp;
        grad_rho_up_grad_rho_dn_tp = grad_rho_up_tp * grad_rho_dn_tp;
        grad_rho_dn_grad_rho_dn_tp = grad_rho_dn_tp * grad_rho_dn_tp;

        vsigma_uu_tp = Ftp(sht__.num_points(), rgrid__);
        vsigma_ud_tp = Ftp(sht__.num_points(), rgrid__);
        vsigma_dd_tp = Ftp(sht__.num_points(), rgrid__);

        /* backward transform Laplacians from Rlm to (theta, phi) */
        lapl_rho_up_tp = transform(sht__, laplacian(rho_up_lm));
        lapl_rho_dn_tp = transform(sht__, laplacian(rho_dn_lm));
    }

    for (auto& ixc: xc_func__) {
        if (ixc.is_lda()) {
            ixc.get_lda(sht__.num_points() * rgrid__.num_points(), rho_up_tp.at(memory_t::host),
                rho_dn_tp.at(memory_t::host), vxc_up_tp.at(memory_t::host), vxc_dn_tp.at(memory_t::host),
                exc_tp.at(memory_t::host));
        }
        if (ixc.is_gga()) {
            /* get the vrho and vsigma */
            ixc.get_gga(sht__.num_points() * rgrid__.num_points(), rho_up_tp.at(memory_t::host),
                    rho_dn_tp.at(memory_t::host), grad_rho_up_grad_rho_up_tp.at(memory_t::host),
                    grad_rho_up_grad_rho_dn_tp.at(memory_t::host), grad_rho_dn_grad_rho_dn_tp.at(memory_t::host),
                    vxc_up_tp.at(memory_t::host), vxc_dn_tp.at(memory_t::host), vsigma_uu_tp.at(memory_t::host),
                    vsigma_ud_tp.at(memory_t::host), vsigma_dd_tp.at(memory_t::host), exc_tp.at(memory_t::host));

            /* directly add to Vxc available contributions */
            vxc_up_tp -= (2.0 * vsigma_uu_tp * lapl_rho_up_tp + vsigma_ud_tp * lapl_rho_dn_tp);
            vxc_dn_tp -= (2.0 * vsigma_dd_tp * lapl_rho_dn_tp + vsigma_ud_tp * lapl_rho_up_tp);

            /* forward transform vsigma to Rlm */
            auto vsigma_uu_lm = transform(sht__, vsigma_uu_tp);
            auto vsigma_ud_lm = transform(sht__, vsigma_ud_tp);
            auto vsigma_dd_lm = transform(sht__, vsigma_dd_tp);

            /* compute gradient of vsgima in spherical harmonics */
            auto grad_vsigma_uu_lm = gradient(vsigma_uu_lm);
            auto grad_vsigma_ud_lm = gradient(vsigma_ud_lm);
            auto grad_vsigma_dd_lm = gradient(vsigma_dd_lm);

            /* backward transform gradient from Rlm to (theta, phi) */
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_uu_tp(sht__.num_points(), rgrid__);
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_ud_tp(sht__.num_points(), rgrid__);
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_dd_tp(sht__.num_points(), rgrid__);
            for (int x = 0; x < 3; x++) {
                grad_vsigma_uu_tp[x] = transform(sht__, grad_vsigma_uu_lm[x]);
                grad_vsigma_ud_tp[x] = transform(sht__, grad_vsigma_ud_lm[x]);
                grad_vsigma_dd_tp[x] = transform(sht__, grad_vsigma_dd_lm[x]);
            }

            /* compute scalar product of two gradients */
            auto grad_vsigma_uu_grad_rho_up_tp = grad_vsigma_uu_tp * grad_rho_up_tp;
            auto grad_vsigma_dd_grad_rho_dn_tp = grad_vsigma_dd_tp * grad_rho_dn_tp;
            auto grad_vsigma_ud_grad_rho_up_tp = grad_vsigma_ud_tp * grad_rho_up_tp;
            auto grad_vsigma_ud_grad_rho_dn_tp = grad_vsigma_ud_tp * grad_rho_dn_tp;

            /* add remaining terms to Vxc */
            vxc_up_tp -= (2.0 * grad_vsigma_uu_grad_rho_up_tp + grad_vsigma_ud_grad_rho_dn_tp);
            vxc_dn_tp -= (2.0 * grad_vsigma_dd_grad_rho_dn_tp + grad_vsigma_ud_grad_rho_up_tp);
        }
        /* genertate magnetic filed and effective potential inside MT sphere */
        for (int ir = 0; ir < rgrid__.num_points(); ir++) {
            for (int itp = 0; itp < sht__.num_points(); itp++) {
                /* Vxc = 0.5 * (V_up + V_dn) */
                vxc_tp(itp, ir) = 0.5 * (vxc_up_tp(itp, ir) + vxc_dn_tp(itp, ir));
                /* Bxc = 0.5 * (V_up - V_dn) */
                double bxc = 0.5 * (vxc_up_tp(itp, ir) - vxc_dn_tp(itp, ir));
                /* get the sign between mag and B */
                auto s = utils::sign((rho_up_tp(itp, ir) - rho_dn_tp(itp, ir)) * bxc);

                vector3d<double> m;
                for (int j = 0; j < num_mag_dims__; j++) {
                    m[j] = rho_tp__[1 + j](itp, ir);
                }
                auto m_len = m.length();
                if (m_len > 1e-8) {
                    for (int j = 0; j < num_mag_dims__; j++) {
                        bxc_tp[j](itp, ir) = std::abs(bxc) * s * m[j] / m_len;
                    }
                } else {
                    for (int j = 0; j < num_mag_dims__; j++) {
                        bxc_tp[j](itp, ir) = 0.0;
                    }
                }
            }
        }
        /* convert magnetic field back to Rlm */
        for (int j = 0; j < num_mag_dims__; j++) {
            *vxc__[j + 1] += transform(sht__, bxc_tp[j]);
        }
        /* forward transform from (theta, phi) to Rlm */
        *vxc__[0] += transform(sht__, vxc_tp);
        exc__ += transform(sht__, exc_tp);
    } // ixc
}

void xc_mt(Radial_grid<double> const& rgrid__, SHT const& sht__, std::vector<XC_functional> const& xc_func__,
        int num_mag_dims__, std::vector<Flm const*> rho__, std::vector<Flm*> vxc__, Flm* exc__)
{
    /* zero the fields */
    exc__->zero();
    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        vxc__[j]->zero();
    }

    std::vector<Ftp> rho_tp(num_mag_dims__ + 1);
    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        /* convert density and magnetization to theta, phi */
        rho_tp[j] = transform(sht__, *rho__[j]);
    }

    /* check if density has negative values */
    double rhomin{0};
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        for (int itp = 0; itp < sht__.num_points(); itp++) {
            rhomin = std::min(rhomin, rho_tp[0](itp, ir));
            /* fix negative density */
            if (rho_tp[0](itp, ir) < 0.0) {
                rho_tp[0](itp, ir) = 0.0;
            }
        }
    }

    if (rhomin < 0.0) {
        std::stringstream s;
        s << "[xc_mt] negative charge density: " << rhomin << std::endl
          << "  current Rlm expansion of the charge density may be not sufficient, try to increase lmax";
        WARNING(s);
    }

    if (num_mag_dims__ == 0) {
        xc_mt_nonmagnetic(rgrid__, sht__, xc_func__, *rho__[0], rho_tp[0], *vxc__[0], *exc__);
    } else {
        xc_mt_magnetic(rgrid__, sht__, num_mag_dims__, xc_func__, rho_tp, vxc__, *exc__);
    }
}

void Potential::xc_mt(Density const& density__)
{
    PROFILE("sirius::Potential::xc_mt");

    #pragma omp parallel for
    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        auto& rgrid = unit_cell_.atom(ia).radial_grid();
        std::vector<Flm const*> rho(ctx_.num_mag_dims() + 1);
        std::vector<Flm*> vxc(ctx_.num_mag_dims() + 1);
        rho[0] = &density__.rho().f_mt(ialoc);
        vxc[0] = &xc_potential_->f_mt(ialoc);
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            rho[j + 1] = &density__.magnetization(j).f_mt(ialoc);
            vxc[j + 1] = &effective_magnetic_field(j).f_mt(ialoc);
        }
        sirius::xc_mt(rgrid, *sht_, xc_func_, ctx_.num_mag_dims(), rho, vxc, &xc_energy_density_->f_mt(ialoc));

        /* z, x, y order */
        std::array<int, 3> comp_map = {2, 0, 1};
        /* add auxiliary magnetic field antiparallel to starting magnetization */
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                effective_magnetic_field(j).f_mt<index_domain_t::local>(0, ir, ialoc) -=
                    aux_bf_(j, ia) * ctx_.unit_cell().atom(ia).vector_field()[comp_map[j]];
            }
        }
    } // ialoc
}

} // namespace sirius
