#ifndef QE_ENERGIES_H
#define QE_ENERGIES_H

namespace sirius {

/// energies coming from QE (when using veff_callback)
struct qe_energies {
    /// the Hartree energy
    double ehart;
    // the exchange and correlation energy
    double etxc;
    /// another exchange - correlation energy
    double vtxc;
    /// the Hubbard contribution to the energy (module ldaU)
    double eth;
    /// energy correction due to the field (module extfield)
    double etotefield;
};


}  // sirius


#endif /* QE_ENERGIES_H */
