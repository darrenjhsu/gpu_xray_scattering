



void xray_scattering (
    int num_atom,    // number of atoms Na
    float *coord,    // coordinates     3Na
    float *Ele,      // Element
    float *vdW,      // van der Waals radius Na
    int num_q,       // number of q vector points
    float *q,        // q vector        Nq
    float *S_calc,   // scattering intensity to be returned, Nq
    int num_raster,  // number of raster points for SASA calculation
    float sol_s,     // solvent radius for SASA (1.8)
    float r_m,       // exclusion radius for C1 (1.62)
    float rho,       // water electron density (e-/A^3, 0.334 at 20 C)
    float c1,        // hyperparameter for C1 solvent exclusion term (1.0)
    float c2        // hyperparameter for hydration shell (2.0)
);


