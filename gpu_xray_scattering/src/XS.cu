
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "kernel.cu"
#include "WaasKirf.hh"
#include "vdW.hh"
#include "XS.hh"
#define PI 3.14159265359 


extern "C" {
void xray_scattering (
    int num_atom,    // number of atoms Na
    float *coord,    // coordinates     3Na
    int *Ele,      // Element
    int num_q,       // number of q vector points
    float *q,        // q vector        Nq
    float *S_calc,   // scattering intensity to be returned, Nq
    int num_raster,  // number of raster points for SASA calculation
    float sol_s,     // solvent radius for SASA (1.8)
    float r_m,       // exclusion radius for C1 (1.62)
    float rho,       // water electron density (e-/A^3, 0.334 at 20 C)
    float c1,        // hyperparameter for C1 solvent exclusion term (1.0)
    float c2         // hyperparameter for hydration shell (2.0)
) {
// This function is called by tclforce script from NAMD. Note that it is only executed every delta_t steps.
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    // In this code pointers with d_ are device pointers. 
    int   num_atom2 = (num_atom + 2047)/2048 * 2048;
    int   num_q2    = (num_q + 31) / 32 * 32;
    //printf("num_atom2 = %d, num_q2 = %d\n", num_atom2, num_q2);

    // Declare cuda pointers //
    float *d_coord;          // Coordinates 3 x num_atom

    int   *d_Ele;            // Element list.
    float *d_q;              // q vector
    float *d_S_calc;         // Calculated scattering curve
    float *d_S_calcc;        // Some intermediate matrices
    float *d_V,              // Exposed surf area (in num of dots) 
          *d_V_s;            // Exposed surf area (in real A^2)

    float *d_WK;             // Waasmaier-Kirfel parameters 

    int   *d_close_flag,     // Flags for atoms close to an atom
          *d_close_num,      // Num of atoms close to an atom
          *d_close_idx;      // Their atomic index
 
    float *d_vdW;            // van der Waals radii

    float *d_FF_table,       // Form factors for each atom type at each q
          *d_FF_full;        /* Form factors for each atom at each q, 
                                considering the SASA an atom has. */
    
    
    // set various memory chunk sizes
    int size_coord       = 3 * num_atom * sizeof(float);
    int size_atom        = num_atom * sizeof(int);
    int size_atom2       = num_atom2 * sizeof(int);
    int size_atom2f      = num_atom2 * sizeof(float);
    int size_atom2xatom2 = 1024 * num_atom2 * sizeof(int); // For d_close_flag
    int size_q           = num_q * sizeof(float); 
    int size_qxatom2     = num_q2 * num_atom2 * sizeof(float);
    int size_FF_table    = (num_ele + 1) * num_q * sizeof(float); // +1 for solvent
    int size_WK          = 11 * num_ele * sizeof(float);
    int size_vdW         = (num_ele + 1) * sizeof(float); // +1 for solvent

    // Allocate local memories
    // S_calc = (float *)malloc(size_q);

    // Allocate cuda memories
    cudaMalloc((void **)&d_coord,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,        size_atom);
    cudaMalloc((void **)&d_q,          size_q);
    cudaMalloc((void **)&d_S_calc,     size_q);

    cudaMalloc((void **)&d_S_calcc,    size_qxatom2);
    cudaMalloc((void **)&d_V,          size_atom2f);
    cudaMalloc((void **)&d_V_s,        size_atom2f);
    cudaMalloc((void **)&d_close_flag, size_atom2xatom2);
    cudaMalloc((void **)&d_close_num,  size_atom2);
    cudaMalloc((void **)&d_close_idx,  size_atom2xatom2);
    cudaMalloc((void **)&d_vdW,        size_vdW);
    cudaMalloc((void **)&d_FF_table,   size_FF_table);
    cudaMalloc((void **)&d_FF_full,    size_qxatom2);
    cudaMalloc((void **)&d_WK,         size_WK);

    // Initialize some matrices
    cudaMemset(d_close_flag, 0,   size_qxatom2);

    cudaMemset(d_S_calc,     0.0, size_q);
    cudaMemset(d_S_calcc,    0.0, size_qxatom2);
    cudaMemset(d_close_num,  0,   size_atom2);
    cudaMemset(d_close_idx,  0,   size_atom2xatom2);
    cudaMemset(d_FF_full,    0.0, size_qxatom2);

    // Copy necessary data
    cudaMemcpy(d_coord,      coord,      size_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW,        vdW,        size_vdW,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,        Ele,        size_atom,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_q,          q,          size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,         WK,         size_WK,    cudaMemcpyHostToDevice);


    // Register what atoms are close to what atoms. 
    dist_calc<<<1024, 1024>>>(
        d_coord, 
        d_close_num, 
        d_close_flag,
        d_close_idx, 
        num_atom,
        num_atom2); 

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Based on the result list of d_close_idx, calculate surface area for each atom
    surf_calc<<<1024,512>>>(
        d_coord, 
        d_Ele, 
        d_close_num, 
        d_close_idx, 
        d_vdW, 
        num_atom, 
        num_atom2, 
        num_raster, 
        sol_s, 
        d_V);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Sum up the surface area for this snapshot, not necessary but useful.
    sum_V<<<1,1024>>>(
        d_V, 
        d_V_s, 
        num_atom, 
        num_atom2, 
        d_Ele,
        sol_s, 
        d_vdW);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Calculate the non-varying part of the form factor.
    // In the future this can be done pre-simulation.
    FF_calc<<<320, 32>>>(
        d_q, 
        d_WK, 
        d_vdW, 
        num_q, 
        num_ele, 
        c1, 
        r_m, 
        d_FF_table,
        rho);

    // Adding the surface area contribution. From this point every atom has a different form factor.
    create_FF_full_FoXS<<<320, 1024>>>(
        d_FF_table, 
        d_V,
        c2, 
        d_Ele, 
        d_FF_full, 
        num_q, 
        num_ele, 
        num_atom, 
        num_atom2);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Actually calculating scattering pattern. This kernel is for single snapshot - 
    // the force is purely based on this one structure.
    scat_calc<<<320, 1024>>>(
        d_coord, 
        d_Ele,
        d_q, 
        d_S_calc, 
        num_atom,  
        num_q,     
        num_ele,  
        d_S_calcc, 
        num_atom2, 
        d_FF_full);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaMemcpy(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);

    /*printf("S_calc: ");
    for (int ii = 0; ii < num_q; ii++) {
        printf("%.3f, ", S_calc[ii]);
    }*/
   

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaFree(d_coord); 
    cudaFree(d_Ele); 
    cudaFree(d_q);
    cudaFree(d_S_calc); 
    cudaFree(d_S_calcc); 
    cudaFree(d_V); cudaFree(d_V_s); 
    cudaFree(d_WK);
    cudaFree(d_close_flag); cudaFree(d_close_num); cudaFree(d_close_idx);
    cudaFree(d_vdW);
    cudaFree(d_FF_table); cudaFree(d_FF_full);
    gettimeofday(&tv2, NULL);
    double time_in_mill = 
         (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0 ;
    //printf("Time elapsed = %.3f ms.\n", time_in_mill);

}
}
