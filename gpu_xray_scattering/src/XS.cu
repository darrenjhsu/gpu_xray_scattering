
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
    int use_oa,      // do orientational averaging literally (calls scat_calc_oa)
    int num_q_raster // number of q_raster points when using scat_calc_oa
) {
// This function is called by tclforce script from NAMD. Note that it is only executed every delta_t steps.
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    // In this code pointers with d_ are device pointers. 
    int num_atom1024  = (num_atom + 1023) / 1024 * 1024;
    int num_q2        = (num_q + 31) / 32 * 32;
    int num_q_raster2 = (num_q_raster + 2047) / 2048 * 2048;

    //printf("num_atom = %d, num_atom2 = %d, num_q2 = %d, num_raster = %d, num_q_raster = %d, num_q_raster2 = %d\n", num_atom, num_atom2, num_q2, num_raster, num_q_raster, num_q_raster2);

    // Declare cuda pointers //
    float *d_coord;          // Coordinates 3 x num_atom

    int   *d_Ele;            // Element list.
    float *d_q;              // q vector
    float *d_S_calc;         // Calculated scattering curve
    float *d_S_calcc;        // Some intermediate matrices

    float *d_WK;             // Waasmaier-Kirfel parameters 

    float *d_FF_table,       // Form factors for each atom type at each q
          *d_FF_full;        /* Form factors for each atom at each q, 
                                considering the SASA an atom has. */
    
    
    // set various memory chunk sizes
    int size_coord       = 3 * num_atom * sizeof(float);
    int size_atom        = num_atom * sizeof(int);
    int size_atomf       = num_atom * sizeof(float);
    int size_q           = num_q * sizeof(float); 
    int size_qxatom2     = num_q2 * 1024 * sizeof(float);
    int size_qxqraster2  = num_q2 * num_q_raster2 * sizeof(float);
    int size_FF_table    = (num_ele) * num_q * sizeof(float);
    int size_FF_full     = num_q * num_atom1024 * sizeof(float);
    int size_WK          = 11 * num_ele * sizeof(float);


    // Allocate cuda memories
    cudaMalloc((void **)&d_coord,      size_coord); // 40 KB
    cudaMalloc((void **)&d_Ele,        size_atom);
    cudaMalloc((void **)&d_q,          size_q);
    cudaMalloc((void **)&d_S_calc,     size_q);
    if (use_oa == 0) {
        cudaMalloc((void **)&d_S_calcc,    size_qxatom2);
    } else {
        cudaMalloc((void **)&d_S_calcc,    size_qxqraster2);
    }
    cudaMalloc((void **)&d_FF_table,   size_FF_table);
    cudaMalloc((void **)&d_FF_full,    size_FF_full);
    cudaMalloc((void **)&d_WK,         size_WK);

    // Initialize some matrices
    
    cudaMemset(d_FF_full,    0.0, size_FF_full);

    cudaMemset(d_S_calc,     0.0, size_q);
    if (use_oa == 0) {
        cudaMemset(d_S_calcc,    0.0, size_qxatom2);
    } else {
        cudaMemset(d_S_calcc,    0.0, size_qxqraster2);
    }

    // Copy necessary data
    cudaMemcpy(d_coord,      coord,      size_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele,        Ele,        size_atom,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_q,          q,          size_q,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,         WK,         size_WK,    cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: setting memory, %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Register what atoms are close to what atoms. 

    // Calculate the non-varying part of the form factor.
    // In the future this can be done pre-simulation.
    FF_calc<<<num_q, 32>>>(
        d_q, 
        d_WK, 
        num_q, 
        num_ele, 
        d_FF_table
        );

    // Adding the surface area contribution. From this point every atom has a different form factor.
    create_FF_full_FoXS<<<num_q, 1024>>>(
        d_FF_table, 
        d_Ele, 
        d_FF_full, 
        num_q, 
        num_ele, 
        num_atom, 
        num_atom1024);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: FF %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Actually calculating scattering pattern. This kernel is for single snapshot  
    if (use_oa == 0) {
        printf("Using vanilla method\n");
        scat_calc<<<num_q, 1024>>>(
            d_coord, 
            d_Ele,
            d_q, 
            d_S_calc, 
            num_atom,  
            num_q,     
            num_ele,  
            d_S_calcc, 
            num_atom1024, 
            d_FF_full);
    } else if (use_oa == 1) {
        printf("Using orientational averaging method 1\n");
        scat_calc_oa<<<num_q, 1024>>>(
            d_coord, 
            d_Ele,
            d_q, 
            d_S_calc, 
            num_atom,  
            num_q,     
            num_ele,  
            d_S_calcc, 
            num_atom1024, 
            d_FF_full,
            num_q_raster,
            num_q_raster2);
    }
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: scat_calc %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaMemcpy(S_calc, d_S_calc, size_q,     cudaMemcpyDeviceToHost);

    //printf("S_calc: ");
    //for (int ii = 0; ii < num_q; ii++) {
    //    printf("%.3f, ", S_calc[ii]);
    //}
   

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
    cudaFree(d_WK);
    cudaFree(d_FF_table); cudaFree(d_FF_full);
    gettimeofday(&tv2, NULL);
    double time_in_mill = 
         (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0 ;
    //printf("Time elapsed = %.3f ms.\n", time_in_mill);

}
}
