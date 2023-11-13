
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
void cross_xray_scattering (
    int num_atom1,    // number of atoms Na
    float *coord1,    // coordinates     3Na
    int num_atom2,
    float *coord2,
    int *Ele1,      // Element
    int *Ele2,
    float *vdW1, 
    float *vdW2,
    float *weight1,   // Weights of the protein and prior points
    float *weight2,   // Weights of the protein and prior points
    int num_q,       // number of q vector points
    float *q,        // q vector        Nq
    float *S_calc1,   // scattering intensity to be returned, Nq
    float *S_calc2,
    float *S_calc12,
    int num_q_raster,// number of q_raster points when using scat_calc_oa
    float rho
) {
// This function is called by tclforce script from NAMD. Note that it is only executed every delta_t steps.
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    // In this code pointers with d_ are device pointers. 
    int num_atom1_pad = (num_atom1 + 512) / 512 * 512;
    int num_atom2_pad = (num_atom2 + 31) / 32 * 32;
    int num_q2        = (num_q + 31) / 32 * 32;
    int num_q_raster2 = (num_q_raster + 1023) / 1024 * 1024;


    // Declare cuda pointers //
    float *d_coord1;          // Coordinates 3 x num_atom
    float *d_coord2;
    int   *d_Ele1;            // Element list.
    int   *d_Ele2;
    float *d_vdW1;
    float *d_vdW2;
    float *d_weight1;
    float *d_weight2;
    float *d_q;              // q vector
    float *d_S_calc1;         // Calculated scattering curve
    float *d_S_calcc1;        // Some intermediate matrices
    float *d_S_calc2;         // Calculated scattering curve
    float *d_S_calcc2;        // Some intermediate matrices
    float *d_S_calc12;         // Calculated scattering curve
    float *d_S_calcc12;        // Some intermediate matrices

    float *d_WK;             // Waasmaier-Kirfel parameters 

    float *d_FF_table,       // Form factors for each atom type at each q
          *d_FF_full1,        /* Form factors for each atom at each q, 
                                considering the SASA an atom has. */
          *d_FF_full2;
    
    // set various memory chunk sizes
    int size_coord1      = 3 * num_atom1 * sizeof(float);
    int size_coord2      = 3 * num_atom2 * sizeof(float);
    int size_atom1       = num_atom1 * sizeof(int);
    int size_atom1f      = num_atom1 * sizeof(float);
    int size_atom2       = num_atom2 * sizeof(int);
    int size_atom2f      = num_atom2 * sizeof(float);
    int size_q           = num_q * sizeof(float); 
    int size_qxqraster2  = num_q2 * num_q_raster2 * sizeof(float);
    int size_FF_table    = (num_ele) * num_q * sizeof(float);
    int size_FF_full1    = num_q * num_atom1_pad * sizeof(float);
    int size_FF_full2    = num_q * num_atom2_pad * sizeof(float);
    int size_WK          = 11 * num_ele * sizeof(float);


    // Allocate cuda memories
    cudaMalloc((void **)&d_coord1,      size_coord1); // 40 KB
    cudaMalloc((void **)&d_coord2,      size_coord2); // 40 KB
    cudaMalloc((void **)&d_Ele1,        size_atom1);
    cudaMalloc((void **)&d_Ele2,        size_atom2);
    cudaMalloc((void **)&d_vdW1,        size_atom1f);
    cudaMalloc((void **)&d_vdW2,        size_atom2f);
    cudaMalloc((void **)&d_weight1,     size_atom1f);
    cudaMalloc((void **)&d_weight2,     size_atom2f);
    cudaMalloc((void **)&d_q,           size_q);
    cudaMalloc((void **)&d_S_calc1,     size_q);
    cudaMalloc((void **)&d_S_calc2,     size_q);
    cudaMalloc((void **)&d_S_calc12,    size_q);
    cudaMalloc((void **)&d_S_calcc1,    size_qxqraster2);
    cudaMalloc((void **)&d_S_calcc2,    size_qxqraster2);
    cudaMalloc((void **)&d_S_calcc12,   size_qxqraster2);
    cudaMalloc((void **)&d_FF_table,    size_FF_table);
    cudaMalloc((void **)&d_FF_full1,    size_FF_full1);
    cudaMalloc((void **)&d_FF_full2,    size_FF_full2);
    cudaMalloc((void **)&d_WK,          size_WK);

    // Initialize some matrices
    
    cudaMemset(d_FF_full1,    0.0, size_FF_full1);
    cudaMemset(d_FF_full2,    0.0, size_FF_full2);

    cudaMemset(d_S_calc1,     0.0, size_q);
    cudaMemset(d_S_calc2,     0.0, size_q);
    cudaMemset(d_S_calc12,    0.0, size_q);
    cudaMemset(d_S_calcc1,    0.0, size_qxqraster2);
    cudaMemset(d_S_calcc2,    0.0, size_qxqraster2);
    cudaMemset(d_S_calcc12,   0.0, size_qxqraster2);

    // Copy necessary data
    cudaMemcpy(d_coord1,     coord1,     size_coord1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord2,     coord2,     size_coord2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele1,       Ele1,       size_atom1,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ele2,       Ele2,       size_atom2,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW1,       vdW1,       size_atom1f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vdW2,       vdW2,       size_atom2f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight1,    weight1,    size_atom1f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2,    weight2,    size_atom2f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q,          q,          size_q,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_WK,         WK,         size_WK,     cudaMemcpyHostToDevice);

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
    FF_calc<<<num_q, 1024>>>(
        d_q, 
        d_WK,
        d_vdW1,
        d_Ele1,
        d_FF_table,
        d_FF_full1,
        num_q, 
        num_ele, 
        num_atom1,
        num_atom1_pad,
        rho
        );

    FF_calc<<<num_q, 1024>>>(
        d_q, 
        d_WK,
        d_vdW2,
        d_Ele2,
        d_FF_table,
        d_FF_full2,
        num_q, 
        num_ele, 
        num_atom2,
        num_atom2_pad,
        rho
        );

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: FF %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    // Actually calculating scattering pattern. This kernel is for single snapshot  
    //printf("Using orientational averaging method 1\n");
    scat_calc_xoa<<<num_q, 1024>>>(
        d_coord1, 
        d_coord2,
        d_Ele1,
        d_Ele2,
        d_weight1,
        d_weight2,
        d_q, 
        d_S_calc1, 
        d_S_calc2, 
        d_S_calc12, 
        num_atom1,
        num_atom2,
        num_q,     
        num_ele,
        d_S_calcc1, 
        d_S_calcc2, 
        d_S_calcc12, 
        num_atom1_pad,
        num_atom2_pad,
        d_FF_full1,
        d_FF_full2,
        num_q_raster,
        num_q_raster2);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: scat_calc %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaMemcpy(S_calc1, d_S_calc1, size_q,     cudaMemcpyDeviceToHost);
    cudaMemcpy(S_calc2, d_S_calc2, size_q,     cudaMemcpyDeviceToHost);
    cudaMemcpy(S_calc12, d_S_calc12, size_q,     cudaMemcpyDeviceToHost);
   

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
       fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
       exit(-1);
    }

    cudaFree(d_coord1); 
    cudaFree(d_coord2); 
    cudaFree(d_Ele1); 
    cudaFree(d_Ele2); 
    cudaFree(d_vdW1); 
    cudaFree(d_vdW2); 
    cudaFree(d_weight1); 
    cudaFree(d_weight2); 
    cudaFree(d_q);
    cudaFree(d_S_calc1); 
    cudaFree(d_S_calc2); 
    cudaFree(d_S_calc12); 
    cudaFree(d_S_calcc1); 
    cudaFree(d_S_calcc2); 
    cudaFree(d_S_calcc12); 
    cudaFree(d_WK);
    cudaFree(d_FF_table); 
    cudaFree(d_FF_full1);
    cudaFree(d_FF_full2);
    gettimeofday(&tv2, NULL);
    double time_in_mill = 
         (tv2.tv_sec - tv1.tv_sec) * 1000.0 + (tv2.tv_usec - tv1.tv_usec) / 1000.0 ;

}
}