
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "WaasKirf.hh"
#define PI 3.14159265359



__global__ void FF_calc (
    float *q, 
    float *WK, 
    int num_q, 
    int num_ele, 
    float *FF_table
    ) {

    // Calculate the non-SASA part of form factors per element

    __shared__ float q_pt, q_WK;
    __shared__ float FF_pt[98]; // num_ele + 1, the last one for water.
    __shared__ float WK_s[1078]; 
    if (blockIdx.x >= num_q) return; // out of q range
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        q_pt = q[ii];
        q_WK = q_pt / 4.0 / PI;
        // FoXS C1 term
        for (int jj = threadIdx.x; jj < 11 * num_ele; jj += blockDim.x) {
            WK_s[jj] = WK[jj];
        } // Copy WK to shared memory for faster access
        __syncthreads();

        // Calculate Form factor for this block (or q vector)
        for (int jj = threadIdx.x; jj < num_ele; jj += blockDim.x) {
            FF_pt[jj] = WK_s[jj*11+5];
            // The part is for excluded volume
            //FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);
            for (int kk = 0; kk < 5; kk++) {
                FF_pt[jj] += WK_s[jj*11+kk] * exp(-WK_s[jj*11+kk+6] * q_WK * q_WK); 
            }
            FF_table[ii*(num_ele)+jj] = FF_pt[jj];
        }
    }
}


__global__ void create_FF_full_FoXS (
    float *FF_table, 
    int *Ele, 
    float *FF_full, 
    int num_q, 
    int num_ele, 
    int num_atom, 
    int num_atom1024) {

    // Add on SASA for each atom

    __shared__ float FF_pt[99];
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        // Get form factor for this block (or q vector)
        if (ii < num_q) {
            for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
                FF_pt[jj] = FF_table[ii*num_ele+jj];
            }
        }
        __syncthreads();
        
        // Calculate atomic form factor for this q
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            int atomt = Ele[jj];
            FF_full[ii*num_atom1024 + jj] = FF_pt[atomt];
        }
    }
}

__global__ void __launch_bounds__(1024,2) scat_calc (
    float *coord, 
    int *Ele,
    float *q,
    float *S_calc, 
    int num_atom,   
    int num_q,     
    int num_ele,   
    float *S_calcc, 
    int num_atom1024,
    float *FF_full) {

    float q_pt;

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q[ii];

        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // for every atom jj
            float atom1x = coord[3*jj];
            float atom1y = coord[3*jj+1];
            float atom1z = coord[3*jj+2];
            float S_calccs = 0.0;
            for (int kk = 0; kk < num_atom; kk++) {
                // for every atom kk
                float FF_kj = FF_full[ii * num_atom1024 + jj] * FF_full[ii * num_atom1024 + kk];
                if (q_pt == 0.0 || kk == jj) {
                    S_calccs += FF_kj;
                } else {
                    float dx = atom1x - coord[3*kk];
                    float dy = atom1y - coord[3*kk+1];
                    float dz = atom1z - coord[3*kk+2];
                    float r = sqrt(dx*dx+dy*dy+dz*dz);
                    float qr = q_pt * r; 
                    float sqr = sin(qr) / qr;
                    S_calccs += FF_kj * sqr;
                }
            }
            S_calcc[ii*blockDim.x+threadIdx.x] += S_calccs;
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * blockDim.x + iAccum] += S_calcc[ii * blockDim.x + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = S_calcc[ii * blockDim.x];
        __syncthreads();


    }
}


__global__ void __launch_bounds__(1024,2) scat_calc_oa (
    float *coord, 
    int *Ele,
    float *q,
    float *S_calc, 
    int num_atom,   
    int num_q,
    int num_ele,   
    float *S_calcc, 
    int num_atom1024,
    float *FF_full,
    int num_q_raster,
    int num_q_raster2) {

    float q_pt;

    // raster of q points
    // if user set num_q_raster > 1024,
    // it'll get reduced to 1024 before the call

    float L = sqrt(num_q_raster * PI);

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q[ii];

        for (int jj = threadIdx.x; jj < num_q_raster; jj += blockDim.x) {
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_q_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // q raster points
            float qx = q_pt * xu;
            float qy = q_pt * yu;
            float qz = q_pt * zu;
            float amp_cos = 0.0; // this q and this q raster point, summed over all atoms
            float amp_sin = 0.0; // this q and this q raster point, summed over all atoms
            for (int kk = 0; kk < num_atom; kk++) {
                float FF = FF_full[ii * num_atom1024 + kk];
                float qrx = -coord[3*kk] * qx;
                float qry = -coord[3*kk+1] * qy;
                float qrz = -coord[3*kk+2] * qz;
                float qr = qrx + qry + qrz;
                amp_cos += FF * cos(qr);
                amp_sin += FF * sin(qr);
            }
            S_calcc[ii*num_q_raster2+jj] = (amp_cos * amp_cos + amp_sin * amp_sin) / float(num_q_raster);
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_q_raster2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_q_raster2 + iAccum] += S_calcc[ii * num_q_raster2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = S_calcc[ii * num_q_raster2];
        __syncthreads();


    }
}


__global__ void __launch_bounds__(1024,2) scat_calc_xoa (
    float *coord1, 
    float *coord2, 
    int *Ele1,
    int *Ele2,
    float *weight,
    float *q,
    float *S_calc1, 
    float *S_calc2, 
    float *S_calc12, 
    int num_atom1,   
    int num_atom2,   
    int num_q,
    int num_ele,   
    float *S_calcc1, 
    float *S_calcc2, 
    float *S_calcc12, 
    int num_atom1_1024,
    int num_atom2_1024,
    float *FF_full1,
    float *FF_full2,
    int num_q_raster,
    int num_q_raster2) {

    float q_pt;

    // raster of q points
    // if user set num_q_raster > 1024,
    // it'll get reduced to 1024 before the call

    float L = sqrt(num_q_raster * PI);

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        q_pt = q[ii];

        for (int jj = threadIdx.x; jj < num_q_raster; jj += blockDim.x) {
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_q_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // q raster points
            float qx = q_pt * xu;
            float qy = q_pt * yu;
            float qz = q_pt * zu;
            float amp_cos1 = 0.0; // this q and this q raster point, summed over all atoms
            float amp_sin1 = 0.0; // this q and this q raster point, summed over all atoms
            float amp_cos2 = 0.0; // this q and this q raster point, summed over all atoms
            float amp_sin2 = 0.0; // this q and this q raster point, summed over all atoms
            for (int kk = 0; kk < num_atom1; kk++) {
                float FF1 = FF_full1[ii * num_atom1_1024 + kk];
                float W1 = weight[kk];
                float qrx1 = -coord1[3*kk] * qx;
                float qry1 = -coord1[3*kk+1] * qy;
                float qrz1 = -coord1[3*kk+2] * qz;
                float qr1 = qrx1 + qry1 + qrz1;
                amp_cos1 += W1 * FF1 * cos(qr1);
                amp_sin1 += W1 * FF1 * sin(qr1);
            }
            for (int ll = 0; ll < num_atom2; ll++) {
                float FF2 = FF_full2[ii * num_atom2_1024 + ll];
                float qrx2 = -coord2[3*ll] * qx;
                float qry2 = -coord2[3*ll+1] * qy;
                float qrz2 = -coord2[3*ll+2] * qz;
                float qr2 = qrx2 + qry2 + qrz2;
                amp_cos2 += FF2 * cos(qr2);
                amp_sin2 += FF2 * sin(qr2);
            }
            S_calcc1[ii*num_q_raster2+jj] = (amp_cos1 * amp_cos1 + amp_sin1 * amp_sin1) / float(num_q_raster);
            S_calcc2[ii*num_q_raster2+jj] = (amp_cos2 * amp_cos2 + amp_sin2 * amp_sin2) / float(num_q_raster);
            S_calcc12[ii*num_q_raster2+jj] = 2.0 * (amp_cos1 * amp_cos2 + amp_sin1 * amp_sin2) / float(num_q_raster);
        }
        
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_q_raster2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc1[ii * num_q_raster2 + iAccum] += S_calcc1[ii * num_q_raster2 + stride + iAccum];
            }
        }
        for (int stride = num_q_raster2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc2[ii * num_q_raster2 + iAccum] += S_calcc2[ii * num_q_raster2 + stride + iAccum];
            }
        }
        for (int stride = num_q_raster2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc12[ii * num_q_raster2 + iAccum] += S_calcc12[ii * num_q_raster2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc1[ii] = S_calcc1[ii * num_q_raster2];
        S_calc2[ii] = S_calcc2[ii * num_q_raster2];
        S_calc12[ii] = S_calcc12[ii * num_q_raster2];
        __syncthreads();


    }
}
