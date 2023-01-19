
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "WaasKirf.hh"
#define PI 3.14159265359


__global__ void dist_calc (
    float *coord, 
    int *close_num,
    int *close_flag, 
    int *close_idx, 
    int num_atom, 
    int num_atom2) {

    // close_flag is a 1024 x num_atom2 int matrix initialized to 0.
    // close_idx: A 128 x num_atom2 int matrix, row i of which only the first close_num[i] elements are defined. (Otherwise it's -1). 
    __shared__ float x_ref, y_ref, z_ref;
    __shared__ int idz;
    __shared__ int temp[2048];

    __shared__ int round;
    round = (num_atom2 > num_atom) ? 1 : (num_atom + num_atom2 - 1) / num_atom2;
    /*if (blockIdx.x == 0) {
        if (threadIdx.x == 0) printf("Number of rounds: %d\n", round);
    }*/
    // Calc distance
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        if (threadIdx.x == 0) {
            x_ref = coord[3*ii  ];
            y_ref = coord[3*ii+1];
            z_ref = coord[3*ii+2];
        }
        int idy = ii % gridDim.x; // This will be what row of close_flag this block is putting its value in.
        __syncthreads();

        for (int rr = 0; rr < round; rr++) {
            for (int jj = threadIdx.x; jj < num_atom2; jj += blockDim.x) { // flush close_flag
                close_flag[idy*num_atom2+jj] = 0;
            }

            for (int jj = threadIdx.x + rr * num_atom2; jj < num_atom; jj += blockDim.x) {
                float r2t = (coord[3*jj  ] - x_ref) * (coord[3*jj  ] - x_ref) + 
                            (coord[3*jj+1] - y_ref) * (coord[3*jj+1] - y_ref) + 
                            (coord[3*jj+2] - z_ref) * (coord[3*jj+2] - z_ref); 
     
                if (r2t < 34.0) {
                    close_flag[idy*num_atom2+jj % num_atom2] = 1; // roughly 2 A + 2 A vdW + 2 * 1.8 A probe
                } else { 
                    close_flag[idy*num_atom2+jj % num_atom2] = 0;
                }
                if (ii == jj) close_flag[idy*num_atom2+jj % num_atom2] = 0;
            }
            __syncthreads();
    
            // Prefix sum: Do pre scan
            idz = 0;
            int temp_sum = 0;
            for (int jj = threadIdx.x; jj < num_atom2; jj += 2 * blockDim.x) {
                int idx = jj % blockDim.x; 
                int offset = 1;
                temp[2 * idx]     = close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx];
                temp[2 * idx + 1] = close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx + 1];
                for (int d = 2 * blockDim.x>>1; d > 0; d >>= 1) { // up-sweep
                    __syncthreads();
                    if (idx < d) {
                        int ai = offset * (2 * idx + 1) - 1;
                        int bi = offset * (2 * idx + 2) - 1;
                        temp[bi] += temp[ai];
                    }
                    offset *= 2;
                }
                __syncthreads();
                temp_sum = close_num[ii];
                __syncthreads();
                if (idx == 0) {
                    close_num[ii] += temp[2 * blockDim.x - 1]; // log the total number of 1's in this blockDim
                    temp[2 * blockDim.x - 1] = 0;
                }
                __syncthreads();
                for (int d = 1; d < blockDim.x * 2; d *= 2) { //down-sweep
                    offset >>= 1;
                    __syncthreads();
                    if (idx < d) {
                        int ai = offset * (2 * idx + 1) - 1;
                        int bi = offset * (2 * idx + 2) - 1;
                        int t    = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                }
            
                __syncthreads();
            
                // Finally assign the indices
                if (close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx] == 1) {
                    close_idx[ii * 128 + temp[2*idx] + temp_sum] = 2 * idx + 2 * blockDim.x * idz;
                }
                if (close_flag[idy * num_atom2 + 2 * blockDim.x * idz + 2 * idx + 1] == 1) {
                    close_idx[ii * 128 + temp[2*idx+1] + temp_sum] = 2 * idx + 1 + 2 * blockDim.x * idz;
                }
                idz++;
                __syncthreads();
            } // prefix sum loop
        } // round loop
    } // block loop
} // function


__global__ void __launch_bounds__(512,4) surf_calc (
    float *coord, 
    int *Ele, 
    int *close_num, 
    int *close_idx, 
    float *vdW, 
    int num_atom, 
    int num_atom2, 
    int num_raster, 
    float sol_s, 
    float *V) {

    // num_raster should be a number of 2^n. 
    // sol_s is solvent radius (default = 1.8 A)
    __shared__ float vdW_s; // vdW radius of the center atom
    __shared__ int pts[512]; // All spherical raster points
    __shared__ float L, r;
    
    if (blockIdx.x >= num_atom) return;
    L = sqrt(num_raster * PI);
    for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
        int atom1t = Ele[ii];
        vdW_s = vdW[atom1t];
        r = vdW_s + sol_s;
        for (int jj = threadIdx.x; jj < num_raster; jj += blockDim.x) {
            int pt = 1;
            
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // vdW points
            float x = vdW_s * xu + coord[3*ii];
            float y = vdW_s * yu + coord[3*ii+1];
            float z = vdW_s * zu + coord[3*ii+2];
            // Solvent center
            float x2 = r * xu + coord[3*ii];
            float y2 = r * yu + coord[3*ii+1];
            float z2 = r * zu + coord[3*ii+2];
            for (int kk = 0; kk < close_num[ii]; kk++) {
                int atom2i = close_idx[ii * 128 + kk];
                int atom2t = Ele[atom2i];
                float dx = (x - coord[3*atom2i]);
                float dy = (y - coord[3*atom2i+1]);
                float dz = (z - coord[3*atom2i+2]);
                float dr2 = dx * dx + dy * dy + dz * dz; 
                float dx2 = (x2 - coord[3*atom2i]);
                float dy2 = (y2 - coord[3*atom2i+1]);
                float dz2 = (z2 - coord[3*atom2i+2]);
                float dr22 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                // vdW points must not cross into other atom
                if (dr2 < vdW[atom2t] * vdW[atom2t]) pt = 0; //pts[jj] = 0;
                // solvent center has to be far enough
                if (dr22 < (vdW[atom2t]+sol_s) * (vdW[atom2t]+sol_s)) pt = 0; //pts[jj] = 0;
                
            }
            pts[jj] = pt;
        }
        // Sum pts == 1, calc surf area and assign to V[ii]
        for (int stride = num_raster / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                pts[iAccum] += pts[stride + iAccum];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            V[ii] = (float)pts[0]/(float)num_raster;// * 4.0 * r * r * PI ;
        }
    }
}


__global__ void sum_V (
    float *V, 
    float *V_s, 
    int num_atom, 
    int num_atom2, 
    int *Ele,
    float sol_s, 
    float *vdW) {

    for (int ii = threadIdx.x; ii < num_atom2; ii += blockDim.x) {
        if (ii < num_atom) {
            int atomi = Ele[ii];
            V_s[ii] = V[ii] * 4.0 * PI * (vdW[atomi]+sol_s) * (vdW[atomi]+sol_s);
        } else {
            V_s[ii] = 0.0;
        }
    }
    for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
            V_s[iAccum] += V_s[stride + iAccum];
        }
    }
    __syncthreads();
    //if (threadIdx.x == 0) printf("Convex contact area = %.3f A^2.\n", V_s[0]);
}

__global__ void FF_calc (
    float *q, 
    float *WK, 
    float *vdW, 
    int num_q, 
    int num_ele, 
    float c1, 
    float r_m, 
    float *FF_table,
    float rho) {

    // Calculate the non-SASA part of form factors per element

    __shared__ float q_pt, q_WK, C1, expC1;
    __shared__ float FF_pt[99]; // num_ele + 1, the last one for water.
    __shared__ float vdW_s[99];
    __shared__ float WK_s[1078]; 
    __shared__ float C1_PI_43_rho;
    if (blockIdx.x >= num_q) return; // out of q range
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        q_pt = q[ii];
        q_WK = q_pt / 4.0 / PI;
        // FoXS C1 term
        expC1 = -powf(4.0 * PI / 3.0, 1.5) * q_WK * q_WK * r_m * r_m * (c1 * c1 - 1.0) / 4.0 / PI;
        C1 = powf(c1,3) * exp(expC1);
        C1_PI_43_rho = C1 * PI * 4.0 / 3.0 * rho;
        for (int jj = threadIdx.x; jj < 11 * num_ele; jj += blockDim.x) {
            WK_s[jj] = WK[jj];
        } // Copy WK to shared memory for faster access
        __syncthreads();

        // Calculate Form factor for this block (or q vector)
        for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
            vdW_s[jj] = vdW[jj];
            if (jj == num_ele) {
                // water
                FF_pt[jj] = WK_s[7*11+5];  // Oxygen
                FF_pt[jj] += 2.0 * WK_s[5];  // Hydrogen
                FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * powf(4.0/3.0*PI, 2.0/3.0) * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);  // Water vdW_s
                for (int kk = 0; kk < 5; kk ++) {
                    FF_pt[jj] += WK_s[7*11+kk] * exp(-WK_s[7*11+kk+6] * q_WK * q_WK); // Oxygen
                    FF_pt[jj] += WK_s[kk] * exp(-WK_s[kk+6] * q_WK * q_WK) * 2.0; // Hydrogen
                }
            } else { 
                FF_pt[jj] = WK_s[jj*11+5];
                // The part is for excluded volume
                FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * powf(4.0/3.0*PI, 2.0/3.0) * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);  // Water vdW_s
                //FF_pt[jj] -= C1_PI_43_rho * powf(vdW_s[jj],3.0) * exp(-PI * vdW_s[jj] * vdW_s[jj] * q_WK * q_WK);
                for (int kk = 0; kk < 5; kk++) {
                    FF_pt[jj] += WK_s[jj*11+kk] * exp(-WK_s[jj*11+kk+6] * q_WK * q_WK); 
                }
            }
            FF_table[ii*(num_ele+1)+jj] = FF_pt[jj];
        }
    }
}


__global__ void create_FF_full_FoXS (
    float *FF_table, 
    float *V, 
    float c2, 
    int *Ele, 
    float *FF_full, 
    int num_q, 
    int num_ele, 
    int num_atom, 
    int num_atom1024) {

    // Add on SASA for each atom

    __shared__ float FF_pt[99];
    float hydration;
    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

        // Get form factor for this block (or q vector)
        if (ii < num_q) {
            for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
                FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
            }
        }
        __syncthreads();
        
        // In FoXS since c2 remains the same for all elements it is reduced to one value.
        hydration = c2 * FF_pt[num_ele];
        
        // Calculate atomic form factor for this q
        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            int atomt = Ele[jj];
            FF_full[ii*num_atom1024 + jj] = FF_pt[atomt] + hydration * V[jj];
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
    __shared__ float q_raster[3072];
    __shared__ float L;
    
    L = sqrt(num_q_raster * PI);

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
            q_raster[3*jj] = q_pt * xu;
            q_raster[3*jj+1] = q_pt * yu;
            q_raster[3*jj+2] = q_pt * zu;
        }
        __syncthreads();
        for (int jj = threadIdx.x; jj < num_q_raster; jj += blockDim.x) {
            // for every atom jj
            float qx = q_raster[3*jj];
            float qy = q_raster[3*jj+1];
            float qz = q_raster[3*jj+2];
            float amp_cos = 0.0; // this q and this q raster point, summed over all atoms
            float amp_sin = 0.0; // this q and this q raster point, summed over all atoms
            for (int kk = 0; kk < num_atom; kk++) {
                float FF = FF_full[ii * num_atom1024 + kk];
                float qrx = -coord[3*kk] * qx;
                float qry = -coord[3*kk+1] * qy;
                float qrz = -coord[3*kk+2] * qz;
                float qr = qrx+qry+qrz;
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

__global__ void __launch_bounds__(1024,2) scat_calc_oa2 (
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
    __shared__ float L;
    
    L = sqrt(num_q_raster * PI);

    // raster of q points
    // if user set num_q_raster > 1024,
    // it'll get reduced to 1024 before the call
    __shared__ float q_raster[3072];
    __shared__ float amp_cos[1024];
    __shared__ float amp_sin[1024];
    

    for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
        for (int jj = threadIdx.x; jj < 1024; jj += blockDim.x) {
            amp_cos[jj] = 0.0;
            amp_sin[jj] = 0.0;
        }
        __syncthreads();

        q_pt = q[ii];
        for (int jj = threadIdx.x; jj < num_q_raster; jj += blockDim.x) {
            float h = 1.0 - (2.0 * (float)jj + 1.0) / (float)num_q_raster;
            float p = acos(h);
            float t = L * p; 
            float xu = sin(p) * cos(t);
            float yu = sin(p) * sin(t);
            float zu = cos(p);
            // q raster points
            q_raster[jj*3] = q_pt * xu;
            q_raster[jj*3+1] = q_pt * yu;
            q_raster[jj*3+2] = q_pt * zu;
        }
        __syncthreads(); 

        for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
            // for every atom jj
            float atomx = coord[3*jj];
            float atomy = coord[3*jj+1];
            float atomz = coord[3*jj+2];
            float FF = FF_full[ii * num_atom1024 + jj];
            for (int kk = 0; kk < num_q_raster; kk++) {
                float qrx = -atomx * q_raster[3*kk];
                float qry = -atomy * q_raster[3*kk+1];
                float qrz = -atomz * q_raster[3*kk+2];
                float qr = qrx+qry+qrz;
                float amp_cos_one = FF * cos(qr);
                float amp_sin_one = FF * sin(qr);
                atomicAdd(&amp_cos[kk], amp_cos_one);
                atomicAdd(&amp_sin[kk], amp_sin_one);
            }
        }

        __syncthreads();
        for (int jj = threadIdx.x; jj < num_q_raster; jj += blockDim.x) {
           S_calcc[ii*num_q_raster2+jj] = (amp_cos[jj] * amp_cos[jj] + amp_sin[jj] * amp_sin[jj]) / float(num_q_raster);
        }    
        // Tree-like summation of S_calcc to get S_calc
        for (int stride = num_q_raster2 / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
                S_calcc[ii * num_q_raster2 + iAccum] += S_calcc[ii * num_q_raster2 + stride + iAccum];
            }
        }
        __syncthreads();
        
        S_calc[ii] = S_calcc[ii * num_q_raster2];
        __syncthreads();


    }
}
