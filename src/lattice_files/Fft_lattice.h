#pragma once

#ifndef __FFT_LATTICE_H__
#define __FFT_LATTICE_H__






class Fft_lattice
{
    public:

    void create_lattice(float *d_latticevol, uint NX, uint NY, uint NZ, uint size, uint lattice_type_index);
    void fft_func(float2 *fft_data);
    void ifft_func(float2 *fft_data);
    void ifft_func_complex(float2 *fft_data);
    void fft_scalar(float2 *fft_data_compute,float scalar_val,int size);
    void fft_fill(float2 *fft_compute, float2 *fft_compute_fill,int Nx, int Ny , int Nz, uint mid_index);
    void indiviual_grating_sum(float2 *sum_gratings,float* fft_function,int Nx,int Ny, int Nz);
    void add_fft_constants(float2* indvidual_grating, float2 fft_constant, int index);

};


#endif /* __FFT_LATTICE_H__ */
