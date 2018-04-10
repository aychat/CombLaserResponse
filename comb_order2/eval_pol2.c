#include <complex.h>
#include <math.h>
#include <stdio.h>

// Complex type
typedef double complex cmplx;

void pol2_1(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_i, const double M_field_j, // Comb parameters
    const cmplx wg_2, const cmplx wg_1, int sign // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
    {
        const double omega = freq[out_i];

        cmplx result = 0. + 0. * I;

        const cmplx term_D = omega - wg_2;

        for(int i = -comb_size; i < comb_size; i++)
        {
            const cmplx term_X = M_field_i + i * delta_freq + gamma * I - wg_1;
            for(int j = -comb_size; j < comb_size; j++)
            {
                const cmplx term_Y = omega - M_field_j - j * delta_freq + gamma * I - wg_1,
                            term_Z = M_field_i + M_field_j + (i+j) * delta_freq - omega + 2. * gamma * I,
                            term_Z_star = -conj(term_Z);

                result += (1./term_D)*(1./(term_X * term_Y) + 1./(term_Y * term_Z_star) + 1./(term_X * term_Z));
            }

        }

        out[out_i] += sign*(-M_PI/2.)*I*result;
    }

}

void pol2_2(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_i, const double M_field_j, // Comb parameters
    const cmplx wg_2, const cmplx wg_1, int sign // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
    {
        const double omega = freq[out_i];

        cmplx result = 0. + 0. * I;

        const cmplx term_D = omega - wg_2;

        for(int i = -comb_size; i < comb_size; i++)
        {
            const cmplx term_X = omega - M_field_i - i * delta_freq + gamma * I - wg_1;
            for(int j = -comb_size; j < comb_size; j++)
            {
                const cmplx term_Y = M_field_j + j * delta_freq + gamma * I - wg_1,
                            term_Z = M_field_i + M_field_j + (i+j) * delta_freq - omega + 2. * gamma * I,
                            term_Z_star = -conj(term_Z);

                result += (1./term_D)*(1./(term_X * term_Y) + 1./(term_Y * term_Z_star) + 1./(term_X * term_Z));
            }

        }

        out[out_i] += sign*(-M_PI/2.)*I*result;
    }

}
void pol2_total(
    cmplx* out, // Array to save the polarizability
    const double* freq, const int freq_size, // Frequency array
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field_i, const double M_field_j, // Comb parameters
    const cmplx wg_nl, const cmplx wg_ml, const cmplx wg_mn, const cmplx wg_nm // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
{
//    pol2_1(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_i, M_field_j, wg_nl, wg_ml, 1);
//    pol2_2(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_i, M_field_j, wg_nl, wg_ml, 1);
//    pol2_1(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_i, M_field_j, wg_mn, wg_nl, -1);
//    pol2_2(out, freq, freq_size, comb_size, delta_freq, gamma, M_field_i, M_field_j, wg_mn, wg_nl, -1);
}