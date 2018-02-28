#include <complex.h>
#include <math.h>
#include <stdio.h>

// Complex type
typedef double complex cmplx;

void pol3(
    cmplx* out, // Array to save the polarizability
    const int freq_size, const double freq_min, const double freq_max, // Frequency array
    const int comb_size, const double delta_freq,
    const double gamma, const double M_field1, const double M_field2, const double M_field3, // Comb parameters
    const cmplx wg_nv, const cmplx wg_mv, const cmplx wg_vl // omega_{ij} + I * gamma_{ij} for each transition from i to j
)
{
    #pragma omp parallel for
    for(int out_i = 0; out_i < freq_size; out_i++)
    {
        const double omega = freq_min + out_i * (freq_max - freq_min) / (freq_size - 1);

        cmplx result = 0. + 0. * I;

        const cmplx term_A = conj(wg_nv) - omega;

        for(int p = -comb_size; p < comb_size; p++)
        {
            const cmplx term_P = M_field1 + p * delta_freq - gamma * I;
            for(int q = -comb_size; q < comb_size; q++)
            {
                const cmplx term_B = omega - M_field2 - q*delta_freq + gamma * I - conj(wg_mv),
                            term_R = conj(wg_mv) - M_field2 - q * delta_freq - gamma * I;
                for(int r = -comb_size; r < comb_size; r++)
                {
                    const cmplx term_Q = omega - M_field2 - M_field3 - (q + r) * delta_freq - 2. * gamma * I,
                                term_s = omega - M_field1 - M_field2 - M_field3 - (p + q + r) * delta_freq,
                                term_W = -2. * term_s / (pow(term_s, 2) + 9.*pow(gamma, 2));

                    //////////////////////////////////////////////////////////////////////////////////////////////
                    //
                    // A2 diagram
                    //
                    /////////////////////////////////////////////////////////////////////////////////////////////
                    result += (term_W + (conj(term_Q)-term_R)/((conj(term_P) - term_Q)*(conj(term_P) - term_R)))/
                            (term_A*term_B*(conj(term_P) - conj(term_Q))*(conj(term_Q) + wg_vl));
                }

            }
        }

        out[out_i] = result;
    }

}
