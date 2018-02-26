#include<math.h>
#include<stdio.h>
#include<complex.h>

struct molecule {
   double w_21;
   double g_20;
   double g_32;
   double g_31;
   double w_23;
   double g_10;
   double g_13;
   double g_12;
   double g_02;
   double g_03;
   double g_01;
   double w_20;
   double w_32;
   double w_31;
   double w_30;
   double g_21;
   double omega_M1;
   double w_13;
   double w_12;
   double w_02;
   double w_03;
   double w_01;
   double g_30;
   double w_10;
   double g_23;
   double omega_M2;
   double gamma;
};

int eval(double* out_r, double* out_i, int size_out, int* index, int x_num, double x_min, double x_max, double h_min, double h_max, struct molecule m)
{
    const double dx = (x_max - x_min) / (x_num - 1);
    printf("%d %d %d \n", index[0], index[1], index[2]);
    printf("%3.2lf \n", m.w_10);

    #pragma omp parallel for
    for(int h_i = 0; h_i < size_out; h_i++){

        const double h = h_min + (h_max - h_min) * h_i / (size_out - 1);
        double complex result = 0. + 0. * I;

        for(double X = x_min; X < x_max + dx; X += dx)
            for(double Y = x_min; Y < x_max + dx; Y += dx)
                for (double Z = x_min; Z < x_max + dx; Z += dx)
                    result += exp(-pow(2. * Z + X + pow(Y, 2) - h, 2)) * (sin(X + Y + 3. * Z + h) * I + pow(Y + Z + h, 2));

        out_r[h_i] = creal(result);
        out_i[h_i] = cimag(result);
    }

    return 0;
}

int calculate_pol3_a2(double* out_r, double* out_i, const int freq_size, const double freq_min, const double freq_max,
        const int* index_permute, const int* index_fields, const int comb_num, const double del_freq, struct molecule m)
{
    double omega_M1, omega_M2, omega_M3;
    double w_n0, w_vn, w_m0, w_nv, w_mv, w_vl, g_n0, g_vn, g_m0, g_nv, g_mv, g_vl;
    int p, q, r;

   /* ------------------------------------------------------------------------------------------------------
    HAVE TO ASSIGN VALUES TO ABOVE VARIABLES DEPENDING ON THE INDICES CORRESPONDING TO POSSIBLE PERMUTATIONS
    -------------------------------------------------------------------------------------------------------*/

    int h_i = 0;
    for(int freq_i = freq_min; freq_i < freq_max + del_freq; freq_i += del_freq){

        h_i += 1;
        double complex result = 0 + 0 * I;
        double complex term_D = 0 + 0 * I;
        double complex term_E = 0 + 0 * I;
        double complex term_R = 0 + 0 * I;
        double complex term_Q = 0 + 0 * I;
        double complex term_s = 0 + 0 * I;
        double complex term_W = 0 + 0 * I;
        double complex term_A = 0 + 0 * I;
        double complex term_B = 0 + 0 * I;

        term_A = w_nv - freq_i - g_nv * I;
        term_E = - w_vl - g_vl * I;

        for(p = -comb_num; p < comb_num + 1; p += 1)
            term_D = omega_M1 + p*del_freq - m.gamma * I;
            for(q = -comb_num; q < comb_num + 1; q += 1)
                term_B = freq_i - w_mv - omega_M2 -q*del_freq + (m.gamma + g_mv) * I;
                for (r = -comb_num; r < comb_num + 1; r += 1)
                    term_Q = freq_i - omega_M2 - omega_M3 - (q+r)*del_freq - 2.*m.gamma*I;
                    term_R = w_mv - omega_M2 - q * del_freq - (m.gamma + g_mv) * I;
                    term_s = freq_i - omega_M1 - omega_M2 - omega_M3 - (p+q+r)*del_freq;
                    term_W = - term_s / (pow(term_s, 2) + 9.*pow(m.gamma, 2));

                    result += ((term_W + (conj(term_Q) - term_R)/((conj(term_D) - term_Q)*(conj(term_D) - term_R)))/
                        ((conj(term_D) - conj(term_Q))*(conj(term_Q) - conj(term_E))));

        out_r[h_i] = creal(result);
        out_i[h_i] = cimag(result);
    }

    return 0;
}