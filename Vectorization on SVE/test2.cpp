#include <stdio.h>
#include <stdio.h>
#include <limits.h>
#include <arm_sve.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace std;



// ###### ALLEQUAL ###### // 
inline void function(const double* a, const double* b) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                          // # double nel registro                      
    int32_t N  = (int32_t)SIZE;                              // # double nel vettore

    for (int32_t i = 0; i < N; i += VL) {        
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }

        /* corpo della funzione */

        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return true; 
}




inline int cmpall(const double* a, const double* b) {
    svbool_t pg = svwhilelt_b64(0, 8);
    svfloat64_t va = svld1(pg, a);  
    svfloat64_t vb = svld1(pg, b); 

    svbool_t mask_gt = svcmpgt(pg, va, vb);  // gt[i]=true se a>b
    svbool_t mask_lt = svcmplt(pg, va, vb);  // gt[i]=true se a<b

    svint64_t indices = svindex_s64(0, 1);
    int64_t gt_index = svmaxv_s64(pg, svsel_s64(mask_gt, indices, svdup_n_s64(0)));
    int64_t lt_index = svmaxv_s64(pg, svsel_s64(mask_lt, indices, svdup_n_s64(0)));

    if (gt_index > lt_index) return 1;          // a > b
    if (lt_index > gt_index) return -1;         // a < b
    return 0;                                   // a = b
}
int main() {
    // Seed the random number generator with the current time
    srand(time(NULL));

    double a[8];
    double b[8];        
    for (int i = 0; i < 10; i++) {
        // Populate a and b with random doubles
        for (int j = 0; j < 8; j++) {
            // Random double between -0.5 and 0.5
            a[j] = (double)rand() / RAND_MAX - 0.5;
            b[j] = (double)rand() / RAND_MAX - 0.5;

            // Occasionally set some numbers to zero
            if (rand() < RAND_MAX / 10) {
                a[j] = 0.0;
            }
            if (rand() < RAND_MAX / 10) {
                b[j] = 0.0;
            }
        }
        printf("<" );
        for (int j = 0; j < 8; j++) {
            printf("%f ", a[j]);
        }
        printf(">\n" );
        printf("<" );
        for (int j = 0; j < 8; j++) {
            printf("%f ", b[j]);
        }
        printf(">\n" );
        int result = cmpall(a, b);
        printf("Test %d: %d\n", i+1, result);
    }
}