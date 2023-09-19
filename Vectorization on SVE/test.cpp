#include <stdio.h>
#include <limits.h>
#include <arm_sve.h>
#include <iostream>
#include <stdio.h>
using namespace std;
#include <stdio.h>
#include <arm_sve.h>

#define SIZE 20


inline int recconvmul(const double* a, const double* b, double* dst) {
    
    svbool_t pg = svwhilelt_b64(0, SIZE);
    svfloat64_t va0;
    svfloat64_t vb07  = svld1_f64 (pg, &b[0]);          //serve caricarlo la prima volta 
    svfloat64_t va0b = svdup_f64(0);
    svfloat64_t accr = svdup_f64(0);

    for (int i = 0 ; i < SIZE ; i++){
        va0  = svdup_f64(a[i]);
        if ( i != 0 ){
            vb07  = svinsr_n_f64(vb07, 0); 
        }
        va0b  = svmul_f64_m (pg, va0,  vb07);
        accr = svadd_f64_m(pg, accr, va0b);

        printf("-----iterazione %d\n", i);
        for (int i = 0; i < SIZE; i++) {
            printf("%lf ", va0[i]);
        }
        printf("\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%lf ", vb07[i]);
        }
        printf("\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%lf ", va0b[i]);
        }
        printf("\n");
    }


    svst1_f64(pg, &dst[0], accr);

    return 0;
}
inline svfloat64_t convsingola(svfloat64_t va, svfloat64_t vb, svfloat64_t dst) {
    
    svbool_t pg = svwhilelt_b64(0, SIZE);
    svfloat64_t va0,va0b;
    for (int i = 0 ; i < SIZE ; i++){
        va0  = svdup_f64(va[i]);
        if ( i != 0 ){
            vb  = svinsr_n_f64(vb, 0); 
        }
        va0b  = svmul_f64_m (pg, va0,  vb);
        
        dst = svadd_f64_m(pg, dst, va0b);

    }
    return dst;
}


inline svfloat64_t convinversa(svfloat64_t va, svfloat64_t vb, svfloat64_t dst) {
    
    svbool_t pg = svwhilelt_b64(0, SIZE);
    svfloat64_t va0,va0b;
    svfloat64_t vb00 = svdup_f64(0);

    for (int i = 0 ; i < SIZE ; i++){
        va0  = svdup_f64(va[i]);
        if ( i != 0 ){
            vb00  = svinsr_n_f64(vb00, vb[i]); 
        }
        va0b  = svmul_f64_m (pg, va0,  vb00);
        dst = svadd_f64_m(pg, dst, va0b);
    }
    return dst;
}


inline int convmul(const double* a, const double* b, double* dst) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();
    int32_t N  = (int32_t)SIZE;

    svfloat64_t va0, vb07, vb07app, vb07r, accr_par;
    svfloat64_t va0b = svdup_f64(0);
    svfloat64_t accr = svdup_f64(0);

    for (int32_t i = 0; i < N; i += VL) {           // per ogni colonna 
        std::cout<< "ciclo "<<i<<std::endl;
        
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }

        // convoluzioni signole 
        int count_r;
        for (int32_t j = 0; j <= i; j += VL) { 
            count_r = i;
            va0b = svld1_f64(pg, &a[j]);   
            vb07 = svld1_f64(pg, &b[i-j]);   
            accr_par = convsingola(va0b, vb07, accr_par);
            
            svadd_f64_m(pg, accr, accr_par);
            std::cout<< "singola"<<std::endl;
            printf("R:\n");
            for (int i = 0; i < SIZE; i++) {
                printf("%f_", accr[i]);
            }
        printf("\n"); 
            if(count_r != 0){
                vb07r = svld1_f64(pg, &b[j]);  
                accr_par = convinversa(va0b,vb07r, accr_par);
                svadd_f64_m(pg, accr, accr_par);
                std::cout<< "inversa"<<std::endl;
                count_r -= VL;
            }
        }

        svst1_f64(pg, &dst[i], accr);

        // Aggiorna per la prossima iterazione
        pg = svwhilelt_b64(i + VL, N);
    }
    return 0;                                               // tutti elementi a 0    
}


 
inline int convmulF(const double* a, const double* b, double* dst) {
    svbool_t pg = svptrue_b64();
    svbool_t pgapp = svptrue_b64();
    int32_t VL = (int32_t)svcntd();
    int32_t N  = (int32_t)SIZE;

    svfloat64_t va0, vb07, vb07app;
    svfloat64_t va0b = svdup_f64(0);
    svfloat64_t accr = svdup_f64(0);
     

    int index_p;
    int VL_fix = (int32_t)svcntd();
    

    for (int32_t i = 0; i < N; i += VL) {           
        index_p= i ;
    
        
        // Lunghezza del vettore per questa iterazione
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }

        vb07 = svld1_f64(pg, &b[i]);          
        accr = svdup_f64(0);
        
        for (int j = 0 ; j < SIZE ; j++){

            va0  = svdup_f64(a[j]);
            va0b = svmul_f64_m (pg, va0,  vb07);
            accr = svadd_f64_m(pg, accr, va0b);
            
            if( j %VL_fix == 0){                                    // caricamento registro appoggio
                if( j == i ) {                                      // convoluzione nella diagonale con 0
                    vb07app = svdup_f64(0);                 
                }
                else if( j < i) {                                   // convoluzione con vettore precedente
                    index_p -= VL_fix;                              // ogni volta devo caricaricare il vettore precedente
                    vb07app = svld1_f64(pgapp, &b[index_p]);     
                }
            }
            
            vb07  = svinsr_n_f64(vb07, vb07app[VL_fix-(j%VL_fix)-1]);     // prendo i valori partendo dal fondo  
        }
        svst1_f64(pg, &dst[i], accr);

        // Aggiorna per la prossima iterazione
        pg = svwhilelt_b64(i + VL, N);
    }
    return 0;                                                 
}   






inline int convmulSS(const double* a, const double* b, double* dst) {
    // Imposta il predicato iniziale per coprire tutti gli elementi del vettore
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();
    int32_t N  = (int32_t)SIZE;
    //int32_t cont;
    svfloat64_t va;
    svfloat64_t vaapp;
    svfloat64_t vb; 
    svfloat64_t vab;
    svfloat64_t accr;
    svfloat64_t parz;
    
    for (int32_t i = 0; i < N; i ++) {
        
        //reinizializzo le condizioni di caricamento
        VL = (int32_t)svcntd();
        pg = svptrue_b64();
        
        if(i % VL == 0)
            vaapp = svld1_f64(pg, &a[i]);
            
        va = svdup_f64(vaapp[i % VL]);
        //cont = 0;
        for (int32_t j = 0; j < N; j+=VL) {
            if (j + VL > N) {
                VL = N - j; // Aggiusta la lunghezza se supera la fine del vettore
                pg = svwhilelt_b64(0, VL); // Aggiorna il predicato per coprire solo gli elementi validi
            }

            // carico vb
            if (j == 0) {
                vb = svld1_f64(pg, &b[j]);
                for(int32_t k = i - j; k > 0; k --)
                    vb = svinsr_n_f64(vb, 0);
            }
            else {
                vb = svld1_f64(pg, &b[j - i + 1]);
                for(int32_t k = i - j; k > 0; k --)
                    vb = svinsr_n_f64(vb, 0);
            }
            
            printf("A:\n");
            for (int l = 0; l < VL; l++)
            {
                printf("%f_", va[l]);
            }
            printf("\n"); 

            printf("B:\n");
            for (int l = 0; l < VL; l++)
            {
                printf("%f_", vb[l]);
            }
            printf("\n"); 
           
            // carico va
            vab = svmul_f64_m(pg, va, vb);
            
            // se non prima iterazione aggiorno somme parziali prendendo le precedenti
            if (i != 0)
                parz = svld1_f64(pg, &dst[j]);
            else 
                parz = svdup_f64(0);
            accr = svadd_f64_m(pg, parz, vab); 
            svst1_f64(pg, &dst[j], accr);

            // Aggiorna il predicato per la prossima iterazione
            pg = svwhilelt_b64(j + VL, N);
        }
    }

    
    return 0;
}
int main() {
    
    double a[SIZE];
    double b[SIZE];     
    double ris[SIZE];  
    int res;  
    for (int i = 0; i < 1; i++) {
        // Populate a and b with random doubles
        
        for (int j = 0; j < SIZE; j++) {
            // Random double between -0.5 and 0.5
            a[j] = j+1;
            b[j] = SIZE-j;        
        }
        
        res = convmulF(a,b,ris);
        printf("A:\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%f ", a[i]);
        }
        printf("\n"); 

        printf("B:\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%f ", b[i]);
        }
        printf("\n"); 

        printf("R:\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%f_", ris[i]);
        }
        printf("\n"); 

        printf("\n"); printf("\n"); 
    }
    
    return 0;
}




    

