#include <velintrin.h>
#include <iostream>
#include <stdio.h>
#define SIZE 8
#define SIZEL 256 // Dimensione del registro vettoriale, ad esempio 256
#define ARRAY_SIZE 8 // Dimensione dell'array che stai replicando
typedef float T;



/* inline bool allzeros(const float* a) {
    double ret[256];
    __vr va = _vel_vldu_vssl(4, &a[0], 256);
    __vr vb = _vel_vbrds_vsl(0.f, 256);
    // if (y = z) { x ← 00...0 }
    __vr mask = _vel_vcmpul_vvvl(va, vb, 256);
    _vel_vst_vssl(mask, 4, &ret[0], 256);
    return true;
} */

// TODO _vel_vcmpul_vvvl 

/*inline int cmpall(const double  * a, const double  * b){
    __vr va =_vel_vld_vssl(8, a, VL);
    __vr vb =_vel_vld_vssl(8, b, VL);
    
    __vr cmp = _vel_vcmpul_vvvl(va, vb, VL);                   // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vm256 mask_g = _vel_vfmklgt_mvl(cmp, VL);                // 0 se va[i]= 0
    __vm256 mask_l = _vel_vfmkllt_mvl(cmp, VL);                // 0 se va[i]= 0
                     
    unsigned long int count_g = _vel_lzvm_sml(mask_g,VL);      // conta gli 0 da sinistra
    unsigned long int count_l = _vel_lzvm_sml(mask_l,VL);      // conta gli 0 da sinistra
    
    if(count_g < count_l) return 1;                            // a>b
    if(count_g > count_l) return -1;                           // a<b
    return 0;                                                  // a=b
}*/

inline int convmul(const double* a,const double* b, double* dst) {

    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb, vab;
    __vr accr = _vel_vbrdd_vsl(0, SIZE);

    for(int i = -SIZE+1; i <= 0; i++){
        if (i == 0)
            vb = _vel_vld_vssl(8, b, SIZE+i);
        else 
            vb = _vel_vmv_vsvl(i, _vel_vld_vssl(8, b, SIZE+i), SIZE);   // _vel_vmv_vsvl(-7) = shift dx di 7 ; _vel_vldu_vssl(SIZE-7) = maschera i primi SIZE-7 elementi
        
        vab = _vel_vfmuld_vsvl(va[-i], vb, SIZE );
        
        accr = _vel_vfaddd_vvvl(accr, vab, SIZE);
         
    }
    
    _vel_vst_vssl(accr, 8, dst, SIZE);

    return 0;
}

inline void mul(const float* a, float n, float* dst) {
    
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    
    __vr accr = _vel_vfmuls_vsvl(n, va, SIZE);
    
    _vel_vstu_vssl(accr, 4, dst, SIZE);
    printf("dst:\n");
        for (int i = 0; i < SIZE; i++) {
            printf("%f_", dst[i]);
        }
            printf("\n");

}

inline int convmul(const float* a,const float* b, float* dst) {

    __vr vb, vab;
    __vr accr = _vel_vbrds_vsl(0, SIZE);

    for(int i = -SIZE+1; i <= 0; i++){
        if (i == 0)
            vb = _vel_vldu_vssl(4, b, SIZE+i);
        else{ 
            vb = _vel_vmv_vsvl(i, _vel_vldu_vssl(4, b, SIZE+i), SIZE);   // _vel_vmv_vsvl(-7) = shift dx di 7 ; _vel_vldu_vssl(SIZE-7) = maschera i primi SIZE-7 elementi
        }
        
        vab = _vel_vfmuls_vsvl(a[-i], vb, SIZE );
        
        accr = _vel_vfadds_vvvl(accr, vab, SIZE);
        
    }
    
    _vel_vstu_vssl(accr, 4, dst, SIZE);

    return 0;
}

inline int convmulMAX(const double* a,const double* b, double* dst) {
    std::cout<< sizeof(T);

    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vld_vssl(8, b, SIZE);
    __vr vab;
    __vr accr = _vel_vbrdd_vsl(0, SIZE);

    for(int i = 0; i < SIZE; i++){
        vab = _vel_vfmuld_vsvl(va[i], vb, SIZE );
        accr = _vel_vfaddd_vvvl(accr, vab, SIZE);
        vb = _vel_vmv_vsvl( -1  , vb , SIZE);
    }
    _vel_vst_vssl(accr, 8, dst, SIZE);
    return 0;
}

inline int convmulMAX(const float* a,const float* b, float* dst) {

    __vr vb = _vel_vldu_vssl(4, b, SIZE);
    __vr vab;
    __vr accr = _vel_vbrds_vsl(0, SIZE);

    for(int i = 0; i < SIZE; i++){
        vab = _vel_vfmuls_vsvl(a[i], vb, SIZE );
        accr = _vel_vfadds_vvvl(accr, vab, SIZE);
        vb = _vel_vmv_vsvl( -1  , vb , SIZE);
    }
    _vel_vstu_vssl(accr, 4, dst, SIZE);
    return 0;
}


inline int convmulF(const double* a, const double* b, double* dst) {
    
    __vr vb ,va ; 
    __vr vab, app;
    __vr accr = _vel_vbrdd_vsl(0, SIZE);
    
    double bufferA[256];
    double bufferB[256];
    double bufferM[256];

    vb = _vel_vld_vssl(8, b, SIZE);
     
    for (int i = 0; i < 64; i += 8) {
        //memcpy(&bufferA[i], a, 64);
    
        va = _vel_vbrdd_vsl(a[(i/8)],SIZE);
        _vel_vst_vssl(va, 8, &bufferA[i], 256);
        
        _vel_vst_vssl(vb, 8, &bufferB[i], SIZE);
        vb = _vel_vmv_vsvl(-1, vb , SIZE);             //(0,-1,-2,-3) quanti 0 in testa
    }

    // Carica il buffer nel registro vettoriale
    va = _vel_vld_vssl(8, bufferA, 256);
    vb = _vel_vld_vssl(8, bufferB, 256);

    vab = _vel_vfmuld_vvvl(va, vb, 64); 

    _vel_vst_vssl(vab, 8, &bufferM, 256);
    
    
    for (int i = 0; i < 256; i += 8) {
        app = _vel_vld_vssl(8, &bufferM[i], 8);
        accr = _vel_vfaddd_vvvl(accr, app, SIZE);
    }
    _vel_vst_vssl(accr, 8, dst, SIZE);
/* 
    printf("A:\n");
    for (int i = 0; i < 8; i++) {
        if ( i %8 ==0)
            printf("||");
        printf("%f_", accr[i]);
    }
    printf("\n");  */
   
   
    return 0;
}



int main() {
    
    double a[SIZE] = {1,2,3,4,5,6,7,8};
    double b[SIZE] = {8,7,6,5,4,3,2,1};
    double sol[SIZE];
    
    printf("A:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f_", a[i]);
    }
    printf("\n"); 

    printf("B:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f_", b[i]);
    }
    printf("\n"); 
    
    
    int ris = convmulF(a,b,sol);
    
    printf("RIS:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("%f_", sol[i]);
    }
    printf("\n"); 
    //8.000000_23.000000_44.000000_70.000000_100.000000_133.000000_168.000000_204.000000_
    return 0;
}