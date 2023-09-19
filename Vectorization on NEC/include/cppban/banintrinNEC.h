#include <velintrin.h>

// ###### VECTORIAL SUM ###### //
inline void vecsum(const double* a, const double* b, double* dst) {
    
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vld_vssl(8, b, SIZE);

    __vr accr = _vel_vfaddd_vvvl(va, vb, SIZE);

    _vel_vst_vssl(accr, 8, dst, SIZE);
}

inline void vecsum(const float* a, const float* b, float* dst) {
    
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    __vr vb = _vel_vldu_vssl(4, b, SIZE);
    
    __vr accr = _vel_vfadds_vvvl(va, vb, SIZE);
    
    _vel_vstu_vssl(accr, 4, dst, SIZE);
    
}

// ###### MUL ###### //
inline void mul(const double* a, double n, double* dst) {
    
    __vr va = _vel_vld_vssl(8, a, SIZE);
    
    __vr accr = _vel_vfmuld_vsvl(n, va, SIZE);

    _vel_vst_vssl(accr, 8, dst, SIZE);
}

inline void mul(const float* a, float n, float* dst) {
    
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    
    __vr accr = _vel_vfmuls_vsvl(n, va, SIZE);

    _vel_vstu_vssl(accr, 4, dst, SIZE);
}

// ###### DIV ###### //
inline void div(const double* a, double n, double* dst) {
    
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vbrdd_vsl(n, SIZE);

    __vr accr = _vel_vfdivd_vvvl(va, vb, SIZE);

    _vel_vst_vssl(accr, 8, dst, SIZE);
}

inline void div(const float* a, float n, float* dst) {
    
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    __vr vb = _vel_vbrds_vsl(n, SIZE);
    
    __vr accr = _vel_vfdivs_vvvl(va, vb, SIZE);

    _vel_vstu_vssl(accr, 4, dst, SIZE);
}

// ###### SUM ###### //
inline void sum(const double* a, const double* b, double* dst, int diff) {
    
    __vr va = _vel_vld_vssl(8, a, SIZE-diff);
    __vr vb = _vel_vld_vssl(8, b, SIZE-diff);

    __vr accr = _vel_vfaddd_vvvl(va, vb, SIZE-diff);

    _vel_vst_vssl(accr, 8, dst, SIZE-diff);
}

inline void sum(const float* a, const float* b, float* dst, int diff) {
    
    __vr va = _vel_vldu_vssl(4, a, SIZE-diff);
    __vr vb = _vel_vldu_vssl(4, b, SIZE-diff);

    __vr accr = _vel_vfadds_vvvl(va, vb, SIZE-diff);

    _vel_vstu_vssl(accr, 4, dst, SIZE-diff);
}

// ###### DENOISE ###### // 

inline void control(const double* a, double tol, double* dst) {
    
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vbrdd_vsl(tol, SIZE);
    __vr vc = _vel_vbrdd_vsl(-tol, SIZE);
    __vr vzero = _vel_vbrdd_vsl(0, SIZE);
    __vr vone = _vel_vbrdd_vsl(1, SIZE);

    __vr vgt = _vel_vfcmpd_vvvl(va, vc, SIZE);                      // vgt[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vr vlt = _vel_vfcmpd_vvvl(va, vb, SIZE);                      // vlt[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vr vmul = _vel_vfmuld_vvvl(vgt, vlt, SIZE);                   // vmul[i] < 0 se a[i] < tol && a[i] > -tol
    
    __vm256 mask = _vel_vfmkdge_mvl(vmul, SIZE);                    // 1 se vsum[i] >= 0

    __vr accr = _vel_vfmuld_vvvmvl(vone, va, mask, vzero, SIZE);    //accr[i] = mask[i] ? vone[i] * va[i] : vzero[i]
    
    _vel_vst_vssl(accr, 8, dst, SIZE);
}

inline void control(const float* a, float tol, float* dst) {
    
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    __vr vb = _vel_vbrds_vsl(tol, SIZE);
    __vr vc = _vel_vbrds_vsl(-tol, SIZE);
    __vr vzero = _vel_vbrds_vsl(0, SIZE);
    __vr vone = _vel_vbrds_vsl(1, SIZE);

    __vr vgt = _vel_vfcmps_vvvl(va, vc, SIZE);                      // vgt[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vr vlt = _vel_vfcmps_vvvl(va, vb, SIZE);                      // vlt[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vr vmul = _vel_vfmuls_vvvl(vgt, vlt, SIZE);                   // vmul[i] < 0 se a[i] < tol && a[i] > -tol

    __vm256 mask = _vel_vfmksge_mvl(vmul, SIZE);                    // 1 se vsum[i] >= 0

    __vr accr = _vel_vfmuls_vvvmvl(vone, va, mask, vzero, SIZE);    //accr[i] = mask[i] ? vone[i] * va[i] : vzero[i]

    _vel_vstu_vssl(accr, 4, dst, SIZE);
}


// ###### ALLZEROS ###### // 
inline bool allzeros( const double* a) {
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vm256 mask = _vel_vfmkdeq_mvl(va, SIZE);                // 1 se va[i]= 0
    unsigned long int ris = _vel_pcvm_sml(mask, SIZE);        // count degli 1
    return ris == SIZE;
}

inline bool allzeros( const float* a) {
    __vr va  = _vel_vldu_vssl(4, a, SIZE);      
    __vm256 mask = _vel_vfmkseq_mvl(va, SIZE);                // 1 se va[i]= 0
    unsigned long int ris = _vel_pcvm_sml(mask, SIZE);        // count degli 1
    return ris == SIZE;
}


// ###### ALLEQUAL ###### //
inline bool allequal(const double* a,const double* b) {
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vld_vssl(8, b, SIZE);
    __vr cmp = _vel_vcmpsl_vvvl(va, vb, SIZE);                // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b 
    __vm256 mask = _vel_vfmkdeq_mvl(cmp, SIZE);               // 1 se va[i]= 0
    unsigned long int ris = _vel_pcvm_sml(mask, SIZE);        // count 1
    return ris == SIZE;
}

inline bool allequal(const float* a,const float* b) {
    __vr va  = _vel_vldu_vssl(4, a, SIZE);  
    __vr vb  = _vel_vldu_vssl(4, b, SIZE); 
    __vr cmp = _vel_vfcmps_vvvl(va, vb, SIZE);                // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vm256 mask = _vel_vfmkseq_mvl(cmp, SIZE);               // 1 se va[i]= 0
    unsigned long int ris = _vel_pcvm_sml(mask, SIZE);        // count 1
    return ris == SIZE;
}


// ###### CMPALL ###### // 
inline int cmpall(const double  * a, const double  * b){
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vld_vssl(8, b, SIZE);

    __vr cmp = _vel_vcmpsl_vvvl(va, vb, SIZE);                   // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b
    __vm256 mask_g = _vel_vfmkdgt_mvl(cmp, SIZE);                // 1 se va[i] > vb[i]
    __vm256 mask_l = _vel_vfmkdlt_mvl(cmp, SIZE);                // 1 se va[i] < vb[i]
    
    unsigned long int count_g = _vel_lzvm_sml(mask_g,SIZE);      // conta gli 0 da sinistra
    unsigned long int count_l = _vel_lzvm_sml(mask_l,SIZE);      // conta gli 0 da sinistra
    
    if(count_g < count_l) return 1;                            // a>b
    if(count_g > count_l) return -1;                           // a<b
    return 0;                                                  // a=b
}

inline int cmpall( const float * a,  const float * b) {
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    __vr vb = _vel_vldu_vssl(4, b, SIZE);
    
    __vr cmp = _vel_vfcmps_vvvl(va, vb, SIZE);                   // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b

    __vm256 mask_g = _vel_vfmksgt_mvl(cmp, SIZE);                // 1 se va[i] > vb[i]
    __vm256 mask_l = _vel_vfmkslt_mvl(cmp, SIZE);                // 1 se va[i] < vb[i]
    
    unsigned long int count_g = _vel_lzvm_sml(mask_g,SIZE);      // conta gli 0 da sinistra
    unsigned long int count_l = _vel_lzvm_sml(mask_l,SIZE);      // conta gli 0 da sinistra
    
    if(count_g < count_l) return 1;                             
    if(count_g > count_l) return -1;
    return 0;
}

// ###### CMP0 ###### // 
inline int cmp0(const double* a) {
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vb = _vel_vbrdd_vsl(0, SIZE);

    __vr cmp = _vel_vcmpsl_vvvl(va, vb, SIZE);                   // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b 

    __vm256 mask_g = _vel_vfmkdgt_mvl(cmp, SIZE);                // 0 se va[i] > 0
    __vm256 mask_l = _vel_vfmkdlt_mvl(cmp, SIZE);                // 0 se va[i] < 0
    
    unsigned long int count_g = _vel_lzvm_sml(mask_g,SIZE);      // conta gli 0 da sinistra
    unsigned long int count_l = _vel_lzvm_sml(mask_l,SIZE);      // conta gli 0 da sinistra

    if(count_g < count_l) return 1;                            // a>0
    if(count_g > count_l) return -1;                           // a<0
    return 0;                                                  // a=0
}

inline int cmp0(const float* a) {    
    
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    __vr vb = _vel_vbrds_vsl(0, SIZE);
    
    __vr cmp = _vel_vfcmps_vvvl(va, vb, SIZE);                  // cmp[i] = p se a>b ; 0 se a=b ; neg se a<b

    __vm256 mask_g = _vel_vfmksgt_mvl(cmp, SIZE);                // 0 se va[i] > 0
    __vm256 mask_l = _vel_vfmkslt_mvl(cmp, SIZE);                // 0 se va[i] < 0
    
    unsigned long int count_g = _vel_lzvm_sml(mask_g,SIZE);      // conta gli 0 da sinistra
    unsigned long int count_l = _vel_lzvm_sml(mask_l,SIZE);      // conta gli 0 da sinistra

    if(count_g < count_l) return 1;                            // a>0
    if(count_g > count_l) return -1;                           // a<0 
    return 0;                                                  // a=0
}


// ###### FIND1°NONZERO ###### // 
inline int findFirstNonZero(const double* a) {
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vm256 mask = _vel_vfmkdne_mvl(va, SIZE);                    // 0 se va[i] = 0
    return _vel_lzvm_sml(mask,SIZE);                              // conta gli 0 da sinistra
}

inline int findFirstNonZero(const float* a) {
    __vr va = _vel_vldu_vssl(4, a, SIZE);
    __vm256 mask = _vel_vfmksne_mvl(va, SIZE);                    // 0 se va[i] = 0
    return _vel_lzvm_sml(mask,SIZE);                              // conta gli 0 da sinistra
}


// ###### CONVMUL ###### //  
inline int convmul(const double* a,const double* b, double* dst) {
    __vr vb = _vel_vld_vssl(8, b, SIZE);
    __vr va = _vel_vld_vssl(8, a, SIZE);
    __vr vab; 
    __vr accr = _vel_vbrdd_vsl(0, SIZE);

    for(int i = 0; i < SIZE; i++){
        vab = _vel_vfmuld_vsvl(va[i], vb, SIZE );
        accr = _vel_vfaddd_vvvl(accr, vab, SIZE);
        vb = _vel_vmv_vsvl(-1, vb, SIZE);
    }
    _vel_vst_vssl(accr, 8, dst, SIZE);

    return 0;

   /*
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

    __vr vb77 = _vel_vmv_vsvl(-7, _vel_vld_vssl(8, b, SIZE-7), SIZE);  // _vel_vmv_vsvl(-7) = shift dx di 7 ; _vel_vldu_vssl(SIZE-7) = maschera i primi SIZE-7 elementi
    __vr vb67 = _vel_vmv_vsvl(-6, _vel_vld_vssl(8, b, SIZE-6), SIZE);
    __vr vb57 = _vel_vmv_vsvl(-5, _vel_vld_vssl(8, b, SIZE-5), SIZE);
    __vr vb47 = _vel_vmv_vsvl(-4, _vel_vld_vssl(8, b, SIZE-4), SIZE);
    __vr vb37 = _vel_vmv_vsvl(-3, _vel_vld_vssl(8, b, SIZE-3), SIZE);
    __vr vb27 = _vel_vmv_vsvl(-2, _vel_vld_vssl(8, b, SIZE-2), SIZE);
    __vr vb17 = _vel_vmv_vsvl(-1, _vel_vld_vssl(8, b, SIZE-1), SIZE);
    __vr vb07 = _vel_vld_vssl(8, b, SIZE);
   
    __vr va0b = _vel_vfmuld_vsvl(a[0], vb07, SIZE );
    __vr va1b = _vel_vfmuld_vsvl(a[1], vb17, SIZE );
    __vr va2b = _vel_vfmuld_vsvl(a[2], vb27, SIZE );
    __vr va3b = _vel_vfmuld_vsvl(a[3], vb37, SIZE );
    __vr va4b = _vel_vfmuld_vsvl(a[4], vb47, SIZE );
    __vr va5b = _vel_vfmuld_vsvl(a[5], vb57, SIZE );
    __vr va6b = _vel_vfmuld_vsvl(a[6], vb67, SIZE );
    __vr va7b = _vel_vfmuld_vsvl(a[7], vb77, SIZE );

    
    __vr accr = _vel_vfaddd_vvvl(va0b, va1b, SIZE);
    accr = _vel_vfaddd_vvvl(accr, va2b, SIZE);
    accr = _vel_vfaddd_vvvl(accr, va3b, SIZE);
    accr = _vel_vfaddd_vvvl(accr, va4b, SIZE);
    accr = _vel_vfaddd_vvvl(accr, va5b, SIZE);
    accr = _vel_vfaddd_vvvl(accr, va6b, SIZE);
    accr = _vel_vfaddd_vvvl(accr, va7b, SIZE);

    _vel_vst_vssl(accr, 8, dst, SIZE);

    return 0;*/
}

inline int convmul(const float* a,const float* b, float* dst) {

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
    /*__vr vb, vab;
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

    __vr vb77 = _vel_vmv_vsvl(-7, _vel_vldu_vssl(4, b, SIZE-7), SIZE);  // _vel_vmv_vsvl(-7) = shift dx di 7 ; _vel_vldu_vssl(SIZE-7) = maschera i primi SIZE-7 elementi
    __vr vb67 = _vel_vmv_vsvl(-6, _vel_vldu_vssl(4, b, SIZE-6), SIZE);
    __vr vb57 = _vel_vmv_vsvl(-5, _vel_vldu_vssl(4, b, SIZE-5), SIZE);
    __vr vb47 = _vel_vmv_vsvl(-4, _vel_vldu_vssl(4, b, SIZE-4), SIZE);
    __vr vb37 = _vel_vmv_vsvl(-3, _vel_vldu_vssl(4, b, SIZE-3), SIZE);
    __vr vb27 = _vel_vmv_vsvl(-2, _vel_vldu_vssl(4, b, SIZE-2), SIZE);
    __vr vb17 = _vel_vmv_vsvl(-1, _vel_vldu_vssl(4, b, SIZE-1), SIZE);
    __vr vb07 = _vel_vldu_vssl(4, b, SIZE);
   
    __vr va0b = _vel_vfmuls_vsvl(a[0], vb07, SIZE );
    __vr va1b = _vel_vfmuls_vsvl(a[1], vb17, SIZE );
    __vr va2b = _vel_vfmuls_vsvl(a[2], vb27, SIZE );
    __vr va3b = _vel_vfmuls_vsvl(a[3], vb37, SIZE );
    __vr va4b = _vel_vfmuls_vsvl(a[4], vb47, SIZE );
    __vr va5b = _vel_vfmuls_vsvl(a[5], vb57, SIZE );
    __vr va6b = _vel_vfmuls_vsvl(a[6], vb67, SIZE );
    __vr va7b = _vel_vfmuls_vsvl(a[7], vb77, SIZE );

    
    __vr accr = _vel_vfadds_vvvl(va0b, va1b, SIZE);
    accr = _vel_vfadds_vvvl(accr, va2b, SIZE);
    accr = _vel_vfadds_vvvl(accr, va3b, SIZE);
    accr = _vel_vfadds_vvvl(accr, va4b, SIZE);
    accr = _vel_vfadds_vvvl(accr, va5b, SIZE);
    accr = _vel_vfadds_vvvl(accr, va6b, SIZE);
    accr = _vel_vfadds_vvvl(accr, va7b, SIZE);

    _vel_vst_vssl(accr, 4, dst, SIZE);

    return 0;
    */
}


// ###### REVERSEORDER ###### //
/*inline __vr reverseOrderF(__vr va){
    va = _vel_vbrds_vsl (va[0], SIZE);
    va = _vel_vbrds_vsvl(va[1], va, SIZE-1);
    va = _vel_vbrds_vsvl(va[2], va, SIZE-2);
    va = _vel_vbrds_vsvl(va[3], va, SIZE-3);
    va = _vel_vbrds_vsvl(va[4], va, SIZE-4);
    va = _vel_vbrds_vsvl(va[5], va, SIZE-5);
    va = _vel_vbrds_vsvl(va[6], va, SIZE-6);
    va = _vel_vbrds_vsvl(va[7], va, SIZE-7);
    return va;
}*/

/*inline __vr reverseOrderD(__vr va){
    va = _vel_vbrdd_vsl (va[0], SIZE);
    va = _vel_vbrdd_vsvl(va[1], va, SIZE-1);
    va = _vel_vbrdd_vsvl(va[2], va, SIZE-2);
    va = _vel_vbrdd_vsvl(va[3], va, SIZE-3);
    va = _vel_vbrdd_vsvl(va[4], va, SIZE-4);
    va = _vel_vbrdd_vsvl(va[5], va, SIZE-5);
    va = _vel_vbrdd_vsvl(va[6], va, SIZE-6);
    va = _vel_vbrdd_vsvl(va[7], va, SIZE-7);
    return va;
}*/