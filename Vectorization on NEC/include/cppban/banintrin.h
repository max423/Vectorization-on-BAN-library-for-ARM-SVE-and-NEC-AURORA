#include <immintrin.h>

inline __m256 reverseOrder(__m256 v) {
    const __m256i idx = _mm256_set_epi32(0,1,2,3,4,5,6,7);
    __m256 reversed = _mm256_permutevar8x32_ps(v,idx);
    return reversed;
}

inline __m512d reverseOrder(__m512d v) {
    const __m512i idx = _mm512_set_epi64(0,1,2,3,4,5,6,7);
    __m512d reversed = _mm512_permutexvar_pd(idx,v);
    return reversed;
}

inline bool allequal(const float* a,const float* b) {
    __m256 va =  _mm256_loadu_ps(&a[0]);
    __m256 vb = _mm256_loadu_ps(&b[0]);
    __m256 mask = _mm256_cmp_ps(va,vb,4);
    int vr = _mm256_testz_ps(mask,mask);
    return vr == 1; 
}

inline bool allequal(const double* a, const double* b) {
    __m512d va = _mm512_loadu_pd(&a[0]);
    __m512d vb = _mm512_loadu_pd(&b[0]);
    __mmask8 mask = _mm512_cmp_pd_mask(va,vb,0);
    return mask == 0xff;
}

inline bool allzeros(const float* a) {
    __m256 va =  _mm256_loadu_ps(&a[0]);
    __m256 vb = _mm256_set1_ps(0.f);
    __m256 mask = _mm256_cmp_ps(va,vb,4);
    int vr = _mm256_testz_ps(mask,mask);
    return vr == 1;
}

inline bool allzeros(const double* a) {
    __m512d va = _mm512_loadu_pd(&a[0]);
    __m512d vb = _mm512_set1_pd(0);
    __mmask8 mask = _mm512_cmp_pd_mask(va,vb,0);
    return mask == 0xff;
}


/*
    Return 1 if a > b
    Return -1 if a < b
    Return 0 if a == b
*/
inline int cmpall(const float* a,const float* b) {
    __m256 va =  _mm256_loadu_ps(&a[0]);
    __m256 vb = _mm256_loadu_ps(&b[0]);
    __m256 maskgt = reverseOrder(_mm256_cmp_ps(va,vb,14));  // -> >  = 1
    __m256 masklt = reverseOrder(_mm256_cmp_ps(va,vb,1));   // -> <  = 1

    int isgt = _mm256_movemask_ps(maskgt);
    int islt = _mm256_movemask_ps(masklt);

    if(isgt > islt) return 1;
    if(islt > isgt) return -1;
    return 0;
}

inline int cmpall(const double* a, const double* b) {
    __m512d va = reverseOrder(_mm512_loadu_pd(&a[0]));
    __m512d vb = reverseOrder(_mm512_loadu_pd(&b[0]));
    __mmask8 isgt = _mm512_cmp_pd_mask(va,vb,14);
    __mmask8 islt = _mm512_cmp_pd_mask(va,vb,1);
    
    if(isgt > islt) return 1;
    if(islt > isgt) return -1;
    return 0;
}

inline int cmp0(const float* a) {
    __m256 va =  _mm256_loadu_ps(&a[0]);
    __m256 vb = _mm256_set1_ps(0);
    __m256 maskgt = reverseOrder(_mm256_cmp_ps(va,vb,14));
    __m256 masklt = reverseOrder(_mm256_cmp_ps(va,vb,1));

    int isgt = _mm256_movemask_ps(maskgt);
    int islt = _mm256_movemask_ps(masklt);

    if(isgt > islt) return 1;
    if(islt > isgt) return -1;
    return 0;
}

inline int cmp0(const double* a) {
    __m512d va = reverseOrder(_mm512_loadu_pd(&a[0]));
    __m512d vb = _mm512_set1_pd(0);

    __mmask8 isgt = _mm512_cmp_pd_mask(va,vb,14);
    __mmask8 islt = _mm512_cmp_pd_mask(va,vb,1);

    if(isgt > islt) return 1;
    if(islt > isgt) return -1;
    return 0;
}


inline int findFirstNonZero(float* a) {
    __m256 va =  _mm256_loadu_ps(&a[0]);
    __m256 vb = _mm256_set_ps(0,0,0,0,0,0,0,0);
    __m256 mask = reverseOrder(_mm256_cmp_ps(va,vb,4));
    int maskm = _mm256_movemask_ps(mask);
    int leading = __builtin_clz(maskm) - 24;
    return leading;
}

inline int findFirstNonZero(double* a) {
    __m512d va = reverseOrder(_mm512_loadu_pd(&a[0]));
    const __m512d vb = _mm512_set_pd(0,0,0,0,0,0,0,0);
    __mmask8 maskm = _mm512_cmp_pd_mask(va,vb,4);
    int leading = __builtin_clz(maskm) - 24;
    return leading;
}

#define masked0 _mm256_set_ps(1,1,1,1,1,1,1,1)
#define masked1 _mm256_set_ps(0,1,1,1,1,1,1,1)
#define masked2 _mm256_set_ps(0,0,1,1,1,1,1,1)
#define masked3 _mm256_set_ps(0,0,0,1,1,1,1,1)
#define masked4 _mm256_set_ps(0,0,0,0,1,1,1,1)
#define masked5 _mm256_set_ps(0,0,0,0,0,1,1,1)
#define masked6 _mm256_set_ps(0,0,0,0,0,0,1,1)
#define masked7 _mm256_set_ps(0,0,0,0,0,0,0,1)

#define perm0 _mm256_set_epi32(7,6,5,4,3,2,1,0)
#define perm1 _mm256_set_epi32(6,5,4,3,2,1,0,7)
#define perm2 _mm256_set_epi32(5,4,3,2,1,0,7,6)
#define perm3 _mm256_set_epi32(4,3,2,1,0,7,6,5)
#define perm4 _mm256_set_epi32(3,2,1,0,7,6,5,4)
#define perm5 _mm256_set_epi32(2,1,0,7,6,5,4,3)
#define perm6 _mm256_set_epi32(1,0,7,6,5,4,3,2)
#define perm7 _mm256_set_epi32(0,7,6,5,4,3,2,1)

inline int convmul(const float* a,const float* b, float* dst) {
    // Fixed width 8 weights
    __m256 va0 = _mm256_set1_ps(a[0]);
    __m256 va1 = _mm256_set1_ps(a[1]);
    __m256 va2 = _mm256_set1_ps(a[2]);
    __m256 va3 = _mm256_set1_ps(a[3]);
    __m256 va4 = _mm256_set1_ps(a[4]);
    __m256 va5 = _mm256_set1_ps(a[5]);
    __m256 va6 = _mm256_set1_ps(a[6]);
    __m256 va7 = _mm256_set1_ps(a[7]);

    // Fixed width b 8 lanes
    __m256 vb07 = _mm256_loadu_ps(&b[0]);
    __m256 vb17 = _mm256_mul_ps(vb07,masked1);
    __m256 vb27 = _mm256_mul_ps(vb07,masked2);
    __m256 vb37 = _mm256_mul_ps(vb07,masked3);
    __m256 vb47 = _mm256_mul_ps(vb07,masked4);
    __m256 vb57 = _mm256_mul_ps(vb07,masked5);
    __m256 vb67 = _mm256_mul_ps(vb07,masked6);
    __m256 vb77 = _mm256_mul_ps(vb07,masked7);

    // {a0*[b0...b7]} {a1*{b1...b7}} ..... {a8*{b7}}
    __m256 va0b = _mm256_mul_ps(va0,vb07);
    __m256 va1b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va1,vb17),perm1);
    __m256 va2b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va2,vb27),perm2);
    __m256 va3b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va3,vb37),perm3);
    __m256 va4b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va4,vb47),perm4);
    __m256 va5b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va5,vb57),perm5);
    __m256 va6b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va6,vb67),perm6);
    __m256 va7b = _mm256_permutevar8x32_ps(_mm256_mul_ps(va7,vb77),perm7);


    __m256 accr = _mm256_add_ps(va0b,va1b);
    accr = _mm256_add_ps(accr,va2b);
    accr = _mm256_add_ps(accr,va3b);
    accr = _mm256_add_ps(accr,va4b);
    accr = _mm256_add_ps(accr,va5b);
    accr = _mm256_add_ps(accr,va6b);
    accr = _mm256_add_ps(accr,va7b);


    _mm256_storeu_ps(&dst[0],accr);

    return 0;
}



#define masked0d _mm512_set_pd(1,1,1,1,1,1,1,1)
#define masked1d _mm512_set_pd(0,1,1,1,1,1,1,1)
#define masked2d _mm512_set_pd(0,0,1,1,1,1,1,1)
#define masked3d _mm512_set_pd(0,0,0,1,1,1,1,1)
#define masked4d _mm512_set_pd(0,0,0,0,1,1,1,1)
#define masked5d _mm512_set_pd(0,0,0,0,0,1,1,1)
#define masked6d _mm512_set_pd(0,0,0,0,0,0,1,1)
#define masked7d _mm512_set_pd(0,0,0,0,0,0,0,1)

#define perm0d _mm512_set_epi64(7,6,5,4,3,2,1,0)
#define perm1d _mm512_set_epi64(6,5,4,3,2,1,0,7)
#define perm2d _mm512_set_epi64(5,4,3,2,1,0,7,6)
#define perm3d _mm512_set_epi64(4,3,2,1,0,7,6,5)
#define perm4d _mm512_set_epi64(3,2,1,0,7,6,5,4)
#define perm5d _mm512_set_epi64(2,1,0,7,6,5,4,3)
#define perm6d _mm512_set_epi64(1,0,7,6,5,4,3,2)
#define perm7d _mm512_set_epi64(0,7,6,5,4,3,2,1)

inline int convmul(const double* a,const double* b, double* dst) {
    // Fixed width 8 weights
    __m512d va0 = _mm512_set1_pd(a[0]);
    __m512d va1 = _mm512_set1_pd(a[1]);
    __m512d va2 = _mm512_set1_pd(a[2]);
    __m512d va3 = _mm512_set1_pd(a[3]);
    __m512d va4 = _mm512_set1_pd(a[4]);
    __m512d va5 = _mm512_set1_pd(a[5]);
    __m512d va6 = _mm512_set1_pd(a[6]);
    __m512d va7 = _mm512_set1_pd(a[7]);

    // Fixed width b 8 lanes
    __m512d vb07 = _mm512_loadu_pd(&b[0]);
    __m512d vb17 = _mm512_mul_pd(vb07,masked1d);
    __m512d vb27 = _mm512_mul_pd(vb07,masked2d);
    __m512d vb37 = _mm512_mul_pd(vb07,masked3d);
    __m512d vb47 = _mm512_mul_pd(vb07,masked4d);
    __m512d vb57 = _mm512_mul_pd(vb07,masked5d);
    __m512d vb67 = _mm512_mul_pd(vb07,masked6d);
    __m512d vb77 = _mm512_mul_pd(vb07,masked7d);

    // {a0*[b0...b7]} {a1*{b1...b7}} ..... {a8*{b7}}
    __m512d va0b = _mm512_mul_pd(va0,vb07);
    __m512d va1b = _mm512_permutexvar_pd(perm1d,_mm512_mul_pd(va1,vb17));
    __m512d va2b = _mm512_permutexvar_pd(perm2d,_mm512_mul_pd(va2,vb27));
    __m512d va3b = _mm512_permutexvar_pd(perm3d,_mm512_mul_pd(va3,vb37));
    __m512d va4b = _mm512_permutexvar_pd(perm4d,_mm512_mul_pd(va4,vb47));
    __m512d va5b = _mm512_permutexvar_pd(perm5d,_mm512_mul_pd(va5,vb57));
    __m512d va6b = _mm512_permutexvar_pd(perm6d,_mm512_mul_pd(va6,vb67));
    __m512d va7b = _mm512_permutexvar_pd(perm7d,_mm512_mul_pd(va7,vb77));


    __m512d accr = _mm512_add_pd(va0b,va1b);
    accr = _mm512_add_pd(accr,va2b);
    accr = _mm512_add_pd(accr,va3b);
    accr = _mm512_add_pd(accr,va4b);
    accr = _mm512_add_pd(accr,va5b);
    accr = _mm512_add_pd(accr,va6b);
    accr = _mm512_add_pd(accr,va7b);


    _mm512_storeu_pd(&dst[0],accr);

    return 0;
}

