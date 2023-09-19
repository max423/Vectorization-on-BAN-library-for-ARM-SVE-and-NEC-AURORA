#include <arm_sve.h>
#include <limits.h>

/* 
uint64_t len = svcntw();
    std::cout << "dimenensione registri = " << len< <std::endl;
 */

// ###### DENOISE ###### // 
inline void control(const double*a,int tol, double* dst){
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                        // # double nel registro 
    int32_t N  = (int32_t)SIZE;                            // # double nel vettore

    for (int32_t i = 0; i < N; i +=VL) {                    
        if (i + VL > N) {   
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // Aggiorna il predicato per coprire solo gli elementi validi
        }
        svfloat64_t va = svld1_f64(pg, &a[i]);
        svfloat64_t vb = svdup_f64(tol);
        svfloat64_t vc = svdup_f64(-tol);
        
        svbool_t mask_gt = svcmpgt(pg, va, vc);             // gt[i]=true se a>b
        svbool_t mask_lt = svcmplt(pg, va, vb);             

        svfloat64_t v0 = svdup_f64(0);
        svfloat64_t vgt = svsel_f64(mask_gt, v0, va);       //0 se va[i] > -tol
        svfloat64_t vlt = svsel_f64(mask_lt, v0, va);       //0 se vai[i] < tol 
        
        svfloat64_t res = svadd_f64_m(pg, vgt, vlt);
        svst1_f64(pg, &dst[i], res);

        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna il predicato per la prossima iterazione
    }
}

inline void control(const float*a, int tol, float* dst){
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # float nel registro
    int32_t N  = (int32_t)SIZE;                             // # float nel vettore

    for (int32_t i = 0; i < N; i +=VL) {
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // Aggiorna il predicato per coprire solo gli elementi validi
        }
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svdup_f32(tol);
        svfloat32_t vc = svdup_f32(-tol);
        
        svbool_t mask_gt = svcmpgt(pg, va, vc);             // gt[i]=true se a>b
        svbool_t mask_lt = svcmplt(pg, va, vb);             

        svfloat32_t v0 = svdup_f32(0);
        svfloat32_t vgt = svsel_f32(mask_gt, v0, va);       //0 se va[i] > -tol
        svfloat32_t vlt = svsel_f32(mask_lt, v0, va);       //0 se vai[i] < tol 
        
        svfloat32_t res = svadd_f32_m(pg, vgt, vlt);
        svst1_f32(pg, &dst[i], res); 

        pg = svwhilelt_b32(i + VL, N);                      // Aggiorna il predicato per la prossima iterazione
    }
}


// ###### SUMVET ###### // 
inline void sumvet(const double* a, const double* b, int p, double* dst) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                          // # double nel registro

    for (int32_t i = 0; i < p; i +=VL) {
        if (i + VL > p) {                                   // Calcola la lunghezza del vettore per questa iterazione
            VL = p - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // Aggiorna il predicato per coprire solo gli elementi validi
        }
        svfloat64_t va = svld1_f64(pg, &a[i]);
        svfloat64_t vb = svld1_f64(pg, &b[i]);
        svfloat64_t res = svadd_f64_m(pg, va, vb);
        svst1_f64(pg, &dst[i], res); 

        pg = svwhilelt_b64(i + VL, p);
    }
}

inline void sumvet(const float* a, const float* b, int p, float* dst) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # float nel registro

    for (int32_t i = 0; i < p; i +=VL) {
        if (i + VL > p) {                                   // Calcola la lunghezza del vettore per questa iterazione
            VL = p - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // Aggiorna il predicato per coprire solo gli elementi validi
        }
        svfloat32_t va = svld1_f32(pg, &a[i]);
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svfloat32_t res = svadd_f32_m(pg, va, vb);
        svst1_f32(pg, &dst[i], res); 

        pg = svwhilelt_b32(i + VL, p);
    }
}


// ###### ALLZEROS ###### // 
inline bool allzeros(const double* a) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                         // # double nel registro 
    int32_t N  = (int32_t)SIZE;                             // # double nel vettore

    for (int32_t i = 0; i < N; i += VL) {
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat64_t va = svld1_f64(pg, &a[i]);  
        svfloat64_t v0 = svdup_f64(0.0);
        svbool_t result = svcmpeq_f64(pg, va, v0);          // result[i] = true se va[i] = 0

        svbool_t result_rev = svnot_b_z(pg, result);        // result[i] = false se va[i] = 0
        if (svptest_any(pg, result_rev) != false) {
            return false;                                   // elemento != 0 -> false 
        }
        
        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return true; 
}

inline bool allzeros(const float* a) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # float nel registro 
    int32_t N  = (int32_t)SIZE;                             // # float nel vettore

    for (int32_t i = 0; i < N; i += VL) {
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t v0 = svdup_f32(0.0f);
        svbool_t result = svcmpeq(pg, va, v0);              // result[i] = true se va[i] = 0

        svbool_t result_rev = svnot_b_z(pg, result);        // result[i] = false se va[i] = 0
        if (svptest_any(pg, result_rev) != false) {
            return false;                                   // elemento != 0 -> false 
        }

        pg = svwhilelt_b32(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return true; 
}


// ###### ALLEQUAL ###### // 
inline bool allequal(const double* a, const double* b) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                          // # double nel registro                      
    int32_t N  = (int32_t)SIZE;                              // # double nel vettore

    for (int32_t i = 0; i < N; i += VL) {        
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat64_t va = svld1(pg, &a[i]);  
        svfloat64_t vb = svld1(pg, &b[i]); 
        svbool_t result = svcmpeq_f64(pg, va, vb);          // result[i] = true se va[i] = vb[i]

        svbool_t result_rev = svnot_b_z(pg, result);        // result[i] = false se va[i] = vb[i]
        if (svptest_any(pg, result_rev) != false) {
            return false;                                   // elemento != 0 -> false 
        }

        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return true; 
}

inline bool allequal(const float* a, const float* b) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # float nel registro
    int32_t N  = (int32_t)SIZE;                             // # float nel vettore

    for (int32_t i = 0; i < N; i += VL) {
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);  
        svbool_t result = svcmpeq(pg, va, vb);              // result[i] = true se va[i] = vb[i]

        svbool_t result_rev = svnot_b_z(pg, result);        // result[i] = false se va[i] = vb[i]
        if (svptest_any(pg, result_rev) != false) {
            return false;                                   // elemento != 0 -> false 
        }

        pg = svwhilelt_b32(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return true; 
}


// ###### CMPALL ###### // 
inline int cmpall(const double* a, const double* b) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                         // # double nel registro 
    int32_t N  = (int32_t)SIZE;                             // # double nel vettore
    
    for (int32_t i = 0; i < N; i += VL) {                   // Lunghezza del vettore per questa iterazione
        
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }

        svfloat64_t va = svld1_f64(pg, &a[i]);  
        svfloat64_t vb = svld1_f64(pg, &b[i]); 

        svbool_t mask_gt = svcmpgt(pg, va, vb);             // gt[i]=true se a>b
        svbool_t mask_lt = svcmplt(pg, va, vb);             // gt[i]=true se a<b

        svuint64_t indices = svindex_u64(0, 1);
        uint32_t gt_index = svminv_u64(pg, svsel_u64(mask_gt, indices, svdup_u64(UINT64_MAX)));
        uint32_t lt_index = svminv_u64(pg, svsel_u64(mask_lt, indices, svdup_u64(UINT64_MAX)));

        if (gt_index < lt_index) return 1;                  // a > b
        if (lt_index < gt_index) return -1;                 // a < b

        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return 0;                                               // a = b    
}

inline int cmpall(const float* a, const float* b) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # float nel registro
    int32_t N  = (int32_t)SIZE;                             // # float nel vettore
    
    for (int32_t i = 0; i < N; i += VL) {                   // Lunghezza del vettore per questa iterazione
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat32_t va = svld1(pg, &a[i]);  
        svfloat32_t vb = svld1(pg, &b[i]); 

        svbool_t mask_gt = svcmpgt(pg, va, vb);             // gt[i]=true se a>b
        svbool_t mask_lt = svcmplt(pg, va, vb);             // gt[i]=true se a<b

        svuint32_t indices = svindex_u32(0, 1);
        uint32_t gt_index = svminv_u32(pg, svsel_u32(mask_gt, indices, svdup_u32(UINT32_MAX)));
        uint32_t lt_index = svminv_u32(pg, svsel_u32(mask_lt, indices, svdup_u32(UINT32_MAX)));

        if (gt_index < lt_index) return 1;                  // a > b
        if (lt_index < gt_index) return -1;                 // a < b

        pg = svwhilelt_b32(i + VL, N);
    }
    return 0;                                               // a = b    
}


// ###### CMP0 ###### // 
inline int cmp0(const double* a) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                         // # double nel registro 
    int32_t N  = (int32_t)SIZE;                             // # double nel vettore
    
    for (int32_t i = 0; i < N; i += VL) {                   // Lunghezza del vettore per questa iterazione
        
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }

        svfloat64_t va = svld1_f64(pg, &a[i]);  
        svfloat64_t vb = svdup_f64(0.0);

        svbool_t mask_gt = svcmpgt(pg, va, vb);             // gt[i]=true se a>b
        svbool_t mask_lt = svcmplt(pg, va, vb);             // gt[i]=true se a<b

        svuint64_t indices = svindex_u64(0, 1);
        uint32_t gt_index = svminv_u64(pg, svsel_u64(mask_gt, indices, svdup_u64(UINT64_MAX)));
        uint32_t lt_index = svminv_u64(pg, svsel_u64(mask_lt, indices, svdup_u64(UINT64_MAX)));

        if (gt_index < lt_index) return 1;                  // a > 0
        if (lt_index < gt_index) return -1;                 // a < 0

        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return 0;                                                // a = 0   
}

inline int cmp0(const float* a) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # float nel registro
    int32_t N  = (int32_t)SIZE;                             // # float nel vettore
    
    for (int32_t i = 0; i < N; i += VL) {                   // Lunghezza del vettore per questa iterazione
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // prende elementi da 0 a VL
        }

        svfloat32_t va = svld1(pg, &a[i]);  
        svfloat32_t vb = svdup_f32(0);

        svbool_t mask_gt = svcmpgt(pg, va, vb);             // gt[i]=true se a>b
        svbool_t mask_lt = svcmplt(pg, va, vb);             // gt[i]=true se a<b

        svuint32_t indices = svindex_u32(0, 1);
        uint32_t gt_index = svminv_u32(pg, svsel_u32(mask_gt, indices, svdup_u32(UINT32_MAX)));
        uint32_t lt_index = svminv_u32(pg, svsel_u32(mask_lt, indices, svdup_u32(UINT32_MAX)));

        if (gt_index < lt_index) return 1;                  // a > 0
        if (lt_index < gt_index) return -1;                 // a < 0

        pg = svwhilelt_b32(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return 0;                                               // a = 0
}


// ###### FIND1°NONZERO ###### // 
inline int findFirstNonZero(const double* a) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                         // # double nel registro 
    int32_t N  = (int32_t)SIZE;                             // # double nel vettore
    
    for (int32_t i = 0; i < N; i += VL) {                   // Lunghezza del vettore per questa iterazione
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat64_t va = svld1(pg, &a[i]);  
        svfloat64_t vb = svdup_f64(0.0);

        svbool_t mask = svcmpne(pg, va, vb);                // mask[i] = true se va[i] != 0
        svuint64_t indices = svindex_u64(0, 1);        
        uint64_t min_index = svminv_u64(pg, svsel(mask, indices, svdup_u64(UINT64_MAX)));

        if( min_index != UINT64_MAX)
            return i + min_index;                           // 0 passati + clz relativo 
        
        pg = svwhilelt_b64(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return N;                                               // tutti elementi a 0    
}

inline int findFirstNonZero(const float* a) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                         // # flaot nel reigstro
    int32_t N  = (int32_t)SIZE;                             // # flaot nel vettore
    
    for (int32_t i = 0; i < N; i += VL) {                   // Lunghezza del vettore per questa iterazione
        if (i + VL > N) {
            VL = N - i;                                     // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                      // prende elementi da 0 a VL
        }
        svfloat32_t va = svld1(pg, &a[i]);  
        svfloat32_t vb = svdup_f32(0);

        svbool_t mask = svcmpne(pg, va, vb);                // mask[i] = true se va[i] != 0
        svuint32_t indices = svindex_u32(0, 1);        
        uint32_t min_index = svminv_u32(pg, svsel(mask, indices, svdup_u32(UINT32_MAX)));

        if( min_index != UINT32_MAX)
            return i + min_index;                           // 0 passati + clz relativo 
        
        pg = svwhilelt_b32(i + VL, N);                      // Aggiorna per la prossima iterazione
    }
    return N;                                               // tutti elementi a 0    
}


//###### CONVMUL ######  
inline int convmul(const double* a, const double* b, double* dst) {
    svbool_t pg = svptrue_b64();
    int32_t VL = (int32_t)svcntd();                                 // # double nel registro 
    int32_t N  = (int32_t)SIZE;                                     // # double nel vettore
    svfloat64_t va, vb, vaxb, vb_app, res_add;  
     
    int index_p, fast_f;                                            // index_p = indice per caricare il vettore precedente ; fast_f = stop condition per convoluzione sotto la diagonale 
    int VL_fix = (int32_t)svcntd();
    svbool_t pg_fix = svptrue_b64();

    for (int32_t i = 0; i < N; i += VL) {           
        index_p= i ;                        
    
        if (i + VL > N) {                                           // Lunghezza del vettore per questa iterazione
            VL = N - i;                                             // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b64(0, VL);                              // prende elementi da 0 a VL
        }

        vb = svld1_f64(pg, &b[i]);                                  // carico b ora perche lo aggiorno alla fine del for  
        res_add = svdup_f64(0);
        
        for (int j = 0 ; j < SIZE ; j++){                           // per ogni riga
            va  = svdup_f64(a[j]);
            vaxb = svmul_f64_m(pg, va,  vb);
            res_add = svadd_f64_m(pg, res_add, vaxb);

            if (j % VL_fix == 0) {                                  // momento di aggiornare il registro di appoggio, se i=j siamo sulla diagnonale   
                index_p = (j == i) ? index_p : index_p - VL_fix;                        // aggiorno index blocco da prendere             
                vb_app = (j == i) ? svdup_f64(0) : svld1_f64(pg_fix, &b[index_p]);      // aggiorno il registro di appoggio
                fast_f = (j == i) ? 4 : UINT64_MAX;                                     // aggiorno stop condition se siamo sulla diagnoale ci mancano 4 operazioni prima del braek;
            }       
            fast_f --;

            vb = svinsr_n_f64(vb, vb_app[VL_fix-(j%VL_fix)-1]);      // prendo i valori partendo dal fondo del blocco 
            if ( fast_f == 0)                                        // alla fine della diagonale stop condition
                break; 
        }
        svst1_f64(pg, &dst[i], res_add);

        pg = svwhilelt_b64(i + VL, N);                               // Aggiorna per la prossima iterazione
    }
    return 0;                                                 
}   

inline int convmul(const float* a, const float* b, float* dst) {
    svbool_t pg = svptrue_b32();
    int32_t VL = (int32_t)svcntw();                                 // # flaot nel reigstro
    int32_t N  = (int32_t)SIZE;                                     // # flaot nel vettore
    svfloat32_t va, vb, vaxb, vb_app, res_add;
    
    int index_p, fast_f;                                            // index_p = indice per caricare il vettore precedente ; fast_f = stop condition per convoluzione sotto la diagonale 
    int VL_fix = (int32_t)svcntd();
    svbool_t pg_fix = svptrue_b32();

    for (int32_t i = 0; i < N; i += VL) {                           
        index_p= i ;                                                

        if (i + VL > N) {                                           // Lunghezza del vettore per questa iterazione
            VL = N - i;                                             // Aggiusta la lunghezza se supera la fine del vettore
            pg = svwhilelt_b32(0, VL);                              // prende elementi da 0 a VL
        }

        vb = svld1_f32(pg, &b[i]);                                  // carico b ora perche lo aggiorno alla fine del for
        res_add = svdup_f32(0);
        
        for (int j = 0 ; j < SIZE ; j++){                           // per ogni riga
            va  = svdup_f32(a[j]);
            vaxb = svmul_f32_m(pg, va,  vb);
            res_add = svadd_f32_m(pg, res_add, vaxb);

            if (j % VL_fix == 0) {                                  // momento di aggiornare il registro di appoggio, se i=j siamo sulla diagnonale 
                index_p = (j == i) ? index_p : index_p - VL_fix;                        // aggiorno index blocco da prendere 
                vb_app  = (j == i) ? svdup_f32(0) : svld1_f32(pg_fix, &b[index_p]);     // aggiorno il registro di appoggio
                fast_f  = (j == i) ? 4 : UINT32_MAX;                                    // aggiorno stop condition se siamo sulla diagnoale ci mancano 4 operazioni prima del braek;
            }   
            fast_f --;

            vb = svinsr_n_f32(vb, vb_app[VL_fix-(j%VL_fix)-1]);     // prendo i valori partendo dal fondo del blocco 
            if ( fast_f == 0)                                       // alla fine della diagonale stop condition
                break; 
        }
        svst1_f32(pg, &dst[i], res_add);

        pg = svwhilelt_b32(i + VL, N);                              // Aggiorna per la prossima iterazione
    }
    return 0;                                                 
}   


/*  NON CANCELLARE
- versione piu leggibile del del doppio controllo nella convoluzione finale 
- versione semplificata convmul , no gestione dim registro
- reverse order 
*/


/*
    VERSIONE PIU LEGGIBLE DEL DOPPIO CONTROLLO

    if( j %VL_fix == 0){                                    // tutte le volte che aggiorno il regisgtro di appoggio

        if( j == i ) {                                      // convoluzione nella diagonale con 0
            vb_app = svdup_f64(0);                          // carico 0
        }
        else if( j < i) {                                   // convoluzione con vettore precedente
            index_p -= VL_fix;                              // ogni blocco carico il vettore precedente
            vb_app = svld1_f64(pg_fix, &b[index_p]);     
        }
    } 
*/


/* VERSIONE SEMPLIFICATA CONV
/###### CONVMUL ######  
inline int convmul(const float* a, const float* b, float* dst) {
    svbool_t pg = svwhilelt_b32(0, SIZE);
    svfloat32_t va0;
    svfloat32_t vb07  = svld1_f32 (pg, &b[0]);          //serve caricarlo la prima volta 
    svfloat32_t va0b = svdup_f32(0);
    svfloat32_t accr = svdup_f32(0);

    for (int i = 0 ; i < SIZE ; i++){                   // per ogni riga 
        va0  = svdup_f32(a[i]);     
        if ( i != 0 ){
            vb07  = svinsr_n_f32(vb07, 0);              // inserisce 0 in testa con shift a dx 
        }
        va0b  = svmul_f32_m (pg, va0,  vb07);   
        accr = svadd_f32_m(pg, accr, va0b);
    }
    svst1_f32(pg, &dst[0], accr);
    return 0;
} 
 
inline int convmul(const double* a, const double* b, double* dst) {
    svbool_t pg = svwhilelt_b64(0, SIZE);
    svfloat64_t va0;
    svfloat64_t vb07  = svld1_f64 (pg, &b[0]);          //serve caricarlo la prima volta 
    svfloat64_t va0b = svdup_f64(0);
    svfloat64_t accr = svdup_f64(0);

    for (int i = 0 ; i < SIZE ; i++){                   // per ogni riga
        va0  = svdup_f64(a[i]);
        if ( i != 0 ){
            vb07  = svinsr_n_f64(vb07, 0);              // inserisce 0 in testa con shift a dx 
        }
        va0b  = svmul_f64_m (pg, va0,  vb07);
        accr = svadd_f64_m(pg, accr, va0b);
    }

    svst1_f64(pg, &dst[0], accr);
    return 0;
}  
*/


// ###### REVERSEORDER ###### // 
/* inline svfloat64_t reverseOrder(svfloat64_t v) {
    svbool_t pg = svwhilelt_b64(0, SIZE);  
    svfloat64_t v_reverse = svrev_f64(v);  
    return v_reverse;
}
inline svfloat32_t reverseOrder(svfloat32_t v) {
    const uint32_t index[] = {7, 6, 5, 4, 3, 2, 1, 0};
    svuint32_t idx = svld1_u32(svwhilelt_b32(0, 8), index);
    return svtbl_f32(v, idx);
} */