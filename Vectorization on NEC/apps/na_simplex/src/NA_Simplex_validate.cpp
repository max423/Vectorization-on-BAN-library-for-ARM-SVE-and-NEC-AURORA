#include "NA_Simplex.h"
#include <Eigen/Eigenvalues>
#define EALL Eigen::placeholders::all

void LU_fact(const Matrix<ban, Dynamic, Dynamic> &A, Matrix<ban, Dynamic, Dynamic> const &LU, vector<uint> &pivot){
    const uint n_constraints = A.rows();
    Matrix<ban, Dynamic, Dynamic> LU_ = A;
    iota(pivot.begin(), pivot.end(), 0);

    for(auto k = 0; k < n_constraints; ++k){
        uint kp = k;
        ban amax = abs(LU_(k,k));
        for(auto i = k+1; i < n_constraints; ++i){
            ban absi = abs(LU_(i,k));
            if(absi > amax) {
                kp = i;
                amax = absi;
            }
        }

        if(LU_(kp,k) != 0){
            if(kp != k){
                auto tmp_pivot = pivot[k];
                pivot[k] = pivot[kp];
                pivot[kp] = tmp_pivot;

                for(auto i = 0; i < n_constraints; ++i){
                    auto tmp = LU_(k,i);
                    LU_(k,i) = LU_(kp, i);
                    LU_(kp,i) = tmp;
                }
            }
            auto Akkinv = 1/LU_(k,k);
            for(auto i = k+1; i < n_constraints; ++i) {
                LU_(i, k) *= Akkinv;
            }
        }
        else{
            throw domain_error("pivot equal to zero detected");
        }
        for(auto j = k+1; j < n_constraints; ++j){
            for(auto i = k+1; i < n_constraints; ++i){
                LU_(i,j) -= LU_(i,k) * LU_(k,j);
            }
        }
    }

    (const_cast< Matrix<ban, Dynamic, Dynamic> &>(LU)) = LU_;
}

void LU_solve(const Matrix<ban, Dynamic, Dynamic> &LU, const Vector<ban, Dynamic> &b, const vector<uint> &pivot, Vector<ban, Dynamic> const &x){
    const uint dim = LU.cols();
    Vector<ban, Dynamic> xx(dim);
    Vector<ban, Dynamic> yy(dim);

    // solve for L
    for(auto i = 0; i < dim; ++i){
        auto bi = b(pivot[i]);
        for(auto j = 0; j < i; ++j){
            bi -= LU(i,j)*yy(j);
        }
        yy(i) = bi;
    }

    // solve for U
    for(int i = dim-1; i >= 0; --i){
        auto yi = yy(i);
        for(int j = dim-1; j > i; --j){
            yi -= LU(i,j)*xx(j);
        }
        xx(i) = yi/LU(i,i);
    }

    (const_cast< Vector<ban, Dynamic> &>(x)) = xx;
}

void LU_trans_solve(const Matrix<ban, Dynamic, Dynamic> &LU, const Vector<ban, Dynamic> &b, const vector<uint> &pivot, Vector<ban, Dynamic> const &x){
    const uint dim = LU.cols();
    Vector<ban, Dynamic> xx(dim);
    Vector<ban, Dynamic> yy(dim);

    // solve for U^T
    for(auto i = 0; i < dim; ++i){
        auto bi = b(i);
        for(auto j = 0; j < i; ++j){
            bi -= LU(j,i)*yy(j);
        }
        yy(i) = bi/LU(i,i);
    }

    // solve for L^T
    for(int i = dim-1; i >= 0; --i){
        auto yi = yy(i);
        for(int j = dim-1; j > i; --j){
            yi -= LU(j,i)*xx(j);
        }
        xx(i) = yi;
    }

    xx(pivot) = xx;
    (const_cast< Vector<ban, Dynamic> &>(x)) = xx;
}

/* The method implements the revised simplex method in Box 7.1 on page 103 of Chvatal

	Revised Simplex
	
	max  c'*x
	s.t. Ax = b
	      x >= 0
*/

void debug(const Vector<ban,Dynamic> &A, const char* s){
    cout << s << ": ";
    for(auto x : A)
        cout << x << " ";
    cout << endl << endl;
}

void debug(const vector<uint> &v, const char* s){
    cout << s << ": ";
    for(auto x : v)
        cout << x << " ";
    cout << endl << endl;
}

bool na_simplex(const Matrix<ban, Dynamic, Dynamic> &A, const Vector<ban, Dynamic> &b, const Vector<ban, Dynamic> &c, vector<uint> &B, T tol,
                Vector<ban, Dynamic> const &x, ban &optimal_value){

	// TODO control dimensions coherence

    const uint n_constraints = A.rows();
    const uint n_variables = A.cols();

    // Assume rank non-deficient initial base matrix
    vector<uint> variables;
    variables.reserve(n_variables);
    for(auto i = 0; i < n_variables; ++i)
        variables.push_back(i);

    vector<uint> N;
    N.reserve(n_variables-n_constraints);
    set_difference(variables.begin(), variables.end(), B.begin(), B.end(), inserter(N, N.begin()));

    vector<uint> pivot(n_constraints);
    Matrix<ban, Dynamic, Dynamic> LU(n_constraints, n_constraints);
    LU_fact(A(EALL, B), LU, pivot);
    Vector<ban, Dynamic> xB(n_constraints);
    LU_solve(LU, b, pivot, xB);
    for(auto& xBi : xB) {
        xBi.denoise(tol);
    }
    Vector<ban, Dynamic> xx = Vector<ban, Dynamic>::Zero(n_variables);
    xx(B) = xB;

    Vector<ban, Dynamic> y(n_constraints), sN(n_variables-n_constraints), d(n_constraints), quality(n_constraints);
    int k;
    uint l;
    ban k_val;
    uint tmp;
    vector<uint> zz;
    bool one_positive;
    Index ii;

    while(true){;
        LU_trans_solve(LU, c(B), pivot, y);
        sN = c(N) - A(EALL, N).transpose() * y;
        //debug(c,"c: ");
        //exit(1);
        //debug(sN,"sN: ");


        // entering index
        k = -1;
        for(auto i=0; i < n_variables - n_constraints; ++i){
            k_val = sN(i).lead_mon();
            if(k_val > tol) {
                k = i;
                break;
            }
        }

        if(k == -1) {
            // solution found
            (const_cast< Vector<ban, Dynamic> &>(x)) = xx;
            optimal_value = (c.transpose() * x)(0);
            return true;
        }

        LU_solve(LU, A(EALL, N[k]), pivot, d);
        //debug(d,"d: ");

        zz.clear();
        one_positive = false;
        for(int i = 0; i < n_constraints; ++i)
            if (d(i) > 0){
                zz.push_back(i);
                one_positive = true;
            }

        if(!one_positive){
            // problem unbounded
            return false;
        }

        quality = xB(zz).array() / d(zz).array();
        quality.minCoeff(& ii);


        //debug(quality,"quality: ");

        
        //Efficient but unstable update when using bans
        /*
        auto t = quality(ii);
        xB -= t * d;
        xx(N[k]) = t;
        */

        // more stable update
        l = zz[ii];
        tmp = B[l];
        B[l] = N[k];

        //debug(B,"B: ");


        N[k] = tmp;

        LU_fact(A(EALL, B), LU, pivot);
        LU_solve(LU, b, pivot, xB);

        xx(N[k]) = 0;

        for(auto& elem : xB) {
            elem.denoise(tol);
        }

        //debug(xx,"xx: ");



        xx(B) = xB;
    }
}

/*  problem form

    max c^T x
    s.t. Ax <= b if t < 0
		 Ax  = b if t = 0
		 Ax >= b if t > 0
		  x >= 0

    Assume b >= 0
*/

void modify(Matrix<ban, Dynamic, Dynamic> &A, Vector<ban, Dynamic> &b, Vector<ban, Dynamic> &c, const vector<int> &t, vector<uint> &B){

    uint n_constraints = A.rows();
    uint n_variables = A.cols();

    auto AA = A;
    auto bb = b;
    auto cc = c;

    vector<int> idx_smaller;
    vector<int> idx_equal;
    vector<int> idx_larger;

    for(auto i = 0; i < t.size(); ++i){
        if(t[i] < 0)
            idx_smaller.push_back(i);
        else if(t[i] == 0)
            idx_equal.push_back(i);
        else
            idx_larger.push_back(i);
    }

    uint n_less = idx_smaller.size();
    uint n_equal = idx_equal.size();
    uint n_larger = idx_larger.size();

    /* Make A as follows
             _					   _
			|  Ale	I	0	0	0	|
	   A = 	|  Aeq	0	I	0	0	|
			|_ Age  0	0	I  -I  _|
	*/


    A = Matrix<ban, Dynamic, Dynamic>::Zero(n_constraints, n_variables + n_less + n_equal + 2 * n_larger);
    A(seq(0, n_less - 1), seq(0, n_variables - 1)) = AA(idx_smaller, EALL);
    A(seq(0, n_less - 1), seq(n_variables, n_variables + n_less - 1)) = Matrix<ban, Dynamic, Dynamic>::Identity(n_less, n_less);
    A(seq(n_less, n_less + n_equal -1), seq(0, n_variables - 1)) = AA(idx_equal, EALL);
    A(seq(n_less, n_less + n_equal -1), seq(n_variables + n_less, n_variables + n_less + n_equal - 1)) = Matrix<ban, Dynamic, Dynamic>::Identity(n_equal, n_equal);
    A(seq(n_less + n_equal, n_less + n_equal + n_larger -1), seq(0, n_variables - 1)) = AA(idx_larger, EALL);
    A(seq(n_less + n_equal, n_less + n_equal + n_larger -1), seq(n_variables + n_less + n_equal, n_variables + n_less + n_equal + n_larger - 1)) =  Matrix<ban, Dynamic, Dynamic>::Identity(n_larger, n_larger);
    A(seq(n_less + n_equal, n_less + n_equal + n_larger -1), seq(n_variables + n_less + n_equal + n_larger, n_variables + n_less + n_equal + 2 * n_larger - 1)) = - Matrix<ban, Dynamic, Dynamic>::Identity(n_larger, n_larger);

    /* Make b as follows

        b =  |_ ble	beq	 bge_|^T

     */

    b(seq(0, n_less - 1)) = bb(idx_smaller);
    b(seq(n_less, n_less + n_equal - 1)) = bb(idx_equal);
    b(seq(n_less + n_equal, n_less + n_equal + n_larger -1)) = bb(idx_larger);


    /* Make c as follows

        c =  | _c 0 -α -α 0 |^T

     */

    c = Vector<ban, Dynamic>::Zero(n_variables + n_less + n_equal + 2 * n_larger);
    c(seq(0, n_variables-1)) = cc;
    //c(seq(n_variables + n_less, n_variables + n_less + n_equal + n_larger - 1), 0) = ALPHA * Matrix<T, Dynamic, 1>::Ones(n_equal + n_larger);
    for(auto &elem : c(seq(n_variables + n_less, n_variables + n_less + n_equal + n_larger - 1)))
        elem = -ALPHA;

    B.reserve(n_constraints);
    for(auto i = n_variables; i < n_constraints + n_variables; ++i)
        B.push_back(i);
}

bool i_big_m(Matrix<ban, Dynamic, Dynamic> A, Vector<ban, Dynamic> b, Vector<ban, Dynamic> c, const vector<int> &t, T tol,
             Vector<ban, Dynamic> const &x, ban &optimal_value){

    // TODO check positivity of b

    // TODO check non-infinity of c

    vector<uint> B;
    modify(A, b, c, t, B);

    return na_simplex(A, b, c, B, tol, x, optimal_value);
}


