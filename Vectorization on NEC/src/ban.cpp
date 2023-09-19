#include "ban.h"

bool Ban::_check_inconsistency(int p, const T num[SIZE]){

    if(num[0]) // num[0] != 0
        return false;

    if(p) // p != 0
        return true;
    for(unsigned i=1; i<SIZE; ++i)
        if(num[i])
            return true;

    return false;
}

// body of any constructor
void Ban::init(int p, const T num[SIZE]){
    this->p = p;
    for(unsigned i=0; i<SIZE; ++i)
        this->num[i] = num[i];
}

// boolean value used only for disambiguation
Ban::Ban(int p, const T num[SIZE], bool check){
    init(p, num);
}

Ban::Ban(int p, const T num[SIZE]){

    if(_check_inconsistency(p, num))
        throw invalid_argument("Inconsistent input for Ban.");

    init(p, num);

}

Ban::Ban(T n){
    p = 0;
    num[0] = n;
    for(unsigned i=1; i<SIZE; ++i)
        num[i] = 0;
}

/*Ban& Ban::operator=(const Ban &b){
    if(this == &b)
        return *this;

    p = b.p;
    for(auto i = 0; i < SIZE; ++i)
        num[i] = b.num[i];

    return *this;
}*/

Ban& Ban::operator=(T n){
    p = 0;
    num[0] = n;

    for(auto i = 1; i < SIZE; ++i)
        num[i] = 0;

    return *this;
}

bool Ban::operator==(const Ban &b) const{
    if(p != b.p)
        return false;

    bool res = true;
    for(unsigned i=0; i<SIZE; ++i)
        if(res && (num[i] != b.num[i])) // res == true
            res = false;

    return res;
}

bool Ban::operator==(T n) const{
    if(p || num[0] != n) // p != 0
        return false;

    if(!n) // n==0, leverages assumption of normal form
        return true;

    bool res = true;
    for(unsigned i=1; i<SIZE; ++i)
        if(res && num[i])  // res == true && num[i] != 0
            res = false;

    return res;
}

// bring a Ban to normal form
void Ban::to_normal_form(){
    if(num[0]) // num[0] != 0
        return;

    // idx memorizes the entry which must be shifted as first
    unsigned idx = 0;

    while(++idx<SIZE && !num[++idx]); // num[idx++] == 0, i.e., not found a nonzero entry


    // execute the left-to-right shift
    unsigned base = 0;
    while(idx < SIZE)
        num[base++] = num[idx++];

    // base now memorizes how many shift has been executed
    // if base == 0, then the inconsistency was on p
    if(!base) // base == 0
        p = 0;
    else{
        p -= SIZE - base;
    }
    // zeroing the shifted entries
    while(base<SIZE)
        num[base++] = 0;

    return;
}

Ban Ban::_sum(const Ban &a, const Ban &b, int diff_p){
    Ban c(a);

    for(unsigned i=0; i<SIZE; ++i){
        if(i>=diff_p)
            c.num[i] += b.num[i-diff_p];
    }

    if(!diff_p) // diff_p == 0
        c.to_normal_form();

    return c;

}

Ban Ban::operator+(const Ban &b) const{
    // check sum with zero to avoid precision loss
    // example: 0 + η^5  = 0 if SIZE <= 5
    if(*this  == 0) // *this == 0
        return b;

    if(b == 0) //b == 0
        return *this;

    int diff_p = p - b.p;

    // if the numbers are too different the precision is not enough to compute the sum
    if(diff_p >= SIZE)
        return *this;

    if(diff_p <= -SIZE)
        return b;

    // code implementation only for one scenario
    if(diff_p < 0)
        return _sum(b, *this, -diff_p);

    return _sum(*this, b, diff_p);
}

Ban Ban::operator-() const{
    Ban b(*this);
    for(T & m : b.num)
            m *= -1;

    return b;
}

void Ban::_mul_conv(const T num_a[SIZE], const T num_b[SIZE], T aux[(SIZE<<1)-1]){

    for(unsigned i=0; i<(SIZE<<1)-1; ++i){
        aux[i] = 0;
        for(unsigned j=0; j<SIZE; ++j)
            if((i-j)>=0 && (i-j)<SIZE)
                aux[i] += num_a[i-j]*num_b[j];
    }
}


void Ban::_mul(const T num_a[SIZE], const T num_b[SIZE], T num_res[SIZE]){
    T aux[(SIZE<<1)-1];

    _mul_conv(num_a, num_b, aux);

    for(unsigned i=0; i<SIZE; ++i)
        num_res[i] = aux[i];
}

Ban Ban::mul_body(const Ban &b) const{
    T num_res[SIZE];
    _mul(num, b.num, num_res);

    Ban c(p+b.p, num_res, false);
    // necessity to check normal form
    // otherwise: (1e-170*α)*(1e-170*α) = α^2(0+0...)
    c.to_normal_form();

    return c;
}

Ban Ban::operator*(const Ban &b) const{
    // introduced for speed-up
    if(*this  == 0 || b == 0) // *this == 0 || b == 0
        return ZERO;

    return this->mul_body(b);
}

void Ban::_div_body(const T num_num[SIZE], const T num_den[SIZE], T num_res[SIZE]){
    T normalizer = num_den[0];
    T den_norm[SIZE], eps[SIZE], eps_tmp[SIZE];
    den_norm[0] = 0;
    for(unsigned i=1; i<SIZE; ++i)
        den_norm[i] = -num_den[i]/normalizer;

    _mul(den_norm, num_num, eps);
    for(unsigned i=0; i<SIZE; ++i)
            num_res[i] += eps[i]; // vectorial sum

    // unrolling of the outer loop to speed up
    for(unsigned j=1; j<=((SIZE-1)>>1); ++j){
        _mul(eps, den_norm, eps_tmp);
        for(unsigned i=0; i<SIZE; ++i)
                num_res[i] += eps_tmp[i]; // vectorial sum  

        _mul(eps_tmp, den_norm, eps);
        for(unsigned i=0; i<SIZE; ++i)
                num_res[i] += eps[i]; // vectorial sum
    }

    // necessary due to unrolling in case SIZE is even
#if EVEN_SIZE
    _mul(eps, den_norm, eps_tmp);
		for(unsigned i=0; i<SIZE; ++i)
			num_res[i] += eps_tmp[i]; // vectorial sum
#endif

    for(unsigned i=0; i<SIZE; ++i)
            num_res[i] /= normalizer; // element-wise division by a real
}

Ban Ban::operator/(const Ban &b) const{
    // check division by/of zero
    if(b == 0) // b == 0
        throw domain_error("division by zero detected");

    if(*this  == 0) // *this == 0
        return ZERO;

    Ban c(*this);

    c.p -= b.p;

    _div_body(this->num, b.num, c.num);

    c.to_normal_form();

    return c;
}

ostream& operator<<(ostream& os, const Ban &b){
    os<<"α^"<<b.p<<'('<<b.num[0];
    for(unsigned i=1; i<SIZE; ++i)
        if(b.num[i] >= 0)
            os<<" + "<<b.num[i]<<"η^"<<i;
        else
            os<<" - "<<-b.num[i]<<"η^"<<i;

    os<<')';

    return os;
}

// readable implementation of file writing
/*
ofstream& operator<<(ofstream& os, const Ban &b){
	os<<"α^"<<b.p<<'('<<b.num[0];
	for(unsigned i=1; i<SIZE; ++i)
		if(b.num[i] >= 0)
			os<<" + "<<b.num[i]<<"η^"<<i;
		else
			os<<" - "<<-b.num[i]<<"η^"<<i;
	os<<")";
	
	return os;
}
*/

// file writing implementation for tests
ofstream& operator<<(ofstream& os, const Ban &b){
    os<<scientific<<setprecision(6);
    os<<" "<<b.p<<" "<<b.num[0]; //initial space because disambiguation
    for(unsigned i=1; i<SIZE; ++i)
        os<<" "<<b.num[i];

    return os;
}

bool Ban::operator<(const Ban &b) const{
    if(p < b.p){
        if(b.num[0] > 0 || (!b.num[0] && num[0] < 0))  // b.num[0] == 0
            return true;

        return false;
    }

    if(p > b.p){
        if(num[0] < 0 || (!num[0] && b.num[0] > 0))  // num[0] == 0
            return true;

        return false;
    }

    bool res = false, solved = false;
    for(unsigned i=0; i<SIZE; ++i){
        if(!solved && num[i] < b.num[i]){
            res = true;
            solved = true;
        }

        if(!solved && num[i] > b.num[i])
            solved = true;
    }

    return res;
}

bool Ban::operator<(T n) const{
    if(p > 0){
        if(num[0] < 0)
            return true;
        return false;
    }

    if(p < 0){
        if(n > 0)
            return true;
        if(n < 0)
            return false;
        // here iff n == 0
        if(num[0] < 0)
            return true;

        return false;
    }
    // here iff p == 0, so comparison among finite numbers
    if(num[0] < n)
        return true;
    if(num[0] > n)
        return false;

    bool res = false, solved = false;
    // num[0] == n, infinitesimal components are crucial
    for(unsigned i=1; i<SIZE; ++i)
        if(!solved && num[i]){  // solved == false && num[i] != 0
            solved = true;
            if(num[i] < 0)
                res = true;
        }
    // here also in case *this == n
    return res;
}

bool operator<(T n, const Ban &b){
    if(b.p > 0){
        if(b.num[0] > 0)
            return true;
        return false;
    }

    if(b.p < 0){
        if(n < 0)
            return true;
        if(n > 0)
            return false;
        // here iff n == 0
        if(b.num[0] > 0)
            return true;

        return false;
    }
    // here iff p == 0, so comparison among finite numbers
    if(b.num[0] > n)
        return true;
    if(b.num[0] < n)
        return false;

    bool res = false, solved = false;
    // num[0] == n, infinitesimal components are crucial
    for(unsigned i=1; i<SIZE; ++i)
        if(!solved && b.num[i]){  // solved == false && b.num[i] != 0
            solved = true;
            if(b.num[i] > 0)
                res = true;
        }
    // here iff *this == n
    return res;
}

Ban abs(const Ban &b){
    if(b.num[0] >= 0)
        return b;

    return -b;
}

void Ban::sum_infinitesimal_real(T num_res[SIZE], T n) const{
    for(unsigned i=SIZE-1; i>0; --i)
        // right-shifting the most significant part of *this by -p
        if(i>=-p)
            num_res[i] = num[i+p];

            // zero padding until first monosemium
        else
            num_res[i] = 0;

    num_res[0] = n;
}

Ban Ban::operator+(T n) const{
    // check sum with zero to avoid precision loss
    // example: η^5 + 0  = 0 if SIZE <= 5
    if(!n)  // n == 0
        return *this;

    if(p >= 0){
        Ban res(*this);
        // *this too big to be affected by n
        if(p-SIZE >= 0)
            return res;

        /*
        T tmp = res.num[p] + n;
        res.num[p]  = tmp;
        */
        res.num[p] += n;
        res.to_normal_form();

        return res;
    }

    T num_res[SIZE];
    this->sum_infinitesimal_real(num_res, n);

    return Ban(0, num_res);
}

Ban Ban::operator*(T n) const{
    /*
    // true speedup ?
    if(n == 0 || *this == 0)
        return ZERO;
    */

    Ban res(*this);
    for(T &m : res.num)
            m *= n;

    res.to_normal_form();

    return res;
}

Ban Ban::operator/(T n) const{
    if(!n)  // n == 0
        throw domain_error("division by zero detected");

    if(*this  == 0)  // *this == 0
        return ZERO;

    Ban res(*this);
    for(T &m : res.num)
            m /= n;

    res.to_normal_form();

    return res;
}

void Ban::denoise(T tol) {
    for(T &m : num)
        if(m < tol && m > -tol){
            m = 0;
        }

    this->to_normal_form();
}

Ban::Ban(const Ban &b) {
    p = b.p;
    memcpy(&(this->num[0]), &b.num[0], sizeof(T)*SIZE);
}