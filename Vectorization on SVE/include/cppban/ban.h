#include <iostream>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstring>
using namespace std;
typedef double T;     // generates inaccuracies in division proportional to 1e-6

class Ban;
typedef Ban ban;

#define SIZE 12
#define EVEN_SIZE !(SIZE & 1u)

#pragma pack(push, 1)
class Ban{

	int p;
	T num[SIZE];

	// utility functions
	void to_normal_form();
	Ban mul_body(const Ban &b) const;
	void sum_infinitesimal_real(T num_res[SIZE], T n) const;

	// static functions
	static bool _check_inconsistency(int p, const T num[SIZE]);
	static Ban _sum(const Ban &a, const Ban &b, int diff_p);
	static void _div_body(const T num_num[SIZE], const T num_den[SIZE], T num_res[SIZE]);
	static void _mul(const T num_a[SIZE], const T num_b[SIZE], T num_res[SIZE]);
	static void _mul_conv(const T num_a[SIZE], const T num_b[SIZE], T aux[]);

	// constructor without consistency check
	Ban(int p, const T num[SIZE], bool check);
	void init(int p, const T num[SIZE]);

	// utility for boolean conversion
	//typedef void (Ban::*bool_type)() const;
    //void this_type_does_not_support_comparisons() const {}
	//explicit operator int() const = delete;

	public:
	
	// constructors
	Ban(){};
	Ban(int p, const T num[SIZE]);
	Ban(T n);
    Ban(const Ban& b);

    // assignment opertator
    Ban& operator=(const Ban &b) = default;
    Ban& operator=(T n);


	// boolean convertion
	//explicit inline operator bool() const{return num[0];};
	//inline operator bool_type() const {
    //  return num[0] ? &Ban::this_type_does_not_support_comparisons : 0;
    //};

	// algebraic operations
	Ban operator+(const Ban &b) const;
	Ban operator-() const;
	inline Ban operator-(const Ban &b) const {return *this+(-b);};
	Ban operator*(const Ban &b) const;
	Ban operator/(const Ban &b) const;
	inline Ban& operator+=(const Ban &b) {*this = *this + b; return *this;};
	inline Ban& operator-=(const Ban &b) {return *this+=-b;};
	inline Ban& operator*=(const Ban &b) {*this = *this * b; return *this;};
	inline Ban& operator/=(const Ban &b) {*this = *this / b; return *this;};
	//Ban operator>>(unsigned i) const;
	//Ban operator<<(unsigned i) const;
	friend Ban abs(const Ban &b);

	// operatori di uscita
	friend ostream& operator<<(ostream &os, const Ban &b);
	friend ofstream& operator<<(ofstream &os, const Ban &b);

	// ordering operators
	bool operator==(const Ban& b) const;
	inline bool operator!=(const Ban& b) const {return !(*this == b);};
	bool operator<(const Ban &b) const;
	inline bool operator>(const Ban &b)  const {return b<*this;};
	inline bool operator<=(const Ban &b) const {return !(b<*this);}; // *this <= b <-> !(*this>b) <-> !(b<*this)
	inline bool operator>=(const Ban &b) const {return !(*this<b);};

	// speedup functions for compuations with reals
	Ban operator+(T n) const;
	inline Ban operator-(T n) const {return *this+(-n);};
	Ban operator*(T n) const;
	Ban operator/(T n) const; // make inline? *this*(1/n)
	inline Ban& operator+=(T n) {*this = *this + n; return *this;};
	inline Ban& operator-=(T n) {*this = *this + (-n); return *this;};
	inline Ban& operator*=(T n) {*this = *this * n; return *this;};
	inline Ban& operator/=(T n) {*this = *this / n; return *this;};

	inline friend Ban operator+(T n, const Ban &b) {return b+n;};
	inline friend Ban operator-(T n, const Ban &b) {return -b+n;};
	inline friend Ban operator*(T n, const Ban &b) {return b*n;};
	inline friend Ban operator/(T n, const Ban &b) {Ban c(n); return c/b;};
	
	bool operator==(T n) const;
	inline bool operator!=(T n) const{return !(*this == n);};
	bool operator<(T n) const;
	inline bool operator>(T n) const {return n<*this;};
	inline bool operator<=(T n) const {return !(n<*this);};
	inline bool operator>=(T n) const {return !(*this<n);};

	inline friend bool operator==(T n, const Ban &b) {return b == n;};
	inline friend bool operator!=(T n, const Ban &b) {return !(b == n);};
	friend bool operator<(T n, const Ban &b);
	inline friend bool operator>(T n, const Ban &b)  {return b<n;};
	inline friend bool operator<=(T n, const Ban &b) {return !(b<n);};
	inline friend bool operator>=(T n, const Ban &b) {return !(n<b);};

	// external functions
	inline int degree() const {return p;};
    inline T lead_mon() const {return num[0];};
    void denoise(T tol);
};
#pragma pack(pop)

// To modify according to SIZE
//constexpr T _[] = {1.0, 0, 0};
//constexpr T __[] = {0.0, 0, 0};
constexpr T _[] = {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0 };
constexpr T __[] = {0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 , 0};
const Ban ALPHA(1, _);
const Ban ETA(-1, _);
const Ban ZERO(0, __);
const Ban ONE(0, _);
