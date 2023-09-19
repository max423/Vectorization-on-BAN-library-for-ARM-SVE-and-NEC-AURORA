#include <ban.h>
#include <NA_Simplex.h>
#include <Eigen/Core>
#include <vector>
#include <iostream>

int main() {

    Matrix<ban, 5, 3> A;
    Matrix<ban, 5, 1> b;
    Matrix<ban, 3, 1> c;

    A << 2, 1, -3,
            2, 3, -2,
            4, 3,  3,
            0, 0,  1,
            1, 2,  1;

    b << 90, 190, 300, 10, 70;

    c << -8, -12, -7;

    vector<int> t = {-1, -1, -1, 0, 1};

    Marix<ban, 5, 1> x;
    ban optimal_value;
    T tol = 1e-4;
    bool flag;

    flag = i_big_m(A, b, c, t, tol, x, optimal_value);

    cout<<x<<endl;
    cout<<optimal_value<<endl;
    cout<<flag<<endl;

    return 0;
}