#include <NA_Simplex.h>
#include <Eigen/Core>
#include <vector>
#include <iostream>

//template <typename Derived>
//template class Eigen::internal::traits<ban>;
void debug1(const Vector<ban,Dynamic> &A, const char* s){
    cout << s << ": ";
    for(auto x : A)
        cout << x << " ";
    cout << endl << endl;
}
int main(int argc, char *argv[]) {

    /*float num[8] ={0.1,0,0,0,0,0,0,0};
    float num2[8] ={0.1,0,0,0,0,0,0,0};


    Ban a{0,num};
    Ban b{-1,num};
    std::cout << 3*ETA+5 << std::endl;
    return 0;*/


    int info = *argv[1] - '0';
    
    ban optimal_value;
    T tol = 1e-4;
    bool flag;

    if(info == 1){
        // standard
        Matrix<ban, 6, 9> A;
        Matrix<ban, 6, 1> b;
        Matrix<ban, 9, 1> c;

        A <<  2,  1, -3, 1, 0, 0, 0, 0, 0,
              2,  3, -2, 0, 1, 0, 0, 0, 0,
              4,  3,  3, 0, 0, 1, 0, 0, 0,
              0,  0,  1, 0, 0, 0, 1, 0, 0,
              0,  0, -1, 0, 0, 0, 0, 1, 0,
             -1, -2, -1, 0, 0, 0, 0, 0, 1;

        b << 90, 190, 300, 10, -10, -70;

        c << 8, 12, 7, 0, 0, 0, 0, 0, 0;

        vector<uint> B = {1, 2, 3, 4, 5, 7};
        Vector<ban, 9> x;

        flag = na_simplex(A, b, c, B, tol, x, optimal_value);

        cout << optimal_value << endl;
        cout << flag << endl;

        return 0;
    }

    if(info == 2){
        // i-big-m standard
        Matrix<ban, 5, 3> A;
        Matrix<ban, 5, 1> b;
        Matrix<ban, 3, 1> c;

        A << 2, 1, -3,
             2, 3, -2,
             4, 3,  3,
             0, 0,  1,
             1, 2,  1;

        b << 90, 190, 300, 10, 70;

        c << 8, 12, 7;

        vector<int> t = {-1, -1, -1, 0, 1};

        Matrix<ban, 5, 1> x;

        flag = i_big_m(A, b, c, t, tol, x, optimal_value);

        //cout<<x<<endl;
        cout<<optimal_value<<endl;
        cout<<flag<<endl;

        return 0;
    }

    if(info == 3){
        // na
        Matrix<ban, 6, 9> A;
        Matrix<ban, 6, 1> b;
        Matrix<ban, 9, 1> c;

        A <<  2,  1, -3, 1, 0, 0, 0, 0, 0,
              2,  3, -2, 0, 1, 0, 0, 0, 0,
              4,  3,  3, 0, 0, 1, 0, 0, 0,
              0,  0,  1, 0, 0, 0, 1, 0, 0,
              0,  0, -1, 0, 0, 0, 0, 1, 0,
             -1, -2, -1, 0, 0, 0, 0, 0, 1;

        b << 90, 190, 300, 10, -10, -70;

        c << 8 + 14 * ETA, 12 +  10 * ETA, 7 + 2 * ETA, 0, 0, 0, 0, 0, 0;

        vector<uint> B = {1, 2, 3, 4, 5, 7};
        Vector<ban, 9> x;

        flag = na_simplex(A, b, c, B, tol, x, optimal_value);

        cout << optimal_value << endl;
        cout << flag << endl;

        return 0;
    }

    if(info == 4){
        // i-big-m na
        Matrix<ban, 5, 3> A;
        Matrix<ban, 5, 1> b;
        Matrix<ban, 3, 1> c;

        A << 2, 1, -3,
             2, 3, -2,
             4, 3,  3,
             0, 0,  1,
             1, 2,  1;

        b << 90, 190, 300, 10, 70;

        c << 8 + 14 * ETA, 12 +  10 * ETA, 7 + 2 * ETA;
        //debug1(c,"c: ");
        //exit(1);
        vector<int> t = {-1, -1, -1, 0, 1};

        Matrix<ban, 5, 1> x;


        flag = i_big_m(A, b, c, t, tol, x, optimal_value);

        //cout<<x<<endl;
        cout<<optimal_value<<endl;
        cout<<flag<<endl;

        return 0;
    }

    if(info == 5){
        // full na
        Matrix<ban, 9, 11> A;
        Vector<ban, 9> b;
        Vector<ban, 11> c;

        A <<  -1,            -1,            1, 0, 0, 0, 0, 0, 0, 0, 0,
             -15,           -30 - ETA,      0, 1, 0, 0, 0, 0, 0, 0, 0,
              1,            -1,             0, 0, 1, 0, 0, 0, 0, 0, 0,
              30 - ETA,     15,             0, 0, 0, 1, 0, 0, 0, 0, 0,
              1,             1,             0, 0, 0, 0, 1, 0, 0, 0, 0,
              5 - 2 * ETA,  10 - 2 * ETA,   0, 0, 0, 0, 0, 1, 0, 0, 0,
              1,             1,             0, 0, 0, 0, 0, 0, 1, 0, 0,
              35,           35 - 2 * ETA,   0, 0, 0, 0, 0, 0, 0, 1, 0,
              -1,            1,             0, 0, 0, 0, 0, 0, 0, 0, 1;

        b << -30,
             -900 + 45 * ETA - ETA * ETA,
             60,
             1800 - 75 * ETA - 2 * ETA * ETA,
             75,
             525 - 145 * ETA,
             70,
             2450 - 70 * ETA - 2 * ETA * ETA,
             70;

        c = Vector<ban, Dynamic>::Zero(11);
        c(0) = 8 + 14 * ETA;
        c(1) = 12 +  10 * ETA;

        vector<uint> B = {0, 1, 4, 5, 6, 7, 8, 9, 10};
        Vector<ban, 9> x;

        flag = na_simplex(A, b, c, B, tol, x, optimal_value);

        cout << optimal_value << endl;
        cout << flag << endl;

        return 0;
    }

    if(info == 6){
        // full na i-big-m
        Matrix<ban, 9, 2> A;
        Matrix<ban, 9, 1> b;
        Matrix<ban, 2, 1> c;

        A <<  1,             1,
                15,            30 + ETA,
                1,            -1,
                30 + ETA,     15,
                1,             1,
                5 - 2 * ETA,  10 - 2 * ETA,
                1,             1,
                35,            35 - 2 * ETA,
                -1,            1;

        b << 30,
            900 -45 * ETA + ETA * ETA,
            60,
            1800 - 75 * ETA - 2 * ETA * ETA,
            75,
            525 - 145 * ETA,
            70,
            2450 - 70 * ETA - 2 * ETA * ETA,
            70;


        vector<int> t = {1, 1, -1, -1, -1, -1, -1, -1, -1};
        return 0;
    }

    cout << "Implementation missing" << endl;

    return 0;
}