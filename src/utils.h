#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <iostream>
#include "mat_io.h"

/**
 * \brief Definition of max, min, abs
 */
#ifndef _MAX_MIN_ABS_
#define _MAX_MIN_ABS_
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define ABS(a) (((a) >= 0.0) ? (a) : -(a))
#endif

// count for the memory usage
static inline void MemUsage();

// check answer function
void check_correctness(int n, int* row_ptr, int* col_idx, double* mat, double* x, double* b);

void check_correctness_complex(int n, int* row_ptr, int* col_idx, double* mat_real, double* mat_imag, 
                               double* x_real, double* x_imag, double* b_real, double* b_imag);

// store vector function
void store_x(int n, double* x, char* filename);
void store_x_complex(int n, double* x_real, double* x_imag, char* filename);

// read vector function
void load_vector(int n, double* vec, char* filename);


static inline void MemUsage()
{
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    long long int mymem = r_usage.ru_maxrss;
#ifdef MPI_USE
    long long int total;
    MPI_Allreduce(&mymem, &total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0) {
        fprintf(stdout, "Total memory usage:                 %.4f MB\n", (double)((double)(total) / (double)(1024)));
        fflush(stdout);
    }
#else
    fprintf(stdout, "Total memory usage:                 %.4f MB\n", (double)((double)(mymem) / (double)(1024)));
    fflush(stdout);
#endif
}

void print_help(){
    std::cout << "please follow this input:\n./solver matrixname.mtx vectorb.rhs [-option]read_base [-option]type [-option]sys_type\n";
    std::cout << "e.g., ./solver matrixname.mtx vectorb.rhs 1 0 0\n";
    std::cout << "\nhelp:\n";
    std::cout << "matrixname.mtx: the filename of matrix A\n";
    std::cout << "vectorb.rhs:    the filename of right-hand side vector b\n";
    std::cout << "read_base:      0-base or 1-base, default 1-base\n";
    std::cout << "type:           type to output time, 0: end to end time, default 0\n";
    std::cout << "                                     1: solver time + solve time\n";
    std::cout << "                                     2: solve time\n";
    std::cout << "sys_type:       type of algebraic systems, 0: real, 1: complex; default 0\n";
}

/*
multiply a csr matrix with a vector x, and get the resulting vector y,
use kekan sum to reduce the floating point error
@param n: the size of matrix
@param row_ptr: the row pointer of the matrix
@param col_idx: the column index of the matrix
@param val: the value of the matrix
@param x: the vector x
@param y: the resulting vector y
*/
void matvec(int n, int* row_ptr, int* col_idx, double* mat, double* x, double* y){
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        double c = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            double num = mat[j] * x[col_idx[j]];
            double z   = num - c;
            double t   = y[i] + z;
            c          = (t - y[i]) - z;
            y[i]       = t;
        }
    }

}

/*
the complex number multiplication
*/
void mul(double* a_real, double* a_imag, double* b_real, double* b_imag, double* c_real, double* c_imag)
{
    *c_real = *a_real * *b_real - *a_imag * *b_imag;
    *c_imag = *a_real * *b_imag + *a_imag * *b_real;
}

/*
the complex number conjugate multiplication
*/
void conj_mul(double* a_real, double* a_imag, double* b_real, double* b_imag, double* c_real, double* c_imag)
{
    *c_real = *a_real * *b_real + (*a_imag * *b_imag);
    *c_imag = *a_real * *b_imag - *a_imag * *b_real;
}


// the complex number addition, use kekan sum
void add(double* add_real, double* add_imag, double* sum_real, double* sum_imag, double* c_real, double* c_imag)
{
    double num_real = *add_real;
    double z_real   = num_real - *c_real;
    double t_real   = *sum_real + z_real;
    *c_real         = (t_real - *sum_real) - z_real;
    *sum_real       = t_real;

    double num_imag = *add_imag;
    double z_imag   = num_imag - *c_imag;
    double t_imag   = *sum_imag + z_imag;
    *c_imag         = (t_imag - *sum_imag) - z_imag;
    *sum_imag       = t_imag;
}


/*
multiply a csr matrix with a vector x, and get the resulting vector y for complex number
*/
void matvec_complex(int n, int* row_ptr, int* col_idx, double* mat_real, double* mat_imag, 
                    double* x_real, double* x_imag, double* y_real, double* y_imag){
    double t[2];
    double c[2];
    for (int i = 0; i < n; i++) {
        y_real[i] = y_imag[i] = 0.0;
        c[0] = c[1] = 0.0;

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            mul(mat_real + j, mat_imag + j, x_real + col_idx[j], x_imag + col_idx[j], t, t + 1);
            add(t, t + 1, y_real + i, y_imag + i, c, c + 1);
        }
    }
} 


// calculate the mode of complex number
void complex_modulus_squared(double* real, double* imag, double* mode) { *mode = *real * *real + (*imag * *imag); }

// calculate the conjugate of complex number
void complex_conjugate(double* num_real, double* num_iamg, double* conj_real, double* conj_imag)
{
    *conj_real = *num_real;
    *conj_imag = -*num_iamg;
}

// calculate the division of complex number
void complext_division(double* a_real, double* a_imag, double* b_real, double* b_imag, double* c_real, double* c_imag)
{
    double  tmp = 0.0;
    double* l;
    l = &tmp;
    complex_modulus_squared(b_real, b_imag, l);
    *c_real = *a_real * *b_real + (*a_imag * *b_imag);
    *c_imag = *a_imag * *b_real - (*a_real * *b_imag);
    *c_real = (*c_real) / (*l);
    *c_imag = (*c_imag) / (*l);
}


/*
calculate the L2 norm of a vector using kekan sum
*/
double vec_l2_norm(double* x, int n){
    double sum = 0.0;
    double c = 0.0;
    for (int i = 0; i < n; i++) {
        double num = x[i] * x[i];
        double z = num - c;
        double t = sum + z;
        c = (t - sum) - z;
        sum = t;
    }
    return sqrt(sum);
}

/*
calculate the L2 norm of a complex vector using kekan sum
*/
void vec_l2_norm_complex(double* x_real, double* x_imag, double* err_real, double* err_imag, int n){
    double sum[2] = {0, 0}, t[2];
    double c[2];
    c[0] = c[1] = 0.0;
    for (int i = 0; i < n; i++) {
        conj_mul(x_real + i, x_imag + i, x_real + i, x_imag + i, t, t + 1);
        add(t, t + 1, sum, sum + 1, c, c + 1);
    }
    *err_real = sqrt(sum[0]);
    *err_imag = sqrt(sum[1]);
}

#include <float.h>
/*
get the max absolute value of a vector
*/
double vec_max_norm(double* x, int n)
{
    double max = DBL_MIN;
    for (int i = 0; i < n; i++) {
        double x_fabs = fabs(x[i]);
        max           = max > x_fabs ? max : x_fabs;
    }
    return max;
}

/*
get the max absolute value of a complex vector
*/
double vec_max_norm_complex(double* x_real, double* x_imag, int n)
{
    double max = DBL_MIN;
    for (int i = 0; i < n; i++) {
        double x_fabs = x_real[i] * x_real[i] + x_imag[i] * x_imag[i];
        max           = max > x_fabs ? max : x_fabs;
    }
    return sqrt(max);
}


/*
check the correctness of the linear system
err1 = sqrt(|| Ax - b ||)
err2 = max(|| Ax - b ||)
err3 = sqrt(|| Ax - b || / || b ||)
precision check using double type
*/
void check_correctness(int n, int* row_ptr, int* col_idx, double* mat, double* x, double* b){
    // create a new vector to store the result of Ax
    double* b_res = (double*)malloc(sizeof(double) * n);
    double* b_err = (double*)malloc(sizeof(double) * n);

    matvec(n, row_ptr, col_idx, mat, x, b_res);
    for (int i = 0; i < n; i++) {
        b_err[i] = b_res[i] - b[i];
    }

    double err_abs = vec_l2_norm(b_err, n);         // absolute l2-norm error
    double err_max = vec_max_norm(b_err, n);        // absolute max-norm error
    double err_rel = err_abs / vec_l2_norm(b, n);   // relative l2-norm error

    fprintf(stdout, "Error || b - Ax ||_2             =  %12.6e\n", err_abs);
    fprintf(stdout, "Error || b - Ax ||_MAX           =  %12.6e\n", err_max);
    fprintf(stdout, "Error || b - Ax ||_2 / || b ||_2 =  %12.6e\n", err_rel);

    // free the memory
    free(b_res);
    free(b_err);
}


/*
check the correctness of the linear system for complex case
err1 = sqrt(|| Ax - b ||)
err2 = max(|| Ax - b ||)
err3 = sqrt(|| Ax - b || / || b ||)
precision check using double type
*/
void check_correctness_complex(int n, int* row_ptr, int* col_idx, double* mat_real, double* mat_imag, 
                               double* x_real, double* x_imag, double* b_real, double* b_imag){
    // create a new vector to store the result of Ax
    double* b_res_real = (double*)malloc(sizeof(double) * n);
    double* b_res_imag = (double*)malloc(sizeof(double) * n);
    double* b_err_real = (double*)malloc(sizeof(double) * n);
    double* b_err_imag = (double*)malloc(sizeof(double) * n);

    matvec_complex(n, row_ptr, col_idx, mat_real, mat_imag, x_real, x_imag, b_res_real, b_res_imag);
    for (int i = 0; i < n; i++) {
        b_err_real[i] = b_res_real[i] - b_real[i];
        b_err_imag[i] = b_res_imag[i] - b_imag[i];
    }

    double err_abs_real = 0.0, err_abs_imag = 0.0;  // absolute l2-norm error
    double b_norm_real = 0.0, b_norm_imag = 0.0;    // relative l2-norm error
    double err_max_real = vec_max_norm_complex(b_err_real, b_err_imag, n); // absolute max-norm error

    vec_l2_norm_complex(b_err_real, b_err_imag, &err_abs_real, &err_abs_imag, n);
    vec_l2_norm_complex(b_real, b_imag, &b_norm_real, &b_norm_imag, n);

    double err_rel_real = err_abs_real / b_norm_real;

    fprintf(stdout, "Error Complex || b - Ax ||_2             =  %12.6e\n", err_abs_real);
    fprintf(stdout, "Error Complex || b - Ax ||_MAX           =  %12.6e\n", err_max_real);
    fprintf(stdout, "Error Complex || b - Ax ||_2 / || b ||_2 =  %12.6e\n", err_rel_real);

    // free the memory
    free(b_res_real);
    free(b_res_imag);
    free(b_err_real);
    free(b_err_imag);
}

/*
store the vector x to a file
*/
void store_x(int n, double* x, char* filename){
    FILE* file_ptr = fopen(filename, "w");
    fprintf(file_ptr, "%d\n", n);
    for (int i = 0; i < n; i++) fprintf(file_ptr, "%lf\n", x[i]);
    fclose(file_ptr);
}

/*
store the vector x to a file for complex number
*/
void store_x_complex(int n, double* x_real, double* x_imag, char* filename){
    FILE* file_ptr = fopen(filename, "w");
    fprintf(file_ptr, "%d\n", n);
    for (int i = 0; i < n; i++) fprintf(file_ptr, "%lf %lf\n", x_real[i], x_imag[i]);
    fclose(file_ptr);
}

/*
load a vector from filename
@param n: the size of vector
@param vec: the vector to store the data
@param filename: the filename to load the data
*/
void load_vector(int n, double* vec, char* filename){
    FILE* file_ptr = fopen(filename, "r");

    int n_right;
    int r = 0;
    char line[MM_MAX_LINE_LENGTH];
    fgets(line, MM_MAX_LINE_LENGTH, file_ptr);

    // the first line is the size of vector
    if (line[0] == '%') {
        r = fscanf(file_ptr, "%d", &n_right);
    } else {
        n_right = atoi(line);
    }

    // check the size of vector is correct
    if (n_right != n) {
        fclose(file_ptr);
        fprintf(stdout, "Invalid size of vector.\n");
        return;
    }

    // load the data from file
    for (int i = 0; i < n_right; i++) r = fscanf(file_ptr, "%lf", &vec[i]);

    fclose(file_ptr);
}

/*
load a complex vector from filename
@param n: the size of vector
@param vec_real: the real part of vector
@param vec_imag: the imaginary part of vector
@param filename: the filename to load the data
*/
void load_vector_complex(int n, double* vec_real, double* vec_imag, char* filename){
    FILE* file_ptr = fopen(filename, "r");

    int n_right;
    int r = 0;
    char line[MM_MAX_LINE_LENGTH];
    fgets(line, MM_MAX_LINE_LENGTH, file_ptr);

    // the first line is the size of vector
    if (line[0] == '%') {
        r = fscanf(file_ptr, "%d", &n_right);
    } else {
        n_right = atoi(line);
    }

    // check the size of vector is correct
    if (n_right != n) {
        fclose(file_ptr);
        fprintf(stdout, "Invalid size of vector.\n");
        return;
    }

    // load the data from file
    for (int i = 0; i < n_right; i++) r = fscanf(file_ptr, "%lf %lf", &vec_real[i], &vec_imag[i]);

    fclose(file_ptr);
}

// return now time
double GetCurrentTime()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
}