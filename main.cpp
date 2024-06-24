#include <iostream>
#include <sys/time.h>
#include <math.h>

#ifdef MPI_USE
#include <mpi.h>
#endif

#include "src/utils.h"

using namespace std;

/*
argv[1] b: the right-hand vector b file
*/
int main(int argc, char **argv) {
    // load vector from file
    cout << "load vector from file" << endl;
    int n = 9104256;
    double *vec = new double[n];
    
    // parse arguments
    char* b_file = strdup(argv[1]);
    load_vector(n ,vec, b_file);

    // print vector (first 10 elements)
    for (int i = 0; i < 10; i++) {
        fprintf(stdout, "vec[%d] = %lf\n", i, vec[i]);
    }

    // get memory usageV
    MemUsage();

    // free memory
    delete[] vec;
    free(b_file);
    
    return 0;
}