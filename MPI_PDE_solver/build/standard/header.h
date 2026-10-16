#ifndef HEADER_H
#define HEADER_H
#include <mpi.h>


// Struct to hold all simulation parameters
typedef struct
{
    int nx;        // Grid points in X (including boundaries)
    int ny;        // Grid points in Y (including boundaries)
    double dx;     // grid spacing in x-direction (x_{i+1} - x_{i})
    double dy;     // grid spacing in y-direction (y_{j+1} - y_{j})
    double cx;     // Convection velocity along X
    double cy;     // Convection velocity along Y
    double tol;    // Convergence tolerance
    double beta;   // SOR relaxation parameter
    int max_iters; // Maximum iterations allowed
    int no_output; // Skip diagnostics gather and images for benchmarks
} Params;

// Struct to hold 2D grid dimensions and data
typedef struct Grid
{
    int local_cols; // Number of owned interior columns
    int x_start;    // Global index of the first owned column

    MPI_Comm comm; //Cartesian commmunicator MPI
    int below, above;
    int left, right;
    int local_rows; // Number of owned interior rows
    int y_start; // Global index of the first owned row
    int rank; // proces rank
    int size; // total number of processes
    int nx;       // Grid points in X (including boundaries)
    int ny;       // Grid points in Y (including boundaries)
    double **val; // 2D array of values (val[j][i])
} Grid;

// Grid memory and I/O utility functions (util.c)
Grid *allocate_grid(int nx, int ny);
Grid *gather_grid(Grid *local, const Params *p);
void free_grid(Grid *grid);
void write_pgm(const char *filename, Grid *grid);

// Solver, boundary, and diagnostic functions (pde_solver.c)
void parse_args(int argc, char **argv, Params *p);
void error_and_exit(const char *msg);
void init_sol(Grid *u);
void set_boundaries(Grid *u);
void set_rhs(Grid *f, Params *p);
void solve_gauss_seidel(Grid *u, Grid *f, const Params *p, int *out_iters, double *out_diff);
void print_canaries(Grid *u, Grid *f, const Params *p);

// Red black ordering
void exchange_colour(Grid *u, int colour, MPI_Datatype row_colour);
void exchange_halo(Grid *u, MPI_Datatype column_type);
void start_halo_exchange(Grid *u, MPI_Datatype column_type, MPI_Request request[8]);
#endif // HEADER_H
