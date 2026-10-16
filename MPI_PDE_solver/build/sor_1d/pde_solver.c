#include "header.h"

#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default parameters
    Params p = {.nx = 128, .ny = 128, .cx = 2.0, .cy = 1.0, .tol = 1e-6, .beta=1.5, .max_iters = 1000000};
    double beta;   // SOR relaxation parameter

    parse_args(argc, argv, &p);
    if (!isfinite(p.beta) || p.beta <= 1.0 || p.beta >= 2.0)
    {
        error_and_exit("SOR requires 1 < beta < 2.");
    }
    
    int interior_rows = p.ny - 2;
    if (p.nx < 3 || interior_rows < size)
    {
        error_and_exit("Need nx >=3 and at least one interior row per process");
    }
    if (p.nx % 2 != 0)
    {
        error_and_exit("Red-black ordering requires an even nx for periodic x.");
    }
    int base_rows = interior_rows / size;
    int remainder = interior_rows % size;

    int local_rows = base_rows + (rank < remainder ? 1 : 0);

    int y_start = 1 + rank * base_rows +
                            (rank < remainder ? rank : remainder);

    printf("Rank %d: owns global rows %d to %d (%d rows)\n",
            rank, y_start, y_start + local_rows -  1, local_rows);
    int rank_sum = 0;
    MPI_Allreduce(&rank, &rank_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("Sum of ranks: %d (expected %d)\n",
                rank_sum, size * (size-1)/2);
    }
    printf("=== 2D Convection-Diffusion Solver ===\n");
    printf("Grid: %d x %d | cx: %.2f | cy: %.2f | Tol: %.1e\n", p.nx, p.ny, p.cx, p.cy, p.tol);

    // Memory Allocation
    Grid *u = allocate_grid(p.nx, local_rows + 2);
    Grid *f = allocate_grid(p.nx, local_rows + 2);

    u->local_rows = f->local_rows=local_rows;
    u->y_start = f->y_start=y_start;
    u->rank = f->rank = rank;
    u->size = f->size= size;
    // Initialization
    init_sol(u);
    set_rhs(f, &p);
    set_boundaries(u);
    
    printf("Rank %d: bottom ghost = %.0f, top ghost = %.0f\n",
           rank, u->val[0][1], u->val[u->ny - 1][1]);

    // Solve PDE with timing
    int iters = 0;
    double final_diff = 0.0;

    double t0 = MPI_Wtime();
    solve_gauss_seidel(u, f, &p, &iters, &final_diff);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    printf("\n--- Solver Results ---\n");
    printf("Status: %s\n", (iters < p.max_iters) ? "CONVERGED" : "MAX ITERS REACHED");
    printf("Iterations: %d\n", iters);
    printf("Final Change: %.4e\n", final_diff);
    printf("Execution Time: %.6f seconds\n", elapsed);

    // Verification and Output
    print_canaries(u, f, &p);
    Grid *global_u = gather_grid(u, &p);
    Grid *global_f = gather_grid(f, &p);

    if (rank==0)
    {
        global_f->y_start =1;
        set_rhs(global_f, &p);

        printf("SOR beta: %.2f\n", p.beta);
        write_pgm("rhs.pgm", global_f);
        write_pgm("solution.pgm", global_u);
        printf("Output images saved to 'rhs.pgm' and 'solutions.pgm'.\n");

        free_grid(global_u);
        free_grid(global_f);
    }

    // Cleanup
    free_grid(u);
    free_grid(f);

    MPI_Finalize();
    return 0;
}

// Command-Line Argument Parser
void parse_args(int argc, char **argv, Params *p)
{
    int opt;
    while ((opt = getopt(argc, argv, "x:y:c:d:r:t:b:h")) != -1)
    {
        switch (opt)
        {
        case 'r':
            p->max_iters = atoi(optarg);
            break;
        case 'x':
            p->nx = atoi(optarg);
            break;
        case 'y':
            p->ny = atoi(optarg);
            break;
        case 'c':
            p->cx = atof(optarg);
            break;
        case 'd':
            p->cy = atof(optarg);
            break; // '-d' for Y convection
        case 't':
            p->tol = atof(optarg);
            break;
        case 'b':
            p->beta= atof(optarg);
            break;
        case 'h':
        default:
            printf("Usage: %s [-x nx] [-y ny] [-c cx] [-d cy] [-t tolerance] [-b beta]\n", argv[0]);
            exit(opt == 'h' ? EXIT_SUCCESS : EXIT_FAILURE);
        }
    }
    p->dx = 1.0 / (double)(p->nx - 2);     // note the periodic BC
    p->dy = 1.0 / (double)(p->ny - 1); // for ny=2, only the Dirichlet points exist
}

/* function that prints an error message and exits */
void error_and_exit(const char *msg)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "The follow error occured on MPI rank %d. Will abort MPI program:\n'%s'\n", rank, msg);
    MPI_Abort(MPI_COMM_WORLD, -1);
}

// Separate function for setting boundary layers cleanly
void init_sol(Grid *u)
{
    for (int j = 0; j < u->ny; j++)
    {
        for (int i = 0; i < u->nx; i++)
        {
            u->val[j][i] = 0.0; // Initial guess
        }
    }
}

// Right-Hand Side source term setup
void set_rhs(Grid *f, Params *p)
{
    for (int j = 0; j < f->ny; j++)
    {
        int global_j = f->y_start +j -1;
        double y = global_j * p->dy;
        for (int i = 0; i < f->nx; i++)
        {
            double x = i * p->dx;
            // Source peak near (0.25, 0.25)
            f->val[j][i] = 100.0 * exp(-1000.0 * ((x - 0.25) * (x - 0.25) + (y - 0.25) * (y - 0.25)));
        }
    }
}

// Set periodic boundary conditions by copying values at x=0 and x=1 to ghost nodes
// at the opposite side of the grid
void set_boundaries(Grid *u)
{
    int below = (u->rank>0) ? u->rank - 1 : MPI_PROC_NULL;
    int above = (u->rank < u->size -1) ? u->rank + 1 : MPI_PROC_NULL;

    // Dirichlet conditions at top (j=ny-1) and bottom (j=0)
    for (int i = 0; i < u->nx; i++)
    {
        u->val[0][i] = 0.0;
        u->val[u->ny - 1][i] = 0.0;
    }

    // Periodic Ghost Node Exchange in X-direction
    // Ghost left (i=0) gets inner right (i=nx-2)
    // Ghost right (i=nx-1) gets inner left (i=1)
    for (int j = 0; j < u->ny; j++)
    {
        u->val[j][0] = u->val[j][u->nx - 2];
        u->val[j][u->nx - 1] = u->val[j][1];
    }
    // Send the first owned row down; receive upper ghost row.
    MPI_Sendrecv(u->val[1], u->nx, MPI_DOUBLE, below, 0,
                u->val[u->ny -1], u->nx, MPI_DOUBLE, above, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send the last owned row up, receive the lower ghost row.
    MPI_Sendrecv(u->val[u->ny-2], u->nx, MPI_DOUBLE, above, 1,
                u->val[0], u->nx, MPI_DOUBLE, below, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

// Gauss-Seidel solver that iteratively solves Au=f, where A is implemented
// using a 5-point stencil representing the finite idfference discretization of
// stationary PDE at hand. The iteration is stopped if ||u_{i+1}-u_{i}||_2 < p->tol * ||f||_2,
// or the maximum iterations (p->max_iter) are reached.
//
// Returns: final solution u, the number of iterations performed (out_iters), and the final
// realtive integral update difference ||u_{it} - u_{it-1}||_2 / ||f||_2
void solve_gauss_seidel(Grid *u, Grid *f, const Params *p, int *out_iters, double *out_diff)
{
    double dx = p->dx;
    double dy = p->dy;
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);

    double cx_plus = (p->cx > 0) ? p->cx : 0.0;
    double cx_minus = (p->cx < 0) ? p->cx : 0.0;
    double cy_plus = (p->cy > 0) ? p->cy : 0.0;
    double cy_minus = (p->cy < 0) ? p->cy : 0.0;

    double a_P = 2.0 * idx2 + 2.0 * idy2 + (cx_plus - cx_minus) / dx + (cy_plus - cy_minus) / dy;

    // Compute initial norm of RHS ||F||_2 for relative residual check
    double f_norm_sq = 0.0;
    for (int j = 1; j < u->ny - 1; j++)
    {
        for (int i = 1; i < u->nx - 1; i++)
        {
            f_norm_sq += f->val[j][i] * f->val[j][i];
        }
    }
    double global_f_norm_sq;
    MPI_Allreduce(&f_norm_sq, &global_f_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double f_norm = sqrt(global_f_norm_sq);
    if (f_norm < 1e-12)
        f_norm = 1.0;

    int iter = 0;
    double rel_res = 1.0;

    MPI_Datatype row_colour;
    MPI_Type_vector((u->nx -2)/2,1,2, MPI_DOUBLE, &row_colour);
    MPI_Type_commit(&row_colour);
    do
    {
        double res_sq = 0.0;

        for (int colour = 0; colour < 2; colour++)
        {
            for (int j = 1; j < u->ny - 1; j++)
            {
                int global_j = u->y_start + j - 1;

                for (int i = 1; i < u->nx - 1; i++)
                {
                    if ((i + global_j) % 2 != colour)
                    {
                        continue;
                    }

                    double rhs_val =
                        f->val[j][i]
                        + (idx2 + cx_plus / dx) * u->val[j][i - 1]
                        + (idx2 - cx_minus / dx) * u->val[j][i + 1]
                        + (idy2 + cy_plus / dy) * u->val[j - 1][i]
                        + (idy2 - cy_minus / dy) * u->val[j + 1][i];

                    double res = rhs_val - a_P * u->val[j][i];
                    res_sq += res * res;

                    double u_old = u->val[j][i];
                    double u_gs = rhs_val / a_P;

                    u->val[j][i] = u_old + p->beta * (u_gs - u_old);
                }
            }
            exchange_colour(u, colour, row_colour);
        }
        double global_res_sq;
        MPI_Allreduce(&res_sq, &global_res_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        rel_res = sqrt(global_res_sq) / f_norm;
        iter++;

    } while (rel_res > p->tol && iter < p->max_iters);
    
    MPI_Type_free(&row_colour);
    *out_iters = iter;
    *out_diff = rel_res;
}

// Computes and prints integral quantities that can be used
// as 'canary' values, i.e., as a quick sanity check that
// you didn't make indexing or communication mistakes when
// parallelizing the code
void print_canaries(Grid *u, Grid *f, const Params *p)
{
    double total_source = 0.0;
    double l2_norm_sq = 0.0;
    double h1_seminorm_sq = 0.0;
    double energy_source_inflow = 0.0;

    double dx = p->dx;
    double dy = p->dy;
    int nx = u->nx;
    int ny = u->ny;

    // Integrate over physical domain (excluding ghost cells)
    for (int j = 1; j < ny - 1; j++)
    {
        for (int i = 1; i < nx - 1; i++)
        {
            double u_val = u->val[j][i];
            double f_val = f->val[j][i];

            total_source += f_val * dx * dy;
            l2_norm_sq += u_val * u_val * dx * dy;
            energy_source_inflow += u_val * f_val * dx * dy;

            double du_dx = (u->val[j][i + 1] - u->val[j][i - 1]) / (2.0 * dx);
            double du_dy = (u->val[j + 1][i] - u->val[j - 1][i]) / (2.0 * dy);

            h1_seminorm_sq += (du_dx * du_dx + du_dy * du_dy) * dx * dy;
        }
    }

    // 2nd-order wall fluxes at y=0 and y=1
    double bottom_wall_flux = 0.0;
    double top_wall_flux = 0.0;

    for (int i = 1; i < nx - 1; i++)
    {
        if (u->rank == 0)
        {
            double dudy_bottom =
                (-3.0 * u->val[0][i]
                 + 4.0 * u->val[1][i]
                 - u->val[2][i]) / (2.0 * dy);

            bottom_wall_flux += dudy_bottom * dx;
        }

        if (u->rank == u->size - 1)
        {
            double dudy_top =
                (3.0 * u->val[ny - 1][i]
                 - 4.0 * u->val[ny - 2][i]
                 + u->val[ny - 3][i]) / (2.0 * dy);

            top_wall_flux += dudy_top * dx;
        }
    }
    double local[6] = {
        total_source,
        l2_norm_sq,
        h1_seminorm_sq,
        energy_source_inflow,
        bottom_wall_flux,
        top_wall_flux
    };

    double global[6];

    MPI_Reduce(local, global, 6, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);

    if (u->rank != 0)
    {
        return;
    }

    total_source = global[0];
    l2_norm_sq = global[1];
    h1_seminorm_sq = global[2];
    energy_source_inflow = global[3];
    bottom_wall_flux = global[4];
    top_wall_flux = global[5];

    double wall_flux_balance = (bottom_wall_flux - top_wall_flux) - total_source;
    double global_energy_balance = h1_seminorm_sq - energy_source_inflow;

    printf("\n========================================================================================================="
           "===============\n");
    printf("                                              CANARY VALIDATION REPORT                                     "
           "             \n");
    printf("==========================================================================================================="
           "=============\n");
    printf("%-25s | %-18s | %-60s\n", "Integral Quantity", "Computed Value", "Expected Behavior / Usage");
    printf("--------------------------+--------------------+-----------------------------------------------------------"
           "-------------\n");
    printf("%-25s | %-18.10e | %-60s\n", "Wall Flux Balance", wall_flux_balance,
           "Mass balance error; converges to 0 at O(h)");
    printf("%-25s | %-18.10e | %-60s\n", "Global Energy Balance", global_energy_balance,
           "Energy balance error; converges to 0 at O(h)");
    printf("%-25s | %-18.10e | %-60s\n", "L2 Norm", sqrt(l2_norm_sq), "L2 norm of u; converges to continuous value");
    printf("%-25s | %-18.10e | %-60s\n", "H1 Semi-Norm", sqrt(h1_seminorm_sq),
           "H1 semi-norm; converges to continuous value");
    printf("==========================================================================================================="
           "=============\n\n");
}

Grid *gather_grid(Grid *local, const Params *p)
{
    Grid *global = NULL;
    int *counts = NULL;
    int *displacements = NULL;

    if (local->rank == 0)
    {
        global = allocate_grid(p->nx, p->ny);

        counts = malloc(local->size * sizeof(int));
        displacements = malloc(local->size * sizeof(int));

        int interior_rows = p->ny -2;
        int base_rows = interior_rows / local -> size;
        int extra_rows = interior_rows % local->size;

        for (int r = 0; r < local->size; r++)
        {
            int rows = base_rows + (r < extra_rows ? 1 : 0);
            int start = 1 + r * base_rows +
                        (r < extra_rows ? r: extra_rows);

            counts[r] = rows * p->nx;
            displacements[r]= start *p->nx;
        }
    }

    MPI_Gatherv(local->val[1],
                local->local_rows * local->nx,
                MPI_DOUBLE,
                local->rank == 0 ? global->val[0] : NULL,
                counts, displacements, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    free(counts);
    free(displacements);

    return global;
}

void exchange_colour(Grid *u, int colour, MPI_Datatype row_colour)
{
    int below = (u->rank > 0) ? u -> rank - 1 : MPI_PROC_NULL;
    int above = (u->rank < u->size -1) ? u->rank + 1 : MPI_PROC_NULL;

    int first_global = u->y_start;
    int last_global = u->y_start + u->local_rows -1;

    // First column of colour
    int first_start = 1 + ((1 + first_global) % 2 != colour);
    int last_start = 1 + ((1 + last_global) % 2 != colour);

    int bottom_start = 1 + ((1 + first_global - 1) % 2 != colour);
    int top_start = 1 + ((1 + last_global - 1) % 2 != colour);

    MPI_Sendrecv(&u->val[1][first_start],
                1, row_colour, below, 0,
                &u->val[u->ny -1][top_start],
                1, row_colour, above, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&u->val[u->ny - 2][last_start],
                1, row_colour, above, 1,
                &u->val[0][bottom_start],
                1, row_colour, below, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Refresh periodic ghost columns after each colour sweep
    for (int j = 0; j < u->ny; j++)
    {
        u->val[j][0] = u->val[j][u->nx -2];
        u->val[j][u->nx -1] = u->val[j][1];
    }
}
