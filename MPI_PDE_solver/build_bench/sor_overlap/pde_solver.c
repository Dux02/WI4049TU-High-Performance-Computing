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
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    // Coordinate order: y, x. Only x is periodic.
    int periods[2] = {0, 1};
    MPI_Comm cart_comm;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods,
                    0, &cart_comm);

    int cart_rank;
    int coords[2];

    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int below, above, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &below, &above);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    int interior_rows = p.ny - 2;
    int interior_cols = p.nx - 2;

    if (interior_rows < dims[0] || interior_cols < dims[1])
    {
        error_and_exit("Each Cartesian block needs at least one owned row and column.");
    }

    if (p.nx % 2 != 0)
    {
        error_and_exit("Red-black ordering requires an even global nx.");
    }

    int base_rows = interior_rows / dims[0];
    int extra_rows = interior_rows % dims[0];

    int local_rows =
        base_rows + (coords[0] < extra_rows ? 1 : 0);

    int y_start =
        1 + coords[0] * base_rows
          + (coords[0] < extra_rows ? coords[0] : extra_rows);

    int base_cols = interior_cols / dims[1];
    int extra_cols = interior_cols % dims[1];

    int local_cols =
        base_cols + (coords[1] < extra_cols ? 1 : 0);

    int x_start =
        1 + coords[1] * base_cols
          + (coords[1] < extra_cols ? coords[1] : extra_cols);

    printf("Rank %d: coords=(%d,%d), x=%d..%d, y=%d..%d\n",
           cart_rank, coords[0], coords[1],
           x_start, x_start + local_cols - 1,
           y_start, y_start + local_rows - 1);

    printf("Rank %d: below=%d above=%d left=%d right=%d\n",
           cart_rank, below, above, left, right);

    if (p.nx < 3 || interior_rows < size)
    {
        error_and_exit("Need nx >=3 and at least one interior row per process");
    }
    if (p.nx % 2 != 0)
    {
        error_and_exit("Red-black ordering requires an even nx for periodic x.");
    }
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
    Grid *u = allocate_grid(local_cols + 2, local_rows + 2);
    Grid *f = allocate_grid(local_cols + 2, local_rows + 2);

    u->local_rows = f->local_rows = local_rows;
    u->local_cols = f->local_cols = local_cols;

    u->y_start = f->y_start = y_start;
    u->x_start = f->x_start = x_start;

    u->rank = f->rank = cart_rank;
    u->size = f->size = size;
    u->comm = f->comm = cart_comm;

    u->below = f->below = below;
    u->above = f->above = above;
    u->left = f->left = left;
    u->right = f->right = right;
    
    // Initialization
    init_sol(u);
    set_rhs(f, &p);

    MPI_Datatype column_type;
    MPI_Type_vector(u->local_rows, 1, u->nx, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    exchange_halo(u, column_type);
    MPI_Type_free(&column_type);

    printf("Rank %d: allocated nx=%d ny=%d; first owned point=(%d,%d)\n",
           cart_rank, u->nx, u->ny, u->x_start, u->y_start);


    // Solve PDE with timing
    int iters = 0;
    double final_diff = 0.0;

    MPI_Barrier(cart_comm);

    double t0 = MPI_Wtime();
    solve_gauss_seidel(u, f, &p, &iters, &final_diff);
    double local_elapsed = MPI_Wtime() - t0;

    double elapsed;
    MPI_Reduce(&local_elapsed, &elapsed, 1,
               MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (cart_rank == 0)
    {
        double points = (double)(p.nx - 2) * (p.ny - 2);
        double updates = points * iters;
        double gflops = 16.0 * updates / elapsed / 1e9;

        printf("\n--- Solver Results ---\n");
        printf("Status: %s\n",
               final_diff <= p.tol ? "CONVERGED" : "MAX ITERS REACHED");
        printf("Iterations: %d\n", iters);
        printf("Final Change: %.10e\n", final_diff);
        printf("Execution Time: %.9f seconds\n", elapsed);
        printf("Performance: %.6f GFLOP/s\n", gflops);

        printf("TIMING,%d,%d,%d,%.6f,%d,%.9f\n",
               size, p.nx, p.ny, p.beta,
               iters, elapsed);
    }
    if (!p.no_output)
    {
        if (getenv("BENCH_QUIET") == NULL)
    {
    // Verification and Output
        print_canaries(u, f, &p);

        Grid *global_u = gather_grid(u, &p);
        Grid *global_f = gather_grid(f, &p);

        if (cart_rank==0)
        {
            global_f->y_start =1;
            global_f->x_start =1;
            set_rhs(global_f, &p);

            printf("SOR beta: %.2f\n", p.beta);
            write_pgm("rhs.pgm", global_f);
            write_pgm("solution.pgm", global_u);
            printf("Output images saved to 'rhs.pgm' and 'solutions.pgm'.\n");

            free_grid(global_u);
            free_grid(global_f);
        }

    } 
    }

    // Cleanup
    free_grid(u);
    free_grid(f);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

// Command-Line Argument Parser
void parse_args(int argc, char **argv, Params *p)
{
    int opt;
    while ((opt = getopt(argc, argv, "x:y:c:d:r:t:b:qh")) != -1)
    {
        switch (opt)
        {
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
        case 'r':
            p->max_iters = atoi(optarg);
            break;
        case 'q':
            p->no_output = 1;
            break;
        case 'h':
        default:
            printf("Usage: %s [-x nx] [-y ny] [-c cx] [-d cy] [-t tolerance] [-r max_iters] [-q no_output] [-b beta]\n", argv[0]);
            exit(opt == 'h' ? EXIT_SUCCESS : EXIT_FAILURE);
        }
    }

    if (p->max_iters < 1)
    {
        error_and_exit("Maximum iteration count must be at least 1.");  
    }
    /* nx includes two periodic ghost columns.
    ny includes the two physical Dirichlet boundary rows. */
    if (p->nx < 4 || p->ny < 3)
    {
        error_and_exit("Need nx >= 4 and ny >= 3.");
    }

    p->dx = 1.0 / (double)(p->nx - 2);
    p->dy = 1.0 / (double)(p->ny - 1);

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
            int global_i = f->x_start + i - 1;
            double x = global_i * p->dx;
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

    double a_W = idx2 + cx_plus / dx;
    double a_E = idx2 - cx_minus / dx;
    double a_S = idy2 + cy_plus / dy;
    double a_N = idy2 - cy_minus / dy;
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
    MPI_Allreduce(&f_norm_sq, &global_f_norm_sq, 1, MPI_DOUBLE, MPI_SUM, u->comm);
    double f_norm = sqrt(global_f_norm_sq);
    if (f_norm < 1e-12)
        f_norm = 1.0;

    int iter = 0;
    double rel_res = 1.0;

    MPI_Datatype column_type;
    MPI_Type_vector(u->local_rows, 1, u->nx, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    do
    {
        double res_sq = 0.0;
            
        double global_res_sq;
        for (int colour = 0; colour < 2; colour++)
        {
            MPI_Request requests[8];
            start_halo_exchange(u, column_type, requests);

            // Phase 0: interior. Phase 1: block edges.
            for (int phase = 0; phase < 2; phase++)
            {
                if (phase == 1)
                {
                    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
                }

                for (int j = 1; j < u->ny - 1; j++)
                {
                    int global_j = u->y_start + j - 1;

                    for (int i = 1; i < u->nx - 1; i++)
                    {
                        int global_i = u->x_start + i - 1;

                        if ((global_i + global_j) % 2 != colour)
                        {
                            continue;
                        }

                        int on_edge =
                            i == 1 || i == u->nx - 2 ||
                            j == 1 || j == u->ny - 2;

                        if ((phase == 0 && on_edge) ||
                            (phase == 1 && !on_edge))
                        {
                            continue;
                        }

                        double rhs_val =
                            f->val[j][i]
                            + a_W * u->val[j][i - 1]
                            + a_E * u->val[j][i + 1]
                            + a_S * u->val[j - 1][i]
                            + a_N * u->val[j + 1][i];

                        double u_old = u->val[j][i];
                        double res = rhs_val - a_P * u_old;
                        res_sq += res * res;

                        double u_gs = rhs_val / a_P;
                        u->val[j][i] =
                            u_old + p->beta * (u_gs - u_old);
                    }
                }
            }
        }
        MPI_Allreduce(&res_sq, &global_res_sq, 1, MPI_DOUBLE, MPI_SUM, u->comm);

        rel_res = sqrt(global_res_sq) / f_norm;
        iter++;

    } while (rel_res > p->tol && iter < p->max_iters);
    
    exchange_halo(u, column_type);
    MPI_Type_free(&column_type);
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
        if (u->below == MPI_PROC_NULL)
        {
            double dudy_bottom =
                (-3.0 * u->val[0][i]
                 + 4.0 * u->val[1][i]
                 - u->val[2][i]) / (2.0 * dy);

            bottom_wall_flux += dudy_bottom * dx;
        }

        if (u->above == MPI_PROC_NULL)
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
               0, u->comm);

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
    // Describe this rank's block.
    int info[4] = {
        local->y_start,
        local->x_start,
        local->local_rows,
        local->local_cols
    };

    int *all_info = NULL;
    Grid *global = NULL;

    if (local->rank == 0)
    {
        all_info = malloc(4 * local->size * sizeof(int));
        global = allocate_grid(p->nx, p->ny);
    }

    MPI_Gather(info, 4, MPI_INT,
               all_info, 4, MPI_INT,
               0, local->comm);

    // Select owned cells from the local array, excluding ghosts.
    int local_sizes[2] = {local->ny, local->nx};
    int local_subsizes[2] = {
        local->local_rows, local->local_cols
    };
    int local_starts[2] = {1, 1};

    MPI_Datatype owned_type;
    MPI_Type_create_subarray(2, local_sizes, local_subsizes,
                             local_starts, MPI_ORDER_C,
                             MPI_DOUBLE, &owned_type);
    MPI_Type_commit(&owned_type);

    if (local->rank == 0)
    {
        // Copy rank 0's own block without sending to itself.
        for (int j = 1; j <= local->local_rows; j++)
        {
            for (int i = 1; i <= local->local_cols; i++)
            {
                int global_j = local->y_start + j - 1;
                int global_i = local->x_start + i - 1;

                global->val[global_j][global_i] = local->val[j][i];
            }
        }

        // Receive each other rank directly into its global rectangle.
        for (int r = 1; r < local->size; r++)
        {
            int global_sizes[2] = {p->ny, p->nx};
            int block_sizes[2] = {
                all_info[4 * r + 2],
                all_info[4 * r + 3]
            };
            int block_starts[2] = {
                all_info[4 * r],
                all_info[4 * r + 1]
            };

            MPI_Datatype destination_type;
            MPI_Type_create_subarray(2, global_sizes, block_sizes,
                                     block_starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &destination_type);
            MPI_Type_commit(&destination_type);

            MPI_Recv(global->val[0], 1, destination_type,
                     r, 100, local->comm, MPI_STATUS_IGNORE);

            MPI_Type_free(&destination_type);
        }

        // Reconstruct periodic x ghost columns for the output image.
        for (int j = 0; j < global->ny; j++)
        {
            global->val[j][0] = global->val[j][global->nx - 2];
            global->val[j][global->nx - 1] = global->val[j][1];
        }
    }
    else
    {
        MPI_Send(local->val[0], 1, owned_type,
                 0, 100, local->comm);
    }

    MPI_Type_free(&owned_type);
    free(all_info);

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

void exchange_halo(Grid *u, MPI_Datatype column_type)
{
    // Send bottom row down, receive top ghost row
    MPI_Sendrecv(&u->val[1][1],
                u->local_cols, MPI_DOUBLE, u->below, 10,
                &u->val[u->ny-1][1],
                u->local_cols, MPI_DOUBLE, u->above, 10,
                u->comm, MPI_STATUS_IGNORE);
    
    // Send top owned row up, receive bottom ghost row
    MPI_Sendrecv(&u->val[u->ny-2][1],
                u->local_cols, MPI_DOUBLE, u->above, 11,
                &u->val[0][1],
                u->local_cols, MPI_DOUBLE, u->below, 11,
                u->comm, MPI_STATUS_IGNORE);

    // Send left owned column left, receive right ghost column
    MPI_Sendrecv(&u->val[1][1],
                1, column_type, u->left, 12,
                &u->val[1][u->nx-1],
                1, column_type, u->right, 12,
                u->comm, MPI_STATUS_IGNORE);

    // Send right owned colum right, receive left ghost column
    MPI_Sendrecv(&u->val[1][u->nx-2],
                1, column_type, u->right, 13,
                &u->val[1][0],
                1, column_type, u->left, 13,
                u->comm, MPI_STATUS_IGNORE);
}

void start_halo_exchange(Grid *u, MPI_Datatype column_type, MPI_Request requests[8])
{
    // Post receiv first
    MPI_Irecv(&u->val[u->ny-1][1],
               u->local_cols, MPI_DOUBLE, u->above, 10,
               u->comm, &requests[0]);
    MPI_Irecv(&u->val[0][1],
              u->local_cols, MPI_DOUBLE, u->below, 11,
              u->comm, &requests[1]);

    MPI_Irecv(&u->val[1][u->nx - 1],
              1, column_type, u->right, 12,
              u->comm, &requests[2]);

    MPI_Irecv(&u->val[1][0],
              1, column_type, u->left, 13,
              u->comm, &requests[3]);

    // Send owned edges.
    MPI_Isend(&u->val[1][1],
              u->local_cols, MPI_DOUBLE, u->below, 10,
              u->comm, &requests[4]);

    MPI_Isend(&u->val[u->ny - 2][1],
              u->local_cols, MPI_DOUBLE, u->above, 11,
              u->comm, &requests[5]);

    MPI_Isend(&u->val[1][1],
              1, column_type, u->left, 12,
              u->comm, &requests[6]);

    MPI_Isend(&u->val[1][u->nx - 2],
              1, column_type, u->right, 13,
              u->comm, &requests[7]);
}
