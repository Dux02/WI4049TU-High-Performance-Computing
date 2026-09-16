#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* simple math functions not in <math.h> */
int max(int a, int b) {return a>=b? a: b;}
int min(int a, int b) {return a>=b? b: a;}

/* Mathematical function to integrate: f(x) = x^2 */
double f(double x) { return x * x; }
/* For checking the results: analytical expression for the indefinite integral */
double F(double x) { return (x * x * x)/3.0; }

/* Helper to allocate 1D contiguous double array */
double *allocate_array(int n_elem)
{
    double *arr = (double *)malloc(n_elem * sizeof(double));
    if (!arr)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Allocation of %ld bytes failed on MPI rank %d!\n", n_elem*sizeof(double), rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return arr;
}

/* Sequential implementation of the trapezoidal rule to compute
 * \int_a^b f(x) dx with grid points given in x[i], 0<=i<=n.
 * a=x[0], b=x[n],
 * and function values in fx[i].
 * Note that this means that x and fx need to have n+1 elements each!
 */
double trapezoidal(int n, double const *x, double const *fx)
{
    double I = 0.0;
    for (int i = 1; i <= n; i++)
    {
        double h = x[i] - x[i-1];
        double area = 0.5 * h * (fx[i - 1] + fx[i]);
        I += area;
    }
    return I;
}

/* parallel numerical integration routine where we assume that 
 * local_x[i+1]>local_x[i], 0<=i<local_n
 * local_x[0] on MPI rank r is equal to local_x[local_n-1] on rank r-1 if r>0,
 * fx[i] = f(x[i]), where f is the functino to be integrated.
 *
 * Returns the value of the complete integral on rank nproc-1.
 */
double integrate_v1(int rank, int nproc, int local_n, double const* local_x, double const* local_fx)
{
    /* compute the integral over the local (owned) part of the interval 
     * [local_x[0], local_x[local_n-1]].
     */
    double local_I = trapezoidal(local_n, local_x, local_fx);

    /* now we're missing an offset from the previous process, so communicate! */
    double offset = 0.0;
    if (rank > 0)
    {
       int tag = 0;
        MPI_Recv(&offset, 1, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_I += offset;
    }

    if (rank < nproc - 1)
    {
       int tag = 0;
        MPI_Send(&local_I, 1, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD);
    }
    return local_I;
}

/* TODO: Add functions integrate_v2 and integrate_v3 as instructed in README.md,
 *       using the same function signature as above.
 */


double integrate_v2(int rank, int nproc, int local_n, double const* local_x, double const* local_fx)
{
    /* compute the integral over the local (owned) part of the interval 
     * [local_x[0], local_x[local_n-1]].
     */
    double local_I = trapezoidal(local_n, local_x, local_fx);

    /* now we're missing an offset from the previous process, so communicate! */

    /* Here is an algorithm that needs log(P) messages
     * and ends up with the full result in rank nproc-1.
     * (It's actually an inclusive Scan operation)
     */

    double offset = 0.0;
    double current_val = local_I;

    for (int mask = 1; mask < nproc; mask <<= 1)
    {
        int partner = rank ^ mask;
        if (partner < nproc)
        {
            double recv_val = 0.0;
            MPI_Sendrecv(&current_val, 1, MPI_DOUBLE, partner, 0, &recv_val, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            current_val += recv_val;
            if (partner < rank)
            {
                offset += recv_val;
            }
        }
    }
    return current_val;
}

double integrate_v3(int rank, int nproc, int local_n, double const* local_x, double const* local_fx)
{
    /* compute the integral over the local (owned) part of the interval 
     * [local_x[0], local_x[local_n-1]].
     */
    double local_I = trapezoidal(local_n, local_x, local_fx);

    /* now we're missing an offset from the previous process, so communicate! */

    /* The most elegant way to do this is the collective MPI_Scan, which is made exactly for this purpose */
    MPI_Scan(MPI_IN_PLACE, &local_I, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return local_I;
}

/* This function runs an integration_v* implementation passed as function pointer <int_fcn>
   a number of <ntimes> times and returns the average (elasepd/wallclock)runtime per call 
   in milliseconds [ms] for given input arguments <local_n>, <local_x>, and <local_fx>.
   The time peasured per call is the maximum across all MPI processors.
 */
double run_benchmark(double (*int_fcn)(int, int, int, double const*, double const*), int ntimes, int rank, int nproc,
                            int local_n, double const* local_x, double const* local_fx)
{
    /* Warm-up run to eliminate cold-start cache/network overhead */
    int_fcn(rank, nproc, local_n, local_x, local_fx);

    double tottime = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int it=0; it<ntimes; it++)
    {
        double t0 = MPI_Wtime();
        int_fcn(rank, nproc, local_n, local_x, local_fx);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        tottime += t1-t0;
    }
    return tottime/(double)ntimes * 1e3; /* [ms] */
}

/* ====================================================================
 * MAIN DRIVER
 * Accepts the number of grid points n on the command-line
 * ==================================================================== */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc != 2)
    {
        if (rank==0)
        {
            fprintf(stderr, "  Usage: %s n, where n>=2 is an integer giving the number of intervals (grid cells).\n"
                          "Example: %s 10000 (or %s 1e4)\n",argv[0],argv[0],argv[0]);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    int global_n = (int)atof(argv[1]);
    int num_runs = max(1,100000000/global_n); /* Number of benchmark iterations to get some runtime together */

    if (global_n < nproc)
    {
        fprintf(stderr, "Error: We need at least n=<nproc> intervals.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_n = global_n / nproc;
    double a = 0.0, b = 1.0; /* global integral bounds */
    double h = (b - a) / global_n;
    double a_local = a + rank * local_n * h;

    /* if there are any "leftovers", give them to the last proc */
    if (rank==nproc-1) local_n += (global_n % nproc);

    /* note that local_n is the number of intervals, but the x[i]
     * are grid points, so there are one more than there are intervals.
     */
    double *local_x = allocate_array(local_n+1);
    double *local_fx = allocate_array(local_n+1);

    for (int i = 0; i <= local_n; i++)
    {
        local_x[i] = a_local + i*h;
        local_fx[i] = f(local_x[i]);
    }

    /* Compute the integral for testing/correctness checking:
     * TODO: Include correctness checks for all your variants.
     *       If you do not like the basic 'print' check, calculate
     *       a tolerance based on the mesh size h and the approximation
     *       order of the trapezoidal method and print an error only if
     *       the tolerance is not met.
     */
    double I_an = F(b) - F(a);
    double I_v1 = integrate_v1(rank, nproc, local_n, local_x, local_fx);
    double I_v2 = integrate_v2(rank, nproc, local_n, local_x, local_fx);
    double I_v3 = integrate_v3(rank, nproc, local_n, local_x, local_fx);
    /* note: We only require the total integral value to be present on the last process! */
    if (rank==nproc-1)
    {
        fprintf(stdout, "       MPI INTEGRATION NUMERICAL RESULTS\n");
        fprintf(stdout, "========================================\n");
        fprintf(stdout, "I_an=%f\n",I_an);
        fprintf(stdout, "T_v1=%f, err=%g\n", I_v1,fabs(I_v1-I_an));
        fprintf(stdout, "T_v2=%f, err=%g\n", I_v2,fabs(I_v2-I_an));
        fprintf(stdout, "T_v3=%f, err=%g\n", I_v3,fabs(I_v3-I_an));

        fprintf(stdout, "\n");
        fprintf(stdout, "       MPI INTEGRATION TIMING RESULTS\n");
        fprintf(stdout, "=====================================\n");
        fprintf(stdout, "  Processes (P)      : %d\n", nproc);
        fprintf(stdout, "  Global Points (N)  : %d\n", global_n);
        fprintf(stdout, "  Local Points/Rank  : %d\n", local_n);
        fprintf(stdout, "  Benchmark Runs     : %d\n", num_runs);
        fprintf(stdout, "  Timer Resolution   : %.3e sec\n", MPI_Wtick());
    }

    /* Run Benchmarks for the different versions */
    double t1 = run_benchmark(integrate_v1, num_runs, rank, nproc, local_n, local_x, local_fx);
    /* TODO: benchmark your different variants in the same way */
    double t2=-1.0;
    double t3=-1.0;
    t2 = run_benchmark(integrate_v2, num_runs, rank, nproc, local_n, local_x, local_fx);
    t3 = run_benchmark(integrate_v3, num_runs, rank, nproc, local_n, local_x, local_fx);

    /* Print Performance Log Table on Rank 0 */
    if (rank == 0)
    {
                  fprintf(stdout, "+---------------------------+------------------------+------+------------+\n");
                  fprintf(stdout, "| Implementation            | Elapsed time/call [ms] | Speedup over v1   |\n");
                  fprintf(stdout, "+---------------------------+------------------------+-------------------+\n");
                  fprintf(stdout, "| Original Impl     (v1)    | %16.3g       | %12.2fx     |\n", t1,1.0);
        if (t2>0) fprintf(stdout, "| Improved P2P      (v2)    | %16.3g       | %12.2fx     |\n", t2,t1/t2);
        if (t3>0) fprintf(stdout, "| Collective Comm.  (v3)    | %16.3g       | %12.2fx     |\n", t3,t1/t3);
        /*... and any other variants you may have tried */
                  fprintf(stdout, "+---------------------------+---------------+---------------+------------+\n");

        /* brief output for constructing tables and plots with external tools
         * (use e.g., srun -n 4 ./integrate.x |grep SUMMARY in a job script for varying values of -n)
         */
         if (nproc==1)
         {
             /* for scaling experiments: Print header for the first (sequential) run only */
             fprintf(stdout, "SUMMARY np, %8s, %8s, %8s, %8s\n","global_n","v1 [ms]","v2 [ms]","v3 [ms]\n");
         }
         fprintf(stdout, "SUMMARY %2d, %8d, %8.4g, %8.4g, %8.4g\n", nproc, global_n, t1, t2, t3);
    }

    MPI_Finalize();
    free(local_x);
    free(local_fx);
    return 0;
}
