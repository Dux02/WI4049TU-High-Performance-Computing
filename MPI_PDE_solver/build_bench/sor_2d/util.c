#include "header.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

// Dynamic 2D Array Allocation
Grid *allocate_grid(int nx, int ny)
{
    Grid *grid = (Grid *)malloc(sizeof(Grid));
    grid->nx = nx;
    grid->ny = ny;
    grid->val = (double **)malloc(ny * sizeof(double *));
    double *data = (double *)calloc(nx * ny, sizeof(double));
    for (int j = 0; j < ny; j++)
    {
        grid->val[j] = &data[j * nx];
    }
    return grid;
}

void free_grid(Grid *grid)
{
    if (grid)
    {
        if (grid->val)
        {
            free(grid->val[0]); // Free contiguous data block
            free(grid->val);    // Free pointers
        }
        free(grid);
    }
}

// Output image in PGM format (Portable Gray Map)
void write_pgm(const char *filename, Grid *grid)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        perror("Failed to open file for PGM output");
        return;
    }

    // P2: ASCII Grayscale Format
    fprintf(fp, "P2\n%d %d\n255\n", grid->nx, grid->ny);

    // Find min and max for scaling
    double min_val = grid->val[0][0], max_val = grid->val[0][0];
    for (int j = 0; j < grid->ny; j++)
    {
        for (int i = 0; i < grid->nx; i++)
        {
            if (grid->val[j][i] < min_val)
                min_val = grid->val[j][i];
            if (grid->val[j][i] > max_val)
                max_val = grid->val[j][i];
        }
    }

    double range = (max_val - min_val < 1e-12) ? 1.0 : (max_val - min_val);

    // Write pixels from top to bottom
    for (int j = grid->ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < grid->nx; i++)
        {
            int pixel = (int)(255.0 * (grid->val[j][i] - min_val) / range);
            fprintf(fp, "%d ", pixel);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
