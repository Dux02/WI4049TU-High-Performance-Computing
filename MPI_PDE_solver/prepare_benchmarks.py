from pathlib import Path
import os
import subprocess

root = Path(__file__).resolve().parent
destination = root / "bin_bench"
destination.mkdir(exist_ok=True)

variants = {
    "standard": ("standard", "MPI_COMM_WORLD", False),
    "sor_1d": ("sor_1d", "MPI_COMM_WORLD", True),
    "sor_2d": ("sor_2d", "cart_comm", True),
    "sor_overlap": ("sor_overlap", "cart_comm", True),
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of: {old!r}")
    return text.replace(old, new, 1)


for name, (build_name, communicator, parallel) in variants.items():
    original = root / "build" / build_name
    folder = root / "build_bench" / name
    folder.mkdir(parents=True, exist_ok=True)

    text = (original / "pde_solver.c").read_text()

    if name != "sor_overlap":
        # The original sequential version uses MPI_Wtime without MPI_Init.
        if not parallel:
            text = replace_once(
                text,
                "    // Default parameters",
                "    MPI_Init(&argc, &argv);\n\n    // Default parameters",
            )
            text = replace_once(
                text,
                "    return 0;",
                "    MPI_Finalize();\n    return 0;",
            )

        # Synchronize before timing and report the slowest rank.
        text = replace_once(
            text,
            "    double t0 = MPI_Wtime();",
            f"    MPI_Barrier({communicator});\n"
            "    double t0 = MPI_Wtime();",
        )

        beta = "p.beta" if parallel else "1.0"

        timing = f"""
    double local_elapsed = t1 - t0;
    double elapsed;
    int benchmark_rank, benchmark_size;
    MPI_Comm_rank({communicator}, &benchmark_rank);
    MPI_Comm_size({communicator}, &benchmark_size);
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, {communicator});

    if (benchmark_rank == 0)
    {{
        printf("TIMING,%d,%d,%d,%.6f,%d,%.9f\\n",
               benchmark_size, p.nx, p.ny, {beta},
               iters, elapsed);
    }}
"""
        text = replace_once(
            text,
            "    double elapsed = t1 - t0;",
            timing,
        )

        # Existing human-readable results are printed by every rank.
        # Give each rank a defined elapsed value for those messages.
        text = replace_once(
            text,
            '    printf("\\n--- Solver Results ---\\n");',
            '    elapsed = local_elapsed;\n'
            '    printf("\\n--- Solver Results ---\\n");',
        )
    else:
        # Current overlap version already has synchronized maximum timing.
        text = replace_once(
            text,
            '        printf("BENCH,%d,%d,%d,%.6f,%d,%.9f,%.6f\\n",',
            '        printf("TIMING,%d,%d,%d,%.6f,%d,%.9f\\n",',
        )
        text = replace_once(
            text,
            "               iters, elapsed, gflops);",
            "               iters, elapsed);",
        )

    # Skip diagnostics and image gathering when BENCH_QUIET is set.
    start = text.index("    // Verification and Output")
    end = text.index("    // Cleanup", start)

    text = (
        text[:start]
        + '    if (getenv("BENCH_QUIET") == NULL)\n    {\n'
        + text[start:end]
        + "    }\n\n"
        + text[end:]
    )

    (folder / "pde_solver.c").write_text(text)

    for filename in ("header.h", "util.c"):
        (folder / filename).write_bytes(
            (original / filename).read_bytes()
        )

    temporary = destination / f"{name}.tmp"
    subprocess.run(
        [
            "mpicc", "-O3",
            str(folder / "pde_solver.c"),
            str(folder / "util.c"),
            "-lm", "-o", str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, destination / name)
    print(f"Built bin_bench/{name}")
