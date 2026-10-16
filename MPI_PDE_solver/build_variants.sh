#!/bin/bash
# We build the different builds based on backups
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

module load intel/oneapi_2026.1
module load mpi/latest

mkdir -p build bin

build_variant() {
    local name="$1"
    local source="$2"
    local header="$3"
    local directory="build/$name"

    mkdir -p "$directory"
    cp "$source" "$directory/pde_solver.c"
    cp "$header" "$directory/header.h"
    cp util.c "$directory/util.c"

    # Apply the agreed spacing convention to the working copy.
    python3 - "$directory/pde_solver.c" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
text, count = re.subn(
    r"p->dx\s*=\s*1\.0\s*/\s*\(double\)\s*\(p->nx\)\s*;",
    "p->dx = 1.0 / (double)(p->nx - 2);",
    text,
)
if count > 1:
    raise SystemExit("Unexpected multiple dx assignments")
if count == 0 and not re.search(
    r"p->dx\s*=\s*1\.0\s*/\s*\(double\)\s*\(p->nx\s*-\s*2\)",
    text,
):
    raise SystemExit(f"Cannot identify dx assignment in {path}")
path.write_text(text)
PY

    mpicc -O3 \
        "$directory/pde_solver.c" "$directory/util.c" \
        -lm -o "bin/$name"

    echo "Built bin/$name"
}

build_variant standard \
    pde_solver_sequential.c.bak header.h

build_variant sor_1d \
    pde_solver_sor_1d.c.bak header_sor_1d.c.bak

build_variant sor_2d \
    pde_solver_2d_gather.c.bak header_2d_gather.h.bak

build_variant sor_overlap \
    pde_solver.c header.h
