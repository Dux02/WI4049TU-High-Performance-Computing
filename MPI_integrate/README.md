# Numerical Integration with MPI

In this exercise we want to numerically compute an integral with the trapezoidal rule:

$$
I_n = \sum_{i=0}^{n-1} 0.5 (f(x_{i+1})+f(x_i)) h \approx \int_a^b f(x) dx
$$

with $n$ the number of grid points $x_i$, and $h = (b-a)/(n-1)$ the regular grid spacing.

- All tasks are to be solved in the source file ``integrate.c``.
- On DelftBlue, you have to load a module to get the MPI compiler ``mpicc``:
```bash
module load 2026 openmpi
```
- To compile the program, just type ``make``. To 
- In the ``Makefile``, you can switch the compiler flags between optimized (fast) and debugging (slow).
- If you want to know the syntax and semantics (functionality) of a specific MPI function, read the ``man`` page
in your terminal:
```bash
man MPI_Init
man MPI_Send
...
```

---

## Task 0: Distributing the data

- Every MPI program must call ``MPI_Init`` before the first, and ``MPI_Finalize`` after the last
  MPI communication call. Insert them in the ``main`` program and obtain the ``rank`` and ``nproc`` values
  using ``MPI_Comm_rank/size``.
- Adapt the program so that every process owns a part of the $n$ global intervals,
  and make sure the distributed arrays are initialized correctly. The program should
  behave correctly for any combination of ``nproc`` and ``n``.

## Task 1: A first attempt

- The function ``integrate_v1`` already contains communication routines. Test if your initialization
works for the different value $n$ and, e.g., ``-np 1``, ``-np 2``, ``-np 3``, etc. You can run
run these small tests (up to $n=10000$ and $-np 10`` or so) on the login node:
```bash
mpirun -np 2 ./integration.x 1e3``
```

## Task 2: Optimizing the communication

Observe the number of communication calls in variant ``v1`` as a function of ``nproc``.
Think about an algorithm that uses pairwise exchange of messages and achieves asymptotically fewer messages.
Implement it as function ``integrate_v2``, and test again that it produces correct results in various combinations.

**Hints:**
- There are many ways to achieve this, remember that we only
demand that the final value of the integral is available on ``rank nproc-1``.
- There is a function ``MPI_Sendrecv`` that is particularly useful for pairwise exchange of data.
- If you produce a _deadlock_ (your program hangs), use CTRL+C in the terminal to kill the running processes.

## Task 3: Can we do this more elegantly?

So far we have programmed using individual messages (point-to-point/P2P communication).
Look at the "zoo" of MPI collectives and pick the most useful one to implement the communciation pattern we need,
and use it to produce ``integrate_v3``, and test it as before.

## Task 4: Let's benchmark this thing!

We have provided a job script ``bench_integrate.slurm`` that runs your program
for different values of ``n`` and ``nproc``, and tabulates the result in the output file
``bench_integrate.slurm``. You can use it as a starting point and adapt it to your needs.
Create some tables and plots to see which variants work best for combinations of smaller/larger ``n``
and `nproc``.
