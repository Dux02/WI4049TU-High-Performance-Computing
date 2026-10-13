CC=mpicc
C_FLAGS=-O3 #-g -fbounds-check -fsanitize=address

%.o: %.c header.h Makefile
	${CC} ${C_FLAGS} -c $<

pde_solver.x: pde_solver.o util.o
	${CC} ${C_FLAGS} -o $@ $^ -lm

clean:
	-rm -f *.x *.o
