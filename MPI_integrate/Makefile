C_FLAGS_OPT=-O3 -march=native -DNDEBUG
C_FLAGS_DBG=-Wall -Werror -O0 -g -fsanitize=address

#pick which flags to use
C_FLAGS=${C_FLAGS_OPT}

# If you just type "make", builds the executable integrate.x
default: integrate.x

# general rule to compile C files into object files
%.o: %.c Makefile
	mpicc ${C_FLAGS} -c $<

# Rule how to link our executable
integrate.x: integrate.o
	mpicc ${C_FLAGS} -o $@ $< -lm

# Rule how to link our executable
integrate2D.x: integrate2D.o
	mpicc ${C_FLAGS} -o $@ $< -lm

# clean up
clean:
	-rm *.o *.x
