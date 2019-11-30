
test_mpi:
	mpicc tester_mpi.c knnring_sequential.c knnring_mpi_v1.c -o test_mpi_v1 -lm -lopenblas
	mpirun -np 4 ./test_mpi_v1
	mpicc tester_mpi.c knnring_sequential.c knnring_mpi.c -o test_mpi -lm -lopenblas
	mpirun -np 4 ./test_mpi
	mpicc tester.c knnring_sequential.c -o test_sequential -lm -lopenblas
	./test_sequential


run:
	mpirun -np 4 ./test_mpi_v1
	mpirun -np 4 ./test_mpi

clean:
	rm test_mpi_v1
	rm test_mpi
	rm test_sequential
