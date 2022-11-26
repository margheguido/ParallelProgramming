#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <mpi.h>

#include "mmio_wrapper.h"
#include "util.h"
#include "parameters.h"
#include "cg.h"
#include "second.h"


/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix market)
Any matrix in that format can be used to test the code
*/
int main ( int argc, char **argv ) {

	double * A;
	double * x;
	double * b;
	double * A_loc;
	FILE * tfile;
	double t_start, t_end, t_tot, t_avg;

// Arrays and parameters to read and store the sparse matrix
	double * val = NULL;
	int * row = NULL;
	int * col = NULL;
	int nz,r;
	const char * element_type ="d"; 
	int symmetrize=1;


	int m,n, n_loc, nelements, N;
	struct size_m sA;
	double h;

	// 	Initialize MPI
	int size, rank;

  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &size); 
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("Hola from process %d of %d\n", rank, size);

	if (rank ==0)
	{
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	else    
	{ 	
	/* 
		// If I create the matrix depending on the grid size N 
		N = atoi(argv[1]);
		A= creat_mat_lap(N);
		m=N*N;
		n=N*N;
		*/
		A = read_mat(argv[1]);
		sA = get_size(argv[1]);
		
	}
	
	if (loadMMSparseMatrix(argv[1], *element_type, true, &N, &N, &nz, &val, &row, &col, symmetrize)){
		fprintf (stderr, "!!!! loadMMSparseMatrix FAILED\n");
		return EXIT_FAILURE;
	} else {
		printf("Matrix loaded from file %s in rank %i\n",argv[1],rank);
		printf("N = %d \n",N);
		printf("nz = %d \n",nz);
		printf("val[0] = %f \n",val[0]);
	}
	m = sA.m;
	n = sA.n;
	
	
	}
	
	// Bcast m,n to all the processors from rank 0
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	printf("Bcast done on process %d \n", rank);

	// Number of columns for each processor
	// Distribute equally the remainder among the first processors
	n_loc = n / size + (rank < n % size ? 1 : 0);
	
	// Number of elements for each local matrix
	nelements = n_loc *n;

	// Split mattrix A among processors
	
	// Local matrix allocation
	A_loc = (double*) malloc(nelements * sizeof(double));
	memset(A_loc, 0., nelements*sizeof(double));

	int *nelem_ranks;
	int *displ_ranks;
	int *displ2_ranks;
	int *n_ranks;
	
	// m_ranks contain the value of_local of all the ranks, displ_ranks contians the displacemnt of where to start the sending buffer of the matrix for each rank, displ2_ranks contians the displacemnt where to start the sending buffer when splitting a vector
	nelem_ranks = malloc(sizeof(int)*size);
	memset(nelem_ranks, 0., size*sizeof(int));
	displ_ranks = malloc(sizeof(int)*size);
	memset(displ_ranks, 0., size*sizeof(int));
	n_ranks = malloc(sizeof(int)*size);
	memset(n_ranks, 0., size*sizeof(int));
	displ2_ranks = malloc(sizeof(int)*size);
	memset(displ2_ranks, 0., size*sizeof(int));
		
	displ_ranks[0] = 0;
	displ2_ranks[0] = 0;
	for (r = 0; r < size; r++)
	{
		nelem_ranks[r] = (m / size + (r < m % size ? 1 : 0))*m;
		n_ranks[r] = (m / size + (r < m % size ? 1 : 0));
	}
		
	for (r = 0; r < size-1; r++)
	{
		displ_ranks[r+1] = displ_ranks[r] + nelem_ranks[r];
		displ2_ranks[r+1] = displ2_ranks[r] + n_ranks[r];
	}
	
	printf("Parameters set on process %d \n", rank);

	//Print the parameters for the matrix splitting 
	/*
	for (int i = 0; i < size; i++) 
	{
        printf("m_ranks[%d] = %d\tdispls[%d] = %d, m_loc =%d\n", i, nelem_ranks[i], i, displ_ranks[i], n_loc);
    }
	*/
	
	// Scatter the matrix A from rank 0 to A_loc in all the processors
	MPI_Scatterv(A,  nelem_ranks, displ_ranks, MPI_DOUBLE, A_loc, nelements, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
    printf("Scatter done on process %d \n", rank);


	h = 1./(double)n;
	b = init_source_term(n,h);

	x = (double*) malloc(n * sizeof(double));
	memset(x, 0., n*sizeof(double));
	
	// Solve with dense cgsovler
	printf("Call cgsolver() on matrix size (%d x %d)\n",m,n);
	t_start = MPI_Wtime();
	cgsolver( A_loc, b, x, m, n ,MPI_COMM_WORLD,n_ranks,displ2_ranks);
	t_end = MPI_Wtime();
	
// Average the solution time among all processors
	t_tot = t_end - t_start;
	MPI_Allreduce(MPI_IN_PLACE, &t_tot, 1, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
	t_avg = t_tot/size;

	if (rank==0)
	{
		printf("Average time for CG (dense solver)  = %f [s]\n",(t_avg));
		tfile = fopen("timers.txt", "a");
		fprintf(tfile,"%f \n", t_avg);
		fclose(tfile);
		free(A);
	}


	free(b);
	free(x);
	free(val);
	free(row);
	free(col);
	free(A_loc);
//	free(x0);


// 	Close MPI
	MPI_Finalize();
	
	return 0;
}


