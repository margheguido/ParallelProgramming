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
	

// Arrays and parameters to read and store the sparse matrix
	double * val = NULL;
	int * row = NULL;
	int * col = NULL;
	int nz;
	const char * element_type ="d"; 
	int symmetrize=1;


	int m,n, r, m_loc, nelements, N,  m_loc0, m_loc1, which_m, which_m0;
	struct size_m sA;
	double h, t_start, t_end,  t_tot,t_avg;
	FILE *tfile;
	MPI_Datatype typeblock0,typeblock0_loc, typeblock1, typeblock1_loc;

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
	{ 	/* 
		// If I create the matrix depending on the grid size N 
		N = atoi(argv[1]);
		A= creat_mat_lap(N);
		m=N*N;
		n=N*N;
		*/
		A = read_mat(argv[1]);
		sA = get_size(argv[1]);
		m = sA.m;
		n = sA.n;
	
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
	
	
	}
	
	// Bcast m,n to all the processors from rank 0
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	printf("Bcast done on process %d \n", rank);
	
	
	// Number of rows for each processor
	// Distribute equally the remainder among the first processors
	m_loc = m / size + (rank < m % size ? 1 : 0);

	// Number of elements for each local matrix
	nelements = m_loc *n;


	//MPI_Type_vector(n,1, m, MPI_DOUBLE, &typerow);
	//MPI_Type_commit(&typerow);

	// 2 possible values of m_loc
	m_loc0 = m / size;
	m_loc1 = m / size +1;

	//Definition of the data types needed to split the matrix among processors 

	// block of m/size rows in the big matrix
	MPI_Type_vector(n,m_loc0, m, MPI_DOUBLE, &typeblock0);
	MPI_Type_commit(&typeblock0);

	// block of m/size rows in the local matrix
	MPI_Type_vector(n,m_loc0, m_loc0, MPI_DOUBLE, &typeblock0_loc);
	MPI_Type_commit(&typeblock0_loc);
	
	// block of m/size+1 rows in the big matrix for the ranks that get the remainders row
	MPI_Type_vector(n,m_loc1 , m, MPI_DOUBLE, &typeblock1);
	MPI_Type_commit(&typeblock1);

	// block of m/size+1 rows in the big matrix for the local matrix of theranks that get the remainders row
	MPI_Type_vector(n,m_loc1 , m_loc1, MPI_DOUBLE, &typeblock1_loc);
	MPI_Type_commit(&typeblock1_loc);
	
	// Local matrix allocation
	A_loc = (double*) malloc(nelements * sizeof(double));
	memset(A_loc, 0., nelements*sizeof(double));

	// m_ranks contain the value of m_local of all the ranks, displ_ranks contians the displacemnt of where to start the sending buffer for each rank
	int *m_ranks;
	int *displ_ranks  ;
	
	m_ranks = malloc(sizeof(int)*size);
	memset(m_ranks, 0., size*sizeof(int));
	displ_ranks = malloc(sizeof(int)*size);
	memset(displ_ranks, 0., size*sizeof(int));

	for (r = 0; r < size; r++)
	{	
		m_ranks[r] = m / size + (r < m % size ? 1 : 0);
	}
	
	displ_ranks[0] = 0;
	for (r = 0; r < size-1; r++)
	{
		displ_ranks[r+1] = displ_ranks[r] + m_ranks[r];
	}
	printf("Parameters set on process %d \n", rank);

	//Print the parameters for the matrix splitting 
	/*
    for (int i = 0; i < size; i++) 
	{
        printf("m_ranks[%d] = %d\tdispls[%d] = %d, m_loc =%d\n", i, m_ranks[i], i, displ_ranks[i], m_loc);
    }
	*/

	// Switch data type rank 0 sends with respect to the value of m_loc of the rank recieving 
	MPI_Datatype blocktypes[2] = {typeblock0, typeblock1};
	MPI_Datatype blocktypes_loc[2] = {typeblock0_loc, typeblock1_loc};


	//Split the matrix from rank 0 among all the processors 
	if (rank==0)
	{	
		// Switch data type with respect to the value of m_loc
		which_m0 = (0 < m % size ? 1 : 0);
		MPI_Sendrecv(&A[displ_ranks[0]], 1, blocktypes[which_m0],0, 0,A_loc, 1, blocktypes_loc[which_m0],0,0, MPI_COMM_WORLD , MPI_STATUS_IGNORE);

		for (r = 1; r < size; r++)
		{	
			//  Switch data type with respect to the value of m_loc: if which_m=1 I use blocktype1
			which_m = (r < m % size ? 1 : 0);
			MPI_Send(&A[displ_ranks[r]], 1, blocktypes[which_m],r, 0, MPI_COMM_WORLD);
		}

	//	printf("Sent from process %d \n", rank);
	//	print_mat( "matrice: \n",A,m, n);
	//	printf("Recieved on process %d \n", rank);
	//	print_mat( "matrice : \n",A_loc,m_loc, n);
	}
	else
	{	
		which_m = (rank < m % size ? 1 : 0);
		MPI_Recv(A_loc, 1, blocktypes_loc[which_m],0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//	printf("Recieved on process %d \n", rank);
	//	print_mat( "matrice : \n",A_loc,m_loc, n);
}

	h = 1./(double)n;
	b = init_source_term(n,h);

	x = (double*) malloc(n * sizeof(double));
	memset(x, 0., n*sizeof(double));

	// Solve with dense cgsovler
	printf("Call cgsolver() on matrix size (%d x %d)\n",m,n);
	t_start = MPI_Wtime();
	cgsolver( A_loc, b, x, m, n ,MPI_COMM_WORLD, m_ranks, displ_ranks);
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
	free(m_ranks);
	free(displ_ranks);


// 	Close MPI
	MPI_Finalize();
	
	return 0;
}


