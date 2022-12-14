#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>

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

	double t1,t2;

// Arrays and parameters to read and store the sparse matrix
	double * val = NULL;
	int * row = NULL;
	int * col = NULL;
	int N;
	int nz;
	const char * element_type ="d"; 
	int symmetrize=1;


	int m,n;
	struct size_m sA;
	double h;

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	else    
	{ 
		A = read_mat(argv[1]);
		sA = get_size(argv[1]);
	}

	if (loadMMSparseMatrix(argv[1], *element_type, true, &N, &N, &nz, &val, &row, &col, symmetrize)){
		fprintf (stderr, "!!!! loadMMSparseMatrix FAILED\n");
		return EXIT_FAILURE;
	} else {
		printf("Matrix loaded from file %s\n",argv[1]);
		printf("N = %d \n",N);
		printf("nz = %d \n",nz);
		printf("val[0] = %f \n",val[0]);
	}


	m = sA.m;
	n = sA.n;

	h = 1./(double)n;
	b = init_source_term(n,h);

	x = (double*) malloc(n * sizeof(double));
	memset(x, 0., n*sizeof(double));

	printf("Call cgsolver() on matrix size (%d x %d)\n",m,n);
	t1 = second();
	cgsolver( A, b, x, m, n );
	t2 = second();
	printf("Time for CG (dense solver)  = %f [s]\n",(t2-t1));
/*
	memset(x, 0., n*sizeof(double));
	printf("Call cgsolver_sparse() on matrix size (%d x %d)\n",m,n);
	t1 = second();
	cgsolver_sparse( val, row, col, b, x, N );
	t2 = second();
	printf("Time for CG (sparse solver) = %f [s]\n",(t2-t1));
*/
	free(A);
	free(b);
	free(x);
	free(val);
	free(row);
	free(col);

//	free(x0);

	return 0;
}


