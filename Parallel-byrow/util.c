#include "util.h"




// ====================================================================
// Implementation of useful functions
// ====================================================================

/*
print a matrix A of size m x n
*/

void print_mat( char title[], double *A, int m, int n )
{
	double x;
	printf("%s",title);

	for ( int i = 0; i < m; i++ ){
		for ( int j = 0; j < n; j++ ){
			x = A[id(m,i,j)];   //if ( abs( x ) < NEARZERO ) x = 0.0;
			printf("%f (%d)\t",x,id(m,i,j));
		}
		printf("\n");
	}
}

/*
print first d elements from a matrix A of size m x n
*/

void print_first( char title[], double *A, int m, int n, int d )
{
	double x;
	printf("%s",title);

	for ( int i = 0; i < d; i++ ){
		x = A[i];   //if ( abs( x ) < NEARZERO ) x = 0.0;
		printf("%f (%d)\t",x,i);
	}
	printf("\n");
}



/*
reads a matrix A of size m x n

inspired by https://math.nist.gov/MatrixMarket/mmio/c/example_read.c
*/

double* read_mat(const char * restrict fn)
{
	double * mat;
	int n,m;

	int ret_code;
	MM_typecode matcode;
	FILE *f;
	int nz;   
	int i, *I, *J;
	double tmp;

	if ((f = fopen(fn, "r")) == NULL) 
	{
		printf("Could not open matrix");
		exit(1);
	}

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}


//	Matrix is sparse
	if (mm_is_matrix(matcode) && mm_is_coordinate(matcode) ) {
		if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0)
		{
			exit(1);
		}else{
			if ( (m==1) || (n==1) ){
				printf("size of vector = %d x %d\n",m,n);
			}else{
				printf("size of matrix = %d x %d\n",m,n);
			}
		}
	}
//	Matrix is not sparse
	else if (mm_is_matrix(matcode) && mm_is_array(matcode) )
	{
//		printf("case : %%MatrixMarket matrix array real symmetric \n");
		if ((ret_code = mm_read_mtx_array_size(f, &m, &n)) !=0)
		{
			exit(1);
		}else{
			if ( (m==1) || (n==1) ){
				printf("size of vector = %d x %d\n",m,n);
			}else{
				printf("size of matrix = %d x %d\n",m,n);
			}
		}
		nz = m*n;
	}
//	Matrix is something else
	else{
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

/* reserve memory for matrices */

	I = (int *) malloc(nz * sizeof(int));
	J = (int *) malloc(nz * sizeof(int));
	mat = (double*) malloc(m*n * sizeof(double));

	for (i=0; i<m*n; i++)
	{
		mat[i] = 0.;
	}


    /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */


// Matrix is symmetric

	if (mm_is_symmetric(matcode)){

		for (i=0; i<nz; i++)
		{
			if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &tmp)==3){
				I[i]--;  /* adjust from 1-based to 0-based */
				J[i]--;
				mat[I[i]*m+J[i]] = tmp;
				mat[J[i]*m+I[i]] = tmp;
			}else if(fscanf(f, "%lg\n",&tmp)==1){
				mat[i] = tmp;
			}else{
				break;
			}
		}
	}

// Matrix is not symmetric

	if (mm_is_general(matcode)){
		for (i=0; i<nz; i++)
		{
			if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &tmp)==3){
				I[i]--;  /* adjust from 1-based to 0-based */
				J[i]--;
				mat[I[i]*m+J[i]] = tmp;
			}else if(fscanf(f, "%lg\n",&tmp)==1){
				mat[i] = tmp;
			}else{
				break;
			}
		}
	}


	if (f !=stdin) fclose(f);

//	print_mat( "matrice : \n", mat, m, n );
	free(I);
	free(J);
	return mat;
}



/*
Reads the size of the matrix
*/

/*
reads a matrix A of size m x n
*/

struct size_m get_size(const char * restrict fn)
{
	int n,m,nz;
	FILE *f;
	struct size_m size;
	int ret_code;
	MM_typecode matcode;

	if ((f = fopen(fn, "r")) == NULL) 
	{
		printf("Could not open matrix");
		exit(1);
	}

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	if (mm_is_matrix(matcode) && mm_is_coordinate(matcode) ) {
		if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0)
		{
			exit(1);
		}
	}
	else if (mm_is_matrix(matcode) && mm_is_array(matcode) )
	{
		if ((ret_code = mm_read_mtx_array_size(f, &m, &n)) !=0)
		{
			exit(1);
		}
	}


//	if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0)
//		exit(1);
	size.m = m;
	size.n = n;

	if (f !=stdin) fclose(f);

	return size;
}


/*
*/

double* creat_mat_lap(int N)
{
	double * mat;
	int mat_size,i,n;   

	mat_size = N*N*N*N;
	n = N*N;
/* reserve memory for matrices */
	mat = (double*) malloc(mat_size * sizeof(double));

	for (i=0; i<mat_size; i++)
	{
		mat[i] = 0.;
	}




	for (i=0; i<n; i++)
	{
		mat[i*n+i] = 4;
		if (i*n+i+1 < mat_size){
			mat[i*n+i+1] = -1;
		}
		if (i*n+i-1 >=  0){
			mat[i*n+i-1] = -1;
		}

		if (i*n+i+N < mat_size){
			mat[i*n+i+N] = -1;
		}
		if (i*n+i-N >=  0){
			mat[i*n+i-N] = -1;
		}

	}
	
	//print_mat( "matrice : \n", mat,n, n);


	return mat;
}

double* creat_mat_lap_rank(int N, int n_loc, MPI_Comm mpicommunicator)
{
	double * mat;
	int mat_size,i,n, i_offset, i_start, i_end;   
	int rank, size;

	MPI_Comm_size(mpicommunicator, &size); 
  	MPI_Comm_rank(mpicommunicator, &rank);
	// n _loc: number of columns
	// n: number of rows
	n = N*N;

	
	mat_size = n_loc *n;

/* reserve memory for matrices */
	mat = (double*) malloc(mat_size * sizeof(double));

	for (i=0; i<mat_size; i++)
	{
		mat[i] = 0.;
	}


	i_offset = rank*mat_size;
	i_start = rank*n_loc*n +rank*n_loc-i_offset;
	i_end = (rank+1)*n_loc*n +(rank+1)*n_loc-i_offset;

	printf("Fill matrix on rank %i, from %i to %i \n", rank, i_start, i_end);
	
	for (i=rank*n_loc; i<(rank+1)*n_loc; i++)
	{
		mat[i*n+i - i_offset] = 4;
		
		if (i*n+i+1 - i_offset< mat_size){
			mat[i*n+i+1  - i_offset ] = -1;
		}
		if (i*n+i-1 - i_offset>=  0){
			mat[i*n+i-1  - i_offset] = -1;
		}

		if (i*n+i+N - i_offset< mat_size){
			mat[i*n+i+N  - i_offset] = -1;
		}
		if (i*n+i-N - i_offset >=  0){
			mat[i*n+i-N  - i_offset] = -1;
		}

	}
	
	//print_mat( "matrice : \n", mat,n, n_loc);


	return mat;
}