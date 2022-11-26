#include "cg.h"

const double TOLERANCE = 1.0e-10;
const double NEARZERO = 1.0e-14;
/*
	cgsolver solves the linear equation A*x = b where A is 
	of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

*/
void cgsolver( double *A, double *b, double *x, int m, int n , MPI_Comm mpicommunicator,  int *n_ranks,int *displ_ranks ){
	double * r;
	double * p;
	double rsold;
	double rsnew;
	double * Ap;
	double * p_loc;
	double * x_loc;
	double * tmp;
	double alpha;

	int lda = m;
	double al = 1.;
	double be = 0.;

	int k = 0;
	int rank;
	int size;
	int n_loc;

	// MPI Size and rank
	MPI_Comm_size(mpicommunicator, &size); 
	MPI_Comm_rank(mpicommunicator, &rank);
	

	// Number of columns for each processor is n_loc
	n_loc = n_ranks[rank];
	printf("CG from process %d of %d with n_loc %d\n", rank, size, n_loc);

	r = (double*) malloc(n* sizeof(double));
	p = (double*) malloc(n* sizeof(double));
	Ap = (double*) malloc(n* sizeof(double));
	tmp = (double*) malloc(n* sizeof(double));
	p_loc = (double*) malloc(n_loc* sizeof(double));
	x_loc = (double*) malloc(n_loc* sizeof(double));

//  r = b - A * x;
	memset(Ap, 0., n * sizeof(double));
	
	// Every processors own a slice of length n_loc of the vector x
	x_loc = (double*) malloc(n_loc * sizeof(double));
	memset(x_loc, 0., n_loc*sizeof(double));
	
	// Local matrix-vector multiplication
	cblas_dgemv (CblasColMajor, CblasNoTrans, m, n_loc, al , A, m, x_loc, 1,  0., Ap, 1);
	
	// Reduce the result of the multiplication on all processors
	MPI_Allreduce(MPI_IN_PLACE, Ap, m, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);

	// Everyone compute the residual
	cblas_dcopy(n, b, 1, tmp, 1);
	cblas_daxpy(n, -1., Ap, 1, tmp, 1 );
	cblas_dcopy(n, tmp, 1, r, 1);

//  p = r;
	cblas_dcopy (n, r, 1, p, 1);

//  rsold = r' * r;
	rsold = cblas_ddot (n, r, 1, r, 1);
	printf(" rank %i, rsold %E\n",rank,rsold);
	

//  for i = 1:length(b)
	while ( k < n ){
//      Ap = A * p;

		// Allocate local portion of vector p 
		memset(p_loc, 0., n_loc * sizeof(double));
		
		// Scatter the vector p among all processors 
		MPI_Scatterv(p,  n_ranks, displ_ranks, MPI_DOUBLE, p_loc, n_loc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Local matrix-vector multiplication
      	memset(Ap, 0., n * sizeof(double));
		cblas_dgemv (CblasColMajor, CblasNoTrans, m, n_loc, al, A, lda, p_loc, 1, be, Ap, 1);
		
		// Reduce the result of the multiplication on all processors
		MPI_Allreduce(MPI_IN_PLACE, Ap, m, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);

//      alpha = rsold / (p' * Ap);
		alpha = rsold / fmax( cblas_ddot(n, p, 1, Ap, 1), rsold * NEARZERO );
		
//      x = x + alpha * p;
		cblas_daxpy(n, alpha, p, 1, x, 1);
//      r = r - alpha * Ap;
		cblas_daxpy(n, -alpha, Ap, 1, r, 1);
//      rsnew = r' * r;
		rsnew = cblas_ddot (n, r, 1, r, 1);
//      if sqrt(rsnew) < 1e-10
//            break;
		
		if ( sqrt(rsnew) < TOLERANCE ) break;             // Convergence test
//      p = r + (rsnew / rsold) * p;
		cblas_dcopy(n, r, 1, tmp, 1);
		cblas_daxpy(n, (double)(rsnew/rsold), p, 1, tmp, 1);
		cblas_dcopy(n, tmp, 1, p, 1);
	
//      rsold = rsnew;
		rsold = rsnew;
		printf("\t[STEP %d] residual = %E\r", k, sqrt(rsold));
		fflush(stdout);
		k++;
	}

	memset(r, 0., n * sizeof(double));
	// Scatter the final solution x among all processors 
	MPI_Scatterv(x,  n_ranks, displ_ranks, MPI_DOUBLE, x_loc, n_loc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Local matrix-vector 
	cblas_dgemv (CblasColMajor, CblasNoTrans, m, n_loc, al, A, lda, x_loc, 1, be, r, 1);

	// Reduce the result of the multiplication to compute the final residual 
	MPI_Allreduce(MPI_IN_PLACE, r, m, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
	cblas_daxpy(n, -1., b, 1, r, 1);
	double res = cblas_ddot(n, r, 1, r, 1);

	if (rank==0)
	{
		printf("\t[STEP %d] residual = %E, ||Ax -b|| = %E\n",k,sqrt(rsold), sqrt(res));
	}
	

	free(r);
	free(p);
	free(Ap);
	free(tmp);

}

/*
Sparse version of the cg solver
*/

void cgsolver_sparse( double *Aval, int *row, int *col, double *b, double *x, int n ){
	double * r;
	double * p;
	double rsold;
	double rsnew;
	double * Ap;
	double * tmp;
	double alpha;

	int incx = 1;
	int incy = 1;

	int k = 0;

	r = (double*) malloc(n* sizeof(double));
	p = (double*) malloc(n* sizeof(double));
	Ap = (double*) malloc(n* sizeof(double));
	tmp = (double*) malloc(n* sizeof(double));

//    r = b - A * x;
	smvm(n, Aval, col, row, x, Ap);
	cblas_dcopy(n, b, 1, r, 1);
	cblas_daxpy(n, -1., Ap, 1, r, 1);

//  p = r;
	cblas_dcopy (n, r, incx, p, incy);

//  rsold = r' * r;
	rsold = cblas_ddot (n, r, 1, r, 1);

//  for i = 1:length(b)
	while ( k < n ){
//      Ap = A * p;
		smvm(n, Aval, col, row, p, Ap);
//      alpha = rsold / (p' * Ap);
		alpha = rsold / fmax( cblas_ddot(n, p, 1, Ap, 1), rsold * NEARZERO);
//      x = x + alpha * p;
		cblas_daxpy(n, alpha, p, 1, x, 1);
//      r = r - alpha * Ap;
		cblas_daxpy(n, -alpha, Ap, 1, r, 1);
//      rsnew = r' * r;
		rsnew = cblas_ddot (n, r, 1, r ,1);
//      if sqrt(rsnew) < 1e-10
//            break;
		if ( sqrt(rsnew) < TOLERANCE ) break;             // Convergence test
//        p = r + (rsnew / rsold) * p;
		cblas_dcopy(n, r, 1, tmp, 1);
		cblas_daxpy(n, rsnew / rsold, p, 1, tmp, 1);
		cblas_dcopy(n, tmp, 1, p, 1);
	
		//cblas_dcopy(n,p,1,pt,1);
//        rsold = rsnew;
		rsold = rsnew;
		printf("\t[STEP %d] residual = %E\r", k, sqrt(rsold));
		k++;
	}

	smvm(n, Aval, col, row, x, r);
	cblas_daxpy(n, -1., b, 1, r, 1);
	double res = cblas_ddot(n, r, 1, r, 1);

	printf("\t[STEP %d] residual = %E, ||Ax -b|| = %E\n",k,sqrt(rsold), sqrt(res));

	free(r);
	free(p);
	free(Ap);
	free(tmp);
}


/*
Sparse matrix vector multiplication
*/

void smvm(int m, const double* val, const int* col, const int* row, const double* x, double* y)
{
	for (int i=0; i<m; ++i) {
		y[i] = 0.0;
		for (int j=row[i]; j<row[i+1]; ++j){
			y[i] += val[j-1]*x[col[j-1]-1];
		}
	}
}




/*
Initialization of the source term b 
*/

double * init_source_term(int n, double h){
	double * f;
	int i;
	f  = (double*) malloc(n*sizeof(double*));

	for(i = 0; i < n; i++) {
		f[i] = (double)i * -2. * M_PI * M_PI * sin(10.*M_PI*i*h) * sin(10.*M_PI*i*h);
	}
	return f;
}


