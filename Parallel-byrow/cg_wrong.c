#include "cg.h"
#include <mpi.h>

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
void cgsolver( double *A, double *b, double *x, int m, int n ){
	double * r;
	double * r_loc;
	double * Ap_loc;
	double * p;
	double rsold;
	double rsnew;
	double * Ap;
	double * tmp;
	double alpha;

	int lda = m;
	double al = 1.;
	double be = 0.;

	int k = 0;
	int rank, size, m_loc;
	MPI_Comm_size(MPI_COMM_WORLD, &size); 
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Number of rows for each processor is m_loc
	m_loc = m/size;

	r = (double*) malloc(m* sizeof(double));
	p = (double*) malloc(m* sizeof(double));
	Ap = (double*) malloc(m* sizeof(double));
	tmp = (double*) malloc(m* sizeof(double));

	Ap_loc = (double*) malloc(m_loc* sizeof(double));
	r_loc = (double*) malloc(m_loc* sizeof(double));

//  r = b - A * x;
	memset(Ap, 0., m * sizeof(double));
	memset(Ap_loc, 0., m_loc * sizeof(double));
	cblas_dgemv (CblasColMajor, CblasNoTrans, m_loc, n, 1. , A, n, x, 1,  0., Ap_loc, 1);
	MPI_Allgather(Ap_loc, m_loc, MPI_DOUBLE, Ap, m_loc, MPI_DOUBLE, MPI_COMM_WORLD);

	cblas_dcopy(m, b, 1, tmp, 1);
	cblas_daxpy(m, -1., Ap, 1, tmp, 1 );
	cblas_dcopy(m, tmp, 1, r, 1);

//  p = r;
	cblas_dcopy (m, r, 1, p, 1);

//  rsold = r' * r;
	rsold = cblas_ddot (m, r, 1, r, 1);


//  for i = 1:length(b)
	while ( k < n ){
//      Ap = A * p;
      	memset(Ap, 0., m * sizeof(double));
		memset(Ap_loc, 0., m_loc * sizeof(double));
		cblas_dgemv (CblasColMajor, CblasNoTrans, m_loc, n, al, A, lda, p, 1, be, Ap_loc, 1);
		MPI_Allgather(Ap_loc, m_loc, MPI_DOUBLE, Ap, m_loc, MPI_DOUBLE, MPI_COMM_WORLD);

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

	memset(r, 0., m * sizeof(double));
	memset(r_loc, 0., m_loc * sizeof(double));
	cblas_dgemv (CblasColMajor, CblasNoTrans, m_loc, n, al, A, lda, x, 1, be, r_loc, 1);
	MPI_Allgather(r_loc, m_loc, MPI_DOUBLE,r, m_loc, MPI_DOUBLE, MPI_COMM_WORLD);

	cblas_daxpy(n, -1., b, 1, r, 1);
	double res = cblas_ddot(n, r, 1, r, 1);

	printf("\t[STEP %d] residual = %E, ||Ax -b|| = %E\n",k,sqrt(rsold), sqrt(res));

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
	int i_start;
	f  = (double*) malloc(n*sizeof(double*));
	i_start =0;
	// when I call this function in parallel n is m_loc
	
	//i_start = rank * n;

	for(i = 0; i < n; i++) {
		f[i] = (double)i * -2. * M_PI * M_PI * sin(10.*M_PI*(i + i_start)*h) * sin(10.*M_PI*(i +i_start)*h);
	}
	return f;
}


