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
void cgsolver( double *A, double *b_loc, double *x, int m, int n , MPI_Comm mpicommunicator, int * m_ranks, int * displ_ranks){
	double * r;
	double * p;
	double rsold;
	double rsnew;
	double * Ap;
	double * p_loc;
	double * Ap_loc;
	double * tmp;
	double alpha;
	double pAp;

	int lda = m_loc;
	double al = 1.;
	double be = 0.;

	int k = 0;
	int rank;
	int size;
	int m_loc;
	printf("In CG \n");
	
	MPI_Comm_size(mpicommunicator, &size); 
	
  	MPI_Comm_rank(mpicommunicator, &rank);
	

	m_loc = m_ranks[rank];
	printf("CG from process %d of %d with m_loc %d\n", rank, size, m_loc);

	r = (double*) malloc(m_loc* sizeof(double));
	p = (double*) malloc(n* sizeof(double));
	Ap = (double*) malloc(n* sizeof(double));
	tmp = (double*) malloc(m_loc* sizeof(double));
	p_loc = (double*) malloc(m_loc* sizeof(double));
	Ap_loc = (double*) malloc(m_loc* sizeof(double));
	
//  r = b - A * x;
//memset(Ap, 0., n * sizeof(double));
	memset(Ap_loc, 0., m_loc* sizeof(double));
	

	
	cblas_dgemv (CblasColMajor, CblasNoTrans, m_loc,n,  al , A, m_loc, x, 1,  0., Ap_loc, 1);
	
	MPI_Allgatherv(Ap_loc,m_loc, MPI_DOUBLE,Ap, m_ranks,displ_ranks,MPI_DOUBLE, MPI_COMM_WORLD);
	/*
	printf("b,r %d in rank	\n",rank);
	for(int i = 0; i < m_loc; i++) {
        printf("%f ", b_loc[i]);
    printf("\n");}
print_mat( "matrice : \n",A,m_loc, n);
*/
	cblas_dcopy(m_loc, b_loc, 1, tmp, 1);
	cblas_daxpy(m_loc, -1., Ap_loc, 1, tmp, 1 );
	cblas_dcopy(m_loc, tmp, 1, r, 1);

	for(int i = 0; i < m_loc; i++) {
        printf("%f ",r[i]);
    printf("\n");}
//  p = r;
	cblas_dcopy (m_loc, r, 1, p_loc, 1);

//  rsold = r' * r;
	rsold = cblas_ddot (m_loc, r, 1, r, 1);


	// Sum all the contribution to the residual 
	MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	printf(" rank %i, rsold %E\n",rank,rsold);
	memset(p, 0., n * sizeof(double));
    MPI_Allgatherv(p_loc,m_loc, MPI_DOUBLE,p, m_ranks,displ_ranks,MPI_DOUBLE, MPI_COMM_WORLD);

//  for i = 1:length(b)
	while ( k < n ){
//      Ap = A * p;

		// All gather the full vector p on every processor


		memset(Ap_loc, 0., n * sizeof(double));
		
		cblas_dgemv (CblasColMajor, CblasNoTrans, m_loc, n, al, A, lda, p, 1, be, Ap_loc, 1);
		
	
		//MPI_Allreduce(MPI_IN_PLACE, Ap, m, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);

//      alpha = rsold / (p' * Ap);
		//SUM ALL THE COMPONENTS OF PAP 
		pAp = fmax( cblas_ddot(n, p, 1, Ap, 1), rsold * NEARZERO );
		MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
		alpha = rsold / pAp;
		//printf("STEP %d, rank %i, alpha %E\n",k,rank,alpha);


//      x = x + alpha * p;
		cblas_daxpy(m, alpha, p, 1, x, 1);
//      r = r - alpha * Ap;

		cblas_daxpy(m_loc, -alpha, Ap_loc, 1, r, 1);
//      rsnew = r' * r;
		rsnew = cblas_ddot (m_loc, r, 1, r, 1);

		MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);

		// SUM ALL THE COMPONENTS OF RSNEW
//      if sqrt(rsnew) < 1e-10
//            break;
		
		if ( sqrt(rsnew) < TOLERANCE ) break;             // Convergence test

		MPI_Allgatherv(r,m_loc, MPI_DOUBLE,r_tot, m_ranks,displ_ranks,MPI_DOUBLE, MPI_COMM_WORLD);

//      p = r + (rsnew / rsold) * p;
		cblas_dcopy(m, r_tot, 1, tmp2, 1);
		cblas_daxpy(n, (double)(rsnew/rsold), p, 1, tmp, 1);
		cblas_dcopy(n, tmp, 1, p, 1);
		MPI_Allgatherv(p_loc,m_loc, MPI_DOUBLE,p, m_ranks,displ_ranks,MPI_DOUBLE, MPI_COMM_WORLD);

//      rsold = rsnew;
		rsold = rsnew;
		printf("\t[STEP %d] residual = %E\r", k, sqrt(rsold));
		fflush(stdout);
		k++;
	}

	memset(r, 0., n * sizeof(double));
	MPI_Scatter(x, n_loc, MPI_DOUBLE, x_loc, n_loc, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	cblas_dgemv (CblasColMajor, CblasNoTrans, m, n_loc, al, A, lda, x_loc, 1, be, r, 1);
	MPI_Allreduce(MPI_IN_PLACE, r, m, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
	cblas_daxpy(n, -1., b, 1, r, 1);
	double res = cblas_ddot(n, r, 1, r, 1);

	printf("\t[STEP %d] residual = %E, ||Ax -b|| = %E\n",k,sqrt(rsold), sqrt(res));
*/

	free(r);
	free(p);
	free(Ap);
	free(tmp);
	free(Ap_loc);
	free(p_loc);
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


