#include "knnring.h"
#include "mpi.h"
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

# define rnd 100000000

void swap(double *a, double *b);
int partition(double *A, int left, int right);
double quickselect(double *A, int left, int right, int k);
int comparator(const void *p,const void *q){
  if (*(double*)p - *(double *)q < 0)
    return -1;
  if (*(double*)p - *(double *)q > 0)
    return 1;

  return 0;
}

knnresult kNN(double * X, double * Y, int n, int m, int d, int k)
{
  if(k>m)
    k=m;
  else if(k<1)
    k=1;

  double *SumX=(double *)malloc(n*sizeof(double));
  for(int i=0;i<n;i++)
  {
    SumX[i]=0;
    for(int j=0;j<d;j++)
    SumX[i]+=pow(X[i+n*j],2);
    //printf("SumX[%d]=%lf \n",i,SumX[i] );
  }

  double *SumY=(double *)malloc(m*sizeof(double));
  for(int i=0;i<m;i++)
  {
    SumY[i]=0;
    for(int j=0;j<d;j++)
    SumY[i]+=pow(Y[i+m*j],2);
    //printf("SumY[%d]=%lf \n",i,SumY[i] );
  }

  double *Drow=(double *)malloc(n*m*sizeof(double));
  double **D=(double **)malloc(m*sizeof(double *));
  double **Dtemp=(double **)malloc(m*sizeof(double *));
  for (int i = 0; i < m; i++)
  {
    D[i]=(double *)malloc(n*sizeof(double));
    Dtemp[i]=(double *)malloc(n*sizeof(double));
  }


  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n,m,d, -2, X,n,Y,m,0,Drow, n);

  for(int i=0; i<n; i++)
    for(int j=0; j <m;j++)
    {
      double dd=(SumX[i] + Drow[i+j*n] +SumY[j]);
      //if (dd < 0) dd=0;

      dd=(double)((int)(dd*rnd))/(rnd); //στρογγυλοποίηση

      //printf("dd =%lf \n",dd );
      D[j][i]=sqrt(dd); // Array nxm with Distances
      Dtemp[j][i]=D[j][i];
    }
  free(SumX);
  free(SumY);
  free(Drow);

  for (int i = 0; i < m; i++) {
    quickselect(D[i],0,n-1,k);
    //printf("dd=%lf\n",dd );
    if(realloc(D[i],k*sizeof(double))==NULL)
        exit(-1);
    qsort(D[i],k,sizeof(double),comparator);
  }

  /*
    // Print D Array - for tests
    for(int i=0; i<m; i++)
    {
      for(int j=0; j <n;j++)
        printf("%lf ", D[i][j]);
      printf("\n" );
    }
    // Print D Array - End
*/



  knnresult knn;

  knn.nidx=(int *)malloc(m*k*sizeof(int));
  knn.ndist=(double *)malloc(m*k*sizeof(double));

  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++) {
      int r=0;
      while (D[i][j]!=Dtemp[i][r])
        r++;
      knn.nidx[i+j*m]=r;
      knn.ndist[i+j*m]=Dtemp[i][r];
      knn.m=m;
      knn.k=k;
    }

  for (int i = 0; i < m; i++)
  {
    free(D[i]);
    free(Dtemp[i]);
  }
  free(D);
  free(Dtemp);

  return knn;

}




int partition(double *A, int left, int right){
  double pivot = A[right];
  int i = left, x;
  for (x = left; x < right; x++)
    if (A[x] < pivot)
      swap(&A[i++], &A[x]);
  swap(&A[i], &A[right]);
  return i;
}

double quickselect(double *A, int left, int right, int k){
  int p = partition(A, left, right);
  if (p == k-1)
    return A[p];
  if (k - 1 < p)
    return quickselect(A, left, p - 1, k);
  return quickselect(A, p + 1, right, k);
  }

void swap(double *a, double *b){
  double temp = *a;
  *a = *b;
  *b = temp;
}
