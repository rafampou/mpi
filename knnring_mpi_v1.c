/*
//
//  1/12/19 - 2η Εργασία Παράλληλα
//  Rafael Boulogeorgos - 9186
//
//
//
*/

//Define tests. set tests=1 to print files for test in Matlab
#define  tests 0

#include "knnring.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <sys/time.h>

/*
//-----------------------------------------------------------------------------
//  distrAllkNN function - START
//
//
*/
knnresult distrAllkNN(double * X,int n, int d, int k)
{
  //  mesure time
  struct timeval start, end, startTRNS, endTRNS;
  double time_trns=0;
  double time_job=0;;

  //  Start Job time
  gettimeofday(&start, NULL);

  //  numtasks   ->  Number of MPI tasks
  //  rank       ->  The id number of this tasks
  //  tag        ->  Set tag=2 for send and receve between tasks
  int numtasks, rank, tag=2;
  MPI_Status Stat;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  //  idPolarization    ->  Offest for ids for each group of corpus
  int idPolarization=(rank+numtasks-1)%numtasks*n;;
  //  numOfIterate      ->  Number of the times that we need to iterate to all points checked
  int numOfIterate=numtasks-1;

  //  2 Buffers for receve and Send
  double *yRecv=(double *)malloc(n*d*sizeof(double));
  double *ySent=(double *)malloc(n*d*sizeof(double));

  // First Recv is our Query
  for (int i = 0; i < n*d; i++)
  yRecv[i]=X[i];


  //printf array for Matlab tests
  #if tests
  char path[10];
  snprintf(path,sizeof(path),"Task%d.txt",rank);
  FILE *fp=fopen(path,"w+");
  fprintf(fp, "\n\nTask in i=%d resv \n X=[",rank);
  for (int i = 0; i < n; i++) {
    fprintf(fp, "\n" );
    //fprintf(fp,"idx=%d ",myKnn.nidx[i] );
    for (int j = 0; j < d; j++)
    fprintf(fp, "%lf ",yRecv[i+j*n]);
    if (i==n-1)  fprintf(fp, "]; \n");
    else  fprintf(fp, ";");
  }
  #endif

  //  myKnn is the kNN for each loop
  knnresult myKnn = kNN(yRecv, X, n, n, d, k);

  //printf array for Matlab tests
  #if tests
  fprintf(fp, "\n\nTask in i=%d knn \n knn=[",rank);
  for (int i = 0; i < n; i++) {
    fprintf(fp, "\n" );
    //fprintf(fp,"idx=%d ",myKnn.nidx[i] );
    for (int j = 0; j < d; j++)
    fprintf(fp, "%lf ",myKnn.ndist[i+j*n]);
    if (i==n-1)  fprintf(fp, "]; \n");
    else  fprintf(fp, ";");
  }
  #endif


  //  Set the Offset for ids of knn
  for (int i = 0; i < n*k; i++)
  myKnn.nidx[i]+=idPolarization;


  //  loop to iterate all points
  for (int l = 0; l < numOfIterate; l++) {

    //  Sent the files that Receve before
    for (int i = 0; i < n*d; i++)
    ySent[i]=yRecv[i];

    // end time job
    gettimeofday(&startTRNS, NULL);

    //  Synchronous Reseve from Even Tasks and Send from Odd Tasks
    if (rank%2) // Odd Tasks
    {
      MPI_Send(ySent, n*d, MPI_DOUBLE, (rank+1)%numtasks , tag , MPI_COMM_WORLD);
      MPI_Recv(yRecv, n*d, MPI_DOUBLE, (rank+numtasks-1)%numtasks, tag, MPI_COMM_WORLD, &Stat);
    }
    else  // Even Tasks
    {
      MPI_Recv(yRecv, n*d, MPI_DOUBLE, (rank+numtasks-1)%numtasks, tag, MPI_COMM_WORLD, &Stat);
      MPI_Send(ySent, n*d, MPI_DOUBLE, (rank+1)%numtasks , tag , MPI_COMM_WORLD);
    }

    // end time job
    gettimeofday(&endTRNS, NULL);
    time_trns += (endTRNS.tv_sec - startTRNS.tv_sec) * 1e6;

    //printf array for Matlab tests
    #if tests
    fprintf(fp, "\n\nTask in i=%d resv \nYrecv=[",l);
    for (int i = 0; i < n; i++) {
      fprintf(fp, "\n" );
      //fprintf(fp,"idx=%d ",myKnn.nidx[i] );
      for (int j = 0; j < d; j++)
      fprintf(fp, "%lf ",yRecv[i+j*n]);
      if (i==n-1)  fprintf(fp, "]; \n");
      else  fprintf(fp, ";");
    }
    #endif

    knnresult newKnn = kNN(yRecv, X, n, n, d, k);

    //printf array for Matlab tests
    #if tests
    fprintf(fp, "\n knn before pol in i=%d knn \n newknn=[",rank);
    for (int i = 0; i < n; i++) {
      fprintf(fp, "\n" );
      for (int j = 0; j < k; j++)
      {
        fprintf(fp,"idx=%d ",newKnn.nidx[i+j*n] );
        fprintf(fp, "%lf ",newKnn.ndist[i+j*n]);
      }
      if (i==n-1)  fprintf(fp, "]; \n");
      else  fprintf(fp, ";");
    }
    #endif

    //  Set new Offset for new corpus
    idPolarization=(rank+numtasks-l-2)%numtasks*n;
    for (int i = 0; i < n*k; i++)
      newKnn.nidx[i]+=idPolarization;

    //printf array for Matlab tests
    #if tests
    fprintf(fp, "\n knn after pol in i=%d knn \n newknn=[",rank);
    for (int i = 0; i < n; i++) {
      fprintf(fp, "\n" );
      for (int j = 0; j < k; j++)
      {
        fprintf(fp,"idx=%d ",newKnn.nidx[i+j*n] );
        fprintf(fp, "%lf ",newKnn.ndist[i+j*n]);
      }
      if (i==n-1)  fprintf(fp, "]; \n");
      else  fprintf(fp, ";");
    }
    #endif

    // This check is only if tasks has deferent number of k.
    int minK= (newKnn.k - myKnn.k > 0) ? (myKnn.k) : (newKnn.k); // maxK <= K
    if(minK != k)
    {
      printf("The K=%d and minK=%d, There is a problem with K\n",k,minK);
      exit(-1);
    }

    //--------------------------------------------------------------------------
    //  Merge Sort for old knn and new knn - START
    //
    //Temp Arrays for merge sort
    double *tempDist=(double *)malloc(n*k*sizeof(double));
    int *tempIdx=(int *)malloc(n*k*sizeof(int));
    // merge oll n queries
    for (int i = 0; i < n; i++) {
      int f=0,s=0;
      // merge each knn
      for (int j = 0; j < k; j++) {
        if(newKnn.ndist[i+f*n] < myKnn.ndist[i+s*n]){
          tempDist[i+j*n]=newKnn.ndist[i+f*n];
          tempIdx[i+j*n]=newKnn.nidx[i+f*n];
          f++;
        }
        else {
          tempDist[i+j*n]=myKnn.ndist[i+s*n];
          tempIdx[i+j*n]=myKnn.nidx[i+s*n];
          s++;
        }
      }
      //  Set up myKnn for next loop
      for (int j = 0; j < k; j++) {
        myKnn.ndist[i+j*n]=tempDist[i+j*n];
        myKnn.nidx[i+j*n]=tempIdx[i+j*n];
      }
    }
    free(tempDist);
    free(tempIdx);
    //  Merge Sort for old knn and new knn - END
    //--------------------------------------------------------------------------
    //

  } // Loop to iterate all points


  free(yRecv);
  free(ySent);


  //printf array for Matlab tests
  #if tests
  fprintf(fp, "myKnn.ndist=[");
  for (int i = 0; i < n; i++) {
    fprintf(fp, "\n" );
    for (int j = 0; j < k; j++)
    {
      fprintf(fp,"idx=%d ",myKnn.nidx[i+j*n] );
      fprintf(fp, "%lf ",myKnn.ndist[i+j*n]);
    }
    if (i==n-1)  fprintf(fp, "] \n");
    else  fprintf(fp, ";");
  }
  fclose(fp);
  #endif

  // end time job
  gettimeofday(&end, NULL);


  time_job = (end.tv_sec - start.tv_sec) * 1e6;
  time_job = (time_job + (end.tv_usec - start.tv_usec)) * 1e-6;

  time_trns = (time_trns + (endTRNS.tv_usec - startTRNS.tv_usec)) * 1e-6;
  printf("Task %d Time of job= %lf and time of Tranfer =%lf\n",rank,time_job,time_trns );

  return myKnn;
} //end distrAllkNN
