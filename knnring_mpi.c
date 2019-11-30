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
  MPI_Request reqs[2];
  MPI_Status stats[2];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  //  idPolarization    ->  Offest for ids for each group of corpus
  int idPolarization=(rank+numtasks-1)%numtasks*n;;
  //  numOfIterate      ->  Number of the times that we need to iterate to all points checked
  int numOfIterate=numtasks-1;

  //  2 Buffers for receve and Send
  double *yRecv=(double *)malloc(n*d*sizeof(double));
  double *ySent=(double *)malloc(n*d*sizeof(double));
  double *resvBuff=(double *)malloc(n*d*sizeof(double));
  // First Recv is our Query
  for (int i = 0; i < n*d; i++)
  yRecv[i]=X[i];

  for (int i = 0; i < n*d; i++)
  ySent[i]=yRecv[i];

  MPI_Isend(ySent, n*d, MPI_DOUBLE, (rank+1)%numtasks , tag , MPI_COMM_WORLD,&reqs[0]);
  MPI_Irecv(resvBuff, n*d, MPI_DOUBLE, (rank+numtasks-1)%numtasks, tag, MPI_COMM_WORLD, &reqs[1]);


  //  myKnn is the kNN for each loop
  knnresult myKnn = kNN(yRecv, X, n, n, d, k);
  //  Set the Offset for ids of knn
  for (int i = 0; i < n*k; i++)
  myKnn.nidx[i]+=idPolarization;

  // end time job
  gettimeofday(&startTRNS, NULL);

  MPI_Waitall(2, reqs, stats);

  // end time job
  gettimeofday(&endTRNS, NULL);
  time_trns += (endTRNS.tv_sec - startTRNS.tv_sec) * 1e6;

  //  loop to iterate all points
  for (int l = 0; l < numOfIterate; l++) {

    //  Sent the files that Receve before
    for (int i = 0; i < n*d; i++)
    {
      ySent[i]=resvBuff[i];
      yRecv[i]=resvBuff[i];
    }

    MPI_Isend(ySent, n*d, MPI_DOUBLE, (rank+1)%numtasks , tag , MPI_COMM_WORLD,&reqs[0]);
    MPI_Irecv(resvBuff, n*d, MPI_DOUBLE, (rank+numtasks-1)%numtasks, tag, MPI_COMM_WORLD, &reqs[1]);

    knnresult newKnn = kNN(yRecv, X, n, n, d, k);

    //  Set new Offset for new corpus
    idPolarization=(rank+numtasks-l-2)%numtasks*n;
    for (int i = 0; i < n*k; i++)
      newKnn.nidx[i]+=idPolarization;

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

    // end time job
    gettimeofday(&startTRNS, NULL);

    MPI_Waitall(2, reqs, stats);

    // end time job
    gettimeofday(&endTRNS, NULL);
    time_trns += (endTRNS.tv_sec - startTRNS.tv_sec) * 1e6;
  } // Loop to iterate all points


  free(yRecv);
  free(ySent);
  free(resvBuff);



  // end time job
  gettimeofday(&end, NULL);


  time_job = (end.tv_sec - start.tv_sec) * 1e6;
  time_job = (time_job + (end.tv_usec - start.tv_usec)) * 1e-6;

  time_trns = (time_trns + (endTRNS.tv_usec - startTRNS.tv_usec)) * 1e-6;
  printf("Task %d Time of job= %lf and time of Tranfer =%lf\n",rank,time_job,time_trns );
  return myKnn;
} //end distrAllkNN
