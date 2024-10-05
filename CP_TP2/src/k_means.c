#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <alloca.h>


int N, K, W = 1;

//int *point_ncluster;
float *point_x;
float *point_y;

int *cluster_size;
float *cluster_centroid_x;
float *cluster_centroid_y;
float *cluster_temp_centroid_x;
float *cluster_temp_centroid_y;

int *mcls;


void init(){


	mcls                    = (int *)   malloc(sizeof(int)   * N);
	//point_ncluster          = (int *)   malloc(sizeof(int)   * N);
	point_x                 = (float *) malloc(sizeof(float) * N);
	point_y                 = (float *) malloc(sizeof(float) * N);
	cluster_size            = (int *)   malloc(sizeof(int)   * K);
	cluster_centroid_x      = (float *) malloc(sizeof(float) * K);
	cluster_centroid_y      = (float *) malloc(sizeof(float) * K);
	cluster_temp_centroid_x = (float *) malloc(sizeof(float) * K);
	cluster_temp_centroid_y = (float *) malloc(sizeof(float) * K);

	srand(10);
	for (int i=0; i<N; i++){
		point_x[i] = (float) rand() / RAND_MAX; 
		point_y[i] = (float) rand() / RAND_MAX;
		//point_x[i+1] = (float) rand() / RAND_MAX; 
		//point_y[i+1] = (float) rand() / RAND_MAX; 
		//point_x[i+2] = (float) rand() / RAND_MAX; 
		//point_y[i+2] = (float) rand() / RAND_MAX; 
		//point_x[i+3] = (float) rand() / RAND_MAX; 
		//point_y[i+3] = (float) rand() / RAND_MAX; 
		//point_ncluster[i] = -1; 
	}

	for (int i=0; i < K; i++){
		cluster_size[i] = 0; 
		cluster_centroid_x[i] = point_x[i];
		cluster_centroid_y[i] = point_y[i];
		cluster_temp_centroid_x[i] = 0; 
		cluster_temp_centroid_y[i] = 0; 
	}
}



void clusterDecide(){

	for (int j = 0; j < K; j++){
		cluster_size[j] = 0;
		cluster_temp_centroid_x[j] = cluster_temp_centroid_y[j] = 0.0;
	}

	#pragma omp parallel for num_threads(W)
	for (int i=0; i < N; i+=4){

		float menorDist1 = 100000000;
		float menorDist2 = 100000000;
		float menorDist3 = 100000000;
		float menorDist4 = 100000000;
		float menorCluster1 = -1;
		float menorCluster2 = -1;
		float menorCluster3 = -1;
		float menorCluster4 = -1;
		float x1 = point_x[i];
		float x2 = point_x[i+1];
		float x3 = point_x[i+2];
		float x4 = point_x[i+3];
		float y1 = point_y[i];
		float y2 = point_y[i+1];
		float y3 = point_y[i+2];
		float y4 = point_y[i+3];

		for (int j=0; j<K; j++){

			float clx = cluster_centroid_x[j];
			float cly = cluster_centroid_y[j];

			float xx1 = x1 - clx;
			float xx2 = x2 - clx;
			float xx3 = x3 - clx;
			float xx4 = x4 - clx;
			float yy1 = y1 - cly;
			float yy2 = y2 - cly;
			float yy3 = y3 - cly;
			float yy4 = y4 - cly;
			float dist1 = xx1 * xx1 + yy1 * yy1;
			float dist2 = xx2 * xx2 + yy2 * yy2;
			float dist3 = xx3 * xx3 + yy3 * yy3;
			float dist4 = xx4 * xx4 + yy4 * yy4;

			menorCluster1 = dist1 < menorDist1 ? j : menorCluster1;
			menorCluster2 = dist2 < menorDist2 ? j : menorCluster2;
			menorCluster3 = dist3 < menorDist3 ? j : menorCluster3;
			menorCluster4 = dist4 < menorDist4 ? j : menorCluster4;
			menorDist1    = dist1 < menorDist1 ? dist1 : menorDist1;
			menorDist2    = dist2 < menorDist2 ? dist2 : menorDist2;
			menorDist3    = dist3 < menorDist3 ? dist3 : menorDist3;
			menorDist4    = dist4 < menorDist4 ? dist4 : menorDist4;
		}
		
		mcls[i] = menorCluster1;
		mcls[i+1] = menorCluster2;
		mcls[i+2] = menorCluster3;
		mcls[i+3] = menorCluster4;

	}

	for (int i = 0; i < N; i+=4){
		int menorCluster1 = mcls[i];
		int menorCluster2 = mcls[i+1];
		int menorCluster3 = mcls[i+2];
		int menorCluster4 = mcls[i+3];
		cluster_size[menorCluster1]++;
		cluster_size[menorCluster2]++;
		cluster_size[menorCluster3]++;
		cluster_size[menorCluster4]++;
		cluster_temp_centroid_x[menorCluster1] += point_x[i];
		cluster_temp_centroid_x[menorCluster2] += point_x[i+1];
		cluster_temp_centroid_x[menorCluster3] += point_x[i+2];
		cluster_temp_centroid_x[menorCluster4] += point_x[i+3];
		cluster_temp_centroid_y[menorCluster1] += point_y[i];
		cluster_temp_centroid_y[menorCluster2] += point_y[i+1];
		cluster_temp_centroid_y[menorCluster3] += point_y[i+2];
		cluster_temp_centroid_y[menorCluster4] += point_y[i+3];
		
		//if (point_ncluster[i] != menorCluster){
		//	l=1;
		//}
		//point_ncluster[i] = menorCluster;
	}


	for (int j=0; j<K; j++){
		cluster_temp_centroid_x[j] /= cluster_size[j];
		cluster_temp_centroid_y[j] /= cluster_size[j];
		cluster_centroid_x[j] = cluster_temp_centroid_x[j];
		cluster_centroid_y[j] = cluster_temp_centroid_y[j];
	}
}



void printInfo(double time_spent, int iteractions){
	printf("N = %d, K = %d\n",N,K);
	for (int i=0;i<K;i++){
		printf("Center: (%.3f, %.3f) : Size : %d\n", cluster_centroid_x[i], cluster_centroid_y[i], cluster_size[i]);
	}
	printf("Time Spent: %f seconds.\n",time_spent);
	printf("Iteractions : %d.\n",iteractions);
}



int main(int argc, char **argv){

	if(argc < 3){
		printf("Invalid Arguments");
		return 1;
	}
	
	N = atoi(argv[1]);
	K = atoi(argv[2]);
	if(argc > 3)
		W = atoi(argv[3]);

	double itime, ftime, exec_time;
    itime = omp_get_wtime();
    int it = 0;

	init();
	
	while(it < 20){
		clusterDecide();
		it++;
	}
	
	ftime = omp_get_wtime();
    exec_time = ftime - itime;
	printInfo(exec_time, 20);
	
	return 0;
}