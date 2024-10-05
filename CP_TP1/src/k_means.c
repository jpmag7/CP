#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 10000000
#define K 4

typedef struct point{
	float x;
	float y;
	int ncluster;
} *POINT;

typedef struct cluster{
	int size;
	POINT temp_centroid;
	POINT centroid;
} *CLUSTER;

POINT pointList[N];
CLUSTER clusterList[K];

void init(){
	srand(10);
	for (int i=0;i<N;i++){
		pointList[i] = malloc(sizeof(struct point));
		pointList[i]->x = (float) rand() / RAND_MAX;
		pointList[i]->y = (float) rand() / RAND_MAX;
		pointList[i]->ncluster = -1;
	}
	for (int i=0; i < K; i++){
		clusterList[i] = malloc(sizeof(struct cluster));
		clusterList[i]->centroid = malloc(sizeof(struct point));
		clusterList[i]->temp_centroid = malloc(sizeof(struct point));
		clusterList[i]->centroid->x = pointList[i]->x;
		clusterList[i]->centroid->y = pointList[i]->y;
	}
}

int clusterDecide(){
	int l = 0;
	for (int j=0;j<K;j++){
		clusterList[j]->size = 0;
		clusterList[j]->temp_centroid->x = clusterList[j]->temp_centroid->y=0.0;
	}
	for (int i=0;i<N;i++){
		float menorDist = (pointList[i]->x - clusterList[0]->centroid->x)*(pointList[i]->x - clusterList[0]->centroid->x)
		 + (pointList[i]->y - clusterList[0]->centroid->y)*(pointList[i]->y - clusterList[0]->centroid->y);
		int menorCluster = 0;
		for (int j=1;j<K;j++){
			float dist = (pointList[i]->x - clusterList[j]->centroid->x)*(pointList[i]->x - clusterList[j]->centroid->x) 
			+ (pointList[i]->y - clusterList[j]->centroid->y)*(pointList[i]->y - clusterList[j]->centroid->y);
			if (dist<menorDist) {
				menorDist = dist;
				menorCluster = j;

			}
		}
		clusterList[menorCluster]->size++;
		clusterList[menorCluster]->temp_centroid->x += pointList[i]->x;
		clusterList[menorCluster]->temp_centroid->y += pointList[i]->y;
		if (pointList[i]->ncluster != menorCluster){
			l=1;
		}
		pointList[i]->ncluster = menorCluster;
	}

	for (int j=0;j<K;j++){
		clusterList[j]->temp_centroid->x /= clusterList[j]->size;
		clusterList[j]->temp_centroid->y /= clusterList[j]->size;
		clusterList[j]->centroid->x = clusterList[j]->temp_centroid->x;
		clusterList[j]->centroid->y = clusterList[j]->temp_centroid->y;
	}	
	return l;
}

void printInfo(double time_spent, int iteractions){
	printf("N = %d, K = %d\n",N,K);
	for (int i=0;i<K;i++){
		printf("Center: (%.3f, %.3f) : Size : %d\n",clusterList[i]->centroid->x,clusterList[i]->centroid->y,clusterList[i]->size);
	}
	printf("Time Spent: %f seconds.\n",time_spent);
	printf("Iteractions : %d.\n",iteractions);
}

int main(){
	clock_t begin = clock();
	int iteractions=0;
	init();
	while (clusterDecide()) iteractions++;
	clock_t end = clock();
	double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
	printInfo(time_spent, iteractions);
	return 0;
}
