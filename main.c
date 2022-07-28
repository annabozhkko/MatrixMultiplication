#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define M 3000
#define N 3000
#define K 3000

#define P0 4
#define P1 4

void multMatrix(double *A, double *B, double *C){
    for (int i = 0; i < N / P0; ++i){
        for (int j = 0; j < N; ++j){
            for (int k = 0; k < K / P1; ++k){
                C[i * (M / P0) + k] += A[i * N + j] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]){
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 1;
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm2d;
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    double *A, *B, *C;
    if (rank == 0){
        A = malloc(M * N * sizeof(double));
        B = malloc(N * K * sizeof(double));
        C = malloc(M * K * sizeof(double));
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A[i * N + j] = 1;
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                B[j * K + k] = 1;
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                C[i * M + k] = 0;
    }

    double timeStart = MPI_Wtime();

    MPI_Comm comm1D[2]; 
    int remains[2] = {0, 1};
	MPI_Cart_sub(comm2d, remains, &comm1D[1]);
    remains[0] = 1;
	remains[1] = 0;
	MPI_Cart_sub(comm2d, remains, &comm1D[0]);

    double *AA, *BB, *CC;   
    int partSizeM = N / P0;
    int partSizeN = N;      
    int partSizeK = K / P1; 

    AA = malloc(partSizeM * partSizeN * sizeof(double));
    BB = malloc(partSizeN * partSizeK * sizeof(double));
    CC = calloc(partSizeK * partSizeM, sizeof(double));

    MPI_Datatype col;
    MPI_Type_vector(N, 1, N, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Datatype colType;
    MPI_Type_create_resized(col, 0, sizeof(double), &colType);
    MPI_Type_commit(&colType);

    if (coords[1] == 0) 
        MPI_Scatter(A, N * partSizeM, MPI_DOUBLE, AA, N * partSizeM, MPI_DOUBLE, 0, comm1D[0]);  //строки

    if (coords[0] == 0) 
        MPI_Scatter(B, partSizeK, colType, BB, partSizeK * N, MPI_DOUBLE, 0, comm1D[1]);  //столбцы

    MPI_Bcast(AA, partSizeM * partSizeN, MPI_DOUBLE, 0, comm1D[1]); 
    MPI_Bcast(BB, partSizeK * partSizeN, MPI_DOUBLE, 0, comm1D[0]); 

    multMatrix(AA, BB, CC);

    MPI_Datatype types;
    MPI_Type_vector(partSizeM, partSizeK, K, MPI_DOUBLE, &types); 

    MPI_Datatype typec;
    MPI_Type_commit(&types);
    MPI_Type_create_resized(types, 0, partSizeK * sizeof(double), &typec);  //квадратик
    MPI_Type_commit(&typec);

    int *countc = malloc(P0 * P1 * sizeof(int));
    int *dispc = malloc(P0 * P1 * sizeof(int));
    for(int i = 0; i < P0; ++i){
        for(int j = 0; j < P1; ++j){
            countc[i * P0 + j] = 1;
            dispc[i * P0 + j] = i * P0 * partSizeK + j;
        }
    }

    MPI_Gatherv(CC, partSizeK * partSizeM, MPI_DOUBLE, C, countc, dispc, typec, 0, comm2d);

    double timeFinish = MPI_Wtime();
    //if(rank == 0)
       // printf("%f\n", timeFinish - timeStart);
    
    if (rank == 0){
        for (int i = 0; i < M; ++i){
            for (int j = 0; j < K; ++j){
                printf("%lf ", C[i * K + j]);
            }
            printf("\n");
        }
    }

    if(rank == 0){
        free(A);
        free(B);
        free(C);
    }

    free(AA);
    free(BB);
    free(CC);

    MPI_Comm_free(&comm2d);
    for (int i = 0; i < 2; i++){
        MPI_Comm_free(&comm1D[i]);
    }

    MPI_Type_free(&colType);
    MPI_Type_free(&col);
    MPI_Type_free(&typec);
    MPI_Finalize();
    return 0;
}
