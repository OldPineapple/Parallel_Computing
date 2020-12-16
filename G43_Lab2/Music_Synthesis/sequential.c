#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define rho 0.5
#define eta 0.0002
#define G 0.75
#define N 4

float u1[4][4];
float u2[4][4];
float u[4][4];

int main(int argc, char **argv)
{
	char* t = argv[1]; 
    int T = atoi(t);

    // int T = 5;
	// First hit get u1 for the first step. Make u1[2][2] to 1, others are all 0
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			u1[i][j] = 0;
			u2[i][j] = 0;
		}
	}
	u1[2][2] = 1;

	// Calculate the values for each iteration.
	for(int iter = 0; iter < T; iter++)
	{
		for(int i = 1; i < N-1; i++)
		{
			for(int j = 1; j < N-1; j++)
			{
				u[i][j] = (rho * (u1[i - 1][j] + u1[i + 1][j] + u1[i][j - 1] + u1[i][j + 1] - 4 * u1[i][j]) + 2 * u1[i][j] - (1 - eta) * u2[i][j]) / (1 + eta);
			}
		}
		
		for(int i = 1; i < N-1; i++)
		{
			u[0][i] = G*u[1][i];
			u[N - 1][i] = G*u[N-2][i];
			u[i][0] = G*u[i][1];
			u[i][N - 1] = G*u[i][N-2];
		}

		u[0][0] = G*u[1][0];
		u[N-1][0] = G*u[N-2][0];
		u[0][N-1] = G*u[0][N-2];
		u[N-1][N-1] = G*u[N-1][N-2];

		// Copy the new matrix to u1
		// Update u1 u2
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < N; j++)
			{
				u2[i][j] = u1[i][j];
				u1[i][j] = u[i][j];
			}
		}

		printf("In iteration %d, u[%d][%d] is %f\n", iter, N/2, N/2, u[N/2][N/2]);
	}

	return 0;
}
