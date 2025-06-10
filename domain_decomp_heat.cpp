#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <mpi.h>
#include "tnt/tnt.h"

using namespace std;
using namespace TNT;

#define getmax(a,b) ((a)>(b)?(a):(b))

int main(int argc, char* argv[])
{
    int taskid, numtasks, rc;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    
    // Problem parameters
    const int M = 1000, N = 1000;
    const int forb_i[] = {700, 300, 200, 500};
    const int forb_j[] = {500, 400, 800, 100};
    const int niter = 100000;
    const int mod = 10;
    const int numrep = 32;  // Now run sequentially
    const float a = 1.9f;
    const float maxerror = 1e-6f;
    const float aover4 = a / 4.0f;
    const float oneminusa = 1.0f - a;
    const int numsites = sizeof(forb_i) / sizeof(int);
    
    // Domain decomposition setup
    int rows_per_proc = N / numtasks;
    int start_row = taskid * rows_per_proc + 1;
    int end_row = (taskid == numtasks - 1) ? N : (taskid + 1) * rows_per_proc;
    int local_rows = end_row - start_row + 1;
    
    if (taskid == 0) {
        printf("Domain decomposition: %d processes, %d rows per process\n", numtasks, rows_per_proc);
    }
    
    // Initialize random seed
    struct timeval start;
    gettimeofday(&start, NULL);
    unsigned int seed = static_cast<unsigned int>(start.tv_usec + taskid * 1000);
    srand(seed);
    
    // Initialize source term array Q (full size for source placement)
    Array2D<double> Q(N + 2, M + 2, 0.0);
    
    // Each processor runs numrep simulations sequentially
    for (int rep = 0; rep < numrep; rep++) {
        if (taskid == 0) {
            printf("Starting simulation %d/%d\n", rep + 1, numrep);
        }
        
        // Local temperature array (only local rows + ghost rows)
        Array2D<double> T_local(local_rows + 2, M + 2, 0.0);
        
        // Set random source terms (only master process)
        if (taskid == 0) {
            // Reset Q to zero
            for (int i = 0; i <= N + 1; i++) {
                for (int j = 0; j <= M + 1; j++) {
                    Q[i][j] = 0.0;
                }
            }
            
            // Set random values at forbidden sites
            for (int l = 0; l < numsites; l++) {
                double random_val = (rand() % 101) + (rand() % 100001) / 100000.0;
                Q[forb_i[l]][forb_j[l]] = random_val;
            }
        }
        
        // Broadcast Q to all processes
        MPI_Bcast(&Q[0][0], (N + 2) * (M + 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Iterative solver loop
        int n;
        for (n = 1; n < niter + 1; n++) {
            double local_max_change = -10000.0;
            
            // Exchange ghost rows with neighbors
            if (taskid > 0) {
                // Send top row to upper neighbor, receive from upper neighbor
                MPI_Sendrecv(&T_local[1][0], M + 2, MPI_DOUBLE, taskid - 1, 0,
                            &T_local[0][0], M + 2, MPI_DOUBLE, taskid - 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            if (taskid < numtasks - 1) {
                // Send bottom row to lower neighbor, receive from lower neighbor
                MPI_Sendrecv(&T_local[local_rows][0], M + 2, MPI_DOUBLE, taskid + 1, 0,
                            &T_local[local_rows + 1][0], M + 2, MPI_DOUBLE, taskid + 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Update local interior points
            for (int i = 1; i <= local_rows; i++) {
                int global_i = start_row + i - 1;
                for (int j = 1; j <= M; j++) {
                    double newt = oneminusa * T_local[i][j] + 
                                 aover4 * (T_local[i-1][j] + T_local[i+1][j] + 
                                          T_local[i][j+1] + T_local[i][j-1] + Q[global_i][j]);
                    
                    if (n % mod == 0) {
                        local_max_change = getmax(abs(newt - T_local[i][j]), local_max_change);
                    }
                    
                    T_local[i][j] = newt;
                }
            }
            
            // Global convergence check
            if (n % mod == 0) {
                double global_max_change;
                MPI_Allreduce(&local_max_change, &global_max_change, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                
                if (global_max_change < maxerror) {
                    if (taskid == 0) {
                        printf("Simulation %d converged at iteration %d with max_change = %e\n", 
                               rep + 1, n, global_max_change);
                    }
                    break;
                }
            }
            
            if (taskid == 0 && n % 1000 == 0) {
                printf("Simulation %d: iteration %d\n", rep + 1, n);
            }
        }
        
        // Gather results to master process
        Array2D<double> T_global;
        if (taskid == 0) {
            T_global = Array2D<double>(N + 2, M + 2, 0.0);
        }
        
        // Collect results from all processes
        for (int proc = 0; proc < numtasks; proc++) {
            int proc_start = proc * rows_per_proc + 1;
            int proc_end = (proc == numtasks - 1) ? N : (proc + 1) * rows_per_proc;
            int proc_rows = proc_end - proc_start + 1;
            
            if (taskid == proc) {
                // Send local data to master
                if (proc == 0) {
                    // Master copies its own data
                    for (int i = 1; i <= local_rows; i++) {
                        for (int j = 0; j <= M + 1; j++) {
                            T_global[start_row + i - 1][j] = T_local[i][j];
                        }
                    }
                } else {
                    // Send to master
                    for (int i = 1; i <= local_rows; i++) {
                        MPI_Send(&T_local[i][0], M + 2, MPI_DOUBLE, 0, proc, MPI_COMM_WORLD);
                    }
                }
            } else if (taskid == 0) {
                // Master receives from other processes
                for (int i = 0; i < proc_rows; i++) {
                    MPI_Recv(&T_global[proc_start + i][0], M + 2, MPI_DOUBLE, proc, proc, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        
        // Master writes results
        if (taskid == 0) {
            stringstream fname;
            fname << "simulation_" << rep + 1 << ".txt";
            ofstream fs(fname.str().c_str());
            
            // Write parameters
            fs << "Simulation " << rep + 1 << " " << M << " " << N << " ";
            for (int l = 0; l < numsites; l++) {
                fs << forb_i[l] << " " << forb_j[l] << " " << Q[forb_i[l]][forb_j[l]] << " ";
            }
            fs << endl;
            
            if (n >= niter) {
                fs << "Completed maximum iterations (" << niter << ") without convergence" << endl;
            }
            
            // Write temperature field
            for (int i = 0; i <= N + 1; i++) {
                for (int j = 0; j <= M + 1; j++) {
                    fs << T_global[i][j] << " ";
                }
                fs << endl;
            }
            fs.close();
            
            printf("Simulation %d complete, saved to %s\n", rep + 1, fname.str().c_str());
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    if (taskid == 0) {
        printf("All %d simulations completed!\n", numrep);
    }
    return 0;
}