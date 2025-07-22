#include "lbm_solver.hpp"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize();

    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        LBMSolver solver(rank, size);  // Pass MPI info
        solver.printInfo();
        solver.runShearWaveDecay(1.0, 200, 20);

    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
