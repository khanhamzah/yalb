#ifndef LBM_SOLVER_HPP
#define LBM_SOLVER_HPP

#include <Kokkos_Core.hpp>
#include <string>
#include <mpi.h>  // ✅ MPI support

class LBMSolver {
private:
    // Configuration constants
    static constexpr int GRID_WIDTH = 15;
    static constexpr int GRID_HEIGHT = 10;
    static constexpr int NUM_VELOCITIES = 9;

    // Kokkos view types
    using DistributionView = Kokkos::View<double***>;
    using ScalarField = Kokkos::View<double**>;
    using VelocityMatrix = Kokkos::View<int**>;

    // Member variables
    DistributionView f;
    ScalarField rho;
    ScalarField v_x;
    ScalarField v_y;
    VelocityMatrix c;

    // ✅ MPI info
    int rank_;
    int size_;
    int local_x_start;
    int local_x_end;

public:
    // ✅ Constructor with MPI parameters
    LBMSolver(int rank, int size);

    // Core LBM operations
    void computeDensity();
    void computeVelocityField();
    void initializeDistribution();
    void initializeShearWave(double epsilon);
    void streaming();
    void collision(double omega);
    void applyBoundaryConditions(double lid_velocity);

    // Output and visualization
    void outputFields(const std::string& filename);
    void createVisualizationScript();

    // Simulations
    void testStreaming();
    void runShearWaveDecay(double omega, int steps, int output_interval);
    void runLidDrivenCavity(double omega, double lid_velocity, int steps, int output_interval);

    // ✅ MPI-aware distributed run
    void runDistributed(int steps, double omega, int output_interval);
    // Debug/info
    void printInfo();
    void runParallelSimulation();

private:
    // ✅ Boundary communication
    void exchangeBoundaries();

    
};

#endif
