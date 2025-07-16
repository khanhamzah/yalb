#include "lbm_solver.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>

LBMSolver::LBMSolver(int rank, int size)
    : rank_(rank), size_(size)
{
    // Corrected domain splitting for non-divisible widths
    int base = GRID_WIDTH / size_;
    int extra = GRID_WIDTH % size_;
    local_x_start = rank * base + std::min(rank, extra);
    local_x_end = local_x_start + base + (rank < extra ? 1 : 0);

    f = DistributionView("f", GRID_WIDTH, GRID_HEIGHT, NUM_VELOCITIES);
    rho = ScalarField("rho", GRID_WIDTH, GRID_HEIGHT);
    v_x = ScalarField("vx", GRID_WIDTH, GRID_HEIGHT);
    v_y = ScalarField("vy", GRID_WIDTH, GRID_HEIGHT);
    c = VelocityMatrix("c", 2, NUM_VELOCITIES);

    auto h_c = Kokkos::create_mirror_view(c);
    h_c(0, 0) = 0;  h_c(0, 1) = 1;  h_c(0, 2) = 0;  h_c(0, 3) = -1; h_c(0, 4) = 0;
    h_c(0, 5) = 1;  h_c(0, 6) = -1; h_c(0, 7) = -1; h_c(0, 8) = 1;
    h_c(1, 0) = 0;  h_c(1, 1) = 0;  h_c(1, 2) = 1;  h_c(1, 3) = 0;  h_c(1, 4) = -1;
    h_c(1, 5) = 1;  h_c(1, 6) = 1;  h_c(1, 7) = -1; h_c(1, 8) = -1;
    Kokkos::deep_copy(c, h_c);
}

void LBMSolver::initializeDistribution() {
    const double u0x = 0.05;
    const double u0y = 0.0;
    const double rho0 = 1.0;
    const double w[9] = {
        4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    };

    Kokkos::parallel_for("initialize", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {local_x_start, 0, 0}, {local_x_end, GRID_HEIGHT, NUM_VELOCITIES}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            double cu = 3.0 * (c(0, i) * u0x + c(1, i) * u0y);
            double u2 = u0x * u0x + u0y * u0y;
            f(x, y, i) = w[i] * rho0 * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
        });

    Kokkos::deep_copy(rho, rho0);
    Kokkos::deep_copy(v_x, u0x);
    Kokkos::deep_copy(v_y, u0y);
}


void LBMSolver::streaming() {
    DistributionView f_new("f_new", GRID_WIDTH, GRID_HEIGHT, NUM_VELOCITIES);

    Kokkos::parallel_for("streaming", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {local_x_start, 0, 0}, {local_x_end, GRID_HEIGHT, NUM_VELOCITIES}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            int sx = (x - c(0, i) + GRID_WIDTH) % GRID_WIDTH;
            int sy = (y - c(1, i) + GRID_HEIGHT) % GRID_HEIGHT;
            f_new(x, y, i) = f(sx, sy, i);
        });

    Kokkos::deep_copy(f, f_new);
}

void LBMSolver::computeDensity() {
    Kokkos::parallel_for("density", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {local_x_start, 0}, {local_x_end, GRID_HEIGHT}),
        KOKKOS_LAMBDA(int x, int y) {
            double d = 0.0;
            for (int i = 0; i < NUM_VELOCITIES; ++i) d += f(x, y, i);
            rho(x, y) = d;
        });
}

void LBMSolver::computeVelocityField() {
    Kokkos::parallel_for("velocity", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {local_x_start, 0}, {local_x_end, GRID_HEIGHT}),
        KOKKOS_LAMBDA(int x, int y) {
            double mx = 0.0, my = 0.0;
            double d = rho(x, y);
            for (int i = 0; i < NUM_VELOCITIES; ++i) {
                mx += f(x, y, i) * c(0, i);
                my += f(x, y, i) * c(1, i);
            }
            v_x(x, y) = (d > 1e-12) ? mx / d : 0.0;
            v_y(x, y) = (d > 1e-12) ? my / d : 0.0;
        });
}

void LBMSolver::collision(double omega) {
    const double w[9] = {
        4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    };

    DistributionView f_post("f_post", GRID_WIDTH, GRID_HEIGHT, NUM_VELOCITIES);
    Kokkos::parallel_for("collision", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {local_x_start, 0}, {local_x_end, GRID_HEIGHT}),
        KOKKOS_LAMBDA(int x, int y) {
            double rho_l = rho(x, y), ux = v_x(x, y), uy = v_y(x, y);
            double u2 = ux * ux + uy * uy;
            for (int i = 0; i < NUM_VELOCITIES; ++i) {
                double cu = c(0, i) * ux + c(1, i) * uy;
                double feq = w[i] * rho_l * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
                f_post(x, y, i) = f(x, y, i) + omega * (feq - f(x, y, i));
            }
        });
    Kokkos::deep_copy(f, f_post);
}

void LBMSolver::outputFields(const std::string& filename_prefix) {
    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_vx = Kokkos::create_mirror_view(v_x);
    auto h_vy = Kokkos::create_mirror_view(v_y);
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_vx, v_x);
    Kokkos::deep_copy(h_vy, v_y);

    std::ostringstream fname;
    fname << filename_prefix << "_rank" << rank_ << ".dat";
    std::ofstream out(fname.str());

    for (int x = local_x_start; x < local_x_end; ++x) {
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            out << x << " " << y << " " << h_rho(x, y) << " " << h_vx(x, y) << " " << h_vy(x, y) << "\n";
        }
    }

    std::cout << "Wrote local results to " << fname.str() << std::endl;
}

void LBMSolver::runDistributed(int steps, double omega, int output_interval) {
    initializeDistribution();
    for (int t = 0; t < steps; ++t) {
        streaming();
        computeDensity();
        computeVelocityField();
        collision(omega);
        if (t % output_interval == 0) {
            outputFields("mpi_output_t" + std::to_string(t));
        }
    }
}

void LBMSolver::printInfo() {
    std::cout << "[Rank " << rank_ << "] Domain: x = " << local_x_start
              << " to " << local_x_end - 1 << ", Total ranks = " << size_ << std::endl;
}

void LBMSolver::runParallelSimulation() {
    const int steps = 100;
    const double omega = 1.0;
    const int output_interval = 20;

    runDistributed(steps, omega, output_interval);
}
