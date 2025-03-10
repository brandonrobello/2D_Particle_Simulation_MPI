// mpi.cpp
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

// Precomputed constants
static const double cutoff2 = cutoff * cutoff;
static const double min_r2 = min_r * min_r;

// Define bin size
static const double bin_size = 1.2 * cutoff;

// Bin struct to hold particles
struct Bin {
    std::vector<particle_t*> particles;
};

// 2D grid of bins
static std::vector<std::vector<Bin>> bins;
static std::vector<std::vector<Bin>>* bins_ptr;
int start_bin, end_bin, bin_count_x, bin_count_y;
double y_min, y_max, y_min_halo, y_max_halo;

// Vector of particles in this process's local memory
static std::vector<particle_t> local_particles;

// Apply the force from neighbor to particle
inline void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff2)
        return;

    r2 = fmax(r2, min_r2);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
inline void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    if (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    if (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Compute bin index for a given position
inline void get_bin_index(double x, double y, int& bx, int& by, double size) {
    // Compute bin indices
    bx = static_cast<int>(x / bin_size);
    by = static_cast<int>(y / bin_size) - start_bin;

    // Ensure bin indices stay within valid range
    bx = std::max(0, std::min(bx, bin_count_x - 1));
    by = std::max(0, std::min(by, bin_count_y - 1));
}

// Apply force with binning for all neighbors of a particle
inline void apply_force_binning(particle_t& particle, Bin& bin) {
    for (auto neighbor : bin.particles) {
        if (neighbor->id != particle.id) {
            apply_force(particle, *neighbor);
        }
    }
}

// Optimized force calculation considering only necessary bins
void compute_forces() {
    for (int i = 0; i < bin_count_x; ++i) {
        for (int j = 0; j < bin_count_y; ++j) {
            Bin& bin = (*bins_ptr)[i][j];

            for (auto p : bin.particles) {
                // Reset acceleration before force accumulation
                p->ax = p->ay = 0;
                // Check self-bin and neighboring bins within cutoff
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int ni = i + dx;
                        int nj = j + dy;
                        if (ni >= 0 && ni < bin_count_x && nj >= 0 && nj < bin_count_y) {
                            apply_force_binning(*p, (*bins_ptr)[ni][nj]);
                        }
                    }
                }
            }
        }
    }
}

// Build the bins with local particles
void build_bins(double size) {
    // Clear the bins
    for (int i = 0; i < bin_count_x; ++i) {
        for (int j = 0; j < bin_count_y; ++j) {
            (*bins_ptr)[i][j].particles.clear();
        }
    }
    
    // Insert particle pointers into bins
    for (size_t i = 0; i < local_particles.size(); ++i) {
        int bx, by;
        get_bin_index(local_particles[i].x, local_particles[i].y, bx, by, size);
        (*bins_ptr)[bx][by].particles.push_back(&local_particles[i]);
    }
}

void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    // Compute the domain and halo region boundaries for this process
    y_min = size * rank / num_procs;
    y_max = size * (rank + 1) / num_procs;
    y_min_halo = (y_min - cutoff < 0) ? 0 : y_min - cutoff;
    y_max_halo = (y_max + cutoff > size) ? size : y_max + cutoff;
    
    // Copy owned and ghost particles into this process's local memory
    local_particles.clear();
    for (int i = 0; i < num_parts; ++i) {
        if (parts[i].y >= y_min_halo && parts[i].y <= y_max_halo)
            local_particles.push_back(parts[i]);
    }
    
    // Initialize and build the bins
    bin_count_x = static_cast<int>(size / bin_size);
    start_bin = static_cast<int>(std::floor(y_min_halo / bin_size)); // First bin that overlaps
    end_bin = static_cast<int>(std::ceil(y_max_halo / bin_size)); // First bin that does not overlap
    bin_count_y = end_bin - start_bin;

    bins.assign(bin_count_x, std::vector<Bin>(bin_count_y));
    bins_ptr = &bins;

    build_bins(size);
}

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    // Compute forces using optimized binning
    compute_forces();

    // Move owned particles and partition into remaining and outgoing
    std::vector<particle_t> remain, send_up, send_down;
    for (size_t i = 0; i < local_particles.size(); ++i) {
        double y = local_particles[i].y;

        // Ownership check
        if ((rank == num_procs - 1 && y >= y_min && y <= y_max) ||
            (rank != num_procs - 1 && y >= y_min && y < y_max)) {
            move(local_particles[i], size);
            y = local_particles[i].y;

            // Keep owned particles and owned particles that move into halo regions
            if (y >= y_min_halo && y <= y_max_halo)
                remain.push_back(local_particles[i]);
            
            // Send owned particles that move into a neighbor's domain or halo region
            if (y >= y_max - cutoff && rank < num_procs - 1)
                send_up.push_back(local_particles[i]);
            if (y <= y_min + cutoff && rank > 0)
                send_down.push_back(local_particles[i]);
        }
    }

    // Exchange particles with immediate neighbors
    std::vector<particle_t> recv_from_lower, recv_from_upper;

    // Upward exchange:
    // All processors except the top-most send their upward-moving particles
    if (rank != num_procs - 1) {
        int count = send_up.size();
        MPI_Send(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        if (count > 0) {
            MPI_Send(send_up.data(), count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD);
        }
    }
    // All processors except the bottom-most receive from below
    if (rank != 0) {
        int count;
        MPI_Recv(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (count > 0) {
            recv_from_lower.resize(count);
            MPI_Recv(recv_from_lower.data(), count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Downward exchange:
    // All processors except the bottom-most send their downward-moving particles
    if (rank != 0) {
        int count = send_down.size();
        MPI_Send(&count, 1, MPI_INT, rank - 1, 2, MPI_COMM_WORLD);
        if (count > 0) {
            MPI_Send(send_down.data(), count, PARTICLE, rank - 1, 3, MPI_COMM_WORLD);
        }
    }
    // All processors except the top-most receive from above
    if (rank != num_procs - 1) {
        int count;
        MPI_Recv(&count, 1, MPI_INT, rank + 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (count > 0) {
            recv_from_upper.resize(count);
            MPI_Recv(recv_from_upper.data(), count, PARTICLE, rank + 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Update local particles
    local_particles = remain;
    local_particles.insert(local_particles.end(), recv_from_lower.begin(), recv_from_lower.end());
    local_particles.insert(local_particles.end(), recv_from_upper.begin(), recv_from_upper.end());

    // Rebuild bins
    build_bins(size);
}

void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    // Filter out ghost particles
    std::vector<particle_t> owned;
    for (size_t i = 0; i < local_particles.size(); ++i) {
        double y = local_particles[i].y;
        if ((rank == num_procs - 1 && y >= y_min && y <= y_max) ||
            (rank != num_procs - 1 && y >= y_min && y < y_max)) {
            owned.push_back(local_particles[i]);
        }
    }

    // Gather each processor's number of owned particles
    int owned_count = owned.size();
    std::vector<int> proc_part_counts;
    if (rank == 0) {
        proc_part_counts.resize(num_procs);
    }
    MPI_Gather(&owned_count, 1, MPI_INT, proc_part_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute displacements and initialize list for all particles
    std::vector<int> displs;
    std::vector<particle_t> all_particles;
    if (rank == 0) {
        displs.resize(num_procs);
        displs[0] = 0;

        for (int i = 1; i < num_procs; ++i) {
            displs[i] = proc_part_counts[i - 1] + displs[i - 1];
        }

        all_particles.resize(displs.back() + proc_part_counts.back());
    }
    
    // Gather particles
    MPI_Gatherv(owned.data(), owned_count, PARTICLE,
                all_particles.data(), proc_part_counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);
    
    // Sort particles and update parts
    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), 
            [](const particle_t &a, const particle_t &b) { return a.id < b.id; }
        );
        for (int i = 0; i < num_parts; ++i) {
            parts[i] = all_particles[i];
        }
    }
}
