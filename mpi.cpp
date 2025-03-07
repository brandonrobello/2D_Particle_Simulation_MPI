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

// 2D grid of bins for two frames
std::vector<std::vector<Bin>> bins_frame_1;
std::vector<std::vector<Bin>>* current_bins;
int bin_count_x, bin_count_y;

// Global vector holding the local copy of particles for this process
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
    // Apply the same reflection logic as move()
    if (x < 0) x = -x;
    if (x > size) x = 2 * size - x;
    if (y < 0) y = -y;
    if (y > size) y = 2 * size - y;

    // Compute bin indices
    bx = static_cast<int>(x / bin_size);
    by = static_cast<int>(y / bin_size);

    // Ensure bin indices stay within valid range
    bx = std::max(0, std::min(bx, bin_count_x - 1));
    by = std::max(0, std::min(by, bin_count_y - 1));
}

// Apply force with binning for all neighbors of a particle
inline void apply_force_binning(particle_t& particle, Bin& bin) {
    for (auto neighbor : bin.particles) {
        if (neighbor != &particle) {
            apply_force(particle, *neighbor);
        }
    }
}

// Optimized force calculation considering only necessary bins
void compute_forces() {
    for (int i = 0; i < bin_count_x; ++i) {
        for (int j = 0; j < bin_count_y; ++j) {
            Bin& bin = (*current_bins)[i][j];

            for (auto p : bin.particles) {
                // Reset acceleration before force accumulation
                p->ax = p->ay = 0;
                // Check self-bin and neighboring bins within cutoff
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int ni = i + dx;
                        int nj = j + dy;
                        if (ni >= 0 && ni < bin_count_x && nj >= 0 && nj < bin_count_y) {
                            apply_force_binning(*p, (*current_bins)[ni][nj]);
                        }
                    }
                }
            }
        }
    }
}

// Build the binning data structures from local_particles
void build_bins(double size) {
    bin_count_x = static_cast<int>(size / bin_size);
    bin_count_y = static_cast<int>(size / bin_size);
    
    bins_frame_1.assign(bin_count_x, std::vector<Bin>(bin_count_y));
    current_bins = &bins_frame_1;
    
    for (int i = 0; i < bin_count_x; ++i) {
        for (int j = 0; j < bin_count_y; ++j) {
            bins_frame_1[i][j].particles.clear();
        }
    }
    
    // Insert pointers to local_particles into the appropriate bins.
    for (size_t i = 0; i < local_particles.size(); ++i) {
        int bx, by;
        get_bin_index(local_particles[i].x, local_particles[i].y, bx, by, size);
        (*current_bins)[bx][by].particles.push_back(&local_particles[i]);
    }
}

void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    // Compute owned domain boundaries for this process (row decomposition)
    double y_min = size * rank / num_procs;
    double y_max = size * (rank + 1) / num_procs;

    // Extend the domain by cutoff (clamping at 0 and size)
    double ext_y_min = (y_min - cutoff < 0) ? 0 : y_min - cutoff;
    double ext_y_max = (y_max + cutoff > size) ? size : y_max + cutoff;
    
    local_particles.clear();
    for (int i = 0; i < num_parts; ++i) {
        if (parts[i].y >= ext_y_min && parts[i].y <= ext_y_max)
            local_particles.push_back(parts[i]);
    }
    
    // Build the bins from the local particles.
    build_bins(size);
}

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    double y_min = size * rank / num_procs;
    double y_max = size * (rank + 1) / num_procs;

    // Compute forces using optimized binning
    compute_forces();

    // Move owned particles
    std::vector<particle_t> owned;
    for (size_t i = 0; i < local_particles.size(); ++i) {
        double y = local_particles[i].y;
        if ((rank == num_procs - 1 && y >= y_min && y <= y_max) ||
            (rank != num_procs - 1 && y >= y_min && y < y_max)) {
            move(local_particles[i], size);
            owned.push_back(local_particles[i]);
        }
    }

    // Partition owned particles into remaining and outgoing
    std::vector<particle_t> remain;
    std::vector<particle_t> send_up;   // particles with y >= y_max - cutoff
    std::vector<particle_t> send_down; // particles with y < y_min + cutoff
    
    for (size_t i = 0; i < owned.size(); ++i) {
        double y = owned[i].y;

        if ((rank == num_procs - 1 && y >= y_min && y <= y_max) ||
            (rank != num_procs - 1 && y >= y_min && y < y_max))
            remain.push_back(owned[i]);
        
        if (y >= y_max - cutoff && rank < num_procs - 1)
            send_up.push_back(owned[i]);

        if (y <= y_min + cutoff && rank > 0)
            send_down.push_back(owned[i]);
    }

    // Exchange particles with immediate neighbors
    std::vector<particle_t> recv_from_lower, recv_from_upper;

    // Upward exchange:
    // All processors except the top-most send their upward-moving particles.
    if (rank != num_procs - 1) {
        int count = send_up.size();
        MPI_Send(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        if (count > 0) {
            MPI_Send(send_up.data(), count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD);
        }
    }
    // All processors except the bottom-most receive from below.
    if (rank != 0) {
        int count;
        MPI_Recv(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (count > 0) {
            recv_from_lower.resize(count);
            MPI_Recv(recv_from_lower.data(), count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Downward exchange:
    // All processors except the bottom-most send their downward-moving particles.
    if (rank != 0) {
        int count = send_down.size();
        MPI_Send(&count, 1, MPI_INT, rank - 1, 2, MPI_COMM_WORLD);
        if (count > 0) {
            MPI_Send(send_down.data(), count, PARTICLE, rank - 1, 3, MPI_COMM_WORLD);
        }
    }
    // All processors except the top-most receive from above.
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

    // Print total number of particles after one step
    // std::vector<particle_t> after_owned;
    // for (size_t i = 0; i < local_particles.size(); ++i) {
    //     double y = local_particles[i].y;
    //     if ((rank == num_procs - 1 && y >= y_min && y <= y_max) ||
    //         (rank != num_procs - 1 && y >= y_min && y < y_max)) {
    //         after_owned.push_back(local_particles[i]);
    //     }
    // }

    // int local_particle_count = after_owned.size();
    // int total_particle_count;
    // MPI_Reduce(&local_particle_count, &total_particle_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // if (rank == 0) {
    //     printf("Total number of particles after simulate_one_step: %d\n", total_particle_count);
    // }

    // Rebuild bins
    build_bins(size);
}

void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    double y_min = size * rank / num_procs;
    double y_max = size * (rank + 1) / num_procs;
    std::vector<particle_t> owned;
    for (size_t i = 0; i < local_particles.size(); ++i) {
        if (local_particles[i].y >= y_min && local_particles[i].y < y_max)
            owned.push_back(local_particles[i]);
    }
    
    int local_owned = owned.size();
    std::vector<int> recv_counts;
    if (rank == 0) {
        recv_counts.resize(num_procs);
    }
    MPI_Gather(&local_owned, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> displs;
    int total_owned = 0;
    if (rank == 0) {
        displs.resize(num_procs);
        for (int i = 0; i < num_procs; ++i) {
            displs[i] = total_owned;
            total_owned += recv_counts[i];
        }
    }
    
    std::vector<particle_t> all_particles;
    if (rank == 0) {
        all_particles.resize(total_owned);
    }
    
    MPI_Gatherv(owned.data(), local_owned, PARTICLE,
                all_particles.data(), recv_counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t &a, const particle_t &b) {
            return a.id < b.id;
        });
        for (int i = 0; i < total_owned && i < num_parts; ++i) {
            parts[i] = all_particles[i];
        }

        // printf("Total number of particles after gather_for_save: %d\n", total_owned);
    }
}
