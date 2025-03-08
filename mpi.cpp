#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <map>

int total_within_cutoff;

// ================== Left Alone ==============================

// Apply the force from neighbor to particle
// void apply_force(particle_t& particle, particle_t& neighbor) {
//     // Calculate Distance
//     double dx = neighbor.x - particle.x;
//     double dy = neighbor.y - particle.y;
//     double r2 = dx * dx + dy * dy;

//     // Check if the two particles should interact
//     if (r2 > cutoff * cutoff){
//         printf("DEBUG: Skipping interaction - Particle ID %ld and Neighbor ID %ld, r2 = %.6f, cutoff^2 = %.6f\n",
//         particle.id, neighbor.id, r2, cutoff * cutoff);
//         return;
//     }

//     r2 = fmax(r2, min_r * min_r);
//     double r = sqrt(r2);

//     // Very simple short-range repulsive force
//     double coef = (1 - cutoff / r) / r2 / mass;
//     particle.ax += coef * dx;
//     particle.ay += coef * dy;
// }

void apply_force(particle_t& p1, particle_t& p2) {
    // Calculate Distance
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff) {
        printf("DEBUG: Skipping interaction - Particle ID %ld and Neighbor ID %ld, r2 = %.6f, cutoff^2 = %.6f\n",
               p1.id, p2.id, r2, cutoff * cutoff);
        return;
    }

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    double fx = coef * dx;
    double fy = coef * dy;

    // Apply forces symmetrically to both particles
    p1.ax += fx;
    p1.ay += fy;
    p2.ax -= fx;  // Apply equal and opposite force
    p2.ay -= fy;
}


// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// ================== Left Alone ==============================

// Define bin size
const double bin_size = 1.2 * cutoff;

// Bin struct to hold particles
struct Bin {
    std::vector<particle_t> particles;
};

// Global bins only used by Rank 0
std::vector<std::vector<Bin>> master_bins_frame;

// Local bins for each rank
std::vector<std::vector<Bin>> local_bins_frame;

// Ghost bins for inter-rank communication
std::vector<std::vector<Bin>> ghost_bins_frame;

// Number of bins in x and y directions
int bin_count_x, bin_count_y;

// Rank-specific bin partitions
std::vector<int> start_bin_yindex;
std::vector<int> end_bin_yindex;
// Compute bin index for a given position
void get_bin_index(double x, double y, int& bx, int& by, double size) {
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

// Distribute bins using MPI_Scatterv instead of manual MPI_Send
void distribute_bins(int rank, int num_procs, double size) {
    int total_particles = 0;
    std::vector<int> send_counts(num_procs, 0);
    std::vector<int> displacements(num_procs, 0);
    std::vector<particle_t> send_particles;

    if (rank == 0) {
        // Flatten particle data for sending
        for (int irank = 0; irank < num_procs; irank++) {
            int start_y = start_bin_yindex[irank];
            int end_y = end_bin_yindex[irank];

            for (int i = 0; i < bin_count_x; i++) {
                for (int j = start_y; j < end_y; j++) {
                    for (auto& p : master_bins_frame[i][j].particles) {
                        send_particles.push_back(p);
                    }
                }
            }
            send_counts[irank] = send_particles.size() - total_particles;
            total_particles = send_particles.size();
        }

        // Compute displacements for MPI_Scatterv
        displacements[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displacements[i] = displacements[i - 1] + send_counts[i - 1];
        }
    }

    // Receive particle counts
    int recv_count;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate buffer and receive particles
    std::vector<particle_t> recv_particles(recv_count);
    MPI_Scatterv(send_particles.data(), send_counts.data(), displacements.data(), PARTICLE,
                 recv_particles.data(), recv_count, PARTICLE, 0, MPI_COMM_WORLD);

    //DEBUG
    // printf("Rank %d: Received %ld particles after distribution\n", rank, recv_particles.size());

    
    // Assign received particles to local bins
    for (auto& p : recv_particles) {
        int bx, by;
        get_bin_index(p.x, p.y, bx, by, size);
        local_bins_frame[bx][by].particles.push_back(p);
    }

    // DEBUG
    // printf("Rank %d: Received %d particles after distribution\n", rank, recv_particles.size());

}


// Initialize bins per MPI rank
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    bin_count_x = static_cast<int>(size / bin_size);
    bin_count_y = static_cast<int>(size / bin_size);

    // Resize global bins (Only Rank 0 needs this)
    if (rank == 0) {
        master_bins_frame.assign(bin_count_x, std::vector<Bin>(bin_count_y));
    }

    // Resize local and ghost bins
    local_bins_frame.assign(bin_count_x, std::vector<Bin>(bin_count_y));
    ghost_bins_frame.assign(bin_count_x, std::vector<Bin>(2));

    // Resize and assign bin indices per rank
    start_bin_yindex.resize(num_procs);
    end_bin_yindex.resize(num_procs);
    int current_start_index = 0;

    for (int irank = 0; irank < num_procs; irank++) {
        start_bin_yindex[irank] = current_start_index;
        int n_bins = bin_count_y / num_procs;
        if (irank < bin_count_y % num_procs) n_bins++;  // Handle remainder bins
        end_bin_yindex[irank] = start_bin_yindex[irank] + n_bins;
        current_start_index = end_bin_yindex[irank];
    }

    // ===================== Sanity Check: Before Distribution =====================
    if (rank == 0) {
        printf("\n[Sanity Check] Checking particle distances BEFORE distribution...\n");

        total_within_cutoff = 0;
        for (int i = 0; i < num_parts; i++) {
            for (int j = i + 1; j < num_parts; j++) {
                if (parts[i].id == parts[j].id) continue;

                double dx = parts[j].x - parts[i].x;
                double dy = parts[j].y - parts[i].y;
                double r2 = dx * dx + dy * dy;

                if (r2 <= cutoff * cutoff) {
                    total_within_cutoff++;
                }
            }
        }
        printf("Rank 0: Total particle pairs within cutoff BEFORE distribution: %d\n\n", total_within_cutoff);
    }
    // ===================== Sanity Check: Before Distribution =====================


    if (rank == 0) {
        // Assign particles to global bins (only Rank 0 does this)
        for (int i = 0; i < num_parts; i++) {
            int bx, by;
            get_bin_index(parts[i].x, parts[i].y, bx, by, size);
            master_bins_frame[bx][by].particles.push_back(parts[i]);
        }
    }

    // // DEBUG
    // if (rank == 0) {
    //     printf("\n--- Rank 0: Verifying All Particles Are in Global Bins ---\n");
    
    //     int total_particles = 0;
    //     for (int j = 0; j < bin_count_y; j++) {
    //         int row_particles = 0;
    //         for (int i = 0; i < bin_count_x; i++) {
    //             row_particles += master_bins_frame[i][j].particles.size();
    //         }
    //         total_particles += row_particles;
    //         printf("Row %d: %d particles\n", j, row_particles);
    //     }
    
    //     printf("Total particles in global bins: %d (should match num_parts = %d)\n", total_particles, num_parts);
    // }

    // Broadcast metadata to all ranks
    MPI_Bcast(&bin_count_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bin_count_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(start_bin_yindex.data(), num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(end_bin_yindex.data(), num_procs, MPI_INT, 0, MPI_COMM_WORLD);

    // Send bins to each rank
    distribute_bins(rank, num_procs, size);


    // ===================== Sanity Check: After Distribution =====================
    if (rank == 0) {
        printf("\n[Sanity Check] Checking particle distances AFTER distribution...\n");

        int total_within_cutoff_after = 0;
        
        // Compare all particles in bins
        for (int i = 0; i < bin_count_x; i++) {
            for (int j = 0; j < bin_count_y; j++) {
                for (size_t p1 = 0; p1 < local_bins_frame[i][j].particles.size(); p1++) {
                    for (size_t p2 = p1 + 1; p2 < local_bins_frame[i][j].particles.size(); p2++) {
                        if (local_bins_frame[i][j].particles[p1].id == local_bins_frame[i][j].particles[p2].id) 
                            continue;  // Skip self-comparison

                        double dx = local_bins_frame[i][j].particles[p2].x - local_bins_frame[i][j].particles[p1].x;
                        double dy = local_bins_frame[i][j].particles[p2].y - local_bins_frame[i][j].particles[p1].y;
                        double r2 = dx * dx + dy * dy;

                        if (r2 <= cutoff * cutoff) {
                            total_within_cutoff_after++;
                        }
                    }
                }
            }
        }
        printf("Rank 0: Total particle pairs within cutoff AFTER distribution: %d\n\n", total_within_cutoff_after);

        if (total_within_cutoff != total_within_cutoff_after) {
            printf("WARNING: Particle interactions BEFORE and AFTER distribution do NOT match!\n");
        } else {
            printf("Particle interactions BEFORE and AFTER distribution MATCH!\n");
        }
        // ===================== Sanity Check: After Distribution =====================

    // // DEBUG
    // MPI_Barrier(MPI_COMM_WORLD); // Ensure Rank 0's output is complete before others start

    // for (int r = 0; r < num_procs; r++) {
    //     if (r == rank) {
    //         printf("\n--- Rank %d: Verifying Particles in Local Bins ---\n", rank);
    //         int local_total_particles = 0;
    
    //         for (int j = start_bin_yindex[rank]; j < end_bin_yindex[rank]; j++) {
    //             int row_particles = 0;
    //             for (int i = 0; i < bin_count_x; i++) {
    //                 row_particles += local_bins_frame[i][j].particles.size();
    //             }
    //             local_total_particles += row_particles;
    //             printf("Rank %d - Row %d: %d particles\n", rank, j, row_particles);
    //         }
    
    //         printf("Rank %d: Total particles in assigned bins = %d\n", rank, local_total_particles);
    //         fflush(stdout); // Ensure output is printed immediately
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD); // Synchronize before next rank prints
    // }
    }
}

void compute_forces(int rank) {

    // DEBUG
    printf("Rank %d: Computing forces for particles in local bins...\n", rank);

    // Loop over all local bins
    for (int i = 0; i < bin_count_x; i++) {
        for (int j = 0; j < bin_count_y; j++) {
            for (auto& p : local_bins_frame[i][j].particles) {
                // p.ax = 0;
                // p.ay = 0;

                // Compute forces within the same bin
                for (size_t n = 0; n < local_bins_frame[i][j].particles.size(); n++) {
                    particle_t& neighbor = local_bins_frame[i][j].particles[n];
                    if (p.id < neighbor.id) {  // Ensure each pair is processed only once

                        // DEBUG
                        printf("Rank %d: Particle %ld in Bin (%d, %d) checking interactions with %ld\n", 
                               rank, p.id, i, j, neighbor.id);
                        printf("Particle p (%f, %f) -> Particle neighbor (%f, %f)\n", 
                               p.x, p.y, neighbor.x, neighbor.y);

                        apply_force(p, neighbor);

                        //DEBUG
                        printf("Rank %d: Particle %ld updated force -> ax: %.6f, ay: %.6f\n", 
                               rank, p.id, p.ax, p.ay);                    
                    }
                }

                // Compute forces from neighboring bins (local only)
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int ni = i + dx;
                        int nj = j + dy;

                        // Ensure within local bin bounds and avoid self-bin duplicate calculations
                        if (ni >= 0 && ni < bin_count_x && nj >= 0 && nj < bin_count_y && (dx != 0 || dy != 0)) {
                            for (auto& neighbor : local_bins_frame[ni][nj].particles) {
                                if (p.id < neighbor.id) {  // Avoid duplicate interactions

                                    // DEBUG
                                    printf("Rank %d: Particle %ld in Bin (%d, %d) checking interactions with %ld\n", 
                                           rank, p.id, i, j, neighbor.id);
                                            
                                    apply_force(p, neighbor);

                                    //DEBUG
                                    printf("Rank %d: Particle %ld updated force -> ax: %.6f, ay: %.6f\n", 
                                           rank, p.id, p.ax, p.ay);            
                                }
                            }
                        }
                    }
                }

                // Handle ghost bins **for edge interactions**
                if (j == 0) {  // If in the first row, use ghost particles from above
                    for (auto& neighbor : ghost_bins_frame[i][0].particles) {
                        if (p.id < neighbor.id) {
                            apply_force(p, neighbor);
                        }
                    }
                }
                if (j == bin_count_y - 1) {  // If in the last row, use ghost particles from below
                    for (auto& neighbor : ghost_bins_frame[i][1].particles) {
                        if (p.id < neighbor.id) {
                            apply_force(p, neighbor);
                        }
                    }
                }
            }
        }
    }

    // DEBUG
    for (int i = 0; i < bin_count_x; i++) {
        for (int j = 0; j < bin_count_y; j++) {
            for (const auto& p : local_bins_frame[i][j].particles) {
                printf("Rank %d: Particle %ld in Bin (%d, %d) -> Final Force: ax = %.6f, ay = %.6f\n",
                       rank, p.id, i, j, p.ax, p.ay);
            }
        }
    }
    
}





// Function to exchange ghost bins between neighboring ranks
void exchange_ghost_bins(int rank, int num_procs) {
    int above_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int below_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    std::vector<particle_t> send_above, send_below, recv_above, recv_below;

    // Identify primary bin row indices
    int top_primary_start = start_bin_yindex[rank];
    int bottom_primary_end = end_bin_yindex[rank] - 1;

    // Count total particles in the top and bottom rows before sending
    int total_top_particles = 0;
    int total_bottom_particles = 0;

    // Collect particles for sending
    if (above_rank != MPI_PROC_NULL) {
        for (int i = 0; i < bin_count_x; i++) {
            for (auto& p : local_bins_frame[i][top_primary_start].particles) {
                send_above.push_back(p);
            }
        }
    }

    if (below_rank != MPI_PROC_NULL) {
        for (int i = 0; i < bin_count_x; i++) {
            for (auto& p : local_bins_frame[i][bottom_primary_end].particles) {
                send_below.push_back(p);
            }
        }
    }

    // Exchange sizes first
    int send_above_size = send_above.size();
    int send_below_size = send_below.size();
    int recv_above_size = 0, recv_below_size = 0;

    // // Debug
    // for (int r = 0; r < num_procs; r++) {
    //     if (r == rank) {
    //         printf("Rank %d: Preparing to send %ld particles above and %ld below\n", 
    //             rank, send_above.size(), send_below.size());
    //         fflush(stdout); // Ensure output is printed immediately
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD); // Synchronize before next rank prints
    // }

    MPI_Request reqs[4];
    MPI_Status stats[4];

    if (above_rank != MPI_PROC_NULL) {
        MPI_Isend(&send_above_size, 1, MPI_INT, above_rank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&recv_above_size, 1, MPI_INT, above_rank, 1, MPI_COMM_WORLD, &reqs[1]);
    } else {
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
    }

    if (below_rank != MPI_PROC_NULL) {
        MPI_Isend(&send_below_size, 1, MPI_INT, below_rank, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&recv_below_size, 1, MPI_INT, below_rank, 0, MPI_COMM_WORLD, &reqs[3]);
    } else {
        reqs[2] = MPI_REQUEST_NULL;
        reqs[3] = MPI_REQUEST_NULL;
    }

    MPI_Waitall(4, reqs, stats);

    // // Debug
    // for (int r = 0; r < num_procs; r++) {
    //     if (r == rank) {
    //         printf("Rank %d: Received %d particles from above, %d from below\n", 
    //             rank, recv_above_size, recv_below_size);
    //         fflush(stdout); // Ensure output is printed immediately
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD); // Synchronize before next rank prints
    // }

    // Ensure correct allocation of receive buffers
    recv_above.resize(recv_above_size);  // Allocate memory
    recv_below.resize(recv_below_size);  // Allocate memory

    if (above_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_above.data(), send_above_size, PARTICLE, above_rank, 2,
                    recv_above.data(), recv_above_size, PARTICLE, above_rank, 3,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (below_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_below.data(), send_below_size, PARTICLE, below_rank, 3,
                    recv_below.data(), recv_below_size, PARTICLE, below_rank, 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    
    if (above_rank != MPI_PROC_NULL) {
        for (const auto& p : recv_above) {
            int bx, by;
            get_bin_index(p.x, p.y, bx, by, bin_count_x * bin_size);
            ghost_bins_frame[bx][0].particles.push_back(p);  // Store by value, not pointer
        }
    }
    
    if (below_rank != MPI_PROC_NULL) {
        for (const auto& p : recv_below) {
            int bx, by;
            get_bin_index(p.x, p.y, bx, by, bin_count_x * bin_size);
            ghost_bins_frame[bx][1].particles.push_back(p);  // Store by value, not pointer
        }
    }

    // // DEBUG
    // for (int r = 0; r < num_procs; r++) {
    //     if (r == rank) {
    //         // Sum the total number of particles in the ghost bins
    //         int total_top_ghost_particles = 0;
    //         int total_bottom_ghost_particles = 0;

    //         for (int i = 0; i < bin_count_x; i++) {
    //             total_top_ghost_particles += ghost_bins_frame[i][0].particles.size();
    //             total_bottom_ghost_particles += ghost_bins_frame[i][1].particles.size();
    //         }

    //         // Print the total number of ghost particles and compare with received counts
    //         printf("Rank %d: Total particles in top ghost row: %d (Expected: %d)\n", 
    //             rank, total_top_ghost_particles, recv_above_size);

    //         printf("Rank %d: Total particles in bottom ghost row: %d (Expected: %d)\n", 
    //             rank, total_bottom_ghost_particles, recv_below_size);
                        
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD); // Synchronize before next rank prints
    // }
}

void move_particles(int rank, int num_procs, double size) {
    std::vector<particle_t> send_above, send_below;

    // Iterate through all local bins and update particle positions
    for (int i = 0; i < bin_count_x; i++) {
        for (int j = 0; j < bin_count_y; j++) {
            auto& bin = local_bins_frame[i][j];
            auto it = bin.particles.begin();

            while (it != bin.particles.end()) {
                move(*it, size);  // Update position using Velocity Verlet

                // Determine new bin index after movement
                int new_bx, new_by;
                get_bin_index(it->x, it->y, new_bx, new_by, size);

                // If particle stays within local bins, reassign it
                if (new_by >= start_bin_yindex[rank] && new_by < end_bin_yindex[rank]) {
                    if (new_bx != i || new_by != j) {
                        local_bins_frame[new_bx][new_by].particles.push_back(*it);
                        it = bin.particles.erase(it);
                    } else {
                        ++it;
                    }
                }
                // If particle moves beyond the rank's partition, store it in send buffer
                else {
                    if (new_by < start_bin_yindex[rank]) {
                        send_above.push_back(*it);  // Send to upper rank
                    } else {
                        send_below.push_back(*it);  // Send to lower rank
                    }
                    it = bin.particles.erase(it);  // Remove from local bin
                }
            }
        }
    }

    // Exchange moved particles with neighboring ranks
    int above_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int below_rank = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    int send_above_size = send_above.size();
    int send_below_size = send_below.size();
    int recv_above_size = 0, recv_below_size = 0;

    MPI_Request reqs[4];
    MPI_Status stats[4];

    // Send sizes of particles being transferred
    if (above_rank != MPI_PROC_NULL) {
        MPI_Isend(&send_above_size, 1, MPI_INT, above_rank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&recv_above_size, 1, MPI_INT, above_rank, 1, MPI_COMM_WORLD, &reqs[1]);
    } else {
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
    }

    if (below_rank != MPI_PROC_NULL) {
        MPI_Isend(&send_below_size, 1, MPI_INT, below_rank, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&recv_below_size, 1, MPI_INT, below_rank, 0, MPI_COMM_WORLD, &reqs[3]);
    } else {
        reqs[2] = MPI_REQUEST_NULL;
        reqs[3] = MPI_REQUEST_NULL;
    }

    MPI_Waitall(4, reqs, stats);

    // Allocate buffers to receive incoming particles
    std::vector<particle_t> recv_above(recv_above_size);
    std::vector<particle_t> recv_below(recv_below_size);

    // Send/receive actual particles
    if (above_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_above.data(), send_above_size, PARTICLE, above_rank, 2,
                     recv_above.data(), recv_above_size, PARTICLE, above_rank, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (below_rank != MPI_PROC_NULL) {
        MPI_Sendrecv(send_below.data(), send_below_size, PARTICLE, below_rank, 3,
                     recv_below.data(), recv_below_size, PARTICLE, below_rank, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Reassign received particles to local bins
    for (const auto& p : recv_above) {
        int bx, by;
        get_bin_index(p.x, p.y, bx, by, size);
        local_bins_frame[bx][by].particles.push_back(p);
    }

    for (const auto& p : recv_below) {
        int bx, by;
        get_bin_index(p.x, p.y, bx, by, size);
        local_bins_frame[bx][by].particles.push_back(p);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function

    exchange_ghost_bins(rank, num_procs);
    // compute_forces(rank);
    // move_particles(rank, num_procs, size);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::vector<int> recv_counts(num_procs, 0);
    std::vector<int> displacements(num_procs, 0);

    // Count number of particles in each rank
    int local_count = 0;
    for (int i = 0; i < bin_count_x; i++) {
        for (int j = 0; j < bin_count_y; j++) {
            local_count += local_bins_frame[i][j].particles.size();
        }
    }

    // Gather particle counts
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacements for gather
    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    // Flatten local particles into a single array
    std::vector<particle_t> local_particles;
    for (int i = 0; i < bin_count_x; i++) {
        for (int j = 0; j < bin_count_y; j++) {
            for (auto& p : local_bins_frame[i][j].particles) {
                local_particles.push_back(p);
            }
        }
    }

    // Rank 0 receives all particles
    std::vector<particle_t> all_particles;
    if (rank == 0) {
        all_particles.resize(num_parts);
    }

    MPI_Gatherv(local_particles.data(), local_count, PARTICLE, 
                all_particles.data(), recv_counts.data(), displacements.data(), PARTICLE, 
                0, MPI_COMM_WORLD);

    // Sort particles by ID to maintain order
    if (rank == 0) {
        std::sort(all_particles.begin(), all_particles.end(), [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });

        // Copy to final array
        std::copy(all_particles.begin(), all_particles.end(), parts);
    }

    // DEBUG
    // if (rank == 0) {
    //     printf("Rank 0: Gathered all %d particles for final output\n", num_parts);
    // }
    
}