#include "common.h"
#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <unordered_set>

// Put any static global variables here that you will use throughout the simulation.
static int grid_dimension;
std::unordered_set<int> rank_part_ids;
std::unordered_set<int> rank_grid_ids;
std::unordered_set<int> rank_ghost_part_ids;
static std::vector<std::vector<particle_t*>> grids;
static std::vector<std::vector<int>> neighbor_grids_indices;

// Helper routine returning the index of the block that particle p belongs to.
int get_block_index(particle_t* p) {
    return floor(p->x / cutoff) + floor(p->y / cutoff) * grid_dimension;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    // Setup grids each with cutoff-by-cutoff dimensions.
    grid_dimension = ceil(size / cutoff);
    int total_grid_count = grid_dimension * grid_dimension;

    // Store particles in a row-major order 2D grid
    grids.resize(total_grid_count);
    for (int i = 0; i < num_parts; i++) {
        particle_t* p = parts + i;
        int block_index = get_block_index(p);
        grids[block_index].push_back(p);
    }

    //neighbor grids are self-EXCLUSIVE
    neighbor_grids_indices.resize(total_grid_count);
    // Calculate neighborhood grid indices
    for (int g = 0; g < total_grid_count; g++) {
        int col_index = g % grid_dimension;
        int row_index = g / grid_dimension;
        int row_min = fmax(0, row_index - 1);
        int row_max = fmin(grid_dimension - 1, row_index + 1);
        if (col_index < grid_dimension - 1) {
            for (int row = row_min; row <= row_max; row++) {
                int grid_index = col_index + 1 + row * grid_dimension;
            neighbor_grids_indices[g].push_back(grid_index);
            }
        }
        if (row_index > 0) {
            int grid_index = g - grid_dimension;
            neighbor_grids_indices[g].push_back(grid_index);
        }
    }

    // Assign particles to rank
    for (int i = rank * grid_dimension; i < grid_dimension * grid_dimension; i += grid_dimension * num_procs) {
        for (int j = 0; (j < grid_dimension) && (i + j < grid_dimension * grid_dimension); j += 1) {
            printf("index: %d, rank: %d, grid_dimension: %d, num_procs: %d, num_parts: %d,\n", i + j, rank, grid_dimension, num_procs, num_parts);
            rank_grid_ids.insert(i + j);
            for (const auto& element : grids[i + j]) {
                rank_part_ids.insert(element->id);
            }
        }
    }
    

}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    printf("Hello, world! rank: %d\n", rank);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // // Write this function such that at the end of it, the master (rank == 0)
    // // processor has an in-order view of all particles. That is, the array
    // // parts is complete and sorted by particle id.

    // // Log how many particles were passerd in for debugging
    // // for (int i = 0; i < num_parts; i += 1) {
    // //     printf("id: %d\n", parts[i].id);
    // // }

    // // Vectors for gathering particles from processors
    // std::vector<particle_t> sending_parts;  // Size depends number of particles owned by each processor
    // std::vector<particle_t> receiving_parts(num_parts); // Size

    // // Add particles from processor to be sent for gathering
    // for (const auto& id : rank_part_ids) {
    //     sending_parts.push_back(parts[id]);
    // }

    // // Use variable gather due to particle count varying for each processor
    // // Create arrays for specifying count and offsets of particles for each processor
    // int* receiving_counts;
    // int* offsets;
    // if (rank == 0) {
    //     receiving_counts = new int[num_procs];
    //     offsets = new int[num_procs];
    // }

    // // Use gather to populate array for specifying count of partilces for each offset
    // int sending_count = sending_parts.size();
    // MPI_Gather(&sending_count, 1, MPI_INT, receiving_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // // Build up offsets array
    // if (rank == 0) {
    //     offsets[0] = 0;
    //     for (int i = 1; i < num_parts; i += 1) {
    //         offsets[i] = offsets[i - 1] + receiving_counts[i - 1];
    //     }
    // }

    // // Variable gather of particles from processors
    // MPI_Gatherv(sending_parts.data(), sending_parts.size(), PARTICLE, receiving_parts.data(), receiving_counts, offsets, PARTICLE, 0, MPI_COMM_WORLD);

    // // Create in-order view of all particles, sorted by particle id
    // if (rank == 0) {
    //     for (int i = 0; i < num_parts; i += 1) {
    //         particle_t curr_part = receiving_parts[i];
    //         parts[curr_part.id - 1] = curr_part;
    //     }
    // }
}