# Distributed Particle Simulation with MPI

## Contributions

* **Write-Up**: All
* **Brandon**: MPI troubleshooting; computation and communication breakdown
* **Kevin**: MPI implementation and debugging, runtime optimization, and benchmarking

---

## Introduction

This assignment extends the work from Homework 2.1 by replacing OpenMP shared-memory parallelism with **MPI-based distributed memory parallelism**.

### Goals:

* Partition a 2D repulsive particle simulation across MPI ranks
* Maintain correct cross-boundary force calculations
* Analyze performance as the number of processes increases

---

## Optimization & Design

### Spatial Binning

As in HW2.1, we use **spatial binning** to reduce the force computation from `O(n²)` to `O(n)` by:

* Dividing space into a 2D grid of bins
* Limiting interactions to particles in the same or adjacent bins

### MPI Domain Decomposition

* Simulation space is partitioned across MPI ranks in **1D along the y-axis**
* Each process is assigned a strip (rows) of the simulation and padded with **halo regions**
* **Halo size = interaction cutoff** to ensure correct force calculations at boundaries

### Key Functions

* `build_bins()` constructs the binning structure for each process’s local + halo region
* `simulate_one_step()`:

  * Filters out ghost particles before updating
  * Identifies particles to retain or send to neighbors
  * Exchanges particles using send buffers

---

## MPI Communication Strategy

Implemented via **MPI point-to-point communication** (`MPI_Send`, `MPI_Recv`):

* 2-stage exchange:

  1. Send particle count
  2. Send actual particle data
* Upward and downward communication phases
* Avoided deadlocks by ordering send/receive operations

### Benefits:

* Only halo particles are exchanged (not entire datasets)
* Dynamic buffer sizes reduce memory and communication cost
* One communication phase per simulation step ensures efficiency

---

## Performance Analysis

### Summary:

* **MPI implementation (1 process)** has slight overhead due to MPI setup, but maintains linear scaling
* **Speedup is most noticeable** going from 1 → 4 → 32 processes
* **Beyond 32 processes**, overhead from partitioning and communication limits additional gains

---

## Scaling Results

### Strong Scaling

* Fixed particle count, increasing process count
* Ideal slope = -1 on log-log plot
* **Best performance** at 100,000 particles
* Overhead dominates at very high or low process counts

### Weak Scaling

* Proportional increase in problem size and process count
* Ideal slope = 0 on log-log plot
* **Good scaling up to 32 processes**, then performance degrades due to sync costs

#### Bottlenecks Identified:

* Current communication order causes cascading delays
* Proposed fix: pair neighboring processes for overlapping send/receive to reduce blocking

---

## Computation vs. Communication Breakdown

Simulation time is broken down into:

* Initialization
* Bin building
* Simulation step:

  * Force computation
  * Partitioning/movement
  * Communication

### Key Observations:

* **At low ranks**: force computation dominates
* **At high ranks (≥128)**: communication becomes the bottleneck
* Partitioning overhead shrinks but shows diminishing returns after 32 processes
* Communication cost increases with rank count and overtakes gains from force distribution

---

## Future Improvements

To achieve better scalability at high ranks:

* Use **asynchronous communication**
* Overlap **computation and communication**
* **Reduce ghost region updates** or their frequency
* Consider a **better communication scheme** to prevent cascading delays

---

This project demonstrates the trade-offs in distributed parallel computing: while computation parallelizes well, communication and synchronization can dominate beyond a certain scale. With further optimization, this framework could support even larger simulations efficiently.
