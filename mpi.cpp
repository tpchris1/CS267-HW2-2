#include "common.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// Put any static global variables here that you will use throughout the simulation.
typedef vector<particle_t*> bin_t;

// total 
double size;
bin_t* bins;
int bin_row_count;
int bin_count;
double bin_size;

// for each process
int rows_per_proc;
int proc_bin_count;
int proc_rows_start;
int proc_rows_end; 

int inline get_row_id_particle(particle_t& particle){
    int y;
    y = particle.y / bin_size;
    if (y == bin_row_count) {
        y--;
    }
    return y;
}
int inline get_bin_id_by_row(int row_id) {
    // return the first bin of given row_id
    // row_id * bin_row_count ~ (row_id +1) * bin_row_count
    return row_id * bin_row_count; 
}

int inline get_bin_id(particle_t& particle) {
    int x, y;
    x = particle.x / bin_size;
    y = particle.y / bin_size;
    if (x == bin_row_count) {
        x--;
    }
    if (y == bin_row_count) {
        y--;
    }
    return y * bin_row_count + x;
}

void reconstruct_bin(particle_t* parts, int num_parts) {
    
    for (int i = 0; i < bin_count; i++) {
        bins[i].clear();
    }

    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0;
    	parts[i].ay = 0;
        int row_id = get_row_id_particle(parts[i]);
        if (row_id >= proc_rows_start && row_id < proc_rows_end ){
            int bin_id = get_bin_id(parts[i]);
            bins[bin_id].push_back(&parts[i]);
        } 
    }
}

bin_t row_to_particle_vec(int row_id){
    bin_t arr;
    int start_bin = get_bin_id_by_row(row_id);
    // cout << "sbin: " << start_bin << endl;
    for(int i=0; i<bin_row_count; i++){
        bin_t cur_bin = bins[i + start_bin];
        copy(cur_bin.begin(), cur_bin.end(), back_inserter(arr));
    }
    for(auto iter: arr){
        cout << iter->x << ' ';
    }
    cout << endl;

    return arr;
}

void init_simulation(particle_t* parts, int num_parts, double size_, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    // for total
    bin_row_count = size_ / cutoff;
    bin_count = bin_row_count * bin_row_count;
    bin_size = size_ / bin_row_count;
    
    // for each process
    rows_per_proc = bin_row_count / num_procs; 
    
    // with remainder
    int remainder = bin_row_count % num_procs; 
    if (remainder != 0 && rank < remainder){
        rows_per_proc += 1;
    }   

    // All proc have a copy of all bins, but only access it's own portion of that
    bins = new bin_t[bin_count];

    // TODO: optimize to process based bin (each proc only have its own copy of bins)
    // proc_bin_count = rows_per_proc * bin_row_count;
    // bins = new bin_t[proc_bin_count];
    
    if (remainder != 0 && rank < remainder){
        proc_rows_start = rank * rows_per_proc;
        proc_rows_end = proc_rows_start + rows_per_proc;
    }
    else{
        int rows_before_me = remainder * (rows_per_proc + 1);
        proc_rows_start = rows_before_me + (rank-remainder) * rows_per_proc;
        proc_rows_end = proc_rows_start + rows_per_proc;
    }   

    // reconstruct bin
    reconstruct_bin(parts, num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    if(rank >= 0 && rank <= 2){
        ////////////////////// SEND
        bin_t arr = row_to_particle_vec(proc_rows_start);
        // for(auto iter: arr){
        //     cout << "rank: " << rank << ' ' << iter->x << ' ';
        // }
        // cout << endl;
        if(proc_rows_start != 0){ // up has proc
            // send my first row
            MPI_Request req;
            MPI_Isend(&arr[0], arr.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD, &req);
        }
        // if(proc_rows_end != bin_row_count){

        // }
        
        ////////////////////// RECV
        bin_t lower_row(num_parts);
        if(proc_rows_start != 0){
            
        }
        // recv first row from proc under me
        if(proc_rows_end != bin_row_count){
            // int recvd_tag, recvd_from, recvd_count;
            // MPI_Status status;
            // MPI_Request req;
            // MPI_Recv(&lower_row[0], num_parts, PARTICLE, rank+1, 0, MPI_COMM_WORLD, &status);
            // recvd_tag = status.MPI_TAG;
            // recvd_from = status.MPI_SOURCE;
            // MPI_Get_count( &status, PARTICLE, &recvd_count);
            // // for(auto iter: lower_row){
            // //     cout << "rank: " << rank << ' ' << iter->x << ' ';
            // // }
            // cout << "rank: " << rank << " cnt: " << recvd_count << endl;
            // for(int i=0;i<recvd_count;i++){
            //     cout << "recvd from: " << recvd_from << ' ' << lower_row[i]->x << ' ';
            // }
            // cout << endl;
        }
    }
    
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}