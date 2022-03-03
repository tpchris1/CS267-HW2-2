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

// e.g if proc 0 has row 0 and row 1 then (proc_rows_start, proc_rows_end) = (0,2)
int proc_rows_start; // left close
int proc_rows_end; //  right open 

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
    
    cout << "rank: " << rank << endl 
         << "rows_per_proc: " << rows_per_proc << endl
         << "proc_rows_start: " << proc_rows_start << endl
         << "proc_rows_end: " << proc_rows_end << endl;
    
    // reconstruct bin
    reconstruct_bin(parts, num_parts);
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    if(rank >= 0 && rank <= 10){
        ////////////////////// SEND
        MPI_Request request[2];
        MPI_Status status[2];

        // up has proc
        if(proc_rows_start != 0){ 
            // send my first row
            bin_t first;
            first = row_to_particle_vec(proc_rows_start);
            for(auto iter: first){
                cout << "rank: " << rank << " (" << iter->x << ',' << iter->y << ") ";
            }
            cout << endl;
            MPI_Isend(&first[0], first.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Wait(&request[0], &status[0]);
        }
        
        // down has proc
        if(proc_rows_end != bin_row_count){ 
            // send my last row
            bin_t last = row_to_particle_vec(proc_rows_end - 1);
            cout << "send last rank: " << rank << endl;
            
            for(auto iter: last){
                cout << "rank: " << rank << " (" << iter->x << ',' << iter->y << ") ";
            }
            cout << endl;
            MPI_Isend(&last[0], last.size(), PARTICLE, rank+1, 1, MPI_COMM_WORLD, &request[1]);
            MPI_Wait(&request[1], &status[1]);
        }
                
        ////////////////////// RECV
        bin_t upper_row(num_parts), lower_row(num_parts);

        // recv last row from proc on top of me
        if(proc_rows_start != 0){
            int recvd_tag, recvd_from, recvd_count;

            MPI_Recv(&upper_row[0], num_parts, PARTICLE, rank-1, 1, MPI_COMM_WORLD, &status[1]);
            recvd_tag = status[1].MPI_TAG;
            recvd_from = status[1].MPI_SOURCE;
            MPI_Get_count( &status[1], PARTICLE, &recvd_count);

            cout << "upper rank: " << rank << " cnt: " << recvd_count << endl;
            for(int i=0;i<recvd_count;i++){
                cout << "upper rank: " << rank << " recvd from: " << recvd_from << " (" << upper_row[i]->x << ',' << upper_row[i]->y << ") ";
            }
            cout << endl << endl;
        }

        // recv first row from proc under me
        if(proc_rows_end != bin_row_count){
            int recvd_tag, recvd_from, recvd_count;
            MPI_Recv(&lower_row[0], num_parts, PARTICLE, rank+1, 0, MPI_COMM_WORLD, &status[0]);
            bin_t temp;
            copy(lower_row.begin(), lower_row.end(), back_inserter(temp));

            recvd_tag = status[0].MPI_TAG;
            recvd_from = status[0].MPI_SOURCE;
            MPI_Get_count( &status[0], PARTICLE, &recvd_count);

            cout << "lower rank: " << rank << " cnt: " << recvd_count << endl;
            for(int i=0;i<recvd_count;i++){
                cout << "lower rank: " << rank << " recvd from: " << recvd_from << " (" << temp[i]->x << ',' << temp[i]->y << ") ";
            }
            cout << endl << endl;
        }
    }
    
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}