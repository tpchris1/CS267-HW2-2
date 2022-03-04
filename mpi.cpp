#include "common.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// type for bin 
typedef vector<particle_t*> bin_t;
typedef vector<particle_t> row_t;

// total 
bin_t* bins;
int bin_row_count;
int bin_count;
double bin_size;
int total_procs_needed;
int remain_procs_count; // procs can not be divided completely by total_procs_needed

// for each process
int rows_per_proc;
int proc_bin_count;

// e.g if proc 0 has row 0 and row 1 then (proc_rows_start, proc_rows_end) = (0,2)
int proc_rows_start=-1; // left close
int proc_rows_end=-1; //  right open 

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

int inline get_bin_id_by_row(int row_id) {
    // return the first bin of given row_id
    // row_id * bin_row_count ~ (row_id +1) * bin_row_count
    return row_id * bin_row_count; 
}

int inline get_row_id(particle_t& particle){
    int y;
    y = particle.y / bin_size;
    if (y == bin_row_count) {
        y--;
    }
    return y;
}

row_t row_to_particle_vec(int row_id){
    row_t arr;
    int start_bin = get_bin_id_by_row(row_id);
    // cout << "sbin: " << start_bin << endl;
    for(int i=0; i<bin_row_count; i++){
        bin_t cur_bin = bins[i + start_bin];
        for(auto iter: cur_bin){
            arr.push_back(*iter);
        }
    }
    return arr;
}

// from rank send row_id to target
void send_row(int rank, int target, int row_id){
    row_t row = row_to_particle_vec(row_id);
    for(auto iter: row){
        cout << "rank: " << rank << " p: " << iter.id << " ";
    }
    cout << endl << endl;

    MPI_Request* request = new MPI_Request();
    MPI_Isend(&row[0], row.size(), PARTICLE, target, rank, MPI_COMM_WORLD, request);
}

void construct_bin(particle_t* parts, int num_parts) {
    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0;
    	parts[i].ay = 0;
        int row_id = get_row_id(parts[i]);
        if (row_id >= proc_rows_start && row_id < proc_rows_end ){
            int bin_id = get_bin_id(parts[i]);
            bins[bin_id].push_back(&parts[i]);
        } 
    }
}

void print_bin(int rank){
    for(int i=proc_rows_start; i<proc_rows_end;i++){ // for each row in current rank
        int start_bin = get_bin_id_by_row(i);
        for(int j=0; j<bin_row_count; j++){ // for each bin in current row
            int cur_bin_id = j + start_bin;
            bin_t &cur_bin = bins[cur_bin_id];
            for (auto it = cur_bin.begin(); it != cur_bin.end(); it++) {
                int bid = get_bin_id(**it);
                int rid = get_row_id(**it);
                cout << "rank: " << rank 
                     << " row: " << rid 
                     << " bin: " << bid 
                     << " p: " << (**it).id << ' ';
            }
        }
    }
    cout << endl << endl;
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // for total
    bin_row_count = size / cutoff;
    bin_size = size / bin_row_count;
    bin_count = bin_row_count * bin_row_count;
    bins = new bin_t[bin_count];
    
    if(num_procs < bin_row_count) total_procs_needed = num_procs;
    else total_procs_needed = bin_row_count;

    if(rank >= total_procs_needed) return; // if out of needed return
    rows_per_proc = bin_row_count / total_procs_needed;
    
    remain_procs_count = bin_row_count % total_procs_needed; 
    if (remain_procs_count != 0 && rank < remain_procs_count){
        rows_per_proc += 1;
    }   

    if (remain_procs_count != 0 && rank < remain_procs_count){
        proc_rows_start = rank * rows_per_proc;
        proc_rows_end = proc_rows_start + rows_per_proc;
    }
    else{
        int rows_before_me = remain_procs_count * (rows_per_proc + 1);
        proc_rows_start = rows_before_me + (rank-remain_procs_count) * rows_per_proc;
        proc_rows_end = proc_rows_start + rows_per_proc;
    }   

    cout << "rank: " << rank << endl 
        << "rows_per_proc: " << rows_per_proc << endl
        << "proc_rows_start: " << proc_rows_start << endl
        << "proc_rows_end: " << proc_rows_end << endl;

    construct_bin(parts, num_parts);
    print_bin(rank);
    
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    //  if(proc_rows_start == 0){ // if current rank is the first row
    //     // send my last row
    //     row_t last = row_to_particle_vec(proc_rows_end - 1);
    //     MPI_Isend(&last[0], last.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD, &request[1]);
    // }
    // else if(proc_rows_end == bin_row_count){ // if current rank is the last row
    //     // send my first row
    //     row_t first = row_to_particle_vec(proc_rows_start);
    //     MPI_Isend(&first[0], first.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD, &request[0]);
    // }
    // else{

    // }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}