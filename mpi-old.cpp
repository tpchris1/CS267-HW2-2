#include "common.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// Put any static global variables here that you will use throughout the simulation.
typedef vector<particle_t*> bin_t;
typedef vector<particle_t> row_t;

// total 
bin_t* bins;
int bin_row_count;
int bin_count;
double bin_size;
int remainder

// for each process
int rows_per_proc;
int proc_bin_count;

// e.g if proc 0 has row 0 and row 1 then (proc_rows_start, proc_rows_end) = (0,2)
int proc_rows_start; // left close
int proc_rows_end; //  right open 

row_t upper_row, lower_row;
row_t recv_upper, recv_lower;

particle_t* recv_all;
int *dispSizes, *migrateSizes;

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

inline void assignPart(particle_t* parts, particle_t &newPart) {
    parts[newPart.id - 1].x = newPart.x;
    parts[newPart.id - 1].y = newPart.y;
    parts[newPart.id - 1].vx = newPart.vx;
    parts[newPart.id - 1].vy = newPart.vy;
    parts[newPart.id - 1].ax = newPart.ax;
    parts[newPart.id - 1].ay = newPart.ay;
}

void reconstruct_row_to_bin(row_t row, int row_count, particle_t* parts){
    for(int i=0; i<row_count; i++){
        particle_t &newp = row[i];
        int cur_bin_id = get_bin_id(newp);
        if(get_row_id_particle(newp) >= proc_rows_start && get_row_id_particle(newp) < proc_rows_end){
            assignPart(parts, newp);
            bins[cur_bin_id].push_back(&parts[newp.id - 1]);
        }
    }
    return;
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

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
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

void inline do_neighbor(int cur_bin_id, int i, int nei_bin_id) {
    for (particle_t* neighbor : bins[nei_bin_id]) {
        apply_force(*bins[cur_bin_id][i], *neighbor);
    }
}

void inline do_cur_bin(int i, int cur_bin_id) {
    if (i > 0){
        for (int j = i - 1; j > -1; --j){
            apply_force(*bins[cur_bin_id][i], *bins[cur_bin_id][j]);
        } 
    }
}

bool inline has_up(int bin_id) {
    return bin_id - bin_row_count > -1;
}
bool inline has_down(int bin_id) {
    return bin_id + bin_row_count < bin_count;
}
bool inline has_left(int bin_id) {
    return bin_id % bin_row_count != 0;
}
bool inline has_right(int bin_id) {
    return bin_id % bin_row_count != bin_row_count - 1;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    // for total
    bin_row_count = size / cutoff;
    bin_size = size / bin_row_count;
    bin_count = bin_row_count * bin_row_count;
    bins = new bin_t[bin_count];
    
    // for each process

    rows_per_proc = bin_row_count / num_procs; 
    
    // with remainder
    int remainder = bin_row_count % num_procs; 
    if (remainder != 0 && rank < remainder){
        rows_per_proc += 1;
    }   

    // All proc have a copy of all bins, but only access it's own portion of that

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

    upper_row.resize(num_parts);
    lower_row.resize(num_parts);
    recv_upper.resize(num_parts);
    recv_lower.resize(num_parts);

    recv_all = new particle_t[num_parts];

    migrateSizes = (int*) malloc(num_procs * sizeof(int));
    dispSizes = (int*) malloc(num_procs * sizeof(int));
    
    // reconstruct bin
    reconstruct_bin(parts, num_parts);
}

void rebin(particle_t* original_parts, int num_parts, int rank, int num_procs) {
    vector<particle_t> particlesToSend;
    for(int i=proc_rows_start; i<proc_rows_end;i++){ // for each row in current rank
        int start_bin = get_bin_id_by_row(i);
        for(int j=0; j<bin_row_count; j++){ // for each bin in current row
            int cur_bin_id = j + start_bin;
            bin_t &cur_bin = bins[cur_bin_id];
            for (auto it = cur_bin.begin(); it != cur_bin.end(); it++) {
                int newBinId = get_bin_id(**it);
                int newRowId = get_row_id_particle(**it);
                if (newRowId < proc_rows_start || newRowId >= proc_rows_end) {
                    particlesToSend.push_back(**it);
                    cur_bin.erase(it--);
                } else if (newBinId != cur_bin_id) {
                    bins[newBinId].push_back(*it);
                    cur_bin.erase(it--);
                }
            }
        }
    }

    int particlesToSendSize = particlesToSend.size();
    MPI_Allgather(&particlesToSendSize, 1, MPI_INT, migrateSizes, 1, MPI_INT, MPI_COMM_WORLD);

    int totalMigrateSize = migrateSizes[0];
    dispSizes[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        dispSizes[i] = dispSizes[i-1] + migrateSizes[i-1];
        totalMigrateSize += migrateSizes[i];
    }

    MPI_Allgatherv(&particlesToSend[0], particlesToSend.size(), PARTICLE, recv_all, migrateSizes, dispSizes, PARTICLE, MPI_COMM_WORLD);

    for (int i = 0; i < totalMigrateSize; i++) {
        int newBinId = get_bin_id(recv_all[i]);
        int newRowId = get_row_id_particle(recv_all[i]);
        if (newRowId >= proc_rows_start && newRowId < proc_rows_end) {
            particle_t &newPart = recv_all[i];
            bins[newBinId].push_back(&original_parts[newPart.id - 1]);
            assignPart(original_parts, newPart);
        }
    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
    /***********************BEGIN - Send first/last row, Recv upper/lower row *************************/
    ////////////////////// SEND
    MPI_Request request[2];
    MPI_Status status[2];

    // 
    if(rank-1 >= 0){ // 
        // clean up procs last row
        int start_bin_id = get_bin_id_by_row(proc_rows_start - 1); 
        for(int i=0;i<bin_row_count;i++){
            bins[start_bin_id + i].clear();
        } 
    }
    if(rank < num_procs-1){
        int start_bin_id = get_bin_id_by_row(proc_rows_end); 
        for(int i=0;i<bin_row_count;i++){
            bins[start_bin_id + i].clear();
        } 
    }


    // cout << "DEBUG rank: " << rank << " part:" << 1 << endl;
    if(proc_rows_start == 0){ // if current rank is the first row
        // send my last row
        row_t last = row_to_particle_vec(proc_rows_end - 1);
        MPI_Isend(&last[0], last.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD, &request[1]);
    }
    else if(proc_rows_end == bin_row_count){ // if current rank is the last row
        // send my first row
        row_t first = row_to_particle_vec(proc_rows_start);
        MPI_Isend(&first[0], first.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD, &request[0]);
    }
    else{
        // send my last row
        row_t last = row_to_particle_vec(proc_rows_end - 1);
        MPI_Isend(&last[0], last.size(), PARTICLE, rank+1, 0, MPI_COMM_WORLD, &request[1]);

        // send my first row
        row_t first = row_to_particle_vec(proc_rows_start);
        MPI_Isend(&first[0], first.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD, &request[0]);
    }
    // cout << "DEBUG rank: " << rank << " part:" << 2 << endl;
    ////////////////////// RECV
    int upper_count=0, lower_count=0;
    if(proc_rows_start == 0){ // if current rank is the first row
        // recv first row from rank 1
        int recvd_from;
        MPI_Recv(&lower_row[0], num_parts, PARTICLE, rank+1, 0, MPI_COMM_WORLD, &status[0]);
        recvd_from = status[0].MPI_SOURCE;
        MPI_Get_count( &status[0], PARTICLE, &lower_count);
    }
    else if(proc_rows_end == bin_row_count){ // if current rank is the last row
        // recv last row from 2nd last rank 
        int recvd_from;
        MPI_Recv(&upper_row[0], num_parts, PARTICLE, rank-1, 0, MPI_COMM_WORLD, &status[1]);
        recvd_from = status[1].MPI_SOURCE;
        MPI_Get_count( &status[1], PARTICLE, &upper_count);

    }
    else{ // middle ranks
        // recv first row from lower rank
        int recvd_from;
        MPI_Recv(&lower_row[0], num_parts, PARTICLE, rank+1, 0, MPI_COMM_WORLD, &status[0]);
        recvd_from = status[0].MPI_SOURCE;
        MPI_Get_count( &status[0], PARTICLE, &lower_count);


        // recv last row from upper rank 
        MPI_Recv(&upper_row[0], num_parts, PARTICLE, rank-1, 0, MPI_COMM_WORLD, &status[1]);
        recvd_from = status[1].MPI_SOURCE;
        MPI_Get_count( &status[1], PARTICLE, &upper_count);

    }

    /***********************END - Send first/last row, Recv upper/lower row *************************/

    // cout << "DEBUG rank: " << rank << " part:" << 3 << endl;
    // put upper row and lower row into bins
    for(int i=0;i<upper_count;i++){
        particle_t &p = upper_row[i];
        int bin_id = get_bin_id(p);
        bins[bin_id].push_back(&p);
    }
    // cout << endl << endl;
    for(int i=0;i<lower_count;i++){
        particle_t &p = lower_row[i];
        int bin_id = get_bin_id(p);
        bins[bin_id].push_back(&p);
    }
    // cout << endl << endl;
    
    // cout << "DEBUG rank: " << rank << " part:" << 4 << endl;

    // calculate the row of particles
    // handle last row problem
    int actual_rows_end = 0;
    // if last row -> need to take care of last row
    if(rank == num_procs -1) actual_rows_end = proc_rows_end - 1; 
    else actual_rows_end = proc_rows_end;
    
    for(int i=proc_rows_start; i<=actual_rows_end;i++){ // for each row in current proc
        int start_bin = get_bin_id_by_row(i);
        for(int j=0; j<bin_row_count; j++){ // for each bin in current row
            int cur_bin_id = j + start_bin;
            bin_t cur_bin = bins[cur_bin_id];
            for(int p=0; p<cur_bin.size(); p++){ // for each particle in current bin
                do_cur_bin(p, cur_bin_id);

                // up
                if (has_up(cur_bin_id)) {
                    do_neighbor(cur_bin_id, p, cur_bin_id - bin_row_count);
                }
                // up right
                if (has_up(cur_bin_id) && has_right(cur_bin_id)) {
                    do_neighbor(cur_bin_id, p, cur_bin_id - bin_row_count + 1);
                }
                // left
                if (has_left(cur_bin_id)) {
                    do_neighbor(cur_bin_id, p, cur_bin_id - 1);
                }
                // up left
                if (has_up(cur_bin_id) && has_left(cur_bin_id)) {
                    do_neighbor(cur_bin_id, p, cur_bin_id - bin_row_count - 1);
                }
            } 
        }
    }
    
    // cout << "DEBUG rank: " << rank << " part:" << 5 << endl;



    // move the row of particles
    for(int i=proc_rows_start; i<proc_rows_end;i++){ // for each row in current rank
        int start_bin = get_bin_id_by_row(i);
        for(int j=0; j<bin_row_count; j++){ // for each bin in current row
            int cur_bin_id = j + start_bin;
            bin_t cur_bin = bins[cur_bin_id];
            for(int p=0; p<cur_bin.size(); p++){ // for each particle in current bin
                if (cur_bin[p]->id == 21){
                    // cout << "!!!!!! rank " << rank << " row: " << i << " cbid: " << cur_bin_id << endl;
                }
                move(*cur_bin[p], size);  
            }
        }
    }

    rebin(parts, num_parts, rank, num_procs);
}




void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // Push local paricles into send_all
    row_t send_all;
    for(int i=proc_rows_start; i<proc_rows_end;i++){ // for each row in current rank
        int start_bin = get_bin_id_by_row(i);
        for(int j=0; j<bin_row_count; j++){ // for each bin in current row
            int cur_bin_id = j + start_bin;
            bin_t cur_bin = bins[cur_bin_id];
            for(int p=0; p<cur_bin.size(); p++){
                send_all.push_back(*cur_bin[p]);
            }
        }
    }

    int send_all_size = send_all.size();
    MPI_Gather(&send_all_size, 1, MPI_INT, migrateSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    dispSizes[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        dispSizes[i] = dispSizes[i-1] + migrateSizes[i-1];
    }

    MPI_Gatherv(&send_all[0], send_all.size(), PARTICLE, recv_all, migrateSizes, dispSizes, PARTICLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < num_parts; i++) {
            particle_t &newPart = recv_all[i];
            assignPart(parts, newPart);
        }
    }
}