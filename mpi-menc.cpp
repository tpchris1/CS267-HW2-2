#include "common.h"
#include <mpi.h>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <set>
#include <iostream>
#include <map>

using namespace std;

double binSize;
int binRowCount, procCount, rowsPerProc, numOfProcHasMoreRows, binCount;
vector<particle_t *> *bin;
vector<int> localBinIds;
int *migrateSizes, *dispSizes;

vector<particle_t> toSend;
particle_t *toRecvFromUp, *toRecvFromDown;

vector<particle_t> particlesSendToUp;
vector<particle_t> particlesSendToDown;

set<int> surroundingBinIds;


inline int getIndex(particle_t &point) {
    int x = floor(point.x/binSize);
    x = (x >= binRowCount) ? x - 1: x;

    int y = floor(point.y/binSize);
    y = (y >= binRowCount) ? y - 1: y;

    return x*binRowCount + y;
}

inline int getProcIndex(int binId) {
    int rowId = binId / binRowCount;
    if (rowId >= numOfProcHasMoreRows * (rowsPerProc + 1)) {
        return (rowId - numOfProcHasMoreRows * (rowsPerProc + 1)) / rowsPerProc + numOfProcHasMoreRows;
    } else {
        return rowId / (rowsPerProc + 1);
    }
}

void getLocalBins(int rank) {
    if (rank < numOfProcHasMoreRows) {
        for (int i=0; i < (rowsPerProc + 1); i++) { // Since every process would be assigned (rowsPerProc + 1) rows
            int rowId = rank * (rowsPerProc + 1) + i;
            for (int j=0; j < binRowCount; j++) {
                localBinIds.push_back(rowId*binRowCount + j);
            }
        }
    } else {
        int padding = numOfProcHasMoreRows*(rowsPerProc+1);
        for (int i=0; i<rowsPerProc; i++) {
            int rowId = padding + (rank - numOfProcHasMoreRows) * rowsPerProc + i;
            for (int j=0; j < binRowCount; j++) {
                localBinIds.push_back(rowId*binRowCount + j);
            }
        }
    }
}

void assignBins(particle_t* parts, int num_parts, int rank) {
    for (int i=0; i<num_parts; i++) {
        particle_t &part = parts[i];
        int binId = getIndex(part);
        int procId = getProcIndex(binId);
        if (procId == rank) {
            bin[binId].push_back(&part);
        }
    }
}


// Put any static global variables here that you will use throughout the simulation.

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    binRowCount = size/cutoff; // Binsize = cutoff
    binSize = size / binRowCount;
    binCount = binRowCount * binRowCount;
    bin = new vector<particle_t *>[binCount];
    procCount = min(num_procs, binRowCount);
    rowsPerProc = binRowCount / procCount;
    numOfProcHasMoreRows = binRowCount % procCount;

    toRecvFromUp = new particle_t[5 * binRowCount];
    toRecvFromDown = new particle_t[5 * binRowCount];

    if (rank < procCount) {
        getLocalBins(rank);
        assignBins(parts, num_parts, rank);
        for (auto binId : localBinIds) {
            for (auto part : bin[binId]) {
                cout << "rank: " << rank 
                     << " bin: " << binId 
                     << " p: " << (*part).id << ' ';
            }
        }
        cout << endl << endl;
    }

    migrateSizes = (int*) malloc(num_procs * sizeof(int));
    dispSizes = (int*) malloc(num_procs * sizeof(int));
}

inline void applyForce(particle_t& particle, particle_t& neighbor) {
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

    neighbor.ax += coef * (-dx);
    neighbor.ay += coef * (-dy);
}

inline void move(particle_t& p, double size) {
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

    p.ax = p.ay = 0;
}

inline bool hasUpProc  (int rank) { return rank >= procCount ? false : rank > 0; }
inline bool hasDownProc(int rank) { return rank < procCount - 1; }
inline int  getUpProc  (int rank) { return rank - 1; }
inline int  getDownProc(int rank) { return rank + 1; }

// if shouldGetLastLayer = false, we'll get first layer
vector<int>* getRowBinIds(int rank, bool shouldGetLastLayer) {
    vector<int> *res = new vector<int>;
    int row;
    if (rank < numOfProcHasMoreRows) {
        row = rank * (rowsPerProc + 1);
        if (shouldGetLastLayer) row += rowsPerProc;
    } else {
        row = numOfProcHasMoreRows * (rowsPerProc + 1) + (rank - numOfProcHasMoreRows) * rowsPerProc;
        if (shouldGetLastLayer) row += rowsPerProc-1;
    }
    for (int i=0; i<binRowCount; i++) {
        res->push_back(row*binRowCount + i);
    }

    return res;
}

void sendParticles(int rank, int direction) {
     vector<int>* layer;

     if (direction == 0) { // direction 0: Up, 4: Down
         layer = getRowBinIds(rank, false);
     } else {
         layer = getRowBinIds(rank, true);
     }

     for (auto binId : *layer) {
        for (auto part : bin[binId]) {
            toSend.push_back(*part);
        }
    }

    MPI_Request* request = new MPI_Request();
    if (direction == 0) {
        MPI_Isend(&toSend[0], toSend.size(), PARTICLE, getUpProc(rank), 0, MPI_COMM_WORLD, request);
    } else {
        MPI_Isend(&toSend[0], toSend.size(), PARTICLE, getDownProc(rank), 0, MPI_COMM_WORLD, request);
    }

    delete layer;
    delete request;
    toSend.clear();
}

void recvParticles(int num_parts, int rank, int direction) {
    MPI_Status status;
    MPI_Request* request = new MPI_Request();

    particle_t* toRecv;

    if (direction == 0) {
        MPI_Irecv(toRecvFromUp,   num_parts, PARTICLE, getUpProc(rank),   0, MPI_COMM_WORLD, request);
        toRecv = toRecvFromUp;
    } else {
        MPI_Irecv(toRecvFromDown, num_parts, PARTICLE, getDownProc(rank), 0, MPI_COMM_WORLD, request);
        toRecv = toRecvFromDown;
    }

    MPI_Wait(request, &status);

    int recvCount = 0;
    MPI_Get_count(&status, PARTICLE, &recvCount);
    for (int i=0; i<recvCount; i++) {
        int binId = getIndex(toRecv[i]);
        bin[binId].push_back(&toRecv[i]);
        surroundingBinIds.insert(binId);
    }

    delete request;
}

inline void calcForceWithSelf(int selfBinId) {
    vector<particle_t *> &particles = bin[selfBinId];
    for (vector<particle_t *>::iterator it = particles.begin(); it != particles.end(); it++) {
        for (vector<particle_t *>::iterator nit = it+1; nit != particles.end(); nit++) {
            applyForce(**nit, **it);
        }
    }
}

inline void loop(particle_t* part, int neighborBinId) {
    for (particle_t* neighbor : bin[neighborBinId]) {
        applyForce(*part, *neighbor);
    }
}

void renewPart(particle_t* original_parts, particle_t &newPart) {
    int binId = getIndex(newPart);
    bin[binId].push_back(&original_parts[newPart.id - 1]);
    original_parts[newPart.id - 1].x = newPart.x;
    original_parts[newPart.id - 1].y = newPart.y;
    original_parts[newPart.id - 1].vx = newPart.vx;
    original_parts[newPart.id - 1].vy = newPart.vy;
    original_parts[newPart.id - 1].ax = newPart.ax;
    original_parts[newPart.id - 1].ay = newPart.ay;
}

void rebin(particle_t* original_parts, int num_parts, int rank, int num_procs) {
    for (int binId: localBinIds) {
        for (auto it = bin[binId].begin(); it != bin[binId].end(); it++) {
            int newBinId = getIndex(**it);
            int newProcId = getProcIndex(newBinId);
            if (newProcId != rank) {
                if (newProcId == rank-1) particlesSendToUp.push_back(**it);
                else particlesSendToDown.push_back(**it);
                bin[binId].erase(it--);
            } else if (newBinId != binId) {
                bin[newBinId].push_back(*it);
                bin[binId].erase(it--);
            }
        }
    }

    MPI_Request* request = new MPI_Request();
    if (hasUpProc(rank)) {
        MPI_Isend(&particlesSendToUp[0],   particlesSendToUp.size(),   PARTICLE, getUpProc(rank),   0, MPI_COMM_WORLD, request);
    }
    if (hasDownProc(rank)) {
        MPI_Isend(&particlesSendToDown[0], particlesSendToDown.size(), PARTICLE, getDownProc(rank), 0, MPI_COMM_WORLD, request);
    }

    int numPartFromUp = 0, numPartFromDown = 0;
    MPI_Request status[2];
    if (hasUpProc(rank)) {
        MPI_Status status;
        MPI_Recv(toRecvFromUp,   num_parts, PARTICLE, getUpProc(rank), 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &numPartFromUp);
    }
    if (hasDownProc(rank)) {
        MPI_Status status;
        MPI_Recv(toRecvFromDown, num_parts, PARTICLE, getDownProc(rank), 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &numPartFromDown);
    }

    for (int i = 0; i < numPartFromUp; i++) {
        particle_t &newPart = toRecvFromUp[i];
        renewPart(original_parts, newPart);
    }

    for (int i = 0; i < numPartFromDown; i++) {
        particle_t &newPart = toRecvFromDown[i];
        renewPart(original_parts, newPart);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    particlesSendToUp.clear();
    particlesSendToDown.clear();
}

inline bool hasUpBin   (int binId) { return binId - binRowCount >= 0; }
inline bool hasDownBin (int binId) { return binId + binRowCount < binCount; }
inline bool hasLeftBin (int binId) { return binId % binRowCount != 0; }
inline bool hasRightBin(int binId) { return binId % binRowCount != binRowCount - 1; }

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if (hasUpProc(rank)) {
        // TODO: Whether needs to store sendBuffer => Change to use MPI_Barrier
        sendParticles(rank, 0);
    }
    if (hasDownProc(rank)) {
        sendParticles(rank, 4);
    }

    if (hasUpProc(rank)) {
        recvParticles(num_parts, rank, 0);
    }

    if (hasDownProc(rank)) {
        recvParticles(num_parts, rank, 4);
    } 

    if (hasUpProc(rank)) {
        vector<int>* layer;
        layer = getRowBinIds(getUpProc(rank), true);
        for (int binId: *layer) {
            for (auto part: bin[binId]) {
                if (hasRightBin(binId)) {
                    loop(part, binId + 1);
                }
                if (hasDownBin(binId) && hasRightBin(binId)) {
                    loop(part, binId + binRowCount + 1);
                }
                if (hasDownBin(binId)) {
                    loop(part, binId + binRowCount);
                }
                if (hasDownBin(binId) && hasLeftBin(binId)) {
                    loop(part, binId + binRowCount - 1);
                }
            }
        }
        delete layer;
    }


    for (int binId: localBinIds) {
        for (auto part: bin[binId]) {
            if (hasRightBin(binId)) {
                loop(part, binId + 1);
            }
            if (hasDownBin(binId) && hasRightBin(binId)) {
                loop(part, binId + binRowCount + 1);
            }
            if (hasDownBin(binId)) {
                loop(part, binId + binRowCount);
            }
            if (hasDownBin(binId) && hasLeftBin(binId)) {
                loop(part, binId + binRowCount - 1);
            }
        }
        calcForceWithSelf(binId);
    }

    for (auto binId : localBinIds) {
        for (auto part : bin[binId]) {
            move(*part, size);
        }
    }


    for (auto binId : surroundingBinIds) {
        bin[binId].clear();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    rebin(parts, num_parts, rank, num_procs);

    surroundingBinIds.clear();
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    vector<particle_t> localParts;
    for (int binId: localBinIds) {
        for (particle_t* p: bin[binId]) {
            localParts.push_back(*p);
        }
    }
    int* gatherPartSizes = new int[num_procs];
    int* gatherDispSizes = new int[num_procs];
    
    int localPartSize = localParts.size();

    // TODO: Combine the next two into MPIReduce
    int error_code = MPI_Gather(&localPartSize, 1, MPI_INT, gatherPartSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != 0) {
        cout << "MPI_Gather error!!" << endl;
    }

    gatherDispSizes[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        gatherDispSizes[i] = gatherDispSizes[i-1] + gatherPartSizes[i-1];
    }

    particle_t* recvBuff = new particle_t[num_parts];
    error_code = MPI_Gatherv(&localParts[0], localParts.size(), PARTICLE, recvBuff,
            gatherPartSizes, gatherDispSizes, PARTICLE, 0, MPI_COMM_WORLD);
    if (error_code != 0) {
        cout << "MPI_Gatherv error!!" << endl;
    }

    if (rank == 0) {
        for (int i = 0; i < num_parts; i++) {
            particle_t &p = recvBuff[i];
            parts[p.id-1].x = p.x;
            parts[p.id-1].y = p.y;
            parts[p.id-1].ax = p.ax;
            parts[p.id-1].ay = p.ay;
            parts[p.id-1].vx = p.vx;
            parts[p.id-1].vy = p.vy;
        }
    }

    delete[] recvBuff;
    delete[] gatherPartSizes;
    delete[] gatherDispSizes;
}