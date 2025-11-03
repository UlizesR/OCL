// Tiled matrix multiplication for better performance

#define TILE_SIZE 16

__kernel void matmul_tiled(__global const float* A,
                          __global const float* B,
                          __global float* C,
                          const int M,
                          const int K,
                          const int N) {
    
    // Thread identifiers
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TILE_SIZE * get_group_id(0) + row;
    const int globalCol = TILE_SIZE * get_group_id(1) + col;
    
    // Local memory for tiles
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];
    
    // Accumulator
    float sum = 0.0f;
    
    // Loop over tiles
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        
        // Load tile from A
        const int tiledRow = globalRow;
        const int tiledCol = t * TILE_SIZE + col;
        if (tiledRow < M && tiledCol < K) {
            Asub[row][col] = A[tiledRow * K + tiledCol];
        } else {
            Asub[row][col] = 0.0f;
        }
        
        // Load tile from B
        const int tiledRow2 = t * TILE_SIZE + row;
        const int tiledCol2 = globalCol;
        if (tiledRow2 < K && tiledCol2 < N) {
            Bsub[row][col] = B[tiledRow2 * N + tiledCol2];
        } else {
            Bsub[row][col] = 0.0f;
        }
        
        // Synchronize to make sure tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Multiply the tiles
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[row][k] * Bsub[k][col];
        }
        
        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store result
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}

