// Parallel prefix sum (scan) - work-efficient algorithm

__kernel void scan_inclusive(__global const float* input,
                            __global float* output,
                            __local float* temp,
                            const int n) {
    
    int thid = get_local_id(0);
    int offset = 1;
    
    // Load input into local memory
    int ai = thid;
    int bi = thid + (n/2);
    
    temp[ai] = (ai < n) ? input[ai] : 0.0f;
    temp[bi] = (bi < n) ? input[bi] : 0.0f;
    
    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear last element
    if (thid == 0) {
        temp[n - 1] = 0.0f;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Write results
    if (ai < n) output[ai] = temp[ai];
    if (bi < n) output[bi] = temp[bi];
}

