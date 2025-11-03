// Parallel reduction kernel - sum all elements

__kernel void reduce_sum(__global const float* input,
                         __global float* output,
                         __local float* scratch,
                         const int length) {
    
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    
    // Load data into local memory
    float value = (global_id < length) ? input[global_id] : 0.0f;
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in local memory
    for (int offset = group_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            scratch[local_id] += scratch[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work-group
    if (local_id == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}

