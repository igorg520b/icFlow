__device__ int nresult;

__inline__ __device__
int warpReduceSum(int val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		//		val += __shfl_down(val, offset);
		val += __shfl_down_sync(0xffffffff, val, offset, warpSize);
	return val;
}

__inline__ __device__
int blockReduceSum(int val) {
	static __shared__ int shared[32];
	int lane = threadIdx.x%warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);

	//write reduced value to shared memory
	if (lane == 0) shared[wid] = val;
	__syncthreads();
	//ensure we only grab a value from shared memory if that warp existed
	val = (threadIdx.x<blockDim.x / warpSize) ? shared[lane] : int(0);
	if (wid == 0) val = warpReduceSum(val);
	return val;
}

extern "C" __global__ void n_kerSum(int *vals, int N) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int sum = 0;
	while (i < N) {
		sum += vals[i];
		i += blockDim.x * gridDim.x;
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0 && sum!=0) atomicAdd(&nresult, sum);
}
