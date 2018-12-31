#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

	__global__ void KernelSoftlight(unsigned char* s, unsigned char* c)
	{
		
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		float cf = c[i], sf = s[i], rf = 0;
		if (cf == 0 && sf == 0) return;
		if (cf <= 128) {
			rf = (255 - 2 * cf) * powf(sf / 255, 2) + 2 * cf * sf / 255;
		}
		else
		{
			rf = (2 * cf - 255) * sqrtf(sf / 255) + 2 * (255 - cf) * sf / 255;
		}
		if (rf < 0) s[i] = 0;
		else if (rf > 255) s[i] = 255; else s[i] = (__int8)rf;
	}

	__global__ void KernelSoftlight(unsigned char* s, unsigned char c)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		float cf = c, sf = s[i], rf = 0;
		if (cf == 0 && sf == 0) return;
		if (cf <= 128) {
			rf = (255 - 2 * cf) * powf(sf / 255, 2) + 2 * cf * sf / 255;
		}
		else
		{
			rf = (2 * cf - 255) * sqrtf(sf / 255) + 2 * (255 - cf) * sf / 255;
		}
		if (rf < 0) s[i] = 0;
		else if (rf > 255) s[i] = 255; else s[i] = (__int8)rf;
	}

	void CudaSoftlight(unsigned char* s, unsigned char c, int length, int threads)
	{
		int blocks = length / threads;
		if (length % threads > 0) blocks += 1;
		unsigned char* buf;
		cudaMalloc((void**)&buf, blocks * threads);
		cudaMemcpy(buf, s, length, cudaMemcpyHostToDevice);
		KernelSoftlight<<<blocks, threads >>>(buf, c);
		cudaDeviceSynchronize();
		cudaMemcpy(s, buf, length, cudaMemcpyDeviceToHost);
		cudaFree(buf);
	}

	void CudaSoftlight(unsigned char* s, unsigned char* c, int length, int threads)
	{
		int blocks = length / threads;
		if (length % threads > 0) blocks += 1;
		unsigned char* buf;
		cudaMalloc((void**)&buf, blocks*threads);
		cudaMemcpy(buf, s, length, cudaMemcpyHostToDevice);
		KernelSoftlight<<<blocks, threads >>>(buf, buf);
		cudaDeviceSynchronize();
		cudaMemcpy(s, buf, length, cudaMemcpyDeviceToHost);
		cudaFree(buf);
	}

	__global__ void reduceChar(unsigned char * input, unsigned int * output) {
		extern __shared__ unsigned int sdata[];

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		sdata[tid] = input[i];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}

	__global__ void reduceInt(unsigned int * input, unsigned int * output) {
		extern __shared__ unsigned int sdata[];

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		sdata[tid] = input[i];
		__syncthreads();
		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			if (tid % (2 * s) == 0)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}

	void CudaSum(unsigned char* s, int length, unsigned long long* result, unsigned int maxthreads) {
		unsigned char* buf = 0;
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		cudaMalloc((void**)&buf, blocks * maxthreads * sizeof(char));
		cudaMemcpy(buf, s, length * sizeof(char), cudaMemcpyHostToDevice);
		unsigned int* reduceout = 0;
		cudaMalloc((void**)&reduceout, blocks * sizeof(int));
		reduceChar<<<blocks, maxthreads, maxthreads * sizeof(int)>>>(buf, reduceout);
		cudaDeviceSynchronize();
		cudaFree(buf);
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reduceout, blocks * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			if (blocks%maxthreads == 0) blocks = blocks / maxthreads; else blocks = blocks / maxthreads + 1;
			unsigned int* reducelast = 0;
			cudaMalloc((void**)&reducelast, blocks * sizeof(int));
			reduceInt<<<blocks, maxthreads, maxthreads * sizeof(int)>>> (reduceout, reducelast);
			cudaDeviceSynchronize();
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reducelast, blocks * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}