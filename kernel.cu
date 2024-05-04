#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <npp.h>


const char* cudaerror() {
	const char* error = cudaGetErrorString(cudaGetLastError());
	return error;
}

//Softlight functions

	__global__ void KernelSoftlightFC_W3C(unsigned char* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float r;
			float g;
			if (a <= 0.25)
				g = ((16.0 * a - 12.0) * a + 4) * a;
			else
				g = sqrtf(a);
			if (b <= 0.5)
				r = (a - (1.0 - 2 * b) * a * (1.0 - a)) * 255;
			else
				r = (a + (2 * b - 1) * (g - a)) * 255;
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8) (r + 0.5);
		}
	}
	__global__ void KernelSoftlightFC_pegtop(unsigned char* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a) * 255;
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}
	__global__ void KernelSoftlightFC_illusionshu(unsigned char* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float r = powf(a, powf(2, (2 * (0.5 - b)))) * 255; //can be fastened
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}
	__global__ void KernelSoftlightF_W3C(Npp32f* s, Npp32f* d, Npp32f clampmin, Npp32f clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i];
			float b = (float)d[i];
			float r;
			float g;
			if (a <= 0.25)
				g = ((16.0 * a - 12.0) * a + 4) * a;
			else
				g = sqrtf(a);
			if (b <= 0.5)
				r = (a - (1.0 - 2 * b) * a * (1.0 - a));
			else
				r = (a + (2 * b - 1) * (g - a));
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = r;
		}
	}
	__global__ void KernelSoftlightF_pegtop(Npp32f* s, Npp32f* d, Npp32f clampmin, Npp32f clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i];
			float b = (float)d[i];
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a);
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = r;
		}
	}
	__global__ void KernelSoftlightF_illusionshu(Npp32f* s, Npp32f* d, Npp32f clampmin, Npp32f clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i];
			float b = (float)d[i];
			float r = powf(a, powf(2, (2 * (0.5 - b))));
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = r;
		}
	}
	__global__ void KernelSoftlight_W3C(unsigned char* s, unsigned char* d, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float b = (float)d[i] / 255;
			float r;
			float g;
			if (a <= 0.25)
				g = ((16.0 * a - 12.0) * a + 4) * a;
			else
				g = sqrtf(a);
			if (b <= 0.5)
				r = (a - (1.0 - 2 * b) * a * (1.0 - a)) * 255;
			else
				r = (a + (2 * b - 1) * (g - a)) * 255;
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_pegtop(unsigned char* s, unsigned char* d, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float b = (float)d[i] / 255;
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a) * 255;
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_illusionshu(unsigned char* s, unsigned char* d, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float b = (float)d[i] / 255;
			float r = powf(a, powf(2, (2 * (0.5 - b)))) * 255;
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}
//10bit
	__global__ void KernelSoftlightFC_W3C(unsigned short* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float r;
			float g;
			if (a <= 0.25)
				g = ((16.0 * a - 12.0) * a + 4) * a;
			else
				g = sqrtf(a);
			if (b <= 0.5)
				r = (a - (1.0 - 2 * b) * a * (1.0 - a)) * 1023;
			else
				r = (a + (2 * b - 1) * (g - a)) * 1023;
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}
	__global__ void KernelSoftlightFC_pegtop(unsigned short* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a) * 1023;
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}
	__global__ void KernelSoftlightFC_illusionshu(unsigned short* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float r = powf(a, powf(2, (2 * (0.5 - b)))) * 1023; //can be fastened
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_W3C(unsigned short* s, unsigned short* d, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float b = (float)d[i] / 1023;
			float r;
			float g;
			if (a <= 0.25)
				g = ((16.0 * a - 12.0) * a + 4) * a;
			else
				g = sqrtf(a);
			if (b <= 0.5)
				r = (a - (1.0 - 2 * b) * a * (1.0 - a)) * 1023;
			else
				r = (a + (2 * b - 1) * (g - a)) * 1023;
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_pegtop(unsigned short* s, unsigned short* d, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float b = (float)d[i] / 1023;
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a) * 1023;
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_illusionshu(unsigned short* s, unsigned short* d, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float b = (float)d[i] / 1023;
			float r = powf(a, powf(2, (2 * (0.5 - b)))) * 1023;
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}


//RGB<->HSV functions

	__global__ void KernelRGB2HSV(unsigned char* R, unsigned char* G, unsigned char* B, Npp32f* H, Npp32f* S, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)255;
		Npp32f double_g = G[i] / (Npp32f)255;
		Npp32f double_b = B[i] / (Npp32f)255;
		Npp32f h, s, v;


		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		v = fmaxf(double_r, fmaxf(double_g, double_b));
		if (fabs(diff) < 0.00001)
		{
			s = 0;
			h = 0;
		}
		else
		{
			if (max == 0) s = 0; else s = (Npp32f)1 - ((Npp32f)1 * min / max);
			Npp32f r_dist = (max - double_r) / diff;
			Npp32f g_dist = (max - double_g) / diff;
			Npp32f b_dist = (max - double_b) / diff;
			if (double_r == max) h = b_dist - g_dist;
			else if (double_g == max) h = 2 + r_dist - b_dist;
			else h = 4 + g_dist - r_dist;
			h = h * 60;
			if (h < 0) h += 360;
		}
		H[i] = h;
		S[i] = s;
		V[i] = v;
	}
	__global__ void KernelRGB2HSV_S(unsigned char* R, unsigned char* G, unsigned char* B, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)255;
		Npp32f double_g = G[i] / (Npp32f)255;
		Npp32f double_b = B[i] / (Npp32f)255;
		Npp32f s;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		if (fabs(diff) < 0.00001)
		{
			s = 0;
		}
		else
		{
			if (max == 0) s = 0; else s = (Npp32f)1 - ((Npp32f)1 * min / max);
		}
		S[i] = s;
	}
	__global__ void KernelRGB2HSV_S10(unsigned short* R, unsigned short* G, unsigned short* B, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)1023;
		Npp32f double_g = G[i] / (Npp32f)1023;
		Npp32f double_b = B[i] / (Npp32f)1023;
		Npp32f s;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		if (fabs(diff) < 0.00001)
		{
			s = 0;
		}
		else
		{
			if (max == 0) s = 0; else s = (Npp32f)1 - ((Npp32f)1 * min / max);
		}
		S[i] = s;
	}
	__global__ void KernelRGB2HSV_HV(unsigned char* R, unsigned char* G, unsigned char* B, Npp32f* H, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)255;
		Npp32f double_g = G[i] / (Npp32f)255;
		Npp32f double_b = B[i] / (Npp32f)255;
		Npp32f h, v;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		v = fmaxf(double_r, fmaxf(double_g, double_b));
		if (fabs(diff) < 0.00001)
		{
			h = 0;
		}
		else
		{
			Npp32f r_dist = (max - double_r) / diff;
			Npp32f g_dist = (max - double_g) / diff;
			Npp32f b_dist = (max - double_b) / diff;
			if (double_r == max) h = b_dist - g_dist;
			else if (double_g == max) h = 2 + r_dist - b_dist;
			else h = 4 + g_dist - r_dist;
			h = h * 60;
			if (h < 0) h += 360;
		}
		H[i] = h;
		V[i] = v;
	}
	__global__ void KernelRGB2HSV_HV10(unsigned short* R, unsigned short* G, unsigned short* B, Npp32f* H, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)1023;
		Npp32f double_g = G[i] / (Npp32f)1023;
		Npp32f double_b = B[i] / (Npp32f)1023;
		Npp32f h, v;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		v = fmaxf(double_r, fmaxf(double_g, double_b));
		if (fabs(diff) < 0.00001)
		{
			h = 0;
		}
		else
		{
			Npp32f r_dist = (max - double_r) / diff;
			Npp32f g_dist = (max - double_g) / diff;
			Npp32f b_dist = (max - double_b) / diff;
			if (double_r == max) h = b_dist - g_dist;
			else if (double_g == max) h = 2 + r_dist - b_dist;
			else h = 4 + g_dist - r_dist;
			h = h * 60;
			if (h < 0) h += 360;
		}
		H[i] = h;
		V[i] = v;
	}
	__global__ void KernelRGB2HSV_HS(unsigned char* R, unsigned char* G, unsigned char* B, Npp32f* H, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)255;
		Npp32f double_g = G[i] / (Npp32f)255;
		Npp32f double_b = B[i] / (Npp32f)255;
		Npp32f h, s;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		if (fabs(diff) < 0.00001)
		{
			s = 0;
			h = 0;
		}
		else
		{
			if (max == 0) s = 0; else s = (Npp32f)1 - ((Npp32f)1 * min / max);
			Npp32f r_dist = (max - double_r) / diff;
			Npp32f g_dist = (max - double_g) / diff;
			Npp32f b_dist = (max - double_b) / diff;
			if (double_r == max) h = b_dist - g_dist;
			else if (double_g == max) h = 2 + r_dist - b_dist;
			else h = 4 + g_dist - r_dist;
			h = h * 60;
			if (h < 0) h += 360;
		}
		H[i] = h;
		S[i] = s;
	}
	__global__ void KernelRGB2HSV_HS10(unsigned short* R, unsigned short* G, unsigned short* B, Npp32f* H, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)1023;
		Npp32f double_g = G[i] / (Npp32f)1023;
		Npp32f double_b = B[i] / (Npp32f)1023;
		Npp32f h, s;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		if (fabs(diff) < 0.00001)
		{
			s = 0;
			h = 0;
		}
		else
		{
			if (max == 0) s = 0; else s = (Npp32f)1 - ((Npp32f)1 * min / max);
			Npp32f r_dist = (max - double_r) / diff;
			Npp32f g_dist = (max - double_g) / diff;
			Npp32f b_dist = (max - double_b) / diff;
			if (double_r == max) h = b_dist - g_dist;
			else if (double_g == max) h = 2 + r_dist - b_dist;
			else h = 4 + g_dist - r_dist;
			h = h * 60;
			if (h < 0) h += 360;
		}
		H[i] = h;
		S[i] = s;
	}
	__global__ void KernelRGB2HSV_SV(unsigned char* R, unsigned char* G, unsigned char* B, Npp32f* S, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)255;
		Npp32f double_g = G[i] / (Npp32f)255;
		Npp32f double_b = B[i] / (Npp32f)255;
		Npp32f s, v;

		// Get the maximum and minimum RGB components.
		Npp32f max = double_r;
		if (max < double_g) max = double_g;
		if (max < double_b) max = double_b;

		Npp32f min = double_r;
		if (min > double_g) min = double_g;
		if (min > double_b) min = double_b;

		Npp32f diff = max - min;

		v = fmaxf(double_r, fmaxf(double_g, double_b));
		if (fabs(diff) < 0.00001)
		{
			s = 0;
		}
		else
		{
			if (max == 0) s = 0; else s = (Npp32f)1 - ((Npp32f)1 * min / max);
		}
		S[i] = s;
		V[i] = v;
	}
	__global__ void KernelRGB2HSV_V(unsigned char* R, unsigned char* G, unsigned char* B, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)255;
		Npp32f double_g = G[i] / (Npp32f)255;
		Npp32f double_b = B[i] / (Npp32f)255;
		Npp32f v;
		v = fmaxf(double_r, fmaxf(double_g, double_b));
		V[i] = v;
	}
	__global__ void KernelRGB2HSV_V10(unsigned short* R, unsigned short* G, unsigned short* B, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i] / (Npp32f)1023;
		Npp32f double_g = G[i] / (Npp32f)1023;
		Npp32f double_b = B[i] / (Npp32f)1023;
		Npp32f v;
		v = fmaxf(double_r, fmaxf(double_g, double_b));
		V[i] = v;
	}
	__global__ void KernelHSV2RGB10(Npp32f* H, Npp32f* S, Npp32f* V, unsigned short* R, unsigned short* G, unsigned short* B, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		Npp32f h, s, v;
		h = H[i];
		s = S[i];
		v = V[i];
		int hi = (int)(floorf(h / 60)) % 6;
		Npp32f f = h / 60 - floorf(h / 60);

		unsigned short vi = (unsigned short)round(v * 1023);
		v = v * (Npp32f)1023;
		unsigned short p = (unsigned short)round(v * ((Npp32f)1 - s));
		unsigned short q = (unsigned short)round(v * ((Npp32f)1 - f * s));
		unsigned short t = (unsigned short)round(v * ((Npp32f)1 - ((Npp32f)1 - f) * s));
		switch (hi)
		{
		case 0:
		{
			R[i] = vi;
			G[i] = t;
			B[i] = p;
			break;
		}
		case 1:
		{
			R[i] = q;
			G[i] = vi;
			B[i] = p;
			break;
		}
		case 2:
		{
			R[i] = p;
			G[i] = vi;
			B[i] = t;
			break;
		}
		case 3:
		{
			R[i] = p;
			G[i] = q;
			B[i] = vi;
			break;
		}
		case 4:
		{
			R[i] = t;
			G[i] = p;
			B[i] = vi;
			break;
		}
		case 5:
		{
			R[i] = vi;
			G[i] = p;
			B[i] = q;
			break;
		}
		default:
		{
			R = 0; G = 0; B = 0;
			break;
		}
		}
	}
	__global__ void KernelHSV2RGB(Npp32f* H, Npp32f* S, Npp32f* V, unsigned char* R, unsigned char* G, unsigned char* B, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		Npp32f h, s, v;
		h = H[i];
		s = S[i];
		v = V[i];
		int hi = (int)(floorf(h / 60)) % 6;
		Npp32f f = h / 60 - floorf(h / 60);

		unsigned char vi = (unsigned char)round(v * 255);
		v = v * (Npp32f)255;
		unsigned char p = (unsigned char)round(v * ((Npp32f)1 - s));
		unsigned char q = (unsigned char)round(v * ((Npp32f)1 - f * s));
		unsigned char t = (unsigned char)round(v * ((Npp32f)1 - ((Npp32f)1 - f) * s));
		switch (hi)
		{
		case 0:
		{
			R[i] = vi;
			G[i] = t;
			B[i] = p;
			break;
		}
		case 1:
		{
			R[i] = q;
			G[i] = vi;
			B[i] = p;
			break;
		}
		case 2:
		{
			R[i] = p;
			G[i] = vi;
			B[i] = t;
			break;
		}
		case 3:
		{
			R[i] = p;
			G[i] = q;
			B[i] = vi;
			break;
		}
		case 4:
		{
			R[i] = t;
			G[i] = p;
			B[i] = vi;
			break;
		}
		case 5:
		{
			R[i] = vi;
			G[i] = p;
			B[i] = q;
			break;
		}
		default:
		{
			R = 0; G = 0; B = 0;
			break;
		}
		}
	}

//RGB<->BGR functions

	__global__ void KernelRGBtoBGR(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, unsigned char* planeBGR, int width, int height, int Rpitch, int BGRpitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Rpitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Rpitch) return; //if i greater than buffer - do nothing

		int row = i / Rpitch;
		int position = i % Rpitch;

		unsigned char R, G, B;
		R = planeR[row * Rpitch + position];
		G = planeG[row * Rpitch + position];
		B = planeB[row * Rpitch + position];

		int bgrsize = height * BGRpitch;
		int pixeloffset = bgrsize - ((row + 1) * BGRpitch) + position * 4;
		planeBGR[pixeloffset] = B;
		planeBGR[pixeloffset + 1] = G;
		planeBGR[pixeloffset + 2] = R;
		planeBGR[pixeloffset + 3] = 255;
	}
	__global__ void KernelBGRtoRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, unsigned char* planeBGR, int width, int height, int Rpitch, int BGRpitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Rpitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Rpitch) return; //if i greater than buffer - do nothing

		int row = i / Rpitch;
		int position = i % Rpitch;

		unsigned char R, G, B;

		int bgrsize = height * BGRpitch;
		int pixeloffset = bgrsize - ((row + 1) * BGRpitch) + position * 4;
		B = planeBGR[pixeloffset];
		G = planeBGR[pixeloffset + 1];
		R = planeBGR[pixeloffset + 2];

		planeR[row * Rpitch + position] = R;
		planeG[row * Rpitch + position] = G;
		planeB[row * Rpitch + position] = B;
	}

//RGB<->YUV
	__global__ void KernelYUV2RGB(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		
		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f C = (Npp32f)planeY[i];
		Npp32f D = (Npp32f)planeU[i] - 128;
		Npp32f E = (Npp32f)planeV[i] - 128;

		Npp32f Rf = round(C + 1.4075 * E);
		Npp32f Gf = round(C - 0.3455 * D - 0.7169 * E);
		Npp32f Bf = round(C + 1.7790 * D);

		unsigned char R, G, B;
		if (Rf > 255) R = 255; else if (Rf < 0) R = 0; else R = (unsigned char)Rf;
		if (Gf > 255) G = 255; else if (Gf < 0) G = 0; else G = (unsigned char)Gf;
		if (Bf > 255) B = 255; else if (Bf < 0) B = 0; else B = (unsigned char)Bf;

		planeR[row * width + position] = R;
		planeG[row * width + position] = G;
		planeB[row * width + position] = B;
	}
	__global__ void KernelYUV420toRGB(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f C = (Npp32f)planeY[i];
		Npp32f D = (Npp32f)planeU[UVoffset] - 128;
		Npp32f E = (Npp32f)planeV[UVoffset] - 128;

		Npp32f Rf = round(C + 1.4075 * E);
		Npp32f Gf = round(C - 0.3455 * D - 0.7169 * E);
		Npp32f Bf = round(C + 1.7790 * D);

		unsigned char R, G, B;
		if (Rf > 255) R = 255; else if (Rf < 0) R = 0; else R = (unsigned char)Rf;
		if (Gf > 255) G = 255; else if (Gf < 0) G = 0; else G = (unsigned char)Gf;
		if (Bf > 255) B = 255; else if (Bf < 0) B = 0; else B = (unsigned char)Bf;

		planeR[row * width + position] = R;
		planeG[row * width + position] = G;
		planeB[row * width + position] = B;
	}
	__global__ void KernelRGB2YUV(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];

		Npp32f Y = round(R * .299000 + G * .587000 + B * .114000);
		Npp32f U = round(R * -.168736 + G * -.331264 + B * .500000 + 128);
		Npp32f V = round(R * .500000 + G * -.418688 + B * -.081312 + 128);
		planeY[row * Ypitch + position] = (unsigned char)Y;
		planeU[row * Ypitch + position] = (unsigned char)U;
		planeV[row * Ypitch + position] = (unsigned char)V;
	}
	__global__ void KernelUVShrink(unsigned char* planeI, unsigned char* planeO, int width, int height, int pitchI, int pitchO, int size)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= size) return; //if i greater than pixels number / 4 (should process only blocks of 4 pixels) size should be width * height / 4

		int blocksInRow = width / 2;
		int row = i / blocksInRow;

		Npp32f A, B, C, D;
		int position = i % blocksInRow;
		A = planeI[row * pitchI * 2 + position * 2];
		B = planeI[row * pitchI * 2 + position * 2 + 1];
		C = planeI[row * pitchI * 2 + pitchI + position * 2];
		D = planeI[row * pitchI * 2 + pitchI + position * 2 + 1];
		Npp32f E = (A + B + C + D) / 4;
		planeO[row * pitchO + position] = (unsigned char)E;
	}
	
	__global__ void KernelYUV420toRGB10(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f C = (Npp32f)planeY[i];
		Npp32f D = (Npp32f)planeU[UVoffset] - 512;
		Npp32f E = (Npp32f)planeV[UVoffset] - 512;

		Npp32f Rf = round(C + 1.4075 * E);
		Npp32f Gf = round(C - 0.3455 * D - 0.7169 * E);
		Npp32f Bf = round(C + 1.7790 * D);

		unsigned short R, G, B;
		if (Rf > 1023) R = 1023; else if (Rf < 0) R = 0; else R = (unsigned short)Rf;
		if (Gf > 1023) G = 1023; else if (Gf < 0) G = 0; else G = (unsigned short)Gf;
		if (Bf > 1023) B = 1023; else if (Bf < 0) B = 0; else B = (unsigned short)Bf;

		planeR[row * width + position] = R;
		planeG[row * width + position] = G;
		planeB[row * width + position] = B;
	}
	__global__ void KernelYUV444toRGB10(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f C = (Npp32f)planeY[i];
		Npp32f D = (Npp32f)planeU[i] - 512;
		Npp32f E = (Npp32f)planeV[i] - 512;

		Npp32f Rf = round(C + 1.4075 * E);
		Npp32f Gf = round(C - 0.3455 * D - 0.7169 * E);
		Npp32f Bf = round(C + 1.7790 * D);

		unsigned short R, G, B;
		if (Rf > 1023) R = 1023; else if (Rf < 0) R = 0; else R = (unsigned short)Rf;
		if (Gf > 1023) G = 1023; else if (Gf < 0) G = 0; else G = (unsigned short)Gf;
		if (Bf > 1023) B = 1023; else if (Bf < 0) B = 0; else B = (unsigned short)Bf;

		planeR[row * width + position] = R;
		planeG[row * width + position] = G;
		planeB[row * width + position] = B;
	}
	__global__ void KernelRGB2YUV10(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];

		Npp32f Y = round(R * .299000 + G * .587000 + B * .114000);
		Npp32f U = round(R * -.168736 + G * -.331264 + B * .500000 + 512);
		Npp32f V = round(R * .500000 + G * -.418688 + B * -.081312 + 512);
		planeY[row * Ypitch + position] = (unsigned short)Y;
		planeU[row * Ypitch + position] = (unsigned short)U;
		planeV[row * Ypitch + position] = (unsigned short)V;
	}
	__global__ void KernelUVShrink10(unsigned short* planeI, unsigned short* planeO, int width, int height, int pitchI, int pitchO, int size)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= size) return; //if i greater than pixels number / 4 (should process only blocks of 4 pixels) size should be width * height / 4

		int blocksInRow = width / 2;
		int row = i / blocksInRow;

		Npp32f A, B, C, D;
		int position = i % blocksInRow;
		A = planeI[row * pitchI * 2 + position * 2];
		B = planeI[row * pitchI * 2 + position * 2 + 1];
		C = planeI[row * pitchI * 2 + pitchI + position * 2];
		D = planeI[row * pitchI * 2 + pitchI + position * 2 + 1];
		Npp32f E = (A + B + C + D) / 4;
		planeO[row * pitchO + position] = (unsigned short)E;
	}

//Parallel sum functions

	__global__ void reduceBlacks(unsigned char* input, unsigned int* output, int length) {
		extern __shared__ unsigned int sdata[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdata[tid] = 0;
		else
		{
			if (input[i] == 0)
				sdata[tid] = 1;
			else sdata[tid] = 0;
		}

		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}
	__global__ void reduceBlacksShorts(unsigned short* input, unsigned int* output, int length) {
		extern __shared__ unsigned int sdata[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdata[tid] = 0;
		else
		{
			if (input[i] == 0)
				sdata[tid] = 1;
			else sdata[tid] = 0;
		}

		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}
	__global__ void reduceChar(unsigned char * input, unsigned int * output, int length) {
		extern __shared__ unsigned int sdata[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdata[tid] = 0;
		else
		sdata[tid] = input[i];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}
	__global__ void reduceShort(unsigned short* input, unsigned int* output, int length) {
		extern __shared__ unsigned int sdata[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdata[tid] = 0;
		else
			sdata[tid] = input[i];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}
	__global__ void reduceFloat(Npp32f* input, Npp64f* output, int length) {
		extern __shared__ Npp64f sdatafloat[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdatafloat[tid] = 0;
		else
			sdatafloat[tid] = input[i];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdatafloat[tid] += sdatafloat[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdatafloat[0];
	}
	__global__ void reduceFloat(Npp64f* input, Npp64f* output, int length) {
		extern __shared__ Npp64f sdatafloat[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdatafloat[tid] = 0;
		else
			sdatafloat[tid] = input[i];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdatafloat[tid] += sdatafloat[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdatafloat[0];
	}
	__global__ void reduceInt(unsigned int * input, unsigned int * output, int length) {
		extern __shared__ unsigned int sdata[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdata[tid] = 0;
		else		
			sdata[tid] = input[i];
		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid == 0) output[blockIdx.x] = sdata[0];
	}
	void CudaSumNV(unsigned char* buf, int length, unsigned long long* result, unsigned int maxthreads) {
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		unsigned int* reduceout = 0;
		int resultblocks = blocks / maxthreads;
		if (blocks % maxthreads > 0) resultblocks += 1;
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * 4); //change size
		reduceChar <<<blocks, maxthreads, maxthreads * 4 >>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reduceout, blocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			unsigned int* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * 4);
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, resultblocks * maxthreads * 4);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}
	void CudaSumNV(Npp32f* buf, int length, Npp64f* result, unsigned int maxthreads) {
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		Npp64f* reduceout = 0;
		int resultblocks = blocks / maxthreads;
		if (blocks % maxthreads > 0) resultblocks += 1;
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * 8); //change size
		reduceFloat <<<blocks, maxthreads, maxthreads * 8>>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			Npp64f* tosum = new Npp64f[blocks];
			cudaMemcpy(tosum, reduceout, blocks * 8, cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			Npp64f* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * 8);
			reduceFloat<<<resultblocks, maxthreads, maxthreads * 8>>> (reduceout, reducelast, resultblocks * maxthreads * 8);
			cudaFree(reduceout);
			Npp64f* tosum = new Npp64f[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 8, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}
	void CudaSumNV(unsigned short* buf, int length, unsigned long long* result, unsigned int maxthreads) {
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		unsigned int* reduceout = 0;
		int resultblocks = blocks / maxthreads;
		if (blocks % maxthreads > 0) resultblocks += 1;
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * 4); //change size
		reduceShort <<<blocks, maxthreads, maxthreads * 4 >>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reduceout, blocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			unsigned int* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * 4);
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, resultblocks * maxthreads * 4);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}
	void CudaCountBlacksNV(unsigned char* buf, int length, unsigned long long* result, unsigned int maxthreads) {
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		unsigned int* reduceout = 0;
		int resultblocks = blocks / maxthreads;
		if (blocks % maxthreads > 0) resultblocks += 1;
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * 4); //change size
		reduceBlacks <<<blocks, maxthreads, maxthreads * 4 >>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reduceout, blocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			unsigned int* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * 4);
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, length);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}
	void CudaCountBlacksNV(unsigned short* buf, int length, unsigned long long* result, unsigned int maxthreads) {
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		unsigned int* reduceout = 0;
		int resultblocks = blocks / maxthreads;
		if (blocks % maxthreads > 0) resultblocks += 1;
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * 4); //change size
		reduceBlacksShorts<<<blocks, maxthreads, maxthreads * 4 >>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reduceout, blocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			unsigned int* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * 4);
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, length);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}
//Other functions

	__global__ void KernelTV2PC(unsigned char* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			unsigned char c = buf[i];
			if (c < 16) c = 0; //lowerst
			else
				if (c > 235) c = 255; //max
				else if (c == 125) c = 128; //middle
				else
				{ //16-235
					Npp32f cf = c;
					cf = (cf - 16) / 220 * 255 + 0.5;
					c = (unsigned char)cf;
				}
			buf[i] = c;
		}
	}
	__global__ void KernelTV2PC(unsigned short* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			unsigned short c = buf[i];
			if (c < 64) c = 0; //lowerst
			else
				if (c > 943) c = 1023; //max
				else if (c == 439) c = 512; //middle
				else
				{ //16-235
					Npp32f cf = c;
					cf = (cf - 64) / 880 * 1023 + 0.5;
					c = (unsigned short)cf;
				}
			buf[i] = c;
		}
	}
	int blocks(int from, int threads) {
		int result = from / threads;
		if (from % threads > 0) result += 1;
		return result;
	}

//Main functions

	void CudaNeutralizeRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		int hsvsize = length * 4;
		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlightF_pegtop <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlightF_illusionshu <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlightF_W3C <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB <<<rgbblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB <<<rgbblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);

		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaNeutralizeRGBwithLight(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack)
	{
		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
				}
			}
		}

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaBoostSaturationRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int formula)
	{
		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		int hsvsize = length * 4;

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
		}
		}

		KernelRGB2HSV_S <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB <<<rgbblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}
	
	void CudaNeutralizeRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack) {

		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength / 4, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrLength);

		int hsvsize = length * 4;

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
		}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
			}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB <<<rgbblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB <<<rgbblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);

		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaNeutralizeRGB32withLight(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack)
	{
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength / 4, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrLength);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);

		KernelBGRtoRGB <<<bgrblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;
		
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
				}
			}
		}

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaBoostSaturationRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int formula)
	{
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength / 4, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrLength);

		int hsvsize = length * 4;

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
		}
		}

		KernelRGB2HSV_S <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB <<<rgbblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}

	void CudaNeutralizeYUV420byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		
		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		
		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		int length = planeYwidth * planeYheight;

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		int hsvsize = length * 4;
		
		KernelYUV420toRGB<<<Yblocks,threads>>>(planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V <<<Yblocks, threads>>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}
		
		if (type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlightF_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlightF_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlightF_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		
		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		KernelRGB2YUV<<<Yblocks,threads>>>(planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);
		
		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYwidth, planeUwidth, Ylength / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYwidth, planeVwidth, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaNeutralizeYUV420byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Ylength);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Ylength);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Ylength);

		KernelYUV420toRGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth);

		int length = planeYwidth * planeYheight;

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);
		
		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
				}
			}
		}

		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYwidth, planeUwidth, Ylength / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYwidth, planeVwidth, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaBoostSaturationYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula) {
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int length = planeYwidth * planeYheight;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Ylength);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Ylength);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Ylength);

		int hsvsize = Ylength * 4;

		KernelYUV420toRGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
		}
		}

		KernelRGB2HSV_S <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYwidth, planeUwidth, Ylength / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYwidth, planeVwidth, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}

	void CudaNeutralizeYUV444byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int length = planeYwidth * planeYheight;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);
		int Ublocks = blocks(Ulength, threads);
		int Vblocks = blocks(Vlength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		int hsvsize = length * 4;

		KernelYUV2RGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;


		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
		}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
			}
			}
		}
		if (type == 0 || type == 2)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaNeutralizeYUV444byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int length = planeYwidth * planeYheight;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);
		int Ublocks = blocks(Ulength, threads);
		int Vblocks = blocks(Vlength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		KernelYUV2RGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			}
			}
		}

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaBoostSaturationYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula) {
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int length = planeYwidth * planeYheight;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);
		int Ublocks = blocks(Ulength, threads);
		int Vblocks = blocks(Vlength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		int hsvsize = length;

		KernelYUV2RGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, length);
		}
		}

		KernelRGB2HSV_S <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}

	void CudaTV2PCYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads) {
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int length = planeYwidth * planeYheight;
		
		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Ylength);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Ylength);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Ylength);

		KernelYUV420toRGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth);

		int rgbblocks = blocks(length, threads);

		KernelTV2PC <<<rgbblocks, threads>>> (planeRnv, length);
		KernelTV2PC <<<rgbblocks, threads>>> (planeGnv, length);
		KernelTV2PC <<<rgbblocks, threads>>> (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYwidth, planeUwidth, Ylength / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYwidth, planeVwidth, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaTV2PCYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads) {
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int length = planeYwidth * planeYheight;

		int Ylength = planeYwidth * planeYheight;
		int Ulength = planeUwidth * planeUheight;
		int Vlength = planeVwidth * planeVheight;

		int Yblocks = blocks(Ylength, threads);
		int Ublocks = blocks(Ulength, threads);
		int Vblocks = blocks(Vlength, threads);

		cudaMalloc(&planeYnv, Ylength);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ulength);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vlength);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		KernelYUV2RGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		int rgbblocks = blocks(length, threads);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length);

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaTV2PCRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {
		const char* error = cudaerror();
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength / 4, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrLength);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length);

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaTV2PCRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}

	void CudaGrayscaleRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {

		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, rgbblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, rgbblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, rgbblocks * threads);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrblocks * threads);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		unsigned char* planeYnv; cudaMalloc(&planeYnv, rgbblocks * threads);
		unsigned char* planeUnv; cudaMalloc(&planeUnv, rgbblocks * threads);
		unsigned char* planeVnv; cudaMalloc(&planeVnv, rgbblocks * threads);

		KernelRGB2YUV <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);
		cudaMemset(planeUnv, 128, length);
		cudaMemset(planeVnv, 128, length);
		KernelYUV2RGB <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);
		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);
		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);

		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaGrayscaleRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		unsigned char* planeYnv; cudaMalloc(&planeYnv, length);
		unsigned char* planeUnv; cudaMalloc(&planeUnv, length);
		unsigned char* planeVnv; cudaMalloc(&planeVnv, length);

		KernelRGB2YUV <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);
		cudaMemset(planeUnv, 128, length);
		cudaMemset(planeVnv, 128, length);
		KernelYUV2RGB <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);

		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}

//Main 10bit

	void CudaNeutralizeYUV420byRGBwithLight10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		KernelYUV420toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;

		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}

		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 1023 - Rsum;
		Gsum = 1023 - Gsum;
		Bsum = 1023 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			}
			}
		}

		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);
		
		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);
		
		int shrinkblocks = blocks(length / 4, threads);

		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2, length / 4);
		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeVwidth / 2, length / 4);
		
		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaNeutralizeYUV420byRGB10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		int hsvsize = length * 4;

		KernelYUV420toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack==0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 1023 - Rsum;
		Gsum = 1023 - Gsum;
		Bsum = 1023 - Bsum;

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
		}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
			}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2, Ylength / 4);
		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeVwidth / 2, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaTV2PCYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads) {
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		KernelYUV420toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		int rgbblocks = blocks(length, threads);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2, Ylength / 4);
		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeVwidth / 2, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaBoostSaturationYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula) {
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		int hsvsize = length * 4;

		KernelYUV420toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			}
		}

		KernelRGB2HSV_S10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2, Ylength / 4);
		KernelUVShrink10 <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeVwidth / 2, Ylength / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}

	void CudaNeutralizeYUV444byRGBwithLight10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		KernelYUV444toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;

		if (skipblack == 0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}

		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 1023 - Rsum;
		Gsum = 1023 - Gsum;
		Bsum = 1023 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			}
			}
		}

		
		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaNeutralizeYUV444byRGB10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack)
	{
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		int hsvsize = length * 4;

		KernelYUV444toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack == 0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 1023 - Rsum;
		Gsum = 1023 - Gsum;
		Bsum = 1023 - Bsum;

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
		}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
			}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}

		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaTV2PCYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads) {
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		KernelYUV444toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		int rgbblocks = blocks(length, threads);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length);

		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaBoostSaturationYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula) {
		unsigned short* planeYnv;
		unsigned short* planeUnv;
		unsigned short* planeVnv;

		int length = planeYwidth / 2 * planeYheight;

		int Ylength = planeYwidth / 2 * planeYheight;
		int Ulength = planeUwidth / 2 * planeUheight;
		int Vlength = planeVwidth / 2 * planeVheight;

		int Yblocks = blocks(Ylength, threads);

		cudaMalloc(&planeYnv, Ylength * 2);
		cudaMemcpy2D(planeYnv, planeYwidth, planeY, planeYpitch, planeYwidth, planeYheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeUnv, Ulength * 2);
		cudaMemcpy2D(planeUnv, planeUwidth, planeU, planeUpitch, planeUwidth, planeUheight, cudaMemcpyHostToDevice);
		cudaMalloc(&planeVnv, Vlength * 2);
		cudaMemcpy2D(planeVnv, planeVwidth, planeV, planeVpitch, planeVwidth, planeVheight, cudaMemcpyHostToDevice);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		int hsvsize = length * 4;

		KernelYUV444toRGB10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
		}
		}

		KernelRGB2HSV_S10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		KernelRGB2YUV10 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYpitch / 2);

		cudaMemcpy2D(planeY, planeYpitch, planeYnv, planeYwidth, planeYwidth, planeYheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeU, planeUpitch, planeUnv, planeUwidth, planeUwidth, planeUheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeV, planeVpitch, planeVnv, planeVwidth, planeVwidth, planeVheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}

	void CudaNeutralizeRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack) {

		int length = planeheight * planewidth / 2;

		int rgbblocks = blocks(length, threads);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		int hsvsize = length * 4;

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, hsvsize);
			cudaMalloc(&planeHSVo_Snv, hsvsize);
			cudaMalloc(&planeHSV_Vnv, hsvsize);
			KernelRGB2HSV_HS10 <<<rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V10 <<<rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack == 0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 1023 - Rsum;
		Gsum = 1023 - Gsum;
		Bsum = 1023 - Bsum;

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlightFC_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
		}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS10 << <rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V10 << <rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop << <rgbblocks, threads >> > (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu << <rgbblocks, threads >> > (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C << <rgbblocks, threads >> > (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
			}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB10 <<<rgbblocks, threads >> > (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB10 << <rgbblocks, threads >> > (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
		}

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);

		if (type == 1)
		{
			cudaFree(planeHSVo_Hnv);
			cudaFree(planeHSVo_Snv);
			cudaFree(planeHSV_Vnv);
		}
		else if (type == 0 || type == 2)
		{
			cudaFree(planeHSVo_Vnv);
			cudaFree(planeHSV_Hnv);
			cudaFree(planeHSV_Snv);
		}
	}
	void CudaNeutralizeRGBwithLight10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack)
	{
		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, length, &Rsum, threads);
		CudaSumNV(planeGnv, length, &Gsum, threads);
		CudaSumNV(planeBnv, length, &Bsum, threads);

		int rlength = length, glength = length, blength = length;
		unsigned long long rblacks = 0, gblacks = 0, bblacks = 0;
		if (skipblack == 0) {
			CudaCountBlacksNV(planeRnv, length, &rblacks, threads);
			CudaCountBlacksNV(planeGnv, length, &gblacks, threads);
			CudaCountBlacksNV(planeBnv, length, &bblacks, threads);
			if (rblacks < length) rlength -= (int)rblacks;
			if (gblacks < length) glength -= (int)gblacks;
			if (bblacks < length) blength -= (int)bblacks;
		}
		Rsum /= rlength;
		Gsum /= glength;
		Bsum /= blength;

		Rsum = 1023 - Rsum;
		Gsum = 1023 - Gsum;
		Bsum = 1023 - Bsum;

		if (type == 0 || type == 1 || type == 3) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop << <rgbblocks, threads >> > (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_pegtop << <rgbblocks, threads >> > (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_pegtop << <rgbblocks, threads >> > (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu << <rgbblocks, threads >> > (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_illusionshu << <rgbblocks, threads >> > (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_illusionshu << <rgbblocks, threads >> > (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C << <rgbblocks, threads >> > (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlightFC_W3C << <rgbblocks, threads >> > (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlightFC_W3C << <rgbblocks, threads >> > (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop << <rgbblocks, threads >> > (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_pegtop << <rgbblocks, threads >> > (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_pegtop << <rgbblocks, threads >> > (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu << <rgbblocks, threads >> > (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_illusionshu << <rgbblocks, threads >> > (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_illusionshu << <rgbblocks, threads >> > (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C << <rgbblocks, threads >> > (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_W3C << <rgbblocks, threads >> > (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_W3C << <rgbblocks, threads >> > (planeBnv, planeBnv, 0, 1023, length);
			}
			}
		}

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaBoostSaturationRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int formula)
	{
		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		int hsvsize = length * 4;

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV << <rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop << <rgbblocks, threads >> > (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_pegtop << <rgbblocks, threads >> > (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_pegtop << <rgbblocks, threads >> > (planeBnv, planeBnv, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu << <rgbblocks, threads >> > (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_illusionshu << <rgbblocks, threads >> > (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_illusionshu << <rgbblocks, threads >> > (planeBnv, planeBnv, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C << <rgbblocks, threads >> > (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_W3C << <rgbblocks, threads >> > (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_W3C << <rgbblocks, threads >> > (planeBnv, planeBnv, 0, 1023, length);
		}
		}

		KernelRGB2HSV_S << <rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB << <rgbblocks, threads >> > (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}
	void CudaTV2PCRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelTV2PC << <rgbblocks, threads >> > (planeRnv, length);
		KernelTV2PC << <rgbblocks, threads >> > (planeGnv, length);
		KernelTV2PC << <rgbblocks, threads >> > (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaGrayscaleRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		unsigned short* planeYnv; cudaMalloc(&planeYnv, length * 2);
		unsigned short* planeUnv; cudaMalloc(&planeUnv, length * 2);
		unsigned short* planeVnv; cudaMalloc(&planeVnv, length * 2);

		KernelRGB2YUV10 << <rgbblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);

		short* gray = (short*)malloc(planewidth * planeheight);
		for (int i = 0; i != length; i++) gray[i] = 512;
		cudaMemcpy(planeUnv, gray, length * 2, cudaMemcpyHostToDevice);
		cudaMemcpy(planeVnv, gray, length * 2, cudaMemcpyHostToDevice);
		KernelYUV444toRGB10 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2, planewidth / 2);
		free(gray);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);

		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}