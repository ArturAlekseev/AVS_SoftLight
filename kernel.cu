#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <npp.h>

//const char* cudaerror() {
//	const char* error = cudaGetErrorString(cudaGetLastError());
//	return error;
//}

//Softlight functions

	//byte array with float
	__global__ void KernelSoftlight_W3C(unsigned char* s, Npp32f b, int clampmin, int clampmax, int length)
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
	__global__ void KernelSoftlight_pegtop(unsigned char* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a) * 255;
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_illusionshu(unsigned char* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 255;
			float r = powf(a, powf(2, (2 * (0.5 - b)))) * 255; //can be fastened
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = (__int8)(r + 0.5);
		}
	}

	//float array with float
	__global__ void KernelSoftlight_W3C(Npp32f* s, Npp32f b, Npp32f clampmin, Npp32f clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f a = s[i];
			Npp32f r;
			Npp32f g;
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
	__global__ void KernelSoftlight_pegtop(Npp32f* s, Npp32f b, Npp32f clampmin, Npp32f clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f a = s[i];
			Npp32f r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a);
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = r;
		}
	}
	__global__ void KernelSoftlight_illusionshu(Npp32f* s, Npp32f b, Npp32f clampmin, Npp32f clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f a = s[i];
			Npp32f r = powf(a, powf(2, (2 * (0.5 - b)))); //can be fastened
			if (r <= clampmin) s[i] = clampmin;
			else if (r >= clampmax) s[i] = clampmax; else s[i] = r;
		}
	}

	//float array with float array
	__global__ void KernelSoftlight_W3C(Npp32f* s, Npp32f* d, Npp32f clampmin, Npp32f clampmax, int length)
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
	__global__ void KernelSoftlight_pegtop(Npp32f* s, Npp32f* d, Npp32f clampmin, Npp32f clampmax, int length)
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
	__global__ void KernelSoftlight_illusionshu(Npp32f* s, Npp32f* d, Npp32f clampmin, Npp32f clampmax, int length)
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

	//byte array with byte array
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
	//short array with float
	__global__ void KernelSoftlight_W3C(unsigned short* s, Npp32f b, int clampmin, int clampmax, int length)
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
	__global__ void KernelSoftlight_pegtop(unsigned short* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float r = ((1.0 - 2.0 * b) * powf(a, 2) + 2 * b * a) * 1023;
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}
	__global__ void KernelSoftlight_illusionshu(unsigned short* s, Npp32f b, int clampmin, int clampmax, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			float a = (float)s[i] / 1023;
			float r = powf(a, powf(2, (2 * (0.5 - b)))) * 1023; //can be fastened
			if (r <= clampmin) s[i] = (unsigned short)clampmin;
			else if (r >= clampmax) s[i] = (unsigned short)clampmax; else s[i] = (unsigned short)(r + 0.5);
		}
	}

	//short array with short array
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
	__global__ void KernelRGB2HSV_HS(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* H, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
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
	__global__ void KernelHSV2RGB(Npp32f* H, Npp32f* S, Npp32f* V, Npp32f* R, Npp32f* G, Npp32f* B, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		Npp32f h, s, v;
		h = H[i];
		s = S[i];
		v = V[i];
		int hi = (int)(floorf(h / 60)) % 6;
		Npp32f f = h / 60 - floorf(h / 60);

		Npp32f vi = v;
		Npp32f p = v * ((Npp32f)1 - s);
		Npp32f q = v * ((Npp32f)1 - f * s);
		Npp32f t = v * ((Npp32f)1 - ((Npp32f)1 - f) * s);
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
	__global__ void KernelRGB2HSV_V(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
		Npp32f v;
		v = fmaxf(double_r, fmaxf(double_g, double_b));
		V[i] = v;
	}
	__global__ void KernelRGB2HSV_HV(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* H, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
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
	__global__ void KernelRGB2HSV_S(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
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
	__global__ void KernelRGB2HSV_HS10(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* H, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
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
	__global__ void KernelRGB2HSV_V10(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
		Npp32f v;
		v = fmaxf(double_r, fmaxf(double_g, double_b));
		V[i] = v;
	}
	__global__ void KernelHSV2RGB10(Npp32f* H, Npp32f* S, Npp32f* V, Npp32f* R, Npp32f* G, Npp32f* B, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		Npp32f h, s, v;
		h = H[i];
		s = S[i];
		v = V[i];
		int hi = (int)(floorf(h / 60)) % 6;
		Npp32f f = h / 60 - floorf(h / 60);

		Npp32f vi = v;
		Npp32f p = v * (1.0 - s);
		Npp32f q = v * (1.0 - f * s);
		Npp32f t = v * (1.0 - (1.0 - f) * s);
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
	__global__ void KernelRGB2HSV_HV10(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* H, Npp32f* V, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
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
	__global__ void KernelRGB2HSV_S10(Npp32f* R, Npp32f* G, Npp32f* B, Npp32f* S, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= length) return;
		// Convert RGB to a 0.0 to 1.0 range.
		Npp32f double_r = R[i];
		Npp32f double_g = G[i];
		Npp32f double_b = B[i];
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
			if (max == 0) s = 0; else s = 1.0 - (1.0 * min / max);
		}
		S[i] = s;
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
//8 bit (Formulas are same - just coefficients are different. Recs are separated in different functions for speedup (not to check selected rec in each CUDA call)
	__global__ void KernelYUV2RGBRec601(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		
		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i] / 255.0;
		Npp32f U = (Npp32f)planeU[i] / 255.0 - .5;
		Npp32f V = (Npp32f)planeV[i] / 255.0 - .5;

		Npp32f Rf = Y + 1.402 * V;
		Npp32f Gf = Y - 0.114 * 1.772 / 0.587 * U - 0.299 * 1.402 / 0.587 * V;
		Npp32f Bf = Y + 1.772 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelYUV2RGBRec601(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i];
		Npp32f U = (Npp32f)planeU[i] - .5;
		Npp32f V = (Npp32f)planeV[i] - .5;

		Npp32f Rf = round((Y + 1.402 * V) * 255);
		Npp32f Gf = round((Y - 0.114 * 1.772 / 0.587 * U - 0.299 * 1.402 / 0.587 * V) * 255);
		Npp32f Bf = round((Y + 1.772 * U) * 255);

		if (Rf < 0) Rf = 0; else if (Rf > 255) Rf = 255;
		if (Gf < 0) Gf = 0; else if (Gf > 255) Gf = 255;
		if (Bf < 0) Bf = 0; else if (Bf > 255) Bf = 255;

		planeR[row * width + position] = (unsigned char)Rf;
		planeG[row * width + position] = (unsigned char)Gf;
		planeB[row * width + position] = (unsigned char)Bf;
	}
	__global__ void KernelYUV420toRGBRec601(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f Y = (Npp32f)planeY[i] / 255.0;
		Npp32f U = (Npp32f)planeU[UVoffset] / 255.0 - .5;
		Npp32f V = (Npp32f)planeV[UVoffset] / 255.0 - .5;

		Npp32f Rf = Y + 1.402 * V;
		Npp32f Gf = Y - 0.114 * 1.772 / 0.587 * U - 0.299 * 1.402 / 0.587 * V;
		Npp32f Bf = Y + 1.772 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelRGB2YUVRec601(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		Npp32f yf = R * .299 + G * .587 + B * .114;
		Npp32f Y = round(yf * 255); //Y
		Npp32f U = round(((B - yf) / 1.772 + .5) * 255); //Cb
		Npp32f V = round(((R - yf) / 1.402 + .5) * 255); //Cr

		if (Y < 0) Y = 0; else if (Y > 255) Y = 255;
		if (U < 0) U = 0; else if (U > 255) U = 255;
		if (V < 0) V = 0; else if (V > 255) V = 255;

		planeY[row * Ypitch + position] = (unsigned char)Y;
		planeU[row * Ypitch + position] = (unsigned char)U;
		planeV[row * Ypitch + position] = (unsigned char)V;
	}
	__global__ void KernelRGB2YUVRec601(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		R /= 255.0;
		G /= 255.0;
		B /= 255.0;

		Npp32f Y = R * .299 + G * .587 + B * .114; //Y
		Npp32f U = (B - Y) / 1.772 + .5; //Cb
		Npp32f V = (R - Y) / 1.402 + .5; //Cr

		planeY[row * Ypitch + position] = Y;
		planeU[row * Ypitch + position] = U;
		planeV[row * Ypitch + position] = V;
	}
	
	__global__ void KernelYUV2RGBRec709(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i] / 255.0;
		Npp32f U = (Npp32f)planeU[i] / 255.0 - .5;
		Npp32f V = (Npp32f)planeV[i] / 255.0 - .5;

		Npp32f Rf = Y + 1.5748 * V;
		Npp32f Gf = Y - 0.0722 * 1.8556 / 0.7152 * U - 0.2126 * 1.5748 / 0.7152 * V;
		Npp32f Bf = Y + 1.8556 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelYUV2RGBRec709(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i];
		Npp32f U = (Npp32f)planeU[i] - .5;
		Npp32f V = (Npp32f)planeV[i] - .5;

		Npp32f Rf = round((Y + 1.5748 * V) * 255);
		Npp32f Gf = round((Y - 0.0722 * 1.8556 / 0.7152 * U - 0.2126 * 1.5748 / 0.7152 * V) * 255);
		Npp32f Bf = round((Y + 1.8556 * U) * 255);

		if (Rf < 0) Rf = 0; else if (Rf > 255) Rf = 255;
		if (Gf < 0) Gf = 0; else if (Gf > 255) Gf = 255;
		if (Bf < 0) Bf = 0; else if (Bf > 255) Bf = 255;

		planeR[row * width + position] = (unsigned char)Rf;
		planeG[row * width + position] = (unsigned char)Gf;
		planeB[row * width + position] = (unsigned char)Bf;
	}
	__global__ void KernelYUV420toRGBRec709(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f Y = (Npp32f)planeY[i] / 255.0;
		Npp32f U = (Npp32f)planeU[UVoffset] / 255.0 - .5;
		Npp32f V = (Npp32f)planeV[UVoffset] / 255.0 - .5;

		Npp32f Rf = Y + 1.5748 * V;
		Npp32f Gf = Y - 0.0722 * 1.8556 / 0.7152 * U - 0.2126 * 1.5748 / 0.7152 * V;
		Npp32f Bf = Y + 1.8556 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelRGB2YUVRec709(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];

		Npp32f yf = 0.2126 * R + 0.7152 * G + 0.0722 * B;
		Npp32f Y = round(yf * 255);
		Npp32f U = round(((B - yf) / 1.8556 + 0.5) * 255);
		Npp32f V = round(((R - yf) / 1.5748 + 0.5) * 255);

		if (Y < 0) Y = 0; else if (Y > 255) Y = 255;
		if (U < 0) U = 0; else if (U > 255) U = 255;
		if (V < 0) V = 0; else if (V > 255) V = 255;

		planeY[row * Ypitch + position] = (unsigned char)Y;
		planeU[row * Ypitch + position] = (unsigned char)U;
		planeV[row * Ypitch + position] = (unsigned char)V;
	}
	__global__ void KernelRGB2YUVRec709(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		R /= 255.0;
		G /= 255.0;
		B /= 255.0;

		Npp32f Y = 0.2126 * R + 0.7152 * G + 0.0722 * B; //Y
		Npp32f U = (B - Y) / 1.8556 + .5; //Cb
		Npp32f V = (R - Y) / 1.5748 + .5; //Cr

		planeY[row * Ypitch + position] = Y;
		planeU[row * Ypitch + position] = U;
		planeV[row * Ypitch + position] = V;
	}

	__global__ void KernelYUV2RGBRec2020(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i] / 255.0;
		Npp32f U = (Npp32f)planeU[i] / 255.0 - .5;
		Npp32f V = (Npp32f)planeV[i] / 255.0 - .5;

		Npp32f Rf = Y + 1.4746 * V;
		Npp32f Gf = Y - 0.0593 * 1.8814 / 0.6780 * U - 0.2627 * 1.4746 / 0.6780 * V;
		Npp32f Bf = Y + 1.8814 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelYUV2RGBRec2020(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i];
		Npp32f U = (Npp32f)planeU[i] - .5;
		Npp32f V = (Npp32f)planeV[i] - .5;

		Npp32f Rf = round((Y + 1.4746 * V) * 255);
		Npp32f Gf = round((Y - 0.0593 * 1.8814 / 0.6780 * U - 0.2627 * 1.4746 / 0.6780 * V) * 255);
		Npp32f Bf = round((Y + 1.8814 * U) * 255);

		if (Rf < 0) Rf = 0; else if (Rf > 255) Rf = 255;
		if (Gf < 0) Gf = 0; else if (Gf > 255) Gf = 255;
		if (Bf < 0) Bf = 0; else if (Bf > 255) Bf = 255;

		planeR[row * width + position] = (unsigned char)Rf;
		planeG[row * width + position] = (unsigned char)Gf;
		planeB[row * width + position] = (unsigned char)Bf;
	}
	__global__ void KernelYUV420toRGBRec2020(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f Y = (Npp32f)planeY[i] / 255.0;
		Npp32f U = (Npp32f)planeU[UVoffset] / 255.0 - .5;
		Npp32f V = (Npp32f)planeV[UVoffset] / 255.0 - .5;

		Npp32f Rf = Y + 1.4746 * V;
		Npp32f Gf = Y - 0.0593 * 1.8814 / 0.6780 * U - 0.2627 * 1.4746 / 0.6780 * V;
		Npp32f Bf = Y + 1.8814 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelRGB2YUVRec2020(unsigned char* planeY, unsigned char* planeU, unsigned char* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];

		Npp32f yf = 0.2627 * R + 0.6780 * G + 0.0593 * B;
		Npp32f Y = round(yf * 255);
		Npp32f U = round(((B - yf) / 1.8814 + 0.5) * 255);
		Npp32f V = round(((R - yf) / 1.4746 + 0.5) * 255);

		if (Y < 0) Y = 0; else if (Y > 255) Y = 255;
		if (U < 0) U = 0; else if (U > 255) U = 255;
		if (V < 0) V = 0; else if (V > 255) V = 255;

		planeY[row * Ypitch + position] = (unsigned char)Y;
		planeU[row * Ypitch + position] = (unsigned char)U;
		planeV[row * Ypitch + position] = (unsigned char)V;
	}
	__global__ void KernelRGB2YUVRec2020(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		R /= 255.0;
		G /= 255.0;
		B /= 255.0;

		Npp32f Y = 0.2627 * R + 0.6780 * G + 0.0593 * B; //Y
		Npp32f U = (B - Y) / 1.8814 + .5; //Cb
		Npp32f V = (R - Y) / 1.4746 + .5; //Cr

		planeY[row * Ypitch + position] = Y;
		planeU[row * Ypitch + position] = U;
		planeV[row * Ypitch + position] = V;
	}

//10 bit
	__global__ void KernelYUV2RGB10Rec601(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i] / 1023.0;
		Npp32f U = (Npp32f)planeU[i] / 1023.0 - .5;
		Npp32f V = (Npp32f)planeV[i] / 1023.0 - .5;

		Npp32f Rf = Y + 1.402 * V;
		Npp32f Gf = Y - 0.114 * 1.772 / 0.587 * U - 0.299 * 1.402 / 0.587 * V;
		Npp32f Bf = Y + 1.772 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelYUV2RGB10Rec601(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i];
		Npp32f U = (Npp32f)planeU[i] - .5;
		Npp32f V = (Npp32f)planeV[i] - .5;

		Npp32f Rf = round((Y + 1.402 * V) * 1023);
		Npp32f Gf = round((Y - 0.114 * 1.772 / 0.587 * U - 0.299 * 1.402 / 0.587 * V) * 1023);
		Npp32f Bf = round((Y + 1.772 * U) * 1023);

		if (Rf < 0) Rf = 0; else if (Rf > 1023) Rf = 1023;
		if (Gf < 0) Gf = 0; else if (Gf > 1023) Gf = 1023;
		if (Bf < 0) Bf = 0; else if (Bf > 1023) Bf = 1023;

		planeR[row * width + position] = (unsigned short)Rf;
		planeG[row * width + position] = (unsigned short)Gf;
		planeB[row * width + position] = (unsigned short)Bf;
	}
	__global__ void KernelYUV420toRGB10Rec601(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f Y = (Npp32f)planeY[i] / 1023.0;
		Npp32f U = (Npp32f)planeU[UVoffset] / 1023.0 - .5;
		Npp32f V = (Npp32f)planeV[UVoffset] / 1023.0 - .5;

		Npp32f Rf = Y + 1.402 * V;
		Npp32f Gf = Y - 0.114 * 1.772 / 0.587 * U - 0.299 * 1.402 / 0.587 * V;
		Npp32f Bf = Y + 1.772 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelRGB2YUV10Rec601(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		Npp32f yf = R * .299 + G * .587 + B * .114;
		Npp32f Y = round(yf * 1023); //Y
		Npp32f U = round(((B - yf) / 1.772 + .5) * 1023); //Cb
		Npp32f V = round(((R - yf) / 1.402 + .5) * 1023); //Cr

		if (Y < 0) Y = 0; else if (Y > 1023) Y = 1023;
		if (U < 0) U = 0; else if (U > 1023) U = 1023;
		if (V < 0) V = 0; else if (V > 1023) V = 1023;

		planeY[row * Ypitch + position] = (unsigned short)Y;
		planeU[row * Ypitch + position] = (unsigned short)U;
		planeV[row * Ypitch + position] = (unsigned short)V;
	}
	__global__ void KernelRGB2YUV10Rec601(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		R /= 1023;
		G /= 1023.0;
		B /= 1023.0;

		Npp32f Y = R * .299 + G * .587 + B * .114; //Y
		Npp32f U = (B - Y) / 1.772 + .5; //Cb
		Npp32f V = (R - Y) / 1.402 + .5; //Cr

		planeY[row * Ypitch + position] = Y;
		planeU[row * Ypitch + position] = U;
		planeV[row * Ypitch + position] = V;
	}

	__global__ void KernelYUV2RGB10Rec709(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i] / 1023.0;
		Npp32f U = (Npp32f)planeU[i] / 1023.0 - .5;
		Npp32f V = (Npp32f)planeV[i] / 1023.0 - .5;

		Npp32f Rf = Y + 1.5748 * V;
		Npp32f Gf = Y - 0.0722 * 1.8556 / 0.7152 * U - 0.2126 * 1.5748 / 0.7152 * V;
		Npp32f Bf = Y + 1.8556 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelYUV2RGB10Rec709(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i];
		Npp32f U = (Npp32f)planeU[i] - .5;
		Npp32f V = (Npp32f)planeV[i] - .5;

		Npp32f Rf = round((Y + 1.5748 * V) * 1023);
		Npp32f Gf = round((Y - 0.0722 * 1.8556 / 0.7152 * U - 0.2126 * 1.5748 / 0.7152 * V) * 1023);
		Npp32f Bf = round((Y + 1.8556 * U) * 1023);

		if (Rf < 0) Rf = 0; else if (Rf > 1023) Rf = 1023;
		if (Gf < 0) Gf = 0; else if (Gf > 1023) Gf = 1023;
		if (Bf < 0) Bf = 0; else if (Bf > 1023) Bf = 1023;

		planeR[row * width + position] = (unsigned short)Rf;
		planeG[row * width + position] = (unsigned short)Gf;
		planeB[row * width + position] = (unsigned short)Bf;
	}
	__global__ void KernelYUV420toRGB10Rec709(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f Y = (Npp32f)planeY[i] / 1023.0;
		Npp32f U = (Npp32f)planeU[UVoffset] / 1023.0 - .5;
		Npp32f V = (Npp32f)planeV[UVoffset] / 1023.0 - .5;

		Npp32f Rf = Y + 1.5748 * V;
		Npp32f Gf = Y - 0.0722 * 1.8556 / 0.7152 * U - 0.2126 * 1.5748 / 0.7152 * V;
		Npp32f Bf = Y + 1.8556 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelRGB2YUV10Rec709(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];

		Npp32f yf = 0.2126 * R + 0.7152 * G + 0.0722 * B;
		Npp32f Y = round(yf * 1023);
		Npp32f U = round(((B - yf) / 1.8556 + 0.5) * 1023);
		Npp32f V = round(((R - yf) / 1.5748 + 0.5) * 1023);

		if (Y < 0) Y = 0; else if (Y > 1023) Y = 1023;
		if (U < 0) U = 0; else if (U > 1023) U = 1023;
		if (V < 0) V = 0; else if (V > 1023) V = 1023;

		planeY[row * Ypitch + position] = (unsigned short)Y;
		planeU[row * Ypitch + position] = (unsigned short)U;
		planeV[row * Ypitch + position] = (unsigned short)V;
	}
	__global__ void KernelRGB2YUV10Rec709(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		R /= 1023.0;
		G /= 1023.0;
		B /= 1023.0;

		Npp32f Y = 0.2126 * R + 0.7152 * G + 0.0722 * B; //Y
		Npp32f U = (B - Y) / 1.8556 + .5; //Cb
		Npp32f V = (R - Y) / 1.5748 + .5; //Cr

		planeY[row * Ypitch + position] = Y;
		planeU[row * Ypitch + position] = U;
		planeV[row * Ypitch + position] = V;
	}

	__global__ void KernelYUV2RGB10Rec2020(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i] / 1023.0;
		Npp32f U = (Npp32f)planeU[i] / 1023.0 - .5;
		Npp32f V = (Npp32f)planeV[i] / 1023.0 - .5;

		Npp32f Rf = Y + 1.4746 * V;
		Npp32f Gf = Y - 0.0593 * 1.8814 / 0.6780 * U - 0.2627 * 1.4746 / 0.6780 * V;
		Npp32f Bf = Y + 1.8814 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelYUV2RGB10Rec2020(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int position = i % Ypitch;
		if (position >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;

		Npp32f Y = (Npp32f)planeY[i];
		Npp32f U = (Npp32f)planeU[i] - .5;
		Npp32f V = (Npp32f)planeV[i] - .5;

		Npp32f Rf = round((Y + 1.4746 * V) * 1023);
		Npp32f Gf = round((Y - 0.0593 * 1.8814 / 0.6780 * U - 0.2627 * 1.4746 / 0.6780 * V) * 1023);
		Npp32f Bf = round((Y + 1.8814 * U) * 1023);

		if (Rf < 0) Rf = 0; else if (Rf > 1023) Rf = 1023;
		if (Gf < 0) Gf = 0; else if (Gf > 1023) Gf = 1023;
		if (Bf < 0) Bf = 0; else if (Bf > 1023) Bf = 1023;

		planeR[row * width + position] = (unsigned short)Rf;
		planeG[row * width + position] = (unsigned short)Gf;
		planeB[row * width + position] = (unsigned short)Bf;
	}
	__global__ void KernelYUV420toRGB10Rec2020(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch, int Upitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;
		int UVoffset = position / 2 + Upitch * (row / 2);

		Npp32f Y = (Npp32f)planeY[i] / 1023.0;
		Npp32f U = (Npp32f)planeU[UVoffset] / 1023.0 - .5;
		Npp32f V = (Npp32f)planeV[UVoffset] / 1023.0 - .5;

		Npp32f Rf = Y + 1.4746 * V;
		Npp32f Gf = Y - 0.0593 * 1.8814 / 0.6780 * U - 0.2627 * 1.4746 / 0.6780 * V;
		Npp32f Bf = Y + 1.8814 * U;

		planeR[row * width + position] = Rf;
		planeG[row * width + position] = Gf;
		planeB[row * width + position] = Bf;
	}
	__global__ void KernelRGB2YUV10Rec2020(unsigned short* planeY, unsigned short* planeU, unsigned short* planeV, Npp32f* planeR, Npp32f* planeG, Npp32f* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];

		Npp32f yf = 0.2627 * R + 0.6780 * G + 0.0593 * B;
		Npp32f Y = round(yf * 1023);
		Npp32f U = round(((B - yf) / 1.8814 + 0.5) * 1023);
		Npp32f V = round(((R - yf) / 1.4746 + 0.5) * 1023);

		if (Y < 0) Y = 0; else if (Y > 1023) Y = 1023;
		if (U < 0) U = 0; else if (U > 1023) U = 1023;
		if (V < 0) V = 0; else if (V > 1023) V = 1023;

		planeY[row * Ypitch + position] = (unsigned short)Y;
		planeU[row * Ypitch + position] = (unsigned short)U;
		planeV[row * Ypitch + position] = (unsigned short)V;
	}
	__global__ void KernelRGB2YUV10Rec2020(Npp32f* planeY, Npp32f* planeU, Npp32f* planeV, unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int width, int height, int Ypitch)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i % Ypitch >= width) return; //if in padding zone - do nothing
		if (i >= height * Ypitch) return; //if i greater than buffer - do nothing

		int row = i / Ypitch;
		int position = i % Ypitch;

		Npp32f R = planeR[row * width + position];
		Npp32f G = planeG[row * width + position];
		Npp32f B = planeB[row * width + position];
		R /= 1023.0;
		G /= 1023.0;
		B /= 1023.0;

		Npp32f Y = 0.2627 * R + 0.6780 * G + 0.0593 * B; //Y
		Npp32f U = (B - Y) / 1.8814 + .5; //Cb
		Npp32f V = (R - Y) / 1.4746 + .5; //Cr

		planeY[row * Ypitch + position] = Y;
		planeU[row * Ypitch + position] = U;
		planeV[row * Ypitch + position] = V;
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
	__global__ void reduceBlacksFloats(Npp32f* input, unsigned int* output, int length) {
		extern __shared__ unsigned int sdata[];
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int tid = threadIdx.x;
		if (i >= length)
			sdata[tid] = 0;
		else
		{
			if (input[i] == 0.0)
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
		cudaMalloc((void**)&reduceout, blocks * 4); //change size
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
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, blocks);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[resultblocks];
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
		cudaMalloc((void**)&reduceout, blocks * 8); //change size
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
			reduceFloat<<<resultblocks, maxthreads, maxthreads * 8>>> (reduceout, reducelast, blocks);
			cudaFree(reduceout);
			Npp64f* tosum = new Npp64f[resultblocks];
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
		cudaMalloc((void**)&reduceout, blocks * 4); //change size
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
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, blocks);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[resultblocks];
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
		cudaMalloc((void**)&reduceout, blocks * 4); //change size
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
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, blocks);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[resultblocks];
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
		cudaMalloc((void**)&reduceout, blocks * 4); //change size
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
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, blocks);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[resultblocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}
	void CudaCountBlacksNV(Npp32f* buf, int length, unsigned long long* result, unsigned int maxthreads) {
		int blocks = length / maxthreads;
		if (length % maxthreads > 0) blocks += 1;
		unsigned int* reduceout = 0;
		int resultblocks = blocks / maxthreads;
		if (blocks % maxthreads > 0) resultblocks += 1;
		cudaMalloc((void**)&reduceout, blocks * 4); //change size
		reduceBlacksFloats <<<blocks, maxthreads, maxthreads * 4 >>> (buf, reduceout, length); //sdata should be 2^x size
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
			reduceInt <<<resultblocks, maxthreads, maxthreads * 4 >>> (reduceout, reducelast, blocks);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[resultblocks];
			cudaMemcpy(tosum, reducelast, resultblocks * 4, cudaMemcpyDeviceToHost);
			for (int i = 0; i != resultblocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reducelast);
		}
	}

//Other functions

	__global__ void KernelTV2PC(unsigned char* buf, int length, int rangemin, int rangemax)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			unsigned char c = buf[i];
			if (c <= rangemin) c = 0; //lowerst
			else
				if (c >= rangemax) c = 255; //max
				else
				{ //16-235
					Npp32f cf = c;
					cf = (cf - rangemin) / (rangemax - rangemin);
					cf = cf * 255;
					c = (unsigned char)cf;
				}
			buf[i] = c;
		}
	}
	__global__ void KernelTV2PC(unsigned short* buf, int length, int rangemin, int rangemax)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			unsigned short c = buf[i];
			if (c <= rangemin) c = 0; //lowerst
			else
				if (c >= rangemax) c = 1023; //max
				else
				{ //16-235
					Npp32f cf = c;
					cf = (cf - rangemin) / (rangemax - rangemin);
					cf = cf * 1023;
					c = (unsigned short)cf;
				}
			buf[i] = c;
		}
	}
	__global__ void KernelTV2PC(Npp32f* buf, int length, int rangemin, int rangemax)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f c = buf[i];
			Npp32f lowlimit = rangemin / 255.0;
			Npp32f highlimit = rangemax / 255.0;
			Npp32f range = (rangemax-rangemin) / 255.0;
			if (c <= lowlimit) c = 0.0; //lowerst
			else
				if (c >= highlimit) c = 1.0; //max
				else
				{ //16-235
					Npp32f cf = c;
					cf = (cf - lowlimit) / range;
					c = cf;
				}
			buf[i] = c;
		}
	}
	__global__ void KernelPC2TV(unsigned char* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i];
			cf = round(cf / 255.0 * 219.0 + 16.0);
			buf[i] = (unsigned char)cf;
		}
	}
	__global__ void KernelPC2TV(unsigned short* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i];
			cf = round(cf / 1023.0 * 879.0 + 64.0);
			buf[i] = (unsigned short)cf;
		}
	}
	__global__ void KernelPC2TV(Npp32f* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f c = buf[i];
			static Npp32f lowlimit = 16.0 / 255.0;
			static Npp32f range = 219.0 / 255.0;
			c = round(c * range + lowlimit);
			buf[i] = c;
		}
	}
	
	__global__ void KernelTVClamp(unsigned char* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			unsigned char cf = buf[i];
			if (cf < 16) buf[i] = 16; else if (cf > 235) buf[i] = 235;
		}
	}
	__global__ void KernelTVClamp(unsigned short* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			unsigned short cf = buf[i];
			if (cf < 64) buf[i] = 64; else if (cf > 943) buf[i] = 943;
		}
	}
	__global__ void KernelTVClamp(Npp32f* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f c = buf[i];
			static Npp32f lowlimit = 16.0 / 255.0;
			static Npp32f highlimit = 235.0 / 255.0;
			if (c < lowlimit) buf[i] = lowlimit; else if (c > highlimit) buf[i] = highlimit;
		}
	}
	
	__global__ void KernelOETF(unsigned char* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i] / 255.0;
			if (cf <= 1.0 && cf >= 0.018) cf = 1.099 * pow(cf, 0.45) - 0.099; else if (cf < 0.018 && cf >= 0) cf *= 4.5;
			cf *= 255;
			cf = round(cf);
			if (cf < 0) cf = 0; else if (cf > 255) cf = 255;
			buf[i] = cf;
		}
	}
	__global__ void KernelOETF(unsigned short* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i] / 1023.0;
			if (cf <= 1.0 && cf >= 0.018) cf = 1.099 * pow(cf, 0.45) - 0.099; else if (cf < 0.018 && cf >= 0) cf *= 4.5;
			cf *= 1023;
			cf = round(cf);
			if (cf < 0) cf = 0; else if (cf > 1023) cf = 1023;
			buf[i] = (unsigned short)cf;
		}
	}
	__global__ void KernelOETF(Npp32f* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i];
			if (cf <= 1.0 && cf >= 0.018) cf = 1.099 * pow(cf, 0.45) - 0.099; else if (cf < 0.018 && cf >= 0) cf *= 4.5;
			buf[i] = cf;
		}
	}
	__global__ void KernelEOTF(unsigned char* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i] / 255.0;
			if (cf < 0.081) cf = cf / 4.5; else cf = pow((cf + 0.099) / 1.099, 1 / 0.45);
			cf *= 255;
			cf = round(cf);
			if (cf < 0) cf = 0; else if (cf > 255) cf = 255;
			buf[i] = cf;
		}
	}
	__global__ void KernelEOTF(unsigned short* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i] / 1023.0;
			if (cf < 0.081) cf = cf / 4.5; else cf = pow((cf + 0.099) / 1.099, 1 / 0.45);
			cf *= 1023;
			cf = round(cf);
			if (cf < 0) cf = 0; else if (cf > 1023) cf = 1023;
			buf[i] = (unsigned short)cf;
		}
	}
	__global__ void KernelEOTF(Npp32f* buf, int length)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < length) {
			Npp32f cf = buf[i];
			if (cf < 0.081) cf = cf / 4.5; else cf = pow((cf + 0.099) / 1.099, 1 / 0.45);
			buf[i] = cf;
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
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
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
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
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
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
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
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
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
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
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
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
					KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
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

	void CudaNeutralizeYUV420byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int memsize = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memsize);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memsize);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memsize);
		
		switch (yuvin)
		{
			case 709: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
			case 601: KernelYUV420toRGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
			case 2020: KernelYUV420toRGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
			default: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, memsize);
			cudaMalloc(&planeHSVo_Snv, memsize);
			cudaMalloc(&planeHSV_Vnv, memsize);
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, memsize);
			cudaMalloc(&planeHSV_Snv, memsize);
			cudaMalloc(&planeHSVo_Vnv, memsize);
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads>>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads>>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads>>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
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
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
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


		switch (yuvout)
		{
			case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
			case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
			case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
			default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}
		
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
	void CudaNeutralizeYUV420byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack,int yuvin, int yuvout)
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

		int memlen = Ylength * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		int length = planeYwidth * planeYheight;

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
				}
			}
		}

		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaBoostSaturationYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout) {
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

		int memlen = Ylength * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, memlen);
		cudaMalloc(&planeHSV_Snv, memlen);
		cudaMalloc(&planeHSV_Vnv, memlen);

		KernelRGB2HSV_HV <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
		}
		}

		KernelRGB2HSV_S <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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

	void CudaNeutralizeYUV444byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		Npp32f* planeHSVo_Hnv;
		Npp32f* planeHSVo_Snv;
		Npp32f* planeHSVo_Vnv;

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		if (type == 1)
		{
			cudaMalloc(&planeHSVo_Hnv, memlen);
			cudaMalloc(&planeHSVo_Snv, memlen);
			cudaMalloc(&planeHSV_Vnv, memlen);
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, memlen);
			cudaMalloc(&planeHSV_Snv, memlen);
			cudaMalloc(&planeHSVo_Vnv, memlen);
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
		}

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;


		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, length);
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
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
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

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaNeutralizeYUV444byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0.0, 1.0, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
			}
			}
		}

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaBoostSaturationYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, memlen);
		cudaMalloc(&planeHSV_Snv, memlen);
		cudaMalloc(&planeHSV_Vnv, memlen);

		KernelRGB2HSV_HV <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0.0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0.0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0.0, 1.0, length);
		}
		}

		KernelRGB2HSV_S <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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

	void CudaTV2PCYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		int rgbblocks = blocks(length, threads);

		if (rangemin >= rangemax)
		{
			rangemin = 16;
			rangemax = 235;
		}

		KernelTV2PC <<<rgbblocks, threads>>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads>>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads>>> (planeBnv, length, rangemin, rangemax);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaTV2PCYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		int rgbblocks = blocks(length, threads);

		if (rangemin >= rangemax)
		{
			rangemin = 16;
			rangemax = 235;
		}

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length, rangemin, rangemax);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaTV2PCRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int rangemin, int rangemax) {
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

		if (rangemin >= rangemax)
		{
			rangemin = 16;
			rangemax = 235;
		}

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length, rangemin, rangemax);

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaTV2PCRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int rangemin, int rangemax) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		if (rangemin >= rangemax)
		{
			rangemin = 16;
			rangemax = 235;
		}

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length, rangemin, rangemax);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}

	void CudaPC2TVYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink << <shrinkblocks, threads >> > (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYwidth, planeUwidth, Ylength / 4);
		KernelUVShrink << <shrinkblocks, threads >> > (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYwidth, planeVwidth, Ylength / 4);

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
	void CudaPC2TVYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaPC2TVRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength / 4, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrLength);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB << <bgrblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		KernelRGBtoBGR << <rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaPC2TVRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}

	void CudaTVClampYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink << <shrinkblocks, threads >> > (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYwidth, planeUwidth, Ylength / 4);
		KernelUVShrink << <shrinkblocks, threads >> > (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYwidth, planeVwidth, Ylength / 4);

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
	void CudaTVClampYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaTVClampRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength / 4, threads);

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrLength);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB << <bgrblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		KernelRGBtoBGR << <rgbblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaTVClampRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}

	void CudaGrayscaleRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int yuvin, int yuvout) {

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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeYnv; cudaMalloc(&planeYnv, memlen);
		Npp32f* planeUnv; cudaMalloc(&planeUnv, memlen);
		Npp32f* planeVnv; cudaMalloc(&planeVnv, memlen);

		switch (yuvin)
		{
		case 709: KernelRGB2YUVRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 601: KernelRGB2YUVRec601 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		default: KernelRGB2YUVRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		}

		Npp32f* temp = (Npp32f*)malloc(length * sizeof(Npp32f));
		for (int i = 0; i != length; i++) temp[i] = 0.5;
		cudaMemcpy(planeUnv, temp, length * sizeof(Npp32f), cudaMemcpyHostToDevice);
		cudaMemcpy(planeVnv, temp, length * sizeof(Npp32f), cudaMemcpyHostToDevice);
		free(temp);

		switch (yuvout)
		{
		case 709: KernelYUV2RGBRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 601: KernelYUV2RGBRec601 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		default: KernelYUV2RGBRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		}
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
	void CudaGrayscaleRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int yuvin, int yuvout) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeYnv; cudaMalloc(&planeYnv, memlen);
		Npp32f* planeUnv; cudaMalloc(&planeUnv, memlen);
		Npp32f* planeVnv; cudaMalloc(&planeVnv, memlen);

		switch (yuvin)
		{
		case 709: KernelRGB2YUVRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 601: KernelRGB2YUVRec601 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		default: KernelRGB2YUVRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		}
		
		Npp32f* temp = (Npp32f*)malloc(length * sizeof(Npp32f));
		for (int i = 0; i != length; i++) temp[i] = 0.5;
		cudaMemcpy(planeUnv, temp, length * sizeof(Npp32f), cudaMemcpyHostToDevice);
		cudaMemcpy(planeVnv, temp, length * sizeof(Npp32f), cudaMemcpyHostToDevice);
		free(temp);
		
		switch (yuvout)
		{
		case 709: KernelYUV2RGBRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 601: KernelYUV2RGBRec601 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		default: KernelYUV2RGBRec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth); break;
		}

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
	
	void CudaOETFRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelOETF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaOETFRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {

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

		KernelOETF <<<bgrblocks, threads >>> (planeRnv, length);
		KernelOETF <<<bgrblocks, threads >>> (planeGnv, length);
		KernelOETF <<<bgrblocks, threads >>> (planeBnv, length);

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaOETFYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout)
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

		int memsize = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memsize);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memsize);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memsize);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		KernelOETF <<<Yblocks, threads >>> (planeRnv, length);
		KernelOETF <<<Yblocks, threads >>> (planeGnv, length);
		KernelOETF <<<Yblocks, threads >>> (planeBnv, length);


		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);


		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaOETFYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		KernelOETF <<<Yblocks, threads >>> (planeRnv, length);
		KernelOETF <<<Yblocks, threads >>> (planeGnv, length);
		KernelOETF <<<Yblocks, threads >>> (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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

	void CudaEOTFRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelEOTF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelEOTF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelEOTF <<<rgbblocks, threads >>> (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaEOTFRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {

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

		KernelEOTF <<<bgrblocks, threads >>> (planeRnv, length);
		KernelEOTF <<<bgrblocks, threads >>> (planeGnv, length);
		KernelEOTF <<<bgrblocks, threads >>> (planeBnv, length);

		KernelRGBtoBGR <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}
	void CudaEOTFYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout)
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

		int memsize = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memsize);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memsize);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memsize);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 601: KernelYUV420toRGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		case 2020: KernelYUV420toRGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		default: KernelYUV420toRGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth, planeUwidth); break;
		}

		KernelEOTF <<<Yblocks, threads >>> (planeRnv, length);
		KernelEOTF <<<Yblocks, threads >>> (planeGnv, length);
		KernelEOTF <<<Yblocks, threads >>> (planeBnv, length);


		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength);
		cudaMalloc(&planeVnvFull, Ylength);


		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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
	void CudaEOTFYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelYUV2RGBRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelYUV2RGBRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelYUV2RGBRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

		KernelEOTF <<<Yblocks, threads >>> (planeRnv, length);
		KernelEOTF <<<Yblocks, threads >>> (planeGnv, length);
		KernelEOTF <<<Yblocks, threads >>> (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 601: KernelRGB2YUVRec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		case 2020: KernelRGB2YUVRec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		default: KernelRGB2YUVRec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYwidth); break;
		}

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

//Main 10bit

	void CudaNeutralizeYUV420byRGBwithLight10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
			}
			}
		}

		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);
		
		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}
		
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
	void CudaNeutralizeYUV420byRGB10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		int hsvsize = length * sizeof(Npp32f);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

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

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
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
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
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

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaTV2PCYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		if (rangemin >= rangemax)
		{
			rangemin = 64;
			rangemax = 943;
		}

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length, rangemin, rangemax);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaPC2TVYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink10 << <shrinkblocks, threads >> > (planeUnvFull, planeUnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2, Ylength / 4);
		KernelUVShrink10 << <shrinkblocks, threads >> > (planeVnvFull, planeVnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeVwidth / 2, Ylength / 4);

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
	void CudaTVClampYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int shrinkblocks = blocks(Ylength / 4, threads);

		KernelUVShrink10 << <shrinkblocks, threads >> > (planeUnvFull, planeUnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2, Ylength / 4);
		KernelUVShrink10 << <shrinkblocks, threads >> > (planeVnvFull, planeVnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeVwidth / 2, Ylength / 4);

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
	void CudaBoostSaturationYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout) {
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

		int hsvsize = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, hsvsize);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, hsvsize);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, hsvsize);
		
		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

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
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
			}
		}

		KernelRGB2HSV_S10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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

	void CudaOETFYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelOETF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaEOTFYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 601: KernelYUV420toRGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		case 2020: KernelYUV420toRGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		default: KernelYUV420toRGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2, planeUwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelOETF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeBnv, length);

		//allocate full UV planes buffers:
		unsigned short* planeUnvFull;
		unsigned short* planeVnvFull;
		cudaMalloc(&planeUnvFull, Ylength * 2);
		cudaMalloc(&planeVnvFull, Ylength * 2);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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

	void CudaNeutralizeYUV444byRGBwithLight10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int memlen = length * sizeof(Npp32f);
		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
			}
			}
		}

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaNeutralizeYUV444byRGB10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout)
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

		int hsvsize = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, hsvsize);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, hsvsize);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, hsvsize);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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

		Npp64f Rsum = 0, Gsum = 0, Bsum = 0;

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

		Rsum = 1.0 - Rsum;
		Gsum = 1.0 - Gsum;
		Bsum = 1.0 - Bsum;

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, (Npp32f)Rsum, 0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, (Npp32f)Gsum, 0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, (Npp32f)Bsum, 0, 1.0, length);
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
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
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

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaTV2PCYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		if (rangemin >= rangemax)
		{
			rangemin = 64;
			rangemax = 943;
		}

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length, rangemin, rangemax);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaPC2TVYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaTVClampYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaBoostSaturationYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout) {
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

		int hsvsize = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, hsvsize);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, hsvsize);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, hsvsize);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
			KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
			KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 1.0, length);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 1.0, length);
		}
		}

		KernelRGB2HSV_S10 <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, length);

		KernelHSV2RGB10 <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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

	void CudaOETFYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelOETF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
	void CudaEOTFYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout) {
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

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeRnv; cudaMalloc(&planeRnv, memlen);
		Npp32f* planeGnv; cudaMalloc(&planeGnv, memlen);
		Npp32f* planeBnv; cudaMalloc(&planeBnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

		int rgbblocks = blocks(length, threads);

		KernelEOTF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelEOTF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelEOTF <<<rgbblocks, threads >>> (planeBnv, length);

		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth / 2, planeYheight, planeYwidth / 2); break;
		}

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
			KernelRGB2HSV_HS10 <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, length); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V10 <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, length); //make original Volume plane
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
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
		}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS10 <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, length); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V10 <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, length);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, length);
			}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB10 <<<rgbblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, length);
		}
		else if (type == 1)
		{
			KernelHSV2RGB10 <<<rgbblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, length);
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
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, (float)Rsum / 1023, 0, 1023, length);
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, (float)Gsum / 1023, 0, 1023, length);
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, (float)Bsum / 1023, 0, 1023, length);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
				KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
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

		KernelRGB2HSV_HV <<<rgbblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, length);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_pegtop <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_illusionshu <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeRnv, planeRnv, 0, 1023, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeGnv, planeGnv, 0, 1023, length);
			KernelSoftlight_W3C <<<rgbblocks, threads >>> (planeBnv, planeBnv, 0, 1023, length);
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
	void CudaTV2PCRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int rangemin, int rangemax) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		if (rangemin >= rangemax)
		{
			rangemin = 64;
			rangemax = 943;
		}

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, length, rangemin, rangemax);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, length, rangemin, rangemax);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaPC2TVRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelPC2TV << <rgbblocks, threads >> > (planeRnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeGnv, length);
		KernelPC2TV << <rgbblocks, threads >> > (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaTVClampRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelTVClamp << <rgbblocks, threads >> > (planeRnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeGnv, length);
		KernelTVClamp << <rgbblocks, threads >> > (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaGrayscaleRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int yuvin, int yuvout) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned short* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned short* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned short* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		int memlen = length * sizeof(Npp32f);

		Npp32f* planeYnv; cudaMalloc(&planeYnv, memlen);
		Npp32f* planeUnv; cudaMalloc(&planeUnv, memlen);
		Npp32f* planeVnv; cudaMalloc(&planeVnv, memlen);

		switch (yuvin)
		{
		case 709: KernelYUV2RGB10Rec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		case 601: KernelYUV2RGB10Rec601 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		case 2020: KernelYUV2RGB10Rec2020 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		default: KernelYUV2RGB10Rec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		}

		Npp32f* gray = (Npp32f*)malloc(planewidth * planeheight);
		for (int i = 0; i != length; i++) gray[i] = 0.5;
		cudaMemcpy(planeUnv, gray, memlen, cudaMemcpyHostToDevice);
		cudaMemcpy(planeVnv, gray, memlen, cudaMemcpyHostToDevice);
		free(gray);
		
		switch (yuvout)
		{
		case 709: KernelRGB2YUV10Rec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		case 601: KernelRGB2YUV10Rec601 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		case 2020: KernelRGB2YUV10Rec2020 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		default: KernelRGB2YUV10Rec709 <<<rgbblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth / 2, planeheight, planewidth / 2); break;
		}

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

	void CudaOETFRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelOETF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelOETF <<<rgbblocks, threads >>> (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}
	void CudaEOTFRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads) {

		int length = planeheight * planewidth / 2;
		int rgbblocks = blocks(length, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, length * 2);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, length * 2);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, length * 2);

		cudaMemcpy2D(planeRnv, planewidth, planeR, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeGnv, planewidth, planeG, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);
		cudaMemcpy2D(planeBnv, planewidth, planeB, planepitch, planewidth, planeheight, cudaMemcpyHostToDevice);

		KernelEOTF <<<rgbblocks, threads >>> (planeRnv, length);
		KernelEOTF <<<rgbblocks, threads >>> (planeGnv, length);
		KernelEOTF <<<rgbblocks, threads >>> (planeBnv, length);

		cudaMemcpy2D(planeR, planepitch, planeRnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeG, planepitch, planeGnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(planeB, planepitch, planeBnv, planewidth, planewidth, planeheight, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
	}