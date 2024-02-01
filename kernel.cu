#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <npp.h>

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
		planeBGR[pixeloffset + 3] = 0;
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

		Npp32f C = (Npp32f)planeY[i] - 16;
		Npp32f D = (Npp32f)planeU[i] - 128;
		Npp32f E = (Npp32f)planeV[i] - 128;

		Npp32f Rf = round(1.164383 * C + 1.596027 * E);
		Npp32f Gf = round(1.164383 * C - (0.391762 * D) - (0.812968 * E));
		Npp32f Bf = round(1.164383 * C + 2.017232 * D);

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

		Npp32f C = (Npp32f)planeY[i] - 16;
		Npp32f D = (Npp32f)planeU[UVoffset] - 128;
		Npp32f E = (Npp32f)planeV[UVoffset] - 128;

		Npp32f Rf = round(1.164383 * C + 1.596027 * E);
		Npp32f Gf = round(1.164383 * C - (0.391762 * D) - (0.812968 * E));
		Npp32f Bf = round(1.164383 * C + 2.017232 * D);

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

		Npp32f Y = round(0.256788 * R + 0.504129 * G + 0.097906 * B) + 16;
		Npp32f U = round(-0.148223 * R - 0.290993 * G + 0.439216 * B) + 128;
		Npp32f V = round(0.439216 * R - 0.367788 * G - 0.071427 * B) + 128;
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
	
//Parallel sum functions	
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
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * sizeof(int)); //change size
		reduceChar <<<blocks, maxthreads, maxthreads * sizeof(int) >>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reduceout, blocks * sizeof(int), cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			unsigned int* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * sizeof(int));
			reduceInt <<<resultblocks, maxthreads, maxthreads * sizeof(int) >>> (reduceout, reducelast, length);
			cudaFree(reduceout);
			unsigned int* tosum = new unsigned int[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * sizeof(int), cudaMemcpyDeviceToHost);
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
		cudaMalloc((void**)&reduceout, resultblocks * maxthreads * sizeof(Npp64f)); //change size
		reduceFloat <<<blocks, maxthreads, maxthreads * sizeof(Npp64f)>>> (buf, reduceout, length); //sdata should be 2^x size
		//final
		if (blocks <= 100) { //if just 100 sum by processor
			Npp64f* tosum = new Npp64f[blocks];
			cudaMemcpy(tosum, reduceout, blocks * sizeof(Npp64f), cudaMemcpyDeviceToHost);
			for (int i = 0; i != blocks; i++) *result += tosum[i];
			delete[] tosum;
			cudaFree(reduceout);
		}
		else {
			Npp64f* reducelast = 0;
			cudaMalloc(&reducelast, resultblocks * sizeof(Npp64f));
			reduceFloat<<<resultblocks, maxthreads, maxthreads * sizeof(Npp64f)>>> (reduceout, reducelast, length);
			cudaFree(reduceout);
			Npp64f* tosum = new Npp64f[blocks];
			cudaMemcpy(tosum, reducelast, resultblocks * sizeof(Npp64f), cudaMemcpyDeviceToHost);
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
					cf = (cf - 16) / 220 * 256 + 0.5;
					c = (unsigned char)cf;
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

	void CudaNeutralizeRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula) {
		
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength, threads);

		int planeYlength = planeheight * planewidth;
		int Yblocks = blocks(planeYlength, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads>>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);
		
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
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, planeYlength); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, planeYlength); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, planeYlength, &Rsum, threads);
		CudaSumNV(planeGnv, planeYlength, &Gsum, threads);
		CudaSumNV(planeBnv, planeYlength, &Bsum, threads);

		int length = planewidth * planeheight;
		Rsum /= length;
		Gsum /= length;
		Bsum /= length;
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >> > (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >> > (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >> > (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
			}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS << <Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, planeYlength); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V << <Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, planeYlength);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop <<<Yblocks, threads >> > (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu <<<Yblocks, threads >> > (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C <<<Yblocks, threads >> > (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
			}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);
		}
		else if (type == 1)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);
		}

		KernelRGBtoBGR <<<Yblocks, threads>>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

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

	void CudaNeutralizeRGB32withLight(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula)
	{
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength, threads);

		int planeYlength = planeheight * planewidth;
		int Yblocks = blocks(planeYlength, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB << <bgrblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, planeYlength, &Rsum, threads);
		CudaSumNV(planeGnv, planeYlength, &Gsum, threads);
		CudaSumNV(planeBnv, planeYlength, &Bsum, threads);

		int length = planewidth * planeheight;
		Rsum /= length;
		Gsum /= length;
		Bsum /= length;
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
					KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
					KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
					KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
					KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
					KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
					KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlight_pegtop << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
					KernelSoftlight_pegtop << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
					KernelSoftlight_pegtop << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
					KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
					KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
					KernelSoftlight_W3C << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
					KernelSoftlight_W3C << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
				}
			}
		}

		KernelRGBtoBGR << <Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}

	void CudaBoostSaturationRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int formula)
	{
		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength, threads);

		int planeYlength = planeheight * planewidth;
		int Yblocks = blocks(planeYlength, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV <<<Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, planeYlength);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop <<<Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
			KernelSoftlight_pegtop <<<Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
			KernelSoftlight_pegtop <<<Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu <<<Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
			KernelSoftlight_illusionshu <<<Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
			KernelSoftlight_illusionshu <<<Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
			KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
		}
		}

		KernelRGB2HSV_S <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, planeYlength);

		KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);

		KernelRGBtoBGR <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
		cudaFree(planeHSV_Hnv);
		cudaFree(planeHSV_Snv);
		cudaFree(planeHSV_Vnv);
	}

	void CudaNeutralizeYUV420byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);
		
		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);
		
		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;
		
		KernelYUV420toRGB<<<Yblocks,threads>>>(planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch, planeUpitch);

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
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, planeYlength); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, planeYlength); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, planeYlength, &Rsum, threads);
		CudaSumNV(planeGnv, planeYlength, &Gsum, threads);
		CudaSumNV(planeBnv, planeYlength, &Bsum, threads);

		int length = planeYwidth * planeYheight;
		Rsum /= length;
		Gsum /= length;
		Bsum /= length;
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
			}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS <<<Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, planeYlength); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V <<<Yblocks, threads>>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, planeYlength);
		}
		
		if (type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlightF_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlightF_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlightF_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
				}
			}
		}

		if (type == 0 || type == 2)
		{
			KernelHSV2RGB << <Yblocks, threads >> > (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);
		}
		else if (type == 1)
		{
			KernelHSV2RGB << <Yblocks, threads >> > (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);
		}
		
		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Yblocks * threads * sizeof(char));
		cudaMalloc(&planeVnvFull, Yblocks * threads * sizeof(char));

		KernelRGB2YUV<<<Yblocks,threads>>>(planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);
		
		int shrinkblocks = blocks(planeYwidth * planeYheight / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYpitch, planeUpitch, planeYwidth * planeYheight / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYpitch, planeVpitch, planeYwidth * planeYheight / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

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
	
	void CudaNeutralizeYUV420byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		KernelYUV420toRGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch, planeUpitch);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, planeYlength, &Rsum, threads);
		CudaSumNV(planeGnv, planeYlength, &Gsum, threads);
		CudaSumNV(planeBnv, planeYlength, &Bsum, threads);

		int length = planeYwidth * planeYheight;
		Rsum /= length;
		Gsum /= length;
		Bsum /= length;
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
				case 0: //pegtop
				{
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
					KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
					break;
				}
				case 1: //illusions.hu
				{
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
					KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
					break;
				}
				case 2: //W3C
				{
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
					KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
				}
			}
		}

		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Yblocks * threads * sizeof(char));
		cudaMalloc(&planeVnvFull, Yblocks * threads * sizeof(char));

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		int shrinkblocks = blocks(planeYwidth * planeYheight / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYpitch, planeUpitch, planeYwidth * planeYheight / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYpitch, planeVpitch, planeYwidth * planeYheight / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	void CudaNeutralizeYUV444byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		KernelYUV2RGB << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, planeYlength, &Rsum, threads);
		CudaSumNV(planeGnv, planeYlength, &Gsum, threads);
		CudaSumNV(planeBnv, planeYlength, &Bsum, threads);

		int length = planeYwidth * planeYheight;
		Rsum /= length;
		Gsum /= length;
		Bsum /= length;
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;

		if (type == 0 || type == 1 || type == 3) {

			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop << <Yblocks, threads >> > (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop << <Yblocks, threads >> > (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop << <Yblocks, threads >> > (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu << <Yblocks, threads >> > (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu << <Yblocks, threads >> > (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu << <Yblocks, threads >> > (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C << <Yblocks, threads >> > (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C << <Yblocks, threads >> > (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C << <Yblocks, threads >> > (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
			}
			}
		}

		if (type == 1 || type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
				KernelSoftlight_pegtop << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
				KernelSoftlight_pegtop << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
				KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
				KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
				KernelSoftlight_W3C << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
				KernelSoftlight_W3C << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
			}
			}
		}

		KernelRGB2YUV << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

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

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;

		KernelYUV420toRGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch, planeUpitch);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, planeYlength);

		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
				KernelSoftlight_pegtop <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
				KernelSoftlight_illusionshu <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeRnv, planeRnv, 0, 255, planeYlength);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeGnv, planeGnv, 0, 255, planeYlength);
				KernelSoftlight_W3C <<<Yblocks, threads >>> (planeBnv, planeBnv, 0, 255, planeYlength);
			}
		}

		KernelRGB2HSV_S <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Snv, planeYlength);

		KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Yblocks * threads * sizeof(char));
		cudaMalloc(&planeVnvFull, Yblocks * threads * sizeof(char));

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		int shrinkblocks = blocks(planeYwidth * planeYheight / 4, threads);

		KernelUVShrink <<<shrinkblocks, threads >>> (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYpitch, planeUpitch, planeYwidth * planeYheight / 4);
		KernelUVShrink <<<shrinkblocks, threads >>> (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYpitch, planeVpitch, planeYwidth * planeYheight / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

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
	
	void CudaBoostSaturationYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula) {
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;

		KernelYUV2RGB << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		Npp32f* planeHSV_Hnv;
		Npp32f* planeHSV_Snv;
		Npp32f* planeHSV_Vnv;

		cudaMalloc(&planeHSV_Hnv, hsvsize);
		cudaMalloc(&planeHSV_Snv, hsvsize);
		cudaMalloc(&planeHSV_Vnv, hsvsize);

		KernelRGB2HSV_HV << <Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Vnv, planeYlength);

		switch (formula) {
		case 0: //pegtop
		{
			KernelSoftlight_pegtop << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
			KernelSoftlight_pegtop << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
			KernelSoftlight_pegtop << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
			break;
		}
		case 1: //illusions.hu
		{
			KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
			KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
			KernelSoftlight_illusionshu << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
			break;
		}
		case 2: //W3C
		{
			KernelSoftlight_W3C << <Yblocks, threads >> > (planeRnv, planeRnv, 0, 255, planeYlength);
			KernelSoftlight_W3C << <Yblocks, threads >> > (planeGnv, planeGnv, 0, 255, planeYlength);
			KernelSoftlight_W3C << <Yblocks, threads >> > (planeBnv, planeBnv, 0, 255, planeYlength);
		}
		}

		KernelRGB2HSV_S << <Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeHSV_Snv, planeYlength);

		KernelHSV2RGB << <Yblocks, threads >> > (planeHSV_Hnv, planeHSV_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);

		KernelRGB2YUV << <Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

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

	void CudaNeutralizeYUV444byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula)
	{
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		int hsvsize = blocks(planeYlength * sizeof(Npp32f), threads) * threads;

		KernelYUV2RGB<<<Yblocks,threads>>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv,planeYwidth, planeYheight, planeYpitch);

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
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Hnv, planeHSVo_Snv, planeYlength); //make Hue & Saturation planes from original planes
		}
		else if (type == 0 || type == 2)
		{
			cudaMalloc(&planeHSV_Hnv, hsvsize);
			cudaMalloc(&planeHSV_Snv, hsvsize);
			cudaMalloc(&planeHSVo_Vnv, hsvsize);
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSVo_Vnv, planeYlength); //make original Volume plane
		}

		unsigned long long Rsum = 0, Gsum = 0, Bsum = 0;

		CudaSumNV(planeRnv, planeYlength, &Rsum, threads);
		CudaSumNV(planeGnv, planeYlength, &Gsum, threads);
		CudaSumNV(planeBnv, planeYlength, &Bsum, threads);

		int length = planeYwidth * planeYheight;
		Rsum /= length;
		Gsum /= length;
		Bsum /= length;
		Rsum = 255 - Rsum;
		Gsum = 255 - Gsum;
		Bsum = 255 - Bsum;


		switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_pegtop <<<Yblocks, threads>>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_illusionshu <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeRnv, (float)Rsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeGnv, (float)Gsum / 255, 0, 255, planeYlength);
				KernelSoftlightFC_W3C <<<Yblocks, threads >>> (planeBnv, (float)Bsum / 255, 0, 255, planeYlength);
			}
		}

		if (type == 0 || type == 2)
		{
			KernelRGB2HSV_HS <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Hnv, planeHSV_Snv, planeYlength); //make Hue & Saturation planes from processed RGB
		}
		else if (type == 1)
		{
			KernelRGB2HSV_V <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeHSV_Vnv, planeYlength);
		}

		if (type == 2) {
			switch (formula) {
			case 0: //pegtop
			{
				KernelSoftlightF_pegtop <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
				break;
			}
			case 1: //illusions.hu
			{
				KernelSoftlightF_illusionshu <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
				break;
			}
			case 2: //W3C
			{
				KernelSoftlightF_W3C <<<Yblocks, threads >>> (planeHSV_Snv, planeHSV_Snv, 0.0, 1.0, planeYlength);
			}
			}
		}
		if (type == 0 || type == 2)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSV_Hnv, planeHSV_Snv, planeHSVo_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);
		}
		else if (type == 1)
		{
			KernelHSV2RGB <<<Yblocks, threads >>> (planeHSVo_Hnv, planeHSVo_Snv, planeHSV_Vnv, planeRnv, planeGnv, planeBnv, planeYlength);
		}

		KernelRGB2YUV <<<Yblocks, threads>>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

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

	void CudaTV2PCYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads) {
		unsigned char* planeYnv;
		unsigned char* planeUnv;
		unsigned char* planeVnv;

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		KernelYUV420toRGB <<<Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch, planeUpitch);

		int rgbblocks = blocks(planeYheight * planeYwidth, threads);

		KernelTV2PC <<<rgbblocks, threads>>> (planeRnv, planeYheight * planeYwidth);
		KernelTV2PC <<<rgbblocks, threads>>> (planeGnv, planeYheight * planeYwidth);
		KernelTV2PC <<<rgbblocks, threads>>> (planeBnv, planeYheight * planeYwidth);

		//allocate full UV planes buffers:
		unsigned char* planeUnvFull;
		unsigned char* planeVnvFull;
		cudaMalloc(&planeUnvFull, Yblocks * threads * sizeof(char));
		cudaMalloc(&planeVnvFull, Yblocks * threads * sizeof(char));

		KernelRGB2YUV << <Yblocks, threads >> > (planeYnv, planeUnvFull, planeVnvFull, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		int shrinkblocks = blocks(planeYwidth * planeYheight / 4, threads);

		KernelUVShrink << <shrinkblocks, threads >> > (planeUnvFull, planeUnv, planeYwidth, planeYheight, planeYpitch, planeUpitch, planeYwidth * planeYheight / 4);
		KernelUVShrink << <shrinkblocks, threads >> > (planeVnvFull, planeVnv, planeYwidth, planeYheight, planeYpitch, planeVpitch, planeYwidth * planeYheight / 4);

		cudaFree(planeUnvFull);
		cudaFree(planeVnvFull);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

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

		int planeYlength = planeYpitch * planeYheight;
		int planeUlength = planeUpitch * planeUheight;
		int planeVlength = planeVpitch * planeVheight;

		int Yblocks = blocks(planeYlength, threads);
		int Ublocks = blocks(planeUlength, threads);
		int Vblocks = blocks(planeVlength, threads);

		cudaMalloc(&planeYnv, Yblocks * threads * sizeof(char));
		cudaMemcpy(planeYnv, planeY, planeYlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeUnv, Ublocks * threads * sizeof(char));
		cudaMemcpy(planeUnv, planeU, planeUlength, cudaMemcpyHostToDevice);

		cudaMalloc(&planeVnv, Vblocks * threads * sizeof(char));
		cudaMemcpy(planeVnv, planeV, planeVlength, cudaMemcpyHostToDevice);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);

		KernelYUV2RGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		int rgbblocks = blocks(planeYheight * planeYwidth, threads);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, planeYheight * planeYwidth);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, planeYheight * planeYwidth);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, planeYheight * planeYwidth);

		KernelRGB2YUV <<<Yblocks, threads >> > (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planeYwidth, planeYheight, planeYpitch);

		cudaMemcpy(planeY, planeYnv, planeYlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeU, planeUnv, planeUlength, cudaMemcpyDeviceToHost);
		cudaMemcpy(planeV, planeVnv, planeVlength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}
	
	void CudaTV2PCRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {

		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength, threads);

		int planeYlength = planeheight * planewidth;
		int Yblocks = blocks(planeYlength, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrblocks * threads);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB << <bgrblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		int rgbblocks = blocks(planeheight * planewidth, threads);

		KernelTV2PC <<<rgbblocks, threads >>> (planeRnv, planeheight * planewidth);
		KernelTV2PC <<<rgbblocks, threads >>> (planeGnv, planeheight * planewidth);
		KernelTV2PC <<<rgbblocks, threads >>> (planeBnv, planeheight * planewidth);

		KernelRGBtoBGR <<<Yblocks, threads >>> (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);
	}

	void CudaGrayscaleRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads) {

		int bgrLength = planeheight * planepitch;
		int bgrblocks = blocks(bgrLength, threads);

		int planeYlength = planeheight * planewidth;
		int Yblocks = blocks(planeYlength, threads);

		unsigned char* planeRnv; cudaMalloc(&planeRnv, Yblocks * threads);
		unsigned char* planeGnv; cudaMalloc(&planeGnv, Yblocks * threads);
		unsigned char* planeBnv; cudaMalloc(&planeBnv, Yblocks * threads);
		unsigned char* planeBGRnv; cudaMalloc(&planeBGRnv, bgrblocks * threads);

		cudaMemcpy(planeBGRnv, plane, bgrLength, cudaMemcpyHostToDevice);
		KernelBGRtoRGB <<<bgrblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);

		unsigned char* planeYnv; cudaMalloc(&planeYnv, Yblocks * threads);
		unsigned char* planeUnv; cudaMalloc(&planeUnv, Yblocks * threads);
		unsigned char* planeVnv; cudaMalloc(&planeVnv, Yblocks * threads);

		KernelRGB2YUV <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);
		cudaMemset(planeUnv, 128, planewidth * planeheight);
		cudaMemset(planeVnv, 128, planewidth * planeheight);
		KernelYUV2RGB <<<Yblocks, threads >>> (planeYnv, planeUnv, planeVnv, planeRnv, planeGnv, planeBnv, planewidth, planeheight, planewidth);
		KernelRGBtoBGR << <Yblocks, threads >> > (planeRnv, planeGnv, planeBnv, planeBGRnv, planewidth, planeheight, planewidth, planepitch);
		cudaMemcpy(plane, planeBGRnv, bgrLength, cudaMemcpyDeviceToHost);

		cudaFree(planeRnv);
		cudaFree(planeGnv);
		cudaFree(planeBnv);
		cudaFree(planeBGRnv);

		cudaFree(planeYnv);
		cudaFree(planeUnv);
		cudaFree(planeVnv);
	}