// SoftLight.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "avisynth.h"
#include <math.h>
#include <cuda_runtime.h>



class SoftLight : public GenericVideoFilter {
public:
	SoftLight(PClip _child, int mode, IScriptEnvironment* env);
	BYTE DoSoftLight(BYTE s, BYTE c);
	~SoftLight();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	int mode = 0;
	bool cuda = false;
	unsigned int maxthreads = 0;
};

extern void CudaSoftlight(unsigned char* s, unsigned char c, int length, int threads);
extern void CudaSoftlight(unsigned char* s, unsigned char* c, int length, int threads);
extern void CudaSum(unsigned char* s, int length, unsigned long long * result, unsigned int maxthreads);

struct PIXEL {
	unsigned char blue;
	unsigned char green;
	unsigned char red;
	unsigned char luma;
};


SoftLight::SoftLight(PClip _child, int mode, IScriptEnvironment* env) : GenericVideoFilter(_child) {
	SoftLight::mode = mode;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus == cudaSuccess) {
		SoftLight::cuda = true;
		cudaDeviceProp prop;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);
		SoftLight::maxthreads = prop.maxThreadsPerBlock;
	}
}

SoftLight::~SoftLight() {}

PVideoFrame __stdcall SoftLight::GetFrame(int n, IScriptEnvironment* env) {

	PVideoFrame src = child->GetFrame(n, env);

	env->MakeWritable(&src);

	if (vi.IsPlanar() && vi.IsYUV()) {

		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
		int p;

		unsigned long long avg[3] = { 0,0,0 };
		unsigned char* srcp;
		int src_pitch, src_width, src_height;

		//calculate average:
		if (mode != 2) {
			if (!SoftLight::cuda) {
				for (p = 1; p < 3; p++) {
					srcp = src->GetWritePtr(planes[p]);
					src_pitch = src->GetPitch(planes[p]);
					src_width = src->GetRowSize(planes[p]);
					src_height = src->GetHeight(planes[p]);
					for (int h = 0; h < src_height; h++) {
						for (int w = 0; w < src_width; w++) {
							avg[p] += srcp[w];
						}
						srcp += src_pitch;
					}
				}
			}
			else {
				for (p = 1; p < 3; p++) {
					srcp = src->GetWritePtr(planes[p]);
					src_pitch = src->GetPitch(planes[p]);
					src_width = src->GetRowSize(planes[p]);
					src_height = src->GetHeight(planes[p]);
					CudaSum(srcp, src_width*src_height, &avg[p], SoftLight::maxthreads);
				}
			}

			//avg[0] /= src_width * src_height;
			avg[1] /= src_width * src_height;
			avg[2] /= src_width * src_height;

			//if (avg[0] > 255) avg[0] = 255;
			if (avg[1] > 255) avg[1] = 255;
			if (avg[2] > 255) avg[2] = 255;

			//neg
			//avg[0] = 255 - avg[0];
			avg[1] = 255 - avg[1];
			avg[2] = 255 - avg[2];
		}

		unsigned char* plane0 = src->GetWritePtr(planes[0]);
		unsigned char* plane1 = src->GetWritePtr(planes[1]);
		unsigned char* plane2 = src->GetWritePtr(planes[2]);
		for (p = 0; p < 3; p++) {
			srcp = src->GetWritePtr(planes[p]);
			src_pitch = src->GetPitch(planes[p]);
			src_width = src->GetRowSize(planes[p]);
			src_height = src->GetHeight(planes[p]);

			if (SoftLight::cuda) {

				switch (mode) {
				case 0: {
					if (p != 0)
						CudaSoftlight(srcp, (unsigned char)avg[p], src_width * src_height, SoftLight::maxthreads);
					break;
				}
				case 1: {
					if (p != 0)
						CudaSoftlight(srcp, (unsigned char)avg[p], src_width * src_height, SoftLight::maxthreads);
					else
						CudaSoftlight(srcp, srcp, src_width * src_height, SoftLight::maxthreads);
					break;
				}
				case 2: {
					if (p == 0) {
						CudaSoftlight(srcp, srcp, src_width * src_height, SoftLight::maxthreads);
					}
					break;
				}

				}
			}
			else
				for (int h = 0; h < src_height; h++) {
					for (int w = 0; w < src_width; w++) {
						switch (mode) {
						case 0: {
							if (p != 0)
								srcp[w] = DoSoftLight(srcp[w], (BYTE)avg[p]);
							break;
						}
						case 1: {
							if (p == 0) {
								srcp[w] = DoSoftLight(srcp[w], srcp[w]);
							}
							else
								srcp[w] = DoSoftLight(srcp[w], (BYTE)avg[p]);
							break;
						}
						case 2: {
							if (p == 0) {
								srcp[w] = DoSoftLight(srcp[w], srcp[w]);
							}
							break;
						}

						}
					}
					srcp += src_pitch;
				}
		}
	}


	return src;
}

AVSValue __cdecl CreateSoftLight(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new SoftLight(args[0].AsClip(), args[1].AsInt(), env);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("SoftLight", "c[mode]i", CreateSoftLight, 0);
	return "SoftLight plugin";
}

BYTE SoftLight::DoSoftLight(BYTE s, BYTE c) {
	double cf = c, sf = s, rf = 0;
	if (c <= 128) {
		rf = (255 - 2 * cf) * pow(sf / 255, 2) + 2 * cf * sf / 255;
	}
	else
	{
		rf = (2 * cf - 255) * sqrt(sf / 255) + 2 * (255 - cf) * sf / 255;
	}
	BYTE r = 0;
	if (rf < 0) r = 0;
	else if (rf > 255) r = 255; else r = (BYTE)rf;
	return r;
}
