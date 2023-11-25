// SoftLight.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <avisynth.h>
#include <math.h>
#include <cuda_runtime.h>

class SoftLight : public GenericVideoFilter {
public:
	SoftLight(PClip _child, int mode, int formula, IScriptEnvironment* env);
	~SoftLight();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	int mode = 0;
	int softlightFormula = 0;
	bool cuda = false;
	unsigned int maxthreads = 0;
};

extern void CudaNeutralizeYUV420byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaNeutralizeYUV444byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaNeutralizeYUV420byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaBoostSaturationYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula);

SoftLight::SoftLight(PClip _child, int mode, int formula, IScriptEnvironment* env) : GenericVideoFilter(_child) {
	SoftLight::mode = mode;
	SoftLight::softlightFormula = formula;

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

SoftLight::~SoftLight() {
}

PVideoFrame __stdcall SoftLight::GetFrame(int n, IScriptEnvironment* env) {

	PVideoFrame src = child->GetFrame(n, env);

	if (vi.IsPlanar() && vi.Is420() && SoftLight::cuda) {
		env->MakeWritable(&src);

		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

		unsigned char* planeY = src->GetWritePtr(planes[0]);
		unsigned char* planeU = src->GetWritePtr(planes[1]);
		unsigned char* planeV = src->GetWritePtr(planes[2]);

		int planeYpitch = src->GetPitch(PLANAR_Y);
		int planeYheight = src->GetHeight(PLANAR_Y);
		int planeYwidth = src->GetRowSize(PLANAR_Y);
		int planeUpitch = src->GetPitch(PLANAR_U);
		int planeUheight = src->GetHeight(PLANAR_U);
		int planeUwidth = src->GetRowSize(PLANAR_U);
		int planeVpitch = src->GetPitch(PLANAR_V);
		int planeVheight = src->GetHeight(PLANAR_V);
		int planeVwidth = src->GetRowSize(PLANAR_V);

		switch (mode) {
		case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV
		{
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
			break;
		}
		case 1: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
		{
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
			break;
		}
		case 2: //YUV -> neutralize(RGB) -> YUV
		{
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
			break;
		}
		case 3:
		{
			//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
			break;
		}
		case 4:
		{
			//YUV -> softlight RGB with RGB -> YUV
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
			break;
		}
		case 5: //boost saturation using softlight
		{
			CudaBoostSaturationYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula);
			break;
		}

		case 10: //Grayscale
		{
			for (int h = 0; h != planeUheight; h++)
				memset((void*)(planeU + h * planeUpitch), 128, planeUwidth);
			for (int h = 0; h != planeVheight; h++)
				memset((void*)(planeV + h * planeVpitch), 128, planeVwidth);
			break;
		}

		}
	}

	if (vi.IsPlanar() && vi.Is444() && SoftLight::cuda) {
		env->MakeWritable(&src);

		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

		unsigned char* planeY = src->GetWritePtr(planes[0]);
		unsigned char* planeU = src->GetWritePtr(planes[1]);
		unsigned char* planeV = src->GetWritePtr(planes[2]);

		int planeYpitch = src->GetPitch(PLANAR_Y);
		int planeYheight = src->GetHeight(PLANAR_Y);
		int planeYwidth = src->GetRowSize(PLANAR_Y);
		int planeUpitch = src->GetPitch(PLANAR_U);
		int planeUheight = src->GetHeight(PLANAR_U);
		int planeUwidth = src->GetRowSize(PLANAR_U);
		int planeVpitch = src->GetPitch(PLANAR_V);
		int planeVheight = src->GetHeight(PLANAR_V);
		int planeVwidth = src->GetRowSize(PLANAR_V);

		switch (mode) {
			case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
		}
	}

	return src;
}

AVSValue __cdecl CreateSoftLight(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new SoftLight(args[0].AsClip(), args[1].AsInt(), (float)args[2].AsFloat(), env);
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("SoftLight", "c[mode]i[formula]i", CreateSoftLight, 0);
	return "SoftLight plugin";
}