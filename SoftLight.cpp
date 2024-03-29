//common include
#include "stdafx.h"
#include <math.h>
#include <cuda_runtime.h>

//avisynth include
#include <avisynth.h>

//vapoursynth include
#include <VapourSynth4.h>
#include <VSHelper4.h>

//avisynth main code:
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

extern void CudaBoostSaturationRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int formula);
extern void CudaNeutralizeRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula);
extern void CudaNeutralizeRGB32withLight(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula);
extern void CudaNeutralizeYUV420byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaNeutralizeYUV444byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaNeutralizeYUV420byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaNeutralizeYUV444byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula);
extern void CudaNeutralizeY(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, int threads, int formula);
extern void CudaBoostSaturationYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula);
extern void CudaBoostSaturationYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula);
extern void CudaTV2PCYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads);
extern void CudaTV2PCYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads);
extern void CudaTV2PCRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaGrayscaleRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads);

//VapourSynth RGB24 only (planar)
extern void CudaNeutralizeRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula);
extern void CudaNeutralizeRGBwithLight(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula);
extern void CudaBoostSaturationRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int formula);
extern void CudaTV2PCRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaGrayscaleRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads);

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

	if (vi.IsPlanar() && vi.Is420() && vi.BitsPerComponent() == 8 && SoftLight::cuda) {
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
		case 1: //YUV -> neutralize(RGB) -> HSV -> restore HS -> RGB -> YUV (normalize lightness/volume)
		{
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
			break;
		}
		case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
		{
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
			break;
		}
		case 3: //YUV -> neutralize(RGB) -> YUV
		{
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
			break;
		}
		case 4:
		{
			//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
			break;
		}
		case 5:
		{
			//YUV -> softlight RGB with RGB -> YUV
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
			break;
		}
		case 6: //boost saturation using softlight
		{
			CudaBoostSaturationYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula);
			break;
		}
		case 8:
		{
			CudaTV2PCYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads);
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

	if (vi.IsPlanar() && vi.Is444() && vi.BitsPerComponent() == 8 && SoftLight::cuda) {
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
			case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV (neutralize colors)
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV (neutralize lightness)
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV (neutralize colors and boost saturation)
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads);
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

	if (!vi.IsPlanar() && vi.IsRGB32() && vi.BitsPerComponent() == 8 && SoftLight::cuda) {
		env->MakeWritable(&src);

		int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
		unsigned char* plane = src->GetWritePtr(planes[0]);
		int planepitch = src->GetPitch(0);
		int planeheight = src->GetHeight(0);
		int planewidth = src->GetRowSize(0) / 4;
		switch (mode) {
			case 0:
			{
				CudaNeutralizeRGB32(plane, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 1:
			{
				CudaNeutralizeRGB32(plane, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 2:
			{
				CudaNeutralizeRGB32(plane, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 3:
			{
				CudaNeutralizeRGB32withLight(plane, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 4:
			{
				CudaNeutralizeRGB32withLight(plane, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 5:
			{
				CudaNeutralizeRGB32withLight(plane, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 6:
			{
				CudaBoostSaturationRGB32(plane, planeheight, planewidth, planepitch, maxthreads, softlightFormula);
				break;
			}
			case 8:
			{
				CudaTV2PCRGB32(plane, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 10:
			{
				CudaGrayscaleRGB32(plane, planeheight, planewidth, planepitch, maxthreads);
			}
		}

	}

	if (vi.IsPlanarRGB() && vi.BitsPerComponent() == 8 && SoftLight::cuda) {
		env->MakeWritable(&src);

		int planes[] = { PLANAR_R, PLANAR_G, PLANAR_B };
		unsigned char* planeR = src->GetWritePtr(planes[0]);
		unsigned char* planeG = src->GetWritePtr(planes[1]);
		unsigned char* planeB = src->GetWritePtr(planes[2]);
		int planepitch = src->GetPitch(0);
		int planeheight = src->GetHeight(0);
		int planewidth = src->GetRowSize(0);
		switch (mode) {
			case 0:
			{
				CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 1:
			{
				CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 2:
			{
				CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 3:
			{
				CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 4:
			{
				CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 5:
			{
				CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 6:
			{
				CudaBoostSaturationRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, softlightFormula);
				break;
			}
			case 8:
			{
				CudaTV2PCRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 10:
			{
				CudaGrayscaleRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
			}
		}
	}
	return src;
}

AVSValue __cdecl CreateSoftLight(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new SoftLight(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), env);
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("SoftLight", "c[mode]i[formula]i", CreateSoftLight, 0);
	return "SoftLight plugin";
}


//vapoursynth

typedef struct {
	VSNode* node;
	const VSVideoInfo* vi;
} FilterData;

static int mode;
static int softlightFormula;
static bool cuda = false;
static unsigned int maxthreads = 0;

static const VSFrame* VS_CC filterGetFrame(int n, int activationReason, void* instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
	FilterData* d = (FilterData*)instanceData;

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, d->node, frameCtx);
	}
	else if (activationReason == arAllFramesReady) {
		const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);
		const VSVideoFormat* fi = vsapi->getVideoFrameFormat(src);
		int height = vsapi->getFrameHeight(src, 0);
		int width = vsapi->getFrameWidth(src, 0);
		VSFrame* dst = vsapi->newVideoFrame(fi, width, height, src, core);

		//yuv420p8 (3 = yuv, 0 = integer, 8 bit, 1,1) / pfYUV420P8 = VS_MAKE_VIDEO_ID(cfYUV, stInteger, 8, 1, 1)
		if (fi->colorFamily == 3 && fi->sampleType == 0 && fi->bitsPerSample == 8 && fi->subSamplingH == 1 && fi->subSamplingW == 1 && cuda)
		{

			const uint8_t* planeYs = vsapi->getReadPtr(src, 0);
			const uint8_t* planeUs = vsapi->getReadPtr(src, 1);
			const uint8_t* planeVs = vsapi->getReadPtr(src, 2);

			//our functions change passed data, so we must first copy source to destination and then pass destination to functions
			uint8_t* planeY = vsapi->getWritePtr(dst, 0);
			uint8_t* planeU = vsapi->getWritePtr(dst, 1);
			uint8_t* planeV = vsapi->getWritePtr(dst, 2);

			int planeYpitch = (int)vsapi->getStride(src, 0);
			int planeYheight = (int)vsapi->getFrameHeight(src, 0);
			int planeYwidth = (int)vsapi->getFrameWidth(src, 0);
			int planeUpitch = (int)vsapi->getStride(src, 1);
			int planeUheight = (int)vsapi->getFrameHeight(src, 1);
			int planeUwidth = (int)vsapi->getFrameWidth(src, 1);
			int planeVpitch = (int)vsapi->getStride(src, 2);
			int planeVheight = (int)vsapi->getFrameHeight(src, 2);
			int planeVwidth = (int)vsapi->getFrameWidth(src, 2);

			memcpy(planeY, planeYs, planeYheight * planeYpitch); //copy source to destination
			memcpy(planeU, planeUs, planeUheight * planeUpitch);
			memcpy(planeV, planeVs, planeVheight * planeVpitch);

			switch (mode) {
			case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV (neutralize colors)
			{
				CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV (neutralize lightness)
			{
				CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV (neutralize colors and boost saturation)
			{
				CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads);
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

		//yuv444p8 (3 = yuv, 0 = integer, 8 bit, 1,1) / pfYUV444P8 = VS_MAKE_VIDEO_ID(cfYUV, stInteger, 8, 0, 0)
		if (fi->colorFamily == 3 && fi->sampleType == 0 && fi->bitsPerSample == 8 && fi->subSamplingH == 0 && fi->subSamplingW == 0 && cuda)
		{

			const uint8_t* planeYs = vsapi->getReadPtr(src, 0);
			const uint8_t* planeUs = vsapi->getReadPtr(src, 1);
			const uint8_t* planeVs = vsapi->getReadPtr(src, 2);

			//our functions change passed data, so we must first copy source to destination and then pass destination to functions
			uint8_t* planeY = vsapi->getWritePtr(dst, 0);
			uint8_t* planeU = vsapi->getWritePtr(dst, 1);
			uint8_t* planeV = vsapi->getWritePtr(dst, 2);

			int planeYpitch = (int)vsapi->getStride(src, 0);
			int planeYheight = (int)vsapi->getFrameHeight(src, 0);
			int planeYwidth = (int)vsapi->getFrameWidth(src, 0);
			int planeUpitch = (int)vsapi->getStride(src, 1);
			int planeUheight = (int)vsapi->getFrameHeight(src, 1);
			int planeUwidth = (int)vsapi->getFrameWidth(src, 1);
			int planeVpitch = (int)vsapi->getStride(src, 2);
			int planeVheight = (int)vsapi->getFrameHeight(src, 2);
			int planeVwidth = (int)vsapi->getFrameWidth(src, 2);

			memcpy(planeY, planeYs, planeYheight * planeYpitch); //copy source to destination
			memcpy(planeU, planeUs, planeUheight * planeUpitch);
			memcpy(planeV, planeVs, planeVheight * planeVpitch);

			switch (mode) {
			case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads);
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

		//RGB24 (2 = RGB, 0 = integer, 8 bit, 0,0) / pfRGB24 = VS_MAKE_VIDEO_ID(cfRGB, stInteger, 8, 0, 0)
		if (fi->colorFamily == 2 && fi->sampleType == 0 && fi->bitsPerSample == 8 && fi->subSamplingH == 0 && fi->subSamplingW == 0 && cuda)
		{

			const uint8_t* planeRs = vsapi->getReadPtr(src, 0);
			const uint8_t* planeGs = vsapi->getReadPtr(src, 1);
			const uint8_t* planeBs = vsapi->getReadPtr(src, 2);

			//our functions change passed data, so we must first copy source to destination and then pass destination to functions
			uint8_t* planeR = vsapi->getWritePtr(dst, 0);
			uint8_t* planeG = vsapi->getWritePtr(dst, 1);
			uint8_t* planeB = vsapi->getWritePtr(dst, 2);

			int planepitch = (int)vsapi->getStride(src, 0);
			int planeheight = (int)vsapi->getFrameHeight(src, 0);
			int planewidth = (int)vsapi->getFrameWidth(src, 0);

			memcpy(planeR, planeRs, planeheight * planepitch); //copy source to destination
			memcpy(planeG, planeGs, planeheight * planepitch); //copy source to destination
			memcpy(planeB, planeBs, planeheight * planepitch); //copy source to destination

			switch (mode) {
				case 0:
				{
					CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula);
					break;
				}
				case 1:
				{
					CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula);
					break;
				}
				case 2:
				{
					CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula);
					break;
				}
				case 3:
				{
					CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula);
					break;
				}
				case 4:
				{
					CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula);
					break;
				}
				case 5:
				{
					CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula);
					break;
				}
				case 6:
				{
					CudaBoostSaturationRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, softlightFormula);
					break;
				}
				case 8:
				{
					CudaTV2PCRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
					break;
				}
				case 10:
				{
					CudaGrayscaleRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				}
			}
		}

		// Release the source frame
		vsapi->freeFrame(src);

		return dst;
	}

	return NULL;
}

static void VS_CC filterFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
	FilterData* d = (FilterData*)instanceData;
	vsapi->freeNode(d->node);
	free(d);
}

static void VS_CC filterCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
	//my code
	int error = 0;
	mode = (int)vsapi->mapGetInt(in, "mode", 0, &error);
	softlightFormula = (int)vsapi->mapGetInt(in, "formula", 0, &error);

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus == cudaSuccess) {
		cuda = true;
		cudaDeviceProp prop;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);
		maxthreads = prop.maxThreadsPerBlock;
	}
	//my code end

	FilterData d;
	FilterData* data;
	d.node = vsapi->mapGetNode(in, "clip", 0, 0);
	d.vi = vsapi->getVideoInfo(d.node);
	data = (FilterData*)malloc(sizeof(d));
	*data = d;
	VSFilterDependency deps[] = { {d.node, rpGeneral} }; /* Depending the the request patterns you may want to change this */
	vsapi->createVideoFilter(out, "Softlight", data->vi, filterGetFrame, filterFree, fmParallel, deps, 1, data, core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
	vspapi->configPlugin("com.Argaricolm.Softlight", "Argaricolm", "Vapoursynth Softlight", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
	vspapi->registerFunction("Softlight", "clip:vnode;mode:int:opt;formula:int:opt;", "clip:vnode;", filterCreate, NULL, plugin);
}



