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
	SoftLight(PClip _child, int mode, int formula, int skipblack, int yuvin, int yuvout, int rangemin, int rangemax, int changerange, IScriptEnvironment* env);
	~SoftLight();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	int changerange = 0;
	int mode = 0;
	int softlightFormula = 0;
	bool cuda = false;
	int skipblack = 0;
	int yuvin = 0;
	int yuvout = 0;
	int rangemin = 0;
	int rangemax = 0;
	unsigned int maxthreads = 0;
};

extern void CudaBoostSaturationRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int formula, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeRGB32withLight(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack, int changerange, int rangemin, int rangemax);
extern void CudaTV2PCRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int rangemin, int rangemax);
extern void CudaPC2TVRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaTVClampRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaGrayscaleRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads, int yuvin, int yuvout);
extern void CudaOETFRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaEOTFRGB32(unsigned char* plane, int planeheight, int planewidth, int planepitch, int threads);

extern void CudaNeutralizeYUV420byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeYUV420byRGB10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeYUV420byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeYUV420byRGBwithLight10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaBoostSaturationYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaBoostSaturationYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaTV2PCYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax);
extern void CudaTV2PCYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax);
extern void CudaPC2TVYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaTVClampYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaPC2TVYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaTVClampYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaOETFYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaEOTFYUV420(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaOETFYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaEOTFYUV42010(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);

extern void CudaNeutralizeYUV444byRGB(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeYUV444byRGB10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeYUV444byRGBwithLight(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeYUV444byRGBwithLight10(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int type, int formula, int skipblack, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaBoostSaturationYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaBoostSaturationYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int formula, int yuvin, int yuvout, int changerange, int rangemin, int rangemax);
extern void CudaTV2PCYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax);
extern void CudaTV2PCYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout, int rangemin, int rangemax);
extern void CudaPC2TVYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaTVClampYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaPC2TVYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaTVClampYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaOETFYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaEOTFYUV444(unsigned char* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned char* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned char* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaOETFYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);
extern void CudaEOTFYUV44410(unsigned short* planeY, int planeYheight, int planeYwidth, int planeYpitch, unsigned short* planeU, int planeUheight, int planeUwidth, int planeUpitch, unsigned short* planeV, int planeVheight, int planeVwidth, int planeVpitch, int threads, int yuvin, int yuvout);

//VapourSynth RGB24 only (planar)
extern void CudaNeutralizeRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeRGBwithLight(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack, int changerange, int rangemin, int rangemax);
extern void CudaBoostSaturationRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int formula, int changerange, int rangemin, int rangemax);
extern void CudaTV2PCRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int rangemin, int rangemax);
extern void CudaPC2TVRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaTVClampRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaGrayscaleRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads, int yuvin, int yuvout);
extern void CudaOETFRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaEOTFRGB(unsigned char* planeR, unsigned char* planeG, unsigned char* planeB, int planeheight, int planewidth, int planepitch, int threads);

extern void CudaNeutralizeRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack, int changerange, int rangemin, int rangemax);
extern void CudaNeutralizeRGBwithLight10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int type, int formula, int skipblack, int changerange, int rangemin, int rangemax);
extern void CudaBoostSaturationRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int formula, int changerange, int rangemin, int rangemax);
extern void CudaTV2PCRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int rangemin, int rangemax);
extern void CudaPC2TVRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaTVClampRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaGrayscaleRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads, int yuvin, int yuvout);
extern void CudaOETFRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads);
extern void CudaEOTFRGB10(unsigned short* planeR, unsigned short* planeG, unsigned short* planeB, int planeheight, int planewidth, int planepitch, int threads);

SoftLight::SoftLight(PClip _child, int mode, int formula, int skipblack, int yuvin, int yuvout, int rangemin, int rangemax, int changerange, IScriptEnvironment* env) : GenericVideoFilter(_child) {
	
	SoftLight::mode = mode;
	SoftLight::softlightFormula = formula;
	SoftLight::skipblack = skipblack;
	SoftLight::yuvin = yuvin;
	SoftLight::yuvout = yuvout;
	SoftLight::rangemin = rangemin;
	SoftLight::rangemax = rangemax;
	SoftLight::changerange = changerange;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus == cudaSuccess) {
		SoftLight::cuda = true;
		cudaDeviceProp prop;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);
		SoftLight::maxthreads = prop.maxThreadsPerBlock;
	} else env->ThrowError("SoftLight: CUDA failed");
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
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 1: //YUV -> neutralize(RGB) -> HSV -> restore HS -> RGB -> YUV (normalize lightness/volume)
		{
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
		{
			CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 3: //YUV -> neutralize(RGB) -> YUV
		{
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 4:
		{
			//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 5:
		{
			//YUV -> softlight RGB with RGB -> YUV
			CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 6: //boost saturation using softlight
		{
			CudaBoostSaturationYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
			break;
		}
		case 7:
		{
			CudaTVClampYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
			break;
		}
		case 8:
		{
			CudaTV2PCYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
			break;
		}
		case 9:
		{
			CudaPC2TVYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
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
		case 11: {
			CudaOETFYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
			break;
		}
		case 12: {
			CudaEOTFYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
			break;
		}
		}
	}
	else
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
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV (neutralize lightness)
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV (neutralize colors and boost saturation)
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
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
			case 11:
			{
				CudaOETFYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
					break;
			}
			case 12:
			{
				CudaEOTFYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			}
		}
		else
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
					CudaNeutralizeRGB32(plane, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
					break;
				}
				case 1:
				{
					CudaNeutralizeRGB32(plane, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
					break;
				}
				case 2:
				{
					CudaNeutralizeRGB32(plane, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
					break;
				}
				case 3:
				{
					CudaNeutralizeRGB32withLight(plane, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
					break;
				}
				case 4:
				{
					CudaNeutralizeRGB32withLight(plane, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
					break;
				}
				case 5:
				{
					CudaNeutralizeRGB32withLight(plane, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
					break;
				}
				case 6:
				{
					CudaBoostSaturationRGB32(plane, planeheight, planewidth, planepitch, maxthreads, softlightFormula, changerange, rangemin, rangemax);
					break;
				}
				case 7:
				{
					CudaTVClampRGB32(plane, planeheight, planewidth, planepitch, maxthreads);
					break;
				}
				case 8:
				{
					CudaTV2PCRGB32(plane, planeheight, planewidth, planepitch, maxthreads, rangemin, rangemax);
					break;
				}
				case 9:
				{
					CudaPC2TVRGB32(plane, planeheight, planewidth, planepitch, maxthreads);
					break;
				}
				case 10:
				{
					CudaGrayscaleRGB32(plane, planeheight, planewidth, planepitch, maxthreads, yuvin, yuvout);
					break;
				}
				case 11:
				{
					CudaOETFRGB32(plane, planeheight, planewidth, planepitch, maxthreads);
					break;
				}
				case 12:
				{
					CudaEOTFRGB32(plane, planeheight, planewidth, planepitch, maxthreads);
					break;
				}
				}

			}
			else
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
						CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
						break;
					}
					case 1:
					{
						CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
						break;
					}
					case 2:
					{
						CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
						break;
					}
					case 3:
					{
						CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
						break;
					}
					case 4:
					{
						CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
						break;
					}
					case 5:
					{
						CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
						break;
					}
					case 6:
					{
						CudaBoostSaturationRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, softlightFormula, changerange, rangemin, rangemax);
						break;
					}
					case 7:
					{
						CudaTVClampRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
						break;
					}
					case 8:
					{
						CudaTV2PCRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, rangemin, rangemax);
						break;
					}
					case 9:
					{
						CudaPC2TVRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
						break;
					}
					case 10:
					{
						CudaGrayscaleRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, yuvin, yuvout);
						break;
					}
					case 11:
					{
						CudaOETFRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
						break;
					}
					case 12:
					{
						CudaEOTFRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
						break;
					}
					}
				}
				else
					if (vi.IsPlanar() && vi.Is420() && vi.BitsPerComponent() == 10 && SoftLight::cuda) {
						env->MakeWritable(&src);

						int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

						unsigned short* planeY = (unsigned short*)src->GetWritePtr(planes[0]);
						unsigned short* planeU = (unsigned short*)src->GetWritePtr(planes[1]);
						unsigned short* planeV = (unsigned short*)src->GetWritePtr(planes[2]);

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
							CudaNeutralizeYUV420byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 1: //YUV -> neutralize(RGB) -> HSV -> restore HS -> RGB -> YUV (normalize lightness/volume)
						{
							CudaNeutralizeYUV420byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
						{
							CudaNeutralizeYUV420byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 3: //YUV -> neutralize(RGB) -> YUV
						{
							CudaNeutralizeYUV420byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 4:
						{
							//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
							CudaNeutralizeYUV420byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 5:
						{
							//YUV -> softlight RGB with RGB -> YUV
							CudaNeutralizeYUV420byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 6: //boost saturation using softlight
						{
							CudaBoostSaturationYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
							break;
						}
						case 7:
						{
							CudaTVClampYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
							break;
						}
						case 8:
						{
							CudaTV2PCYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
							break;
						}
						case 9:
						{
							CudaPC2TVYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
							break;
						}
						case 10: //Grayscale
						{
							for (int h = 0; h != planeUheight; h++)
								for (int w = 0; w != planeUwidth / 2; w++)
									planeU[h * planeUpitch / 2 + w] = 512;
							for (int h = 0; h != planeVheight; h++)
								for (int w = 0; w != planeVwidth / 2; w++)
									planeV[h * planeVpitch / 2 + w] = 512;
							break;
						}
						case 11: {
							CudaOETFYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
							break;
						}
						case 12: {
							CudaEOTFYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
							break;
						}
						}
					}
					else
						if (vi.IsPlanar() && vi.Is444() && vi.BitsPerComponent() == 10 && SoftLight::cuda) {
							env->MakeWritable(&src);

							int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

							unsigned short* planeY = (unsigned short*)src->GetWritePtr(planes[0]);
							unsigned short* planeU = (unsigned short*)src->GetWritePtr(planes[1]);
							unsigned short* planeV = (unsigned short*)src->GetWritePtr(planes[2]);

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
								CudaNeutralizeYUV444byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 1: //YUV -> neutralize(RGB) -> HSV -> restore HS -> RGB -> YUV (normalize lightness/volume)
							{
								CudaNeutralizeYUV444byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
							{
								CudaNeutralizeYUV444byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 3: //YUV -> neutralize(RGB) -> YUV
							{
								CudaNeutralizeYUV444byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 4:
							{
								//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
								CudaNeutralizeYUV444byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 5:
							{
								//YUV -> softlight RGB with RGB -> YUV
								CudaNeutralizeYUV444byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 6: //boost saturation using softlight
							{
								CudaBoostSaturationYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
								break;
							}
							case 7:
							{
								CudaTVClampYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
								break;
							}
							case 8:
							{
								CudaTV2PCYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
								break;
							}
							case 9:
							{
								CudaPC2TVYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
								break;
							}
							case 10: //Grayscale
							{
								for (int h = 0; h != planeUheight; h++)
									for (int w = 0; w != planeUwidth / 2; w++)
										planeU[h * planeUpitch / 2 + w] = 512;
								for (int h = 0; h != planeVheight; h++)
									for (int w = 0; w != planeVwidth / 2; w++)
										planeV[h * planeVpitch / 2 + w] = 512;
								break;
							}
							case 11: {
								CudaOETFYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
								break;
							}
							case 12: {
								CudaEOTFYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
								break;
							}
							}
						}
						else
							if (vi.IsPlanarRGB() && vi.BitsPerComponent() == 10 && SoftLight::cuda) {
								env->MakeWritable(&src);

								int planes[] = { PLANAR_R, PLANAR_G, PLANAR_B };
								unsigned short* planeR = (unsigned short*)src->GetWritePtr(planes[0]);
								unsigned short* planeG = (unsigned short*)src->GetWritePtr(planes[1]);
								unsigned short* planeB = (unsigned short*)src->GetWritePtr(planes[2]);
								int planepitch = src->GetPitch(0);
								int planeheight = src->GetHeight(0);
								int planewidth = src->GetRowSize(0);
								switch (mode) {
									case 0:
									{
										CudaNeutralizeRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
										break;
									}
									case 1:
									{
										CudaNeutralizeRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
										break;
									}
									case 2:
									{
										CudaNeutralizeRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
										break;
									}
									case 3:
									{
										CudaNeutralizeRGBwithLight10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
										break;
									}
									case 4:
									{
										CudaNeutralizeRGBwithLight10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
										break;
									}
									case 5:
									{
										CudaNeutralizeRGBwithLight10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
										break;
									}
									case 6:
									{
										CudaBoostSaturationRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, softlightFormula, changerange, rangemin, rangemax);
										break;
									}
									case 7:
									{
										CudaTVClampRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
										break;
									}
									case 8:
									{
										CudaTV2PCRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, rangemin, rangemax);
										break;
									}
									case 9:
									{
										CudaPC2TVRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
										break;
									}
									case 10:
									{
										CudaGrayscaleRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, yuvin, yuvout);
										break;
									}
									case 11: {
										CudaOETFRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
										break;
									}
									case 12: {
										CudaEOTFRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
										break;
									}
								}
							}
							else env->ThrowError("SoftLight: not supported input format.");
	return src;
}

AVSValue __cdecl CreateSoftLight(AVSValue args, void* user_data, IScriptEnvironment* env) {
	return new SoftLight(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), args[3].AsBool(), args[4].AsInt(), args[5].AsInt(), args[6].AsInt(), args[7].AsInt(), args[8].AsInt(), env);
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("SoftLight", "c[mode]i[formula]i[skipblack]i[yuvin]i[yuvout]i[rangemin]i[rangemax]i[changerange]i", CreateSoftLight, 0);
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
static int skipblack = true;
static int yuvin = 0;
static int yuvout = 0;
static int rangemin = 0;
static int rangemax = 0;
static int changerange = 0;

static const VSFrame* VS_CC filterGetFrame(int n, int activationReason, void* instanceData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
	FilterData* d = (FilterData*)instanceData;
	int dsize = sizeof(FilterData);

	unsigned char* params = (unsigned char*)instanceData + dsize;

	mode = *(unsigned int*)(params);
	softlightFormula = *(unsigned int*)(params + 4 * 1);
	skipblack = *(unsigned int*)(params + 4 * 2);
	yuvin = *(unsigned int*)(params + 4 * 3);
	yuvout = *(unsigned int*)(params + 4 * 4);
	rangemin = *(unsigned int*)(params + 4 * 5);
	rangemax = *(unsigned int*)(params + 4 * 6);
	changerange = *(unsigned int*)(params + 4 * 7);

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
				CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV (neutralize lightness)
			{
				CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV (neutralize colors and boost saturation)
			{
				CudaNeutralizeYUV420byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV420byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula,yuvin,yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads,yuvin,yuvout, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
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
			case 11: {
				CudaOETFYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 12: {
				CudaEOTFYUV420(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			}

		}
		else
		//yuv420p10 (3 = yuv, 0 = integer, 10 bit, 1,1) / pfYUV420P8 = VS_MAKE_VIDEO_ID(cfYUV, stInteger, 10, 1, 1)
		if (fi->colorFamily == 3 && fi->sampleType == 0 && fi->bitsPerSample == 10 && fi->subSamplingH == 1 && fi->subSamplingW == 1 && cuda)
		{

			const uint16_t* planeYs = (uint16_t*)vsapi->getReadPtr(src, 0);
			const uint16_t* planeUs = (uint16_t*)vsapi->getReadPtr(src, 1);
			const uint16_t* planeVs = (uint16_t*)vsapi->getReadPtr(src, 2);

			//our functions change passed data, so we must first copy source to destination and then pass destination to functions
			uint16_t* planeY = (uint16_t*)vsapi->getWritePtr(dst, 0);
			uint16_t* planeU = (uint16_t*)vsapi->getWritePtr(dst, 1);
			uint16_t* planeV = (uint16_t*)vsapi->getWritePtr(dst, 2);

			int planeYpitch = (int)vsapi->getStride(src, 0);
			int planeYheight = (int)vsapi->getFrameHeight(src, 0);
			int planeYwidth = (int)vsapi->getFrameWidth(src, 0) * 2;
			int planeUpitch = (int)vsapi->getStride(src, 1);
			int planeUheight = (int)vsapi->getFrameHeight(src, 1);
			int planeUwidth = (int)vsapi->getFrameWidth(src, 1) * 2;
			int planeVpitch = (int)vsapi->getStride(src, 2);
			int planeVheight = (int)vsapi->getFrameHeight(src, 2);
			int planeVwidth = (int)vsapi->getFrameWidth(src, 2) * 2;

			memcpy(planeY, planeYs, planeYheight * planeYpitch); //copy source to destination
			memcpy(planeU, planeUs, planeUheight * planeUpitch);
			memcpy(planeV, planeVs, planeVheight * planeVpitch);

			switch (mode) {
			case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV (neutralize colors)
			{
				CudaNeutralizeYUV420byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV (neutralize lightness)
			{
				CudaNeutralizeYUV420byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV (neutralize colors and boost saturation)
			{
				CudaNeutralizeYUV420byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV420byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV420byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV420byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 10: //Grayscale
			{
				for (int h = 0; h != planeUheight; h++)
					for (int w = 0; w != planeUwidth / 2; w++)
						planeU[h * planeUpitch / 2 + w] = 512;
				for (int h = 0; h != planeVheight; h++)
					for (int w = 0; w != planeVwidth / 2; w++)
						planeV[h * planeVpitch / 2 + w] = 512;
				break;
			}
			case 11: {
				CudaOETFYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 12: {
				CudaEOTFYUV42010(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			}

		}
		else
		//yuv444p8 (3 = yuv, 0 = integer, 8 bit, 0,0) / pfYUV444P8 = VS_MAKE_VIDEO_ID(cfYUV, stInteger, 8, 0, 0)
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
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
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
			case 11: {
				CudaOETFYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 12: {
				CudaEOTFYUV444(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			}

		}
		else
		//yuv444p10 (3 = yuv, 0 = integer, 10 bit, 0,0) / pfYUV444P10 = VS_MAKE_VIDEO_ID(cfYUV, stInteger, 10, 0, 0)
		if (fi->colorFamily == 3 && fi->sampleType == 0 && fi->bitsPerSample == 10 && fi->subSamplingH == 0 && fi->subSamplingW == 0 && cuda)
		{

			const uint16_t* planeYs = (uint16_t*)vsapi->getReadPtr(src, 0);
			const uint16_t* planeUs = (uint16_t*)vsapi->getReadPtr(src, 1);
			const uint16_t* planeVs = (uint16_t*)vsapi->getReadPtr(src, 2);

			//our functions change passed data, so we must first copy source to destination and then pass destination to functions
			uint16_t* planeY = (uint16_t*)vsapi->getWritePtr(dst, 0);
			uint16_t* planeU = (uint16_t*)vsapi->getWritePtr(dst, 1);
			uint16_t* planeV = (uint16_t*)vsapi->getWritePtr(dst, 2);

			int planeYpitch = (int)vsapi->getStride(src, 0);
			int planeYheight = (int)vsapi->getFrameHeight(src, 0);
			int planeYwidth = (int)vsapi->getFrameWidth(src, 0) * 2;
			int planeUpitch = (int)vsapi->getStride(src, 1);
			int planeUheight = (int)vsapi->getFrameHeight(src, 1);
			int planeUwidth = (int)vsapi->getFrameWidth(src, 1) * 2;
			int planeVpitch = (int)vsapi->getStride(src, 2);
			int planeVheight = (int)vsapi->getFrameHeight(src, 2);
			int planeVwidth = (int)vsapi->getFrameWidth(src, 2) * 2;

			memcpy(planeY, planeYs, planeYheight * planeYpitch); //copy source to destination
			memcpy(planeU, planeUs, planeUheight * planeUpitch);
			memcpy(planeV, planeVs, planeVheight * planeVpitch);

			switch (mode) {
			case 0: //YUV -> neutralize(RGB) -> HSV -> restore V -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 1: //YUV -> neutralize(RGB) -> HSV -> restore H&S -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 2: //YUV -> neutralize(RGB) -> HSV -> boost S + restore V -> RGB -> YUV
			{
				CudaNeutralizeYUV444byRGB10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 3: //YUV -> neutralize(RGB) -> YUV
			{
				CudaNeutralizeYUV444byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 0, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				//YUV -> neutralize(RGB) + softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 1, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				//YUV -> softlight RGB with RGB -> YUV
				CudaNeutralizeYUV444byRGBwithLight10(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, 2, softlightFormula, skipblack, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 6: //boost saturation using softlight
			{
				CudaBoostSaturationYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, softlightFormula, yuvin, yuvout, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 8:
			{
				CudaTV2PCYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 10: //Grayscale
			{
				for (int h = 0; h != planeUheight; h++)
					for (int w = 0; w != planeUwidth / 2; w++)
						planeU[h * planeUpitch / 2 + w] = 512;
				for (int h = 0; h != planeVheight; h++)
					for (int w = 0; w != planeVwidth / 2; w++)
						planeV[h * planeVpitch / 2 + w] = 512;
				break;
			}
			case 11: {
				CudaOETFYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 12: {
				CudaEOTFYUV44410(planeY, planeYheight, planeYwidth, planeYpitch, planeU, planeUheight, planeUwidth, planeUpitch, planeV, planeVheight, planeVwidth, planeVpitch, maxthreads, yuvin, yuvout);
				break;
			}
			}
		}
		else
		//RGB24 (2 = RGB, 0 = integer, 8 bit, 0,0) / pfRGB24 = VS_MAKE_VIDEO_ID(cfRGB, stInteger, 8, 0, 0) - same as RGBP8 in AviSynth (planar RGB)
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
				CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 1:
			{
				CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 2:
			{
				CudaNeutralizeRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 3:
			{
				CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				CudaNeutralizeRGBwithLight(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 6:
			{
				CudaBoostSaturationRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, softlightFormula, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 8:
			{
				CudaTV2PCRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 10:
			{
				CudaGrayscaleRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 11:
			{
				CudaOETFRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 12:
			{
				CudaEOTFRGB(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			}
		}
		else
		//RGB30 (2 = RGB, 0 = integer, 10 bit, 0,0) / pfRGB30 = VS_MAKE_VIDEO_ID(cfRGB, stInteger, 10, 0, 0) - same as RGBP10 in AviSynth (planar RGB)
		if (fi->colorFamily == 2 && fi->sampleType == 0 && fi->bitsPerSample == 10 && fi->subSamplingH == 0 && fi->subSamplingW == 0 && cuda)
		{

			const uint16_t* planeRs = (uint16_t*)vsapi->getReadPtr(src, 0);
			const uint16_t* planeGs = (uint16_t*)vsapi->getReadPtr(src, 1);
			const uint16_t* planeBs = (uint16_t*)vsapi->getReadPtr(src, 2);

			//our functions change passed data, so we must first copy source to destination and then pass destination to functions
			uint16_t* planeR = (uint16_t*)vsapi->getWritePtr(dst, 0);
			uint16_t* planeG = (uint16_t*)vsapi->getWritePtr(dst, 1);
			uint16_t* planeB = (uint16_t*)vsapi->getWritePtr(dst, 2);

			int planepitch = (int)vsapi->getStride(src, 0);
			int planeheight = (int)vsapi->getFrameHeight(src, 0);
			int planewidth = (int)vsapi->getFrameWidth(src, 0) * 2;

			memcpy(planeR, planeRs, planeheight * planepitch); //copy source to destination
			memcpy(planeG, planeGs, planeheight * planepitch); //copy source to destination
			memcpy(planeB, planeBs, planeheight * planepitch); //copy source to destination

			switch (mode) {
			case 0:
			{
				CudaNeutralizeRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 1:
			{
				CudaNeutralizeRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 2:
			{
				CudaNeutralizeRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 3:
			{
				CudaNeutralizeRGBwithLight10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 0, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 4:
			{
				CudaNeutralizeRGBwithLight10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 1, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 5:
			{
				CudaNeutralizeRGBwithLight10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, 2, softlightFormula, skipblack, changerange, rangemin, rangemax);
				break;
			}
			case 6:
			{
				CudaBoostSaturationRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, softlightFormula, changerange, rangemin, rangemax);
				break;
			}
			case 7:
			{
				CudaTVClampRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 8:
			{
				CudaTV2PCRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, rangemin, rangemax);
				break;
			}
			case 9:
			{
				CudaPC2TVRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 10:
			{
				CudaGrayscaleRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads, yuvin, yuvout);
				break;
			}
			case 11:
			{
				CudaOETFRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
			}
			case 12:
			{
				CudaEOTFRGB10(planeR, planeG, planeB, planeheight, planewidth, planepitch, maxthreads);
				break;
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
	skipblack = (int)vsapi->mapGetInt(in, "skipblack", 0, &error);
	yuvin = (int)vsapi->mapGetInt(in, "yuvin", 0, &error);
	yuvout = (int)vsapi->mapGetInt(in, "yuvout", 0, &error);
	rangemin = (int)vsapi->mapGetInt(in, "rangemin", 0, &error);
	rangemax = (int)vsapi->mapGetInt(in, "rangemax", 0, &error);
	changerange = (int)vsapi->mapGetInt(in, "changerange", 0, &error);

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
	unsigned char* data;
	d.node = vsapi->mapGetNode(in, "clip", 0, 0);
	d.vi = vsapi->getVideoInfo(d.node);

	int dsize = sizeof(d);
	
	//mode + formula + skipblack + yuvin + yuvout + rangemin + rangemax (4 * 7)

	data = (unsigned char*)malloc(sizeof(d) + 4 * 8);
	memcpy_s(data, dsize, &d, dsize);
	*(unsigned int*)(data + dsize + 4 * 0) = mode;
	*(unsigned int*)(data + dsize + 4 * 1) = softlightFormula;
	*(unsigned int*)(data + dsize + 4 * 2) = skipblack;
	*(unsigned int*)(data + dsize + 4 * 3) = yuvin;
	*(unsigned int*)(data + dsize + 4 * 4) = yuvout;
	*(unsigned int*)(data + dsize + 4 * 5) = rangemin;
	*(unsigned int*)(data + dsize + 4 * 6) = rangemax;
	*(unsigned int*)(data + dsize + 4 * 7) = changerange;

	VSFilterDependency deps[] = { {d.node, rpGeneral} }; /* Depending the the request patterns you may want to change this */
	vsapi->createVideoFilter(out, "Softlight", ((FilterData*)data)->vi, filterGetFrame, filterFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
	vspapi->configPlugin("com.Argaricolm.Softlight", "Argaricolm", "Vapoursynth Softlight", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
	vspapi->registerFunction("Softlight", "clip:vnode;mode:int:opt;formula:int:opt;skipblack:int:opt;yuvin:int:opt;yuvout:int:opt;rangemin:int:opt;rangemax:int:opt;changerange:int:opt", "clip:vnode;", filterCreate, NULL, plugin);
}



