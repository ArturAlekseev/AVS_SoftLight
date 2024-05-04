# AVS_SoftLight
AviSynth Softlight plugin

Realization of cuda soflight negative average blend.
Plugin is x64 (CUDA toolkit 12.4 & 11.8)

You could see on youtube videos about removing color cast using photoshops softlight blend of negative average. This is a cuda realization of it that process every frame.

There are 6 modes:

Softlight(0)-Softlight(5)

Mode 0 is default.

I'll explain first mode in detail.

Steps done in first mode:
1. YUV->RGB conversion
2. Calculates sums of all pixels in R,G,B planes (for each).
3. Get average from these sums (sum / number of pixels).
4. Get negative from this sum (255 - sum)
5. Use softlight blend of each plane with above negative. After this step we have same as photoshop does. But brightness of frame will be changed. To have brightness intact we need to restore it to original. That what other steps do.
6. We get HSV planes. V plane from orignal image (RGB > V). And HS from result after softlight. Then we do HS(changed) + V(original) -> RGB -> YUV

So first mode will neutralize only colors (hue + saturation) in frame and not brightness (volume).

Also keep in mind that you need to remove black bars in video for correct processing (if there are any). Or they will affect average sum.

1 mode: Same as mode 0 but planes S & V restored to their original values. So this mode only normalizes lightness/brightness and does not change colors.

2 mode: Same as mode 0. But plane S is also boosted (softlight is done for each pixel with itself).

3 mode: Same as first but without brightness restoration. Use it if you want to make brigtness also average (makes dark frames brighter).

4 mode: Same as mode 3 but each of RGB planes are boosted using softlight.

5 mode: YUV->RGB->softlight each RGB plane with itself->YUV (color/contrast boost)

6 mode: YUV->RGB->HSV->boost S->RGB->YUV

8 mode: TV to PC color range conversion (use it on videos where you see no total black and only grays).

10 mode: Grayscale.

For RGB32 - this mode uses RGB -> YUV444 -> RGB cuda conversion. U & V planes are set to 128 (and 512 on 10 bit).

For YUV - just U & V planes are set to 128 (or 512 for 10bit) without cuda.


You can use 3 different softlight formulas:

formula = 0,1,2

0 - pegtop

1 - illusions.hu

2 - W3C

In my opinion - pegtop fomula is the best.

Also mode 1 & mode 3 are my favourite.

Photoshop formula was removed because of discontinuity of local contrast.

Formulas are explained here: https://en.wikipedia.org/wiki/Blend_modes


Usage AviSynth:

Softlight() same as SoftLight(0,0,1) same as SoftLight(mode=0,formula=0,skipblack=1)


Usage VapourSynth:

video = core.Argaricolm.Softlight(video) or core.Argaricolm.Softlight(video,mode=0,formula=0,skipblack=1)

Skipblack option is a new enhancement for averate calculation. By default skipblack = 0 and it means it is activated.

To disable it - set it to anything not zero (like 1).

What it does is calculates how many channel elements are zero. Then they will not be counted in average calculation.

Example:

Original: (1 + 2 + 0) / 3 = 1 average

With skipblack enabled: (1 + 2 + 0) / 2 = 1.5 average


Color modes supported so far:

Avisynth:

* Planar YUV 420 8 bit and 10 bit (YUV420P8, YUV420P10)
* Planar YUV 444 8 bit and 10 bit (YUV444P8, YUV444P10)
* Not planar RGB32 (BGR32) - this one is default you get by using ConvertToRGB() or ConvertToRGB32()
* Planar RGB 8 bit and 10 bit (you get it by using ConvertToPlanarRGB()

Same for VapourSynth except BGR32 (Fredrik "asked" not to implement it in VapourSynth plugins)


Compilation:

Install CUDA Toolkit and just compile.

If you want to compile for 11.8 version just change 12.4 to 11.8 in vcxproj file.