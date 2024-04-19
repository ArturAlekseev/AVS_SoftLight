# AVS_SoftLight
AviSynth Softlight plugin

Realization of cuda soflight negative average blend.
Plugin is x64 (CUDA toolkit 12.3)

You could see on youtube videos about removing color cast using photoshops softlight blend of negative average. This is a cuda realization of it that process every frame.

Input is YUV420.

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

1 mode:
Same as mode 0 but planes S & V restored to their original values. So this mode only normalizes lightness/brightness and does not change colors.

2 mode:
Same as mode 0. But plane S is also boosted (softlight is done for each pixel with itself).

3 mode:
Same as first but without brightness restoration. Use it if you want to make brigtness also average (makes dark frames brighter).

4 mode:
Same as mode 3 but each of RGB planes are boosted using softlight.

5 mode:
YUV->RGB->softlight each RGB plane with itself->YUV (color/contrast boost)

6 mode:
YUV->RGB->HSV->boost S->RGB->YUV

8 mode:
TV to PC color range conversion (use it on videos where you see no total black and only grays).

10 mode:
Grayscale.
For RGB32 - this mode uses RGB -> YUV444 -> RGB cuda conversion. U & V planes are set to 128.
For YUV - just U & V planes are set to 128 without cuda.

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

Skipblack option:
This option is used only when average is calculated.
By default this option is true (1). When true - color level 0 (in 0-255) will be skipped from average calculation. This results in more "correct" average (and will not count black bars if they are 0 black). But will be a little slower.