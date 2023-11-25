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
Same as first. But plane S is also boosted (softlight is done for each pixel with itself).

2 mode:
Same as first but without brightness restoration. Use it if you want to make brigtness also average (makes dark frames brighter).

3 mode:
Same as mode 3 but each of RGB planes are boosted using softlight.

4 mode:
YUV->RGB->softlight each RGB plane with itself->YUV (color/contrast boost)

5 mode:
YUV->RGB->HSV->boost S->RGB->YUV

You can use 3 different softlight formulas:
formula = 0,1,2
0 - pegtop
1 - illusions.hu
2 - W3C

Photoshop formula was removed because of discontinuity of local contrast.

Formulas are explained here: https://en.wikipedia.org/wiki/Blend_modes