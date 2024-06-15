# AVS_SoftLight
AviSynth and VapourSynth Softlight plugin

Realization of CUDA soflight negative average blend.

Plugin is x64 (CUDA toolkit 12.4 & 11.8)

You could see on Youtube videos about removing color cast using Photoshops softlight blend of negative average. This is a CUDA realization of it that process every frame.

Input should be in PC color range (output will be too)! Use mode 8 & 9 to convert to full range and back.

And I suggest to remove noise from input before processing.

**Parameters:**

Softlight(mode, formula, skipblack, yuvin, yuvout, rangemin, rangemax)

All parameters are optional.

mode = 0,1,2,3,4,5,6,8,9,10,11,12 (7 is not used so far, 0 is default)

Can be used like this: Softlight() same as Softlight(0)

**Mode 0** (default):
1. YUV->RGB conversion
2. Calculates sums of all pixels in R,G,B planes (for each plane).
3. Get average from these sums (sum / number of pixels).
4. Get negative from this sum (255 - sum)
5. Use softlight blend of each plane with above negative. After this step we have same as photoshop does. But brightness of frame will be changed. To have brightness intact we need to restore it to original. That what other steps do.
6. We get HSV planes. V plane from orignal image (RGB => V). And HS from result after softlight. Then we do HS(changed) + V(original) -> RGB -> YUV

So first mode will neutralize only colors (hue + saturation) in frame and not brightness (volume).

Also keep in mind that you better remove black bars in video for correct processing (if there are any). Or they will affect average sum.

**1 mode:** Same as mode 0 but planes S & V restored to their original values. So this mode only normalizes lightness/brightness and does not change colors.

**2 mode:** Same as mode 0. But plane S is also boosted (softlight is done for each pixel with itself). So it neutralises colors and boost contrast.

**3 mode:** Same as mode 0 but without brightness restoration. Use it if you want to make brigtness also average (makes dark frames brighter).

**4 mode:** Same as mode 3 but each of RGB planes are boosted using softlight (contrast boost).

**5 mode:** YUV->RGB->softlight each RGB plane with itself->YUV (color/contrast boost).

**6 mode:** YUV->RGB->HSV->boost S->RGB->YUV (boost saturation).

**8 mode:** TV to PC color range conversion (use it on videos where you see no total black and only grays). Or check video using ShowChannels plugin (if minimum in Y is 16 or 15 - then your source is in limited range). You can change input levels using rangemin & rangemax params. They are used only in this mode. If they are not specified or wrong (rangemin>=rangemax) then default will be used.

**9 mode:** PC to TV color range conversion

**10 mode:** Grayscale.

For RGB32 - this mode uses RGB -> YUV444 -> RGB cuda conversion. U & V planes are set to 128 (and 512 on 10 bit).

For YUV - just U & V planes are set to 128 (or 512 for 10bit) without cuda.

**11 mode:** OETF function is applied to each pixel.

**12 mode:** EOTF function is applied to each pixel.


You can use 3 different softlight formulas:

**formula** = 0,1,2

0 - pegtop

1 - illusions.hu

2 - W3C

In my opinion - pegtop fomula is the best.

Also mode 1 & mode 3 are my favourite.

Photoshop formula was removed because of discontinuity of local contrast.

Formulas are explained here: https://en.wikipedia.org/wiki/Blend_modes


**Usage in AviSynth:**

Softlight() same as SoftLight(0,0,0) same as SoftLight(mode=0,formula=0,skipblack=0,yuvin=0,yuvout=0)


**Usage in VapourSynth:**

video = core.Argaricolm.Softlight(video) or core.Argaricolm.Softlight(video,mode=0,formula=0,skipblack=0)


**Skipblack** option is a new enhancement for averate calculation. By default skipblack = 0 and it means it is activated.

To disable it - set it to anything not zero (like 1).

What it does is calculates how many plane (channel) elements are zero. Then they will not be counted in average calculation.

Example:

Original: (1 + 2 + 0) / 3 = 1 average

With skipblack enabled: (1 + 2 + 0) / 2 = 1.5 average


**Color modes** supported so far:

Avisynth:

* Planar YUV 420 8 bit and 10 bit (YUV420P8, YUV420P10)
* Planar YUV 444 8 bit and 10 bit (YUV444P8, YUV444P10)
* Not planar RGB32 (BGR32) - this one is default you get by using ConvertToRGB() or ConvertToRGB32()
* Planar RGB 8 bit and 10 bit (you get it by using ConvertToPlanarRGB()

Same for VapourSynth except BGR32 (Fredrik "asked" not to implement it in VapourSynth plugins)


**yuvin & yuvout** options are used for modes where yuv <-> rgb conversion is used and they define formula used for decode and encode

0 = Default is Rec.709.

Or you can select 601, 709, 2020. Like Softlight(yuvin=601,yuvout=601)


About **OETF & EOTF** functions.

They are added just to play with. EOTF is a reverse of OETF.

Try OETF function when your source is converted to PC range. To convert to PC range use Softlight(8). If result after OETF lacks of contrast then try to change black level higher that 16 like so:

Softlight(8, rangemin=16, rangemax=235)

Softlight(11)


**Compilation:**

Install CUDA Toolkit and just compile.

If you want to compile for 11.8 version just change 12.4 to 11.8 in vcxproj file.