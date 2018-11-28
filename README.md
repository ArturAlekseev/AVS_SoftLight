# AVS_SoftLight
AviSynth Softlight plugin

This plugin has 3 modes so far.
Usage:
SoftLight() - default mode 0
SoftLight(1)
SoftLight(2)

Mode 0 - process only color channels.
Mode 1 - process color channels and luma channel.
Mode 2 - process only luma channel.

For color channels it makes softlight blend over original and negative of original average.
For luma channel it just makes softlight blend original with original.
