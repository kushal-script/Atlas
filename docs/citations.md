# Citations

Every augmentation choice, noise model and structural parameter in the generator, and the core algorithmic choices of the localizer, mapped to public literature. Reference list at the end; bracketed numbers link each choice to its sources.

## SEM image formation

Edge brightening via the secant law. Secondary electron yield rises with the local tilt of the surface approximately as 1 over cos of the tilt angle, because inclined surfaces give escaping secondaries a larger interaction volume near the surface. We therefore compute the height map gradient, convert it to a tilt angle, and multiply the emitted signal by sec of that angle with a clamp near grazing incidence. Bright feature edges then emerge physically instead of being painted with an edge filter. [1] [2] [3]

Material dependent yields. Secondary electron yield differs by material at fixed beam energy; oxides yield notably more than clean silicon, metals sit in between depending on energy. We assign relative yields (silicon 1.0, silicon dioxide 1.45, gate metal 1.2, tungsten 1.5, nitride 1.3) with a per pair random jitter of 10 percent, giving the material contrast visible in real SEM images while acknowledging the published spread. [1] [3]

Beam point spread function. The finite probe is modelled as an anisotropic Gaussian whose sigma is drawn from 1.5 to 2.6 nm for the high magnification capture and 4 to 12 nm for the wide field capture, with up to 35 percent astigmatic ellipticity, following the Gaussian probe treatment used in SEM image simulation for resolution metrology. [2] [4]

Poisson shot noise scaled by dose. SEM noise is dominated by counting statistics of the primary and secondary electrons, so pixel noise is Poisson in the collected signal with a Gaussian detector and amplifier contribution on top. We draw an electron dose per pixel (700 to 2000 electrons for the reference, 80 to 300 for the search capture), sample Poisson counts of signal times dose, and add Gaussian read noise of 2 to 8 electrons. The search image is therefore always the noisier capture, as the problem statement requires of the test data, and the noise fields of the two captures come from independent generators because they are separate physical scans. [5] [1] [2]

Dielectric charging. Insulating regions accumulate charge under the beam and shift their apparent brightness on the scale of the scan, so we multiply oxide regions by a smooth random field of 3 to 9 percent amplitude with 120 to 320 nm correlation length. [6] [2]

Scan drift, line jitter and vibration. Stage and column drift during frame acquisition displaces successive scan lines, and environmental vibration adds a periodic component; artificial SEM image generators include exactly these distortions. We add a smooth drift trajectory (up to 3 reference pixels, up to 1.2 search pixels over the frame), an autoregressive per line jitter and a sinusoidal vibration term, applied as per line sampling offsets before the noise stage, and the recorded ground truth accounts for them. This is also thematically the point of the exercise, since stage drift is the reason navigation error recovery exists. [4] [7] [8]

Aliasing preserved at the wide field of view. At 10 nm per pixel, fin and line pitches of 26 to 60 nm sit near the sampling limit. The wide capture is rendered on a 2 nm supersampled grid, blurred by the beam, then point sampled to the 10 nm grid, so the aliasing a real point sampling scan produces is preserved instead of suppressed by area averaging. [2] [8]

Tone mapping. Operators set brightness and contrast per capture, so each image is percentile normalized with random headroom before 8 bit quantization, which decorrelates the gray scales of the two captures. [1]

## DRAM structural parameters

Cell geometry. The layout follows the 6F2 buried word line DRAM cell: word line pitch 2F, bit line pitch 3F, with F drawn from 16 to 22 nm to span reported product nodes. Storage node contacts sit between bit lines at half pitch offsets with 4 percent size variation and a small missing contact probability as defects. [9] [10] [11]

Array organisation. DRAM arrays are multi divided into mats bounded by bit line sense amplifier stripes and local word line driver stripes, with local lines typically 256 to 512 cells long; shorter local lines appear in speed optimised designs. We draw mat sizes of 4.5 to 9 um, corresponding to roughly 128 to 250 cells per local line at the simulated feature sizes, with stripe widths of 300 to 560 nm filled with quasi random periphery blocks. These stripes are the aperiodic anchors that make localization inside an otherwise repeating array possible at all. [10] [11] [23]

Defect density. Storage node contacts carry 4 percent size variation and a small missing contact probability. The upper part of the probability range reflects that navigation error recovery by definition happens at sites an inspection tool chose to revisit, which are disproportionately sites with measurable anomalies. [9] [11]

## FinFET structural parameters

Fin and gate grids. Fin pitch is drawn from 26 to 36 nm and contacted gate pitch from 50 to 60 nm, spanning reported 10 nm and 7 nm class technologies; fin width is 30 to 40 percent of pitch and fin height 46 nm within the reported 40 to 55 nm range. [12] [13] [14]

Standard cell structure. Logic is organised in rows of 6 to 9 fins with cells of 2 to 9 gate pitches, diffusion breaks of 0.8 gate pitch at cell boundaries, trench contacts between gates with 60 percent occupancy and sparse vias, following published standard cell construction for FinFET nodes. One rectangular SRAM block is rendered perfectly regular to provide the highly periodic hard region the test set is stated to contain. [14] [12]

Line edge roughness. Fin, gate, word line and bit line edges carry correlated roughness with sigma 1 to 2.4 nm and correlation length 15 to 40 nm, matching reported LER magnitudes and correlation lengths for litho and etch defined lines. [15] [16] [17]

## Localizer choices

Normalized cross correlation. NCC is the standard robust similarity for template localization under linear intensity changes, computed with the FFT accelerated formulation. [18]

Rotation and scale search. The relative pose between captures is handled by a hypothesis grid over rotation and scale with coarse to fine refinement, the discretised counterpart of FFT based rotation and scale registration. [19] [20]

Why not sparse features. Keypoint descriptors such as SIFT are ambiguous on repeating structures because hundreds of near identical keypoints exist per frame, which is the documented failure regime for periodic patterns; dense correlation with a tie break rule is the appropriate tool. [21]

Matched formation template. Blurring the reference to the search optics resolution and point sampling it onto the search grid reproduces the degradation chain of the search image, following the matched filter principle that correlation is optimal when the template matches the observed signal formation. [18] [2]

## References

All entries verified against the publisher or an authoritative index.

1. J. I. Goldstein, D. E. Newbury, J. R. Michael, N. W. M. Ritchie, J. H. J. Scott, D. C. Joy, Scanning Electron Microscopy and X Ray Microanalysis, 4th edition, Springer, New York, 2018. ISBN 978 1 4939 6674 5.
2. L. Reimer, Scanning Electron Microscopy: Physics of Image Formation and Microanalysis, 2nd edition, Springer Series in Optical Sciences vol. 45, Springer, Berlin, 1998. ISBN 978 3 540 63976 3.
3. H. Seiler, Secondary electron emission in the scanning electron microscope, Journal of Applied Physics, vol. 54, no. 11, pp. R1 to R18, 1983. doi 10.1063/1.332840.
4. P. Cizmar, A. E. Vladar, B. Ming, M. T. Postek, Simulated SEM images for resolution measurement, Scanning, vol. 30, no. 5, pp. 381 to 391, 2008. doi 10.1002/sca.20120.
5. F. Timischl, M. Date, S. Nemoto, A statistical model of signal noise in scanning electron microscopy, Scanning, vol. 34, no. 3, pp. 137 to 144, 2012. doi 10.1002/sca.20282.
6. J. Cazaux, Charging in scanning electron microscopy from inside and outside, Scanning, vol. 26, no. 4, pp. 181 to 203, 2004. doi 10.1002/sca.4950260406.
7. P. Cizmar, A. E. Vladar, M. T. Postek, Real time scanning charged particle microscope image composition with correction of drift, Microscopy and Microanalysis, vol. 17, no. 2, pp. 302 to 308, 2011. doi 10.1017/S1431927610094250.
8. M. T. Postek, A. E. Vladar, Does your SEM really tell the truth? How would you know? Part 1, Scanning, vol. 35, no. 6, pp. 355 to 361, 2013. doi 10.1002/sca.21075. Part 3 of the series, Proc. SPIE 9636, Scanning Microscopies, 2015, doi 10.1117/12.2195512, covers vibration and drift artifacts specifically.
9. T. Schloesser et al., 6F2 buried wordline DRAM cell for 40nm and beyond, IEEE International Electron Devices Meeting, pp. 809 to 812, 2008. doi 10.1109/IEDM.2008.4796820.
10. K. Itoh, VLSI Memory Chip Design, Springer Series in Advanced Microelectronics vol. 5, Springer, Berlin, 2001. ISBN 978 3 540 67820 5.
11. B. Jacob, S. W. Ng, D. T. Wang, Memory Systems: Cache, DRAM, Disk, Morgan Kaufmann, Burlington, 2007. ISBN 978 0 12 379751 3.
12. D. Hisamoto, W. C. Lee, J. Kedzierski, H. Takeuchi, K. Asano, C. Kuo, E. Anderson, T. J. King, J. Bokor, C. Hu, FinFET, a self aligned double gate MOSFET scalable to 20 nm, IEEE Transactions on Electron Devices, vol. 47, no. 12, pp. 2320 to 2325, 2000.
13. S. Natarajan et al., A 14nm logic technology featuring 2nd generation FinFET, air gapped interconnects, self aligned double patterning and a 0.0588 um2 SRAM cell size, IEEE International Electron Devices Meeting, 2014. Reports 42 nm fin pitch, 70 nm contacted gate pitch, 42 nm fin height.
14. C. Auth et al., A 10nm high performance and low power CMOS technology featuring 3rd generation FinFET transistors, Self Aligned Quad Patterning, contact over active gate and cobalt local interconnects, IEEE International Electron Devices Meeting, 2017. Reports 34 nm fin pitch, 54 nm contacted gate pitch, 46 nm fin height, 7 nm fin width, single dummy gate diffusion break.
15. V. Constantoudis, G. P. Patsis, A. Tserepi, E. Gogolides, Quantification of line edge roughness of photoresists. II. Scaling and fractal analysis and the best roughness descriptors, Journal of Vacuum Science and Technology B, vol. 21, no. 3, 2003.
16. C. A. Mack, Fundamental Principles of Optical Lithography: The Science of Microfabrication, John Wiley and Sons, Chichester, 2007. ISBN 978 0 470 01893 4. Section 9.8 reports 3 sigma LER of about 4 nm as common and specifications near 5 percent of nominal CD.
17. IEEE, International Roadmap for Devices and Systems, 2021 Update, Lithography chapter, 2021. Table LITH 1 gives metal line width roughness 3 sigma targets of 1.8 nm in 2022 falling to 1.2 nm by 2028.
18. J. P. Lewis, Fast Template Matching, Proceedings of Vision Interface 95, pp. 120 to 123, 1995. Expanded version circulated as Fast Normalized Cross Correlation.
19. B. S. Reddy, B. N. Chatterji, An FFT based technique for translation, rotation, and scale invariant image registration, IEEE Transactions on Image Processing, vol. 5, no. 8, pp. 1266 to 1271, 1996. doi 10.1109/83.506761.
20. C. D. Kuglin, D. C. Hines, The phase correlation image alignment method, Proceedings IEEE International Conference on Cybernetics and Society, pp. 163 to 165, 1975.
21. D. G. Lowe, Distinctive image features from scale invariant keypoints, International Journal of Computer Vision, vol. 60, no. 2, pp. 91 to 110, 2004. doi 10.1023/B:VISI.0000029664.99615.94.
22. L. T. Clark, V. Vashishtha, L. Shifren, A. Gujja, S. Sinha, B. Cline, C. Ramamurthy, G. Yeric, ASAP7: A 7 nm finFET predictive process design kit, Microelectronics Journal, vol. 53, pp. 105 to 115, 2016. Reports 27 nm fin pitch, 54 nm contacted poly pitch, 7.5 track standard cells.
23. T. Vogelsang, Understanding the energy consumption of dynamic random access memories, Proceedings of the 43rd Annual IEEE ACM International Symposium on Microarchitecture, 2010. Describes mats bounded by bit line sense amplifier stripes and local word line driver stripes with local lines typically 256 to 512 cells long.
