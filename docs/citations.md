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

## Phase 2 degradation ladder

The addendum names the degraded set's corruption categories, charging, scan distortion, defocus, elevated shot noise and polygon scaling to twenty percent, over four undisclosed severity levels, and every category maps onto a mechanism already in the imaging model rather than onto a new image filter. The ladder scales the search capture only, following the addendum's construction in which the reference is a clean crop and the corruption arrives on the wide image; what a corrupted reference would cost was measured separately and is recorded under `experiments/20260831_degraded_reference`.

Elevated shot noise as dose reduction. Severity multiplies the electron dose down to 15 percent of nominal, which at the search capture's 80 to 300 electrons per pixel leaves 12 to 45 electrons at the hardest rung, and the Poisson statistics of the imaging model then produce the noise rather than additive noise being painted on. Single image SNR estimation work on SEM images grounds both the counting statistics treatment and the signal to noise regimes such doses imply. [5] [28] [1]

Defocus and spot growth. Severity multiplies the beam point spread sigma to three times nominal, the Gaussian probe treatment used in SEM simulation for resolution metrology, since defocus and astigmatism manifest as probe broadening at the specimen plane. [2] [4]

Charging, drift and jitter. Severity multiplies the charging amplitude to four times nominal within the published charging treatment, and pushes the per frame drift and per line jitter that artificial SEM generators include as the standard acquisition distortions. [6] [4] [7] [8]

Polygon scaling as width change at fixed pitch. Severity scales drawn feature widths by up to twenty percent while pitches stay fixed, because critical dimension is the litho and etch controlled quantity that drifts between visits while pitch is set by the patterning periodicity and does not; CD control and roughness specifications in the lithography literature and the IRDS roadmap bound how far such width excursions plausibly run. [16] [17]

The severity parameters themselves are undisclosed by the organisers, so the four rungs are this repository's own reading, spanning from mild to well past where the Phase 1 noise sat; the ladder values are recorded in the generator beside the mechanism each one scales.

## DRAM structural parameters

Cell geometry. The layout follows the 6F2 buried word line DRAM cell: word line pitch 2F, bit line pitch 3F, with F drawn from 16 to 22 nm to span reported product nodes. Storage node contacts sit between bit lines at half pitch offsets with 4 percent size variation and a small missing contact probability as defects. [9] [10] [11]

Array organisation. DRAM arrays are multi divided into mats bounded by bit line sense amplifier stripes and local word line driver stripes, with local lines typically 256 to 512 cells long; shorter local lines appear in speed optimised designs. We draw mat sizes of 4.5 to 9 um, corresponding to roughly 128 to 250 cells per local line at the simulated feature sizes, with stripe widths of 300 to 560 nm filled with quasi random periphery blocks. These stripes are the aperiodic anchors that make localization inside an otherwise repeating array possible at all. [10] [11] [23]

Defect density. Storage node contacts carry 4 percent size variation and a small missing contact probability. The upper part of the probability range reflects that navigation error recovery by definition happens at sites an inspection tool chose to revisit, which are disproportionately sites with measurable anomalies. [9] [11]

## FinFET structural parameters

Fin and gate grids. Fin pitch is drawn from 26 to 36 nm and contacted gate pitch from 50 to 60 nm, spanning reported 10 nm and 7 nm class technologies; fin width is 30 to 40 percent of pitch and fin height 46 nm within the reported 40 to 55 nm range; the ASAP7 predictive kit reports 27 nm fin pitch and 54 nm contacted poly pitch, inside the same ranges. [12] [13] [14] [22]

Standard cell structure. Logic is organised in rows of 6 to 9 fins with cells of 2 to 9 gate pitches, diffusion breaks of 0.8 gate pitch at cell boundaries, trench contacts between gates with 60 percent occupancy and sparse vias, following published standard cell construction for FinFET nodes. One rectangular SRAM block is rendered perfectly regular to provide the highly periodic hard region the test set is stated to contain. [14] [12]

Line edge roughness. Fin, gate, word line and bit line edges carry correlated roughness with sigma 1 to 2.4 nm and correlation length 15 to 40 nm, matching reported LER magnitudes and correlation lengths for litho and etch defined lines. [15] [16] [17]

## Optical modality (bonus)

Diffraction limited resolution. The brightfield point spread is modelled as a Gaussian of sigma 0.21 lambda over NA per channel, the standard Gaussian approximation of the Airy pattern, with chromatic variation following wavelength. At visible wavelengths and NA near 0.9 this is roughly 130 nm, so nanometer scale array features are unresolved and only mesoscale structure (mats, stripes, blocks) carries localization information, which matches the production division of labor between optical coarse alignment and electron beam fine addressing. [24] [25] [27]

Thin film interference color. Oxide regions are colored by two beam interference with phase 4 pi n t over lambda for film thickness t and index 1.46, which is why wafer dielectric films show thickness dependent colors under brightfield inspection. [24] [26]

Photon noise and camera effects. Per channel Poisson shot noise scaled by exposure (the wide field capture receives far fewer photons per pixel), small Gaussian read noise, radial vignetting, illumination tilt and white balance jitter model the camera chain; optical tools are full field cameras, so no scan line artifacts apply. [25] [26]

## Localizer choices

Normalized cross correlation. NCC is the standard robust similarity for template localization under linear intensity changes, computed with the FFT accelerated formulation. [18]

Rotation and scale search. The relative pose between captures is handled by a hypothesis grid over rotation and scale with coarse to fine refinement, the discretised counterpart of FFT based rotation and scale registration. [19] [20]

Why not sparse features. Keypoint descriptors such as SIFT are ambiguous on repeating structures because hundreds of near identical keypoints exist per frame, which is the documented failure regime for periodic patterns; dense correlation with a tie break rule is the appropriate tool. [21]

Sub pixel peak localization. The correlation peak is refined by fitting a low order surface to the peak and its neighbours, the standard sub pixel estimator in registration and particle image velocimetry, where fitting the peak neighbourhood is shown to reach well under a tenth of a pixel and to control the pixel locking bias that integer argmax carries. [29] [30]

Presence probability and the score column. The found decision is a logistic model over diagnostics of the correlation surface, which is the sigmoid mapping of classifier evidence to a calibrated posterior introduced as Platt scaling, and the score column reports confidence in the decision actually made on a 0 to 1 scale so that the calibration component can grade whether it rises and falls with correctness; the evaluation of such probabilities is standard in the calibration literature. [31] [32]

Matched formation template. Blurring the reference to the search optics resolution and point sampling it onto the search grid reproduces the degradation chain of the search image, following the matched filter principle that correlation is optimal when the template matches the observed signal formation. [18] [2]

## Presence decision choices

Period aware second peak ratio. The ratio of the best match to the second best is the standard verification test for whether a match is distinctive, shown to remove about ninety percent of false matches at small cost when the runner up is drawn from known incorrect candidates. [21] On a periodic layout that premise fails by construction, because the runner up at the naive radius is a lattice replica of the chosen site whether or not the site is right; the localizer therefore excludes one full measured lattice period in each axis before taking the runner up, the same reasoning that scales a correlation filter's sidelobe exclusion window to the structure of the scene. [33]

Peak sharpness. The peak to sidelobe family of statistics is reported by its authors as uninformative when the correlation peak is broad, so the curvature of the surface at the peak accompanies the ratio, letting the fitted model separate a decisive peak from a broad one that produces the same ratio. [33]

Architecture as a covariate rather than a router. The spectral balance of the reference separates the two dimensional DRAM lattice from the one dimensional FinFET line family cheaply, but fitting separate decision models per detected class is the scheme the mixture of experts literature was created to replace: specialisation pays only when the classes demand materially different predictors, and hard routing both halves the data behind every fitted weight and concentrates router error exactly in the class overlap. [36] [37] Measured on this repository the two per architecture threshold optima sit at 0.310 and 0.390 and every hard specialisation lost held out credit, so the balance value enters the single pooled model as a continuous feature. Industrial pattern matching on periodic layouts takes the same shape, detecting periodicity and switching only a deterministic tie break rather than the fitted decision itself, [38] and golden template inspection likewise derives its template from the measured periodicity of the scene. [39]

Reject threshold placement. With a calibrated posterior, a single threshold is the optimal reject rule, so effort belongs in features and calibration rather than richer gates. [34] The F1 optimal threshold equals half the achievable F1, which places it below one half and lower on harder problems, consistent with the fitted optima this repository measures near a third; the shipped operating point is chosen on the plateau of a sweep pooled over disjoint suites. [35]

Sequential search agents, surveyed and declined. Registration by a learned sequential agent beats exhaustive search where the pose space is too large to enumerate and one evaluation is expensive, reported at two to three orders of magnitude in high dimensional spaces. [40] [41] [42] [43] At four degrees of freedom with an FFT evaluated similarity the whole disclosed range fits inside the runtime budget, the agents' robustness baselines are local optimizers rather than dense global search, and none of the surveyed agents decides that a target is absent, which carries a quarter of the Phase 2 score; the survey and the decline are recorded in experiments/20260901_rl_layered_search.

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
24. M. Born, E. Wolf, Principles of Optics: Electromagnetic Theory of Propagation, Interference and Diffraction of Light, 7th expanded edition, Cambridge University Press, 1999.
25. J. W. Goodman, Introduction to Fourier Optics, 3rd edition, Roberts and Company, 2005.
26. A. C. Diebold, editor, Handbook of Silicon Semiconductor Metrology, Marcel Dekker, New York, 2001.
27. B. Zhang, J. Zerubia, J. C. Olivo Marin, Gaussian approximations of fluorescence microscope point spread function models, Applied Optics, vol. 46, no. 10, pp. 1819 to 1829, 2007.
28. J. T. L. Thong, K. S. Sim, J. C. H. Phang, Single image signal to noise ratio estimation, Scanning, vol. 23, no. 5, pp. 328 to 336, 2001. doi 10.1002/sca.4950230506.
29. H. Foroosh, J. B. Zerubia, M. Berthod, Extension of phase correlation to subpixel registration, IEEE Transactions on Image Processing, vol. 11, no. 3, pp. 188 to 200, 2002.
30. H. Nobach, M. Honkanen, Two dimensional Gaussian regression for sub pixel displacement estimation in particle image velocimetry or particle position estimation in particle tracking velocimetry, Experiments in Fluids, vol. 38, no. 4, pp. 511 to 515, 2005. doi 10.1007/s00348-005-0942-3.
31. J. C. Platt, Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods, in Advances in Large Margin Classifiers, MIT Press, 2000, first circulated 1999.
32. A. Niculescu Mizil, R. Caruana, Predicting good probabilities with supervised learning, Proceedings of the 22nd International Conference on Machine Learning, pp. 625 to 632, 2005. doi 10.1145/1102351.1102430.
33. D. S. Bolme, J. R. Beveridge, B. A. Draper, Y. M. Lui, Visual object tracking using adaptive correlation filters, IEEE Conference on Computer Vision and Pattern Recognition, pp. 2544 to 2550, 2010. doi 10.1109/CVPR.2010.5539960.
34. C. K. Chow, On optimum recognition error and reject tradeoff, IEEE Transactions on Information Theory, vol. IT 16, no. 1, pp. 41 to 46, 1970. doi 10.1109/TIT.1970.1054406.
35. Z. C. Lipton, C. Elkan, B. Naryanaswamy, Optimal thresholding of classifiers to maximize F1 measure, Machine Learning and Knowledge Discovery in Databases, Lecture Notes in Computer Science vol. 8725, pp. 225 to 239, 2014. doi 10.1007/978-3-662-44851-9_15.
36. R. A. Jacobs, M. I. Jordan, S. J. Nowlan, G. E. Hinton, Adaptive mixtures of local experts, Neural Computation, vol. 3, no. 1, pp. 79 to 87, 1991. doi 10.1162/neco.1991.3.1.79.
37. Z. Chen, Y. Deng, Y. Wu, Q. Gu, Y. Li, Towards understanding the mixture of experts layer in deep learning, Advances in Neural Information Processing Systems 35, 2022. arXiv 2208.02813.
38. A. Sugiyama, H. Shindo, H. Komuro, T. Sutani, H. Morokuma, Pattern matching method and computer program for executing pattern matching, United States Patent 7,925,095 B2, assigned to Hitachi High Technologies Corporation, granted April 12, 2011.
39. P. Xie, S. U. Guan, A golden template self generating method for patterned wafer inspection, Machine Vision and Applications, vol. 12, no. 3, pp. 149 to 156, 2000. doi 10.1007/s001380050133.
40. R. Liao, S. Miao, P. de Tournemire, S. Grbic, A. Kamen, T. Mansi, D. Comaniciu, An artificial agent for robust image registration, Proceedings of the AAAI Conference on Artificial Intelligence, vol. 31, no. 1, 2017. doi 10.1609/aaai.v31i1.11230.
41. K. Ma, J. Wang, V. Singh, B. Tamersoy, Y. J. Chang, A. Wimmer, T. Chen, Multimodal image registration with deep context reinforcement learning, Medical Image Computing and Computer Assisted Intervention, Lecture Notes in Computer Science vol. 10433, pp. 240 to 248, 2017. doi 10.1007/978-3-319-66182-7_28.
42. J. Krebs, T. Mansi, H. Delingette, L. Zhang, F. C. Ghesu, S. Miao, A. K. Maier, N. Ayache, R. Liao, A. Kamen, Robust non rigid registration through agent based action learning, Medical Image Computing and Computer Assisted Intervention, Lecture Notes in Computer Science vol. 10433, pp. 344 to 352, 2017. doi 10.1007/978-3-319-66182-7_40.
43. F. C. Ghesu, B. Georgescu, Y. Zheng, S. Grbic, A. Maier, J. Hornegger, D. Comaniciu, Multi scale deep reinforcement learning for real time 3D landmark detection in CT scans, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 1, pp. 176 to 189, 2019. doi 10.1109/TPAMI.2017.2782687.
