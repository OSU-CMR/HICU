# HICU (HIgh-dimensional Fast ConvolUtional Framework for Calibraless MRI)
HICU is a computational procedure for accelerated, calibrationless magnetic resonance image reconstruction that is fast, memory efficient, and ready to scale to highdimensional imaging. For demonstration, HICU is applied to multi-coil 2D static imaging and multi-coil 2D dyanmic (2D+T) imaging. 

* Shen Zhao, The Ohio State University (zhao.1758@osu.edu)
* Lee C. Potter, The Ohio State University (potter.36@osu.edu)
* Rizwan Ahmad, The Ohio State University (ahmad.46@osu.edu)

## 2D
1. One 2D T2-weighted parallel MRI k-space dataset (https://fastmri.med.nyu.edu/) and four sampling masks are in the folder 2D/2D_data. 
2. The HICU reconstruction subroutine for 2D is in **2D/HICUsubroutine_2D.m**. 
3. The optional nondecimated wavelet denoiser is in the file **2D/NWTDenoiser.m**.
4. To implement the reconstruction for 2D, run file **2D/main.m**.

## 2D+t
1. One 2D+T cardiac cine parallel MRI k-space and two sampling masks are in **2D_T/2D+T_Data**.
2. The HICU reconstruction subroutine for 2D+T is in **2D+T/HICUsubroutine_2D_T.m**.
3. To implement the reconstruction for 2D+T, run file **2D+T/main.m**


## References
1. https://arxiv.org/abs/2002.03225
2. https://arxiv.org/abs/2004.08962
