function Kdata = SWT_denoiser(Kdata,Step_size,Lam_1,Lam_2)
% This function provide capability of regularizing the least square with
% stationary wavelet transform (SWT), which is similar to proximal operator
% See the accompanying license.txt for additional license information.
% The software is available from https://github.com/OSU-CMR/HICU
% Author: Shen Zhao, 04/22/2020, Email: zhao.1758@osu.edu
% Input -------------------------------------------------------------------
% Kdata:     k-space data                                   (tensor: #kx x #ky x #coil)
% Step_size: The step size for the latest gradient descent  (scaler)
% Lam_1:     The lagrange multiplier for band LL            (scaler)
% Lam_2:     The lagrange multiplier for band LH HL HH      (scaler)
% Output ------------------------------------------------------------------
% Kdata:     k-space data                                   (tensor: #kx x #ky x #coil)

persistent Basis_fft2                                                                                                  % the common fft2 of the basis for efficiency

switch 1
    case 1 % direct convolution to calculate the soft-thresholding, lower flops but sequential, slower when wavelet basis is large
        I = K2I(Kdata);
        c = max(abs(Kdata),[],'all');                                                                                  % the largest absolute value in k-space, for normalizzation        
        I = I/c;                                                                                                       % normalize
        I_cp = padarray(I,[1,1,0],'circular','pre');
        
        %Generate stationary wavelet bands
        I_LL = convn(I_cp,[1  1;  1  1]/4,'valid');
        I_LH = convn(I_cp,[1  1; -1 -1]/4,'valid');
        I_HL = convn(I_cp,[1 -1;  1 -1]/4,'valid');
        I_HH = convn(I_cp,[1 -1; -1  1]/4,'valid');
        
        % Soft threshold each band
        I_LL = max(abs(I_LL)-Lam_1*abs(Step_size),0).*sign(I_LL);
        I_LH = max(abs(I_LH)-Lam_2*abs(Step_size),0).*sign(I_LH);
        I_HL = max(abs(I_HL)-Lam_2*abs(Step_size),0).*sign(I_HL);
        I_HH = max(abs(I_HH)-Lam_2*abs(Step_size),0).*sign(I_HH);
        
        I = I_LL + I_LH + I_HL + I_HH;
        I = I*c;
        
        Kdata = I2K(I);                                                                                                % back to k-space        
    case 2 % FFT based convolution to calculate the soft thersholding, highly vectorized, speed is slower than direct convolution but does not change when wavelet baisis is large                              
        if isempty(Basis_fft2)
            Basis_fft2 = fft2(cat(4,[1 1; 1  1],[1 1; -1 -1],[1 -1; 1 -1],[1 -1; -1 1])/4,size(Kdata,1),size(Kdata,2));
        end
        c = max(abs(Kdata),[],'all');                                                                                  % the largest absolute value in k-space, for normalizzation        
        I_bands = ifft2(fft2(K2I(Kdata)).* Basis_fft2);                                                                % generate stationary wavelet bands LL LH HL HH                
        I_bands = max(abs(I_bands)-reshape([Lam_1, Lam_2, Lam_2, Lam_2]*abs(Step_size)*c,1,1,1,4), 0).*sign(I_bands);  % soft threshold each band
        Kdata = I2K(sum(I_bands,4));                                                                                   % back to k-space
end
end

function I = K2I(Kdata)                                                                                                % k-space to image domain
I = sqrt(size(Kdata,1)*size(Kdata,2))*fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(Kdata,1),2),[],1),[],2),1),2);
end

function Kdata = I2K(I)                                                                                                % image domain to k-sapce
Kdata = 1/sqrt(size(I,1)*size(I,2))*fftshift(fftshift(fft(fft(ifftshift(ifftshift(I,2),1),[],2),[],1),2),1);
end
