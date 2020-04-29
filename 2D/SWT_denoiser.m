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


I = K2I(Kdata);
c = max(abs(I),[],'all');                   % the max absolute value
I = I/c;                                    % normalize
I_cp = padarray(I,[1,1,0],'circular','pre');

% Generate station wavelet bands
I_LL = convn(I_cp,[1 1; 1 1]/4,'valid');
I_LH = convn(I_cp,[1 1; -1 -1]/4,'valid');
I_HL = convn(I_cp,[1 -1; 1 -1]/4,'valid');
I_HH = convn(I_cp,[1 -1;-1 1]/4,'valid');

% Soft threshold each band
I_LL = max(abs(I_LL)-Lam_1*abs(Step_size),0).*exp(1j*angle(I_LL));
I_LH = max(abs(I_LH)-Lam_2*abs(Step_size),0).*exp(1j*angle(I_LH));
I_HL = max(abs(I_HL)-Lam_2*abs(Step_size),0).*exp(1j*angle(I_HL));
I_HH = max(abs(I_HH)-Lam_2*abs(Step_size),0).*exp(1j*angle(I_HH));

I = I_LL + I_LH + I_HL + I_HH;
I = I*c;
Kdata = I2D(I);
end

function I = K2I(Kdata)% k-space to image domain
I = sqrt(prod(size(Kdata,[1,2])))*fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(Kdata,1),2),[],1),[],2),1),2);
end

function D = I2D(I)    % image domain to k-sapce
D = 1/sqrt(prod(size(I,[1,2])))*fftshift(fftshift(fft(fft(ifftshift(ifftshift(I,2),1),[],2),[],1),2),1);
end
