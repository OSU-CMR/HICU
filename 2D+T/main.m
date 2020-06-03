clear all
close all
clc

%% Load Data
addpath('2D+T_Data');
load('Kdata.mat');                                                              % k-space
load('R6.mat');                                                                 % sampling pattern: can also load R8

Center = 1/4;                                                                   % [tunable] occupies Center kx by Center ky: smaller value, faster reconstruction
[Nx,Ny,Nt,Nc] = size(Kdata);                                                    % kx ky time coil dimensions
X_keep = round(Nx*(1/2-Center/2)): round(Nx*(1/2+Center/2)-1);                  % x coordiantes of center region
Y_keep = round(Ny*(1/2-Center/2)): round(Ny*(1/2+Center/2)-1);                  % y coordinates of center region

Mask_c = Mask(X_keep, Y_keep,:,:);                                              % center mask
Kdata_c = Kdata(X_keep, Y_keep,:,:);                                            % center k-space
Kdata_ob = Kdata.*Mask;                                                         % observed k-space with zero filling
Kdata_ob_c = Kdata_ob(X_keep, Y_keep,:,:);                                      % center observed k-space with zero filling

Kdata_hat = Kdata_ob;                                                           % estimaition of k-space
Kdata_hat(~Mask) = nan;                                                         % set unobserved k-space to be nan
Kdata_hat = permute(repmat(squeeze(nanmean(Kdata_hat,3)),[1,1,1,Nt]),[1,2,4,3]);% time average excluding nan
Kdata_hat(Mask) = Kdata_ob(Mask);                                               % replace observed k-space
Kdata_hat(isnan(Kdata_hat)) = 0;                                                % set remained nan value to be 0
Kdata_c_hat = Kdata_hat(X_keep, Y_keep,:,:);                                    % estimiation of center k-space

%% HICU Reconstruction
Kernel_size = [5,5,5,Nc];                                                       % [tunable] kernel size: [5,5,5,Nc] is empirically large enough for most 2D+T imaging
Rank = 130;                                                                     % [tunable] rank
Proj_dim = 4*Nc;                                                                % [tunable] projected nullspace dimension: Nc~4*Nc empirically balances between SNR and speed for 2D+T
Denoiser = [];                                                                  % [tunable] denoising subroutine (optional), no denoiser G = []
Iter_1 = 100;                                                                   % [tunable] number of iterations: 100 works for R6 and R8
Iter_2 = 3;                                                                     % [tunable] number of iterations for gradient descent (GD) + exact line search (ELS)
GD_option = 2;                                                                  % [tunable] options of calculating Grammian and GD, 1: without padding -> accurate & slow, 2. with circular padding approximation applied to Grammain and GD calculation using FFT -> less accurate and fast with large kernels. To reproduce the results in Ref [2], GD_option = 1
ELS_frequency = 6;                                                              % [tunable] Every ELS_Update_frequency steps of gradient descent, the step size is updated via ELS. Higher frequency -> more computation & less accurate step size, too large -> diverge. To reproduce the results in Ref [2], ELS_frequency = 1.

% Warm start using center of k-space
disp('Process the center k-space......');tic
[Kdata_c_hat, Null_c] = Public_HICUsubroutine_2D_T(Kdata_ob_c, Mask_c, Kdata_c_hat, [], Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, ELS_frequency);
disp(['HICU reconstructed center k-space SNR (dB) is ', num2str(SNR(Kdata_c_hat,Kdata_c))])

% Form k-space estimation by replacing center region
Kdata_hat = Kdata_ob;
Kdata_hat(X_keep, Y_keep,:,:) = Kdata_c_hat;

% Process on full k-space array
Iter_1 = 64;                                                                    % [tunable] number of iterations
Iter_2 = 1;                                                                     % [tunable] number of iterations for gradient descent + exact line search

disp('Process the full k-space......')
[Kdata_hat, Null] = HICUsubroutine_2D_T(Kdata_ob, Mask, Kdata_hat, Null_c, Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, ELS_frequency);
disp(['HICU reconstructed k-space SNR (dB) is ', num2str(SNR(Kdata_hat,Kdata))])
disp(['HICU reconstruction time (s) is: ' num2str(toc)]);

%% Plot
FS = SSoS(K2I(Kdata));                                                          % coil-comibined fully sampled image
HICU = SSoS(K2I(Kdata_hat));                                                    % coil-combined HICU reconstructed image

figure;
Row1 = cat(2,FS(:,:,9),HICU(:,:,9));
Row2 = 10*abs(Row1 - cat(2,FS(:,:,9),FS(:,:,9)));
imagesc(cat(1,Row1,Row2).^0.618, [0,0.6*max(Row1(:)).^0.618]);
colormap gray
axis ('image', 'off')
title('Frame 9, Reference (left), HICU (right), and $10~\times$ Error (second row)','Interpreter','latex')

%% Function
function I = K2I(Kdata)                                                         % k-space to image domain
I = sqrt(size(Kdata,1)*size(Kdata,2))*fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(Kdata,1),2),[],1),[],2),1),2);
end

function I_SSoS = SSoS(I)                                                       % SSoS comibining
I_SSoS = sum(abs(I).^2,ndims(I)).^0.5;
end

function snr = SNR(x,ref)                                                       % calculate the SNR
snr = -20*log10(norm(x(:)-ref(:))/norm(ref(:)));
end

% [1] Zhao, Shen, et al. "Convolutional Framework for Accelerated Magnetic Resonance Imaging." arXiv preprint arXiv:2002.03225 (2020).
% [2] Zhao, Shen, et al. "High-dimensional fast convolutional framework for calibrationless MRI." arXiv preprint arXiv:2004.08962 (2020).
