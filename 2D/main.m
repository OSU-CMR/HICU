clear all
close all
clc

%% Load Data
addpath('2D_Data');
load('file_brain_AXT2_200_2000019.mat')                                % k-space the file is downloaded from NYU fastMRI, Ref:https://arxiv.org/abs/1811.08839.
load('S1R3.mat');                                                      % sampling pattern: can also load S1R5, S2R3, S2R5

Center = 1/6;                                                          % [tunable] occupies Center kx by Center ky: smaller value, faster reconstruction
[Nx,Ny,Nc] = size(Kdata);                                              % kx ky coil dimensions
X_keep = round(Nx*(1/2-Center/2)): round(Nx*(1/2+Center/2)-1);         % x coordiantes of center region
Y_keep = round(Ny*(1/2-Center/2)): round(Ny*(1/2+Center/2)-1);         % y coordinates of center region

Mask_c = Mask(X_keep, Y_keep,:);                                       % center mask
Kdata_c = Kdata(X_keep, Y_keep,:);                                     % center k-space
Kdata_ob = Kdata.*Mask;                                                % observed k-space with zero filling
Kdata_ob_c = Kdata_ob(X_keep, Y_keep,:);                               % center observed k-space with zero filling

%% HICU Reconstruction
Kernel_size = [5,5,Nc];                                                % [tunable] kernel size: [5,5,Nc] is empirically large enough for most 2D parallel imaging
Rank = 40;                                                             % [tunable] rank
Proj_dim = 2*Nc;                                                       % [tunable] projected nullspace dimension: Nc~2*Nc empirically balances between SNR and speed for 2D. If Proj_dim = nullity, then no projection.
Denoiser = @(I,Step_size)SWT_denoiser(I,Step_size, 1.044e-9, 1.044e-7);% [tunable] denoising subroutine (optional), no denoiser G = [], paired denoiser -> better SNR, lower speed
Iter_1 = 100;                                                          % [tunable] number of iterations: S1R3:100, S1R5:400, S2R3:160, S2R5:1200
Iter_2 = 3;                                                            % [tunable] number of iterations for gradient descent + exact line search
GD_option = 3;                                                         % [tunable] options of calculating graident, 1: without padding -> accurate & slow, 2. with zero padding approximation -> less accurate & fast 3. with circular padding approximation using FFT -> less accurate and fast with large kernels
ELS_frequency = 6;                                                     % [tunable] Every ELS_Update_Frequency steps of gradient descent, the step size is updated via ELS. Higher frequency -> more computation & less accurate step size, too large -> diverge

% Warm start using center of k-space
disp('Process the center k-space......');tic
[Kdata_c_hat, Null_c] = HICUsubroutine_2D(Kdata_ob_c, Mask_c, Kdata_ob_c, [], Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, ELS_frequency);
disp(['HICU reconstructed center k-space SNR (dB) is ', num2str(SNR(Kdata_c_hat,Kdata_c))])

% Form k-space estimation by replacing center region
Kdata_hat = Kdata_ob;
Kdata_hat(X_keep, Y_keep,:) = Kdata_c_hat;

% Process on full k-space array
Iter_1 = 50;                                                           % [tunable] number of iterations
Iter_2 = 1;                                                            % [tunable] number of iterations for gradient descent + exact line search

disp('Process the full k-space......')
[Kdata_hat, Null] = HICUsubroutine_2D(Kdata_ob, Mask, Kdata_hat, Null_c, Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, ELS_frequency);
disp(['HICU reconstructed k-space SNR (dB) is ', num2str(SNR(Kdata_hat,Kdata))])
disp(['HICU reconstruction time (s) is: ' num2str(toc)]);

%% Plot
FS = SSoS(K2I(Kdata));                                                 % coil-comibined fully sampled image
HICU = SSoS(K2I(Kdata_hat));                                           % coil-combined HICU reconstructed image

figure;
Row1 = cat(2,FS,HICU);
Row2 = 10*abs(Row1 - cat(2,FS,FS));
imagesc(cat(1,Row1,Row2));
colormap gray
axis ('image', 'off')
title('Reference (left), HICU (right), and $10~\times$ Error (second row)','Interpreter','latex')

%% Function
function I = K2I(Kdata)                                                % k-space to image domain
I = sqrt(size(Kdata,1)*size(Kdata,2))*fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(Kdata,1),2),[],1),[],2),1),2);
end

function I_SSoS = SSoS(I)                                              % SSoS comibining
I_SSoS = sum(abs(I).^2,ndims(I)).^0.5;
end

function snr = SNR(X,Ref)                                              % calculate the SNR
snr = -20*log10(norm(X(:)-Ref(:))/norm(Ref(:)));
end
