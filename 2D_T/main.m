close all
clear all
clc

rng(1993)
addpath('2D_T_Data');

%% load data
load('Kdata.mat') % C5

switch 1
    case 1
        load('S3R6.mat');
    case 2
        load('S3R6.mat');
    case 3
        load('S3R6.mat');
    case 4
        load('S3R6.mat');
end
    
disp(['Net Acceleration Rate R', num2str(numel(Mask)/sum(Mask,'all'))]);

%% HICU Reconstruction
Center = 1/4;                                                                                   % [tunable] occupies Center kx by Center ky: smaller value, faster reconstruction
[Nx,Ny,Nt,Nc] = size(Kdata);                                                                    % kx ky time coil dimensions
X_keep = round(Nx*(1/2-Center/2)): round(Nx*(1/2+Center/2)-1);                                  % x coordiantes of center region
Y_keep = round(Ny*(1/2-Center/2)): round(Ny*(1/2+Center/2)-1);                                  % y coordinates of center region

Mask_c = Mask(X_keep, Y_keep,:,:);                                                              % center mask
Kdata_c = Kdata(X_keep, Y_keep,:,:);                                                            % center k-space
Kdata_ob = Kdata.*Mask;                                                                         % observed k-space with zero filling
Kdata_ob_c = Kdata_ob(X_keep, Y_keep,:,:);                                                      % center observed k-space with zero filling

Kdata_hat = Kdata_ob;                                                                           % estimaition of k-space
Kdata_hat(~Mask) = nan;                                                                         % set unobserved k-space to be nan
Kdata_hat = permute(repmat(squeeze(nanmean(Kdata_hat,3)),[1,1,1,Nt]),[1,2,4,3]);                % time average excluding nan
Kdata_hat(Mask) = Kdata_ob(Mask);                                                               % replace observed k-space
Kdata_hat(isnan(Kdata_hat)) = 0;                                                                % set remained nan value to be 0
Kdata_hat_init = Kdata_hat;
Kdata_c_hat = Kdata_hat(X_keep, Y_keep,:,:);                                                    % estimiation of center k-space

%% HICU Reconstruction
Kernel_size = [5,5,5,Nc];                                                                       % [tunable] kernel size: [5,5,5,Nc] is empirically large enough for most 2D+T imaging
Rank = 130;                                                                                     % [tunable] rank
Proj_dim = 1*Nc;                                                                                % [tunable] projected nullspace dimension: Nc~4*Nc empirically balances between SNR and speed for 2D+T
Denoiser = [];                                                                                  % [tunable] denoising subroutine (optional), no denoiser G = []
Iter_1 = 100;                                                                                   % [tunable] number of iterations: 100 works for R6 and R8
Iter_2 = 5;                                                                                     % [tunable] number of iterations for gradient descent (GD) + exact line search (ELS)
GD_option = 1;                                                                                  % [tunable] options of calculating Grammian and GD, 1: without padding -> accurate & slow, 2. with circular padding approximation applied to Grammain and GD calculation using FFT -> less accurate and fast with large kernels. To reproduce the results in Ref [2], GD_option = 1
Max_time = 1e6;                                                                                 % [tunable] maximum running time (optional) if not empty the algorithm will stop either at maximum number of iterations or maximum running time
Ref = Kdata_c;

% Warm start using center of k-space
disp('Process the center k-space......');
[Kdata_c_hat, Null_c, SNR_c, Time_c] = HICUsubroutine_2D_T(Kdata_ob_c, Mask_c, Kdata_c_hat, [], Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, Max_time, Ref);

% Form k-space estimation by replacing center region
Kdata_hat(X_keep, Y_keep,:,:) = Kdata_c_hat;

% Process on full k-space array
Iter_1 = 1;                                                                                     % [tunable] number of iterations
Iter_2 = 100;                                                                                   % [tunable] number of iterations for gradient descent + exact line search
Proj_dim = 1*Nc;                                                                                % [tunable] projected nullspace dimension: Nc~4*Nc empirically balances between SNR and speed for 2D+T
Max_time = 1e6;                                                                  % [tunable] maximum running time (optional) if not empty the algorithm will stop either at maximum number of iterations or maximum running time
Ref = Kdata;

disp('Process the full k-space......')
[Kdata_hat, Null, SNR_o, Time_o] = HICUsubroutine_2D_T(Kdata_ob, Mask, Kdata_hat, Null_c, Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, Max_time, Ref);

% Combine SNR and Time
Center_selection = false(Nx,Ny,Nt,Nc);
Center_selection(X_keep, Y_keep,:,:) = true;
HICU_SNR = cat(1,-20*log10((10.^(2*SNR_c/(-20))*norm(Kdata_c(:)).^2 + norm(Kdata_hat_init(~Center_selection)-Kdata(~Center_selection)).^2).^0.5/norm(Kdata(:))),SNR_o);
HICU_Time = cat(1, Time_c, Time_c(end)+Time_o);

disp(['HICU reconstructed k-space SNR (dB) is ', num2str(HICU_SNR(end))])
disp(['HICU reconstruction time (s) is: ' num2str(HICU_Time(end))]);

figure
plot(HICU_Time, HICU_SNR,'-.o'); grid on; drawnow


%% Function
function snr = SNR(x,ref)                                                                               % calculate the SNR
snr = -20*log10(norm(x(:)-ref(:))/norm(ref(:)));
end

% [1] Zhao, Shen, et al. "Convolutional Framework for Accelerated Magnetic Resonance Imaging." arXiv preprint arXiv:2002.03225 (2020).
% [2] Zhao, Shen, et al. "High-dimensional fast convolutional framework for calibrationless MRI." arXiv preprint arXiv:2004.08962 (2020).