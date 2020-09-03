close all
clear all
clc

rng(1993)
addpath('2D_Data')

Data_i = 1;                                                                                                        % index of dataset
switch Data_i
    case 1
        fName = 'file_brain_AXT2_200_2000019';
    case 2
        fName = 'file_brain_AXT2_200_2000021';
    case 3
        fName = 'file_brain_AXT2_200_2000022';
    case 4
        fName = 'file_brain_AXT2_200_2000024';
    case 5
        fName = 'file_brain_AXT2_200_2000031';
end

% read raw k-space data
disp(['Processing the file: ',fName]);
load([fName,'.mat'],'Kdata')
I = K2I(Kdata);

Samp_i = 4;
switch Samp_i                                                                                                   % index of sampling pattern
    case 1
        load('S1R3.mat');
        Iter_1 = 40;                                                                                            % [tunable] number of iterations could be smaller (retunted later!!!!!!!!!)
    case 2
        load('S1R5.mat');
        Iter_1 = 100;                                                                                           % [tunable] number of iterations
    case 3
        load('S2R3.mat');
        Iter_1 = 50;                                                                                            % [tunable] number of iterations
    case 4
        load('S2R5.mat');
        Iter_1 = 200;                                                                                           % [tunable] number of iterations
end

%% HICU reconstruction
Center = 1/4;                                                                                                   % [tunable] occupies Center kx by Center ky: smaller value, faster reconstruction.
[Nx,Ny,Nc] = size(Kdata);                                                                                       % kx ky coil dimensions
X_keep = round(Nx*(1/2-Center/2)): round(Nx*(1/2+Center/2)-1);                                                  % x coordiantes of center region
Y_keep = round(Ny*(1/2-Center/2)): round(Ny*(1/2+Center/2)-1);                                                  % y coordinates of center region

Mask_c = Mask(X_keep, Y_keep,:);                                                                                % center mask
Kdata_c = Kdata(X_keep, Y_keep,:);                                                                              % center k-space
Kdata_ob = Kdata.*Mask;                                                                                         % observed k-space with zero filling
Kdata_ob_c = Kdata_ob(X_keep, Y_keep,:);                                                                        % center observed k-space with zero filling

%% HICU Reconstruction
Kernel_size = [5,5,Nc];                                                                                         % [tunable] kernel size: [5,5,Nc] is empirically large enough for most 2D parallel imaging
Rank = 30;                                                                                                      % [tunable] rank
Proj_dim = 1*Nc;                                                                                                % [tunable] projected nullspace dimension: Nc~2*Nc empirically balances between SNR and speed for 2D. If Proj_dim = nullity, then no projection.
%         Denoiser = @(I,Step_size)SWT_denoiser(I,Step_size, 1.044e-9, 1.044e-7);                                         % [tunable] denoising subroutine (optional), no denoiser G = [], paired denoiser -> better SNR, lower speed
Denoiser = [];                                                                                                  % [tunable] denoising subroutine (optional), no denoiser G = [], paired denoiser -> better SNR, lower speed
Iter_2 = 5;                                                                                                     % [tunable] number of iterations for gradient descent + exact line search
GD_option = 1;                                                                                                  % [tunable] options of calculating graident, 1: without padding -> accurate & slow, 2. with circular padding approximation using FFT -> less accurate and fast with large kernels 3. with zero padding approximation -> less accurate & fast. To reproduce the results in Ref [2], GD_option = 1.
Max_time = 1e6;                                                                                                 % [tunable] maximum running time (optional) if not empty the algorithm will stop either at maximum number of iterations or maximum running time
Ref = Kdata_c;                                                                                                  % [tunable] ground truth data (optional) if not empty the algorithm will generate SNR in k-space for each iteration

% Warm start using center of k-space
disp('Process the center k-space......');
[Kdata_c_hat, Null_c, SNR_c, Time_c] = HICUsubroutine_2D(Kdata_ob_c, Mask_c, Kdata_ob_c, [], Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, Max_time, Ref);

% Form k-space estimation by replacing center region
HICU_Kdata_hat = Kdata_ob;
HICU_Kdata_hat(X_keep, Y_keep,:) = Kdata_c_hat;

% Process on full k-space array
Iter_1 = 2;                                                                                                     % [tunable] number of iterations
Iter_2 = 10;                                                                                                    % [tunable] number of iterations for gradient descent + exact line search
Proj_dim = 4*Nc;                                                                                                % [tunable] projected nullspace dimension: Nc~2*Nc empirically balances between SNR and speed for 2D. If Proj_dim = nullity, then no projection.
Max_time = 1e6;                                                                                                 % [tunable] maximum running time (optional) if not empty the algorithm will stop either at maximum number of iterations or maximum running time
Ref = Kdata;                                                                                                    % [tunable] ground truth data (optional) if not empty the algorithm will generate SNR in k-space for each iteration

disp('Process the full k-space......')
clear SWT_denoiser                                                                                              % clear persistent variable inside the denoiser function
[HICU_Kdata_hat, Null, SNR_o, Time_o] = HICUsubroutine_2D(Kdata_ob, Mask, HICU_Kdata_hat, Null_c, Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, Max_time, Ref);

% Combine SNR and Time
Center_selection = false(Nx,Ny,Nc);
Center_selection(X_keep, Y_keep,:) = true;
HICU_SNR = cat(1,-20*log10((10.^(2*SNR_c/(-20))*norm(Kdata_c(:)).^2 + norm(Kdata_ob(~Center_selection)-Kdata(~Center_selection)).^2).^0.5/norm(Kdata(:))),SNR_o);
HICU_Time = cat(1, Time_c, Time_c(end)+Time_o);

disp(['HICU reconstructed k-space SNR (dB) is ', num2str(HICU_SNR(end))])
disp(['HICU reconstruction time (s) is: ' num2str(HICU_Time(end))]);

% Plot
figure
plot(HICU_Time, HICU_SNR,'-.o'); grid on;
title('k-space SNR vs. Time')

figure
I_SSOS = SSOS(I);
I_HICU_SSOS = SSOS(K2I(HICU_Kdata_hat));
Row_1 = cat(2,I_SSOS, I_HICU_SSOS);
Row_2 = abs(Row_1 - repmat(I_SSOS,1,2));
Scaling = max(Row_1(:))/max(Row_2(:));

imagesc(cat(1,Row_1, Row_2*Scaling));
title(['Scaling ',num2str(Scaling), ', k-space SNR ', num2str(HICU_SNR(end))])
axis image off
colormap gray



%% Functions
function I = K2I(Kdata)                                                                                                 % k-space to image domain
I = sqrt(size(Kdata,1)*size(Kdata,2))*fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(Kdata,1),2),[],1),[],2),1),2);
end

function I_SSOS = SSOS(I)
I_SSOS = sum(abs(I).^2,ndims(I)).^0.5;
end