close all
clear all
clc

addpath('3D_Data')

Kdata = single(readReconData('kspace'));                                                        % The data is downloaded from mridata.org
Option = 2;
switch Option
    case 1
        load('R3.mat')
    case 2
        load('R3_ACS.mat')
end

disp(['Process Dataset, net Acceleration Rate R', num2str(numel(Mask)/sum(Mask,'all'))]);

%% HICU Reconstruction
Center = 1/4;                                                                                   % [tunable] occupies Center kx by Center ky: smaller value, faster reconstruction
[Nx,Ny,Nz,Nc] = size(Kdata);                                                                    % kx ky time coil dimensions
X_keep = round(Nx*(1/2-Center/2)): round(Nx*(1/2+Center/2)-1);                                  % x coordiantes of center region
Y_keep = round(Ny*(1/2-Center/2)): round(Ny*(1/2+Center/2)-1);                                  % y coordinates of center region
Z_keep = round(Nz*(1/2-Center/2)): round(Nz*(1/2+Center/2)-1);                                  % y coordinates of center region

Mask_c = Mask(X_keep, Y_keep, Z_keep,:);                                                        % center mask
Kdata_c = Kdata(X_keep, Y_keep, Z_keep,:);                                                      % center k-space
Kdata_ob = Kdata.*Mask;                                                                         % observed k-space with zero filling
Kdata_ob_c = Kdata_ob(X_keep, Y_keep, Z_keep,:);                                                % center observed k-space with zero filling

Kdata_hat = Kdata_ob;                                                                           % estimaition of k-space
Kdata_c_hat = Kdata_hat(X_keep, Y_keep, Z_keep,:);                                              % estimiation of center k-space

%% HICU Reconstruction
Kernel_size = [5,5,5,Nc];                                                                       % [tunable] kernel size: [5,5,5,Nc] is empirically large enough for most 2D+T imaging
Rank = 150;                                                                                     % [tunable] rank
Proj_dim = 1*Nc;                                                                                % [tunable] projected nullspace dimension: Nc~4*Nc empirically balances between SNR and speed for 2D+T
Denoiser = [];                                                                                  % [tunable] denoising subroutine (optional), no denoiser G = []

switch Option
    case 1
        Iter_1 = 50;                                                                            % [tunable]
    case 2
        Iter_1 = 50;                                                                            % [tunable]
end
Iter_2 = 5;                                                                                     % [tunable] number of iterations for gradient descent (GD) + exact line search (ELS)
GD_option = 1;                                                                                  % [tunable] options of calculating Grammian and GD, 1: without padding -> accurate & slow, 2. with circular padding approximation applied to Grammain and GD calculation using FFT -> less accurate and fast with large kernels. To reproduce the results in Ref [2], GD_option = 1
Max_time = 1e6;                                                                                 % [tunable] maximum running time (optional) if not empty the algorithm will stop either at maximum number of iterations or maximum running time
Ref = Kdata_c;

% Warm start using center of k-space
disp('Process the center k-space......');
[Kdata_c_hat, Null_c, SNR_c, Time_c] = HICUsubroutine_3D(Kdata_ob_c, Mask_c, Kdata_c_hat, [], Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, Max_time, Ref);

% Form k-space estimation by replacing center region
Kdata_hat(X_keep, Y_keep, Z_keep,:) = Kdata_c_hat;

% Process on full k-space array
Iter_1 = 1;                                                                                     % [tunable] number of iterations
Iter_2 = 20;                                                                                    % [tunable] number of iterations for gradient descent + exact line search
Proj_dim = 1*Nc;                                                                                % [tunable] projected nullspace dimension: Nc~4*Nc empirically balances between SNR and speed for 2D+T
Max_time = 3.6e3- Time_c(end);                                                                  % [tunable] maximum running time (optional) if not empty the algorithm will stop either at maximum number of iterations or maximum running time
Ref = Kdata;

disp('Process the full k-space......')
clear SWT_denoiser
[Kdata_hat, Null, SNR_o, Time_o] = HICUsubroutine_3D(Kdata_ob, Mask, Kdata_hat, Null_c, Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, GD_option, Max_time, Ref);

% Combine SNR and Time
Center_selection = false(Nx,Ny,Nz,Nc);
Center_selection(X_keep, Y_keep,Z_keep,:) = true;
HICU_SNR = cat(1,-20*log10((10.^(2*SNR_c/(-20))*norm(Kdata_c(:)).^2 + norm(Kdata_ob(~Center_selection)-Kdata(~Center_selection)).^2).^0.5/norm(Kdata(:))),SNR_o);
HICU_Time = cat(1, Time_c, Time_c(end)+Time_o);

disp(['HICU reconstructed k-space SNR (dB) is ', num2str(HICU_SNR(end))])
disp(['HICU reconstruction time (s) is: ' num2str(HICU_Time(end))]);

figure
plot(HICU_Time, HICU_SNR,'-.o'); grid on; drawnow

I_hat_ssos = ssos(K2I(Kdata_hat));
I_ssos = ssos(K2I(Kdata));
comp = cat(1,I_ssos, I_hat_ssos);

%% Save Variable
switch Option
    case 1
        File_Title = 'R3_HICU_Reconstruction.mat';
    case 2
        File_Title = 'R3_ACS_HICU_Reconstruction.mat';
        
end
save(File_Title, 'Kdata_hat', 'HICU_Time','HICU_SNR','comp');

%% Functions
function I = K2I(Kdata)                                                                                                                                 % k-space to image domain
I = prod(size(Kdata,1,2,3))^0.5*fftshift3(ifft3(ifftshift3(Kdata)));
end

function Kdata = I2K(I)                                                                                                                                     % image domain to k-sapce
Kdata = 1/prod(size(I,1,2,3))^0.5*fftshift3(fft3(ifftshift3(I)));
end

function K = fft3(I,l1,l2,l3)                                                                                                                           % calculate fft along the first three dimension
switch nargin
    case 1
        K = fft(fft(fft(I,[],1),[],2),[],3);
    case 4
        K = fft(fft(fft(I,l1,1),l2,2),l3,3);
end
end

function I = ifft3(K,l1,l2,l3)                                                                                                                          % calculate ifft along the first three dimension
switch nargin
    case 1
        I = ifft(ifft(ifft(K,[],1),[],2),[],3);
    case 4
        I = ifft(ifft(ifft(K,l1,1),l2,2),l3,3);
end
end

function Kdata = fftshift3(Kdata)
Kdata = fftshift(fftshift(fftshift(Kdata,1),2),3);
end

function I = ifftshift3(I)
I = ifftshift(ifftshift(ifftshift(I,1),2),3);
end

function I_ssos = ssos(I)
I_ssos = sum(abs(I).^2,ndims(I)).^0.5;
end