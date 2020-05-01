function [Kdata,Null] = HICUsubroutine_2D_T(Kdata_ob, Mask, Kdata, Null_learned, Kernel_size, Rank, Proj_dim, Denoiser, Iter_1, Iter_2, ELS_Frequency)
% This function provides capabilities to reconstruct undersampled multi-channel 2D+T k-space using HICU
% The problem formulations implemented by this softwere originally reported in:
% [1] Zhao, Shen, et al. "Convolutional Framework for Accelerated Magnetic Resonance Imaging." arXiv preprint arXiv:2002.03225 (2020).
% [2] Zhao, Shen, et al. "High-dimensional fast convolutional framework for calibrationless MRI." arXiv preprint arXiv:2004.08962 (2020).
% Use of this software (or its derivatives) in your own work requires that you at least cite [1] or [2]
% See the accompanying license.txt for additional license information.
% The software is available from https://github.com/OSU-CMR/HICU
% Author: Shen Zhao, 04/22/2020, Email: zhao.1758@osu.edu
%
% Input -------------------------------------------------------------------
% Kdata_ob:      observed k-space data with zero filling                         (tensor: #kx x #ky x #frame x #coil)
% Mask:          sampling mask for k-space data, True: sampled, False: unsampled (tensor: #kx x #ky x #frame x #coil)
% Kdata:         initial estimiation of k-space data                             (tensor: #kx x #ky x #frame x #coil)
% Null_learned:  learned/ extracted null space                                   (matrix: prod(Kernek_size) x (prod(Kernek_size) -r))
% Kernel_size:   kernel size                                                     (vector: 1 x 4)
% Rank:          rank                                                            (scaler)
% Proj_dim:      projected nullspace dimension                                   (scaler)
% Denoiser:      denoising subroutine                                            (function handle)
% Iter_1:        number of iterations                                            (scaler)
% Iter_2:        number of iterations for gradient descent + exact line search   (scaler)
% ELS_Frequency: frequency of updating step size using exact line search         (scaler)
% Output ------------------------------------------------------------------
% Kdata:         estimation of k-space data                                      (tensor: #kx x #ky x #frame x #coil)
% Null:          output null space                                               (tensor: prod(Kernek_size) x (prod(Kernek_size) -r))

Kdata_cp = CP(Kdata,Kernel_size);
Mask_cp = CP(Mask,Kernel_size);
Data_cp_size = size(Kdata_cp);         % kx ky time coil dimensions of circular padded k-space
Diff_size = Data_cp_size - Kernel_size;% difference between kx ky coil dimensions of circular padded k-space and kernel

for i = 1:Iter_1
    disp(['Iteration ',num2str(i,'%04d')])
    %% Nullspace Extraction
    if isempty(Null_learned)
        % Build Grammian
        switch 2
            case 1 % build Grammian from convolution operator: memory efficient but relative slow in Matalb
                Gram = zeros(prod(Kernel_size), 'like', Kdata_ob);
                for l = 1:prod(Kernel_size)
                    [coord_1,coord_2,coord_3,coord_4] = ind2sub(Kernel_size,l);                    % coordinate inside the kernel
                    Kdata_cp_part = Kdata_cp(coord_1+(0:Diff_size(1)), coord_2+(0:Diff_size(2)),...
                        coord_3+(0:Diff_size(3)), coord_4+(0:Diff_size(4)));                       % part of the k-space
                    Kdata_cp_part(end:-1:1) = Kdata_cp_part;                                       % flip in all dimension
                    Gram(:,l) = reshape(convn(conj(Kdata_cp), Kdata_cp_part,'valid'), [],1);
                end
            case 2 % build Grammian from explicit matrix: memory inefficient but relative fast in Matalb
                A = zeros(prod(Data_cp_size-Kernel_size+1 ), prod(Kernel_size), 'like', Kdata_ob );
                for l = 1:prod(Kernel_size)
                    [coord_1,coord_2,coord_3,coord_4] = ind2sub(Kernel_size,l);                    % coordinate inside the kernel
                    A(:,l) = reshape(Kdata_cp(coord_1+(0:Diff_size(1)), coord_2+(0:Diff_size(2)),...
                        coord_3+(0:Diff_size(3)), coord_4+(0:Diff_size(4))), [],1);
                end
                Gram = A'*A;
        end
        % Eigendecomposition
        [V,Lam] = eig(Gram);
        [~,ind] = sort(real(diag(Lam)),'ascend');                                                  % enforce real due to possible unexpected round-off error for case 1 above
        V = V(:,ind);
        Null = V(:,1:prod(Kernel_size)-Rank);
    else
        Null = Null_learned;
    end
    
    %% Nullspace Dimensionality Reduction
    if Proj_dim == prod(Kernel_size)-Rank                                                          % Proj_dim = nullity then no random projection
        Null_tilde = Null;
    else
        Null_tilde = Null*randn(size(Null,2),Proj_dim)/sqrt(size(Null,2));                         % project to Proj_dim dimension
    end
    F = reshape(flip(Null_tilde,1),[Kernel_size,Proj_dim]);                                        % flip and reshape to filters
    F_Hermitian = reshape(conj(Null_tilde),[Kernel_size,Proj_dim]);                                % Hermitian of filters
    
    %% Solving Least-Squares Subproblem
    for j = 1:Iter_2
        % Calculate gradient
        GD_cp = zeros(Data_cp_size,'like',Kdata_cp); % gradient for the circular padded k-space        
        for k = 1:Proj_dim            
            GD_cp = GD_cp + 2*convn(convn(Kdata_cp,F(:,:,:,:,k),'valid'),F_Hermitian(:,:,:,:,k)).*(~Mask_cp);            
        end
        
        GD = GD_cp(:,:,Kernel_size(3):end,:);                                                     % gradient for the k-space
        GD(:,:,end-Kernel_size(3)+2:end,:) = GD_cp(:,:,end-Kernel_size(3)+2:end,:) + GD_cp(:,:,1:Kernel_size(3)-1,:);
        GD_cp = CP(GD,Kernel_size);                                                               % circular pad the gradient since padded k-space share the same gradient
        
        % ELS: Exact Line Search
        if mod((i-1)*Iter_2+j-1,ELS_Frequency) == 0                                               % whether update step size via ELS
            Denominator = 0;                                                                      % For ||Ax-b||^2, numeraotr should be \nabla f(x)^H \nabla f(x)
            for k = 1:Proj_dim
                Denominator = Denominator+ 2*sum(abs(convn(GD_cp,F(:,:,:,:,k),'valid')).^2,'all');% For ||Ax-b||^2, denominator should be 2\nabla f(x)^H A^H A \nabla f(x)
            end
            Numerator = sum(abs(GD).^2,'all');
            Step_ELS = -Numerator/Denominator;                                                    % optimal step size
        end
        Kdata = Kdata + Step_ELS*GD;
        Kdata_cp = CP(Kdata,Kernel_size);
        
        % Denoising (Denoiseing with GD+ELS is similar to proximal gradient descent)
        if ~isempty(Denoiser)
            Kdata = Denoiser(Kdata);     % denoise
            Kdata(Mask) = Kdata_ob(Mask);% enforce data consistency
        end
    end
end
end

%% Function
function Y_cp = CP(Y,m)                                                                          % circular padding along time dimension (for circular convolution along time dimension)
Y_cp = padarray(Y,[0 0 m(3)-1 0],'circular','pre');
end
