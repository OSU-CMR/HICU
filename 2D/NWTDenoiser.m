function Kdata = NWTDenoiser(Kdata,Lam1,Lam2) % NWT denoiser
I = K2I(Kdata);
c = max(abs(I),[],'all'); % the max absolute value
I = I/c;                  % normalize
I_cp = CP(I);

% Soft threshold for each band
I_LL = convn(I_cp,[1 1; 1 1]/4,'valid');
I_LH = convn(I_cp,[1 1; -1 -1]/4,'valid');
I_HL = convn(I_cp,[1 -1; 1 -1]/4,'valid');
I_HH = convn(I_cp,[1 -1;-1 1]/4,'valid');

I_LL = max(abs(I_LL)-Lam1,0).*exp(1j*angle(I_LL));
I_LH = max(abs(I_LH)-Lam2,0).*exp(1j*angle(I_LH));
I_HL = max(abs(I_HL)-Lam2,0).*exp(1j*angle(I_HL));
I_HH = max(abs(I_HH)-Lam2,0).*exp(1j*angle(I_HH));

I = I_LL + I_LH + I_HL + I_HH;
I = I*c;
Kdata = I2D(I);
end

function I = K2I(Kdata)% k-space to image domain
I = fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(Kdata,1),2),[],1),[],2),1),2);
end

function D = I2D(I)    % image domain to k-sapce
D = fftshift(fftshift(fft(fft(ifftshift(ifftshift(I,2),1),[],2),[],1),2),1);
end

function I_cp = CP(I)  % circular pad
I_cp = zeros(size(I)+[1,1,0],'like', I);
for ii = 1:size(I,3)
    I_cp(:,:,ii) = padarray(I(:,:,ii),[1,1],'circular','pre');
end
end
