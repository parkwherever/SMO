function [ didUpdate ] = takeStep( i1, i2 )
%TAKESTEP Summary of this function goes here
%   Detailed explanation goes here
%i1 and i2 are the indeces of the points we are processing
%Y is a vector of the correct labels for our training set
%C is the margin constant
%K is an N*N matrix of our kernel function values k(xi,xj) i from 1 to N
%alphas is a vector of our current alphas that respresent weights for each
%   training point
%E is our vector of cached error values that we update iteration to
%   iteration
% b is offset value for current iteration
% w is normal vector for current iteration
global K E Alphas Labels b Eps C;

%initialize return params
didUpdate = 0;

if (i1 == i2)
    return
end

alph1 = Alphas(i1);
alph2 = Alphas(i2);
Y1 = Labels(i1);
Y2 = Labels(i2);
s = Y1*Y2;

%will need to calculate if not present in cache
%FIXME is this initilization in trainSVM right?
%What is the best way initialize the cache/check if a value exists?
E1 = E(i1);
E2 = E(i2);

if(Y1 ~= Y2)
   L = max(0,alph2-alph1);
   H = min(C,C+(alph2-alph1));
else
   L = max(0,alph2+alph1-C);
   H = min(C,alph2+alph1);
end

if (L == H)
    return
end

eta = K(i1,i1) + K(i2,i2) - 2*K(i1,i2);

if(eta > 0 )
   a2 = alph2 + Y2*(E1-E2)/eta;
   if (a2 < L)
       a2 = L;
   elseif (a2 > H)
       a2 = H;
   end
else
   % calculate obj functions since objective function is negative
   s = Y1* Y2;
   f1 = Y1 * (E1 +b) - alph1 * K(i1,i1) - s* alph2 * K(i1,i2);
   f2 = Y2 * (E2 +b) - s *alph1 * K(i1,i2) - alph2 * K(i2,i2);
   L1 = alph1 + s * (alph2 - L);
   H1 = alph1 + s * (alph2 - H);
   objL = L1*f1 + L*f2 + 0.5 * L1^2 * K(i1,i1) + 0.5 * L^2 * K(i2,i2) + s * L * L1 * K(i1,i2);
   objH = H1*f1 + H*f2 + 0.5 * H1^2 * K(i1,i1) + 0.5 * H^2 * K(i2,i2) + s * H * H1 * K(i1,i2);
   
   %set a2
   if (objL < objH - Eps)
       a2 = L;
   elseif(objL > objH +Eps)
       a2 = H;
   else
       a2 = alph2;
   end
end

 %check to see if update was made between a2 and alph2, if not, return
 if (abs(a2 -alph2) < Eps * (a2 + alph2 + Eps))
     return;
 end
 
a1 = alph1+ s * (alph2-a2);

%update threshold to reflect change in lagrange
b1 = E(i1) + Y1 * (a1 - alph1) * K(i1,i1) + Y2 * (a2 - alph2) * K(i1,i2) + b;
b2 = E(i2) + Y1 * (a1 - alph1) * K(i1,i2) + Y2 * (a2 - alph2) * K(i2,i2) + b;

if (b1 == b2)
    b = b1;
else
    b = (b1+b2)/2;
end

%update made, update alphas and error cache
Alphas(i1) = a1;
Alphas(i2) = a2;
E(i2) = SVMOutput(i2) - Y2;

didUpdate = 1;
return;

