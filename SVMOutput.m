function [ value ] = SVMOutput( i )
%SVMOUTPUT Summary of this function goes here
%   Detailed explanation goes here
    
global Labels Alphas K b

value = sum(Labels .*Alphas' .* K(i,:)) - b;

end

