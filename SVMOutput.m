function [ value ] = SVMOutput( i )
%SVMOUTPUT Summary of this function goes here
%   Detailed explanation goes here
    
global Labels Alphas K B

value = sum(Labels .* Alphas .* K(i,:)') - B;

end

