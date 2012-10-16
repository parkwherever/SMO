function [alphas,b]=trainSVM(Kern,Cslack, trainLabels)
% K = kernel(trainPoints,trainPoints);
% c = margin
%TRAINSVM Summary of this function goes here
%   Detailed explanation goes here

%main function
global K E B Alphas Labels Eps C;
Labels = trainLabels;
K = Kern;
C = Cslack;

%precision point
Eps = 10^(-3);

% equivalent to size(Labels,2)
dataSetSize = length(Labels);

B = 0;
Alphas = zeros(dataSetSize,1);

% set error to be the worst, so it will be improved
%FIXME: does this work?
E = -Labels;

numChanged = 0;
examineAll = 1;


while numChanged >0 || examineAll
    numChanged = 0;
    if (examineAll)
        % loop over all training examples
        for i=1:length(Labels)
            numChanged = numChanged + examineExample(i);
        end
    else
        % loop over examples where alpha is not 0 and not C
        for i=1:length(Labels)
            if (Alphas(i)> 0 && Alphas(i) < C)
                numChanged = numChanged + examineExample(i);
            end
        end
    end
    
    if (examineAll == 1)
        examineAll = 0;
    elseif (numChanged == 0)
        examineAll = 1;
    end
end

b = B;
alphas = Alphas;

