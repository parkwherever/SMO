function [ didUpdate ] = examineExample( i2 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

global Alphas E Labels Eps C; 

y2 = Labels(i2);
alph2 = Alphas(i2);
E2 = E(i2);
r2 = E2*y2;

didUpdate = 0;
if( r2 < -Eps && alph2 < C) || (r2 > Eps && alph2 > 0)
    % find non zero indexes
    nonZeroIndexes = find (Alphas ~= 0 & (Alphas < C | Alphas > -C));
    if ~isempty(nonZeroIndexes)
        % find second choice heuristic - most likely to maximize step size
        % choose max of negative error if label positive, and vice versa
        % sort in ascending order, and get indexes of sort order
        [~, indexes] = sort(E);
        if E(i2) > 0
            if indexes(1) == i2
                i1 = indexes(2);
            else
                i1 = indexes(1);
            end
        else
            if (indexes(end) == i2)
                i1 = indexes(end-1);
            else
                i1 = indexes(end);
            end
        end
        
        % if update, then successful heuristic approximation
        if (takeStep(i1,i2) == 1)
            didUpdate = 1;
            return;
        end
    end

    % loop over all possible non zero and non-c alpha, starting at random point
    randomIndexes = randperm(length(nonZeroIndexes));
    for j=1:length(nonZeroIndexes)
        if (takeStep(randomIndexes(j),i2) == 1)
            didUpdate = 1;
            return
        end
    end

    % loop over all possible i1, starting at random point
    randomIndexes = randperm(length(Alphas));
    for j=1:length(Alphas)
        if (takeStep(randomIndexes(j),i2) == 1)
            didUpdate = 1;
            return
        end
    end  
end

return

