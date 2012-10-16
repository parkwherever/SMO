function [trainPoints, trainLabels,fun] = svm_kernel(varargin)
% function [X, y,h] = svm_kernel(varargin)
%
% Inputs:
% 'X',X : defines feature vectors as columns of matrix X (dxn)
% 'Y',y : defines labels as one 1xn vector
% 'lambda',C : sets the loss/regularization trade-off (default=1)
% Visualization:
% 'vismargin',true/false : visualizes the margin of 1
% 'viscolor',true/false : generates color plots
% Kernel parameters:
% 'kernel',s : sets kernel to either 'rbf','polynomial','linear','mkl'
% 'sigma',sigma : rbf lernel width
% 'degree',d : sets degree of polynomial kernel
% 
% Outputs:
% X : feature vectors
% y : labels
% h : svm classifier function
%
% example:
% []

pars.X=[];
pars.Y=[];
pars.bias=1;
pars.C=1;
pars.sigma=0.5;
pars.kernel='rbf';
pars.degree=4;
pars.viscolor=false;
pars.vismargin=true;
pars=extractpars(varargin,pars);


figure;
% Initialize training data to empty; will get points from user
% Obtain points froom the user:
trainPoints=pars.X;
trainLabels=pars.Y;
clf;
axis([-5 5 -5 5]);
if isempty(trainPoints)
	% Define the symbols and colors we'll use in the plots later
	symbols = {'o','x'};
	classvals = [-1 1];
	trainLabels=[];
    hold on; % Allow for overwriting existing plots
    xlim([-5 5]); ylim([-5 5]);
    
    for c = 1:2
        title(sprintf('Click to create points from class %d. Press enter when finished.', c));
        [x y] = getpts;
        
        plot(x,y,symbols{c},'LineWidth', 2, 'Color', 'black');
        
        % Grow the data and label matrices
        trainPoints = vertcat(trainPoints, [x y]);
        trainLabels = vertcat(trainLabels, repmat(classvals(c), numel(x), 1));        
    end

end

switch pars.kernel
    case 'rbf'
        kernel=@(x,z) exp(-distance(x',z')./(2*pars.sigma^2));
        disp('RBF');
    case 'linear'
        kernel=@(x,z) x*z';
        disp('linear');
    case 'polynomial'
        kernel=@(x,z) (x*z'+1).^pars.degree;
        disp('polynomial');
    case 'mkl'
		resc=@(K) K./max(max(K));
        kernel=@(x,z) resc((x*z'+1).^pars.degree)+resc(exp(-distance(x',z')./(2*pars.sigma^2)));
        disp('mkl');
end;
K=kernel(trainPoints,trainPoints);

% This is where your work begins
[alphas,b]=trainSVM(K,pars.C,trainLabels); % you have to implement this function
%find distance to the plane
fun=@(Xt) SVMOutput(Xt);
% This is where your work ends

% classification function
% visualization
title('Decision Boundary');
visdecision(trainPoints,trainLabels,fun,'viscolor',pars.viscolor,'vismargin',pars.vismargin);


