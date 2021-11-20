%% ECE-474-Proj#1-Victor_Zhang

% The project is to implement conjugate estimators for the following scenarios:
% Binomial 
% Gaussian with known variance (i.e. estimate the mean)
% Gaussian with known mean (i.e. estimate the variance)
% 
% You should plot the mean squared error both the ML and conjugate prior estimates, all on one plot, with a legend. For each scenario, choose 2-3 different values for the hyper parameters. 
% 
% Additionally, for at least one set of hyperparameter per scenario, plot the posterior density as it changes with observations. The easy way to do this is to just plot the pdf a few times, for different #s of observations.
% 
% Stretch Goal #1: Make a movie showing the posterior converge (2 points)
% Stretch Goal #2: Implement the conjugate prior estimator for the unknown mean and variance case. Plot the pdf as it changes as in the first part. Note this will be a 3-Dim pdf, so use a heatmap or someting similar. (3 points)
%%
%Note: Load the project folder and its subfolders as working directory to
%run this script
%%
%Binomial
clc;
clear;
close all;
%setting hyper-parameters, binomial: mu(real mean) a,b(initial param)
n_iters=100;
n_obs=100;
mu = [0.2 0.5 0.7];
a = [1 2 9];
b = [2 5 10];
%generate random data to be formatted and used later
data = rand([n_iters n_obs]); % uniform random -> threshold for set probability
obs = [0:1:n_obs];
x = linspace(0,1,1000); %x-axis space for testing pdf
ev = repmat([0:1:n_obs],[n_iters 1]); %matrix of 1....n, to remove for loops
figure(1);
hdl1=axes; %allow ploting to specified figure
hold on;
Z=[];Z_T=[]; %initialize these variables outside loop for later ploting
ll=[];
for i=[1:1:length(mu)]
    ll=[ll "ML#"+i "CP#"+i]; %generating legend for graph
    Y = data<mu(i);
    %Binomail estimate mean
    MLE_A = cumsum(Y,2); %running sum
    MLE_B = ev(:,2:end); %padded with 0 in first column for initial values
    MLE = MLE_A./MLE_B;
    MLE_est = mean((MLE-mu(i)).^2,1); %ML method estimation
    plot(hdl1,obs(2:end),MLE_est);
    Z = [zeros([n_iters 1]) MLE_A];
    Z_T = ev-Z + b(i);
    Z = Z + a(i);
    est = Z./(Z+Z_T);
    est_m = mean((est-mu(i)).^2,1); %Bayes method
    figure(1);
    plot(hdl1,obs,est_m);
end
%Format plots
legend(hdl1,ll);
xlabel(hdl1,"Number of Obs");
ylabel(hdl1,"MSE");
title(hdl1,"Binomial MSE/Obs");
hold off; % redundancy
makegif(@betapdf,x,Z(1,:),Z_T(1,:),"Binomail.gif",0.1);%Call makegif to output gif at working directory
%%
%Gaussian known variance
clc;
clear;
%hyper-params
n_iters=100;
n_obs=100;
%true values
mean_true = [2 5 7];
variance = [3 7 16];
sdiv = sqrt(variance);
data = randn([[n_iters n_obs]]);
x = linspace(0,20,1000); %x-axis space for pdf
%initial guess
m0=[2 10 9];
var0=[3 5 16];
sdiv0 = sqrt(var0);
ev = repmat([1:1:n_obs], [n_iters 1]);
%initial gamma params
a = [4 8 10];
b = [2 3 5];
% allow ploting to different figures
figure(2);
hdl2=axes;
hold on;
figure(3);
hdl3=axes;
hold on;
%initialize variables for changing scope
ll1=[];
a_N=[];
b_N=[];

for i=[1:1:length(mean_true)]
    %formatting generated normal random with chosen mean and variance
    X = data .* sdiv(i) + mean_true(i);
    mu_ML = zeros(size(X));
    mu_ML(:,1) = X(:,1);
    var_ML_N = (X-mean_true(i)).^2;
    var_ML_N = cumsum(var_ML_N,2);% var_ML_N = estimated variance * N (current number of obs)
    var_ML = var_ML_N ./ ev;
    for j = [2:1:n_obs]
        mu_ML(:,j) = mu_ML(:,j-1) + (X(:,j) - mu_ML(:,j-1))/j; %ML method mean estimation
    end
    % Bayes mean estimate for known variance
    %implement equations 2.141-2.142
    mean_est = variance(i) * m0(i) ./ (ev * var0(i) + variance(i)) + (ev * var0(i) .* mu_ML ./(ev * var0(i) + variance(i)));
    var_est = 1./ (1/var0(i) + ev/variance(i));
    plot(hdl2,ev(1,:),mean((mu_ML-mean_true(i)).^2,1));
    plot(hdl2,ev(1,:),mean((mean_est-mean_true(i)).^2,1));
    % Bayes variance estimate (known mean)
    % 2.152 update equations
    a_N = a(i) + ev(1,:)/2;
    b_N = b(i) + var_ML_N/2;
    mean_est2 = a_N./b_N;%calc mean/variance for plotting
    var_est2 = mean_est2./b_N;
    plot(hdl3,ev(1,:),mean((var_ML-variance(i)).^2,1));
    plot(hdl3,ev(1,:),mean((1./mean_est2-variance(i)).^2,1));
    %Legend
    ll1=[ll1 "ML#"+i "CP#"+i];
end
%format mean estimate plot
legend(hdl2,ll1);
xlabel(hdl2,"Number of Obs");
ylabel(hdl2,"MSE");
title(hdl2,"Gaussian w/ known variance MSE/Obs");
hold off;
% format variance estimate plot
legend(hdl3,ll1);
xlabel(hdl3,"Number of Obs");
ylabel(hdl3,"MSE");
title(hdl3,"Gaussian w/ known mean MSE/Obs");
hold off;
%movie
makegif(@normpdf,x,mean_est(1,:),var_est(1,:),"Gaussian_1.gif",0.1); %gif for known variance
x = linspace(0,1,1000); %change x-axis space for variance
makegif(@gampdf,x,a_N(1,:),1./b_N(1,:),"Gaussian_2.gif",0.1); % gif for known mean
%%
clc;
clear;
% same set of hyper-params for generating random data
n_iters=1;
n_obs=100;
mean_true = 2; % only one set of hyperparameters since no MSE evaluation
variance = 5;
sdiv = sqrt(variance);
data = randn([[n_iters n_obs]]);
x = linspace(0,5,1000);
ev = repmat([1:1:n_obs], [n_iters 1]); % for loop -> matrix

% initial guess
mu0 = 2;
k0 = 0.1;
a0 = 10;
b0 = 1;

x = linspace(0,3,1000); % mean space for pdf
sig = linspace(1e-5,10,n_obs); % variance space for pdf
%set variance & mean
X = data*sdiv+mean_true;
%generate the posteriors (matrix instead of loops)
%update equations & prior & posteriors from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
X_runsum = cumsum(X,2);
X_mean = X_runsum./ev;

mu_N = (mu0*k0 + X_runsum) ./ (k0 + ev);
k_N = k0 + ev;
a_N = a0 + ev/2;
%initialize the gif process
h = figure();
axis tight manual;
%initialize memory
prob = zeros([n_obs length(x)]);
for i = [1:1:n_obs]
    b_N = b0 + (cumsum((X-X_mean(i)).^2,2) + (k0*ev)./(k0 + ev) .* (mu0 - X_mean).^2)/2;
    for j = [1:1:n_obs]
        prob(j,:) = normpdf(x,mu_N(i),1/(k_N(i)*sig(j))) * gampdf(sig(j),a_N(i),b_N(i)); %Normal-Gamma pdf
    end
    heatmap(prob,'GridVisible','off','XLabel','Mean','YLabel','Variance','Title',"Unknown Mean & Variance"); %drawing heatmap frame
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));% remove all axis values (otherwise its a black bar)
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    drawnow;
    frame = getframe(h);
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256);
    %draw
    if i == 1 
        imwrite(imind,cm,"Gaussian_3.gif",'gif', 'DelayTime', 0.1,'Loopcount',inf); 
    else 
        imwrite(imind,cm,"Gaussian_3.gif",'gif','DelayTime', 0.1, 'WriteMode','append');
    end
        clf;
end
%% 
% 

function makegif(funct,x,a,b,fname,delay)
% makegif iterate through a,b arrays and for each pair generate graph
% funct(x,a,b) and write to gif of fname with specified dalay
    h = figure();
    axis tight manual;
    for i = [1:1:length(a)]
        plot(x,funct(x,a(i),b(i)));
        xlabel("X");
        ylabel("Beta PDF");
        title("PDF GIF");
        drawnow;
        frame = getframe(h);
        im = frame2im(frame); 
        [imind,cm] = rgb2ind(im,256);
        if i == 1 
            imwrite(imind,cm,fname,'gif', 'DelayTime', delay,'Loopcount',inf); 
        else 
            imwrite(imind,cm,fname,'gif','DelayTime', delay, 'WriteMode','append');
        end
            clf;
    end
end