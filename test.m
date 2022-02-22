
%%
clc;
clear all;
close all;

inp1=imgetfile;    
a=imread(inp1);%%image read
figure('name','input','numbertitle','off');%separate figure window
imshow(a);impixelinfo;%to show an image


%%
%%%%%%%%%%%%%%%%%%%PREPROCESS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

b=rgb2gray(a);%RGB TO GRAY
figure('name','gray','numbertitle','off');
imshow(b);impixelinfo;

c = imresize(b, [500 500]);%RESIZING
figure('name','Resized Image','numbertitle','off');
imshow(c);impixelinfo;

d = imnoise(c,'salt & pepper',0.02);%NOISE ADDING 
figure('name','Noisy','numbertitle','off');
imshow(d);impixelinfo;


e = medfilt2(d);%%median filter
figure('name','Filtered Image','numbertitle','off');
imshow(e);impixelinfo;


%%
%%%%%%KMEANS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b=kmeansclustering(e)


%%
% % % % GLCM 

glcms = graycomatrix(b);

stats = graycoprops(glcms,'Contrast Correlation');

stats1 = graycoprops(glcms,'Energy Homogeneity');

conts=stats.Contrast;

corre=stats.Correlation;

en=stats1.Energy;

ho=stats1.Homogeneity;


%%
% % % fetaures 

bw1=b;
me=mean2(bw1);

st=std2(bw1);

va=var(var(double(bw1)));

sk=skewness(skewness(double(bw1)));

ku=kurtosis(kurtosis(double(bw1)));


%%
%%%%%all features

QF=[me st va sk ku conts corre en ho]

k=double(QF);

TestSet=k;

%%

%CLASSIFICATION

GroupTrain=[ones(14,1);2*ones(12,1)];%1,2,3,4 class

load traintree.mat
S=x;
Mdl = fitcknn(S,GroupTrain,'NSMethod','exhaustive','Distance','cosine');%KNN classifier
result = predict(Mdl,TestSet)

%%
%%%%%%%result

if result==1
    msgbox('cancer')
elseif result==2
    msgbox('healthy')
elseif result==3

    end

%%