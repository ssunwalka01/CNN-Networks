function lgraph = fist_network()
Layers =[...
    imageInputLayer([48 48 1],'Name','input')];  %inputs image to a network and apply normalization%

lgraph=layerGraph(Layers);  %object layer
%   A layer graph describes the architecture of a directed acyclic graph (DAG)
%   network for deep learning. After you create a LayerGraph object, you can use
%   object functions to add layers to a graph, connect and disconnect layers in a graph,
%   remove layers from a graph, and plot the graph. To train the network, use the layer 
%   graph as the layers input argument to trainNetwork.%


%--------------------------------------------LAYER1-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------%
% SYNTAX:- layer = convolution2dLayer(filterSize,numFilters,Name,Value)
conv1 = convolution2dLayer(7,1,'Padding',[3 0 3 0],'Stride',2,'Name','conv1');
%   convolution2dLayer(11,96,'Stride',4,'Padding',1) creates a 2-D convolutional layer with 96 filters 
%   of size [11 11], a stride of [4 4], and zero padding of size 1 along all edges of the layer input.
%   You can specify multiple name-value pairs

lgraph=addLayers(lgraph,conv1); % connects layers in a graph sequentially
lgraph=connectLayers(lgraph,'input','conv1');

%pool_1_1 = maxPooling2dLayer(3,'padding','same','Stride',2,'Name','pool_1_1');

pool1=maxPooling2dLayer(3,'padding',[1 0 1 0],'Stride',2,'Name','pool1');
lgraph = addLayers(lgraph,pool1);
lgraph = connectLayers(lgraph,'conv1','pool1');
%--------------------------------------------LAYER-2----------------------------------------------------%
conv_2 = convolution2dLayer(3,1,'Padding',1,'Stride',1,'Name','conv_2');
lgraph = addLayers(lgraph,conv_2);
lgraph = connectLayers(lgraph,'pool1','conv_2');
% pool2
pool2=maxPooling2dLayer(3,'padding',[1 0 1 0],'Stride',2,'Name','pool2');
lgraph = addLayers(lgraph,pool2);
lgraph = connectLayers(lgraph,'conv_2','pool2');

%----------------------------------------LAYER-3(INCEPTION LAYER)---------------------------------------------------------%

% connecting output at  pool2 with conv1x1 %
conv1x1 = convolution2dLayer(1,64,'Padding',0,'Stride',1,'Name','conv1x1');
lgraph=addLayers(lgraph,conv1x1);
lgraph=connectLayers(lgraph,'pool2','conv1x1');

% connecting output at pool2 with 3x3reduce %
% TAKING 3X3REDUCE AS conv3x3z%
conv3x3z = convolution2dLayer(3,96,'Padding',1,'Stride',1,'Name','conv3x3z');
lgraph = addLayers(lgraph,conv3x3z);
lgraph = connectLayers(lgraph,'pool2','conv3x3z');
%TAKING 5X5REDUCE AS RELu%
conv5x5z = convolution2dLayer(5,16,'Padding',2,'Stride',1,'Name','conv5x5z');
lgraph = addLayers(lgraph,conv5x5z);
lgraph = connectLayers(lgraph,'pool2','conv5x5z');
% Pool proj code%
relu3x3a = reluLayer('Name','relu3x3a');
lgraph = addLayers(lgraph,relu3x3a);
lgraph = connectLayers(lgraph,'pool2','relu3x3a');
%----------------------------------------- height6 finish-------------------------------------------------%



% connecting 3x3 reduce to 3x3conv %
conv3x3a = convolution2dLayer(3,128,'Padding',1,'Stride',1,'Name','conv3x3a');
lgraph=addLayers(lgraph,conv3x3a);
lgraph=connectLayers(lgraph,'conv3x3z','conv3x3a');


% connecting 5x5 reduce to 5x5conv %
conv5x5a = convolution2dLayer(5,32,'Padding',2,'Stride',1,'Name','conv5x5a');
lgraph=addLayers(lgraph,conv5x5a);
lgraph=connectLayers(lgraph,'conv5x5z','conv5x5a');

% connecting pool proj with conv1x1a%
conv1x1a = convolution2dLayer(1,64,'Padding',0,'Stride',1,'Name','conv1x1a');
lgraph=addLayers(lgraph,conv1x1a);
lgraph=connectLayers(lgraph,'relu3x3a','conv1x1a');

%----------------------------------------- height7 finish-------------------------------------------------%

%-----------------------------------CONCAT-1----------------------------------------------%
concat1=depthConcatenationLayer(4,'Name','concat1');
lgraph = addLayers(lgraph,concat1);
lgraph = connectLayers(lgraph,'conv1x1','concat1/in1');
lgraph = connectLayers(lgraph,'conv3x3a','concat1/in2');
lgraph = connectLayers(lgraph,'conv5x5a','concat1/in3');
lgraph = connectLayers(lgraph,'conv1x1a','concat1/in4');

%----Here output size is (6x6x(64+128+32+64))--------------------------------------------------------------------------------------%
%----------------------------------------- height8 finish-------------------------------------------------%


%------------------------------------------------------------------------------------------%
%connecting output from concat1 with conv1x1b %
conv1x1b = convolution2dLayer(1,128,'Padding',0,'Stride',1,'Name','conv1x1b');
lgraph=addLayers(lgraph,conv1x1b);
lgraph=connectLayers(lgraph,'concat1','conv1x1b');

% connecting output at concat1 with 3x3reduce %

% TAKING 3X3REDUCE AS conv3x3z%
conv3x3y = convolution2dLayer(3,128,'Padding',1,'Stride',1,'Name','conv3x3y');
lgraph = addLayers(lgraph,conv3x3y);
lgraph = connectLayers(lgraph,'concat1','conv3x3y');
%TAKING 5X5REDUCE AS RELu%
conv5x5y = convolution2dLayer(5,32,'Padding',2,'Stride',1,'Name','conv5x5y');
lgraph = addLayers(lgraph,conv5x5y);
lgraph = connectLayers(lgraph,'concat1','conv5x5y');
% Pool proj code%
relu3x3ab = reluLayer('Name','relu3x3ab');
lgraph = addLayers(lgraph,relu3x3ab);
lgraph = connectLayers(lgraph,'concat1','relu3x3ab');
%---------------------------------------- height 9 finish-------------------------------------------------%



% connecting 3x3 reduce to 3x3conv %
conv3x3ab = convolution2dLayer(3,192,'Padding',1,'Stride',1,'Name','conv3x3ab');
lgraph=addLayers(lgraph,conv3x3ab);
lgraph=connectLayers(lgraph,'conv3x3y','conv3x3ab');


% connecting 5x5 reduce to 5x5conv %
conv5x5ab = convolution2dLayer(5,96,'Padding',2,'Stride',1,'Name','conv5x5ab');
lgraph=addLayers(lgraph,conv5x5ab);
lgraph=connectLayers(lgraph,'conv5x5y','conv5x5ab');

% connecting pool proj with conv1x1a%
conv1x1ab = convolution2dLayer(1,128,'Padding',0,'Stride',1,'Name','conv1x1ab');
lgraph=addLayers(lgraph,conv1x1ab);
lgraph=connectLayers(lgraph,'relu3x3ab','conv1x1ab');

%----------------------------------------- height 10 finish-------------------------------------------------%
%-------------------------------Concat-2-----------------------------------------------------%
concat2=depthConcatenationLayer(4,'Name','concat2');
lgraph = addLayers(lgraph,concat2);
lgraph = connectLayers(lgraph,'conv1x1b','concat2/in1');
lgraph = connectLayers(lgraph,'conv3x3ab','concat2/in2');
lgraph = connectLayers(lgraph,'conv5x5ab','concat2/in3');
lgraph = connectLayers(lgraph,'conv1x1ab','concat2/in4');
%---------------------------------------------------------------------------------------------%
%----------------------------------------height 11 finish-------------------------------------------------%
% connect concat2 with pool3%

pool3=maxPooling2dLayer(3,'padding',[1 0 1 0],'Stride',2,'Name','pool3');
lgraph = addLayers(lgraph,pool3);
lgraph = connectLayers(lgraph,'concat2','pool3');




%----------------------------------------- height 12 finish-------------------------------------------------%



%-------------------INCEPTION LAYER 3-------------------------------------------------------%
%------------------------------------------------------------------------------------------%

%------------------------------------------------------------------------------------------%
%connecting output from pool3 with conv1x1c %
conv1x1c = convolution2dLayer(1,192,'Padding',0,'Stride',1,'Name','conv1x1c');
lgraph=addLayers(lgraph,conv1x1c);
lgraph=connectLayers(lgraph,'pool3','conv1x1c');



% TAKING 3X3REDUCE AS conv3x3x%
conv3x3x = convolution2dLayer(3,96,'Padding',1,'Stride',1,'Name','conv3x3x');
lgraph = addLayers(lgraph,conv3x3x);
lgraph = connectLayers(lgraph,'pool3','conv3x3x');
%TAKING 5X5REDUCE AS RELu%
conv5x5x = convolution2dLayer(5,16,'Padding',2,'Stride',1,'Name','conv5x5x');
lgraph = addLayers(lgraph,conv5x5x);
lgraph = connectLayers(lgraph,'pool3','conv5x5x');
% Pool proj code%
relu3x3abc = reluLayer('Name','relu3x3abc');
lgraph = addLayers(lgraph,relu3x3abc);
lgraph = connectLayers(lgraph,'pool3','relu3x3abc');
%---------------------------------------- height 13 finish-------------------------------------------------%



% connecting 3x3 reduce to 3x3conv %
conv3x3abc = convolution2dLayer(3,208,'Padding',1,'Stride',1,'Name','conv3x3abc');
lgraph=addLayers(lgraph,conv3x3abc);
lgraph=connectLayers(lgraph,'conv3x3x','conv3x3abc');


% connecting 5x5 reduce to 5x5conv %
conv5x5abc = convolution2dLayer(5,48,'Padding',2,'Stride',1,'Name','conv5x5abc');
lgraph=addLayers(lgraph,conv5x5abc);
lgraph=connectLayers(lgraph,'conv5x5x','conv5x5abc');

% connecting pool proj with conv1x1a%
conv1x1abc = convolution2dLayer(1,192,'Padding',0,'Stride',1,'Name','conv1x1abc');
lgraph=addLayers(lgraph,conv1x1abc);
lgraph=connectLayers(lgraph,'relu3x3abc','conv1x1abc');

%----------------------------------------- height 14 finish-------------------------------------------------%

%-----------------------------CONCAT 3-----------------------------------------------------%
concat3=depthConcatenationLayer(4,'Name','concat3');
lgraph = addLayers(lgraph,concat3);
lgraph = connectLayers(lgraph,'conv1x1c','concat3/in1');
lgraph = connectLayers(lgraph,'conv3x3abc','concat3/in2');
lgraph = connectLayers(lgraph,'conv5x5abc','concat3/in3');
lgraph = connectLayers(lgraph,'conv1x1abc','concat3/in4');
%-------------------------------------------------------------------------------------------%


% connect concat3 with pool 4%
pool4=maxPooling2dLayer(3,'padding',0,'Stride',2,'Name','pool4');
lgraph = addLayers(lgraph,pool4);
lgraph = connectLayers(lgraph,'concat3','pool4');
% two inner product and output is remaining %


fc1 = fullyConnectedLayer(4096,'Name','fc1');
lgraph = addLayers(lgraph,fc1);
lgraph = connectLayers(lgraph,'pool4','fc1');
relu_4 = reluLayer('Name','relu_4');
lgraph = addLayers(lgraph,relu_4);
lgraph = connectLayers(lgraph,'fc1','relu_4');


fc2 = fullyConnectedLayer(1024,'Name','fc2');
lgraph = addLayers(lgraph,fc2);
lgraph = connectLayers(lgraph,'relu_4','fc2');
%----------------------------------------- height = 18 -----------------------------------------------------------------%

softmax = softmaxLayer('Name','softmax');
lgraph = addLayers(lgraph,softmax);
lgraph = connectLayers(lgraph,'fc2','softmax');

classOutput = classificationLayer('Name','classOutput');
lgraph = addLayers(lgraph,classOutput);
lgraph = connectLayers(lgraph,'softmax','classOutput');

figure
plot(lgraph)
end
