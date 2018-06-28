--[[
-- Training
-- by socurites
--]]

-- Load data file
require 'model.chap1.mnist_load_data'
local trainSize = 600
local testSize = 100
local datasets = loadDataSet(trainSize, testSize)
local trainSet = datasets[1]
local testSet = datasets[2]

--------------------------------------------------------------------

-- Initialize the network
require 'nn'
---- # of input channels of image
local input_channel = 1
---- # of output classes
local outputs = 10

-- Configure model network
local model = nn.Sequential()
---- First convloution layer
------ 1 channel, 6 output channels, 5x5 convolution kernel
model:add(nn.SpatialConvolution(input_channel, 6, 5, 5))
model:add(nn.ReLU())
------ A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialMaxPooling하(2,2,2,2))
---- Second convloution layer
------ 6 channels, 16 output channels, 5x5 convolution kernel
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
---- First fully connected layer
------ reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model:add(nn.View(16*5*5))
model:add(nn.Linear(16*5*5, 120))
model:add(nn.ReLU())
---- Second fully connected layer
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())
---- Gaussian layer 
model:add(nn.Linear(84, outputs))
model:add(nn.LogSoftMax())

---------------------------------------------------------------------

-- Define loss function
local loss = nn.ClassNLLCriterion()


-- Transform trainSet to readble by trainer
setmetatable(trainSet, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);

function trainSet:size() 
    return self.data:size(1) 
end

-- Train the network
print("== start training")
local trainer = nn.StochasticGradient(model, loss)
trainer.learningRate = 0.01
trainer.maxIteration = 10

trainer:train(trainSet)
print("== end training")

---------------------------------------------------------------------

-- Evaluate trained model
print("== start evaluation")
local correct = 0
for i=1, testSet.data:size(1) do
    local groundtruth = testSet.labels[i]
    local prediction = model:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print("== end evaluation")
print(string.format("Evaluation: %.5f%s", 100 * correct / testSet.data:size(1), "% correct"))

