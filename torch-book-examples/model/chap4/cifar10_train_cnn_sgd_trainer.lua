--[[
-- Training
-- by socurites
--]]

-- Load data file
require 'paths'

local trainSet = torch.load(paths.concat('data/cifar-10/', 'cifar10-train.t7'))
local testSet = torch.load(paths.concat('data/cifar-10/', 'cifar10-test.t7'))
local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

--------------------------------------------------------------------

-- Preprocessing data: normalized and transform for StochasticGradient trainer
---- Normalize trainSet
mean = {}
stdv = {}
trainSet.data = trainSet.data:double()
for i=1,3 do
    mean[i] = trainSet.data[{ {}, {i}, {}, {}  }]:mean()
    stdv[i] = trainSet.data[{ {}, {i}, {}, {}  }]:std()

    trainSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    trainSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

---- Normalize testSet
testSet.data = testSet.data:double()
for i=1,3 do
    mean[i] = testSet.data[{ {}, {i}, {}, {}  }]:mean()
    stdv[i] = testSet.data[{ {}, {i}, {}, {}  }]:std()

    testSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    testSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

---- Transform for StochasticGradient trainer
setmetatable(trainSet, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

function trainSet:size() 
    return self.data:size(1) 
end


--------------------------------------------------------------------

-- Initialize the network
require 'nn'
local model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 6, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(16*5*5))
model:add(nn.Linear(16*5*5, 120))
model:add(nn.ReLU())
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())
model:add(nn.Linear(84, 10))
model:add(nn.LogSoftMax())

---------------------------------------------------------------------

-- Define loss function
local loss = nn.ClassNLLCriterion()


-- Train the network
local trainer = nn.StochasticGradient(model, loss)
trainer.learningRate = 0.01
trainer.maxIteration = 5
trainer:train(trainSet)

---------------------------------------------------------------------

-- Evaluate trained model
correct = 0
for i=1, testSet.data:size(1) do
    local groundtruth = testSet.label[i]
    local prediction = model:forward(testSet.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(string.format("Evaluation: %.5f%s", 100 * correct / testSet.data:size(1), "% correct"))

