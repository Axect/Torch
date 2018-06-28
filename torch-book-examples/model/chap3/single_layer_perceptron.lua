--[[
-- Training
-- by socurites
--]]

-- Load data file
X = torch.DoubleTensor{ {0.1, 0.3}, {0.8, 1.2}, {2.5, 0.3}, {1.2, 4.2}, {0.4, 0.8}, {3.2, 1.2}, {1.7, 0.1}, {3.1, 5.7}, {2.9, 0.5} }
Y = torch.DoubleTensor{ 0, 0, 1, 0, 0, 1, 1, 0, 1 }


--------------------------------------------------------------------

-- Initialize the network
-- # of input features
n = 2
-- # of output classes
K = 1

require 'nn'
-- Configure model network
model = nn.Sequential()
model:add(nn.Linear(n, K)) 
model:add(nn.Sigmoid()) 

---------------------------------------------------------------------

-- 1st epoch: Train the network
h_x = model:forward(X)

-- Define loss function
loss = nn.MSECriterion()

-- Define learning rate
optimState = {learningRate = 0.35}

-- Find loss
J = loss:forward(h_x, Y)
print(string.format("current loss: %.2f", J))

-- Update parameters using back propagation
dJ_dh_x = loss:backward(h_x, Y)
model:backward(X, dJ_dh_x)
model:updateParameters(optimState.learningRate)


maxIteration = 10
for epoch = 2, maxIteration do
    -- initalized gradWeight, gradBias to zero which are used updating parameters
    -- this is equal: theta, gradTheta = model:getParameters(); gradTheta:zero()
    model:get(1).gradWeight:zero()
    model:get(1).gradBias:zero()
    h_x = model:forward(X)    
    -- Find loss
    J = loss:forward(h_x, Y)
    print(string.format("current loss: %.2f", J))
    -- Update parameters using back propagation
    dJ_dh_x = loss:backward(h_x, Y)
    model:backward(X, dJ_dh_x)
    model:updateParameters(optimState.learningRate)
end


-- should be closer to 0.5
model:forward(torch.Tensor{3.9, 2.7})

-- should be larger than 0.5
model:forward(torch.Tensor{3.9, 1.9})
