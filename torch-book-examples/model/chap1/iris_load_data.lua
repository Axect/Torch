--[[
-- Load datasets into trainSet and testSet by 80:20 ratios
-- Converts labes type from text into Tensor
-- by socurites
--]]

-- Training dataset
local trainSet = {data = nil, label = nil}
-- test dataset
local testSet = {data = nil, label = nil}

-- Load data file
local file = io.open(paths.concat('data/iris', 'iris.data'))

math.randomseed(os.time())
-- Convert comman-sperated data into Tensor
for line in file:lines() do 
    if (string.len(line) > 0) then
        local x1, x2, x3, x4, species = unpack(line:split(","))
        local input = torch.Tensor{{x1, x2, x3, x4}};

		local output = nil
        if (species == "Iris-setosa") then
            output = torch.Tensor{1}
        elseif (species == "Iris-versicolor") then
            output = torch.Tensor{2}
        else
            output = torch.Tensor{3}
        end

        if (math.random() > 0.2) then
            trainSet.data = ( trainSet.data == nil and input or torch.cat(trainSet.data, input, 1) )
            trainSet.label = ( trainSet.label == nil and output or torch.cat(trainSet.label, output, 1) )
        else
            testSet.data = ( testSet.data == nil and input or torch.cat(testSet.data, input, 1) )
            testSet.label = ( testSet.label == nil and output or torch.cat(testSet.label, output, 1) )
        end
    end
end

return {trainSet, testSet}
