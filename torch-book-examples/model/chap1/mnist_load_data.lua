--[[
-- Load datasets into trainSet and testSet respectively
-- Normalize datasets with mean of 0.0 and standard deveiation of 1.0
-- by socurites
--]]

-- Load data file
require 'paths'




function loadDataSet(trainSize, testSize)
    local trainSetSize = trainSize
    local testSetSize = testSize

    local trainFile = paths.concat('data/mnist.t7/', 'train_32x32.t7')
    local trainSet = torch.load(trainFile, 'ascii')
    trainSet.data = trainSet.data:double()
    trainSet.data = trainSet.data[{{1, trainSetSize}, {}, {}, {}}]
    trainSet.labels = trainSet.labels:double()
    trainSet.labels = trainSet.labels[{{1, trainSetSize}}]


    local testFile = paths.concat('data/mnist.t7/', 'test_32x32.t7')
    local testSet = torch.load(testFile, 'ascii')
    testSet.data = testSet.data:double()
    testSet.data = testSet.data[{{1, testSetSize},{},{},{}}]
    testSet.labels = testSet.labels:double()
    testSet.labels = testSet.labels[{{1, testSetSize}}]

    ---- normalize training data
    local mean = trainSet.data[{ {}, {1}, {}, {}  }]:mean()
    trainSet.data[{ {}, {1}, {}, {}  }]:add(-mean)
    local stdv = trainSet.data[{ {}, {1}, {}, {}  }]:std() -- std estimation
    trainSet.data[{ {}, {1}, {}, {}  }]:div(stdv)

    ---- normalize test data
    mean = testSet.data[{ {}, {1}, {}, {}  }]:mean()
    testSet.data[{ {}, {1}, {}, {}  }]:add(-mean)
    stdv = testSet.data[{ {}, {1}, {}, {}  }]:std()
    testSet.data[{ {}, {1}, {}, {}  }]:div(stdv)

    return {trainSet, testSet}
end







