local M = {}

function M.pow(n)
  return function(x)
           local sum = 1
           for i = 1, n do
             sum = sum * x
           end
         return sum
         end
end

return M
