require 'torch';
require 'nn';
-- @param k := depth of feature maps for hidden layers. AlphaGo used k=192.
k = 192

net = nn.Sequential()
-- layer 1
net:add(nn.SpatialZeroPadding(2,2,2,2))      -- pad 48x19x19 tensor into 48x23x23 tensor
net:add(nn.SpatialConvolution(49, k, 5, 5))  -- convolve: #filters = k, size = 5x5, stride = 1x1
net:add(nn.ReLU())

-- layers 2-12
for i=2,12 do
  net:add(nn.SpatialZeroPadding(1,1,1,1))    -- pad kx19x19 tensor into kx21x21 tensor
  net:add(nn.SpatialConvolution(k, k, 3, 3)) -- convolve: #filters = k, size = 3x3, stride = 1x1
  net:add(nn.ReLU())
end

-- layer 13
net:add(nn.SpatialConvolution(k, 1, 1, 1))   -- convolve. #filters = 1, size = 1x1, stride = 1x1
net:add(nn.View(19*19))                      -- reshape into 1 dimension
net:add(nn.Linear(19*19,256))
net:add(nn.ReLU())
net:add(nn.Linear(256,1))
net:add(nn.Tanh())

print(net:__tostring())
