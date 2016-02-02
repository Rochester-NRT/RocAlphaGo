--------------------------------------------
-- Architecture of fast SL policy network --
--------------------------------------------
require 'torch';
require 'nn';
require 'cunn';

net = nn.Sequential()
-- TODO architecture goes here
net:add(nn.SoftMax())

net = net:cuda() -- put it on GPU

print(net:__tostring())
