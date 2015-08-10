require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  -- if x_i < 0, x_i = 0
  -- else x_i = x_i^2
  self.output[torch.le(self.output, 0)] = 0
  self.output:cmul(self.output)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- if x_i > 0 then dz_i/dx_i = 2x_i
  -- else dz_i/dx_i = 0
  self.gradInput[torch.le(input, 0)] = 0
  self.gradInput:mul(2)
  self.gradInput:cmul(input)
  return self.gradInput
end

