local M = {}
math.randomseed(os.time())

-- Utils
local utils = {}

function utils.gaussian(m, s)
  local u1, u2 = math.random(), math.random()
  if u1 < 1e-12 then u1 = 1e-12 end
  return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2) * s + m
end

function utils.deepCopy(o)
  if type(o) ~= "table" then return o end
  local c = {}
  for k, v in pairs(o) do
    c[utils.deepCopy(k)] = utils.deepCopy(v)
  end
  return c
end

local function isArray(t)
  local n = 0
  for k, _ in pairs(t) do
    if type(k) ~= "number" then return false end
    if k > n then n = k end
  end
  for i = 1, n do
    if t[i] == nil then return false end
  end
  return true
end

local function serialize(v)
  if type(v) == "table" then
    if isArray(v) then
      local s = "{"
      for _, val in ipairs(v) do
        s = s .. serialize(val) .. ","
      end
      return s .. "}"
    else
      local s = "{"
      for k, val in pairs(v) do
        s = s .. "[" .. serialize(k) .. "]=" .. serialize(val) .. ","
      end
      return s .. "}"
    end
  elseif type(v) == "string" then
    return string.format("%q", v)
  else
    return tostring(v)
  end
end

local function tanh_safe(x)
  local e2x = math.exp(2 * x)
  return (e2x - 1) / (e2x + 1)
end

local function leaky_relu(x, a) return x > 0 and x or a * x end

local function softmax(vec)
  local m = -math.huge
  for _, v in ipairs(vec) do if v > m then m = v end end
  local exps = {}
  local sum = 0
  for i, v in ipairs(vec) do
    local e = math.exp(v - m)
    exps[i] = e
    sum = sum + e
  end
  for i = 1, #exps do exps[i] = exps[i] / sum end
  return exps
end

local activations = {
  relu    = function(x) return x > 0 and x or 0 end,
  sigmoid = function(x) return 1 / (1 + math.exp(-x)) end,
  tanh    = tanh_safe,
  linear  = function(x) return x end,
  leaky_relu = function(x) return leaky_relu(x, 0.01) end,
  softmax = softmax
}

local function initWeightStdByType(typeName, inSz, outSz)
  if typeName == "he" then
    return math.sqrt(2 / inSz)
  elseif typeName == "xavier" then
    return math.sqrt(2 / (inSz + outSz))
  else
    return math.sqrt(1 / inSz)
  end
end

local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

function NeuralNetwork.new(i, h, o, a, opts)
  opts = opts or {}
  local self = setmetatable({}, NeuralNetwork)
  self.arch = { i, unpack(h), o }
  self.actNames = a or {}
  self.weights, self.biases = {}, {}
  self.initType = opts.initType or "relu" -- keep backward compat: "relu" uses old init heuristic
  self.initDist = opts.initDist or "gaussian"
  for L = 1, #self.arch - 1 do
    local inSz, outSz = self.arch[L], self.arch[L + 1]
    local actName = self.actNames[L] or "relu"
    local std = initWeightStdByType(self.initType, inSz, outSz)
    self.weights[L], self.biases[L] = {}, {}
    for j = 1, outSz do
      self.biases[L][j] = 0
      self.weights[L][j] = {}
      for k = 1, inSz do
        if self.initDist == "gaussian" then
          self.weights[L][j][k] = utils.gaussian(0, std)
        elseif self.initDist == "uniform" then
          local lim = std * math.sqrt(3)
          self.weights[L][j][k] = (math.random() * 2 - 1) * lim
        else
          self.weights[L][j][k] = utils.gaussian(0, std)
        end
      end
    end
  end
  return self
end

function NeuralNetwork:forward(x, opts)
  opts = opts or {}
  local a = x
  local preActs = {}
  for L = 1, #self.weights do
    local nA = {}
    for j, w in ipairs(self.weights[L]) do
      local s = self.biases[L][j]
      for i, v in ipairs(w) do
        s = s + v * a[i]
      end
      preActs[L] = preActs[L] or {}
      preActs[L][j] = s
      local act = activations[self.actNames[L]] or activations.relu
      if self.actNames[L] == "softmax" then
        nA[j] = s
      else
        nA[j] = act(s)
      end
    end
    a = nA
  end
  local lastL = #self.weights
  if self.actNames[lastL] == "softmax" then
    a = activations.softmax(a)
  end
  if opts.returnLogits then
    return a, preActs
  else
    return a
  end
end

function NeuralNetwork:clone()
  local c = setmetatable({}, NeuralNetwork)
  c.arch     = utils.deepCopy(self.arch)
  c.weights  = utils.deepCopy(self.weights)
  c.biases   = utils.deepCopy(self.biases)
  c.actNames = utils.deepCopy(self.actNames)
  c.initType = self.initType
  c.initDist = self.initDist
  if self.inMin then
    c.inMin, c.inMax = utils.deepCopy(self.inMin), utils.deepCopy(self.inMax)
    c.outMin, c.outMax = utils.deepCopy(self.outMin), utils.deepCopy(self.outMax)
  end
  return c
end

function NeuralNetwork.crossover(A, B)
  local c = A:clone()
  for L = 1, #c.weights do
    for j = 1, #c.weights[L] do
      if math.random() < 0.5 then
        c.weights[L][j] = utils.deepCopy(B.weights[L][j])
        c.biases[L][j] = B.biases[L][j]
      end
    end
  end
  return c
end

local function onePointCrossover(A, B)
  local c = A:clone()
  local g = 0
  for L = 1, #A.weights do
    for j = 1, #A.weights[L] do
      g = g + #A.weights[L][j] + 1
    end
  end
  local cp = math.random(1, g)
  local idx = 0
  for L = 1, #c.weights do
    for j = 1, #c.weights[L] do
      for i = 1, #c.weights[L][j] do
        idx = idx + 1
        if idx > cp then
          c.weights[L][j][i] = B.weights[L][j][i]
        end
      end
      idx = idx + 1
      if idx > cp then
        c.biases[L][j] = B.biases[L][j]
      end
    end
  end
  return c
end

function NeuralNetwork:mutate(r, s, opts)
  opts = opts or {}
  local layerRates = opts.layerRates
  local decay = opts.decay or 1
  for L = 1, #self.weights do
    local lr = layerRates and (layerRates[L] or r) or r
    for j = 1, #self.weights[L] do
      for i = 1, #self.weights[L][j] do
        if math.random() < lr then
          self.weights[L][j][i] = self.weights[L][j][i] + utils.gaussian(0, s * decay)
        end
      end
      if math.random() < lr then
        self.biases[L][j] = self.biases[L][j] + utils.gaussian(0, s * decay)
      end
    end
  end
end

function NeuralNetwork:setNormalization(inMin, inMax, outMin, outMax)
  self.inMin, self.inMax = inMin, inMax
  self.outMin, self.outMax = outMin, outMax
end

function NeuralNetwork:predict(xRaw)
  local x = xRaw
  local yN = self:forward(x)
  if self.outMin and self.outMax then
    local y = {}
    for i = 1, #yN do
      y[i] = yN[i] * (self.outMax[i] - self.outMin[i]) + self.outMin[i]
    end
    return y
  else
    return yN
  end
end

local Population = {}
Population.__index = Population

function Population.new(cfg)
  local p = setmetatable({}, Population)
  p.cfg = cfg
  p.individuals = {}
  for i = 1, cfg.popSize do
    p.individuals[i] = NeuralNetwork.new(cfg.inputSize, cfg.hiddenLayers, cfg.outputSize, cfg.actNames, cfg.nnOpts)
  end
  return p
end

function Population:evaluate(sim)
  self.fitness = {}
  for i, ind in ipairs(self.individuals) do
    self.fitness[i] = { score = sim(ind), idx = i }
  end
  table.sort(self.fitness, function(a, b) return a.score > b.score end)
end

local function uniformCrossover(A, B)
  local c = A:clone()
  for L = 1, #c.weights do
    for j = 1, #c.weights[L] do
      for i = 1, #c.weights[L][j] do
        c.weights[L][j][i] = math.random() < 0.5 and A.weights[L][j][i] or B.weights[L][j][i]
      end
      c.biases[L][j] = math.random() < 0.5 and A.biases[L][j] or B.biases[L][j]
    end
  end
  return c
end

function Population:selectOne()
  if self.cfg.selectionType == "roulette" then
    local sum = 0
    for _, r in ipairs(self.fitness) do sum = sum + r.score end
    if sum <= 0 then
      return self.individuals[self.fitness[1].idx]
    end
    local pick = math.random() * sum
    local cum = 0
    for _, r in ipairs(self.fitness) do
      cum = cum + r.score
      if cum >= pick then
        return self.individuals[r.idx]
      end
    end
    return self.individuals[self.fitness[#self.fitness].idx]
  else
    local best, bs = nil, -math.huge
    for _ = 1, self.cfg.tournamentK do
      local c = self.fitness[math.random(#self.fitness)]
      if c.score > bs then best, bs = self.individuals[c.idx], c.score end
    end
    return best
  end
end

local function blendCrossover(A, B, alpha)
  alpha = alpha or 0.5
  local c = A:clone()
  for L = 1, #c.weights do
    for j = 1, #c.weights[L] do
      for i = 1, #c.weights[L][j] do
        local wa, wb = A.weights[L][j][i], B.weights[L][j][i]
        c.weights[L][j][i] = wa * alpha + wb * (1 - alpha)
      end
      local ba, bb = A.biases[L][j], B.biases[L][j]
      c.biases[L][j] = ba * alpha + bb * (1 - alpha)
    end
  end
  return c
end

function Population:evolve(gen)
  local cfg = self.cfg
  local rate = cfg.mutRateStart * (1 - gen / cfg.generations)
  if rate < cfg.mutRateMin then rate = cfg.mutRateMin end

  local nextGen = {}
  local mean = 0
  for _, r in ipairs(self.fitness) do mean = mean + r.score end
  mean = mean / #self.fitness

  local eliteCount = cfg.eliteCount
  if cfg.eliteAdaptativo then
    local delta = self.fitness[1].score - mean
    if delta > (cfg.eliteBoostThreshold or 0.2) then
      eliteCount = math.min(cfg.eliteCountMax or cfg.eliteCount, eliteCount + 1)
    elseif delta < (cfg.eliteDropThreshold or 0.05) then
      eliteCount = math.max(cfg.eliteCountMin or 1, eliteCount - 1)
    end
  end
  for i = 1, eliteCount do
    nextGen[i] = self.individuals[self.fitness[i].idx]:clone()
  end

  while #nextGen < cfg.popSize do
    local a, b = self:selectOne(), self:selectOne()
    local child
    if cfg.crossoverType == "onepoint" then
      child = onePointCrossover(a, b)
    elseif cfg.crossoverType == "uniform" then
      child = uniformCrossover(a, b)
    elseif cfg.crossoverType == "blend" then
      child = blendCrossover(a, b, cfg.blendAlpha or 0.5)
    else
      child = NeuralNetwork.crossover(a, b)
    end
    child:mutate(rate, cfg.mutStd, cfg.mutOpts)
    table.insert(nextGen, child)
  end
  self.individuals = nextGen
end

function M.saveModel(nn, fname)
  local f = assert(io.open(fname, "w"))
  f:write(
    "return {arch=" .. serialize(nn.arch) ..
    ",weights=" .. serialize(nn.weights) ..
    ",biases=" .. serialize(nn.biases) ..
    ",actNames=" .. serialize(nn.actNames) ..
    ",inMin=" .. serialize(nn.inMin) ..
    ",inMax=" .. serialize(nn.inMax) ..
    ",outMin=" .. serialize(nn.outMin) ..
    ",outMax=" .. serialize(nn.outMax) ..
    "}"
  )
  f:close()
end

function M.loadModel(fname)
  local chunk, e = loadfile(fname)
  if not chunk then error("Erro ao carregar modelo: " .. e) end
  local t = chunk()
  return setmetatable(t, NeuralNetwork)
end

function M.new(i, h, o, a, opts)
  return NeuralNetwork.new(i, h, o, a, opts)
end

function M.train(cfg, sim)
  local pop = Population.new(cfg)
  local best = -math.huge
  local noImprovement = 0
  local maxNoImprovement = cfg.maxNoImprovement or 5

  for gen = 1, cfg.generations do
    pop:evaluate(sim)
    local mx = pop.fitness[1].score
    local mean = 0
    for _, r in ipairs(pop.fitness) do mean = mean + r.score end
    mean = mean / #pop.fitness

    print(string.format("Gen %3d | max=%.4f | mean=%.4f", gen, mx, mean))

    if mx > best then
      best = mx
      noImprovement = 0
      M.saveModel(pop.individuals[pop.fitness[1].idx], cfg.saveFile or "best_model.lua")
    else
      noImprovement = noImprovement + 1
      print("Sem melhoria - contador:", noImprovement)
      if noImprovement >= maxNoImprovement then
        print("Parando treino após " .. maxNoImprovement .. " gerações sem progresso")
        break
      end
    end

    if gen < cfg.generations then pop:evolve(gen) end
  end

  print("Melhor fitness:", best)
end

function M.runModelOnGame(file, game)
  local nn = M.loadModel(file)
  game.reset()
  while not game.isOver() do
    local o = nn:predict(game.getState())
    game.act(o)
  end
  return game.getFitness()
end

function M.blendCrossover(A, B, alpha) return blendCrossover(A, B, alpha) end
function M.getSummary(nn) return nn:summary() end

return M
