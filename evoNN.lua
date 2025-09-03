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

-- Ativações
local function tanh_safe(x)
  -- tanh(x) = (e^{2x}-1)/(e^{2x}+1); forma estável
  local e2x = math.exp(2 * x)
  return (e2x - 1) / (e2x + 1)
end

local activations = {
  relu    = function(x) return x > 0 and x or 0 end,
  sigmoid = function(x) return 1 / (1 + math.exp(-x)) end,
  tanh    = tanh_safe,
  linear  = function(x) return x end
}

local function initWeightStd(a, i)
  return a == "relu" and math.sqrt(2 / i) or math.sqrt(1 / i)
end

-- Neural Network
local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

function NeuralNetwork.new(i, h, o, a)
  local self = setmetatable({}, NeuralNetwork)
  self.arch = { i, table.unpack(h), o }
  self.actNames = a or {}
  self.weights, self.biases = {}, {}
  for L = 1, #self.arch - 1 do
    local inSz, outSz = self.arch[L], self.arch[L + 1]
    local actName = self.actNames[L] or "relu"
    local std = initWeightStd(actName, inSz)
    self.weights[L], self.biases[L] = {}, {}
    for j = 1, outSz do
      self.biases[L][j] = 0
      self.weights[L][j] = {}
      for k = 1, inSz do
        self.weights[L][j][k] = utils.gaussian(0, std)
      end
    end
  end
  return self
end

function NeuralNetwork:forward(x)
  local a = x
  for L = 1, #self.weights do
    local nA = {}
    for j, w in ipairs(self.weights[L]) do
      local s = self.biases[L][j]
      for i, v in ipairs(w) do
        s = s + v * a[i]
      end
      local act = activations[self.actNames[L]] or activations.relu
      nA[j] = act(s)
    end
    a = nA
  end
  return a
end

function NeuralNetwork:clone()
  local c = setmetatable({}, NeuralNetwork)
  c.arch     = utils.deepCopy(self.arch)
  c.weights  = utils.deepCopy(self.weights)
  c.biases   = utils.deepCopy(self.biases)
  c.actNames = utils.deepCopy(self.actNames)
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

function NeuralNetwork:mutate(r, s)
  for L = 1, #self.weights do
    for j = 1, #self.weights[L] do
      for i = 1, #self.weights[L][j] do
        if math.random() < r then
          self.weights[L][j][i] = self.weights[L][j][i] + utils.gaussian(0, s)
        end
      end
      if math.random() < r then
        self.biases[L][j] = self.biases[L][j] + utils.gaussian(0, s)
      end
    end
  end
end

function NeuralNetwork:setNormalization(inMin, inMax, outMin, outMax)
  self.inMin, self.inMax = inMin, inMax
  self.outMin, self.outMax = outMin, outMax
end

function NeuralNetwork:predict(xRaw)
  local x
  if self.inMin and self.inMax then
    x = {}
    for i = 1, #xRaw do
      local denom = (self.inMax[i] - self.inMin[i])
      x[i] = denom ~= 0 and (xRaw[i] - self.inMin[i]) / denom or 0
    end
  else
    x = xRaw
  end
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

-- Population / GA
local Population = {}
Population.__index = Population

function Population.new(cfg)
  local p = setmetatable({}, Population)
  p.cfg = cfg
  p.individuals = {}
  for i = 1, cfg.popSize do
    p.individuals[i] = NeuralNetwork.new(cfg.inputSize, cfg.hiddenLayers, cfg.outputSize, cfg.actNames)
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
      -- fallback: escolhe o melhor quando não há soma positiva
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

function Population:evolve(gen)
  local cfg = self.cfg
  local rate = cfg.mutRateStart * (1 - gen / cfg.generations)
  if rate < cfg.mutRateMin then rate = cfg.mutRateMin end

  local nextGen = {}
  local mean = 0
  for _, r in ipairs(self.fitness) do mean = mean + r.score 
    mean = mean / #self.fitness
  end
    -- Elitismo adaptativo
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
    else
      child = NeuralNetwork.crossover(a, b)
    end
  child:mutate(rate, cfg.mutStd)
    table.insert(nextGen, child)
  end
  self.individuals = nextGen
end

-- API
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
  -- devolve já com os métodos habilitados
  return setmetatable(t, NeuralNetwork)
end

function M.new(i, h, o, a)
  return NeuralNetwork.new(i, h, o, a)
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
  local nn = M.loadModel(file) -- já vem com metatable
  game.reset()
  while not game.isOver() do
    local o = nn:predict(game.getState())
    game.act(o)
  end
  return game.getFitness()
end

return M