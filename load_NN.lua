local M = require("evoNN")

-- Função de simulação: avalia a rede
local function simulate(nn)
  nn:setNormalization({0, 0}, {1, 1}, {0}, {1}) -- normalização
  local entrada = {0.5, 0.5}
  local alvo = {0.8}
  local saida = nn:predict(entrada)
  local erro = math.abs(saida[1] - alvo[1])
  return -erro
end

-- Configuração do treinamento
local cfg = {
  inputSize     = 2,
  hiddenLayers  = {4, 4},
  outputSize    = 1,
  actNames      = {"relu", "sigmoid"},
  popSize       = 30,
  eliteCount    = 2,
  mutRateStart  = 0.3,
  mutRateMin    = 0.05,
  mutStd        = 0.1,
  generations   = 50,
  selectionType = "tournament", -- ou "roulette"
  tournamentK   = 3,
  crossoverType = "onepoint",   -- ou "uniform"
  saveFile      = "melhor_modelo.lua"
}

-- Inicia o treinamento
M.train(cfg, simulate)