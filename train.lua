local M = require("evoNN")

local function simulate(nn)
  nn:setNormalization({0, 0}, {1, 1}, {0}, {1})
  local entrada = {0.5, 0.5}
  local alvo = {0.8}
  local saida = nn:predict(entrada)
  local erro = math.abs(saida[1] - alvo[1])
  return -erro
end

local cfg = {
  inputSize = 2,
  hiddenLayers  = {4, 4},
  outputSize = 1,
  actNames = {"relu", "sigmoid"},
  popSize = 30, -- tamanho da população
  eliteCount = 2, -- número inicial de elites copiados diretamente para a próxima geração
  eliteAdaptativo = true, -- habilita ajuste dinâmico do elitismo conforme desempenho
  eliteCountMin = 1, -- mínimo de elites quando adaptativo
  eliteCountMax = 6, -- máximo de elites quando adaptativo
  eliteBoostThreshold = 0.2, -- se (melhor - média) > threshold, aumenta eliteCount
  eliteDropThreshold = 0.05, -- se (melhor - média) < threshold, diminui eliteCount
  mutRateStart  = 0.3, -- taxa de mutação inicial
  mutRateMin = 0.05, -- taxa de mutação mínima
  mutStd = 0.1, -- desvio padrão para ruído gaussiano adicionado nas mutações
  generations = 50, -- número máximo de gerações
  selectionType = "tournament", -- método de seleção: "tournament" ou "roulette"
  tournamentK = 3, -- tamanho do torneio
  crossoverType = "uniform", -- tipo de crossover: "uniform", "onepoint", "blend" ou padrão (mix)
  saveFile = "melhor_modelo.lua",
  maxNoImprovement = 5,
  nnOpts = { initType = "he", initDist = "gaussian" }
} -- define como os pesos da rede são inicializados: método de escala (e.g., "he"/"xavier") e distribuição de amostragem (e.g., "gaussian"/"uniform").

M.train(cfg, simulate)
