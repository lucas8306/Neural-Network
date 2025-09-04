local M = require("evoNN")

-- Carrega o modelo salvo
local nn = M.loadModel("melhor_modelo.lua")

-- Define a entrada de teste
local entrada = {0.5, 0.5}

-- Normalização
nn:setNormalization({0, 0}, {1, 1}, {0}, {1})

-- Faz a predição
local saida = nn:predict(entrada)

-- Exibe o resultado
print("Entrada:", table.concat(entrada, ", "))
print("Saída prevista:", table.concat(saida, ", "))