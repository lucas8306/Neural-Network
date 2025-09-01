# Neural-Network

Este repositório contém o projeto Neural Network, desenvolvido e mantido exclusivamente por mim.

Sobre:

Este projeto implementa redes neurais com algoritmos genéticos, inteiramente escrito em Lua. O objetivo é explorar técnicas de aprendizado e otimização inspiradas na evolução biológica, com foco na simplicidade e controle total.

A biblioteca foi criada com ajuda de inteligência artificial, mas toda a estrutura, adaptação e validação foram feitas manualmente por mim. Além disso, editei diversas partes para garantir a clareza e simplicidade.

Funcionalidades e características:

- Inicialização com pesos gaussianos.
- Ativações: relu, sigmoid, tanh, linear.
- Mutação adaptativa e crossover genético (uniform ou onepoint)
- Seleção por torneio ou roleta.
- Normalização de entrada e saída.
- Salvamento e carregamento de modelos.
- Simulação customizável.
- Treinamento com elitismo e controle de gerações.
- Sem dependências externas.

Na linha 322 contém:

`
print("Sem melhoria na geração", gen, "- interrompendo treino")
break
`

Esse break interrompe o treinamento caso não haja melhoria de fitness em uma geração. Embora isso possa acelerar testes, é recomendável remover essa linha para permitir que o algoritmo evolua por todas as gerações.
