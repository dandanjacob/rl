# Reinformenforcement Learning - ALE-Assault

Crie o ambiente virtual com o comando:
```bash
    python3 -m venv venv
```

Ative o ambiente virtual com o comando:
```bash - Linux
    source venv/bin/activate
```
```bash - Windows
    venv\Scripts\activate
```
Instale as dependências com o comando:
```bash
    pip install -r requirements.txt

```

## Resumo
 Trabalho elaborado por Breno Marquez Azevedo e Daniel Jacob Tonn para avaliação da disciplina
de Reforcement Learning, coordenada pelo professor Flávio Codeço Coelho, com objetivo de  criar
um ambiente de simulação onde agentes de aprendizado de máquina podem aprender a jogar o jogo com 
eficiência. O projeto consiste em criar um ambiente de aprendizado de máquina para o jogo "Assault"
utilizando as bibliotecas Gymnasium e Pytorch, ambas em Python. 

## Implementação
O ambiente foi preparado usando a biblioteca Gymnasium[atari], enquanto o restante do código foi feito usando a biblioteca Pytorch além de, claro, bibliotecas adicionais como os, numpy, etc. O modelo de prendizagem utilizado foi o DQN, que é um algoritmo de aprendizado por reforço que utiliza redes neurais, onde os pesos fornecidos pela rede neural são os valores Q, que são os valores que o agente recebe do ambiente pra decidir qual ação tomar.

## O jogo
Assault consiste em jogo de tiro em 2D, onde o jogador controla um personagem, normalmente
 presente na parte inferior da tela, movendo-o para os lados e atirando  para cima e para os lados 
 contra inimigos (estespresentes majoritariamente na parte superior da tela). A complexidade dos 
 movimentos, tiros e inimigos pode variar conforme as fases do jogo avançarem. O objetivo final 
 é alcançar a maior pontuação - avançar o maior número de fases - sem que o personagem controlado
  seja atingido. 

### Estados
Os estados são as informações que o agente recebe do ambiente. No jogo "Assault", os estados são
 as informações sobre a posição do personagem, a posição dos inimigos, a quantidade de munição e
 a quantidade de vidas restantes. Essas informações são passadas através de uma imagem, um espaço de 
 observação é matriz (array) de 3 canais (RGB), onde cada canal é um número inteiro sem sinal (uint8)
  variando de 0 a 255. A forma da matriz é (210, 160, 3), ou seja, 210 linhas, 160 colunas.
O espaço de observação é contínuo e, portanto, você pode ter qualquer valor real dentro do intervalo
 especificado. Em outras palavras, o espaço de observação é uma imagem.


### Ações
As ações são as escolhas que os agentes podem fazer em um determinado estado. No jogo "Assault", as
 ações são atirar e andar, como descritas abaixo.
 Num  | Action
 0    | NOOP
 1    | FIRE
 2    | UP
 3    | RIGHT
 4    | LEFT
 5    | RIGHTFIRE
 6    | LEFTFIRE

### Recompensas
As recompensas são os valores que o agente recebe do ambiente após realizar uma ação.
Aqui, a recompensa é quando ele acerta algum inimigo, o que faz com que sua recompensa seja também seu score.




#referencias
-> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
-> https://www.gymlibrary.dev/environments/atari/assault/
-> https://github.com/Gouet/PPO-pytorch

