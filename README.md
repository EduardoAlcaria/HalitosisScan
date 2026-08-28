# Hality

Estimativa de indício de halitose a partir de uma foto da língua.

Este repositório contém o redesenho do projeto. A implementação anterior está em
`Hality-Project-main/` e foi mantida apenas como referência histórica — os problemas
que motivaram a reescrita estão listados mais abaixo.

## O que o projeto faz

Uma pessoa fotografa a própria língua. O sistema devolve uma de três respostas:

- **indício de halitose**, com uma medida de confiança
- **sem indício**, com uma medida de confiança
- **inconclusivo**, quando não há base suficiente para responder

Junto com a resposta vai sempre uma orientação para procurar avaliação odontológica.
O sistema **não diagnostica** — ele faz triagem. A diferença importa: um diagnóstico
afirma uma condição, uma triagem sinaliza que vale a pena um profissional olhar.

## Por que uma foto da língua diz alguma coisa

Uma língua com muita saburra — a camada esbranquiçada que se acumula no dorso — é
mais clara e menos saturada que uma língua limpa. Essa diferença é visível na foto e
tem ligação clínica conhecida com halitose.

Medimos e confirmamos: a característica que mais pesa no modelo é o percentil 10 da
saturação da língua, ou seja, **quão pálida é a porção mais pálida**. Isso é a camada
branca, medida de forma contínua.

Também medimos que a foto sozinha prevê melhor que o questionário clínico inteiro:

| Entrada | AUC |
|---|---|
| Foto da língua | 0,797 |
| Questionário completo (sem a pergunta que vaza o rótulo) | 0,615 |

## Os dados

- 345 anamneses com nota clínica de 1 a 3, atribuída por avaliador humano
- 318 fotos utilizáveis, casadas com essas anamneses
- Origem: clínica parceira na PUCRS

O conjunto é pequeno, e isso é o teto do projeto. Não existe nenhum dataset público
com rótulo de halitose — procuramos. Datasets públicos de língua existem aos milhares
de imagens, mas nenhum traz o rótulo que precisamos. Eles servem para outras partes
do sistema, não para o classificador.

## Como funciona

O sistema tem três modelos, não um. Isso é deliberado.

| Modelo | Pergunta que responde | Treinado com |
|---|---|---|
| Gate | "isso é uma língua?" | ~2.300 línguas e ~4.000 não-línguas |
| Segmentador | "onde exatamente está a língua?" | modelo público pronto, ou datasets públicos |
| Classificador | "essa língua indica halitose?" | as 318 fotos com nota clínica |

O motivo de separar: só o último modelo depende das 318 amostras. Os outros dois têm
ordens de magnitude mais dados disponíveis. Se tudo fosse uma rede só, as 318 seriam
o teto de todo o sistema.

O caminho de uma foto até a resposta:

```
foto
 │
 ├─ 1. normalização        decodifica, corrige rotação, redimensiona
 ├─ 2. qualidade           tremida? escura? estourada?        → rejeita
 ├─ 3. gate                tem uma língua aí?                 → rejeita
 ├─ 4. fora-de-distribuição  parece com o que vimos no treino? → rejeita
 ├─ 5. segmentação         recorta a língua
 ├─ 6. sanidade da máscara  deu para delimitar?               → rejeita
 ├─ 7. características     cor e textura, só dentro da língua
 ├─ 8. classificador       probabilidade calibrada
 └─ 9. decisão             faixa de incerteza → inconclusivo
```

Rejeitar é uma resposta bem-sucedida, não um erro, e vem sempre com um motivo
específico — "foto tremida" ajuda, "erro" não.

## O que aprendemos medindo

Estas conclusões vieram de experimentos, não de opinião. Cada uma mudou o desenho.

**Modelo maior não ajuda.** Um backbone pré-treinado grande (DINOv2) teve desempenho
pior (AUC 0,761) que 35 características de cor calculadas à mão (0,801). Esses modelos
são treinados para ignorar cor — e aqui a cor é justamente o sinal.

**Não use augmentation de cor.** Testamos cinco intensidades. Quanto mais forte o
jitter de cor, pior o resultado, sem exceção. O valor usado no pipeline anterior
(±20%) custa 0,036 de AUC. Augmentation geométrica, por outro lado, ajuda.

**Cuidado com confundidores.** Este foi o aprendizado mais caro, e aconteceu três vezes.

Um confundidor é quando as classes do dataset foram coletadas de formas diferentes —
câmeras diferentes, resoluções diferentes, recortes diferentes. Aí o modelo aprende a
reconhecer *a forma de coletar* em vez de reconhecer *a condição*. Ele acerta quase
tudo na validação e desaba em produção, porque a pista desaparece.

Onde encontramos:

- Um dataset externo de língua que tínhamos: 99,5% do rótulo era previsível apenas
  pelo tamanho do arquivo. Descartado como fonte de rótulo.
- Nossa primeira montagem do conjunto do gate: os negativos eram quadrados pequenos e
  os positivos retângulos grandes. O gate deu AUC 1,0000 — resultado sem valor.
- O pipeline anterior do projeto: escolhia o melhor modelo e reportava a métrica no
  mesmo conjunto de validação, o que infla o número por construção.

**O teste que pega isso** custa segundos: tente prever o rótulo usando *apenas* os
metadados do arquivo — largura, altura, bytes, formato. Nenhum pixel. Se der para
acertar, o dataset está enviesado.

Nosso conjunto principal passa: AUC 0,498, ou seja, acaso puro. Ele tem variação
enorme (de 0,12 a 12 megapixels, brilho de 75 a 174, quatro formatos), mas essa
variação está distribuída por igual entre as classes. É esse o objetivo — não é
eliminar variação, é impedir que ela acompanhe o rótulo.

## Limitações que você precisa saber

**Todas as fotos vêm de um mesmo ambiente clínico.** A variação que temos é de
aparelho, não de cenário. Uma selfie caseira, com luz de teto, à noite, não está
representada em nenhuma das 318 fotos. Não temos como estimar quanto o desempenho cai
nesse cenário, porque não existe amostra rotulada dele.

**Prevalência muda tudo.** Na população de clínica, onde 58% dos casos são graves, o
valor preditivo positivo é 0,76 no ponto de operação escolhido. Numa população aberta,
com 20% de prevalência, o mesmo ponto cai para 0,36 — dois terços dos sinalizados não
teriam a condição. Isso é aritmética de triagem e nenhum modelo melhor contorna. É a
razão pela qual a saída é "procure um dentista" e não um veredito.

**Métrica sem cobertura é meia informação.** O classificador é avaliado sobre fotos que
já passaram pelos filtros. Quanto mais o sistema rejeita, melhor ele parece. Um modelo
com AUC 0,90 que rejeita 60% das fotos é pior, como produto, que um com 0,80 que
rejeita 5%. Por isso todo número reportado vem acompanhado da cobertura.

## Como usar

Treinar (uma vez, nesta ordem):

```bash
.venv312/Scripts/python.exe -m hality.train_segmenter    # ~10 min
.venv312/Scripts/python.exe -m hality.train_classifier   # ~2 min
.venv312/Scripts/python.exe -m hality.train_gate         # ~10 min
```

Subir a API e analisar uma foto:

```bash
.venv312/Scripts/python.exe -m uvicorn hality.api:app --port 8000
curl -F "foto=@lingua.jpg" http://127.0.0.1:8000/analisar
```

Resposta:

```json
{
  "veredito": "indicio",
  "motivo": "Indicio compativel com halitose. Procure um dentista para avaliacao.",
  "probabilidade": 0.7627,
  "area_lingua": 0.3672,
  "nitidez": 17.5,
  "confianca_lingua": 0.9992
}
```

`veredito` é um de: `indicio`, `sem_indicio`, `inconclusivo`, `rejeitado`.
Rejeição volta como HTTP 200 — o serviço funcionou e concluiu que não dá para avaliar.

Cada módulo tem um auto-teste: `python -m hality.features`, `python -m hality.data`,
`python -m hality.pipeline`.

## Resultados medidos

| Componente | Métrica |
|---|---|
| Segmentador (U-Net, 483k parâmetros) | IoU 0,842 na validação |
| Classificador — teste trancado (n=48) | AUC 0,871 |
| Classificador — CV repetida (n=318) | AUC 0,797 [0,779 – 0,820] |
| Classificador — validação (n=43) | AUC 0,700 |
| Gate de língua — recall nas fotos próprias | 97,8% |
| Gate de língua — falso-aceite em fotos do COCO | 0,0% |

**Cite ~0,80, não 0,871.** As três estimativas do mesmo classificador variam de 0,700 a
0,871. Isso não é o modelo melhorando entre medições — é o ruído de amostra pequena:
com ~45 exemplos por partição, o intervalo de confiança do AUC é da ordem de ±0,10. A
estimativa confiável é a da validação cruzada repetida, que usa as 318 amostras em 50
divisões diferentes.

Duas fraquezas conhecidas do estado atual:

- **Especificidade 0,500** no ponto de operação. O limiar foi calibrado para
  sensibilidade alta (0,964 no teste), e o custo é que metade dos casos sem indício
  também é sinalizada. É o trade-off correto para triagem, mas precisa estar visível.
- **Os negativos do gate são fotos do COCO** — cenas arbitrárias, separação fácil. O
  caso difícil, um rosto de boca fechada ou sem a língua para fora, não existe no
  conjunto e não foi testado. Colete esses negativos antes de abrir ao público.

## Estrutura

```
hality/
  features.py              34 features de cor, textura e setor
  data.py                  tabela mestra e divisão por paciente
  segmenter.py             U-Net
  train_segmenter.py       treina o segmentador
  train_classifier.py      treina o Modelo B e abre o teste trancado
  train_gate.py            treina o Modelo A
  pipeline.py              inferência ponta a ponta
  api.py                   FastAPI
models/                    artefatos treinados
docs/ARQUITETURA.md        contrato dos módulos, features, protocolo, resultados
docs/superpowers/specs/    especificação de projeto e o racional das decisões
Hality-Project-main/       implementação anterior (referência histórica)
data_ext/                  datasets externos baixados
.venv312/                  ambiente Python 3.12
```

Uma ressalva importante antes de ler qualquer número: **não sabemos qual protocolo
produziu a nota clínica** — se foi teste organoléptico, halímetro ou julgamento do
dentista. Isso muda o que o sistema pode afirmar e qual é o seu teto de desempenho.
Está detalhado na seção 1 de `docs/ARQUITETURA.md` e é a pendência número um do projeto.

## Ambiente

Python 3.12, PyTorch em CPU.

```bash
py -3.12 -m venv .venv312
.venv312/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv312/Scripts/python.exe -m pip install numpy pandas scikit-learn pillow opencv-python transformers matplotlib
```

Duas observações sobre o ambiente:

- **Python 3.14 não serve.** O ecossistema de ML ainda não tem suporte; o PyTorch
  instalado nele apresenta falha de segmentação no passo de treino.
- **A GPU AMD não é utilizável para treino.** Não há CUDA, e `torch-directml` não tem
  distribuição para Python 3.12 nem 3.14. Isso não é um problema prático: o modelo
  principal treina em segundos de CPU.

## Pendências

1. Obter o enunciado das perguntas Q1 a Q10 do questionário. A pergunta Q6 sozinha
   prevê boa parte do rótulo, e precisamos saber se é um sintoma relatado (feature
   legítima) ou uma avaliação profissional (vazamento do rótulo). Está fora do modelo
   até a confirmação.
2. Rotacionar a chave de API da Roboflow, que está exposta em texto claro em
   `Hality-Project-main/Segmentation.ipynb:58`.
3. Coletar mais dados. É a única alavanca que levanta o teto de desempenho.
