# Hality

Triagem de indício de halitose a partir de uma foto da língua.

Este repositório contém o redesenho do projeto: pipeline, modelos, API e imagem Docker.
A implementação anterior (`Hality-Project-main/`) fica fora do git, como referência local,
porque contém dados de pacientes.

## O que o projeto faz

Uma pessoa fotografa a própria língua. O sistema devolve uma de quatro respostas:

- **indício de halitose**, com uma probabilidade
- **sem indício**, com uma probabilidade
- **inconclusivo**, quando não há base suficiente para responder
- **rejeitado**, quando a foto não serve, sempre com o motivo ("foto tremida", "não
  identifiquei uma língua")

Junto com a resposta vai sempre a orientação de procurar avaliação odontológica. O sistema
**não diagnostica**, ele faz triagem: sinaliza quem vale a pena um profissional examinar.

## Por que uma foto da língua diz alguma coisa

Uma língua com muita saburra, a camada esbranquiçada que se acumula no dorso, é mais clara
e menos saturada que uma língua limpa. Essa diferença aparece na foto e tem ligação clínica
conhecida com halitose: a saburra abriga as bactérias que produzem os compostos sulfurados
voláteis (VSC) responsáveis pelo odor.

A característica que mais pesa no modelo é o percentil 10 da saturação dentro da língua, ou
seja, **quão pálida é a porção mais pálida**. É a camada branca, medida de forma contínua.

A foto sozinha prevê melhor que o questionário clínico inteiro:

| Entrada | AUC |
|---|---|
| Foto da língua | 0,77 |
| Questionário completo (sem a pergunta que vaza o rótulo) | 0,615 |

## Os dados

- 345 anamneses com nota clínica de 1 a 3, atribuída por avaliador humano
- 321 fotos casadas com essas anamneses; 306 utilizáveis depois do filtro de máscara
- distribuição das 306: nota 1 = 22, nota 2 = 107, nota 3 = 177
- origem: clínica parceira na PUCRS

O alvo é binário: nota 3 contra o resto. Três classes não funcionam, porque a nota 1 tem só
22 fotos. Dos falsos positivos atuais, a maioria vem da nota 2 (moderado), não da nota 1:
o modelo erra na fronteira entre moderado e grave, não entre limpo e grave.

O conjunto é pequeno, e esse é o teto do projeto. A curva de aprendizado ainda sobe com 100%
dos dados (AUC 0,761 com 208 fotos de treino, 0,773 com 244), então mais fotos rotuladas
ajudam, com retorno decrescente. Não existe dataset público com rótulo de halitose.

## Como funciona

São três modelos, de propósito:

| Modelo | Pergunta | Treinado com |
|---|---|---|
| Gate | "isto é uma língua?" | 2.331 fotos de língua contra 5.000 fotos do COCO |
| Segmentador (TongueNet) | "onde exatamente está a língua?" | fotos da clínica |
| Classificador | "esta língua indica halitose?" | as 306 fotos com nota clínica |

Só o classificador depende das 306 notas. Os outros dois têm muito mais dado disponível.

O caminho de uma foto até a resposta:

```
foto
 |
 |- 1. normalização      decodifica (JPEG, PNG, BMP, HEIC), corrige rotação, redimensiona
 |
 |  as etapas 2 a 7 rodam 5 vezes, sobre variações geométricas da mesma foto
 |- 2. qualidade          muito escura? muito clara? tremida?        -> rejeita
 |- 3. gate               tem uma língua aí?                          -> rejeita
 |- 4. segmentação        TongueNet recorta a língua
 |- 5. sanidade           a máscara tem área e forma plausíveis?      -> rejeita
 |- 6. características    34 medidas de cor e textura, só dentro da língua
 |- 7. classificador      probabilidade calibrada
 |
 |- 8. decisão            voto de maioria entre as 5 passagens
                          sem maioria clara ou probabilidade perto do limiar -> inconclusivo
```

As cinco passagens existem porque a mesma língua, fotografada duas vezes, nem sempre recebia
o mesmo veredito. O voto de maioria reduziu essa instabilidade pela metade.

Rejeitar é resposta bem-sucedida, não erro. A API devolve HTTP 200 com o motivo.

## Resultados medidos

**Segmentação**

| Segmentador | IoU contra máscara de especialista (BioHit, n=30) | IoU contra máscara Roboflow (nossa validação, n=45) |
|---|---|---|
| TongueNet (DeepLabV3-ResNet50) | **0,903** | 0,823 |
| U-Net própria (reserva) | 0,729 | 0,843 |

A máscara Roboflow tem defeitos conhecidos, por isso a coluna do especialista é a régua
honesta. Nas 15 fotos em que a segmentação antiga só encontrava fragmentos, o TongueNet
recupera a língua inteira em 14.

**Classificador, pipeline completo** (validação + teste, n=93, fotos nunca vistas no treino
do classificador)

| Métrica | Valor |
|---|---|
| Cobertura (fotos com veredito definitivo) | 72,0% |
| Acurácia | 0,776 |
| F1 macro | 0,759 |
| Sensibilidade | 0,875 |
| Especificidade | 0,630 |
| Precisão (dos sinalizados, quantos tinham) | 0,778 |
| AUC | 0,748 |

Referência: chutar sempre a classe majoritária acerta 0,597 e tem F1 macro 0,374.

**AUC por validação cruzada repetida** (306 fotos, 5 folds × 20 repetições): **0,77**,
intervalo 0,68 a 0,86. É a estimativa mais estável, e o número a citar.

**Gate:** recall de 97,8% nas fotos próprias, 0,0% de falso-aceite em fotos do COCO.

**Constância:** entre refotos da mesma língua, o desvio da probabilidade é 0,040. Com o
segmentador anterior, 25,7% das refotos mudavam de veredito; com o TongueNet a porcentagem
ainda não foi medida.

**Latência:** cerca de 3 s por foto aceita em CPU local e 5,6 s dentro do Docker. Foto
rejeitada leva menos de 1 s.

**Ressalva importante sobre esses números:** o TongueNet foi treinado com as fotos da
clínica, inclusive as de validação e teste do classificador. Os números do pipeline estão
otimistas até uma reavaliação com o TongueNet retreinado sem essas fotos.

## O que aprendemos medindo

**Modelo maior não ajuda no classificador.** O DINOv2 (backbone pré-treinado grande) ficou em
AUC 0,761 contra 0,801 das características de cor feitas à mão. Esses modelos são treinados
para ignorar cor, e aqui a cor é o sinal.

**Mexer na cor piora.** Normalizar iluminação derrubou o AUC de 0,786 para cerca de 0,69.
Jitter de cor piorou em todas as intensidades testadas (±5% a ±35%). Aumentar a saburra de
fotos nota 2 e rotulá-las como nota 3 derrubou o AUC em 0,055. Variação geométrica, por outro
lado, é segura.

**Também testados, sem ganho:** segmentar a saburra explicitamente (duas implementações),
modelo ordinal em vez de binário, restrição monotônica, aumento contrafactual que preserva o
rótulo.

**Confundidores.** Quando as classes de um dataset foram coletadas de formas diferentes, o
modelo aprende a forma de coletar em vez da condição. O teste: prever o rótulo usando só os
metadados do arquivo (largura, altura, bytes, formato). Nosso conjunto passa (AUC 0,498, acaso
puro). Um dataset externo de língua falhou (0,995) e foi descartado como fonte de rótulo.

## Limitações

**O protocolo do rótulo é desconhecido.** Não sabemos se a nota 1 a 3 veio de halímetro,
teste organoléptico ou impressão clínica. Se o avaliador olhou a língua para dar a nota, parte
do resultado é circular. Isso define o que o sistema pode afirmar.

**Todas as fotos vêm da mesma clínica.** Selfie caseira, com luz de teto, não está
representada. Não há como estimar a queda de desempenho nesse cenário.

**Prevalência.** Na população da clínica (58% de casos graves), 78% dos sinalizados realmente
têm a condição. Numa população aberta com 20% de prevalência, esse número cai para cerca de
37%. É aritmética de triagem, por isso a resposta é "procure um dentista" e não um veredito.

**O gate nunca viu boca fechada.** Os negativos dele são fotos do COCO, separação fácil.

**Métrica sem cobertura é meia informação.** O classificador só responde 72% das fotos.
Quanto mais o sistema rejeita, melhor ele parece. Todo número aqui vem com a cobertura.

## Rodar com Docker

A imagem `hality-ai` vem com os modelos treinados e o DINOv2 embutidos, e roda sem internet.
Ela é distribuída como asset do release, fora do git, porque os pesos passam do limite de
100 MB por arquivo do GitHub e foram treinados com dados de pacientes. Não redistribua fora do
grupo.

Baixar e carregar:

```bash
gh release download v0.2.0 --repo EduardoAlcaria/HalitosisScan --pattern "hality-ai.tar.gz"
docker load -i hality-ai.tar.gz
```

Subir o serviço na porta 8000:

```bash
docker run -d --name hality-ai -p 8000:8000 --restart unless-stopped hality-ai:latest
```

A subida leva alguns segundos, porque os modelos são carregados e aquecidos. Conferir:

```bash
curl http://127.0.0.1:8000/saude
curl -F "foto=@lingua.jpg" http://127.0.0.1:8000/analisar
```

Parar e remover:

```bash
docker rm -f hality-ai
```

Construir a imagem a partir do código (exige os pesos, que estão no asset
`hality-models.tar.gz` do mesmo release):

```bash
gh release download v0.2.0 --repo EduardoAlcaria/HalitosisScan --pattern "hality-models.tar.gz"
tar -xzf hality-models.tar.gz
docker build -t hality-ai:latest .
docker save hality-ai:latest | gzip > hality-ai.tar.gz
```

## API

| Rota | O que faz |
|---|---|
| `POST /analisar` | recebe a foto (multipart, campo `foto`, até 20 MB) e devolve o veredito |
| `GET /saude` | informa o segmentador carregado, o limiar e a faixa de abstenção |
| `GET /docs` | documentação interativa (Swagger) |

Exemplo de resposta:

```json
{
  "veredito": "indicio",
  "motivo": "Indicio compativel com halitose. Procure um dentista para avaliacao. Isto nao e um diagnostico.",
  "probabilidade": 0.8995,
  "area_lingua": 0.4512,
  "nitidez": 210.4,
  "confianca_lingua": 0.9994,
  "dispersao": 0.0049,
  "passagens": 5,
  "arquivo": "lingua.jpg",
  "tempo_ms": 3120,
  "aviso": "Triagem, nao diagnostico. Procure um dentista para avaliacao."
}
```

- `veredito`: `indicio`, `sem_indicio`, `inconclusivo` ou `rejeitado`
- `probabilidade`: média das 5 passagens; `null` quando rejeitado
- `dispersao`: desvio da probabilidade entre as passagens; alto significa foto instável
- rejeição volta como HTTP 200; erro HTTP só para arquivo vazio, acima de 20 MB ou modelo não
  carregado

## Rodar sem Docker

Python 3.12 (o 3.14 não serve: o PyTorch falha no treino).

```bash
py -3.12 -m venv .venv312
.venv312/Scripts/python.exe -m pip install -r requirements.txt
tar -xzf hality-models.tar.gz      # pesos em models/
.venv312/Scripts/python.exe -m uvicorn hality.api:app --port 8000
```

Sem `models/tonguenet.pt`, o sistema cai para a U-Net de reserva (`models/segmentador.pt`).

Retreinar (precisa das fotos da clínica em `Hality-Project-main/`, fora do git):

```bash
.venv312/Scripts/python.exe -m hality.train_classifier   # ~3 min
.venv312/Scripts/python.exe -m hality.train_gate         # ~10 min
```

Auto-testes: `python -m hality.features`, `hality.data`, `hality.pipeline`,
`hality.segmentacao`.

A GPU AMD não serve para treinar com PyTorch (não há CUDA, e `torch-directml` não suporta
Python 3.12). Ela serve para inferência via ONNX Runtime + DirectML: o encoder do SAM roda 35
vezes mais rápido nela. O pipeline principal roda em CPU.

## Estrutura

```
hality/
  pipeline.py              inferência ponta a ponta, voto de 5 passagens
  api.py                   FastAPI
  segmentacao.py           TongueNet, com U-Net de reserva
  features.py              34 características de cor, textura e setor
  data.py                  tabela mestra e divisão por paciente
  train_classifier.py      treina o classificador
  train_gate.py            treina o gate de língua
  segmenter.py             U-Net de reserva
  train_segmenter.py       treina a U-Net
  coating.py, raios.py     experimentos sem ganho, mantidos como registro
  sam_segmenter.py         SAM com gerador de pontos (experimental)
  sam_onnx.py              encoder do SAM em ONNX na GPU (experimental)
  train_deeplab.py         experimento de segmentador alternativo
  train_segformer.py       experimento de segmentador alternativo
Dockerfile                 imagem da API com os modelos
requirements.txt           dependências da API
docs/DECISOES.md           cada decisão de projeto com a medição que a sustenta
docs/ARQUITETURA.md        contrato dos módulos e protocolo de dados
docs/SISTEMA.md            visão geral do sistema
```

## Pendências

1. Descobrir com a clínica o protocolo da nota 1 a 3, e se o avaliador via a língua.
2. Retreinar o TongueNet sem as fotos de validação e teste e remedir tudo.
3. Medir a constância com o TongueNet.
4. Coletar negativos difíceis para o gate (boca fechada, língua não exposta).
5. Coletar mais fotos rotuladas, de preferência fora da clínica.
6. Revogar a chave antiga da Roboflow, que ficou exposta no notebook do projeto anterior.
