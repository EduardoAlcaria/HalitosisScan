# Hality — documentação técnica

Contrato dos módulos, protocolo de dados e resultados medidos.
Para a visão geral, ver `README.md`. Para o racional das decisões, ver
`docs/superpowers/specs/2026-08-28-hality-redesign-design.md`.

---

## 1. O rótulo: o que sabemos e o que não sabemos

Esta seção vem primeiro porque tudo depende dela.

### 1.1 O que está estabelecido

O alvo é a coluna `Classificação` de `Hality-Project-main/data/TabelaHality.csv`,
com valores 1, 2 e 3, distribuídos em 27 / 119 / 199 sobre 345 anamneses.

Está estabelecido que **o rótulo não é derivável do questionário**:

| Verificação | Resultado |
|---|---|
| Árvore de decisão sem limite de profundidade sobre `Q1`–`Q10`, medida no próprio treino | 81,7% |
| Padrões de resposta idênticos com nota diferente | 31 padrões |
| Linhas envolvidas nesses conflitos | 181 de 345 |

Uma árvore autorizada a memorizar o conjunto inteiro chega a 81,7%. Se o rótulo fosse
uma função do questionário, chegaria a 100%. Existe informação externa. Isso é
consistente com atribuição por um avaliador humano.

### 1.2 O que NÃO está estabelecido — lacuna bloqueante

**Não sabemos qual protocolo produziu essa nota.** As possibilidades, e o que cada uma
implicaria:

| Protocolo possível | Natureza | O que a nota 1–3 significaria | Confiabilidade |
|---|---|---|---|
| Teste organoléptico | Avaliador treinado cheira o hálito a distância padronizada | Escala de intensidade percebida | Subjetivo; exige concordância entre avaliadores para ser aferida |
| Halímetro | Medição de compostos sulfurados voláteis em ppb | Faixas de ppb convertidas em categorias | Objetivo; os pontos de corte precisam ser conhecidos |
| Julgamento clínico do dentista | Avaliação global, possivelmente incorporando exame bucal | Impressão profissional | Depende do protocolo interno da clínica |

Isso não é preciosismo documental. Três consequências diretas:

1. **Define o que se pode afirmar.** Um modelo treinado sobre medição de halímetro
   estima concentração de compostos sulfurados. Um modelo treinado sobre teste
   organoléptico estima a percepção de um avaliador. São afirmações diferentes e o
   texto do produto precisa corresponder à correta.

2. **Define o teto de desempenho.** Se o rótulo for organoléptico, ele carrega a
   variabilidade do avaliador. Nenhum modelo pode superar a confiabilidade do próprio
   rótulo — se dois avaliadores concordariam em 80% dos casos, 80% é o teto, não 100%.

3. **Define o significado dos limiares.** As classes 1, 2 e 3 correspondem a cortes em
   alguma escala. Sem saber quais, não dá para justificar por que colapsamos para
   `classe 3 vs resto` além do argumento estatístico.

**Ação necessária:** obter da clínica parceira o protocolo de avaliação, os pontos de
corte entre as três classes e, se houve mais de um avaliador, alguma medida de
concordância. Até lá, toda métrica neste documento deve ser lida como "concordância com
a nota da clínica", não como "acurácia diagnóstica".

### 1.3 O modelo é apenas de imagem — a anamnese foi removida

`Q6` correlaciona 0,528 com o rótulo — muito acima de todas as outras, que ficam entre
−0,01 e 0,15.

```
        classe 1   2    3
Q6=0        25   34   14
Q6=1         2   85  185
```

Se `Q6` for um sintoma relatado pelo paciente, é feature legítima. Se for do tipo "um
profissional já lhe disse que você tem mau hálito", é o rótulo reentrando disfarçado de
feature.

**Os enunciados de `Q1`–`Q10` não existem em nenhum arquivo do repositório e não estão
disponíveis.** Sem eles, a hipótese de vazamento não pode ser descartada, e `Q6` está
permanentemente fora.

Sem `Q6`, a contribuição da anamnese é nula dentro do ruído:

| Entrada | AUC |
|---|---|
| Imagem sozinha | 0,797 |
| Imagem + anamnese sem `Q6` | 0,799 |
| Anamnese sozinha sem `Q6` | 0,615 |

**Decisão: a anamnese sai inteira do modelo.** Ganho de 0,002 não justifica manter uma
superfície de vazamento que não pode ser auditada. O Modelo B passa a operar somente
sobre features de imagem. `predict()` perde o parâmetro `anamnese`.

Consequência secundária, favorável ao produto: o sistema deixa de depender de
questionário preenchido, e a entrada passa a ser apenas a fotografia.

---

## 2. Contrato dos módulos

Cada módulo é uma função pura ou um objeto com estado carregado uma vez na
inicialização. Nenhum conhece os vizinhos além da estrutura que troca.

### `normalize(raw: bytes) -> ndarray[H,W,3] uint8`

Decodifica JPEG, PNG, BMP e HEIC. Aplica a rotação indicada no EXIF. Reamostra
preservando a proporção original, com o lado maior limitado.

Falha com erro de requisição se o formato não for decodificável. É a única etapa que
levanta erro em vez de devolver rejeição.

### `image_quality(img) -> Ok | Reject(motivo)`

Lógica pura, sem modelo. Dois indicadores:

- **Nitidez:** variância do Laplaciano sobre o canal de luminância. Abaixo do limiar,
  a imagem está tremida ou fora de foco.
- **Exposição:** fração de pixels saturados no branco somada à fração colada no preto.
  Acima do limiar, a informação cromática foi perdida.

Os limiares são calibrados sobre a distribuição das 318 imagens próprias e expostos
como configuração.

Esta etapa existe porque nem o gate nem a sanidade da máscara medem qualidade. Uma foto
tremida cuja língua ainda seja segmentável passaria pelas duas e chegaria ao
classificador com a cor comprometida — e a cor é o sinal (ver 5.2).

### `tongue_gate(img) -> (bool, float)`

Modelo A. Classificador binário sobre a imagem inteira, antes de qualquer segmentação.

Não é derivado da confiança do segmentador. Um segmentador treinado apenas com línguas
nunca viu um negativo e produz confiança não calibrada fora da distribuição. Sem
controle sobre a captura, entrada arbitrária é o caso normal.

### `ood_check(embedding) -> Ok | Abstain(motivo)`

Distância ao conjunto de treino. Responde o que o gate não responde: não "há uma
língua", mas "esta imagem se parece com aquelas sobre as quais o modelo foi calibrado".

Motivado pela limitação da seção 6.3: o conjunto de treino cobre um único ambiente.
Uma foto doméstica pode conter uma língua perfeitamente válida e ainda assim estar
fora da distribuição em que as métricas valem.

### `segment(img) -> (mask: ndarray[H,W] bool, conf: float)`

Interface estável independentemente da implementação. Primeira opção é TongueSAM
(MIT, pesos pré-treinados, zero-shot). Alternativa é treinar sobre BioHit e TongueSet3
somados às máscaras boas do projeto.

Substitui a chamada à API da Roboflow do pipeline anterior, que implicava latência de
rede por requisição, custo por chamada e envio de imagem clínica a terceiro.

### `mask_sanity(mask, conf) -> Ok | Reject(motivo)`

Lógica pura. Verifica área mínima, conectividade e fragmentação.

**Não impõe limite superior de área.** A inspeção visual (seção 6.2) mostrou que
máscaras cobrindo 62–70% do quadro são close-ups legítimos e estão entre as melhores
imagens do conjunto. Um limite superior rejeitaria justamente essas.

### `extract_features(img, mask) -> ndarray[F] float`

Vetor de features nomeadas. Os nomes e a ordem fazem parte do contrato: o modelo
treinado depende deles, e alterar a extração invalida o artefato.

### `predict(features) -> float`

Modelo B. Devolve probabilidade calibrada. Carrega artefato versionado.

Opera somente sobre features de imagem. A anamnese foi removida por decisão da seção
1.3: sem `Q6` ela contribui 0,002 de AUC, e `Q6` não pode ser auditada porque os
enunciados do questionário não estão disponíveis.

### `decide(prob) -> Positivo | Negativo | Inconclusivo`

Lógica pura, limiares por configuração. A faixa central devolve inconclusivo.

---

## 3. Features do Modelo B

Todas calculadas **apenas sobre os pixels dentro da máscara**.

| Grupo | Quantidade | Descrição |
|---|---|---|
| RGB por canal | 12 | média, desvio, percentil 10, percentil 90 |
| HSV por canal | 12 | média, desvio, percentil 10, percentil 90 |
| Globais | 3 | saturação média, fração de área da máscara, mais uma reservada |
| Textura | 2 | média e desvio do gradiente vertical |
| Por setor | 6 | saturação e brilho na ponta, no meio e na base |

Importância por permutação, queda de AUC ao embaralhar cada feature, média de 5 folds:

```
HSV_S_p10        +0,0622
HSV_S_desvio     +0,0480
HSV_H_p10        +0,0204
area_mascara     +0,0183
RGB_R_media      +0,0165
RGB_R_p10        +0,0157
textura_desvio   +0,0136
```

`HSV_S_p10` — o percentil 10 da saturação — domina. Interpretação: quão pálida é a
porção mais pálida da língua. É a camada de saburra, medida de forma contínua.

**Uma feature explícita de saburra foi testada e descartada.** A contagem
`fração de pixels com saturação < 60 e brilho > 120` produz:

| Conjunto | AUC |
|---|---|
| Somente essa feature | 0,553 |
| Todas menos ela | 0,799 |
| Todas | 0,801 |

Ela não agrega nada. Binarizar com limiar fixo descarta a informação que o percentil
preserva. **Consequência de projeto:** medidas contínuas por percentil substituem
contagens limiarizadas em qualquer feature nova.

---

## 4. Protocolo de dados

### 4.1 Unidade de amostragem

O paciente, não a foto. Todas as imagens e todos os recortes derivados de um mesmo
paciente ficam do mesmo lado da divisão.

Hoje cada paciente tem uma foto, então isso não altera o Modelo B. Importa desde já no
Modelo A, onde uma foto produz um positivo e até seis negativos: se esses sete recortes
se espalhassem entre treino e teste, o modelo veria o mesmo lábio, a mesma luz e a
mesma pessoa dos dois lados, e o teste mediria memorização.

### 4.2 Divisão

Estratificada pelo rótulo binário, 70 / 15 / 15. Os identificadores de cada partição
são gravados em arquivo versionado e reutilizados por todos os experimentos. O conjunto
de teste não é consultado até a avaliação final.

### 4.3 Tabela mestra

Um passo único de reconciliação liga `ID_ANAMNESE`, caminho da imagem original, caminho
da imagem normalizada, rótulo e partição.

As pastas `data/Imagens lingua/`, `data/Classificacao/`, `data/classificacao_rotulados/`
e os três diretórios `dataset*` do drive são entrada bruta e não são consumidos
diretamente pelo treino. Os três `dataset*` contêm divisões distintas do mesmo material
sem registro de qual produziu qual modelo; são descartados.

### 4.4 Construção do conjunto do Modelo A

Positivos: recorte quadrado centrado na máscara, lado igual à maior dimensão da língua.
Origem: imagens próprias com máscara sadia, mais imagens de datasets públicos de língua.

Negativos, duas naturezas:

- **Hard negatives**, minerados das próprias fotos: recortes quadrados de mesmo lado
  que o positivo daquela foto, sem interseção com a máscara. Produzem lábio, dente,
  pele, queixo, dedo enluvado, gaze e fundo. Mesma câmera, mesma iluminação, mesmo
  indivíduo. São os negativos difíceis e os mais valiosos.
- **Negativos genéricos**, do COCO val2017: recortes quadrados aleatórios de fotos
  arbitrárias. Cobrem a captura acidental — bolso, chão, teto.

Duas restrições obrigatórias na mineração:

1. Fotos com fração de máscara abaixo de 0,15 são excluídas da mineração. Nelas a
   segmentação falhou por sub-segmentação, e "fora da máscara" ainda é língua — o
   negativo minerado teria rótulo invertido.
2. Geometria e resolução efetiva devem ser iguais entre as classes por construção.
   Ver 5.4.

---

## 5. Resultados medidos

Todos os números vêm de execuções neste repositório.

### 5.1 Desempenho do Modelo B

Validação cruzada estratificada repetida, 5 folds × 10 repetições, tarefa
`classe 3 vs resto`:

| Entrada | AUC | IC95 |
|---|---|---|
| Imagem | 0,797 | 0,779 – 0,820 |
| Imagem + anamnese sem `Q6` | 0,799 | 0,779 – 0,823 |
| Somente anamnese sem `Q6` | 0,615 | 0,584 – 0,647 |

Baseline majoritário: F1 macro 0,368.

Em três classes o desempenho colapsa: F1 macro 0,496 contra baseline 0,245, com a
classe 1 (22 amostras) recebendo 4 acertos. O recorte binário é o único sustentado.

Pontos de operação, prevalência de 58%:

| Sensibilidade | Especificidade | VPP | VPN |
|---|---|---|---|
| 0,95 | 0,44 | 0,70 | 0,87 |
| 0,90 | 0,60 | 0,76 | 0,81 |
| 0,85 | 0,65 | 0,77 | 0,75 |

Com abstenção descartando os 20% mais incertos: AUC 0,859 e acurácia 0,787 sobre os
80% cobertos.

### 5.2 Cor é o sinal; normalizá-la o destrói

| Correção de iluminação | AUC |
|---|---|
| Nenhuma | 0,786 |
| Gray-world | 0,683 |
| Referência tomada fora da língua | 0,700 |
| White-patch | 0,687 |

Augmentation de cor, duas cópias por imagem de treino, aplicada apenas às dobras de
treino:

| Augmentation | AUC |
|---|---|
| Nenhuma | 0,791 |
| Geométrica (espelhamento) | **0,799** |
| Jitter ±5% | 0,785 |
| Jitter ±10% | 0,779 |
| Jitter ±20% (valor do pipeline anterior) | 0,763 |
| Jitter ±35% | 0,764 |

As diferenças pequenas estão dentro do IC de ±0,02 e não são conclusivas isoladamente.
A monotonicidade nos cinco pontos é o que sustenta a regra: **augmentation geométrica
sim, de cor não.**

### 5.3 Capacidade de modelo não é o gargalo

`facebook/dinov2-small` como extrator congelado, 768 dimensões, com regressão logística:
**AUC 0,761**, contra 0,801 das features à mão.

Backbones de propósito geral são treinados para invariância cromática. Aqui a cor é o
sinal — toda língua tem forma de língua. Consequência: uma CNN só entra no pipeline se
superar 0,797 em conjunto de teste isolado.

### 5.4 Confundidores encontrados

O teste: prever o rótulo usando apenas metadados de arquivo ou geometria de recorte,
sem nenhum pixel.

| Conjunto | AUC só com metadados | Situação |
|---|---|---|
| Conjunto próprio, `classe 3 vs resto` | 0,498 | Limpo |
| Dataset chinês `染苔/非染苔` | 0,995 | Descartado como rótulo |
| Primeira montagem do conjunto do Modelo A | 0,9991 | Refeita |

**Dataset chinês:** mediana de 20,67 MP numa classe contra 0,20 MP na outra. As classes
vieram de equipamentos distintos. As 2008 imagens seguem úteis como positivos do gate —
são línguas reais — mas o rótulo é inutilizável.

**Primeira montagem do Modelo A:** produziu AUC 1,0000 com zero erro em 6.334 amostras.

```
             aspecto (L/A)   desvio   lado menor (mediana)
proprio          0,87         0,27          660 px
chines           0,97         0,16          717 px
hardneg          1,00         0,00          167 px
coco             1,33         0,37          427 px
```

Hard negatives eram quadrados exatos e pequenos; positivos, retângulos grandes; COCO
entrava como imagem inteira em 4:3. As classes eram separáveis pela geometria.

Correção: recorte quadrado em todos os grupos; negativo com o mesmo lado do positivo da
mesma foto; COCO reduzido a recorte quadrado; e resolução efetiva comum aplicada a
todos os recortes antes do modelo.

**Regra:** resultado próximo da separação perfeita é motivo de suspeita, não de
aceitação. O teste vale para conjuntos construídos internamente, não só para os
externos.

---

## 6. Limites conhecidos

### 6.1 Tamanho da amostra

318 imagens utilizáveis. Um conjunto de teste de 15% tem 48 exemplos, o que equivale a
cerca de ±7 pontos de ruído em acurácia. Métrica de ponto único não é informativa nesse
tamanho.

Protocolo adotado: validação cruzada repetida com intervalo de confiança para toda
decisão de modelagem, e um conjunto de teste trancado aberto uma única vez ao final.

Não existe dataset público com rótulo de halitose. A busca foi feita e é conclusiva.
Datasets públicos de língua servem ao gate e ao segmentador, nunca ao Modelo B. Coleta
clínica adicional é a única alavanca que levanta o teto.

### 6.2 Qualidade da segmentação existente

Inspeção visual de 36 imagens amostradas nos extremos e na mediana da fração de área:

| Faixa | Situação |
|---|---|
| Mediana ≈ 0,42 | Máscaras corretas, com buracos internos |
| Alta, 0,62–0,70 | Máscaras corretas; close-ups legítimos, não falhas |
| Baixa, 0,00–0,10 | Falhas reais, ~12 imagens (4%); sub-segmentação |

Nas imagens de área baixa a língua está nítida e centralizada, mas a máscara capturou
apenas um fragmento, tipicamente sobre a mancha de saburra. A falha é de um lado só.

### 6.3 Cobertura de um único ambiente

A variação interna do conjunto é grande e neutra em relação ao rótulo:

```
amplitude:  0,12 a 12,04 MP  |  brilho médio 75 a 174  |  bmp, jpg, heic, png

               n     MP mediana   brilho médio   desvio do brilho
classe 1+2   134        0,75         123,7           16,6
classe 3     186        0,92         121,1           16,9
```

Mas a variação é de aparelho, não de cenário. Inspeção visual mostra dedos enluvados,
gaze e contexto de consultório. Nenhuma imagem representa captura doméstica.

**O teste de metadados não detecta isso** — ele compara classes dentro do conjunto, não
o conjunto contra o mundo. A queda de desempenho em cenário doméstico não é estimável a
partir dos dados existentes.

### 6.4 Prevalência

Os valores preditivos da seção 5.1 valem para prevalência de 58%. A 20% de prevalência,
no ponto de sensibilidade 0,90, o VPP cai para aproximadamente 0,36.

É aritmética de triagem e nenhum ganho realista de AUC a contorna.

### 6.5 Métrica e cobertura são inseparáveis

O Modelo B é treinado e avaliado sobre imagens que passaram pelos filtros — a mesma
distribuição de produção. Isso é correto e cria um risco de leitura: quanto mais casos
difíceis as etapas anteriores rejeitam, melhor o classificador parece.

Toda métrica reportada vem acompanhada da cobertura que a produziu, na mesma frase. O
relatório final traz a curva de métrica contra cobertura, não um ponto isolado.

---

## 7. Ambiente

Python 3.12, PyTorch em CPU. Ver `README.md` para instalação.

**Python 3.14 não é utilizável.** O PyTorch instalado nele apresenta falha de
segmentação no passo de backward, e `torchvision` não tem distribuição.

**A GPU AMD não é utilizável para treino.** Não há CUDA. `torch-directml` não tem
distribuição para Python 3.12 nem 3.14; exigiria Python 3.10 ou 3.11 e rebaixar o
restante da pilha. Sem impacto prático: o Modelo B treina em segundos de CPU e a
extração de embeddings sobre milhares de imagens leva minutos. Caso o treino de um
segmentador próprio se prove caro, a saída é ambiente de nuvem, não infraestrutura
local.

---

## 8. Problemas do pipeline anterior

Registrados para que não se repitam.

| # | Problema | Localização |
|---|---|---|
| 1 | Sem conjunto de teste; early stopping seleciona e reporta no mesmo conjunto de validação | `modeling_efnet.ipynb` |
| 2 | `ColorJitter(brightness=0.2, contrast=0.2)`, que custa 0,036 de AUC | `modeling_efnet.ipynb` |
| 3 | `Resize((224,224))` sem preservar proporção | `modeling_efnet.ipynb` |
| 4 | Três classes, com 27 amostras na minoritária | todo o pipeline |
| 5 | `model.predict(tmp_png, confidence=0)` aceita qualquer predição | `Segmentation.ipynb` |
| 6 | Chave de API da Roboflow em texto claro | `Segmentation.ipynb:58` |
| 7 | Dependência de API externa no caminho de inferência | `Segmentation.ipynb` |
| 8 | `stats.binom_test`, removido no SciPy 1.12 | `modeling_efnet.ipynb` |

O item 6 exige rotação da credencial, independentemente do cronograma do redesenho.

---

## 9. Pendências

| # | Pendência | Bloqueia |
|---|---|---|
| 1 | Protocolo de medição do rótulo (seção 1.2) | A afirmação que o produto pode fazer |
| 2 | Rotação da chave da Roboflow | Nada; é urgente por si |
| 3 | Verificar TCM-Tongue com o teste de metadados | Uso do dataset externo |
| 4 | Correção manual das ~12 máscaras sub-segmentadas | Recuperação de 4% do conjunto |

O enunciado de `Q1`–`Q10` deixou de ser pendência: os enunciados não estão disponíveis,
e a consequência já foi absorvida no desenho — a anamnese saiu do modelo (seção 1.3).

**A pendência 1 é a única que muda o que o produto pode afirmar.** É uma pergunta à
clínica parceira, com cinco itens:

1. O método usado — teste organoléptico, halímetro, ou julgamento do dentista.
2. Se halímetro: qual aparelho, e os pontos de corte em ppb entre as classes 1, 2 e 3.
3. Se organoléptico: qual escala, a que distância, sob que protocolo.
4. Quantos avaliadores participaram.
5. Se algum paciente foi avaliado por mais de um avaliador — esse dado fornece a
   concordância entre avaliadores, que é o teto de desempenho alcançável por qualquer
   modelo treinado sobre esse rótulo.
