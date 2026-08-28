# Hality — Redesign do pipeline de triagem de halitose por foto da língua

**Data:** 2026-08-28
**Status:** Aprovado para planejamento
**Escopo:** modelos e pipeline de inferência. Não há aplicativo neste escopo.

## 1. Contexto

O Hality estima indício de halitose a partir de uma foto da língua. Existe uma implementação anterior (`Hality-Project-main/`) com notebooks de exploração, segmentação via API remota e um classificador EfficientNet-B0, que será substituída.

As imagens vêm de uma clínica parceira na PUCRS. O uso é de pesquisa.

Este documento registra o diagnóstico do estado atual, as medições que sustentam as decisões, e o desenho do novo pipeline. Todo número aqui foi medido neste repositório, não estimado.

## 2. Ativos existentes

| Ativo | Conteúdo |
|---|---|
| `data/TabelaHality.csv` | 345 anamneses: `ID_ANAMNESE`, `IDADE`, `Q1`–`Q10`, `Classificação` (1–3) |
| `data/TabelaHality_Clean.csv` | Mesma tabela com categóricos mapeados para inteiros |
| `data/Classificacao/` | 323 fotos nomeadas `S<id>`, casáveis com a anamnese |
| `data/Imagens lingua/1..4/` | 306 fotos cruas de celular, sem nomeação padronizada |
| `data/classificacao_rotulados/1,2,3/` | 322 imagens organizadas pela nota (23 / 112 / 187) |
| `recortadas/recortadas/` | 322 tríades `_cut.png`, `_mask.png`, `_coords.txt` |
| `models/*.pth` | Dois checkpoints EfficientNet-B0, procedência de dados incerta |
| `drive-download.../中医舌诊染苔数据/` | 2008 imagens chinesas. **Rótulo inutilizável** — ver 4.5 |
| `drive-download.../dataset*` | Três splits do mesmo material, sem registro de qual gerou qual modelo |

## 3. Natureza do rótulo

`Classificação` não é derivável do questionário. Uma árvore de decisão sem limite de profundidade, autorizada a memorizar o dataset inteiro, atinge apenas 81,7% usando `Q1`–`Q10`. Existem 31 padrões de resposta idênticos com nota diferente, envolvendo 181 das 345 linhas.

O rótulo carrega informação externa ao questionário, consistente com atribuição por avaliador humano. É o alvo supervisionado do projeto.

`Q6` é o preditor isolado mais forte (correlação 0,528):

```
        classe 1   2    3
Q6=0        25   34   14
Q6=1         2   85  185
```

Sem `Q6`, o questionário inteiro vale AUC 0,615 — quase acaso. Isso levanta a suspeita de que `Q6` seja uma pergunta do tipo "um profissional já lhe disse que você tem mau hálito", o que a tornaria o rótulo entrando disfarçado de feature.

**Decisão:** `Q6` fica fora do conjunto de features até que o enunciado seja confirmado. Obter o enunciado de `Q1`–`Q10` é item pendente.

## 4. Medições

### 4.1 O sinal da imagem é real

35 features de cor, textura e setor extraídas sob a máscara, com `HistGradientBoostingClassifier`, validação cruzada estratificada repetida 5×10:

| Entrada | AUC | IC95 |
|---|---|---|
| Imagem | 0,797 | 0,779 – 0,820 |
| Imagem + anamnese sem `Q6` | 0,799 | 0,779 – 0,823 |
| Somente anamnese sem `Q6` | 0,615 | 0,584 – 0,647 |

A imagem não complementa a anamnese; ela a supera com folga. Este é o resultado que sustenta a existência de um produto foto-first.

Em três classes o desempenho colapsa (F1 macro 0,496 contra baseline 0,245; a classe 1 tem 22 amostras e o modelo acerta 4). O recorte binário é o único sustentado pelos dados.

### 4.2 Não é confundidor de câmera

Prever `classe 3 vs resto` usando apenas metadados de arquivo (formato, largura, altura, bytes), sem ler um pixel: **AUC 0,498**. Acaso puro. A proporção de classe 3 é 56,7% em `.bmp` e 60,8% em `.jpg`.

Se a sessão de captura fosse a pista real, teria vazado neste teste. Não vazou. Ressalva: metadado é proxy imperfeito de sessão — duas sessões com o mesmo equipamento não seriam detectadas.

### 4.3 Modelo maior perde

`facebook/dinov2-small` como extrator congelado (768-d) com regressão logística: **AUC 0,761**, contra 0,801 das 35 features à mão.

Backbones de propósito geral são treinados para invariância a cor. Aqui a cor é o sinal — toda língua tem forma de língua; o que separa é a cor e a textura fina da saburra. O gargalo é qualidade do sinal cromático, não capacidade de modelo.

### 4.4 Normalizar iluminação destrói o sinal

| Correção | AUC |
|---|---|
| Nenhuma | 0,786 |
| Gray-world | 0,683 |
| Referência fora da língua | 0,700 |
| White-patch | 0,687 |

Combinado com 4.2, a leitura é que o brilho e a cor absolutos carregam informação clínica genuína — uma língua muito saburrosa é literalmente mais clara e menos saturada.

**Consequência direta:** augmentation de cor degrada o modelo. Medido diretamente, com duas cópias aumentadas por imagem de treino e augmentation aplicada apenas às dobras de treino:

| Augmentation | AUC |
|---|---|
| Nenhuma | 0,791 |
| Geométrica apenas (espelhamento) | **0,799** |
| Jitter de cor ±5% | 0,785 |
| Jitter de cor ±10% | 0,779 |
| Jitter de cor ±20% (o de `modeling_efnet.ipynb`) | 0,763 |
| Jitter de cor ±35% | 0,764 |

As diferenças pequenas estão dentro do ruído — 0,785 contra 0,791 não é conclusivo isoladamente, dado o IC de ±0,02. O que sustenta a conclusão é a monotonicidade: cada aumento de intensidade piora, sem exceção, e as pontas estão claramente fora do ruído. O `ColorJitter(brightness=0.2, contrast=0.2)` do pipeline anterior custa 0,036 de AUC.

**Regra:** augmentation geométrica (espelhamento horizontal, rotação pequena) sim; augmentation de cor não. Variação de captura precisa ser coletada, não simulada — simular variação cromática destrói o mesmo canal que carrega o rótulo.

### 4.4b Onde o sinal mora: a camada branca, medida por percentil

Importância por permutação (queda de AUC ao embaralhar cada feature), média de 5 folds:

```
HSV_S_p10        +0.0622
HSV_S_desvio     +0.0480
HSV_H_p10        +0.0204
area_mascara     +0.0183
RGB_R_media      +0.0165
```

Ablação da feature de saburra construída à mão (`fração de pixels com saturação < 60 e brilho > 120`):

| Conjunto de features | AUC |
|---|---|
| Somente a feature de saburra | 0,553 |
| Tudo menos ela | 0,799 |
| Tudo | 0,801 |

A feature explícita de saburra é inútil e pode ser removida sem custo. O que domina é `HSV_S_p10` — o percentil 10 da saturação, isto é, quão pálida é a porção mais pálida da língua.

Isso *é* a camada branca. A hipótese clínica está correta; a forma de medir é que estava errada. Binarizar com limiar fixo descarta a informação que o percentil preserva de forma contínua.

**Consequência para o desenho das features:** medidas contínuas baseadas em percentil, calculadas por setor, substituem contagens limiarizadas. E abre-se um caminho de melhoria com fundamento — segmentar a saburra dentro da língua, obtendo área, distribuição e espessura reais em vez do proxy que `HSV_S_p10` aproxima por acidente. Fica como evolução do Modelo B, condicionada a superar 0,797 em teste isolado.

### 4.4c O dataset varia em câmera, não em ambiente

A variação interna do conjunto é grande e, o que importa mais, é neutra em relação ao rótulo:

```
amplitude no conjunto inteiro:  0,12 a 12,04 MP  |  brilho médio 75 a 174  |  bmp, jpg, heic, png

                 n     MP mediana   brilho médio   desvio do brilho
classe 1+2     134        0,75         123,7            16,6
classe 3       186        0,92         121,1            16,9
```

É isso que produz AUC 0,498 no teste de metadados: não ausência de variação, mas variação distribuída por igual entre as classes. Um conjunto perfeitamente uniforme também passaria no teste e ainda assim seria pior, porque o modelo nunca teria visto uma câmera diferente.

**O limite que o teste de metadados não alcança.** Ele compara classes dentro do conjunto. Não detecta que o conjunto inteiro provém de um único ambiente. E provém: a inspeção visual mostra dedos enluvados, gaze e contexto de consultório. A variação disponível é de aparelho, não de cenário — iluminação de clínica, distância padronizada, profissional posicionando a língua.

Nenhuma imagem representa o cenário doméstico: luz amarela de teto, sem apoio, à noite. Não há como estimar a queda de desempenho nesse cenário a partir dos dados existentes, porque não existe nenhuma amostra rotulada dele. Qualquer número seria invenção.

**Consequência para o desenho:**

1. O primeiro uso deve ficar restrito ao ambiente de origem, onde a distribuição corresponde à medida.
2. O pipeline precisa de detecção de fora-de-distribuição além do gate. O gate responde "há uma língua"; falta responder "esta imagem se parece com as do treino". Distância dos embeddings ao conjunto de treino, com abstenção acima de um limiar, cobre esse caso.
3. Coleta no ambiente de destino é o único conserto real, e cada imagem nova acompanhada de avaliação profissional vale mais que qualquer melhoria de arquitetura.
4. Ao chegar o primeiro lote de imagens domésticas, recalibrar o limiar de operação antes de retreinar. O ponto de corte degrada antes do modelo.

### 4.5 O dataset chinês é inutilizável como rótulo

| Classe | Megapixels medianos | Resolução dominante |
|---|---|---|
| Não-tingida | 20,67 | 5568×3712 (614 imagens) |
| Tingida | 0,20 | 365×365 (44 imagens) |

Prever a classe apenas com metadados: **AUC 0,995**. As duas classes vêm de equipamentos diferentes. Qualquer modelo treinado nesse rótulo aprende o equipamento e reporta acurácia alta enquanto não aprende nada sobre línguas.

As 2008 imagens permanecem úteis como positivos do gate — são línguas reais, independentemente do rótulo. O rótulo `染苔/非染苔` é descartado.

**Regra de processo derivada:** todo dataset passa pelo teste de metadados antes de ser incorporado. Custa segundos e evita construir sobre um confundidor.

### 4.5b A regra vale para os conjuntos que nós mesmos construímos

A primeira montagem do conjunto de treino do Modelo A produziu AUC 1,0000, com zero erro em 6.334 amostras. O resultado era artefato de construção:

```
                aspecto (L/A)   desvio    lado menor (mediana)
proprio             0,87         0,27           660 px
chines              0,97         0,16           717 px
hardneg             1,00         0,00           167 px
coco                1,33         0,37           427 px

prever língua/não-língua apenas com largura, altura e proporção:  AUC = 0,9991
```

Os hard negatives eram quadrados exatos por construção, e pequenos; os positivos eram retângulos grandes; o COCO entrava como foto inteira em 4:3. As classes eram separáveis pela geometria do recorte, sem olhar um pixel.

**Correção adotada na mineração:** recorte quadrado em todos os grupos; negativo com o mesmo lado do positivo extraído da mesma foto; COCO reduzido a recorte quadrado aleatório em vez da imagem inteira; e todos os recortes passam por uma resolução efetiva comum antes de chegar ao modelo, de modo que a quantidade de detalhe disponível deixe de ser sinal.

**Regra:** o teste de confundidor é obrigatório também sobre conjuntos construídos internamente, e a mineração de negativos precisa igualar geometria e resolução entre as classes por construção. Um resultado próximo da separação perfeita é motivo de suspeita, não de aceitação.

### 4.6 Qualidade da segmentação existente

Inspeção visual de 36 imagens, amostradas nos extremos e na mediana da fração de área da máscara:

- **Mediana (≈0,42):** máscaras corretas, com buracos internos — precisamente o que `Mask_hull+fill.ipynb` corrige.
- **Área alta (0,62–0,70):** máscaras corretas. São close-ups legítimos onde a língua ocupa dois terços do quadro. **Não são falhas.**
- **Área baixa (0,00–0,10):** falhas reais, cerca de 12 imagens (4%). A língua está nítida e centralizada em todas; a máscara capturou apenas um fragmento, tipicamente sobre a mancha de saburra. Sub-segmentação, não objeto errado.

A falha é de um lado só. Um limiar superior de área rejeitaria justamente as melhores fotos do conjunto.

### 4.7 Ambiente

- GPU disponível: AMD Radeon RX 9070 XT. Não há CUDA. `torch-directml` não tem distribuição para Python 3.12 nem 3.14; exigiria Python 3.10/3.11 e rebaixar todo o resto.
- O ambiente Python 3.14 original apresenta *segmentation fault* no passo de backward do PyTorch.
- Ambiente de trabalho: venv em `.venv312/` com Python 3.12, torch 2.13.0+cpu, torchvision 0.28.0+cpu, scikit-learn 1.9.0.
- **Treino em CPU.** O modelo vencedor treina em segundos; a extração de embeddings sobre centenas de imagens roda em menos de um minuto. A GPU não é gargalo deste projeto. Se o treino de um segmentador próprio se provar caro, a saída é Colab, não infraestrutura local.

## 5. Problemas do pipeline anterior

1. **Sem conjunto de teste.** `modeling_efnet.ipynb` divide apenas em treino e validação; o early stopping seleciona pesos pela acurácia de validação e o relatório final é calculado sobre o mesmo conjunto. A métrica é otimista por construção.
2. **Augmentation de cor**, que a medição 4.4 mostra ser destrutiva.
3. **`Resize((224,224))`** sem preservar proporção, distorcendo forma de modo desigual entre imagens.
4. **Classe minoritária inviável** — 27 amostras na classe 1.
5. **Segmentação sem controle** — `model.predict(tmp_png, confidence=0)` aceita qualquer predição.
6. **Credencial exposta** — chave de API da Roboflow em texto claro em `Segmentation.ipynb:58`.
7. **Dependência externa na inferência** — latência de rede, custo por requisição, e envio de imagem clínica a terceiro.
8. **Código morto** — `stats.binom_test` foi removido do SciPy 1.12.

## 6. Decisões de design

### 6.1 Saída binária com probabilidade calibrada

O corte é `classe 3 vs resto`: o caso de maior consequência e o de melhor separabilidade. Três níveis não são sustentados pelos dados.

### 6.2 Dois modelos treinados, mais um segmentador

**Modelo A — gate de língua.** Classificador binário treinado com negativos reais, operando sobre a imagem inteira antes de qualquer outra etapa.

A alternativa de derivar o gate de um limiar sobre a confiança do segmentador foi considerada e rejeitada. Um segmentador treinado apenas com línguas nunca viu um negativo e produz confiança não confiável fora da distribuição. Sem controle sobre a captura, entradas arbitrárias são o caso normal, não a exceção. O gate precisa ser um classificador com negativos no treino.

**Modelo B — classificador de halitose.** Features interpretáveis sobre a região segmentada, com gradient boosting.

**Segmentador.** Primeira opção é [TongueSAM](https://github.com/cshan-github/TongueSAM) (licença MIT, pesos pré-treinados, zero-shot), que elimina a dependência da Roboflow sem treino. Se os pesos forem inviáveis de obter — estão hospedados no Baidu Pan —, o plano alternativo é treinar sobre BioHit (300 imagens com máscara manual) e TongueSet3 (1000 imagens em ambiente livre, capturadas com celular em ângulos variados, domínio próximo ao nosso), usando as 310 máscaras boas do projeto como dado adicional.

### 6.3 Features interpretáveis, não CNN

Medido em 4.3: um backbone pré-treinado grande perde para 35 features de cor. O ganho está em ampliar as features — espaço Lab além de HSV, mais percentis, textura multiescala, grade de setores mais fina —, não em capacidade de modelo.

Uma CNN só entra se superar AUC 0,797 em conjunto de teste isolado. A baseline de features é critério de admissão, não etapa descartável.

### 6.4 Sem augmentation de cor

Consequência de 4.4. Apenas transformações geométricas: espelhamento horizontal e rotação pequena. Redimensionamento preserva proporção.

### 6.5 Abstenção explícita

Descartando os 20% de predições mais incertas, o AUC sobe de 0,797 para **0,859** e a acurácia para **0,787** nos 80% restantes. A faixa central retorna resultado inconclusivo com orientação de repetir a foto ou procurar avaliação profissional.

### 6.6 Conjunto de teste isolado e intervalos de confiança

Divisão estratificada feita uma vez, persistida em arquivo versionado, não consultada até a avaliação final.

Com 318 amostras, um teste de 15% tem 48 exemplos e cerca de ±7 pontos de ruído em acurácia. Métrica de ponto único é ficção nesse tamanho. O protocolo é validação cruzada estratificada repetida com intervalo de confiança para toda decisão de modelagem, mais um conjunto de teste trancado usado uma única vez ao final.

## 7. Arquitetura

```
entrada: bytes de imagem
   │
1. normalização
     decodifica JPEG/PNG/BMP/HEIC, corrige orientação EXIF,
     reamostra preservando proporção
   │
2. verificação de qualidade da imagem
     nitidez e exposição — dois números, sem modelo
     ├─ tremida ou mal exposta → rejeição com motivo
     └─ ok ▼
3. MODELO A — gate de língua
     ├─ não é língua → rejeição com motivo
     └─ é língua ▼
4. segmentação
     → máscara + confiança
   │
5. verificação de sanidade da máscara
     máscara vazia ou fragmentada → rejeição com motivo
   │
6. extração de features
     cor, textura e setor sob a máscara → vetor nomeado
   │
7. MODELO B — classificador
     → probabilidade calibrada
   │
8. decisão com abstenção
     → positivo / negativo / inconclusivo + motivo
```

O pipeline tem três pontos de saída por rejeição — qualidade, gate, sanidade da máscara — mais a abstenção ao final. Rejeitar com um motivo específico é preferível a devolver uma predição confiante sobre uma entrada que não a sustenta.

**O teste de metadados da seção 4.5 não faz parte deste pipeline.** Ele é uma auditoria de conjunto de treino, executada offline quando dados novos são incorporados, e nunca toca uma imagem em tempo de inferência.

### 7.1 Componentes

**`normalize`** — bytes de imagem para array RGB canônico. Decodificação multiformato, orientação EXIF, reamostragem com proporção preservada.

**`image_quality`** — array RGB para aprovação ou motivo de rejeição. Lógica pura, sem modelo, dois indicadores:

- *Nitidez:* variância do Laplaciano. Abaixo do limiar, a imagem está tremida ou fora de foco.
- *Exposição:* fração de pixels saturados no branco somada à fração colada no preto. Acima do limiar, a informação de cor foi perdida — e a medição 4.4 mostra que cor é o sinal.

Ambos os limiares são calibrados sobre a distribuição das 318 imagens próprias e expostos como configuração. Esta etapa existe porque nem o gate nem a sanidade da máscara medem qualidade: uma foto tremida com uma língua nítida o bastante para ser segmentada passaria pelos dois e chegaria ao classificador com a cor comprometida.

**`tongue_gate`** — array RGB para veredito e confiança. Encapsula o Modelo A.

**`ood_check`** — embedding da imagem para aprovação ou abstenção. Mede a distância ao conjunto de treino e abstém acima de um limiar. Responde à pergunta que o gate não responde: não "há uma língua", mas "esta imagem se parece com aquelas sobre as quais o modelo foi calibrado". Motivada por 4.4c: o conjunto de treino cobre um único ambiente, e uma imagem doméstica pode conter uma língua legítima e ainda assim estar fora da distribuição em que as métricas valem.

**`segment`** — array RGB para máscara booleana e confiança. Encapsula o segmentador. Interface estável independentemente da implementação por trás.

**`mask_sanity`** — máscara e confiança para aprovação ou motivo de rejeição. Lógica pura, sem estado. Verifica área mínima, conectividade e fragmentação. **Não impõe limite superior de área** — a medição 4.6 mostra que área alta indica close-up legítimo.

**`extract_features`** — array RGB e máscara para vetor de features nomeadas. Os nomes e a ordem são parte do contrato: o modelo treinado depende deles.

**`predict`** — vetor de features para probabilidade calibrada. Carrega artefato versionado.

**`decide`** — probabilidade para veredito com abstenção. Lógica pura, limiares por configuração.

## 8. Protocolo de dados

**Unidade de amostragem:** o paciente, não a foto. Todas as imagens de um paciente ficam do mesmo lado da divisão.

**Divisão:** estratificada pelo rótulo binário, 70/15/15. Identificadores de cada partição gravados em arquivo versionado e reutilizados por todos os experimentos.

**Tabela mestra:** um passo único de reconciliação liga `ID_ANAMNESE`, caminho da imagem original, caminho normalizado, rótulo e partição. As pastas `Imagens lingua/`, `Classificacao/`, `classificacao_rotulados/` e os três diretórios `dataset*` são entrada bruta e não são consumidos diretamente pelo treino.

**Correção manual de máscaras:** as ~12 máscaras sub-segmentadas identificadas em 4.6 são corrigidas ou descartadas manualmente. Recupera 4% do dataset por cerca de meia hora de trabalho, rendimento superior a qualquer troca de modelo.

**Dados do Modelo A:**

- *Positivos:* as ~310 imagens próprias com máscara boa, mais as 2008 chinesas (usadas pelas imagens, nunca pelo rótulo), mais TCM-Tongue e TongueSet3 se aprovados no teste de metadados.
- *Hard negatives:* recortes tomados fora da máscara nas imagens de máscara sadia — lábio, dente, queixo, dedo enluvado, fundo. Mesmo equipamento, mesma iluminação, mesmos indivíduos. As imagens de área baixa são **excluídas** desta mineração: nelas, "fora da máscara" é a própria língua.
- *Negativos genéricos:* **item aberto.** Os hard negatives ensinam que lábio não é língua, mas não ensinam que parede não é língua. Falta uma fonte de imagens genéricas não-língua. É a única peça do plano sem solução definida.

**Datasets externos:** nenhum entra sem passar pelo teste de metadados de 4.5. Não existe dataset público com rótulo de halitose; a busca foi feita e é conclusiva. As 318 amostras próprias são o teto do Modelo B, e mais coleta clínica é a única alavanca que o levanta.

## 9. Métricas e critérios de aceite

**Modelo B**, métrica primária AUC em teste isolado, tarefa `classe 3 vs resto`:

| Critério | Alvo |
|---|---|
| AUC em teste | ≥ 0,78 |
| F1 macro em teste | ≥ 0,70 |
| Sensibilidade no ponto de operação | ≥ 0,85 |
| Taxa de abstenção | ≤ 0,20 |
| Latência de inferência, CPU, por foto | ≤ 1,0 s |

**Modelo A**, gate:

| Critério | Alvo |
|---|---|
| Recall de língua | ≥ 0,98 |
| Falso-aceite de não-língua | ≤ 0,05 |

Referências obrigatórias no relatório final: baseline majoritário (F1 macro 0,368) e modelo apenas com anamnese sem `Q6` (AUC 0,615).

Sensibilidade tem precedência sobre especificidade: deixar de sinalizar é pior que sinalizar um caso que o dentista descartará.

### 9.0 Métrica e cobertura são inseparáveis

O Modelo B é treinado e avaliado sobre imagens que passaram pelas etapas de qualidade, gate e sanidade da máscara — a mesma distribuição que verá em produção. Isso é correto, e cria um risco de leitura.

Filtrar eleva a métrica sem que o sistema tenha melhorado: quanto mais casos difíceis as etapas anteriores rejeitam, melhor parece o classificador. Um modelo com AUC 0,90 que rejeita 60% das entradas é pior, como produto, que um com AUC 0,80 que rejeita 5% — o primeiro simplesmente não responde quando é difícil.

**Regra:** toda métrica reportada vem acompanhada da cobertura que a produziu, na mesma frase. "AUC 0,86 com 80% de cobertura" é afirmação completa. "AUC 0,86" após filtragem é omissão. O relatório final traz a curva de métrica contra cobertura, não um ponto isolado.

### 9.1 O que os números significam na prática

Pontos de operação medidos, à prevalência de 58% do dataset:

| Sensibilidade | Especificidade | VPP | VPN |
|---|---|---|---|
| 0,95 | 0,44 | 0,70 | 0,87 |
| 0,90 | 0,60 | 0,76 | 0,81 |
| 0,85 | 0,65 | 0,77 | 0,75 |

**Prevalência é o limite prático.** Esses valores preditivos valem para população de clínica. A 20% de prevalência, no ponto de sensibilidade 0,90, o VPP cai para cerca de 0,36 — dois terços dos sinalizados não teriam a condição. É aritmética de triagem, não defeito do modelo, e nenhum ganho realista de AUC a contorna.

O desempenho é adequado para triagem com encaminhamento. Não é adequado para diagnóstico, e o teto está no dataset e no rótulo, não no modelo.

## 10. Tratamento de erros

| Situação | Comportamento |
|---|---|
| Formato não decodificável | Erro de requisição com formatos aceitos na mensagem |
| Nitidez abaixo do limiar | "Foto tremida ou fora de foco; refaça segurando firme" |
| Exposição estourada | "Foto muito clara; evite luz direta ou flash muito próximo" |
| Exposição escura demais | "Foto muito escura; procure um ambiente mais iluminado" |
| Modelo A rejeita | "Não foi identificada uma língua na imagem" |
| Máscara vazia ou fragmentada | "Não foi possível delimitar a língua; refaça a foto" |
| Área da máscara abaixo do mínimo | "Aproxime a câmera" |
| Probabilidade na faixa de incerteza | Inconclusivo, com orientação |
| Artefato de modelo ausente ou corrompido | Falha na inicialização, não em tempo de requisição |

Rejeição é resposta bem-sucedida e caminho normal do pipeline, não exceção.

## 11. Testes

- `image_quality`, `mask_sanity` e `decide` são lógica pura: testes de tabela cobrindo os valores exatos de fronteira.
- `image_quality` tem fixtures dedicadas: uma imagem nítida, uma desfocada por convolução gaussiana, uma superexposta e uma subexposta, verificando que cada uma recebe o motivo de rejeição correto.
- `normalize` é verificado contra fixtures cobrindo JPEG, PNG, BMP, HEIC e uma imagem com orientação EXIF rotacionada.
- `extract_features` tem teste de estabilidade: imagem e máscara conhecidas produzem vetor conhecido, de modo que mudanças na extração sejam detectadas antes de invalidarem um modelo treinado.
- Teste de ponta a ponta sobre uma foto de fixture, verificando o contrato de resposta.
- Teste de regressão de métrica, que falha se o AUC em teste cair abaixo do critério de aceite.
- Teste de confundidor: verifica que metadados de arquivo não predizem o rótulo acima de AUC 0,55 em qualquer conjunto usado no treino.

## 12. Fora de escopo

- Aplicativo, captura guiada e interface. Este escopo entrega modelos e pipeline de inferência.
- CNN, até que a baseline de features seja superada em teste isolado.
- Classificação em três níveis.
- Chamada à API da Roboflow em tempo de inferência.
- Rotulagem de índice de saburra (WTCI). Considerada e desnecessária: o rótulo clínico existente já é aprendível.
- Pré-treino no rótulo do dataset chinês, invalidado por 4.5.
- Uso da GPU AMD para treino, inviabilizado por 4.7.

## 13. Riscos

| Risco | Mitigação |
|---|---|
| AUC 0,797 vem de CV e pode não se sustentar em teste isolado | Critério fixado em 0,78, com margem; teste separado antes de qualquer ajuste |
| O recorte binário foi escolhido após observar o resultado de três classes | Reconfirmação obrigatória em holdout limpo |
| `Q6` pode ser vazamento do rótulo | Excluída até confirmação do enunciado |
| Sem negativos genéricos, o gate falha em entradas fora do domínio | Item aberto e bloqueante para o Modelo A; precisa de fonte definida |
| Metadados são proxy imperfeito de sessão de captura | Reexecutar o teste de 4.2 sempre que novas imagens forem incorporadas |
| 318 amostras limitam o teto | Coleta clínica contínua; é a única alavanca real |
| Pesos do TongueSAM hospedados no Baidu Pan podem ser inviáveis de obter | Plano alternativo já definido em 6.2 |

## 14. Pendências

1. **Enunciado de `Q1`–`Q10`.** Decide se `Q6` é sintoma relatado ou avaliação profissional, e portanto se pode voltar ao conjunto de features.
2. **Fonte de negativos genéricos** para o Modelo A. Bloqueante.
3. **Rotacionar a chave da Roboflow** exposta em `Segmentation.ipynb:58`, independentemente do cronograma.
4. **Verificar TCM-Tongue** com o teste de metadados antes de baixar 2,34 GB e planejar em cima.
