# Decisões de projeto

Cada decisão aqui saiu de uma medição feita neste repositório, e cada uma mudou o
código. O código não repete estas justificativas: elas moram aqui.

Índice rápido por arquivo:

| Arquivo | Decisões que ele encarna |
|---|---|
| `hality/features.py` | 1, 2 |
| `hality/data.py` | 3, 4 |
| `hality/segmenter.py`, `train_segmenter.py` | 5, 6 |
| `hality/train_gate.py` | 7, 8 |
| `hality/train_classifier.py` | 9, 10 |
| `hality/pipeline.py` | 11, 12, 13, 14, 15 |
| `hality/coating.py`, `raios.py` | 16, 17 (resultados negativos) |
| `hality/sam_segmenter.py`, `sam_onnx.py` | 18, 19 |

---

## 1. Percentil no lugar de limiar fixo

A característica dominante do classificador é `HSV_S_p10`, o percentil 10 da saturação
dentro da máscara: quão pálida é a porção mais pálida da língua. Isso é a camada de
saburra, medida de forma contínua.

Uma versão limiarizada da mesma ideia (`fração de pixels com saturação < 60 e brilho
> 120`) foi testada e descartada:

| Conjunto de features | AUC |
|---|---|
| Somente a feature limiarizada | 0,553 |
| Todas menos ela | 0,799 |
| Todas | 0,801 |

Ela não agrega nada. Binarizar descarta a informação que o percentil preserva.

**Regra:** medidas contínuas por percentil, nunca contagens limiarizadas.

## 2. O contrato das features é nome e ordem

`FEATURE_NAMES` é gravado junto com o modelo. Alterar a extração invalida o artefato
treinado, e o teste de estabilidade em `features.demo()` existe para que a mudança seja
detectada antes de virar um modelo silenciosamente errado.

## 3. Divisão por paciente, nunca por foto

Hoje cada paciente tem uma foto, então a regra não muda nada no classificador. Ela
importa desde já no gate, onde uma foto gera um positivo e vários negativos: se esses
recortes se espalhassem entre treino e teste, o modelo veria a mesma pele, a mesma luz
e o mesmo lábio dos dois lados, e o teste mediria memorização.

Fica escrita para quando chegar um segundo lote com várias fotos por pessoa.

## 4. Filtro de máscara sub-segmentada

`MIN_AREA_MASCARA = 0.15` remove 15 das 321 imagens. Inspeção visual mostrou que nelas
a segmentação da Roboflow capturou apenas um fragmento, tipicamente sobre a mancha de
saburra, embora a língua esteja nítida e centralizada.

Essas imagens ficam fora do treino e fora da mineração de negativos: nelas, "fora da
máscara" ainda é língua, e o negativo minerado teria rótulo invertido.

## 5. U-Net pequena em vez de fine-tune de backbone grande

483 mil parâmetros a 192×192. A tarefa é um blob central grande e de alto contraste,
então capacidade não é o gargalo, e o modelo cabe em CPU. 306 pares de treino não
sustentam um backbone grande.

IoU 0,842 na validação.

## 6. O segmentador treina só na partição de treino

Usar imagens de teste aqui vazaria: o segmentador produziria máscaras melhores
justamente nas fotos onde o classificador é medido.

As máscaras de referência vieram da API da Roboflow, então isto é destilação. Herdamos
o teto de qualidade dela, mas eliminamos a latência de rede por foto, o custo por
chamada, uma chave exposta em texto claro e o envio de imagem clínica a terceiro.

## 7. O gate é um modelo treinado, não um limiar

A alternativa seria derivar o gate da confiança do segmentador. Rejeitada: um
segmentador treinado apenas com línguas nunca viu um negativo e produz confiança não
calibrada fora da distribuição. Sem controle sobre a captura, entrada arbitrária é o
caso normal.

Recall 97,8% nas fotos próprias, falso-aceite 0,0% no COCO.

## 8. O gate classifica a FOTO INTEIRA, não recortes

Três tentativas anteriores falharam, e a causa foi conceitual.

| Versão | Resultado | Causa |
|---|---|---|
| v1 | AUC 1,0000 | positivo era recorte justo da língua, negativo era quadrado pequeno. Prever o rótulo só por largura e altura dava AUC 0,9991 |
| v2 | 0 negativos | tentei igualar o lado do negativo ao do positivo. Impossível: em close-up não existe região não-língua do tamanho da língua |
| v3 | recall 85% no domínio próprio | randomizei a resolução, mas o modelo aprendeu o domínio de estúdio das imagens chinesas |

A pergunta de produção nunca foi "este pedaço é língua", e sim "esta foto tem uma
língua". Na versão final a unidade é a imagem inteira e existe uma única função de
preparo, cega ao rótulo: a fração do recorte e a resolução saem do mesmo sorteio para
as duas classes. Não existe caminho de código que trate positivo diferente de negativo,
então o viés não pode ser construído por engano.

**Limite conhecido:** os negativos são cenas arbitrárias do COCO. Rua contra close de
boca é separação fácil. O caso difícil, rosto de boca fechada, não existe no conjunto e
não foi testado.

## 9. Alvo binário

Três classes colapsam: a classe 1 tem 22 amostras e o modelo acerta 4. O corte é
`nota == 3` contra o resto, que é o caso de maior consequência e o de melhor
separabilidade.

**Ressalva metodológica:** o recorte binário foi escolhido depois de observar o
desempenho de três classes. É decisão informada pelos dados e precisa de reconfirmação
em holdout limpo.

## 10. Somente imagem, sem anamnese

| Entrada | AUC |
|---|---|
| Imagem sozinha | 0,797 |
| Imagem + anamnese sem `Q6` | 0,799 |
| Anamnese sozinha sem `Q6` | 0,615 |

`Q6` correlaciona 0,528 com o rótulo, muito acima das demais (entre −0,01 e 0,15). Se
for um sintoma relatado, é feature legítima; se for do tipo "um profissional já lhe
disse que você tem mau hálito", é o rótulo reentrando disfarçado.

Os enunciados de `Q1`–`Q10` não existem em nenhum arquivo e não estão disponíveis. Sem
poder auditar `Q6`, ela fica permanentemente fora. E sem `Q6` a anamnese contribui
0,002, que é ruído.

**Decisão:** a anamnese sai inteira. A entrada do sistema é apenas a fotografia.
Efeito colateral favorável: o produto deixa de depender de questionário preenchido.

## 11. Exposição antes de nitidez

Uma foto escura tem pouco contraste e reprovaria no teste de nitidez, devolvendo "foto
tremida" para quem só precisa de mais luz. A ordem das checagens foi invertida depois
que o teste de rejeição mostrou exatamente esse motivo errado.

## 12. Sem limite superior de área da máscara

Inspeção por faixa de área mostrou que máscaras cobrindo 62% a 70% do quadro são
close-ups legítimos, e estão entre as melhores imagens do conjunto. Uma versão anterior
do desenho propunha rejeitar acima de 60%, o que teria descartado as melhores fotos.

## 13. Realce de contraste só para segmentar

| | AUC |
|---|---|
| Sem correção de iluminação | 0,786 |
| Gray-world | 0,683 |
| Referência fora da língua | 0,700 |
| White-patch | 0,687 |

Normalizar iluminação destrói o sinal, porque o brilho absoluto carrega informação
clínica: língua muito saburrosa é literalmente mais clara.

Mas o segmentador procura FORMA, e para ele contraste baixo é apenas ruído. Daí a
cascata: segmenta normal, e se a área sair implausível, tenta de novo com realce.
Aplicar realce sempre custaria IoU (0,8397 para 0,8085 em 40 fotos que já funcionavam);
como resgate, melhora 14 das 15 fotos difíceis sem pagar nas fáceis.

**Regra:** realce entra na segmentação, nunca na extração de features.

## 14. Augmentation geométrica sim, de cor não

Duas cópias aumentadas por imagem de treino, aplicadas apenas às dobras de treino:

| Augmentation | AUC |
|---|---|
| Nenhuma | 0,791 |
| Geométrica (espelhamento) | **0,799** |
| Jitter de cor ±5% | 0,785 |
| Jitter de cor ±10% | 0,779 |
| Jitter de cor ±20% (o do pipeline anterior) | 0,763 |
| Jitter de cor ±35% | 0,764 |

As diferenças pequenas estão dentro do IC de ±0,02. O que sustenta a regra é a
monotonicidade: cada aumento de intensidade piora, sem exceção.

## 15. Voto de maioria sobre cinco passagens

Teste de constância: a mesma língua refotografada com variação realista (enquadramento
3% a 7%, rotação até 4°, recompressão JPEG, exposição automática).

| Versão | Veredito mudou |
|---|---|
| Uma passagem | 48,3% |
| Média da probabilidade sobre 5 passagens | 36,7% |
| Voto de maioria sobre 5 passagens | **23,3%** |

Média da probabilidade sozinha não bastou porque parte da instabilidade não está no
classificador e sim nas PORTAS: uma foto no limite da nitidez ou da área de máscara é
aceita numa variante e rejeitada na outra. Votar sobre o veredito completo cobre as
duas fontes.

Quando não há maioria clara, a própria ausência de consenso é a resposta: inconclusivo.

A faixa de abstenção passou a ser dimensionada pelo ruído medido (`limiar ± 2 × 0,043`)
em vez de por percentil da validação. A versão anterior tinha largura 0,132 enquanto a
probabilidade oscilava 0,175, ou seja, a fronteira de decisão era mais estreita que o
próprio ruído.

**23,3% ainda é alto. É o principal item aberto do projeto.**

## 16. Resultado negativo: segmentação explícita de saburra

`hality/coating.py` fica no repositório como resultado negativo documentado. Não entra
no modelo.

| Conjunto | AUC |
|---|---|
| 34 features base | 0,766 |
| Saburra por k-means em Lab (nossa) | 0,532 |
| Saburra por `B − R − G` (SelectorNet, publicado, MIT) | 0,607 |
| Base + saburra SelectorNet | 0,767 |
| Base + ambas | 0,762 |

Duas implementações independentes, uma delas publicada em artigo revisado, chegam ao
mesmo lugar: nenhuma agrega. Dentro das features de saburra, `saburra_croma` domina, e
croma é essencialmente saturação. Ou seja, é um caminho mais longo para medir o que
`HSV_S_p10` já media.

A hipótese clínica está certa. A implementação não adiciona informação.

## 17. Resultado negativo: contorno por raios radiais

`hality/raios.py` fica como resultado negativo documentado.

A ideia era partir de um ponto dentro da língua e caminhar para fora em todas as
direções até a cor deixar de ser língua, obtendo um contorno fechado por construção.
Isso resolveria o problema do fragmento.

| Versão | IoU |
|---|---|
| Cor absoluta (distância ao modelo da semente) | 0,37 |
| Cor relativa (dois modelos, língua contra entorno) | 0,35 |
| U-Net | 0,85 |

Falhou, e a medição da premissa explica por quê:

```
d' entre a língua e o vizinho imediato:
  dente      3,26      separa muito bem
  cavidade   3,92      separa muito bem
  lábio      0,54      não separa

composição do perímetro da língua:
  lábio 60,7%   cavidade 35,6%   dente 1,7%
```

Textura também não resgata, em resolução nenhuma (d' entre 0,18 e 0,54, piorando
conforme a resolução sobe, porque o ruído cresce mais rápido que o sinal da papila).

A fronteira língua/lábio não é uma propriedade local da imagem. Ela é anatômica, e o
lábio é a maior parte do contorno. Nenhum algoritmo que olhe vizinhança local encontra
o que não está codificado localmente.

**O que sobrou e é usado:** cavidade e dente separam com d' acima de 3, então servem
para afirmar com alta confiança o que NÃO é língua. Isso virou o gerador de pontos
negativos por tipo de tecido em `sam_segmenter.py`.

## 18. SAM com gerador de prompts

A U-Net acerta na maioria, mas nas fotos difíceis devolve um fragmento, e features de
cor tiradas de um fragmento produzem veredito confiante e errado.

O SAM segmenta a partir de pontos. A U-Net, mesmo falhando, entrega exatamente isso: o
fragmento que ela acha está dentro da língua.

```
U-Net acha um pedaço  →  vira ponto positivo  →  SAM completa o órgão
```

Pontos negativos vêm de cavidade e dente (identificados por cor, onde d' > 3) e das
bordas do quadro. Sem eles o SAM tende a devolver o rosto inteiro.

Resultado nas 6 fotos mais difíceis: área sai de 0,13–0,33 para 0,40–0,51, contra
mediana de 0,42 nas fotos que sempre funcionaram. Quatro ficaram corretas, uma vazou
parcialmente, uma falhou.

Nas fotos fáceis o SAM é ligeiramente pior que a U-Net (IoU 0,8075 contra 0,8571), o
que aponta cascata em vez de substituição. **Ainda não integrado ao pipeline.**

## 19. Encoder do SAM em ONNX, na GPU AMD

A placa é uma Radeon RX 9070 XT. Não há CUDA, e `torch-directml` não tem distribuição
para Python 3.12 nem 3.14, então o PyTorch fica preso na CPU. Mas o DirectML roda em
qualquer GPU DX12, e o ONNX Runtime o expõe como `DmlExecutionProvider`.

Só o image encoder vai para a GPU, porque é cerca de 90% do custo do SAM. O prompt
decoder recebe uns poucos pontos e roda em microssegundos.

```
encoder CPU (PyTorch):   5,80 s/imagem
encoder GPU (DirectML):  0,16 s/imagem      35,3x
```

---

## Regras de processo

**Teste de confundidor obrigatório.** Antes de incorporar qualquer conjunto, tentar
prever o rótulo usando apenas metadados de arquivo. Vale também para conjuntos que nós
mesmos construímos.

| Conjunto | AUC só com metadados | Situação |
|---|---|---|
| Nosso, `classe 3 vs resto` | 0,498 | limpo |
| Dataset chinês `染苔/非染苔` | 0,995 | rótulo descartado |
| Primeira montagem do gate | 0,9991 | refeita |

Resultado próximo da separação perfeita é motivo de suspeita, não de aceitação.

**Métrica e cobertura são inseparáveis.** O classificador é avaliado sobre imagens que
passaram pelos filtros. Quanto mais casos difíceis as etapas anteriores rejeitam, melhor
ele parece. Um modelo com AUC 0,90 que rejeita 60% das entradas é pior, como produto,
que um com 0,80 que rejeita 5%.

**Intervalo, não ponto.** Com 306 amostras, uma partição de teste tem 48 exemplos, o
que dá cerca de ±0,10 de ruído no AUC. As três estimativas do mesmo modelo variaram de
0,700 a 0,878. O número que sustenta é o da validação cruzada repetida.
