# Projeto de Deep Learning — Pokémon
### **Alunos:** Jorge Luiz Tessmann e Guilherme dos Passos Persch



## 1. Objetivo

O projeto utiliza uma rede neural para classificar Pokémon em três perfis: ofensivo físico, ofensivo especial e defensivo.

Com essa classificação, o programa monta três tipos de times: equilibrado, ofensivo e defensivo.

A escolha do tema permite trabalhar com atributos numéricos, tipos de Pokémon e técnicas de classificação, utilizando os resultados para montar equipes com características diferentes.




## 2. Dataset

Foram utilizados dois arquivos CSV:

- `status.csv`: contém informações dos Pokémon, como nome, tipo e atributos.
- `efetividade_tipos.csv`: contém os valores de efetividade dos ataques contra cada tipo.

Os arquivos são carregados da seguinte forma:

```python
dataset_status = pd.read_csv("status.csv")
efetividade = pd.read_csv("efetividade_tipos.csv")
```

O arquivo de status fornece os dados usados na classificação. Já o arquivo de efetividade permite calcular as fraquezas, resistências e imunidades de cada Pokémon.

O dataset foi escolhido porque reúne informações que podem ser usadas para comparar os Pokémon e montar times com características diferentes.

**Observação:** o código não informa a origem dos arquivos CSV nem a quantidade total de registros. Essas informações devem ser acrescentadas antes da entrega.



## 3. Preparação dos Dados

Primeiro, foram selecionadas as colunas necessárias para o projeto, como tipos, habilidades e atributos:

```python
colunas = [
    "No", "Name", "Generation",
    "Type1", "Type2",
    "Ability1", "Ability2", "Ability_Hidden",
    "Category", "Region_Form", "Mega_Evolution_Flag",
    "HP", "Attack", "Defense",
    "SP_Attack", "SP_Defense", "Speed", "Total"
]

dataset_ia = dataset_status[colunas].copy()
```

Os nomes das colunas foram traduzidos para facilitar o uso no código.

Também foi feita uma verificação dos valores ausentes. Nos campos em que a ausência representa que o Pokémon não possui aquela característica, o valor foi substituído por `"Nenhum"`:

```python
colunas_texto = [
    "Tipo_1", "Tipo_2", "Habilidade_2",
    "Habilidade_Oculta", "Forma_Regional",
    "Mega_Evolucao"
]

dataset_ia[colunas_texto] = dataset_ia[colunas_texto].fillna("Nenhum")
```

Depois, o programa calculou a efetividade dos ataques considerando os dois tipos de cada Pokémon.

A partir desses valores, foram criados três indicadores:

- **Fraquezas:** tipos de ataque que causam dano dobrado ou maior.
- **Resistências:** tipos de ataque que causam menos dano.
- **Imunidades:** tipos de ataque que não causam dano.

Esses indicadores também são usados para calcular uma pontuação defensiva:

```python
dataset_ia["Pontuacao_Defensiva"] = (
    dataset_ia["Resistencias"] * 2
    + dataset_ia["Imunidades"] * 3
    - dataset_ia["Fraquezas"] * 2
)
```

Essa pontuação considera as resistências e imunidades como vantagens e as fraquezas como desvantagens.



## 4. Classificação dos Pokémon

Para treinar a rede neural, cada Pokémon precisava ter uma classificação. Essa classificação foi criada usando os próprios atributos do dataset.

O programa compara os valores de Ataque, Ataque Especial, Vida, Defesa e Defesa Especial. O perfil com maior resultado é atribuído ao Pokémon:

```python
perfis = pd.DataFrame({
    "Ofensivo_Fisico": dataset_ia["Ataque"].rank(pct=True),
    "Ofensivo_Especial": dataset_ia["Ataque_Especial"].rank(pct=True),
    "Defensivo": pd.concat([
        dataset_ia["Vida"].rank(pct=True),
        dataset_ia["Defesa"].rank(pct=True),
        dataset_ia["Defesa_Especial"].rank(pct=True)
    ], axis=1).mean(axis=1)
})

dataset_ia["Perfil"] = perfis.idxmax(axis=1)
```

Assim, cada Pokémon recebe um dos três perfis: ofensivo físico, ofensivo especial ou defensivo.

Os seis atributos principais e os tipos são utilizados como entradas do modelo. Como os tipos são informações de texto, eles são convertidos em colunas numéricas com `get_dummies()`.

Essa etapa permite que a rede neural trabalhe com informações numéricas em vez de nomes de tipos.



## 5. Modelo e Treinamento

Os dados foram separados em 75% para treinamento e 25% para teste:

```python
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
    previsores,
    classe,
    test_size=0.25,
    random_state=42,
    stratify=classe
)
```

Os atributos numéricos foram normalizados com `StandardScaler`, para que diferenças de escala não atrapalhassem o treinamento.

A rede neural foi criada com três camadas:

- Primeira camada: 16 neurônios.
- Segunda camada: 8 neurônios.
- Camada de saída: 3 neurônios, representando os três perfis.

O código utilizado foi:

```python
modelo = Sequential()

modelo.add(Dense(
    16,
    input_dim=previsores.shape[1],
    activation="relu"
))

modelo.add(Dense(8, activation="relu"))

modelo.add(Dense(3, activation="softmax"))
```

As camadas internas utilizam a função de ativação ReLU. Já a camada de saída utiliza Softmax para indicar a probabilidade de cada perfil.

O modelo utiliza o otimizador Adam, a função de perda `sparse_categorical_crossentropy` e a acurácia como métrica:

```python
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

O treinamento foi configurado para 100 épocas, com lotes de 16 registros:

```python
historico = modelo.fit(
    x_treinamento,
    y_treinamento,
    epochs=100,
    batch_size=16,
    validation_data=(x_teste, y_teste)
)
```

No código atual, os dados de teste também são utilizados durante o treinamento para acompanhar a validação. Uma possível melhoria seria separar um conjunto próprio para validação e deixar os dados de teste reservados para a avaliação final.



## 6. Avaliação

Após o treinamento, o modelo faz previsões para os dados de teste.

A matriz de confusão compara as classificações previstas com as classificações originais:

```python
confusao = confusion_matrix(
    y_teste_matrix,
    y_previsao_matrix
)
```

Também é calculada a acurácia, que mostra a proporção de previsões corretas:

```python
acuracia = accuracy_score(
    y_teste_matrix,
    y_previsao_matrix
)

print("Acurácia:", acuracia)
```

A matriz de confusão permite observar quais perfis foram classificados corretamente e quais foram confundidos pelo modelo.

O arquivo enviado contém o código que calcula essas métricas, mas não contém os resultados da execução. Por isso, não é possível informar aqui o valor da acurácia nem os números da matriz de confusão.



## 7. Montagem dos Times

Depois da classificação, o programa calcula uma pontuação para ajudar a escolher os Pokémon.

Essa pontuação considera o total dos atributos, com peso de 60%, e a pontuação defensiva, com peso de 40%:

```python
dataset_ia["Pontuacao_Final"] = (
    dataset_ia["Total_Normalizado"] * 0.6
    + dataset_ia["Defesa_Normalizada"] * 0.4
)
```

A função `montar_time()` seleciona os Pokémon com as maiores pontuações dentro de cada perfil.

Caso não consiga completar seis integrantes com a divisão definida, o programa escolhe outros Pokémon com as maiores pontuações.

As três composições são:

**Time equilibrado**
- 2 ofensivos físicos.
- 2 ofensivos especiais.
- 2 defensivos.

**Time ofensivo**
- 3 ofensivos físicos.
- 2 ofensivos especiais.
- 1 defensivo.

**Time defensivo**
- 4 defensivos.
- 1 ofensivo físico.
- 1 ofensivo especial.

O código também gera gráficos para comparar os atributos dos Pokémon, a quantidade de Pokémon por tipo, a composição dos times e a matriz de confusão.



## 8. Conclusão

O projeto utiliza uma rede neural para classificar os Pokémon em três perfis e usa essa classificação para montar times com propostas diferentes.

Durante o desenvolvimento, foram trabalhadas etapas de seleção e preparação dos dados, cálculo de efetividade dos tipos, normalização, treinamento e avaliação do modelo.

Também foram criados gráficos para facilitar a análise dos dados e dos times.

Uma limitação do projeto é que os perfis são criados por regras baseadas nos próprios atributos dos Pokémon. Portanto, o modelo aprende a reproduzir essas regras. Uma boa acurácia não significa necessariamente que a classificação seja a melhor para batalhas reais.

Outra melhoria seria separar um conjunto de validação próprio, deixando os dados de teste reservados para a avaliação final.



## Informações Complementar



- **Fonte do dataset:** link de onde foram obtidos os arquivos CSV.
- **Quantidade de registros:** 1025.
- **Acurácia:** 0.9435215946843853.
- **Matriz de confusão:**    

> | 67 | 2 | 2 |
> |:---:|:---:|:---:|
> | 3 | 110 | 4 |
> | 4 | 2 | 107 |

