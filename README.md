# TCC-DSA

Este repositório contém o código e os dados utilizados no projeto de TCC do MBA em Data Science e Analytics da USP/Esalq.

TRata-se de uma aplicação para análise de dados financeiros. O projeto utiliza Python e diversas bibliotecas para realizar ETL (Extração, Transformação e Carga) e análise de dados.

# Estrutura do Projeto

## ETL

### Extração

Dados dos preços das ações obtidos atravéz da API da plataforma [BRAPI]("https://brapi.com.br/"), cliquei aqui para saber mais.

Dados armazenados em formato JSON para posterior tratamento.

### Transformação

Dos dados em JSON são extraídos os preços históricos das ações e armazenadas em parquet.

Após o armazenamento os dados em Parquet são mergeados em um único arquivo onde o ídice são as datas e as colunas são as ações.