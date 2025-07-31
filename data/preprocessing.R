library(dplyr)
library(janitor)
library(survminer)
library(caret)

# Preparando os dados
dados_aux <- read.csv("data/dataset_clean.csv") |>
  clean_names() |>
  mutate(
    sexo = factor(case_when(
      sexo == 1 ~ "masc",
      sexo == 2 ~ "fem"
    ), levels = c("masc", "fem")),
    cateatend = factor(case_when(
      cateatend == 1 | cateatend == 3 ~ "convenio_ou_particular",
      cateatend == 2 ~ "sus",
      cateatend == 9 ~ "sem_informacao"
    ), levels = c("convenio_ou_particular", "sus", "sem_informacao")),
    diagprev = factor(case_when(
      diagprev == 1 ~ "sem_diag_e_sem_trat",
      diagprev == 2 ~ "com_diag_e_sem_trat"
    ), levels = c("sem_diag_e_sem_trat", "com_diag_e_sem_trat")),
    ecgrup = factor(ecgrup),
    cirurgia = factor(case_when(
      cirurgia == 0 ~ "nao",
      cirurgia == 1 ~ "sim"
    ), levels = c("nao", "sim")),
    hormonio = factor(case_when(
      hormonio == 0 ~ "nao",
      hormonio == 1 ~ "sim"
    ), levels = c("nao", "sim")),
    quimio = factor(case_when(
      quimio == 0 ~ "nao",
      quimio == 1 ~ "sim"
    ), levels = c("nao", "sim")),
    radio = factor(case_when(
      radio == 0 ~ "nao",
      radio == 1 ~ "sim"
    ), levels = c("nao", "sim")),
    outros = factor(case_when(
      outros == 0 ~ "nao",
      outros == 1 ~ "sim"
    ), levels = c("nao", "sim")),
    recnenhum = factor(case_when(
      recnenhum == 0 ~ "nao",
      recnenhum == 1 ~ "sim"
    ), levels = c("nao", "sim")),
    escolari_2 = factor(case_when(
      escolari_2 == 1 ~ "analfabeto",
      escolari_2 == 2 ~ "ens_fund_incompleto",
      escolari_2 == 3 ~ "ens_fund_completo",
      escolari_2 == 4 ~ "ens_medio",
      escolari_2 == 5 ~ "ens_superior"
    ), levels = c("analfabeto", "ens_fund_incompleto", "ens_fund_completo", "ens_medio", "ens_superior")),
    idade_cat = factor(case_when(
      idade <= 49 ~ "0_a_49_anos",
      idade >= 50 & idade <= 74 ~ "50_a_74_anos",
      idade >= 75 ~ "75_anos_mais"
    ), levels = c("0_a_49_anos", "50_a_74_anos", "75_anos_mais")),
    tratcons_cat = factor(case_when(
      tratcons <= 60 ~ "ate_60_dias",
      tratcons > 60 ~ "mais_de_60_dias"
    ), levels = c("ate_60_dias", "mais_de_60_dias"))
  ) |>
  select(
    time_years, falha = status_cancer_specific, anodiag, cateatend, cirurgia,
    diagprev, diagtrat, ecgrup, escolari_2, hormonio, idade_cat, outros,
    quimio, radio, recnenhum, sexo, tratcons_cat
  )

## Encontrando pontos de corte ótimos para anodiag e diagtrat utilizando o pacote maxstat
cutpoints <- surv_cutpoint(
  dados_aux,
  time = "time_years",
  event = "falha",
  c("anodiag", "diagtrat"),
  minprop = 0.2
)

cutpoints

## Adicionando as novas categorizações ao dataframe original e mudando as categorias de referência
dados <- dados_aux |>
  mutate(
    anodiag_cat = factor(
      ifelse(anodiag <= 2006, "ate_2006", "apos_2006"),
      levels = c("ate_2006", "apos_2006")
    ),
    .after = "anodiag"
  ) |>
  mutate( 
    diagtrat_cat = factor(
      ifelse(diagtrat <= 81, "ate_81_dias", "mais_de_81_dias"),
      levels = c("ate_81_dias", "mais_de_81_dias")
    ),
    .after = "diagtrat"
  ) |>
  select(!c(anodiag, diagtrat)) |>
  mutate(
    anodiag_cat = relevel(anodiag_cat, "apos_2006"),
    cateatend = relevel(cateatend, "convenio_ou_particular"),
    cirurgia = relevel(cirurgia, "sim"),
    diagprev = relevel(diagprev, "com_diag_e_sem_trat"),
    diagtrat_cat = relevel(diagtrat_cat, "ate_81_dias"),
    ecgrup = relevel(ecgrup, "I"),
    escolari_2 = relevel(escolari_2, "ens_superior"),
    hormonio = relevel(hormonio, "sim"),
    idade_cat = relevel(idade_cat, "0_a_49_anos"),
    outros = relevel(outros, "sim"),
    quimio = relevel(quimio, "nao"),
    radio = relevel(radio, "nao"),
    recnenhum = relevel(recnenhum, "sim"),
    sexo = relevel(sexo, "fem"),
    tratcons_cat = relevel(tratcons_cat, "mais_de_60_dias")
  ) 

rm(cutpoints, dados_aux)

## Discretizando o tempo de sobrevida
dados <- dados |> mutate (
  sobrevida = case_when(
    time_years < 1 ~ "<1ano",
    time_years >= 1 & time_years < 3 ~ "<3anos",
    time_years >= 3 & time_years < 5 ~ "<5anos",
    time_years >= 5 ~ ">5anos"
  )
)

## Escrevendo as bases
## Desconsiderando a cesura
dados |>
  select(-falha, -time_years) |>
  write.csv("data/sem_censura/dataset_sem_censura.csv", row.names = FALSE)

## Considerando a censura
dados |>
  filter(falha == 1 | sobrevida == ">5anos") |>
  select(-falha, -time_years) |>
  write.csv("data/com_censura/dataset_com_censura.csv", row.names = FALSE)
