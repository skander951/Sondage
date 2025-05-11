import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np  

# Dans votre fichier app.py
st.set_page_config(
    page_title="Sampling App Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Titre de l'application
st.title("📊 Sampling App")
st.write("""
Bienvenue dans **Sampling App**, une application interactive elaborée par **Skander Kolsi** et **Khalil Fadhlaoui**
         conçue pour faciliter le tirage d’échantillons à partir de données géographiques.""")

st.header("Fonctionnalités :")
st.write("""1. **Aléatoire simple sans remise (SAS)**
2. **Stratification à allocation proportionnelle**
3. **Sondage Saskatchewan (Sondage systématique stratifié)**
         
Vous pouvez visualiser, comparer et télécharger les échantillons générés.
""")

# Chargement des données
data = pd.read_excel("data/cadre_echantillon.xlsx")
coord_df = pd.read_excel("data/tn.xlsx")
# Fusion sur le nom du gouvernorat
df = data.merge(coord_df[['city', 'lat', 'lng']], left_on='GOVERNORATE', right_on='city', how='left')
# Renommer pour st.map()
df = df.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# Afficher la carte avec les points géographiques
st.header("Visualisation des données géographiques")
st.write("Les points sur la carte représentent les gouvernorats de la Tunisie.")
st.map(df.dropna(subset=['latitude', 'longitude']))

# Afficher un aperçu des données
if st.checkbox("Afficher un aperçu des données"):
    st.write("Aperçu du cadre d'échantillonnage :")
    st.dataframe(df.head())

# Afficher les informations sur le cadre d'échantillonnage
st.header("Informations sur le cadre d'échantillonnage")
st.metric("###### Nombre total d'enregistrements :", len(df))
st.metric("###### Nombre de variables :", len(df.columns))
st.markdown("###### Types de variables")
st.write(df.dtypes)
st.markdown("###### Statistiques descriptives des variables qualitatives")
var_quali = ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area','Block']
st.write(df[var_quali].describe().loc[['count','unique', 'top', 'freq']])
st.markdown("###### Statistiques descriptives des variables quantitatives")
var_quanti = ['pop_block','Lodging','Cumulative population']
st.write(df[var_quanti].describe().loc[['mean', 'std','min', '25%', '50%', '75%', 'max']])

# Méthode 1: Aléatoire simple sans remise (SAS)
st.header("Aléatoire Simple Sans Remise (SAS)")
# Inputs
sample_size_sas = st.number_input("Sélectionnez la taille de l'échantillon", 
                                 min_value=1, 
                                 max_value=len(df), 
                                 value=min(100, len(df)//2))
var_comparative = st.selectbox("Sélectionnez la variable comparative", 
                              ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area'])

if st.button("Tirer l'échantillon SAS"):
    # Tirage aléatoire
    sample_sas = df.sample(n=sample_size_sas, random_state=42)
    # Statistiques descriptives
    st.markdown("###### Échantillon tiré")
    st.dataframe(sample_sas)
    # Affichage de la carte avec l'échantillon*
    st.write("Les points sur la carte représentent les gouvernorats de l'échantillon.")
    st.map(sample_sas.dropna(subset=['latitude', 'longitude']))
    
    # statistiques descriptives
    st.markdown("###### Statistiques descriptives des variables qualitatives")
    var_quali = ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area','Block']
    st.write(sample_sas[var_quali].describe().loc[['count','unique', 'top', 'freq']])
    st.markdown("###### Statistiques descriptives des variables quantitatives")
    var_quanti = ['pop_block','Lodging','Cumulative population']
    st.write(sample_sas[var_quanti].describe().loc[['mean', 'std','min', '25%', '50%', '75%', 'max']])

    # Tableau comparatif
    st.markdown("###### Tableau comparatif des proportions")

    # Calcul des proportions
    pop_prop = df[var_comparative].value_counts(normalize=True).reset_index()
    pop_prop.columns = [var_comparative, 'Proportion population']
    
    sample_prop = sample_sas[var_comparative].value_counts(normalize=True).reset_index()
    sample_prop.columns = [var_comparative, 'Proportion échantillon']
    
    comparison_df = pd.merge(pop_prop, sample_prop, on=var_comparative, how='outer').fillna(0)
    
    st.dataframe(comparison_df)
    
    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.set_index(var_comparative).plot(kind='bar', ax=ax)
    ax.set_title(f"Comparaison des proportions: {var_comparative}")
    ax.set_ylabel("Proportion")
    st.pyplot(fig)

     # Téléchargement de l'échantillon
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sample_sas.to_excel(writer, sheet_name='Echantillon_SAS', index=False)
    st.download_button(
        label="Télécharger l'échantillon SAS",
        data=output.getvalue(),
        file_name="echantillon_SAS.xlsx",
        mime="application/vnd.ms-excel"
    )

# Méthode 2: Stratification à allocation proportionnelle
st.header("Stratification à Allocation Proportionnelle")

# Inputs
sample_size_strat = st.number_input("Sélectionnez la taille de l'échantillon)", 
                                   min_value=1, 
                                   max_value=len(df), 
                                   value=min(100, len(df)//2))

strat_var = st.selectbox("Sélectionnez la variable de stratification", 
                        ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area'])

if st.button("Tirer l'échantillon stratifié"):
    # Calcul des allocations proportionnelles
    strata_sizes = df[strat_var].value_counts()
    allocations = (strata_sizes / strata_sizes.sum() * sample_size_strat).round().astype(int)
    
    # Ajustement pour que la somme soit égale à sample_size_strat
    diff = sample_size_strat - allocations.sum()
    if diff != 0:
        allocations.iloc[0] += diff

    st.markdown("###### Tableau d'allocations par strate")
    allocation_df = pd.DataFrame({
        'Strate': allocations.index,
        'Taille population': strata_sizes,
        'Allocation (nh)': allocations
    })
    st.dataframe(allocation_df)
    
    # Tirage stratifié
    samples = []
    for stratum, size in allocations.items():
        stratum_df = df[df[strat_var] == stratum]
        if len(stratum_df) >= size:
            sample_stratum = stratum_df.sample(n=size, random_state=42)
        else:
            sample_stratum = stratum_df  # Si la strate est plus petite que l'allocation
        samples.append(sample_stratum)
    
    sample_strat = pd.concat(samples)
    
    # Affichage de l'échantillon
    st.markdown("###### Échantillon stratifié tiré")
    st.dataframe(sample_strat)

    # Affichage de la carte avec l'échantillon stratifié
    st.write("Les points sur la carte représentent les gouvernorats de l'échantillon stratifié.")
    st.map(sample_strat.dropna(subset=['latitude', 'longitude']))

    # statistiques descriptives
    st.markdown("###### Statistiques descriptives des variables qualitatives")
    var_quali = ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area','Block']
    st.write(sample_strat[var_quali].describe().loc[['count','unique', 'top', 'freq']])
    st.markdown("###### Statistiques descriptives des variables quantitatives")
    var_quanti = ['pop_block','Lodging','Cumulative population']
    st.write(sample_strat[var_quanti].describe().loc[['mean', 'std','min', '25%', '50%', '75%', 'max']])
    
    # Visualisation des allocations
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Diagramme des allocations
    allocation_df.plot(x='Strate', y='Allocation (nh)', kind='bar', ax=ax)
    ax.set_title("Allocation par strate")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Téléchargement de l’échantillon
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sample_strat.to_excel(writer, sheet_name='Echantillon_Stratifie', index=False)
        allocation_df.to_excel(writer, sheet_name='Allocations', index=False)
    st.download_button(
        label="Télécharger l'échantillon stratifié",
        data=output.getvalue(),
        file_name="echantillon_stratifie.xlsx",
        mime="application/vnd.ms-excel"
    )

st.header("Sondage Saskatchewan (Sondage systématique stratifié)")

sample_size_sask = st.number_input("Taille de l'échantillon pour Saskatchewan", 
                                   min_value=1, 
                                   max_value=len(df), 
                                   value=min(100, len(df)//2))

strat_var_sask = st.selectbox("Variable de stratification (Saskatchewan)", 
                              ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area'], key="sask_strat")

ordre_var = st.selectbox("Variable d'ordre croissant dans chaque strate", 
                         ['pop_block','Lodging','Cumulative population'], key="sask_order")

if st.button("Tirer l'échantillon Saskatchewan"):
    strata_sizes = df[strat_var_sask].value_counts()
    allocations = (strata_sizes / strata_sizes.sum() * sample_size_sask).round().astype(int)
    diff = sample_size_sask - allocations.sum()
    if diff != 0:
        allocations.iloc[0] += diff

    st.markdown("###### Allocations par strate")
    allocation_df_sask = pd.DataFrame({
        'Strate': allocations.index,
        'Taille population': strata_sizes,
        'Allocation (nh)': allocations
    })
    st.dataframe(allocation_df_sask)

    # Tirage Saskatchewan
    sask_samples = []
    for stratum, nh in allocations.items():
        stratum_df = df[df[strat_var_sask] == stratum].sort_values(by=ordre_var).reset_index(drop=True)
        Nh = len(stratum_df)
        if nh == 0 or Nh == 0:
            continue
        k = Nh / nh
        r = np.random.uniform(0, k)
        indices = [int(r + i * k) for i in range(nh) if int(r + i * k) < Nh]
        sask_sample = stratum_df.iloc[indices]
        sask_samples.append(sask_sample)

    sask_final_sample = pd.concat(sask_samples)

    st.markdown("###### Échantillon Saskatchewan")
    st.dataframe(sask_final_sample)

    # Affichage de la carte avec l'échantillon Saskatchewan
    st.write("Les points sur la carte représentent les gouvernorats de l'échantillon Saskatchewan.")
    st.map(sask_final_sample.dropna(subset=['latitude', 'longitude']))

    # statistiques descriptives
    st.markdown("###### Statistiques descriptives des variables qualitatives")
    var_quali = ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area','Block']
    st.write(sask_final_sample[var_quali].describe().loc[['count','unique', 'top', 'freq']])
    st.markdown("###### Statistiques descriptives des variables quantitatives")
    var_quanti = ['pop_block','Lodging','Cumulative population']
    st.write(sask_final_sample[var_quanti].describe().loc[['mean', 'std','min', '25%', '50%', '75%', 'max']])

    # Visualisation des allocations
    fig, ax = plt.subplots(figsize=(10, 6))
    allocation_df_sask.plot(x='Strate', y='Allocation (nh)', kind='bar', ax=ax)
    ax.set_title("Allocation par strate")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # téléchargement de l'échantillon
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sask_final_sample.to_excel(writer, sheet_name='Echantillon_Saskatchewan', index=False)
        allocation_df_sask.to_excel(writer, sheet_name='Allocations', index=False)
    st.download_button(
        label="Télécharger l'échantillon Saskatchewan",
        data=output.getvalue(),
        file_name="echantillon_saskatchewan.xlsx",
        mime="application/vnd.ms-excel"
    )




    
    