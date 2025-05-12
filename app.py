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
3. **Sondage à Probabilités Inégales (PPS)**
         
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

# Méthode 3 : Sondage à Probabilités Inégales (PPS)
st.header("Sondage à Probabilités Inégales (PPS)")

# Choix de la variable de taille
pps_size_var = st.selectbox("Selectionner la Variable", 
                            ['Region', 'GOVERNORATE', 'DELEGATION', 'SECTOR', 'Area'])

# Taille de l'échantillon
sample_size_pps = st.number_input("Taille de l'échantillon PPS", 
                                  min_value=1, 
                                  max_value=len(df), 
                                  value=min(100, len(df)//2))

if st.button("Tirer l'échantillon PPS"):
    df_valid = df[df[pps_size_var] > 0].copy()
    
    if df_valid.empty or len(df_valid) < sample_size_pps:
        st.error("Pas assez d'observations valides pour tirer l'échantillon.")
    else:
        # Calcul des probabilités de sélection
        weights = df_valid[pps_size_var]
        probabilities = weights / weights.sum()

        # Tirage sans remise selon les probabilités
        sample_pps = df_valid.sample(n=sample_size_pps, weights=probabilities, random_state=42)

        st.markdown("###### Échantillon PPS tiré")
        st.dataframe(sample_pps)

        # Carte
        st.write("Les points sur la carte représentent les unités tirées avec PPS.")
        st.map(sample_pps.dropna(subset=['latitude', 'longitude']))

        # Téléchargement
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            sample_pps.to_excel(writer, sheet_name='Echantillon_PPS', index=False)
        st.download_button(
            label="Télécharger l'échantillon PPS",
            data=output.getvalue(),
            file_name="echantillon_pps.xlsx",
            mime="application/vnd.ms-excel"
        )

        # Statistiques descriptives
        st.markdown("###### Statistiques descriptives")
        st.write(sample_pps.describe(include='all'))

        # Comparaison des proportions sur une variable catégorielle
        var_pps_comp = st.selectbox("Variable comparative pour PPS", 
                                    df.select_dtypes(include=['object', 'category']).columns)

        pop_prop = df[var_pps_comp].value_counts(normalize=True).reset_index()
        pop_prop.columns = [var_pps_comp, 'Proportion population']

        sample_prop = sample_pps[var_pps_comp].value_counts(normalize=True).reset_index()
        sample_prop.columns = [var_pps_comp, 'Proportion échantillon']

        comparison_df = pd.merge(pop_prop, sample_prop, on=var_pps_comp, how='outer').fillna(0)

        st.markdown("###### Tableau comparatif des proportions (PPS)")
        st.dataframe(comparison_df)

        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.set_index(var_pps_comp).plot(kind='bar', ax=ax)
        ax.set_title(f"Comparaison des proportions: {var_pps_comp} (PPS)")
        ax.set_ylabel("Proportion")
        st.pyplot(fig)





    
    