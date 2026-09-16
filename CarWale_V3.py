#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


raw_df = pd.read_csv("carwale_raw.csv")
ext_df = pd.read_csv("carwale_extracted_data.csv")


# In[3]:


print("Raw Dataset")
display(raw_df.head())

print(raw_df.shape)

print(raw_df.info())

print(raw_df.isnull().sum())


# In[4]:


print("Extracted Dataset")

display(ext_df.head())

print(ext_df.shape)

print(ext_df.info())

print(ext_df.isnull().sum())


# In[5]:


raw_df.columns = raw_df.columns.str.strip()
ext_df.columns = ext_df.columns.str.strip()

print(raw_df.columns)
print(ext_df.columns)


# In[6]:


raw = raw_df.copy()
ext = ext_df.copy()


# In[7]:


ext.rename(columns={'Engine': 'Engine (cc)'}, inplace=True)


# In[8]:


print(ext.columns)


# In[9]:


raw.dtypes


# In[10]:


ext.dtypes


# In[11]:


ext['Brand'] = (
    ext['Brand']
        .str.lower()
        .str.replace('-', ' ', regex=False)
        .str.replace('_', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
)

raw['Brand'] = (
    raw['Brand']
        .str.lower()
        .str.replace('-', ' ', regex=False)
        .str.replace('_', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
)


# In[12]:


import re

def clean_name(name):
    if pd.isna(name):
        return name

    name = name.lower()

    name = name.replace('-', ' ')
    name = name.replace('_', ' ')

    name = re.sub(r'[^a-z0-9 ]', '', name)

    name = re.sub(r'\s+', ' ', name)

    return name.strip()

raw['Car_Name_Clean'] = raw['Car Name'].apply(clean_name)

ext['Car_Name_Clean'] = ext['Car Name'].apply(clean_name)


# In[14]:


ext['Engine (cc)'] = (
    ext['Engine (cc)']
        .astype(str)
        .str.extract(r'(\d+)')[0]
)

ext['Engine (cc)'] = pd.to_numeric(ext['Engine (cc)'], errors='coerce')


# In[15]:


ext['Mileage'] = (
    ext['Mileage']
        .astype(str)
        .str.extract(r'([\d.]+)')[0]
)

ext['Mileage'] = pd.to_numeric(ext['Mileage'], errors='coerce')


# In[16]:


ext['User Rating'] = (
    ext['User Rating']
        .replace('Not Rated', np.nan)
        .astype(str)
        .str.extract(r'([\d.]+)')[0]
)

ext['User Rating'] = pd.to_numeric(ext['User Rating'], errors='coerce')


# In[17]:


ext['Car_Name_Clean'].duplicated().sum()


# In[18]:


duplicates = ext[ext.duplicated('Car_Name_Clean', keep=False)]

duplicates.sort_values('Car_Name_Clean')


# In[19]:


def first_non_null(series):
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return series.iloc[0]

lookup = (
    ext
    .groupby("Car_Name_Clean", as_index=False)
    .agg(first_non_null)
)


# In[20]:


lookup.shape


# In[21]:


lookup[lookup["Car_Name_Clean"] == "zest"]


# In[22]:


def extract_model(row):
    car = str(row["Car_Name_Clean"])
    brand = str(row["Brand"])

    # Remove brand only if it appears at the beginning
    if car.startswith(brand):
        model = car[len(brand):].strip()
    else:
        model = car

    return model


# In[23]:


raw["Model_Name"] = raw.apply(extract_model, axis=1)


# In[24]:


lookup["Model_Name"] = lookup["Car_Name_Clean"]


# In[25]:


raw[["Brand", "Car Name", "Model_Name"]].head(20)


# In[26]:


lookup_small = lookup[
    [
        "Model_Name",
        "Fuel Type",
        "Transmission",
        "Engine (cc)",
        "Mileage",
        "User Rating"
    ]
]


# In[27]:


merged = raw.merge(
    lookup_small,
    on="Model_Name",
    how="left",
    suffixes=("", "_new")
)


# In[28]:


print("Raw rows:", len(raw))
print("Merged rows:", len(merged))


# In[29]:


merged["Fuel Type_new"].notna().sum()


# In[30]:


merged["Engine (cc)_new"].notna().sum()


# In[31]:


columns = [
    "Fuel Type",
    "Transmission",
    "Engine (cc)",
    "Mileage",
    "User Rating"
]

for col in columns:
    merged[col] = merged[col].fillna(merged[f"{col}_new"])


# In[32]:


merged.drop(
    columns=[
        "Car_Name_Clean",
        "Model_Name",
        "Fuel Type_new",
        "Transmission_new",
        "Engine (cc)_new",
        "Mileage_new",
        "User Rating_new"
    ],
    inplace=True
)


# In[33]:


merged.to_csv("carwale_cleaned.csv", index=False)

print("✅ carwale_cleaned.csv created successfully.")


# In[ ]:





# In[64]:


df = pd.read_csv("carwale_cleaned.csv")


# In[66]:


electric_keywords = [
    "ev",
    "electric",
    "e tron",
    "etron",
    "ioniq",
    "atto",
    "seal",
    "sealion",
    "ix",
    "i4",
    "i5",
    "i7",
    "ix1",
    "ix3",
    "eq",
    "taycan"
]
def is_electric(car_name):
    car_name = str(car_name).lower()

    for word in electric_keywords:
        if word in car_name:
            return True

    return False


# In[67]:


mask = (
    df["Fuel Type"].isna() &
    df["Car Name"].apply(is_electric)
)

df.loc[mask, "Fuel Type"] = "Electric"

print("Filled:", mask.sum())


# In[68]:


import re

def clean_name(name):
    name = str(name).lower()
    name = name.replace("-", " ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()

df["Model"] = df["Car Name"].apply(clean_name)


# In[69]:


def extract_model(row):
    car = row["Model"]
    brand = row["Brand"]

    if car.startswith(brand):
        return car[len(brand):].strip()

    return car

df["Model"] = df.apply(extract_model, axis=1)


# In[70]:


fuel_lookup = (
    df
    .dropna(subset=["Fuel Type"])
    .groupby("Model")["Fuel Type"]
    .first()
)


# In[71]:


mask = df["Fuel Type"].isna()

df.loc[mask, "Fuel Type"] = (
    df.loc[mask, "Model"]
      .map(fuel_lookup)
)

print("Filled:", mask.sum() - df["Fuel Type"].isna().sum())


# In[ ]:





# In[72]:


trans_lookup = (
    df
    .dropna(subset=["Transmission"])
    .groupby("Model")["Transmission"]
    .first()
)

mask = df["Transmission"].isna()

df.loc[mask, "Transmission"] = (
    df.loc[mask, "Model"]
      .map(trans_lookup)
)

print("Remaining:", df["Transmission"].isna().sum())


# In[73]:


engine_lookup = (
    df
    .dropna(subset=["Engine (cc)"])
    .groupby("Model")["Engine (cc)"]
    .first()
)

mask = df["Engine (cc)"].isna()

df.loc[mask, "Engine (cc)"] = (
    df.loc[mask, "Model"]
      .map(engine_lookup)
)

print("Remaining:", df["Engine (cc)"].isna().sum())


# In[74]:


mileage_lookup = (
    df
    .dropna(subset=["Mileage"])
    .groupby("Model")["Mileage"]
    .first()
)

mask = df["Mileage"].isna()

df.loc[mask, "Mileage"] = (
    df.loc[mask, "Model"]
      .map(mileage_lookup)
)

print("Remaining:", df["Mileage"].isna().sum())


# In[75]:


df.drop(columns=["Model"], inplace=True)


# In[76]:


df.isnull().sum()


# In[77]:


missing_before = merged.isnull().sum()
missing_after = df.isnull().sum()

comparison = pd.DataFrame({
    "Before": missing_before,
    "After": missing_after,
    "Filled": missing_before - missing_after
})

print(comparison)


# In[ ]:





# In[78]:


import re

variant_words = [
    "fearless", "creative", "smart", "pure", "adventure",
    "accomplished", "dark", "plus", "prime", "signature",
    "style", "sport", "luxury", "limited", "edition",
    "facelift", "automatic", "manual", "diesel", "petrol",
    "ev", "electric", "hybrid", "turbo", "awd", "4x4",
    "zx", "vx", "ex", "sx", "sxo", "ax3", "ax5", "ax7",
    "mx", "lx", "rx", "rs"
]

def create_base_model(row):

    car = str(row["Car Name"]).lower()
    brand = str(row["Brand"]).lower()

    # Remove brand only from the beginning
    if car.startswith(brand):
        car = car[len(brand):].strip()

    # Remove text inside brackets
    car = re.sub(r"\[.*?\]", "", car)

    # Remove punctuation
    car = re.sub(r"[^a-z0-9 ]", " ", car)

    # Remove variant words
    words = []
    for word in car.split():
        if word not in variant_words:
            words.append(word)

    car = " ".join(words)

    # Remove extra spaces
    car = re.sub(r"\s+", " ", car).strip()

    return car

df["Base_Model"] = df.apply(create_base_model, axis=1)


# In[79]:


df[["Car Name", "Base_Model"]].sample(30, random_state=42)


# In[80]:


import re

# Models that genuinely have two words
two_word_models = [
    "grand vitara",
    "range rover",
    "discovery sport",
    "discovery svx",
    "defender 90",
    "defender 110",
    "defender 130",
    "urban cruiser",
    "fortuner legender",
    "3 series",
    "5 series",
    "6 series",
    "7 series",
    "2 series",
    "xuv 3xo",
    "xev 9e"
]

def create_base_model(row):

    car = str(row["Car Name"]).lower()
    brand = str(row["Brand"]).lower()

    # Remove brand only from beginning
    if car.startswith(brand):
        car = car[len(brand):].strip()

    # Remove text inside brackets
    car = re.sub(r"\[.*?\]", "", car)

    # Remove punctuation
    car = re.sub(r"[^a-z0-9 ]", " ", car)

    # Normalize spaces
    car = re.sub(r"\s+", " ", car).strip()

    # Preserve known two-word models
    for model in two_word_models:
        if car.startswith(model):
            return model

    # Otherwise use first word
    return car.split()[0] if car else car


# In[81]:


df[df["Car Name"].str.contains("Creta", case=False, na=False)][
    ["Car Name", "Base_Model"]
]


# In[ ]:





# In[ ]:





# In[82]:


import re

# Models that genuinely have two words
two_word_models = [
    "grand vitara",
    "range rover",
    "discovery sport",
    "discovery svx",
    "defender 90",
    "defender 110",
    "defender 130",
    "urban cruiser",
    "fortuner legender",
    "3 series",
    "5 series",
    "6 series",
    "7 series",
    "2 series",
    "xuv 3xo",
    "xev 9e"
]

def create_base_model(row):

    car = str(row["Car Name"]).lower()
    brand = str(row["Brand"]).lower()

    # Remove brand only from beginning
    if car.startswith(brand):
        car = car[len(brand):].strip()

    # Remove text inside brackets
    car = re.sub(r"\[.*?\]", "", car)

    # Remove punctuation
    car = re.sub(r"[^a-z0-9 ]", " ", car)

    # Normalize spaces
    car = re.sub(r"\s+", " ", car).strip()

    # Preserve known two-word models
    for model in two_word_models:
        if car.startswith(model):
            return model

    # Otherwise use first word
    return car.split()[0] if car else car


# In[83]:


df["Base_Model"] = df.apply(create_base_model, axis=1)


# In[84]:


df[
    ["Car Name", "Base_Model"]
].sample(40, random_state=10)


# In[85]:


df[df["Car Name"].str.contains("Creta", case=False, na=False)][
    ["Car Name", "Base_Model"]
]


# In[86]:


fuel_lookup = (
    df.dropna(subset=["Fuel Type"])
      .groupby("Base_Model")["Fuel Type"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
)


# In[87]:


mask = df["Fuel Type"].isna()

df.loc[mask, "Fuel Type"] = (
    df.loc[mask, "Base_Model"].map(fuel_lookup)
)


# In[88]:


trans_lookup = (
    df.dropna(subset=["Transmission"])
      .groupby("Base_Model")["Transmission"]
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
)

mask = df["Transmission"].isna()

df.loc[mask, "Transmission"] = (
    df.loc[mask, "Base_Model"].map(trans_lookup)
)


# In[89]:


engine_lookup = (
    df.dropna(subset=["Engine (cc)"])
      .groupby("Base_Model")["Engine (cc)"]
      .median()
)

mask = df["Engine (cc)"].isna()

df.loc[mask, "Engine (cc)"] = (
    df.loc[mask, "Base_Model"].map(engine_lookup)
)


# In[90]:


mileage_lookup = (
    df.dropna(subset=["Mileage"])
      .groupby("Base_Model")["Mileage"]
      .median()
)

mask = df["Mileage"].isna()

df.loc[mask, "Mileage"] = (
    df.loc[mask, "Base_Model"].map(mileage_lookup)
)


# In[91]:


comparison = pd.DataFrame({
    "Before": merged.isnull().sum(),
    "After": df.isnull().sum()
})

comparison["Filled"] = comparison["Before"] - comparison["After"]

comparison


# In[92]:


# Remove temporary helper columns before saving
columns_to_drop = [
    "Base_Model",
    "Model_Name",
    "Car_Name_Clean"
]

existing_columns = [col for col in columns_to_drop if col in df.columns]

df_save = df.drop(columns=existing_columns)

# Save the cleaned dataset
df_save.to_csv("carwale_cleaned_v2.csv", index=False)

print("✅ carwale_cleaned_v2.csv saved successfully!")


# In[93]:


cleaned = pd.read_csv("carwale_cleaned_v2.csv")
cleaned.isnull().sum()


# In[94]:


df = cleaned.copy()


# In[95]:


electric_keywords = [
    "ev", "electric", "e-tron", "etron",
    "ioniq", "ix", "i4", "i5", "i7", "ix1", "ix3",
    "atto", "seal", "sealion",
    "eq", "taycan",
    "e6", "emax",
    "bev"
]


# In[96]:


def detect_electric(car_name):
    car_name = str(car_name).lower()

    for word in electric_keywords:
        if word in car_name:
            return "Electric"

    return None


# In[97]:


mask = df["Fuel Type"].isna()

df.loc[mask, "Fuel Type"] = (
    df.loc[mask, "Car Name"]
      .apply(detect_electric)
)

print(df["Fuel Type"].isna().sum())


# In[98]:


hybrid_keywords = [
    "hybrid",
    "strong hybrid",
    "plug in hybrid",
    "phev"
]


# In[99]:


def detect_hybrid(car_name):

    car_name = str(car_name).lower()

    for word in hybrid_keywords:

        if word in car_name:
            return "Hybrid"

    return None


# In[100]:


mask = df["Fuel Type"].isna()

df.loc[mask, "Fuel Type"] = (
    df.loc[mask, "Car Name"]
      .apply(detect_hybrid)
)

print(df["Fuel Type"].isna().sum())


# In[101]:


import re

two_word_models = [
    "grand vitara",
    "range rover",
    "discovery sport",
    "3 series",
    "5 series",
    "6 series",
    "7 series",
    "2 series",
    "xuv 3xo",
    "xev 9e"
]

def create_base_model(row):

    car = str(row["Car Name"]).lower()
    brand = str(row["Brand"]).lower()

    if car.startswith(brand):
        car = car[len(brand):].strip()

    car = re.sub(r"\[.*?\]", "", car)
    car = re.sub(r"[^a-z0-9 ]", " ", car)
    car = re.sub(r"\s+", " ", car).strip()

    for model in two_word_models:
        if car.startswith(model):
            return model

    return car.split()[0] if car else car

df["Base_Model"] = df.apply(create_base_model, axis=1)


# In[102]:


fuel_lookup = (
    df.dropna(subset=["Fuel Type"])
      .groupby(["Brand", "Base_Model"])["Fuel Type"]
      .agg(lambda x: x.mode().iloc[0])
)


# In[103]:


mask = df["Fuel Type"].isna()

df.loc[mask, "Fuel Type"] = (
    df.loc[mask]
      .set_index(["Brand", "Base_Model"])
      .index
      .map(fuel_lookup)
)

print(df["Fuel Type"].isna().sum())


# In[104]:


print("Remaining Fuel Type:", df["Fuel Type"].isna().sum())


# In[105]:


trans_group = (
    df.dropna(subset=["Transmission"])
      .groupby(["Brand", "Engine (cc)", "Fuel Type"])["Transmission"]
      .nunique()
      .reset_index()
)

trans_group.head()


# In[106]:


unique_trans = trans_group[trans_group["Transmission"] == 1]

print("Unique groups:", len(unique_trans))


# In[107]:


trans_lookup = (
    df.dropna(subset=["Transmission"])
      .groupby(["Brand", "Engine (cc)", "Fuel Type"])["Transmission"]
      .first()
)


# In[108]:


mask = df["Transmission"].isna()

df.loc[mask, "Transmission"] = (
    df.loc[mask]
      .set_index(["Brand", "Engine (cc)", "Fuel Type"])
      .index
      .map(trans_lookup)
)


# In[109]:


before = cleaned["Transmission"].isna().sum()
after = df["Transmission"].isna().sum()

print(f"Before : {before}")
print(f"After  : {after}")
print(f"Filled : {before - after}")


# In[110]:


df.isnull().sum()


# In[111]:


df.to_csv("carwale_cleaned_v3.csv",index = False)


# In[112]:


df.shape


# In[113]:


complete_rows = df.dropna()

print("Number of complete rows:", len(complete_rows))


# In[114]:


# Fill categorical/object columns
df["Fuel Type"] = df["Fuel Type"].fillna("Unknown")
df["Transmission"] = df["Transmission"].fillna("Unknown")
df["User Rating"] = df["User Rating"].fillna("Not Rated")

# Fill numeric columns with "Unknown"
df["Engine (cc)"] = df["Engine (cc)"].astype(object).fillna("Unknown")
df["Mileage"] = df["Mileage"].astype(object).fillna("Unknown")
df["Seating Capacity"] = df["Seating Capacity"].astype(object).fillna("Unknown")


# In[115]:


df.isnull().sum()


# In[116]:


df.to_csv("carwale_cleaned_v3.csv",index = False)


# In[ ]:




