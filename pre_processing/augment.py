#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd

df_raw = pd.read_csv("../data/modern_laptops.csv", index_col=0)
df_raw = pd.read_csv("./data/modern_laptops.csv", index_col=0)
df_raw.head()


# In[34]:


df = df_raw.rename(
    columns={
        "processor_name": "cpu",
        "ram(GB)": "ram",
        "ssd(GB)": "ssd_gb",
        "Hard Disk(GB)": "hdd_gb",
        "Operating System": "os",
        "graphics": "gpu",
        "screen_size(inches)": "screen_size_in",
        "no_of_cores": "cpu_cores",
        "no_of_threads": "cpu_threads",
    }
)


# In[35]:


df = df.dropna(subset=["resolution (pixels)"])
df = df[df["gpu"].str.lower() != "missing"].reset_index(drop=True)


# In[36]:


df["resolution_w"] = (
    df["resolution (pixels)"].str.split("x").str[0].str.strip().astype(int)
)
df["resolution_h"] = (
    df["resolution (pixels)"].str.split("x").str[1].str.strip().astype(int)
)


# In[37]:


mask = df["resolution_w"] < df["resolution_h"]
df.loc[mask, ["resolution_w", "resolution_h"]] = df.loc[
    mask, ["resolution_h", "resolution_w"]
].values


# In[38]:


def get_resolution_type(w, h):
    if w >= 3840 and h >= 2160:
        return "4K"
    elif w >= 2560 and h >= 1440:
        return "QHD"
    elif w >= 1920 and h >= 1080:
        return "FHD"
    else:
        return "HD"


df["resolution_type"] = df.apply(
    lambda row: get_resolution_type(row["resolution_w"], row["resolution_h"]), axis=1
)


# In[39]:


df["os"] = df["os"].replace({"Ubuntu": "Linux", "DOS": "Linux"})


# In[40]:


import re


def normalize_cpu(cpu: str) -> str:
    cpu = cpu.strip()

    # Already normalized
    if cpu.startswith(("Intel ", "AMD ", "Apple ")):
        return cpu

    # Handle Intel generational format: "11th Gen Core i5" -> "Intel Core i5 11th Gen"
    gen_match = re.match(
        r"(\d+)(?:th|st|nd|rd)\s+Gen\s+(Core i[3579])", cpu, re.IGNORECASE
    )
    if gen_match:
        gen = f"{gen_match.group(1)}th Gen"
        tier = gen_match.group(2)
        # Ensure 'i' is lowercase: "Core I5" -> "Core i5"
        tier = re.sub(r"([iI])(\d)", r"i\2", tier)
        return f"Intel {tier} {gen}"

    # Intel: Celeron or Pentium
    if re.search(r"\b[Cc]eleron\b|\b[Pp]entium\b", cpu):
        return "Intel " + re.sub(r"^(Intel\s*)?", "", cpu)

    # AMD: Ryzen or Athlon
    if re.search(r"\b[Rr]yzen\b|\b[Aa]thlon\b", cpu):
        return "AMD " + re.sub(r"^(AMD\s*)?", "", cpu)

    # If none match, return as is
    return cpu


# In[41]:


df["cpu"] = df["cpu"].apply(normalize_cpu)


# In[42]:


cpu_series = df["cpu"].fillna("").astype(str)
mask = cpu_series.str.match(r"^\s*\d")
indices = df.index[mask].tolist()
print(f"Found {len(indices)} matches at indices: {indices}")
matches = (
    df.loc[mask, ["model_name", "brand", "cpu"]]
    .reset_index()
    .rename(columns={"index": "orig_index"})
)
matches


# In[43]:


df.at[129, "cpu"] = "Intel Core i7 10th Gen"
df.at[355, "cpu"] = "Intel Core i5 12th Gen"
df.at[555, "cpu"] = "Intel Core i7 12th Gen"
df.at[822, "cpu"] = "Intel Core i3 11th Gen"

df.loc[[129, 166, 555, 822], ["model_name", "brand", "cpu"]]
#


# In[44]:


df["price_euro"] = df["price_inIndianRupees"] * 0.0097
df["price_euro"] = df["price_euro"].round(2)


# In[45]:


import numpy as np


def extract_vram(gpu_str):
    if pd.isna(gpu_str):
        return pd.Series([np.nan, np.nan])

    match = re.match(r"^(\d+\s*GB)\s+(.*)", gpu_str)
    if match:
        return pd.Series([match.group(1), match.group(2)])
    else:
        return pd.Series([np.nan, gpu_str])


df[["gpu_vram", "gpu"]] = df["gpu"].apply(extract_vram)


# In[46]:


df["gpu_vram_gb"] = (
    df["gpu_vram"]
    .astype(str)
    .str.replace(r"[^\d]", "", regex=True)
    .replace("", np.nan)
    .astype("float")
    .astype("Int64")
)


df = df.drop(columns=["gpu_vram"])


df[["gpu_vram_gb", "gpu"]].head()


# In[47]:


df.drop(columns=["price_inIndianRupees", "resolution (pixels)"])


# In[48]:


df = df[
    [
        "model_name",
        "brand",
        "cpu",
        "cpu_cores",
        "cpu_threads",
        "ram",
        "ssd_gb",
        "hdd_gb",
        "os",
        "gpu",
        "gpu_vram_gb",
        "screen_size_in",
        "resolution_w",
        "resolution_h",
        "resolution_type",
        "spec_score",
        "price_euro",
    ]
]


# In[49]:


df.head()


# In[50]:


total_dup = df.duplicated().sum()
dup_indices = df.index[df.duplicated()].tolist()
print(f"Total exact duplicate rows: {total_dup}, indices: {dup_indices}")


# In[51]:


subset_cols = ["model_name", "brand", "cpu", "ram", "ssd_gb", "hdd_gb"]
dupes = (
    df.groupby(subset_cols)
    .size()
    .reset_index(name="count")
    .query("count>1")
    .sort_values("count", ascending=False)
)
dupes.head(15)


# In[52]:


df[df.duplicated(subset=subset_cols, keep=False)].sort_values(subset_cols).head()


# In[53]:


subset_dup_indices = df.index[df.duplicated(subset=subset_cols)].tolist()
print("Indices duplicate by subset (keeping first):", subset_dup_indices)


# In[54]:


df = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)


# In[55]:


df.to_csv("../data/laptops_cleaned.csv", index=False)
