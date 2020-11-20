# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Script: rastertab
# Author: Alexander Yoshizumi
# Date: 20 November 2020
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ------------------------------------------------------------------------------------------
# Import dependencies.
# ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask

# ------------------------------------------------------------------------------------------
# Create a vectorized function for checking if something is numeric. 
# ------------------------------------------------------------------------------------------

# Define function for checking if something is numeric or not.
def is_numeric(x):
    if x == None:
        return False
    else:
        try:
            float(x)
            return True
        except ValueError:
            return False

# Vectorize the function for use on an array.
is_numeric = np.vectorize(is_numeric)

# ------------------------------------------------------------------------------------------
# Create function that counts pixels for each category by zone.
# ------------------------------------------------------------------------------------------

def ZonalCategoricalCount(img, gdf, id = ""):
    
    # Record every possible category in the data and the number of unique values.
    categories = np.unique(img.read(1))
    categories_count = len(categories)

    # Set up new columns that will hold pixel counts.
    for x in range(categories_count):
        gdf[str(categories[x])] = 0

    # Create loop that will run pixel counts for every feature in the supplied geodataframe.
    for i in range(len(gdf)):
        # Count number of bands in image.
        band_count = img.count

        # Extract geometry of each feature in the geodataframe.
        geometries = gdf.geometry.values
        feature = [mapping(geometries[i])]

        # Create a mask based on the geometry of the feature.
        out_image, out_transform = mask(img, feature, crop = True)

        # Eliminate all the pixels outside of the mask for all bands.
        out_image_trimmed = out_image[:,~np.all(out_image == 0, axis=0)]

        # Reshape the array to [pixel count, bands].
        out_image_reshaped = out_image_trimmed.reshape(-1, band_count)

        # Store unique cateogires, a count of pixels per unique value,
        # and the total number of unique categories.
        unique_categories, pixel_counts = np.unique(out_image_reshaped, return_counts = True)
        unique_length = len(unique_categories)

        # Assign values to each row.
        for category in unique_categories:
            gdf.at[gdf.index[i],str(category)] = pixel_counts[list(unique_categories).index(category)]

    # Rename columns by the name provided.
    for x in range(categories_count):
        gdf.rename(
            columns = {str(categories[x]):'Value_'+str(categories[x])+str(id)},
            inplace = True)

# ------------------------------------------------------------------------------------------
# Create function that tabulates sum of NDVI by year, vegetation pixel count, and total pixel count.
# ------------------------------------------------------------------------------------------

def TabulateNDVI(img, gdf, band_names, red_band, NIR_band, id = "", threshold = 0):

    # Set up new column that will hold sum of NDVI.
    gdf['NDVI_Sum'+str(id)] = 0.0
    gdf['Veg_Pixels'+str(id)] = 0
    gdf['Tot_Pixels'+str(id)] = 0

    for i in range(len(gdf)):
        # Count number of bands in image.
        band_count = img.count

        # Extract geometry of each feature in the geodataframe.
        geometries = gdf.geometry.values
        feature = [mapping(geometries[i])]

        # Create a mask based on the geometry of the feature.
        out_image, out_transform = mask(img, feature, crop = True)

        # Eliminate all the pixels outside of the mask for all bands.
        out_image_trimmed = out_image[:,~np.all(out_image == 0, axis=0)]

        # Reshape the array to [pixel count, bands].
        out_image_reshaped = out_image_trimmed.reshape(-1, band_count)

        # Form reshaped array into dataframe.
        df = pd.DataFrame(data = out_image_reshaped, columns = band_names)

        # Retain only the "R" and "NIR" band
        df = df[[red_band,NIR_band]]

        # Calculate NDVI.
        NDVI = (df['NIR'] - df['R']) / (df['NIR'] + df['R'])
        df['NDVI'] = NDVI

        # Calculate NDVI sum and vegetation pixel count.
        gdf.at[gdf.index[i],'NDVI_Sum'+str(id)] = np.sum(df['NDVI'])
        gdf.at[gdf.index[i],'Veg_Pixels'+str(id)] = np.sum(df['NDVI'] >= threshold)
        try:
            gdf.at[gdf.index[i],'Tot_Pixels'+str(id)] = np.sum(is_numeric(df['NDVI']))
        except:
            gdf.at[gdf.index[i],'Tot_Pixels'+str(id)] = 0