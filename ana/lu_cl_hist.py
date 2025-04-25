import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
# from collection import counter
from collections import defaultdict
import os
from tqdm import tqdm
import pandas as pd

def load_rasters(folder_path):
    rasters=[]
    for ras in tqdm(sorted(os.listdir(folder_path)), desc=f"Processing LU maps"):
        pth = os.path.join(folder_path, ras)
        with rasterio.open(pth) as src:
            rasters.append({
                'name': os.path.basename(pth)
                'data': src.read(1),
                'meata': src.meta,
                'transform': src.transform,
                'crs': src.crs,
                'path': pth
            })
    return rasters


def check_compatability (rasters):
    base = rasters[0]
    for r in rasters[1:]:
        if base['transform'] != r['transform'] or base['crs'] != r['crs'] or base['meta']['width'] != r['meta']['width'] or base['meta']['height'] != r['meta']['height']:
           raise ValueError(f"Raster {r['path']} does not math the extent or crs of base raster.")
    print('All rasters are compatible.')


def ana_classes (rasters):
    hists=[]
    for r in rasters:
        unique, counts = np.unique(r['data'], return_counts=True)
        hists.append((unique, counts))
    return hists


def plt_hists(hists, rasters):
    plt.figure(figsize=(12,6))
    for i, (unique, counts) in enumerate(hists):
        plt.bar(unique + i*0.3, counts, width=0.3, label=os.path.basename(rasters[i]['path']))
    plt.xlabel("LU Class")
    plt.ylabel("Pixle Count")
    plt.title("LU Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


def ana_lu(folder_path):
    area_dict = defaultdict(dict)
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".tif"):
            year = int(''.join(filter(str.isdigit, file)))
            filepath = os.path.join(folder_path, file)
            with rasterio.open(filepath) as src:
                data = src.read(1)
                pixel_area = abs(src.transform[0] * src.transform[4])

                unique, counts = np.unique(data, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    if cls == src.nodata:
                        continue
                    area = cnt * pixel_area / 1e6 # convert to sq.km
                    area_dict[int(cls)][year] = area

    return area_dict

def plot_lu_byear(area_dict):
    df = pd.DataFrame(area_dict).fillna(0).T.sort_index()
    years = sorted(next(iter(area_dict.values())).keys())

    # Plot bar for each year: class distribution
    for year in years:
        plt.figure(figsize=(10, 6))
        class_areas = {cls: area_dict[cls].get(year, 0) for cls in area_dict}
        plt.bar(class_areas.keys(), class_areas.values())
        plt.title(f"Land Use Class Area Distribution - {year}")
        plt.xlabel("Class")
        plt.ylabel("Area (km²)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"class_distribution_{year}.png")
        plt.close()

    # Plot change per class over years
    df = pd.DataFrame(area_dict).fillna(0)
    for cls in df.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(df.index, df[cls])
        plt.title(f"Cange in Area of Class {cls} Over Time")
        plt.xlabel("Year")
        plt.ylabel("Area (km²)")
        plt.tight_layout()
        plt.savefig(f"class_{cls}_trend.png")
        plt.close()


if __name__=="__main__":
    folder_path = "/tudelft.net/staff-umbrella/EDT Veluwe/testbed/luclhist"
    rasters = load_rasters(folder_path)
#    check_compatability(rasters)
    histograms = ana_classes(rasters)
    plt_hists(histograms, rasters)


if __name__=="__main__":
    folder_path = "/tudelft.net/staff-umbrella/EDT Veluwe/testbed/lgntest"
    area_data = ana_lu(folder_path)
    plot_lu_byear(area_data)


#raslist = sorted(os.listdir(folder_path))
#print(raslist)


==========================================================================

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_specific_classes(folder_path, target_classes=[11, 12]):
    class_areas = {cls: {} for cls in target_classes}  # {class: {year: area}}

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".tif"):
            year = int(''.join(filter(str.isdigit, file)))
            filepath = os.path.join(folder_path, file)
            with rasterio.open(filepath) as src:
                data = src.read(1)
                pixel_area = abs(src.transform[0] * src.transform[4]) / 1e6  # in sq.km

                for cls in target_classes:
                    count = np.sum(data == cls)
                    class_areas[cls][year] = count * pixel_area

    return class_areas


def plot_loofbos_naalbos_trend(class_areas):
    years = sorted(next(iter(class_areas.values())).keys())

    plt.figure(figsize=(10, 6))
    for cls, areas in class_areas.items():
        area_list = [areas.get(year, 0) for year in years]
        label = "Loofbos (Class 11)" if cls == 11 else "Naalbos (Class 12)"
        plt.plot(years, area_list, marker='o', label=label, linewidth=2)

    plt.xlabel("Year")
    plt.ylabel("Area (sq.km)")
    plt.title("Change in Forest Area (Loofbos & Naalbos)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loofbos_naalbos_trend.png")
    plt.show()





def analyze_specific_classes(folder_path, target_classes=[321, 323]):
    class_areas = {cls: {} for cls in target_classes}  # {class: {year: area}}

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".tif"):
            year = int(''.join(filter(str.isdigit, file)))
            filepath = os.path.join(folder_path, file)
            with rasterio.open(filepath) as src:
                data = src.read(1)
                pixel_area = abs(src.transform[0] * src.transform[4]) / 1e6  # in sq.km

                for cls in target_classes:
                    count = np.sum(data == cls)
                    class_areas[cls][year] = count * pixel_area

    return class_areas

def plot_vegetatie_trend(class_areas):
    years = sorted(next(iter(class_areas.values())).keys())

    plt.figure(figsize=(10, 6))
    for cls, areas in class_areas.items():
        area_list = [areas.get(year, 0) for year in years]
        label = "struikvegtaie in hoogveengebied (Class 321)" if cls == 321 else "struikvegtaie in moerasgebied (Class 323)"
        plt.plot(years, area_list, marker='o', label=label, linewidth=2)

    plt.xlabel("Year")
    plt.ylabel("Area (sq.km)")
    plt.title("Change in struikvegetatie (Laag)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("str-vegetatie-hoog_moer_laag_trend.png")
    plt.show()

# Example usage
folder_path = "/tudelft.net/staff-umbrella/EDT Veluwe/testbed/lgntest"
class_data = analyze_specific_classes(folder_path)
plot_loofbos_naalbos_trend(class_data)
plot_vegetatie_trend(class_data)

