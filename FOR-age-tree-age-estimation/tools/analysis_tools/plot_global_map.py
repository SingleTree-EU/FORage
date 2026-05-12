#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

# ========== (A) 原始数据 (与之前一致) ==========
datasets_raw = [
    ("CULS",         "Czech Republic",  "ULS", (49.8,   15.5)),
    ("BlueCat",      "Czech Republic",  "TLS", (49.85,  15.4)),
    ("NIBIO",        "Norway",          "ULS", (61.3,   8.5)),
    ("NIBIO2",       "Norway",          "ULS", (61.35,  8.4)),
    ("NIBIO_MLS",    "Norway",          "MLS", (61.25,  8.6)),
    ("RMIT",         "Australia",       "ULS", (-37.8, 144.9)),
    ("SCION",        "New Zealand",     "ULS", (-38.0, 175.5)),
    ("TUWIEN",       "Austria",         "ULS", (48.2,   16.4)),
    ("LAUTx",        "Austria",         "MLS", (48.25,  16.45)),
    ("Yuchen",       "French Guiana",   "ULS", (4.0,   -53.0)),
    ("Wytham woods", "United Kingdom",  "TLS", (51.7,   -1.34))
]

# ========== (B) 定义欧洲区域的边界 ==========
eu_lon_min, eu_lon_max = -10, 25
eu_lat_min, eu_lat_max = 45, 65

# ========== (C) 合并到国家级别，做传感器组合颜色(与之前相同) ==========
region_data = {}
for (name, country, sensor, (lat, lon)) in datasets_raw:
    if country not in region_data:
        region_data[country] = {
            "coords": [],
            "sensors": set()
        }
    region_data[country]["coords"].append((lat, lon))
    region_data[country]["sensors"].add(sensor)

combined_data = []
for country, info in region_data.items():
    coords = info["coords"]
    sensors = sorted(list(info["sensors"]))  
    combo_str = "/".join(sensors)  
    lat_avg = np.mean([c[0] for c in coords])
    lon_avg = np.mean([c[1] for c in coords])
    combined_data.append((country, combo_str, (lat_avg, lon_avg)))

# ========== (D) 定义颜色表 ==========
combo_colors = {
    "ULS":          "#e41a1c",  # 红
    "MLS":          "#377eb8",  # 蓝
    "TLS":          "#4daf4a",  # 绿
    "MLS/TLS":      "#984ea3",  # 紫
    "MLS/ULS":      "#ff7f00",  # 橙
    "TLS/ULS":      "#984ea3",  # (原本写的是黄, 但这里重复了, 你可自行调整)
    "MLS/TLS/ULS":  "#a65628"   # 棕
}

# ========== (E) 设置全球地图范围 ==========
# 过滤掉欧洲区域后剩下的点
lon_min, lon_max = -55, 185
lat_min, lat_max = -45, 70

fig = plt.figure(figsize=(7, 4), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 极简：仅海岸线与国界
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black")
ax.add_feature(cfeature.BORDERS, linewidth=0.5, color="black")

# 画回归线(±23.5)、极圈(±66.5)和赤道(0)
special_lats = [-66.5, -23.5, 0, 23.5, 66.5]
for lat_line in special_lats:
    if lat_min <= lat_line <= lat_max:
        ax.plot([lon_min, lon_max], [lat_line, lat_line],
                transform=ccrs.PlateCarree(),
                linestyle='--', linewidth=0.7, color='gray')

# ========== (F) 绘制圆点，但跳过那些落在欧洲边界内的点 ==========
used_labels = set()
for (country, combo_str, (lat_avg, lon_avg)) in combined_data:
    # 判断是否在欧洲范围
    in_europe = (eu_lon_min <= lon_avg <= eu_lon_max) and (eu_lat_min <= lat_avg <= eu_lat_max)
    if in_europe:
        # 跳过，不画
        continue

    color = combo_colors.get(combo_str, "#999999") 
    label = combo_str if combo_str not in used_labels else None
    used_labels.add(combo_str)

    ax.plot(lon_avg, lat_avg,
            marker='o', markersize=5,
            linestyle='None',
            color=color,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            label=label)

# 图例在下方中间
#plt.legend(
#    loc="lower center", 
#    bbox_to_anchor=(0.72, 0),
#    frameon=True,
#    fontsize=5,
#    title="Sensor",
#    title_fontsize=5,
#    ncol=1
#)

out_file = "global_map_no_europe.png"
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"地图已保存到: {out_file}")
