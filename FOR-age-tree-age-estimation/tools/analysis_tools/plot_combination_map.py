import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

# ========== 1) 原始数据 ========== 
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

# ========== 2) 按国家合并传感器类型 ========== 
region_data = {}
for (name, country, sensor, (lat, lon)) in datasets_raw:
    if country not in region_data:
        region_data[country] = {"coords": [], "sensors": set()}
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

# ========== 3) 传感器组合 -> 颜色 (可选) ==========
combo_colors = {
    "ULS":          "#e41a1c",
    "MLS":          "#377eb8",
    "TLS":          "#4daf4a",
    "MLS/TLS":      "#984ea3",
    "MLS/ULS":      "#ff7f00",
    "TLS/ULS":      "#984ea3",
    "MLS/TLS/ULS":  "#a65628"
}

# ========== 4) 自定义地图范围 ==========
lon_min, lon_max = -55, 185
lat_min, lat_max = -45, 70

# ========== 5) 绘图: 极简 + 浅灰填充陆地 ==========
fig = plt.figure(figsize=(7, 4), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

# 1) 用浅灰色填充陆地
#    resolution='110m' 可能显示得更快, 也可改成 '50m' or '10m' 视你的环境
land_feature = cfeature.NaturalEarthFeature(
    "physical", "land", "110m",
    edgecolor="none",
    facecolor="#f0f0f0"  # 浅灰
)
ax.add_feature(land_feature, zorder=0)

# 2) 海岸线、国界用略深的灰(如 #888888)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="#888888")
ax.add_feature(cfeature.BORDERS, linewidth=0.5, color="#888888")

# 3) 绘制回归线、极圈、赤道，用更浅的银色
special_lats = [-66.5, -23.5, 0, 23.5, 66.5]
for lat_line in special_lats:
    if lat_min <= lat_line <= lat_max:
        ax.plot([lon_min, lon_max], [lat_line, lat_line],
                transform=ccrs.PlateCarree(),
                linestyle='--', linewidth=0.8, color="#bbbbbb")

# 4) 绘制圆点(不同传感器组合 -> 不同颜色)
used_labels = set()
for (country, combo_str, (lat_avg, lon_avg)) in combined_data:
    color = combo_colors.get(combo_str, "#999999")
    label = combo_str if combo_str not in used_labels else None
    used_labels.add(combo_str)

    ax.plot(lon_avg, lat_avg,
            marker='o',
            markersize=5,
            linestyle='None',
            color=color,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            label=label)

# 5) 去掉投影外框
ax.set_frame_on(False)

# 若要隐藏所有轴脊线(有时 Cartopy 不把边框当 spines):
# for spine in ax.spines.values():
#     spine.set_visible(False)

# ========== 6) 保存 ==========
plt.savefig("forinstance_combination_map_gray.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("地图已保存: forinstance_combination_map_gray.png")
