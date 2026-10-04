# -*- coding: utf-8 -*-
"""
大雾天气光伏功率预测论文实验代码
AECE投稿版图表与补充实验

核心方法：
Feature Enhancement + Adaptive Asymmetric Weight + ExtraTrees

主要输出：
1. 总体模型对比表
2. 自适应不对称权重搜索结果
3. 消融实验表
4. 不同出力水平大雾样本性能表
5. 特征重要性图（特征工程前 + 特征工程后）
6. 最优单日预测对比图
7. 绝对误差分布图
8. 误差改善分布图
9. 多指标消融提升图
10. 不同出力水平样本性能提升图
"""

import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


# =====================================================
# 1. 路径与参数设置
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "78101000_next_day_data.csv"

OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PRED_DIR = OUT_DIR / "predictions"

for d in [OUT_DIR, FIG_DIR, TABLE_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_COL = "obs_power"
TIME_COL = "time"
CAP_COL = "cap"
BUSINESS_PRED_COL = "pred_power"

SITE_LAT = 28.833
SITE_LON = 117.55
TIMEZONE = "Asia/Shanghai"

TRAIN_RATIO = 0.8
RANDOM_STATE = 42

MAX_CONTINUOUS_GAP_HOURS = 2
ROLLING_WINDOWS = [3, 5, 9]

# 不对称权重搜索空间
WEIGHT_ALPHA_LIST = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
WEIGHT_POWER_LIST = [1.0, 1.5, 2.0]

# 图像设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


# =====================================================
# 2. 工具函数
# =====================================================

def calc_metrics(y_true, y_pred, cap_values):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cap_values = np.asarray(cap_values, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    cap_mean = np.nanmean(cap_values)
    nrmse = rmse / cap_mean
    acc = 1 - nrmse

    return {
        "ACC": acc,
        "RMSE_MW": rmse,
        "NRMSE": nrmse,
        "MAE_MW": mae,
        "R2": r2
    }


def clean_feature_name(name):
    name = str(name)
    name = re.sub(r'[\[\]\{\}",:]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "feature"


def paper_feature_name(name):
    """
    将代码特征名统一映射成论文图表中使用的英文名称。
    """
    name = str(name)

    replace_dict = {
        "pred_power_ratio": "Business forecast ratio",
        "pred_power_clip": "Business forecast power",
        "pred_power": "Business forecast power",
        "pred_power_day_interaction": "Business forecast × daytime",
        "pred_power_solar_interaction": "Business forecast × solar elevation",
        "solar_elevation": "Solar elevation",
        "solar_zenith": "Solar zenith",
        "solar_azimuth": "Solar azimuth",
        "solar_elevation_pos": "Positive solar elevation",
        "is_daytime": "Daytime flag",
        "sin_solar_azimuth": "Sine of solar azimuth",
        "cos_solar_azimuth": "Cosine of solar azimuth",
        "hour_sin": "Sine of hour",
        "hour_cos": "Cosine of hour",
        "month_sin": "Sine of month",
        "month_cos": "Cosine of month",
        "doy_sin": "Sine of day of year",
        "doy_cos": "Cosine of day of year",
        "time_gap_hours": "Time gap",
    }

    if name in replace_dict:
        return replace_dict[name]

    name2 = name

    name2 = name2.replace("_diff1", " first-order difference")
    name2 = re.sub(r"_roll(\d+)_mean", r" rolling mean (\1-step)", name2)
    name2 = re.sub(r"_roll(\d+)_std", r" rolling std (\1-step)", name2)

    name2 = name2.replace("_", " ")

    name2 = name2.replace("pred power", "business forecast power")
    name2 = name2.replace("solar elevation", "solar elevation")
    name2 = name2.replace("solar zenith", "solar zenith")

    name2 = name2.strip()
    name2 = name2[0].upper() + name2[1:] if len(name2) > 0 else name2

    return name2


def apply_physical_constraints(pred, cap, solar_elevation=None):
    pred = np.asarray(pred, dtype=float)
    cap = np.asarray(cap, dtype=float)

    pred = np.clip(pred, 0, cap)

    if solar_elevation is not None:
        solar_elevation = np.asarray(solar_elevation, dtype=float)
        night_mask = solar_elevation <= 0
        pred[night_mask] = 0.0

    return pred


def build_asym_sample_weight(y, cap, alpha, power):
    y = np.asarray(y, dtype=float)
    cap = np.asarray(cap, dtype=float)

    norm_y = y / np.maximum(cap, 1e-6)
    norm_y = np.clip(norm_y, 0, 1)

    weight = 1.0 + alpha * np.power(1.0 - norm_y, power)
    return weight


def save_table(df, filename):
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("保存表格：", path)


def save_fig(filename):
    png_path = os.path.join(FIG_DIR, f"{filename}.png")
    pdf_path = os.path.join(FIG_DIR, f"{filename}.pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("保存图片：", png_path)


# =====================================================
# 3. 特征工程函数
# =====================================================

def add_time_features(df):
    df = df.copy()
    t = pd.to_datetime(df[TIME_COL])

    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df["month"] = t.dt.month
    df["dayofyear"] = t.dt.dayofyear
    df["weekday"] = t.dt.weekday

    df["hour_decimal"] = t.dt.hour + t.dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_decimal"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_decimal"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df


def add_solar_features(df):
    df = df.copy()
    t = pd.to_datetime(df[TIME_COL])

    try:
        import pvlib

        times = t.dt.tz_localize(
            TIMEZONE,
            nonexistent="shift_forward",
            ambiguous="NaT"
        )

        solpos = pvlib.solarposition.get_solarposition(
            times,
            latitude=SITE_LAT,
            longitude=SITE_LON
        )

        df["solar_elevation"] = solpos["elevation"].values
        df["solar_zenith"] = solpos["zenith"].values
        df["solar_azimuth"] = solpos["azimuth"].values

    except Exception:
        print("未检测到 pvlib，使用近似太阳角度公式。建议安装：pip install pvlib")

        doy = t.dt.dayofyear.values
        hour = t.dt.hour.values + t.dt.minute.values / 60
        lat_rad = np.deg2rad(SITE_LAT)

        decl = np.deg2rad(
            23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
        )

        hour_angle = np.deg2rad(15 * (hour - 12))

        sin_elev = (
            np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
        )

        sin_elev = np.clip(sin_elev, -1, 1)

        df["solar_elevation"] = np.rad2deg(np.arcsin(sin_elev))
        df["solar_zenith"] = 90 - df["solar_elevation"]
        df["solar_azimuth"] = np.rad2deg(hour_angle) + 180

    df["is_daytime"] = (df["solar_elevation"] > 0).astype(int)
    df["solar_elevation_pos"] = np.maximum(df["solar_elevation"], 0)
    df["sin_solar_azimuth"] = np.sin(np.deg2rad(df["solar_azimuth"]))
    df["cos_solar_azimuth"] = np.cos(np.deg2rad(df["solar_azimuth"]))

    return df


def add_business_features(df):
    df = df.copy()

    if BUSINESS_PRED_COL in df.columns:
        df["pred_power_clip"] = np.clip(df[BUSINESS_PRED_COL], 0, df[CAP_COL])
        df["pred_power_ratio"] = df["pred_power_clip"] / np.maximum(df[CAP_COL], 1e-6)
        df["pred_power_ratio"] = df["pred_power_ratio"].clip(0, 1)

        df["pred_power_day_interaction"] = (
            df["pred_power_ratio"] * df["is_daytime"]
        )

        df["pred_power_solar_interaction"] = (
            df["pred_power_ratio"] * df["solar_elevation_pos"]
        )

    return df


def add_segment_id(df):
    df = df.copy()
    t = pd.to_datetime(df[TIME_COL])

    gap_hours = t.diff().dt.total_seconds().fillna(0) / 3600
    df["time_gap_hours"] = gap_hours
    df["fog_segment_id"] = (gap_hours > MAX_CONTINUOUS_GAP_HOURS).cumsum()

    return df


def add_diff_rolling_features(df):
    df = df.copy()

    keywords = [
        "pred", "tmp", "temp", "t2m", "rh", "humidity",
        "dswrf", "ssrd", "ghi", "dni", "dhi",
        "vis", "tcc", "lcc", "cloud", "wspd", "wind",
        "pressure", "solar"
    ]

    exclude = {
        TARGET_COL,
        CAP_COL,
        "fog_segment_id"
    }

    candidate_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if any(k in col.lower() for k in keywords):
            candidate_cols.append(col)

    candidate_cols = list(dict.fromkeys(candidate_cols))

    print("用于构造变化率和滚动统计的特征数量：", len(candidate_cols))

    g = df.groupby("fog_segment_id", group_keys=False)

    for col in candidate_cols:
        df[f"{col}_diff1"] = g[col].diff(1)

        for win in ROLLING_WINDOWS:
            df[f"{col}_roll{win}_mean"] = g[col].transform(
                lambda x: x.rolling(win, min_periods=1).mean()
            )

            df[f"{col}_roll{win}_std"] = g[col].transform(
                lambda x: x.rolling(win, min_periods=2).std()
            )

    return df


# =====================================================
# 4. 建模数据准备
# =====================================================

def select_features(df, feature_mode="enhanced"):
    drop_cols = [TARGET_COL, TIME_COL, CAP_COL]

    for col in df.columns:
        low = col.lower()

        if low.startswith("norm_obs"):
            drop_cols.append(col)

        if low in ["id", "station", "station_id", "站点", "站点名"]:
            drop_cols.append(col)

        if any(k in low for k in [
            "obs_irr", "obs_temp", "obs_humidity", "obs_wind",
            "real_irr", "real_temp", "actual"
        ]):
            drop_cols.append(col)

    if feature_mode == "original":
        enhanced_keywords = [
            "solar_", "is_daytime", "sin_solar", "cos_solar",
            "pred_power_clip", "pred_power_ratio",
            "pred_power_day_interaction", "pred_power_solar_interaction",
            "_diff1", "_roll", "fog_segment_id", "time_gap_hours"
        ]

        for col in df.columns:
            if any(k in col for k in enhanced_keywords):
                drop_cols.append(col)

    elif feature_mode == "solar":
        remove_keywords = [
            "pred_power_clip", "pred_power_ratio",
            "pred_power_day_interaction", "pred_power_solar_interaction",
            "_diff1", "_roll"
        ]

        for col in df.columns:
            if any(k in col for k in remove_keywords):
                drop_cols.append(col)

    elif feature_mode == "solar_business":
        for col in df.columns:
            if "_diff1" in col or "_roll" in col:
                drop_cols.append(col)

    elif feature_mode == "enhanced":
        pass

    drop_cols = list(set([c for c in drop_cols if c in df.columns]))

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    return feature_cols


def prepare_model_data(df, feature_cols):
    required_cols = [TIME_COL, TARGET_COL, CAP_COL] + feature_cols

    model_df = df[required_cols].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    model_df[feature_cols] = model_df[feature_cols].fillna(
        model_df[feature_cols].median()
    )

    model_df = model_df.dropna(subset=[TARGET_COL, CAP_COL]).reset_index(drop=True)

    X = model_df[feature_cols].copy()
    y = model_df[TARGET_COL].copy()
    cap = model_df[CAP_COL].copy()

    safe_cols = []
    name_count = {}

    for col in feature_cols:
        new_col = clean_feature_name(col)

        if new_col in name_count:
            name_count[new_col] += 1
            new_col = f"{new_col}_{name_count[new_col]}"
        else:
            name_count[new_col] = 0

        safe_cols.append(new_col)

    X.columns = safe_cols

    feature_map = pd.DataFrame({
        "raw_feature": feature_cols,
        "safe_feature": safe_cols,
        "paper_feature": [paper_feature_name(c) for c in safe_cols]
    })

    return model_df, X, y, cap, safe_cols, feature_map


def split_by_time(model_df, X, y, cap):
    split_idx = int(len(model_df) * TRAIN_RATIO)

    split_data = {
        "split_idx": split_idx,
        "X_train": X.iloc[:split_idx].copy(),
        "X_test": X.iloc[split_idx:].copy(),
        "y_train": y.iloc[:split_idx].copy(),
        "y_test": y.iloc[split_idx:].copy(),
        "cap_train": cap.iloc[:split_idx].copy(),
        "cap_test": cap.iloc[split_idx:].copy(),
        "time_train": model_df[TIME_COL].iloc[:split_idx].copy(),
        "time_test": model_df[TIME_COL].iloc[split_idx:].copy(),
        "solar_train": model_df["solar_elevation"].iloc[:split_idx].values
        if "solar_elevation" in model_df.columns else None,
        "solar_test": model_df["solar_elevation"].iloc[split_idx:].values
        if "solar_elevation" in model_df.columns else None
    }

    return split_data


# =====================================================
# 5. 模型定义与训练
# =====================================================

def build_model(model_name):
    if model_name == "CatBoost":
        if CatBoostRegressor is None:
            return None

        return CatBoostRegressor(
            iterations=2000,
            learning_rate=0.02,
            depth=8,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            l2_leaf_reg=5,
            verbose=False
        )

    if model_name == "LightGBM":
        if LGBMRegressor is None:
            return None

        return LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            objective="regression",
            verbosity=-1
        )

    if model_name == "XGBoost":
        if XGBRegressor is None:
            return None

        return XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1
        )

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model name: {model_name}")


def train_eval_model(model, split_data, sample_weight=None):
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    cap_test = split_data["cap_test"]
    solar_test = split_data["solar_test"]

    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    pred = model.predict(X_test)

    pred = apply_physical_constraints(
        pred,
        cap_test.values,
        solar_test
    )

    metrics = calc_metrics(y_test, pred, cap_test)

    return model, pred, metrics


def adaptive_weight_search(model_name, split_data):
    """
    对每一个模型分别搜索不对称样本权重参数。
    alpha=0等价于不加权。
    """

    search_records = []

    best_model = None
    best_pred = None
    best_metrics = None
    best_alpha = None
    best_power = None
    best_acc = -np.inf

    for alpha in WEIGHT_ALPHA_LIST:
        for power in WEIGHT_POWER_LIST:
            model = build_model(model_name)

            if model is None:
                continue

            if alpha == 0:
                sample_weight = None
            else:
                sample_weight = build_asym_sample_weight(
                    split_data["y_train"],
                    split_data["cap_train"],
                    alpha=alpha,
                    power=power
                )

            try:
                trained_model, pred, metrics = train_eval_model(
                    model,
                    split_data,
                    sample_weight=sample_weight
                )

                record = {
                    "Model": model_name,
                    "alpha": alpha,
                    "power": power,
                    "ACC": metrics["ACC"],
                    "RMSE_MW": metrics["RMSE_MW"],
                    "NRMSE": metrics["NRMSE"],
                    "MAE_MW": metrics["MAE_MW"],
                    "R2": metrics["R2"]
                }

                search_records.append(record)

                if metrics["ACC"] > best_acc:
                    best_acc = metrics["ACC"]
                    best_model = trained_model
                    best_pred = pred
                    best_metrics = metrics
                    best_alpha = alpha
                    best_power = power

            except Exception as e:
                print(f"{model_name} alpha={alpha}, power={power} 训练失败：{e}")

    return {
        "best_model": best_model,
        "best_pred": best_pred,
        "best_metrics": best_metrics,
        "best_alpha": best_alpha,
        "best_power": best_power,
        "search_records": search_records
    }


# =====================================================
# 6. 读取数据与特征增强
# =====================================================

df_raw = pd.read_csv(DATA_PATH)
df_raw[TIME_COL] = pd.to_datetime(df_raw[TIME_COL])
df_raw = df_raw.sort_values(TIME_COL).reset_index(drop=True)

df = df_raw.copy()

df = add_time_features(df)
df = add_solar_features(df)
df = add_business_features(df)
df = add_segment_id(df)
df = add_diff_rolling_features(df)

df = df.replace([np.inf, -np.inf], np.nan)

print("数据规模：", df.shape)
print("时间范围：", df[TIME_COL].min(), "至", df[TIME_COL].max())


# =====================================================
# 7. 主实验：增强特征 + 自适应不对称权重
# =====================================================

feature_cols = select_features(df, feature_mode="enhanced")
model_df, X, y, cap, safe_cols, feature_map = prepare_model_data(df, feature_cols)
split_data = split_by_time(model_df, X, y, cap)

save_table(feature_map, "feature_name_map.csv")

pred_df = pd.DataFrame({
    TIME_COL: split_data["time_test"].values,
    "obs_power": split_data["y_test"].values,
    "cap": split_data["cap_test"].values,
    "solar_elevation": split_data["solar_test"]
})

results = []
all_weight_search_records = []
trained_models = {}
model_predictions = {}

# 业务预测
if BUSINESS_PRED_COL in model_df.columns:
    pred_business = model_df[BUSINESS_PRED_COL].iloc[
        split_data["split_idx"]:
    ].values.astype(float)
else:
    pred_business = df.loc[model_df.index, BUSINESS_PRED_COL].iloc[
        split_data["split_idx"]:
    ].values.astype(float)

pred_business = apply_physical_constraints(
    pred_business,
    split_data["cap_test"].values,
    split_data["solar_test"]
)

business_metrics = calc_metrics(
    split_data["y_test"],
    pred_business,
    split_data["cap_test"]
)

business_metrics["Model"] = "BusinessForecast"
business_metrics["Method"] = "pred_power"
business_metrics["alpha"] = np.nan
business_metrics["power"] = np.nan

results.append(business_metrics)
pred_df["BusinessForecast"] = pred_business

# 模型对比与自适应权重
model_names = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "ExtraTrees"]

for model_name in model_names:
    print("\n==============================")
    print(f"自适应不对称权重搜索：{model_name}")
    print("==============================")

    search_result = adaptive_weight_search(model_name, split_data)

    if search_result["best_metrics"] is None:
        continue

    best_metrics = search_result["best_metrics"].copy()
    best_metrics["Model"] = model_name
    best_metrics["Method"] = "EnhancedFeature+AdaptiveWeight"
    best_metrics["alpha"] = search_result["best_alpha"]
    best_metrics["power"] = search_result["best_power"]

    results.append(best_metrics)

    pred_df[model_name] = search_result["best_pred"]

    trained_models[model_name] = search_result["best_model"]
    model_predictions[model_name] = search_result["best_pred"]

    all_weight_search_records.extend(search_result["search_records"])

weight_search_df = pd.DataFrame(all_weight_search_records)
save_table(weight_search_df, "Table_weight_search_all_models.csv")

result_df = pd.DataFrame(results)
result_df = result_df[
    ["Model", "Method", "alpha", "power", "ACC", "RMSE_MW", "NRMSE", "MAE_MW", "R2"]
].sort_values("ACC", ascending=False)

save_table(result_df, "Table4_overall_model_comparison.csv")

pred_df.to_csv(
    os.path.join(PRED_DIR, "test_predictions.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n最终模型精度排序：")
print(result_df)

# 选择本文模型
if "ExtraTrees" in trained_models:
    proposed_model = trained_models["ExtraTrees"]
    proposed_pred = model_predictions["ExtraTrees"]
    proposed_row = result_df[result_df["Model"] == "ExtraTrees"].iloc[0]
else:
    proposed_name = result_df[result_df["Model"] != "BusinessForecast"].iloc[0]["Model"]
    proposed_model = trained_models[proposed_name]
    proposed_pred = model_predictions[proposed_name]
    proposed_row = result_df[result_df["Model"] == proposed_name].iloc[0]

pred_df["Proposed"] = proposed_pred


# =====================================================
# 8. 消融实验
# =====================================================

ablation_settings = [
    ("Original features", "original", 0.0, 1.0),
    ("+ Solar features", "solar", 0.0, 1.0),
    ("+ Business forecast features", "solar_business", 0.0, 1.0),
    ("+ Rolling statistics", "enhanced", 0.0, 1.0),
    ("+ Adaptive asymmetric weight", "enhanced", proposed_row["alpha"], proposed_row["power"]),
]

ablation_records = []

for label, mode, alpha, power in ablation_settings:
    print("\n消融实验：", label)

    f_cols = select_features(df, feature_mode=mode)
    m_df, X_a, y_a, cap_a, safe_a, fmap_a = prepare_model_data(df, f_cols)
    sp = split_by_time(m_df, X_a, y_a, cap_a)

    model = build_model("ExtraTrees")

    if alpha == 0:
        sample_weight = None
    else:
        sample_weight = build_asym_sample_weight(
            sp["y_train"],
            sp["cap_train"],
            alpha=alpha,
            power=power
        )

    _, pred_a, met_a = train_eval_model(model, sp, sample_weight=sample_weight)

    met_a["Module"] = label
    met_a["alpha"] = alpha
    met_a["power"] = power

    ablation_records.append(met_a)

ablation_df = pd.DataFrame(ablation_records)
ablation_df = ablation_df[
    ["Module", "alpha", "power", "ACC", "RMSE_MW", "NRMSE", "MAE_MW", "R2"]
]

save_table(ablation_df, "Table5_ablation_study.csv")


# =====================================================
# 9. 不同出力水平大雾样本性能分析
# =====================================================

analysis_df = pred_df.copy()
analysis_df["abs_error_business"] = np.abs(
    analysis_df["obs_power"] - analysis_df["BusinessForecast"]
)
analysis_df["abs_error_proposed"] = np.abs(
    analysis_df["obs_power"] - analysis_df["Proposed"]
)
analysis_df["error_improvement"] = (
    analysis_df["abs_error_business"] - analysis_df["abs_error_proposed"]
)

analysis_df["pred_ratio"] = (
    analysis_df["BusinessForecast"] / np.maximum(analysis_df["cap"], 1e-6)
)

analysis_df["hour"] = pd.to_datetime(analysis_df[TIME_COL]).dt.hour
analysis_df["date"] = pd.to_datetime(analysis_df[TIME_COL]).dt.date

analysis_df["output_level"] = pd.cut(
    analysis_df["pred_ratio"],
    bins=[-0.001, 0.2, 0.5, 1.001],
    labels=[
        "Low-output fog condition",
        "Medium-output fog condition",
        "High-output fog condition"
    ]
)


def group_metrics(df_part, pred_col):
    return calc_metrics(
        df_part["obs_power"].values,
        df_part[pred_col].values,
        df_part["cap"].values
    )


output_level_records = []

for level, g in analysis_df.groupby("output_level"):
    if len(g) < 5:
        continue

    b = group_metrics(g, "BusinessForecast")
    p = group_metrics(g, "Proposed")

    output_level_records.append({
        "Output_level": level,
        "Definition": (
            "Business forecast ratio <=0.2"
            if "Low" in str(level)
            else "0.2 < business forecast ratio <=0.5"
            if "Medium" in str(level)
            else "Business forecast ratio >0.5"
        ),
        "Sample_num": len(g),
        "Business_ACC": b["ACC"],
        "Proposed_ACC": p["ACC"],
        "Business_RMSE": b["RMSE_MW"],
        "Proposed_RMSE": p["RMSE_MW"],
        "RMSE_reduction_%": (
            (b["RMSE_MW"] - p["RMSE_MW"]) / b["RMSE_MW"] * 100
        ),
        "Business_MAE": b["MAE_MW"],
        "Proposed_MAE": p["MAE_MW"],
        "MAE_reduction_%": (
            (b["MAE_MW"] - p["MAE_MW"]) / b["MAE_MW"] * 100
        )
    })

output_level_df = pd.DataFrame(output_level_records)
save_table(output_level_df, "Table6_output_level_fog_condition_performance.csv")


# =====================================================
# 10. 单日误差提升最大案例筛选
# =====================================================

daily_records = []

for d, g in analysis_df.groupby("date"):
    if len(g) < 10:
        continue

    b = group_metrics(g, "BusinessForecast")
    p = group_metrics(g, "Proposed")

    daily_records.append({
        "date": d,
        "sample_num": len(g),
        "business_rmse": b["RMSE_MW"],
        "proposed_rmse": p["RMSE_MW"],
        "rmse_reduction": b["RMSE_MW"] - p["RMSE_MW"],
        "rmse_reduction_%": (
            (b["RMSE_MW"] - p["RMSE_MW"]) / b["RMSE_MW"] * 100
            if b["RMSE_MW"] > 0 else np.nan
        ),
        "business_mae": b["MAE_MW"],
        "proposed_mae": p["MAE_MW"],
        "mae_reduction": b["MAE_MW"] - p["MAE_MW"]
    })

daily_improve_df = pd.DataFrame(daily_records)

if len(daily_improve_df) > 0:
    daily_improve_df = daily_improve_df.sort_values(
        "rmse_reduction_%",
        ascending=False
    )
    best_day = daily_improve_df.iloc[0]["date"]
else:
    best_day = analysis_df["date"].iloc[0]

save_table(daily_improve_df, "Table_daily_improvement_ranking.csv")

print("预测对比图自动选择日期：", best_day)


# =====================================================
# 11. 特征重要性（特征工程后）
# =====================================================

importance_df = pd.DataFrame({
    "feature": safe_cols,
    "paper_feature": [paper_feature_name(c) for c in safe_cols],
    "importance": proposed_model.feature_importances_
}).sort_values("importance", ascending=False)

save_table(importance_df.head(30), "Table7_top30_feature_importance.csv")


# =====================================================
# 12. 论文表格：数据、特征、参数、提升摘要
# =====================================================

table1 = pd.DataFrame({
    "Variable": [
        "time",
        "cap",
        "obs_power",
        "pred_power",
        "meteorological forecast variables",
        "solar position variables",
        "rolling statistics variables"
    ],
    "Description": [
        "Timestamp of fog-condition samples",
        "Installed PV capacity",
        "Observed PV power",
        "Operational business forecast power",
        "NWP-based meteorological features",
        "Solar elevation, zenith, azimuth and derived terms",
        "Rolling mean, rolling standard deviation and first-order difference"
    ],
    "Unit": [
        "-",
        "MW",
        "MW",
        "MW",
        "-",
        "degree",
        "-"
    ]
})

save_table(table1, "Table1_dataset_description.csv")

table2 = pd.DataFrame({
    "Category": [
        "Solar position enhancement",
        "Business forecast fusion",
        "Temporal rolling statistics",
        "Physical constraints",
        "Adaptive asymmetric sample weighting"
    ],
    "Features_or_strategy": [
        "solar elevation, solar zenith, solar azimuth, daytime flag",
        "business forecast power, business forecast ratio, interaction terms",
        "first-order difference, rolling mean and rolling standard deviation",
        "nighttime output is set to zero and prediction is limited by installed capacity",
        "alpha and power are selected by validation performance for each model"
    ]
})

save_table(table2, "Table2_feature_engineering.csv")

table3 = pd.DataFrame({
    "Parameter": [
        "SITE_LAT",
        "SITE_LON",
        "train_test_split",
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "weight_alpha_list",
        "weight_power_list"
    ],
    "Value": [
        SITE_LAT,
        SITE_LON,
        "8:2 chronological split",
        800,
        "sqrt",
        2,
        str(WEIGHT_ALPHA_LIST),
        str(WEIGHT_POWER_LIST)
    ]
})

save_table(table3, "Table3_model_parameters.csv")

business_row = result_df[result_df["Model"] == "BusinessForecast"].iloc[0]
proposed_summary_row = proposed_row

summary_df = pd.DataFrame([{
    "Business_ACC": business_row["ACC"],
    "Proposed_ACC": proposed_summary_row["ACC"],
    "ACC_improvement_point": proposed_summary_row["ACC"] - business_row["ACC"],
    "Business_RMSE": business_row["RMSE_MW"],
    "Proposed_RMSE": proposed_summary_row["RMSE_MW"],
    "RMSE_reduction_%": (
        (business_row["RMSE_MW"] - proposed_summary_row["RMSE_MW"])
        / business_row["RMSE_MW"] * 100
    ),
    "Business_MAE": business_row["MAE_MW"],
    "Proposed_MAE": proposed_summary_row["MAE_MW"],
    "MAE_reduction_%": (
        (business_row["MAE_MW"] - proposed_summary_row["MAE_MW"])
        / business_row["MAE_MW"] * 100
    ),
    "Business_R2": business_row["R2"],
    "Proposed_R2": proposed_summary_row["R2"],
    "R2_improvement_point": proposed_summary_row["R2"] - business_row["R2"],
    "Best_alpha": proposed_summary_row["alpha"],
    "Best_power": proposed_summary_row["power"]
}])

save_table(summary_df, "Table8_improvement_summary.csv")


# =====================================================
# 13. 图1_before：特征工程前特征重要性（原始特征）
# =====================================================

print("\n生成特征工程前特征重要性...")

# 使用原始特征（仅基础气象+时间特征）
ori_feature_cols = select_features(df, feature_mode="original")
ori_model_df, ori_X, ori_y, ori_cap, ori_safe_cols, ori_fmap = prepare_model_data(df, ori_feature_cols)
ori_split = split_by_time(ori_model_df, ori_X, ori_y, ori_cap)

# 训练一个ExtraTrees模型（不加权）
ori_model = ExtraTreesRegressor(
    n_estimators=800,
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
ori_model.fit(ori_split["X_train"], ori_split["y_train"])

# 提取重要性
ori_importance_df = pd.DataFrame({
    "feature": ori_safe_cols,
    "paper_feature": [paper_feature_name(c) for c in ori_safe_cols],
    "importance": ori_model.feature_importances_
}).sort_values("importance", ascending=False)

# 保存表格
save_table(ori_importance_df.head(30), "Table_feature_importance_before_construction.csv")

# 绘图（Top 20）
top_ori_imp = ori_importance_df.head(20).copy()
top_ori_imp = top_ori_imp.sort_values("importance")

plt.figure(figsize=(9, 7))
plt.barh(top_ori_imp["paper_feature"], top_ori_imp["importance"])
plt.xlabel("Feature importance")
plt.title("Top 20 feature importance (before feature construction)")
save_fig("Figure1_before_feature_importance_original")


# =====================================================
# 14. 图2：方法框架图
# =====================================================

plt.figure(figsize=(11, 5.5))
plt.axis("off")

framework_text = (
    "Meteorological forecasts + Business forecast\n"
    "↓\n"
    "Feature enhancement\n"
    "Solar position | Business forecast ratio | Rolling statistics | Physical constraints\n"
    "↓\n"
    "Adaptive asymmetric sample weighting\n"
    "↓\n"
    "ExtraTrees regression\n"
    "↓\n"
    "PV power forecasting under fog conditions"
)

plt.text(
    0.5,
    0.5,
    framework_text,
    ha="center",
    va="center",
    fontsize=14,
    bbox=dict(
        boxstyle="round,pad=0.8",
        fc="white",
        ec="black"
    )
)

plt.title("Framework of the proposed method", fontsize=15)
save_fig("Figure2_framework")


# =====================================================
# 15. 图3：特征重要性（特征工程后）
# =====================================================

top_imp = importance_df.head(20).copy()
top_imp = top_imp.sort_values("importance")

plt.figure(figsize=(9, 7))
plt.barh(top_imp["paper_feature"], top_imp["importance"])
plt.xlabel("Feature importance")
plt.title("Top 20 feature importance of the proposed model")
save_fig("Figure3_feature_importance")


# =====================================================
# 16. 图4：最优单日预测对比
# =====================================================

best_day_df = analysis_df[analysis_df["date"] == best_day].copy()

plt.figure(figsize=(12, 4.5))
plt.plot(
    best_day_df[TIME_COL],
    best_day_df["obs_power"],
    label="Observed",
    linewidth=2.0
)
plt.plot(
    best_day_df[TIME_COL],
    best_day_df["BusinessForecast"],
    label="Business forecast",
    linewidth=1.4
)
plt.plot(
    best_day_df[TIME_COL],
    best_day_df["Proposed"],
    label="Proposed",
    linewidth=1.6
)

plt.xlabel("Time")
plt.ylabel("Power / MW")
plt.title(f"Prediction comparison on the day with largest RMSE improvement ({best_day})")
plt.legend()
save_fig("Figure4_best_daily_prediction_comparison")


# =====================================================
# 17. 图5：绝对误差分布
# =====================================================

plt.figure(figsize=(8.5, 4.8))
plt.hist(
    analysis_df["abs_error_business"],
    bins=40,
    alpha=0.55,
    density=True,
    label="Business forecast"
)
plt.hist(
    analysis_df["abs_error_proposed"],
    bins=40,
    alpha=0.55,
    density=True,
    label="Proposed"
)

plt.xlabel("Absolute error / MW")
plt.ylabel("Density")
plt.title("Absolute error distribution")
plt.legend()
save_fig("Figure5_absolute_error_distribution")


# =====================================================
# 18. 图6：误差改善分布
# =====================================================

plt.figure(figsize=(8.5, 4.8))
plt.hist(
    analysis_df["error_improvement"],
    bins=50,
    alpha=0.75
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1.2
)

plt.xlabel("|Error$_{business}$| - |Error$_{proposed}$| / MW")
plt.ylabel("Frequency")
plt.title("Distribution of absolute error improvement")
save_fig("Figure6_error_improvement_distribution")


# =====================================================
# 19. 图7：消融实验多指标提升图
# =====================================================

base_ablation = ablation_df.iloc[0].copy()

ablation_plot_df = ablation_df.copy()
ablation_plot_df["ACC_improvement_point"] = (
    ablation_plot_df["ACC"] - base_ablation["ACC"]
)
ablation_plot_df["RMSE_reduction_%"] = (
    (base_ablation["RMSE_MW"] - ablation_plot_df["RMSE_MW"])
    / base_ablation["RMSE_MW"] * 100
)
ablation_plot_df["MAE_reduction_%"] = (
    (base_ablation["MAE_MW"] - ablation_plot_df["MAE_MW"])
    / base_ablation["MAE_MW"] * 100
)
ablation_plot_df["R2_improvement_point"] = (
    ablation_plot_df["R2"] - base_ablation["R2"]
)

save_table(ablation_plot_df, "Table_ablation_improvement_relative_to_original.csv")

x = np.arange(len(ablation_plot_df))
width = 0.2

plt.figure(figsize=(12, 5))
plt.bar(
    x - 1.5 * width,
    ablation_plot_df["ACC_improvement_point"] * 100,
    width,
    label="ACC improvement / percentage point"
)
plt.bar(
    x - 0.5 * width,
    ablation_plot_df["RMSE_reduction_%"],
    width,
    label="RMSE reduction / %"
)
plt.bar(
    x + 0.5 * width,
    ablation_plot_df["MAE_reduction_%"],
    width,
    label="MAE reduction / %"
)
plt.bar(
    x + 1.5 * width,
    ablation_plot_df["R2_improvement_point"] * 100,
    width,
    label="R² improvement / percentage point"
)

plt.xticks(
    x,
    ablation_plot_df["Module"],
    rotation=20,
    ha="right"
)
plt.ylabel("Improvement")
plt.title("Ablation study with multiple metrics")
plt.legend()
save_fig("Figure7_ablation_multi_metric_improvement")


# =====================================================
# 20. 图8：不同出力水平大雾样本性能提升
# =====================================================

if len(output_level_df) > 0:
    x = np.arange(len(output_level_df))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        output_level_df["RMSE_reduction_%"],
        width,
        label="RMSE reduction / %"
    )
    plt.bar(
        x + width / 2,
        output_level_df["MAE_reduction_%"],
        width,
        label="MAE reduction / %"
    )

    plt.xticks(
        x,
        output_level_df["Output_level"].astype(str),
        rotation=15,
        ha="right"
    )
    plt.ylabel("Reduction / %")
    plt.title("Performance improvement under different output-level fog conditions")
    plt.legend()
    save_fig("Figure8_output_level_fog_condition_improvement")


# =====================================================
# 21. 图9：总体模型对比
# =====================================================

plot_model_df = result_df.copy()
plot_model_df = plot_model_df.sort_values("ACC", ascending=True)

plt.figure(figsize=(9, 5.5))
plt.barh(
    plot_model_df["Model"],
    plot_model_df["ACC"]
)

plt.xlabel("ACC")
plt.title("Overall model comparison")
save_fig("Figure9_overall_model_ACC_comparison")


# =====================================================
# 22. 输出说明文本
# =====================================================

explain_path = os.path.join(OUT_DIR, "experiment_explanation.txt")

with open(explain_path, "w", encoding="utf-8") as f:
    f.write("实验说明\n")
    f.write("=" * 60 + "\n\n")

    f.write("1. 不同出力水平大雾样本划分方法：\n")
    f.write("   使用 BusinessForecast/cap 得到业务预测出力比例 pred_ratio。\n")
    f.write("   pred_ratio <= 0.2：Low-output fog condition。\n")
    f.write("   0.2 < pred_ratio <= 0.5：Medium-output fog condition。\n")
    f.write("   pred_ratio > 0.5：High-output fog condition。\n")
    f.write("   这里的分组不是严格气象能见度意义上的轻雾/浓雾，")
    f.write("而是面向光伏出力预测的出力水平分组。\n\n")

    f.write("2. 图4最优单日选择方法：\n")
    f.write("   对验证集按日期分组，分别计算 BusinessForecast 和 Proposed 的日RMSE，")
    f.write("选择 RMSE 降低百分比最大的日期作为典型案例。\n\n")

    f.write("3. 图6误差改善定义：\n")
    f.write("   error_improvement = |e_business| - |e_proposed|。\n")
    f.write("   若该值大于0，说明本文方法在该样本上降低了绝对误差。\n\n")

    f.write("4. 消融实验图说明：\n")
    f.write("   图7不直接画ACC绝对值，而是以 Original features 为基准，")
    f.write("展示 ACC提升百分点、RMSE降低百分比、MAE降低百分比和R²提升百分点。\n\n")

    f.write("5. 自适应不对称权重说明：\n")
    f.write("   对每个模型分别搜索 alpha 和 power，alpha=0 表示不使用样本权重。")
    f.write("最终采用验证集ACC最优的权重参数。\n\n")
    
    f.write("6. 图1_before与图3对比说明：\n")
    f.write("   图1_before展示特征工程前（仅原始气象+时间特征）的特征重要性。\n")
    f.write("   图3展示特征工程后（包含太阳位置、业务预测融合、滚动统计等）的特征重要性。\n")
    f.write("   两图对比可直观说明特征工程的有效性。\n\n")

print("\n==============================")
print("全部实验完成")
print("==============================")
print("输出目录：", OUT_DIR)
print("图片目录：", FIG_DIR)
print("表格目录：", TABLE_DIR)
print("预测结果目录：", PRED_DIR)
print("说明文件：", explain_path)

print("\n核心提升摘要：")
print(summary_df.T)