"""
Kapsamlı Senaryo Üretim Sistemi
================================
3 farazi coin (KAPCOIN, METUCOIN, TAMCOIN) için sonraki 1 yıl (2027-03-16 → 2028-03-15)
tahmin verisi üretir. 10 farklı piyasa rejimi, çoklu faz yapısı ve seed tabanlı
deterministik üretim ile milyarlarca benzersiz senaryo üretilebilir.

Kullanım:
    python generate_predictions.py              # 1000 senaryo üret
    python generate_predictions.py --count 5000 # 5000 senaryo üret
    python generate_predictions.py --count 100 --start-id 2001  # ID 2001'den başla
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────── Sabitler ────────────────────────────────────────

DATA_DIR = Path(__file__).parent
CNLIB_DATA = None  # Otomatik bulunur

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
PREDICTION_DAYS = 365
PREDICTION_START = "2027-03-16"
PREDICTION_END = "2028-03-15"

OUTPUT_DIR = DATA_DIR / "predicted_data" / "scenarios"
METADATA_FILE = DATA_DIR / "predicted_data" / "metadata.csv"
CONFIG_FILE = DATA_DIR / "predicted_data" / "generator_config.json"


# ─────────────────────────── Rejim Tanımları ─────────────────────────────────


@dataclass
class RegimeParams:
    """Bir piyasa rejiminin parametre aralıkları."""
    name: str
    daily_drift_min: float       # Günlük ortalama getiri alt sınırı
    daily_drift_max: float       # Günlük ortalama getiri üst sınırı
    daily_vol_min: float         # Günlük volatilite alt sınırı
    daily_vol_max: float         # Günlük volatilite üst sınırı
    volume_multiplier_min: float # Hacim çarpanı alt
    volume_multiplier_max: float # Hacim çarpanı üst
    description: str = ""
    # Özel davranışlar
    is_two_phase: bool = False   # pump_dump gibi 2 fazlı rejimler için
    min_floor_pct: Optional[float] = None  # death_spiral: minimum fiyat yüzdesi


REGIMES = {
    "steady_bull": RegimeParams(
        name="steady_bull",
        daily_drift_min=0.0015, daily_drift_max=0.0040,
        daily_vol_min=0.015, daily_vol_max=0.030,
        volume_multiplier_min=0.8, volume_multiplier_max=1.5,
        description="Kararlı yükseliş trendi"
    ),
    "explosive_bull": RegimeParams(
        name="explosive_bull",
        daily_drift_min=0.004, daily_drift_max=0.010,
        daily_vol_min=0.030, daily_vol_max=0.060,
        volume_multiplier_min=1.5, volume_multiplier_max=3.0,
        description="Parabolik ralli, çok yüksek hacim"
    ),
    "sideways": RegimeParams(
        name="sideways",
        daily_drift_min=-0.0005, daily_drift_max=0.0005,
        daily_vol_min=0.010, daily_vol_max=0.025,
        volume_multiplier_min=0.5, volume_multiplier_max=1.0,
        description="Yatay seyir, düşük hacim"
    ),
    "steady_bear": RegimeParams(
        name="steady_bear",
        daily_drift_min=-0.0040, daily_drift_max=-0.0015,
        daily_vol_min=0.015, daily_vol_max=0.035,
        volume_multiplier_min=0.8, volume_multiplier_max=1.5,
        description="Kontrollü düşüş trendi"
    ),
    "capitulation": RegimeParams(
        name="capitulation",
        daily_drift_min=-0.015, daily_drift_max=-0.005,
        daily_vol_min=0.040, daily_vol_max=0.080,
        volume_multiplier_min=2.0, volume_multiplier_max=4.0,
        description="Panik satışı, yüksek hacim ve volatilite"
    ),
    "death_spiral": RegimeParams(
        name="death_spiral",
        daily_drift_min=-0.030, daily_drift_max=-0.010,
        daily_vol_min=0.050, daily_vol_max=0.100,
        volume_multiplier_min=1.0, volume_multiplier_max=2.5,
        description="Sıfıra yaklaşma, proje çöküşü",
        min_floor_pct=0.001  # Fiyat başlangıcın %0.1'ine kadar düşebilir
    ),
    "recovery": RegimeParams(
        name="recovery",
        daily_drift_min=0.002, daily_drift_max=0.006,
        daily_vol_min=0.025, daily_vol_max=0.050,
        volume_multiplier_min=1.2, volume_multiplier_max=2.0,
        description="Çöküş sonrası toparlanma"
    ),
    "volatile_chop": RegimeParams(
        name="volatile_chop",
        daily_drift_min=-0.0005, daily_drift_max=0.0005,
        daily_vol_min=0.040, daily_vol_max=0.070,
        volume_multiplier_min=1.5, volume_multiplier_max=3.0,
        description="Yoğun dalgalanma, net yatay"
    ),
    "pump_dump": RegimeParams(
        name="pump_dump",
        daily_drift_min=0.005, daily_drift_max=0.020,
        daily_vol_min=0.050, daily_vol_max=0.100,
        volume_multiplier_min=3.0, volume_multiplier_max=5.0,
        description="Ani yükseliş sonrası sert düşüş",
        is_two_phase=True
    ),
    "accumulation": RegimeParams(
        name="accumulation",
        daily_drift_min=0.0002, daily_drift_max=0.0010,
        daily_vol_min=0.008, daily_vol_max=0.015,
        volume_multiplier_min=0.3, volume_multiplier_max=0.7,
        description="Düşük hacim sıkışma, breakout öncesi"
    ),
}

# Rejim geçiş olasılıkları: Hangi rejimden hangi rejime geçiş mantıklı?
REGIME_TRANSITIONS = {
    "steady_bull":    ["sideways", "explosive_bull", "steady_bear", "volatile_chop"],
    "explosive_bull": ["capitulation", "steady_bear", "volatile_chop", "sideways"],
    "sideways":       ["steady_bull", "steady_bear", "accumulation", "volatile_chop", "explosive_bull"],
    "steady_bear":    ["sideways", "capitulation", "recovery", "accumulation"],
    "capitulation":   ["recovery", "death_spiral", "sideways", "volatile_chop"],
    "death_spiral":   ["recovery", "sideways"],  # Ya toparlanır ya da kalır
    "recovery":       ["steady_bull", "sideways", "volatile_chop"],
    "volatile_chop":  ["steady_bull", "steady_bear", "sideways", "capitulation"],
    "pump_dump":      ["capitulation", "steady_bear", "sideways", "volatile_chop"],
    "accumulation":   ["steady_bull", "explosive_bull", "sideways"],
}


# ─────────────────────── Veri Yükleyici ──────────────────────────────────────


def find_cnlib_data() -> Path:
    """cnlib paket verisinin yerini bulur."""
    try:
        import cnlib
        pkg_dir = Path(cnlib.__file__).parent / "data"
        if pkg_dir.exists():
            return pkg_dir
    except ImportError:
        pass

    # Fallback: site-packages'da ara
    for p in sys.path:
        candidate = Path(p) / "cnlib" / "data"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "cnlib data dizini bulunamadı. Lütfen `pip install cnlib` yapın."
    )


def load_training_data() -> dict[str, pd.DataFrame]:
    """Orijinal training verisini yükler."""
    data_dir = find_cnlib_data()
    result = {}
    for coin in COINS:
        fpath = data_dir / f"{coin}.parquet"
        df = pd.read_parquet(fpath)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        result[coin] = df
    return result


# ──────────────────── Senaryo Üretici ────────────────────────────────────────


@dataclass
class PhaseSpec:
    """Bir faz tanımı."""
    regime_name: str
    days: int
    drift: float
    volatility: float
    volume_multiplier: float


@dataclass
class ScenarioSpec:
    """Tam bir senaryo tanımı (3 coin)."""
    scenario_id: int
    seed: int
    coin_phases: dict[str, list[PhaseSpec]] = field(default_factory=dict)


class ScenarioGenerator:
    """
    Deterministik, seed tabanlı senaryo üretici.

    Her seed benzersiz bir senaryo üretir. Aynı seed = aynı sonuç.
    """

    def __init__(self, training_data: dict[str, pd.DataFrame]):
        self.training_data = training_data
        # Her coin'in son verilerinden istatistik çıkar
        self.coin_stats = {}
        for coin, df in training_data.items():
            last_price = float(df["Close"].iloc[-1])
            avg_volume = float(df["Volume"].tail(90).mean())
            daily_ret = df["Close"].pct_change().dropna()
            hist_vol = float(daily_ret.std())
            h_c_ratio = float((df["High"] / df["Close"]).mean())
            l_c_ratio = float((df["Low"] / df["Close"]).mean())
            o_c_ratio = float((df["Open"] / df["Close"]).mean())
            self.coin_stats[coin] = {
                "last_price": last_price,
                "avg_volume": avg_volume,
                "hist_vol": hist_vol,
                "h_c_ratio": h_c_ratio,
                "l_c_ratio": l_c_ratio,
                "o_c_ratio": o_c_ratio,
            }

    def _pick_phases(self, rng: np.random.Generator, coin: str) -> list[PhaseSpec]:
        """Bir coin için 2-4 fazlı yıl planı üretir."""
        num_phases = rng.choice([2, 3, 3, 4])  # 3 faz daha olası

        # İlk rejim: coin'in mevcut durumuna göre ağırlıklı seçim
        # Son 90 gün trendi analiz et
        df = self.training_data[coin]
        last_90 = df.tail(90)
        trend = (last_90["Close"].iloc[-1] / last_90["Close"].iloc[0]) - 1

        # Mevcut trende göre başlangıç rejimi olasılıkları
        if trend < -0.3:  # Sert düşüş: recovery veya devam olasılıkları
            start_weights = {
                "recovery": 0.25, "sideways": 0.15, "steady_bear": 0.15,
                "capitulation": 0.10, "volatile_chop": 0.15,
                "accumulation": 0.10, "steady_bull": 0.05,
                "death_spiral": 0.05,
            }
        elif trend < -0.1:  # Hafif düşüş
            start_weights = {
                "sideways": 0.20, "recovery": 0.15, "steady_bear": 0.15,
                "accumulation": 0.15, "volatile_chop": 0.15,
                "steady_bull": 0.10, "capitulation": 0.05,
                "death_spiral": 0.05,
            }
        elif trend < 0.1:  # Yatay
            start_weights = {
                "sideways": 0.20, "steady_bull": 0.15, "steady_bear": 0.15,
                "accumulation": 0.15, "volatile_chop": 0.15,
                "explosive_bull": 0.05, "recovery": 0.05,
                "capitulation": 0.05, "pump_dump": 0.05,
            }
        else:  # Yükseliş
            start_weights = {
                "steady_bull": 0.25, "explosive_bull": 0.15, "sideways": 0.15,
                "volatile_chop": 0.10, "steady_bear": 0.10,
                "pump_dump": 0.10, "accumulation": 0.10,
                "capitulation": 0.05,
            }

        # Rastgele seçim fonksiyonu
        def weighted_choice(weights_dict):
            names = list(weights_dict.keys())
            probs = np.array([weights_dict[n] for n in names])
            probs /= probs.sum()
            return rng.choice(names, p=probs)

        # Fazları oluştur
        phases = []
        remaining_days = PREDICTION_DAYS

        for i in range(num_phases):
            if i == 0:
                regime_name = weighted_choice(start_weights)
            else:
                # Geçiş olasılıklarından seç
                prev = phases[-1].regime_name
                transitions = REGIME_TRANSITIONS.get(prev, list(REGIMES.keys()))
                # Eşit ağırlıkla seç
                regime_name = rng.choice(transitions)

            regime = REGIMES[regime_name]

            # Faz uzunluğu
            if i == num_phases - 1:
                days = remaining_days
            else:
                # Kalan günlerin %20-%50'si arası
                min_days = max(30, remaining_days // (num_phases - i + 1))
                max_days = max(min_days + 1, int(remaining_days * 0.55))
                days = int(rng.integers(min_days, max_days))
                remaining_days -= days

            # Rejim parametreleri (aralıktan rastgele)
            drift = rng.uniform(regime.daily_drift_min, regime.daily_drift_max)
            vol = rng.uniform(regime.daily_vol_min, regime.daily_vol_max)
            vol_mult = rng.uniform(regime.volume_multiplier_min, regime.volume_multiplier_max)

            phases.append(PhaseSpec(
                regime_name=regime_name,
                days=days,
                drift=drift,
                volatility=vol,
                volume_multiplier=vol_mult,
            ))

        return phases

    def _generate_ohlcv_phase(
        self,
        rng: np.random.Generator,
        phase: PhaseSpec,
        start_price: float,
        base_volume: float,
        coin_stats: dict,
        start_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Tek bir faz için OHLCV verisi üretir."""

        dates = pd.date_range(start=start_date, periods=phase.days, freq="D")
        regime = REGIMES[phase.regime_name]

        if regime.is_two_phase and phase.regime_name == "pump_dump":
            return self._generate_pump_dump(
                rng, phase, start_price, base_volume, coin_stats, dates
            )

        # GBM ile fiyat simülasyonu
        dt = 1.0  # 1 gün
        prices = np.zeros(phase.days)
        prices[0] = start_price

        for i in range(1, phase.days):
            # Drift'e hafif mean reversion ekle
            noise = rng.normal(0, 1)
            log_return = (phase.drift - 0.5 * phase.volatility**2) * dt + phase.volatility * noise * np.sqrt(dt)
            prices[i] = prices[i-1] * np.exp(log_return)

            # Death spiral: minimum fiyat kontrolü
            if regime.min_floor_pct is not None:
                floor = start_price * regime.min_floor_pct
                prices[i] = max(prices[i], floor)

        # OHLCV oluştur
        h_c_mean = coin_stats["h_c_ratio"]
        l_c_mean = coin_stats["l_c_ratio"]

        rows = []
        for i in range(phase.days):
            close = prices[i]
            if i == 0:
                open_price = start_price * (1 + rng.normal(0, 0.003))
            else:
                # Open = önceki close + küçük gap
                open_price = prices[i-1] * (1 + rng.normal(0, 0.002))

            # High ve Low: volatiliteye bağlı spread
            intraday_vol = phase.volatility * rng.uniform(0.5, 1.5)
            high_spread = abs(rng.normal(0, intraday_vol)) + abs(close - open_price) * 0.3
            low_spread = abs(rng.normal(0, intraday_vol)) + abs(close - open_price) * 0.3

            high = max(open_price, close) + close * high_spread * 0.5
            low = min(open_price, close) - close * low_spread * 0.5

            # Sanity: Low > 0
            low = max(low, close * 0.90)  # en fazla %10 spike
            high = max(high, max(open_price, close) * 1.001)
            low = min(low, min(open_price, close) * 0.999)
            low = max(low, 0.001)  # Sıfır altı olamaz

            # Volume: base × multiplier × rastgele
            vol_noise = rng.lognormal(0, 0.3)
            volume = base_volume * phase.volume_multiplier * vol_noise
            # Büyük fiyat hareketi = büyük hacim
            price_change = abs(close / (prices[i-1] if i > 0 else start_price) - 1)
            if price_change > 0.03:
                volume *= (1 + price_change * 10)

            rows.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Close": round(close, 6),
                "High": round(high, 6),
                "Low": round(low, 6),
                "Open": round(open_price, 6),
                "Volume": round(volume, 2),
            })

        return pd.DataFrame(rows)

    def _generate_pump_dump(
        self,
        rng: np.random.Generator,
        phase: PhaseSpec,
        start_price: float,
        base_volume: float,
        coin_stats: dict,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Pump & dump rejimi: önce hızlı yükseliş, sonra sert düşüş."""
        days = len(dates)
        pump_days = int(days * rng.uniform(0.2, 0.4))
        dump_days = days - pump_days

        # Pump fazı
        pump_mult = rng.uniform(2.0, 5.0)  # 2x-5x yükseliş
        pump_drift = np.log(pump_mult) / pump_days
        pump_vol = phase.volatility * 0.8

        # Dump fazı: neredeyse başlangıca dön
        dump_target = start_price * rng.uniform(0.3, 0.8)
        peak_price = start_price * pump_mult

        prices = np.zeros(days)
        prices[0] = start_price

        for i in range(1, days):
            if i < pump_days:
                drift = pump_drift
                vol = pump_vol
            else:
                # Peak'ten target'a düşüş
                remaining = days - i
                if remaining > 0:
                    drift = np.log(dump_target / prices[i-1]) / max(remaining, 1) * 0.3
                else:
                    drift = 0
                vol = phase.volatility * 1.5

            noise = rng.normal(0, 1)
            log_return = (drift - 0.5 * vol**2) + vol * noise
            prices[i] = prices[i-1] * np.exp(log_return)
            prices[i] = max(prices[i], 0.001)

        rows = []
        for i in range(days):
            close = prices[i]
            open_price = prices[i-1] * (1 + rng.normal(0, 0.003)) if i > 0 else start_price
            intraday_vol = phase.volatility * rng.uniform(0.5, 1.5)
            high = max(open_price, close) * (1 + abs(rng.normal(0, intraday_vol * 0.5)))
            low = min(open_price, close) * (1 - abs(rng.normal(0, intraday_vol * 0.5)))
            low = max(low, 0.001)

            vol_mult = phase.volume_multiplier * rng.lognormal(0, 0.4)
            volume = base_volume * vol_mult

            rows.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Close": round(close, 6),
                "High": round(high, 6),
                "Low": round(low, 6),
                "Open": round(open_price, 6),
                "Volume": round(volume, 2),
            })

        return pd.DataFrame(rows)

    def generate_scenario(self, scenario_id: int, seed: int | None = None) -> ScenarioSpec:
        """
        Tek bir senaryo üretir (3 coin için tam 365 günlük veri).

        Returns:
            ScenarioSpec + self.generated_data[coin] DataFrame'leri
        """
        if seed is None:
            seed = scenario_id * 7919 + 42  # Deterministik ama dağınık

        rng = np.random.default_rng(seed)
        spec = ScenarioSpec(scenario_id=scenario_id, seed=seed)
        self.generated_data: dict[str, pd.DataFrame] = {}

        for coin in COINS:
            # Fazları üret
            coin_rng = np.random.default_rng(rng.integers(0, 2**31))
            phases = self._pick_phases(coin_rng, coin)
            spec.coin_phases[coin] = phases

            stats = self.coin_stats[coin]
            start_price = stats["last_price"]
            base_volume = stats["avg_volume"]

            # Her fazı sırayla üret
            all_phase_dfs = []
            current_price = start_price
            current_date = pd.Timestamp(PREDICTION_START)

            for phase in phases:
                phase_rng = np.random.default_rng(coin_rng.integers(0, 2**31))
                phase_df = self._generate_ohlcv_phase(
                    phase_rng, phase, current_price, base_volume, stats, current_date
                )

                all_phase_dfs.append(phase_df)
                current_price = float(phase_df["Close"].iloc[-1])
                current_date = pd.Timestamp(phase_df["Date"].iloc[-1]) + pd.Timedelta(days=1)

            # Fazları birleştir
            predicted_df = pd.concat(all_phase_dfs, ignore_index=True)

            # Faz geçişlerini yumuşat (5 günlük pencere)
            predicted_df = self._smooth_transitions(predicted_df, phases)

            self.generated_data[coin] = predicted_df

        return spec

    def _smooth_transitions(self, df: pd.DataFrame, phases: list[PhaseSpec]) -> pd.DataFrame:
        """Faz geçişlerindeki sert kırılmaları yumuşatır."""
        df = df.copy()
        transition_points = []
        day = 0
        for phase in phases[:-1]:
            day += phase.days
            transition_points.append(day)

        for tp in transition_points:
            # Geçiş noktasının etrafında 5 günlük pencerede yumuşatma
            window = 3
            start = max(0, tp - window)
            end = min(len(df), tp + window)

            for col in ["Close", "High", "Low", "Open"]:
                values = df[col].values.astype(float)
                # Basit hareketli ortalama yumuşatması
                if end - start > 2:
                    smoothed = pd.Series(values[start:end]).rolling(
                        min(3, end - start), center=True, min_periods=1
                    ).mean().values
                    values[start:end] = smoothed

                    # OHLC tutarlılığını koru
                    df[col] = values

        # Son OHLC tutarlılık kontrolü
        for i in range(len(df)):
            o = df.at[i, "Open"]
            c = df.at[i, "Close"]
            h = df.at[i, "High"]
            l = df.at[i, "Low"]

            df.at[i, "High"] = max(h, o, c)
            df.at[i, "Low"] = min(l, o, c)
            df.at[i, "Low"] = max(df.at[i, "Low"], 0.001)

        return df

    def save_scenario(
        self,
        spec: ScenarioSpec,
        output_dir: Path,
    ) -> Path:
        """Senaryoyu disk'e kaydeder (train + predicted birleşik)."""
        scenario_dir = output_dir / f"scenario_{spec.scenario_id:04d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for coin in COINS:
            train_df = self.training_data[coin].copy()
            pred_df = self.generated_data[coin].copy()

            # Kolon sırası orijinalle aynı olsun
            col_order = train_df.columns.tolist()

            # Tarih formatı string olarak
            train_df["Date"] = train_df["Date"].dt.strftime("%Y-%m-%d")

            # Birleştir
            combined = pd.concat([train_df[col_order], pred_df[col_order]], ignore_index=True)

            # Parquet olarak kaydet
            combined.to_parquet(scenario_dir / f"{coin}.parquet", index=False)

        return scenario_dir


# ─────────────────── Toplu Üretim ────────────────────────────────────────────


def generate_metadata_row(spec: ScenarioSpec) -> dict:
    """Senaryo metadata'sını tek satır olarak döndürür."""
    row = {"scenario_id": spec.scenario_id, "seed": spec.seed}
    for coin in COINS:
        phases = spec.coin_phases.get(coin, [])
        regime_names = [p.regime_name for p in phases]
        row[f"{coin}_regimes"] = "|".join(regime_names)
        row[f"{coin}_num_phases"] = len(phases)
        # İlk ve son faz drift'leri
        if phases:
            row[f"{coin}_first_drift"] = phases[0].drift
            row[f"{coin}_last_drift"] = phases[-1].drift
            row[f"{coin}_avg_vol"] = np.mean([p.volatility for p in phases])
    return row


def batch_generate(
    count: int = 1000,
    start_id: int = 1,
    output_dir: Path | None = None,
) -> None:
    """Toplu senaryo üretimi."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  CNLIB Senaryo Üretim Sistemi                          ║")
    print(f"║  {count} senaryo × 3 coin × 365 gün                     ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print()

    # Training verisi yükle
    print("📥 Training verisi yükleniyor...")
    training_data = load_training_data()
    for coin, df in training_data.items():
        print(f"   {coin}: {len(df)} satır, "
              f"{df['Date'].min().strftime('%Y-%m-%d')} → {df['Date'].max().strftime('%Y-%m-%d')}, "
              f"Son fiyat: ${df['Close'].iloc[-1]:,.2f}")
    print()

    # Generator oluştur
    generator = ScenarioGenerator(training_data)

    # Output dizini
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metadata toplama
    metadata_rows = []

    # İlerleme takibi
    start_time = time.time()

    for i in range(count):
        scenario_id = start_id + i
        seed = scenario_id * 7919 + 42

        spec = generator.generate_scenario(scenario_id, seed)
        scenario_path = generator.save_scenario(spec, output_dir)
        metadata_rows.append(generate_metadata_row(spec))

        # İlerleme raporu
        if (i + 1) % 50 == 0 or i == 0 or i == count - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (count - i - 1) / rate if rate > 0 else 0
            print(f"   ✅ {i+1}/{count} senaryo üretildi "
                  f"({rate:.1f}/sn, kalan: {remaining:.0f}sn)")

        # İlk senaryo detaylarını göster
        if i == 0:
            print(f"\n   📊 Örnek senaryo #{scenario_id}:")
            for coin in COINS:
                phases = spec.coin_phases[coin]
                regimes = " → ".join(
                    f"{p.regime_name}({p.days}d)" for p in phases
                )
                pred_df = generator.generated_data[coin]
                start_p = float(pred_df["Close"].iloc[0])
                end_p = float(pred_df["Close"].iloc[-1])
                change = (end_p / start_p - 1) * 100
                print(f"      {coin}:")
                print(f"        Rejimler: {regimes}")
                print(f"        Fiyat: ${start_p:,.2f} → ${end_p:,.2f} ({change:+.1f}%)")
            print()

    # Metadata kaydet
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_dir.parent / "metadata.csv"
    if metadata_path.exists():
        existing = pd.read_csv(metadata_path)
        metadata_df = pd.concat([existing, metadata_df], ignore_index=True)
        metadata_df = metadata_df.drop_duplicates(subset=["scenario_id"], keep="last")
    metadata_df.to_csv(metadata_path, index=False)

    # Config kaydet
    config = {
        "regimes": {k: asdict(v) for k, v in REGIMES.items()},
        "prediction_start": PREDICTION_START,
        "prediction_end": PREDICTION_END,
        "prediction_days": PREDICTION_DAYS,
        "coins": COINS,
        "total_scenarios": len(metadata_df),
    }
    config_path = output_dir.parent / "generator_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - start_time
    total_rows = count * 3 * (1570 + 365)

    print(f"\n{'='*60}")
    print(f"✅ TAMAMLANDI!")
    print(f"   Üretilen senaryo: {count}")
    print(f"   Toplam veri satırı: {total_rows:,}")
    print(f"   Toplam dosya: {count * 3}")
    print(f"   Süre: {elapsed:.1f} saniye")
    print(f"   Çıktı: {output_dir}")
    print(f"   Metadata: {metadata_path}")
    print(f"{'='*60}")


# ───────────────── Doğrulama ─────────────────────────────────────────────────


def validate_scenario(scenario_dir: Path) -> list[str]:
    """Bir senaryo dizinini doğrular. Hata mesajları döndürür."""
    errors = []

    for coin in COINS:
        fpath = scenario_dir / f"{coin}.parquet"
        if not fpath.exists():
            errors.append(f"{coin}.parquet bulunamadı")
            continue

        df = pd.read_parquet(fpath)

        # Kolon kontrolü
        required = {"Date", "Close", "High", "Low", "Open", "Volume"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"{coin}: eksik kolonlar: {missing}")

        # Satır sayısı
        expected = 1570 + 365
        if len(df) != expected:
            errors.append(f"{coin}: satır sayısı {len(df)}, beklenen {expected}")

        # OHLC tutarlılığı
        for i in range(len(df)):
            o, h, l, c = df.iloc[i][["Open", "High", "Low", "Close"]]
            if h < max(o, c) - 0.01:
                errors.append(f"{coin} satır {i}: High ({h}) < max(Open,Close) ({max(o,c)})")
                break
            if l > min(o, c) + 0.01:
                errors.append(f"{coin} satır {i}: Low ({l}) > min(Open,Close) ({min(o,c)})")
                break

        # Tarih sürekliliği (prediction kısmı)
        dates = pd.to_datetime(df["Date"])
        pred_dates = dates.iloc[1570:]
        if len(pred_dates) > 0:
            expected_start = pd.Timestamp("2027-03-16")
            if pred_dates.iloc[0] != expected_start:
                errors.append(f"{coin}: prediction başlangıç {pred_dates.iloc[0]}, "
                            f"beklenen {expected_start}")

        # Negatif fiyat kontrolü
        for col in ["Open", "High", "Low", "Close"]:
            if (df[col] <= 0).any():
                errors.append(f"{coin}: negatif/sıfır {col} değeri var")
                break

    return errors


# ───────────────── CLI ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="CNLIB coin senaryosu üretici"
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Üretilecek senaryo sayısı (varsayılan: 1000)"
    )
    parser.add_argument(
        "--start-id", type=int, default=1,
        help="Başlangıç senaryo ID'si (varsayılan: 1)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Çıktı dizini (varsayılan: predicted_data/scenarios)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Üretim sonrası doğrulama yap"
    )

    args = parser.parse_args()

    output = Path(args.output) if args.output else OUTPUT_DIR

    batch_generate(
        count=args.count,
        start_id=args.start_id,
        output_dir=output,
    )

    if args.validate:
        print("\n🔍 Doğrulama başlatılıyor...")
        all_ok = True
        for i in range(args.start_id, args.start_id + args.count):
            scenario_dir = output / f"scenario_{i:04d}"
            errs = validate_scenario(scenario_dir)
            if errs:
                all_ok = False
                print(f"   ❌ scenario_{i:04d}: {errs}")
        if all_ok:
            print(f"   ✅ Tüm {args.count} senaryo doğrulandı!")


if __name__ == "__main__":
    main()
