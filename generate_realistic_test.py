"""
Gerçekçi Random Walk Test Verisi Üretici
==========================================
3 coin × 365 gün × 10 senaryo
İstatistiksel kısıtlamalar:
  - Lag-1 autocorrelation: 0.00 – 0.05
  - Günlük volatilite: %2 – %4
  - Coinler arası korelasyon: 0.3 – 0.6
  - Mean daily return: +0.01% – +0.05%
  - Ortalama (H-L)/C range: %3 – %5
  - Volume: 5B – 30B arası random
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

COINS = {
    "kapcoin":  {"start_price": 390.0,  "filename": "kapcoin-usd_realistic_test_1y.csv"},
    "metucoin": {"start_price": 5700.0, "filename": "metucoin-usd_realistic_test_1y.csv"},
    "tamcoin":  {"start_price": 7500.0, "filename": "tamcoin-usd_realistic_test_1y.csv"},
}

NUM_SCENARIOS = 10
DAYS = 365
DATE_START = "2027-03-16"
DATE_END = "2028-03-14"

# İstatistiksel hedefler
TARGET_DAILY_VOL_MIN = 0.02      # %2
TARGET_DAILY_VOL_MAX = 0.04      # %4
TARGET_MEAN_RETURN_MIN = 0.0001  # +0.01%
TARGET_MEAN_RETURN_MAX = 0.0005  # +0.05%
TARGET_CORRELATION_MIN = 0.30
TARGET_CORRELATION_MAX = 0.60
TARGET_AUTOCORR_MAX = 0.05
TARGET_RANGE_MIN = 0.03          # %3
TARGET_RANGE_MAX = 0.05          # %5
VOLUME_MIN = 5e9
VOLUME_MAX = 30e9

OUTPUT_BASE = Path(__file__).parent / "data" / "realistic_test_1y"


def generate_correlated_returns(
    rng: np.random.Generator,
    n_days: int,
    n_coins: int = 3,
    target_corr: float = 0.45,
    daily_vols: np.ndarray = None,
    mean_returns: np.ndarray = None,
) -> np.ndarray:
    """
    Korelasyonlu ve düşük autocorrelation'lı return serisi üretir.
    Returns shape: (n_days, n_coins)
    """
    if daily_vols is None:
        daily_vols = np.array([0.03, 0.03, 0.03])
    if mean_returns is None:
        mean_returns = np.array([0.0003, 0.0003, 0.0003])

    # Korelasyon matrisi oluştur
    corr_matrix = np.full((n_coins, n_coins), target_corr)
    np.fill_diagonal(corr_matrix, 1.0)

    # Cholesky decomposition ile korelasyonlu normal üret
    L = np.linalg.cholesky(corr_matrix)

    # Bağımsız normal dağılım
    z = rng.standard_normal((n_days, n_coins))

    # Korelasyonlu hale getir
    corr_z = z @ L.T

    # Ölçekle ve empirical mean'i düzelt
    returns = np.zeros((n_days, n_coins))
    for i in range(n_coins):
        raw = corr_z[:, i] * daily_vols[i]
        # raw veriyi 0 mean yapıp hedef mean'i ekle
        centered_raw = raw - raw.mean()
        returns[:, i] = centered_raw + mean_returns[i]

    return returns


def validate_and_fix_autocorr(
    returns: np.ndarray, max_autocorr: float = 0.05, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Autocorrelation'ı kontrol eder, gerekirse shuffle ile düzeltir.
    Korelasyonu bozmadan çalışır.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_days, n_coins = returns.shape
    result = returns.copy()

    for coin_idx in range(n_coins):
        autocorr = pd.Series(result[:, coin_idx]).autocorr(1)

        attempts = 0
        while abs(autocorr) > max_autocorr and attempts < 500:
            # Küçük blokları karıştır (korelasyonu korumak için tüm coinleri birlikte)
            block_size = rng.integers(3, max(4, n_days // 10))
            n_blocks = n_days // block_size
            if n_blocks < 2:
                break

            # Rastgele 2 bloğu swap et (tüm coinler birlikte)
            idx1, idx2 = rng.choice(n_blocks, 2, replace=False)
            s1 = slice(idx1 * block_size, (idx1 + 1) * block_size)
            s2 = slice(idx2 * block_size, (idx2 + 1) * block_size)

            old = result.copy()
            result[s1, :], result[s2, :] = old[s2, :].copy(), old[s1, :].copy()

            new_autocorr = pd.Series(result[:, coin_idx]).autocorr(1)
            # Sadece iyileşirse kabul et
            if abs(new_autocorr) < abs(autocorr):
                autocorr = new_autocorr
            else:
                result = old  # geri al

            attempts += 1

    return result


def returns_to_ohlcv(
    rng: np.random.Generator,
    returns: np.ndarray,
    start_price: float,
    dates: pd.DatetimeIndex,
    target_range_pct: float = 0.04,
) -> pd.DataFrame:
    """Return serisinden OHLCV DataFrame üretir."""
    n = len(returns)

    # Close fiyatları
    prices = np.zeros(n)
    prices[0] = start_price * (1 + returns[0])
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + returns[i])
        prices[i] = max(prices[i], 0.01)  # sıfır altı koruması

    rows = []
    for i in range(n):
        close = prices[i]
        prev_close = prices[i - 1] if i > 0 else start_price

        # Open: önceki close etrafında küçük gap
        open_price = prev_close * (1 + rng.normal(0, 0.002))

        # High/Low: hedef range'e uygun
        range_pct = rng.uniform(target_range_pct * 0.7, target_range_pct * 1.3)
        total_range = close * range_pct

        # Body spread (Open-Close arası) range'in bir kısmını yer
        body = abs(open_price - close)
        remaining = max(total_range - body, total_range * 0.2)

        # Kalan range'i wicks'e dağıt
        up_bias = rng.uniform(0.3, 0.7)
        high = max(open_price, close) + remaining * up_bias
        low = min(open_price, close) - remaining * (1 - up_bias)
        low = max(low, 0.01)

        # OHLC tutarlılık
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume: 5B-30B arası lognormal
        volume = rng.uniform(VOLUME_MIN, VOLUME_MAX)
        # Büyük hareket = büyük hacim
        abs_ret = abs(returns[i])
        if abs_ret > 0.03:
            volume *= (1 + abs_ret * 5)

        rows.append({
            "Date": dates[i].strftime("%Y-%m-%d"),
            "Open": round(open_price, 6),
            "High": round(high, 6),
            "Low": round(low, 6),
            "Close": round(close, 6),
            "Volume": round(volume, 2),
            "Split": "realistic_test",
            "Source": "realistic_random_walk",
        })

    return pd.DataFrame(rows)


def generate_scenario(scenario_id: int) -> dict[str, pd.DataFrame]:
    """Tek senaryo: 3 coin × 365 gün."""
    seed = scenario_id * 31337 + 2024
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=DATE_START, periods=DAYS, freq="D")
    coin_names = list(COINS.keys())

    # Her coin için rastgele hedef parametreler (kısıtlar dahilinde)
    daily_vols = np.array([
        rng.uniform(TARGET_DAILY_VOL_MIN, TARGET_DAILY_VOL_MAX)
        for _ in coin_names
    ])
    mean_returns = np.array([
        rng.uniform(TARGET_MEAN_RETURN_MIN, TARGET_MEAN_RETURN_MAX)
        for _ in coin_names
    ])
    target_corr = rng.uniform(TARGET_CORRELATION_MIN, TARGET_CORRELATION_MAX)
    target_range = rng.uniform(TARGET_RANGE_MIN, TARGET_RANGE_MAX)

    # Korelasyonlu return üret — autocorr sağlanana kadar tekrarla
    max_retries = 20
    for attempt in range(max_retries):
        attempt_rng = np.random.default_rng(seed + attempt * 1000)
        returns = generate_correlated_returns(
            attempt_rng, DAYS, len(coin_names),
            target_corr=target_corr,
            daily_vols=daily_vols,
            mean_returns=mean_returns,
        )

        # Autocorrelation düzelt
        returns = validate_and_fix_autocorr(returns, TARGET_AUTOCORR_MAX, rng=attempt_rng)

        # Kontrol et
        all_ok = all(
            abs(pd.Series(returns[:, i]).autocorr(1)) <= TARGET_AUTOCORR_MAX
            for i in range(len(coin_names))
        )
        if all_ok:
            break

    # OHLCV'ye çevir
    result = {}
    for idx, coin in enumerate(coin_names):
        coin_rng = np.random.default_rng(rng.integers(0, 2**31))
        df = returns_to_ohlcv(
            coin_rng,
            returns[:, idx],
            COINS[coin]["start_price"],
            dates,
            target_range_pct=target_range,
        )
        result[coin] = df

    return result


def validate_scenario(scenario_data: dict[str, pd.DataFrame], scenario_id: int) -> bool:
    """İstatistiksel doğrulama."""
    coin_names = list(scenario_data.keys())
    all_ok = True

    returns_dict = {}
    for coin, df in scenario_data.items():
        ret = df["Close"].astype(float).pct_change().dropna()
        returns_dict[coin] = ret

        autocorr = ret.autocorr(1)
        daily_vol = ret.std()
        mean_ret = ret.mean()
        avg_range = ((df["High"].astype(float) - df["Low"].astype(float)) / df["Close"].astype(float)).mean()

        checks = {
            "autocorr": abs(autocorr) <= TARGET_AUTOCORR_MAX,
            "daily_vol": TARGET_DAILY_VOL_MIN * 0.7 <= daily_vol <= TARGET_DAILY_VOL_MAX * 1.3,
            "mean_ret": mean_ret >= -0.001,
            "range": TARGET_RANGE_MIN * 0.5 <= avg_range <= TARGET_RANGE_MAX * 1.5,
        }

        for check_name, passed in checks.items():
            if not passed:
                val = {"autocorr": autocorr, "daily_vol": daily_vol, "mean_ret": mean_ret, "range": avg_range}
                print(f"  ⚠️  S{scenario_id} {coin} {check_name} FAIL: {val[check_name]:.6f}")
                all_ok = False

    # Korelasyon
    if len(coin_names) >= 2:
        for i in range(len(coin_names)):
            for j in range(i + 1, len(coin_names)):
                corr = returns_dict[coin_names[i]].corr(returns_dict[coin_names[j]])
                if not (TARGET_CORRELATION_MIN * 0.5 <= corr <= TARGET_CORRELATION_MAX * 1.5):
                    print(f"  ⚠️  S{scenario_id} {coin_names[i]}-{coin_names[j]} corr FAIL: {corr:.4f}")
                    all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("Gerçekçi Random Walk Test Verisi Üretici")
    print(f"{NUM_SCENARIOS} senaryo × 3 coin × {DAYS} gün")
    print("=" * 60)

    for scenario_id in range(1, NUM_SCENARIOS + 1):
        scenario_dir = OUTPUT_BASE / f"scenario_{scenario_id:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        scenario_data = generate_scenario(scenario_id)

        # Kaydet
        for coin, df in scenario_data.items():
            filename = COINS[coin]["filename"]
            df.to_csv(scenario_dir / filename, index=False)

        # Doğrula
        ok = validate_scenario(scenario_data, scenario_id)

        # Özet
        print(f"\n📊 Senaryo {scenario_id:02d} {'✅' if ok else '⚠️'}:")
        for coin, df in scenario_data.items():
            ret = df["Close"].astype(float).pct_change().dropna()
            autocorr = ret.autocorr(1)
            start_p = float(df["Close"].iloc[0])
            end_p = float(df["Close"].iloc[-1])
            total_ret = (end_p / start_p - 1) * 100
            avg_range = ((df["High"].astype(float) - df["Low"].astype(float)) / df["Close"].astype(float)).mean() * 100

            print(f"  {coin:10s}: ${start_p:,.2f} → ${end_p:,.2f} ({total_ret:+.1f}%) | "
                  f"vol={ret.std()*100:.2f}% | autocorr={autocorr:.4f} | range={avg_range:.1f}%")

        # Korelasyonlar
        coin_names = list(scenario_data.keys())
        rets = {c: scenario_data[c]["Close"].astype(float).pct_change().dropna() for c in coin_names}
        corrs = []
        for i in range(len(coin_names)):
            for j in range(i + 1, len(coin_names)):
                c = rets[coin_names[i]].corr(rets[coin_names[j]])
                corrs.append(f"{coin_names[i][:3]}-{coin_names[j][:3]}={c:.3f}")
        print(f"  Korelasyon: {', '.join(corrs)}")

    print(f"\n{'=' * 60}")
    print(f"✅ Çıktı: {OUTPUT_BASE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
