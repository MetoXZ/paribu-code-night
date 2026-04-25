"""
Ters Köşe (Trick) Senaryoları Üretici
======================================
Makine Öğrenmesi algoritmasının "trend bağımlılığını" ve "yüksek otokorelasyon
ezberini" cezalandırmak için tasarlanmış, manuel olarak manipüle edilmiş (ters köşe) 5 senaryo.
"""

import numpy as np
import pandas as pd
from pathlib import Path

COINS = {
    "kapcoin":  {"start_price": 390.0,  "filename": "kapcoin-usd_trick_test_1y.csv"},
    "metucoin": {"start_price": 5700.0, "filename": "metucoin-usd_trick_test_1y.csv"},
    "tamcoin":  {"start_price": 7500.0, "filename": "tamcoin-usd_trick_test_1y.csv"},
}

DAYS = 365
DATE_START = "2027-03-16"
OUTPUT_BASE = Path(__file__).parent / "data" / "trick_test_1y"

def build_df(prices, dates, coin_name, desc):
    """Fiyat dizisinden OHLCV DataFrame üretir."""
    rows = []
    rng = np.random.default_rng(len(prices))
    for i in range(len(prices)):
        close = prices[i]
        prev_close = prices[i-1] if i > 0 else close

        # Çok düşük rastgele range (%1)
        half_range = close * 0.01
        
        # Flash crash veya pump varsa wick'ler büyük olsun
        if i > 0 and abs(prices[i]/prices[i-1] - 1) > 0.05:
            half_range = close * 0.05
            
        open_price = prev_close * (1 + rng.normal(0, 0.001))
        
        high = max(open_price, close) + rng.uniform(0, half_range)
        low = min(open_price, close) - rng.uniform(0, half_range)
        low = max(low, 0.0001)

        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Hacim normalde 10B, büyük harekette 50B
        volume = rng.uniform(5e9, 15e9)
        if i > 0 and abs(prices[i]/prices[i-1] - 1) > 0.10:
            volume *= 5

        rows.append({
            "Date": dates[i].strftime("%Y-%m-%d"),
            "Open": round(open_price, 6),
            "High": round(high, 6),
            "Low": round(low, 6),
            "Close": round(close, 6),
            "Volume": round(volume, 2),
            "Split": "trick_test",
            "Source": desc,
        })
    return pd.DataFrame(rows)

def make_scenario(scen_idx, dates):
    df_dict = {}
    for coin, info in COINS.items():
        rng = np.random.default_rng(scen_idx * 100 + len(coin))
        prices = np.zeros(DAYS)
        prices[0] = info["start_price"]
        
        # Senaryo 1: Flash Crash (Günde %80 Düşüş)
        # 200 gün boyunca aşırı trend %65 otokorelasyon benzeri yükselir, sonra aniden sıfırlanır
        if scen_idx == 1:
            desc = "flash_crash"
            crash_day = 200
            for i in range(1, DAYS):
                if i < crash_day:
                    # %0.2 istikrarlı büyüme + az vol
                    prices[i] = prices[i-1] * (1 + 0.002 + rng.normal(0, 0.01))
                elif i == crash_day:
                    # Tek günde %80 düşüş!
                    prices[i] = prices[i-1] * 0.20
                else:
                    # Yatay seyir
                    prices[i] = prices[i-1] * (1 + rng.normal(0, 0.005))
                    
        # Senaryo 2: Pump and Dump (Hileli)
        # Dümdüz giderken 150. günde 5 gün boyunca günde %50 fırlar, sonra tek günde eski yerine döner
        elif scen_idx == 2:
            desc = "pump_and_dump"
            pump_start = 150
            for i in range(1, DAYS):
                if pump_start <= i <= pump_start + 5:
                    prices[i] = prices[i-1] * 1.50 # %50 zıplama
                elif i == pump_start + 6:
                    prices[i] = info["start_price"] # Orijinal fiyata geri çakılma (dump)
                else:
                    prices[i] = prices[i-1] * (1 + rng.normal(0, 0.005))

        # Senaryo 3: Rug Pull (Delist / Yavaş Sürekli Ölüm)
        # 100. güne kadar tavan, sonra günde -%5 garanti şekilde kaybeder ve sıfıra gider
        elif scen_idx == 3:
            desc = "rug_pull"
            rug_day = 100
            for i in range(1, DAYS):
                if i < rug_day:
                    prices[i] = prices[i-1] * (1 + rng.normal(0.002, 0.02))
                else:
                    prices[i] = prices[i-1] * 0.95 # Her gün garanti %5 düşüş

        # Senaryo 4: Volatility Shock (Fırtına)
        # 180 gün sıfır volatilite (dümdüz), sonra devasa %20'lik çılgın günlük dalgalanmalar (yönsüz)
        elif scen_idx == 4:
            desc = "volatility_shock"
            shock_day = 180
            for i in range(1, DAYS):
                if i < shock_day:
                    prices[i] = prices[i-1] * (1 + rng.normal(0, 0.001)) # %0.1 vol
                else:
                    prices[i] = prices[i-1] * (1 + rng.choice([-1, 1]) * rng.uniform(0.15, 0.25))

        # Senaryo 5: Slow Bleed (Kesin Zarar Döngüsü)
        # İlk gün %100 uçar, sonra model trend var sanıp "Long" açar ama coin her gün eksiksiz %1 düşer
        elif scen_idx == 5:
            desc = "slow_bleed"
            for i in range(1, DAYS):
                if i == 5:
                    prices[i] = prices[i-1] * 2.0
                elif i > 5:
                    # Düzenli günlük azalan, modelin long kapatmasına izin vermeyecek kadar "yavaş ama garanti" düşüş
                    prices[i] = prices[i-1] * 0.99
                else:
                    prices[i] = prices[i-1] * (1 + rng.normal(0, 0.01))

        # Sıfır koruması
        prices = np.clip(prices, 0.0001, None)
        df_dict[coin] = build_df(prices, dates, coin, desc)

    return df_dict

def main():
    print("=" * 50)
    print("Ters Köşe (Trick) 5 Senaryo Üretiliyor...")
    print("=" * 50)
    dates = pd.date_range(DATE_START, periods=DAYS, freq="D")

    for i in range(1, 6):
        out_dir = OUTPUT_BASE / f"scenario_{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        dfs = make_scenario(i, dates)
        for coin, df in dfs.items():
            df.to_csv(out_dir / COINS[coin]["filename"], index=False)
            
        print(f"✅ Senaryo {i:02d} oluşturuldu: {df.iloc[0]['Source']}")
        
    print("==================================================")
    print("Bütün dosyalar 'data/trick_test_1y' içine kaydedildi.")

if __name__ == "__main__":
    main()
