**CNLIB**

**cnlib Kullanım Kılavuzu**

**Kurulum**

pip install cnlib

Kurulumla birlikte 4 yıllık training datası da gelir, ayrıca indirmenize
gerek yoktur.

**Temel Yapı**

Strateji yazmak için BaseStrategy\'den miras almanız yeterlidir:

from cnlib.base_strategy import BaseStrategy

from cnlib import backtest

class Stratejim(BaseStrategy):

def predict(self, data: dict) -\> list\[dict\]:

return \[\]

strategy = Stratejim()

result = backtest.run(strategy=strategy, initial_capital=3000.0)

result.print_summary()

**predict() Metodu**

predict(), her candle kapanışında otomatik olarak çağrılır. Göreviniz bu
metodun içinde mevcut veriye bakıp hangi coinlerde işlem açacağınıza
veya kapatacağınıza karar vermektir.

**Parametre**

def predict(self, data: dict) -\> list\[dict\]:

data → her coin için o ana kadarki tüm fiyat geçmişini içeren bir
sözlük:

{

\"kapcoin-usd_train\": pd.DataFrame, \# o ana kadar tüm geçmiş

\"metucoin-usd_train\": pd.DataFrame,

\"tamcoin-usd_train\": pd.DataFrame,

}

**DataFrame Kolonları**

  ------------------------------
  **Kolon**   **Açıklama**
  ----------- ------------------
  Date        Tarih

  Open        Açılış fiyatı

  High        Günün en yüksek
              fiyatı

  Low         Günün en düşük
              fiyatı

  Close       Kapanış fiyatı

  Volume      İşlem hacmi
  ------------------------------

**Dönüş Değeri**

Her candle\'da **3 coinin tamamı** listede yer almalıdır:

\[

{\"coin\": \"kapcoin-usd_train\", \"signal\": 1, \"allocation\": 0.4,
\"leverage\": 2},

{\"coin\": \"metucoin-usd_train\", \"signal\": -1, \"allocation\": 0.3,
\"leverage\": 1, \"take_profit\": coin\*0.2},

{\"coin\": \"tamcoin-usd_train\", \"signal\": 0, \"allocation\": 0.0,
\"leverage\": 1, \"stop_loss\": coin\*0.2},

\]

//TP SL opsiyonel

  ------------------------------------------------------------
  **Alan**     **Değerler**   **Açıklama**
  ------------ -------------- --------------------------------
  coin         coin adı       Hangi coin
               (string)       

  signal       1, -1, 0       1=long, -1=short, 0=kapat / açma

  allocation   0.0 -- 1.0     Mevcut cash\'in ne kadarını
                              kullanacağınız

  leverage     1, 2, 3, 5, 10 Kaldıraç
  ------------------------------------------------------------

**Önemli Kurallar**

- Her candle\'da **3 coin de listede olmalıdır**, eksik coin hata verir

- signal=0 → pozisyon açıksa kapatır, kapalıysa dokunmaz

- signal=1 veya 1 → pozisyon zaten açıksa tutmaya devam eder, kapalıysa
  yeni açar

- signal=0 olan coinlerin allocation değeri 0.0 olmalıdır

- Tüm aktif allocation değerlerinin toplamı **1.0\'ı geçemez**

**Kullanılabilir Veri**

data\[coin\] size o ana kadar tüm geçmişi verir. Örnek kullanımlar:

df = data\[\"kapcoin-usd_train\"\]

\# Son kapanış fiyatı

son_fiyat = df\[\"Close\"\].iloc\[-1\]

\# Son 20 günün ortalaması

ort_20 = df\[\"Close\"\].iloc\[-20:\].mean()

\# Tüm kapanış serisi

kapanislar = df\[\"Close\"\]

**Machine Learning ile Kullanım**

Model eğitimini \_\_init\_\_ içinde yapın, predict() içinde sadece
tahmin yapın. predict() her candle\'da çağrıldığı için eğitimi orada
yapmak çok yavaş olur.

from sklearn.ensemble import RandomForestClassifier

import numpy as np

from cnlib.base_strategy import BaseStrategy

from cnlib import backtest

class MLStratejim(BaseStrategy):

def \_\_init\_\_(self):

super().\_\_init\_\_()

self.modeller = {}

def egit(self):

\"\"\"get_data() çağrısından sonra çağırın.\"\"\"

for coin, df in self.coin_data.items():

X, y = self.\_ozellikler(df)

model = RandomForestClassifier()

model.fit(X\[:-1\], y\[:-1\]) \# son candle\'ı test için bırak

self.modeller\[coin\] = model

def \_ozellikler(self, df):

closes = df\[\"Close\"\]

\# Örnek: 5 ve 20 günlük MA farkı, momentum

ma5 = closes.rolling(5).mean()

ma20 = closes.rolling(20).mean()

X = np.column_stack(\[ma5, ma20, closes.pct_change()\])

y = (closes.shift(-1) \> closes).astype(int) \# yarın yükselecek mi?

mask = \~np.isnan(X).any(axis=1) & \~np.isnan(y)

return X\[mask\], y\[mask\]

def predict(self, data: dict) -\> list\[dict\]:

decisions = \[\]

for coin, df in data.items():

if coin not in self.modeller or len(df) \< 21:

decisions.append({\"coin\": coin, \"signal\": 0, \"allocation\": 0.0,
\"leverage\": 1})

continue

X, \_ = self.\_ozellikler(df)

tahmin = self.modeller\[coin\].predict(\[X\[-1\]\])\[0\]

if tahmin == 1:

decisions.append({\"coin\": coin, \"signal\": 1, \"allocation\": 0.3,
\"leverage\": 1})

else:

decisions.append({\"coin\": coin, \"signal\": 0, \"allocation\": 0.0,
\"leverage\": 1})

return decisions

strategy = MLStratejim()

strategy.get_data() \# önce data yükle

strategy.egit() \# sonra modeli eğit

result = backtest.run(strategy=strategy, initial_capital=3000.0)

result.print_summary()

**Not:** backtest.run() içinde get_data() otomatik çağrılır. ML modeli
kullanıyorsanız get_data()\'yı kendiniz çağırıp ardından egit()\'i
çalıştırın, sonra backtest.run()\'a geçin.

**Backtest Sonuçları**

result.print_summary()

=======================================================

BACKTEST RESULTS

=======================================================

Initial Capital : \$ 3,000.00

Final Portfolio : \$ 5,619.00

Net P&L : \$ +2,619.00

Return : +87.30%

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

Total Candles : 365

Total Trades : 68

Liquidations : 0

Liquidation Loss : \$ 0.00

Validation Errors : 0

Strategy Errors : 0

=======================================================

result üzerinden detaylı veriye de erişebilirsiniz:

result.portfolio_series \# her candle\'da portföy değeri

result.trade_history \# sadece işlem olan candle\'lar

result.return_pct \# toplam getiri yüzdesi
