#!/usr/bin/env python3
"""
PAPÍR-KERESKEDŐ ÜGYNÖKÖK  v2  (nincs valódi pénz, nincs API-kulcs, csak Python 3.8+)

Több ügynök kereskedik virtuális pénzzel (kripto: BTC, ETH, SOL), egymástól
függetlenül, és egy ranglista mutatja, ki teljesít jobban. Az egyik ügynök
tanul (Q-learning), a többi fix stratégia. A VÉLETLEN ügynök és a 30 rejtett
zaj-ügynök kontroll: ha egy "okos" ügynök nem veri őket, nincs értéke.

v2 változások (mind ügynökre, a zaj-ügynökökre is ugyanúgy vonatkoznak):
  - alap időskála 1 órás gyertya (kevesebb ügylet = kevesebb jutalék)
  - végrehajtás a KÖVETKEZŐ gyertya nyitóárán (a döntés a zárásra épül)
  - cooldown két ügylet között (érménként), stop-loss, volatilitáshoz igazított méret
  - a belépési küszöbök figyelembe veszik a költséget (jutalék + csúszás)
  - Visszatérő: nincs lefelé átlagolás
  - Tanuló: a 'prev' állapot mentve (--once módban is tanul), élő módban epszilon = 0
    (felfedezés csak a bemelegítő visszatöltés alatt, vagy --explore kapcsolóval),
    a jutalomban a valós költség (jutalék + csúszás)
  - átlagos kitettség (%) ügynökönként

Használat:
  python kereskedo_ugynokok.py --sim 5000     # gyors teszt szintetikus adattal
  python kereskedo_ugynokok.py                # valódi árak (Binance publikus adat),
                                              # először 1000 gyertya "bemelegítés",
                                              # utána folyamatosan fut
  python kereskedo_ugynokok.py --interval 4h  # 1m, 5m, 15m, 1h (alap), 4h
  python kereskedo_ugynokok.py --report       # csak a ranglista
  python kereskedo_ugynokok.py --once         # egyszeri futás (felhős ütemezéshez, lásd .github/workflows)
  python kereskedo_ugynokok.py --explore      # élőben is engedi a tanuló véletlen lépéseit
  python kereskedo_ugynokok.py --reset        # törli a mentett állapotot

Ha a mentett állapot régi verziójú, vagy más időskálájú, a program az archivum/
mappába tolja, és tiszta lappal indul (a régi és új mérés nem összehasonlítható).
Az állapot a state.json-ban van, az egyenleggörbe az equity.csv-ben.
Leállítás: Ctrl+C. Újraindításkor ott folytatja, ahol abbahagyta.
"""
import argparse, csv, json, math, os, random, time, urllib.parse, urllib.request
from collections import defaultdict

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_CASH = 10_000.0        # virtuális pénz ügynökönként
FEE = 0.001                  # 0,1% jutalék / ügylet (reális spot díj)
SLIP = 0.0005                # 0,05% csúszás
EDGE = 2 * (FEE + SLIP)      # egy teljes kör (vétel + eladás) költsége; a belépési küszöbök ennél nagyobbat várnak
TRADE_FRAC = 0.20            # egy vételnél a tőke 20%-a (volatilitással skálázva, lásd TARGET_VOL)
MAX_POS_FRAC = 0.40          # egy érmében max a tőke 40%-a
WARMUP = 30                  # ennyi gyertya kell, mielőtt döntenek
COOLDOWN = 4                 # két ügylet közt legalább ennyi gyertya (érménként, mindenkinél)
STOP_LOSS = 0.04             # ha a pozíció az átlagos belépési ár alá esik ennyivel, kényszerű eladás
TARGET_VOL_15M = 0.003       # célvolatilitás gyertyánként 15 percen (más időskálán sqrt(idő)-vel skálázva)
VER = 2                      # szabályverzió: ha változik, az állapot archiválódik
DEFAULT_INTERVAL = "1h"
STATE_FILE, EQUITY_FILE, DASH_FILE, NOISE_FILE = "state.json", "equity.csv", "dashboard_data.js", "noise.csv"
ARCHIVE_DIR = "archivum"
NOISE_N = 30                 # ennyi rejtett véletlen ügynök adja a "zajsávot"
BASE = os.environ.get("BINANCE_BASE", "https://api.binance.com")   # felhőben: https://data-api.binance.vision
SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
TICK = 5                     # ennyi másodpercenként frissül az élő ár / dashboard
COST_PCT = (FEE + SLIP) * 100   # egy ügyletoldal költsége százalékban (a tanuló jutalmához)


# ---------- segédfüggvények ----------
def sma(xs, n):
    w = xs[-n:]
    return sum(w) / len(w)

def std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def rets(xs, n):
    w = xs[-n - 1:]
    return [w[i] / w[i - 1] - 1 for i in range(1, len(w))]


# ---------- virtuális számla ----------
class Portfolio:
    def __init__(self):
        self.cash, self.pos = START_CASH, {}
        self.fees, self.trades = 0.0, 0
        self.peak, self.maxdd = START_CASH, 0.0
        self.entry = {}       # átlagos belépési ár érménként (stop-losshoz)
        self.last = {}        # utolsó végrehajtás gyertya-sorszáma érménként (cooldownhoz)
        self.pending = {}     # a következő gyertya nyitóján végrehajtandó megbízások: {érme: [irány, méretszorzó]}
        self.expo_sum, self.expo_n = 0.0, 0   # átlagos kitettséghez

    def equity(self, prices):
        return self.cash + sum(q * prices[s] for s, q in self.pos.items())

    def can_buy(self, s, price, eq):
        cur = self.pos.get(s, 0) * price
        return min(eq * TRADE_FRAC, self.cash, max(0.0, eq * MAX_POS_FRAC - cur)) >= 10

    def buy(self, s, price, eq, mult=1.0, idx=0):
        cur = self.pos.get(s, 0) * price
        spend = min(eq * TRADE_FRAC * mult, self.cash, max(0.0, eq * MAX_POS_FRAC - cur))
        if spend < 10:
            return False
        fee = spend * FEE
        fill = price * (1 + SLIP)
        qty = (spend - fee) / fill
        old = self.pos.get(s, 0)
        self.entry[s] = (self.entry.get(s, fill) * old + fill * qty) / (old + qty) if old > 0 else fill
        self.cash -= spend
        self.pos[s] = old + qty
        self.fees += fee
        self.trades += 1
        self.last[s] = idx
        return True

    def sell(self, s, price, idx=0):
        qty = self.pos.get(s, 0)
        if qty <= 0:
            return False
        gross = qty * price * (1 - SLIP)
        fee = gross * FEE
        self.cash += gross - fee
        self.pos[s] = 0.0
        self.entry.pop(s, None)
        self.fees += fee
        self.trades += 1
        self.last[s] = idx
        return True

    def track(self, eq):
        self.peak = max(self.peak, eq)
        self.maxdd = max(self.maxdd, 1 - eq / self.peak)
        if eq > 0:
            self.expo_sum += max(0.0, (eq - self.cash) / eq)
            self.expo_n += 1

    @property
    def expo(self):
        return self.expo_sum / self.expo_n * 100 if self.expo_n else 0.0

    def to_dict(self):
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d):
        p = Portfolio()
        p.__dict__.update(d)
        return p


# ---------- ügynökök ----------
class Agent:
    name = "ügynök"
    def decide(self, s, hist, holding):
        """1 = vétel, -1 = eladás (mindent), 0 = vár"""
        return 0

class RandomAgent(Agent):
    name = "Véletlen (kontroll)"
    def __init__(self):
        self.rng = random.Random()
    def decide(self, s, hist, holding):
        r = self.rng.random()
        return 1 if r < 0.03 else (-1 if r > 0.97 else 0)

class Momentum(Agent):
    name = "Trendkövető"
    def decide(self, s, hist, holding):
        fast, slow = sma(hist, 10), sma(hist, 30)
        if fast > slow * (1 + EDGE):            # a trendnek a költségnél nagyobbnak kell lennie
            return 0 if holding else 1
        return -1 if fast < slow * 0.999 else 0

class MeanReversion(Agent):
    name = "Visszatérő (mean-rev.)"
    def decide(self, s, hist, holding):
        w = hist[-20:]
        m = sma(w, 20)
        z = (hist[-1] - m) / (std(w) or 1e-9)
        # csak akkor vesz, ha a visszatérés a költséget is fedezné, és NEM átlagol lefelé
        if z < -1.5 and (m - hist[-1]) / hist[-1] > EDGE:
            return 0 if holding else 1
        return -1 if (z > 0 and holding) else 0

class Breakout(Agent):
    name = "Kitörés"
    def decide(self, s, hist, holding):
        if hist[-1] > max(hist[-21:-1]) * (1 + EDGE / 2):
            return 0 if holding else 1
        return -1 if hist[-1] < min(hist[-11:-1]) else 0

class Learner(Agent):
    """Táblázatos Q-learning. Állapot: trend, rövid lendület, volatilitás, tart-e pozíciót.
    Akciók: 0 vár, 1 vesz, 2 elad. Jutalom: a pozíció hozama a következő lépésben, mínusz a valós költség
    (jutalék + csúszás oldalanként). A 'prev' állapot mentődik, így --once módban is tanul.
    Felfedezés (epszilon) csak ha self.explore igaz; élő módban alapból kikapcsolt."""
    name = "TANULÓ (Q-learning)"
    ALPHA, GAMMA = 0.1, 0.9

    def __init__(self, q=None, steps=0, prev=None, vs=1.0):
        self.q = defaultdict(lambda: [0.0, 0.0, 0.0])
        for k, v in (q or {}).items():
            self.q[k] = v
        self.steps, self.prev = steps, dict(prev or {})
        self.vs = vs              # időskála-szorzó a küszöbökhöz (15 perc = 1.0)
        self.explore = False
        self.opens = {}           # az aktuális gyertya nyitóárai (a motor állítja be)

    @property
    def eps(self):
        return max(0.03, 0.3 * 0.9995 ** self.steps) if self.explore else 0.0

    @staticmethod
    def _b(x, t):
        return 0 if x < -t else (2 if x > t else 1)

    def state(self, hist, holding):
        vs = self.vs
        trend = hist[-1] / sma(hist, 20) - 1
        mom = hist[-1] / hist[-6] - 1
        v = std(rets(hist, 20))
        vol = 0 if v < 0.002 * vs else (1 if v < 0.006 * vs else 2)
        return f"{self._b(trend, 0.01 * vs)}{self._b(mom, 0.005 * vs)}{vol}{int(holding)}"

    def decide(self, s, hist, holding):
        st = self.state(hist, holding)
        if s in self.prev:                               # tanulás az előző lépésből
            ps, pa, pos_after, p0, traded, _h = self.prev[s]
            ref = self.opens.get(s, p0) if pa == 1 else p0     # vételnél a fill ára a nyitó
            reward = pos_after * (hist[-1] / ref - 1) * 100 - (COST_PCT if traded else 0)
            target = reward + self.GAMMA * max(self.q[st])
            self.q[ps][pa] += self.ALPHA * (target - self.q[ps][pa])
        a = random.randrange(3) if random.random() < self.eps else max(range(3), key=lambda i: self.q[st][i])
        self.steps += 1
        pos_after = 1 if a == 1 else (0 if a == 2 else int(holding))
        traded = (a == 1) or (a == 2 and holding)
        self.prev[s] = [st, a, pos_after, hist[-1], traded, int(holding)]
        return {0: 0, 1: 1, 2: -1}[a]

    def executed(self, s, fa):
        """A motor visszajelzi, mi lett ténylegesen a döntésből (cooldown, stop-loss, készpénzhiány
        felülírhatja): fa = 1 vétel, -1 eladás, 0 semmi. A tanuló azt a lépést kapja jóvá, ami tényleg történt."""
        if s not in self.prev:
            return
        st, _pa, _pos, p0, _tr, h = self.prev[s]
        pa = {0: 0, 1: 1, -1: 2}[fa]
        pos_after = 1 if fa == 1 else (0 if fa == -1 else h)
        traded = (fa == 1) or (fa == -1 and bool(h))
        self.prev[s] = [st, pa, pos_after, p0, traded, h]


# ---------- szimulált adat (csak teszthez!) ----------
def sim_series(n, seed=1):
    rng = random.Random(seed)
    price = {"BTCUSDT": 60000.0, "ETHUSDT": 3000.0, "SOLUSDT": 150.0}
    beta = {"BTCUSDT": 1.0, "ETHUSDT": 1.2, "SOLUSDT": 1.5}
    out, regime, left, last = [], "chop", 0, 0.0
    for t in range(n):
        if left <= 0:
            regime, left = rng.choice(["up", "down", "chop"]), rng.randint(60, 300)
        left -= 1
        drift = {"up": 0.0006, "down": -0.0006, "chop": 0.0}[regime]
        m = drift + rng.gauss(0, 0.004)
        if regime == "chop":
            m = -0.3 * last + rng.gauss(0, 0.004)
        last = m
        for s in price:
            price[s] *= math.exp(beta[s] * m + rng.gauss(0, 0.003))
        out.append((t, dict(price)))
    return out


# ---------- Binance publikus adat ----------
def fetch_candles(sym, interval, limit=1000):
    """{nyitási idő: (nyitó, záró)} csak lezárt gyertyákra."""
    url = f"{BASE}/api/v3/klines?symbol={sym}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.load(r)
    now = time.time() * 1000
    return {int(k[0]): (float(k[1]), float(k[4])) for k in data if k[6] < now}


# ---------- motor ----------
class Engine:
    def __init__(self, interval=DEFAULT_INTERVAL, archive=True, explore=False):
        self.interval, self.sec = interval, SEC[interval]
        self.target_vol = TARGET_VOL_15M * math.sqrt(self.sec / 900)
        self.force_explore = explore
        self.agents = {}
        self.ports, self.hist = {}, {s: [] for s in SYMBOLS}
        self.last_t, self.p0, self.live_mark = None, None, None
        self.prices, self.live_ok, self.next_candle = None, None, None
        self.noise, self.noise_agents = None, [RandomAgent() for _ in range(NOISE_N)]
        self.live_n, self.mode, self.idx = 0, "live", 0
        self._series, self._dirty = None, True
        self._load(archive)

    def _make_agents(self, q=None, steps=0, prev=None):
        self.learner = Learner(q, steps, prev, vs=math.sqrt(self.sec / 900))
        for a in (RandomAgent(), Momentum(), MeanReversion(), Breakout(), self.learner):
            self.agents[a.name] = a
            self.ports.setdefault(a.name, Portfolio())

    def _archive(self, old_interval):
        ts = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        for f in (STATE_FILE, EQUITY_FILE, NOISE_FILE):
            if os.path.exists(f):
                os.replace(f, os.path.join(ARCHIVE_DIR, f"{ts}_{f}"))
        print(f"A régi állapot (időskála: {old_interval}, régi szabályverzió) az {ARCHIVE_DIR}/ mappába került, "
              f"tiszta lappal indul ({self.interval}).")

    def _load(self, archive):
        d = None
        if os.path.exists(STATE_FILE):
            d = json.load(open(STATE_FILE))
            if d.get("ver") != VER or d.get("interval", "15m") != self.interval:
                if archive:
                    self._archive(d.get("interval", "15m"))
                    d = None
                else:
                    print("FIGYELEM: a mentett állapot régi verziójú / más időskálájú; a --report ezt mutatja.")
        if d:
            self.ports = {k: Portfolio.from_dict(v) for k, v in d["ports"].items()}
            self.hist, self.last_t = d["hist"], d["last_t"]
            self.p0, self.live_mark = d["p0"], d.get("live_mark")
            self.live_n, self.idx = d.get("live_n", 0), d.get("idx", 0)
            self.noise = [Portfolio.from_dict(x) for x in d["noise"]] if "noise" in d else [Portfolio() for _ in range(NOISE_N)]
            self._make_agents(d["q"], d["steps"], d.get("prev"))
        else:
            self.noise = [Portfolio() for _ in range(NOISE_N)]
            self._make_agents()

    def save(self):
        d = dict(ver=VER, interval=self.interval, idx=self.idx,
                 ports={k: v.to_dict() for k, v in self.ports.items()},
                 hist={s: h[-300:] for s, h in self.hist.items()},
                 last_t=self.last_t, p0=self.p0, live_mark=self.live_mark,
                 q={k: v for k, v in self.learner.q.items()}, steps=self.learner.steps,
                 prev={s: list(v) for s, v in self.learner.prev.items()},
                 noise=[p.to_dict() for p in self.noise], live_n=self.live_n)
        tmp = STATE_FILE + ".tmp"
        json.dump(d, open(tmp, "w"))
        os.replace(tmp, STATE_FILE)
        self._dirty = True
        self.export()

    @staticmethod
    def _quant(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]

    def _sharpe(self, vals):
        r = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
        if len(r) < 30:
            return None
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / len(r))
        return None if sd < 1e-12 else m / sd * math.sqrt(365 * 24 * 3600 / self.sec)

    def export(self, live=None):
        """Adatok a dashboard.html-nek. Élő árral újraszámolja az egyenlegeket."""
        if not self.hist[SYMBOLS[0]] or not os.path.exists(EQUITY_FILE):
            return
        try:
            if self._dirty or self._series is None:
                rows = list(csv.reader(open(EQUITY_FILE)))
                head, body = rows[0], rows[1:][-800:]
                ser = {h: [float(r[i]) for r in body] for i, h in enumerate(head)}
                ser["t"] = [int(x) for x in ser["t"]]
                nz = {}
                if os.path.exists(NOISE_FILE):
                    for r in list(csv.reader(open(NOISE_FILE)))[1:]:
                        nz[int(r[0])] = (float(r[1]), float(r[2]), float(r[3]))
                for j, k in enumerate(("noise_mean", "noise_lo", "noise_hi")):
                    ser[k] = [nz[t][j] if t in nz else None for t in ser["t"]]
                self._series, self._dirty = ser, False
            series = {k: list(v) for k, v in self._series.items()}
            pr = dict(live) if live else {s: self.hist[s][-1] for s in SYMBOLS}
            now_ms = int(time.time() * 1000)
            nlive = None
            if self.live_mark and "noise" in self.live_mark:
                nl = [(p.equity(pr) / m - 1) * 100 for p, m in zip(self.noise, self.live_mark["noise"])]
                mu = sum(nl) / len(nl)
                sd = math.sqrt(sum((x - mu) ** 2 for x in nl) / len(nl))
                nlive = dict(mean=mu, std=sd, p5=self._quant(nl, .05), p95=self._quant(nl, .95), n=len(nl))
            ags = []
            for n, p in self.ports.items():
                eq = p.equity(pr)
                lv = (eq / self.live_mark[n] - 1) * 100 if self.live_mark and n in self.live_mark else None
                z = (lv - nlive["mean"]) / nlive["std"] if lv is not None and nlive and nlive["std"] > 1e-9 else None
                ags.append(dict(name=n, equity=eq, ret=(eq / START_CASH - 1) * 100, live=lv, z=z,
                                dd=max(p.maxdd, 1 - eq / max(p.peak, eq)) * 100, trades=p.trades, fees=p.fees,
                                cash=p.cash, qty={s: p.pos.get(s, 0) for s in SYMBOLS},
                                expo=p.expo,
                                mark=self.live_mark.get(n) if self.live_mark else None,
                                sharpe=self._sharpe(self._series[n]) if n in self._series else None))
            b = self.bench(pr)
            blv = (b / self.live_mark["bench"] - 1) * 100 if self.live_mark else None
            if series["t"] and series["t"][-1] > 1e11 and now_ms > series["t"][-1]:   # élő pont a görbe végén
                series["t"].append(now_ms)
                series["buyhold"].append(round(b, 2))
                for a_ in ags:
                    if a_["name"] in series:
                        series[a_["name"]].append(round(a_["equity"], 2))
                for k in ("noise_mean", "noise_lo", "noise_hi"):
                    series[k].append(series[k][-1])
            d = dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), ts=now_ms, mode=self.mode, sec=self.sec,
                     interval=self.interval,
                     feed=("elo" if live and self.live_ok else "gyertya"), next_candle=self.next_candle,
                     agents=ags, p0=self.p0, start=START_CASH, live_n=self.live_n, noise=nlive,
                     bench=dict(equity=b, ret=(b / START_CASH - 1) * 100, live=blv,
                                mark=self.live_mark.get("bench") if self.live_mark else None),
                     series=series, prices=pr,
                     rules=dict(cooldown=COOLDOWN, stop_loss=STOP_LOSS * 100, edge=EDGE * 100),
                     learner=dict(steps=self.learner.steps, eps=self.learner.eps, states=len(self.learner.q),
                                  explore=bool(self.learner.explore)))
            tmp = DASH_FILE + ".tmp"
            open(tmp, "w", encoding="utf-8").write("window.DATA=" + json.dumps(d, ensure_ascii=False))
            os.replace(tmp, DASH_FILE)
        except Exception as e:
            print("Dashboard-export hiba:", e)

    def _fill_pending(self, opens):
        """Az előző gyertya zárásakor feladott megbízások végrehajtása a mostani gyertya NYITÓÁRÁN."""
        for port in list(self.ports.values()) + self.noise:
            if not port.pending:
                continue
            for s, (a, _m) in port.pending.items():           # előbb az eladások (készpénzt szabadítanak fel)
                if a == -1:
                    port.sell(s, opens[s], self.idx)
            for s, (a, mult) in port.pending.items():
                if a == 1:
                    port.buy(s, opens[s], port.equity(opens), mult, self.idx)
            port.pending = {}

    def _decide(self, ag, port, prices):
        """Döntések a gyertya zárása után; a megbízások a következő nyitón hajtódnak végre.
        Minden ügynökre (a zaj-ügynökökre is) ugyanazok a szabályok: cooldown, stop-loss, méretezés."""
        eq = port.equity(prices)
        is_l = isinstance(ag, Learner)
        for s in SYMBOLS:
            h, px = self.hist[s], prices[s]
            holding = port.pos.get(s, 0) * px > 10
            a = ag.decide(s, h, holding)
            if holding and port.entry.get(s) and px < port.entry[s] * (1 - STOP_LOSS):
                a = -1                                          # stop-loss mindent felülír
            elif a != 0 and self.idx - port.last.get(s, -10 ** 9) < COOLDOWN:
                a = 0                                           # cooldown
            if a == 1 and not port.can_buy(s, px, eq):
                a = 0
            if a == -1 and not holding:
                a = 0
            if is_l:
                ag.executed(s, a)
            if a != 0:
                v = std(rets(h, 20))
                mult = max(0.5, min(1.5, self.target_vol / v)) if v > 1e-9 else 1.0   # volatilitáshoz igazított méret
                port.pending[s] = [a, mult]

    def step(self, t, opens, closes):
        """t = a gyertya nyitási ideje; opens/closes = nyitó/záró árak érménként."""
        if self.p0 is None:
            self.p0 = dict(closes)
        self.idx += 1
        self._fill_pending(opens)
        for s in SYMBOLS:
            self.hist[s].append(closes[s])
            self.hist[s] = self.hist[s][-300:]
        self.last_t = t
        tc = t + self.sec * 1000            # az egyenleg a ZÁRÁSKOR érvényes, ezért a zárás idejét írjuk
        self.learner.opens = opens
        self.learner.explore = self.force_explore or self.live_mark is None   # felfedezés csak bemelegítéskor
        if len(self.hist[SYMBOLS[0]]) >= WARMUP:
            for name, ag in self.agents.items():
                self._decide(ag, self.ports[name], closes)
            for ag, port in zip(self.noise_agents, self.noise):
                self._decide(ag, port, closes)
        ne = []
        for port in self.noise:
            e = port.equity(closes)
            port.track(e)
            ne.append(e)
        newn = not os.path.exists(NOISE_FILE)
        with open(NOISE_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if newn:
                w.writerow(["t", "mean", "p5", "p95"])
            w.writerow([tc, round(sum(ne) / len(ne), 2), round(self._quant(ne, .05), 2), round(self._quant(ne, .95), 2)])
        if self.live_mark:
            self.live_n += 1
        eqs = {}
        for name, port in self.ports.items():
            eqs[name] = port.equity(closes)
            port.track(eqs[name])
        new = not os.path.exists(EQUITY_FILE)
        with open(EQUITY_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["t", "buyhold"] + list(eqs))
            w.writerow([tc, round(self.bench(closes), 2)] + [round(v, 2) for v in eqs.values()])
        self.prices = closes

    def bench(self, prices):
        return START_CASH * sum(prices[s] / self.p0[s] for s in SYMBOLS) / len(SYMBOLS)

    def mark_live(self):
        self.live_mark = {n: p.equity(self.prices) for n, p in self.ports.items()}
        self.live_mark["bench"] = self.bench(self.prices)
        self.live_mark["noise"] = [p.equity(self.prices) for p in self.noise]
        self.live_n = 0

    def report(self):
        pr = getattr(self, "prices", None) or {s: self.hist[s][-1] for s in SYMBOLS if self.hist[s]}
        if not pr:
            print("Még nincs adat.")
            return
        b = self.bench(pr)
        rows = sorted(((n, p.equity(pr), p) for n, p in self.ports.items()), key=lambda r: -r[1])
        print(f"\n{'Ügynök':26}{'Egyenleg':>11}{'Hozam':>9}{'Élő óta':>9}{'MaxDD':>8}{'Ügylet':>8}{'Jutalék':>9}{'Kitett.':>9}")
        for n, eq, p in rows:
            lv = f"{(eq / self.live_mark[n] - 1) * 100:+.1f}%" if self.live_mark and n in self.live_mark else "-"
            print(f"{n:26}{eq:11,.0f}{(eq / START_CASH - 1) * 100:+8.1f}%{lv:>9}{p.maxdd * 100:7.1f}%{p.trades:8d}{p.fees:9,.0f}{p.expo:8.0f}%")
        lv = f"{(b / self.live_mark['bench'] - 1) * 100:+.1f}%" if self.live_mark else "-"
        print(f"{'Vásárol és tart (viszonyítás)':26}{b:11,.0f}{(b / START_CASH - 1) * 100:+8.1f}%{lv:>9}")
        if self.live_mark and "noise" in self.live_mark:
            nl = sorted((p.equity(pr) / m - 1) * 100 for p, m in zip(self.noise, self.live_mark["noise"]))
            print(f"Zajsáv ({len(nl)} véletlen ügynök, élő óta): átlag {sum(nl)/len(nl):+.1f}%, "
                  f"5–95%: {self._quant(nl, .05):+.1f}% … {self._quant(nl, .95):+.1f}%  "
                  f"(mérés: {self.live_n} gyertya; ~200 alatt még nem megbízható)")
        print(f"Tanuló: {self.learner.steps} döntés, felfedezési arány {self.learner.eps:.2f}"
              f"{' (felfedezés BE)' if self.learner.explore else ' (felfedezés KI)'}, "
              f"{len(self.learner.q)} megtanult állapot\n")


# ---------- futtatás ----------
def fetch_live():
    q = urllib.parse.quote(json.dumps(SYMBOLS, separators=(",", ":")))
    with urllib.request.urlopen(f"{BASE}/api/v3/ticker/price?symbols={q}", timeout=10) as r:
        return {x["symbol"]: float(x["price"]) for x in json.load(r)}


def catch_up(eng, interval):
    """Feldolgozza az összes új lezárt gyertyát (akkor is, ha a program közben állt)."""
    series = {s: fetch_candles(s, interval) for s in SYMBOLS}
    times = sorted(set.intersection(*[set(v) for v in series.values()]))
    todo = [t for t in times if eng.last_t is None or t > eng.last_t]
    if eng.last_t and times and times[0] > eng.last_t + SEC[interval] * 1000 * 1.5:
        print("FIGYELEM: túl hosszú kiesés, a hiányzó gyertyák már nem kérhetők le (max ~1000 db).")
    for t in todo:
        eng.step(t, {s: series[s][t][0] for s in SYMBOLS}, {s: series[s][t][1] for s in SYMBOLS})
    if eng.live_mark is None and todo:
        eng.mark_live()
    eng.save()
    if todo:
        print(time.strftime("%Y-%m-%d %H:%M"), f"- {len(todo)} új gyertya feldolgozva")
        eng.report()
    return todo


def run_once(eng, interval):
    """Egyszeri futás (felhőbe / ütemezőbe): behozza a lemaradást, ment, kiír, kilép."""
    eng.mode, eng.sec = "once", SEC[interval]
    catch_up(eng, interval)
    try:
        eng.live_ok = True
        eng.export(fetch_live())
    except Exception as e:
        print("Élő ár hiba:", e)
        eng.export()


def run_live(eng, interval):
    sec, next_check = SEC[interval], 0
    eng.sec = sec
    print(f"Élő mód: {TICK} mp-enként frissül a dashboard, {interval}-os gyertyánként döntenek az ügynökök. Leállítás: Ctrl+C")
    while True:
        now = time.time()
        if now >= next_check:
            try:
                catch_up(eng, interval)
                next_check = now - now % sec + sec + 5
            except Exception as e:
                print("Hiba (újrapróbálom 30 mp múlva):", e)
                next_check = now + 30
            eng.next_candle = int(next_check * 1000)
        try:
            live = fetch_live()
            eng.live_ok = True
        except Exception as e:
            live, eng.live_ok = None, False
            print("Élő ár hiba:", e)
        if eng.prices:
            eng.export(live or eng.prices)
        time.sleep(TICK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=int, help="szintetikus teszt N lépéssel")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL, choices=SEC)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--once", action="store_true", help="egyszeri futás (felhős ütemezéshez)")
    ap.add_argument("--explore", action="store_true", help="a tanuló élőben is felfedez (alapból kikapcsolva)")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    global STATE_FILE, EQUITY_FILE, DASH_FILE, NOISE_FILE
    if a.sim:   # a szimuláció külön fájlokba ír, nem rontja el az éles állapotot
        STATE_FILE, EQUITY_FILE, DASH_FILE, NOISE_FILE = ("sim_state.json", "sim_equity.csv", "sim_dashboard_data.js", "sim_noise.csv")
        for f in (STATE_FILE, EQUITY_FILE, NOISE_FILE):
            if os.path.exists(f):
                os.remove(f)
    if a.reset:
        for f in (STATE_FILE, EQUITY_FILE, NOISE_FILE):
            if os.path.exists(f):
                os.remove(f)
        print("Állapot törölve.")
        return
    eng = Engine(a.interval, archive=not a.report, explore=a.explore)
    if a.report:
        eng.report()
    elif a.sim:
        cut = int(a.sim * 0.7)       # az első 70% a "tanítás" (felfedezéssel), az utolsó 30% a nem látott "teszt"
        prev_p = None
        for i, (t, p) in enumerate(sim_series(a.sim)):
            eng.step(t, dict(prev_p or p), p)    # a szimulált nyitó = az előző zárás
            prev_p = p
            if i == cut:
                eng.mark_live()
        print(f"Az 'Élő óta' oszlop itt a TESZT-szakasz (utolsó 30%, {a.sim - cut} lépés); a 'Hozam' a teljes, tanítással együtt.")
        eng.save()
        eng.report()
        print("FIGYELEM: szintetikus adat, ebből nem lehet a valós piacra következtetni.")
    elif a.once:
        run_once(eng, a.interval)
    else:
        run_live(eng, a.interval)


if __name__ == "__main__":
    main()
