#!/usr/bin/env python3
"""Vrumona Packaging Optimiser – met drukvariatie & opslagcapaciteit
===========================================================================
Toevoegingen:
- Interne opslagcapaciteit in m³ (met automatische overschakeling naar extern)
- Variabiliteit in drukmetingen (CV) bepaalt een veiligheidsfactor f
- f beïnvloedt de maximaal toelaatbare stapelhoogte
- Losse input voor product- en verpakkingsgewicht
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

# Constants
g = 9.81  # m/s²
t_pallet_base_mm = 120  # hoogte houten pallet in mm

@dataclass
class Box:
    length_mm: float
    width_mm: float
    height_mm: float
    thickness_mm: float
    ect_knpm: float

    @property
    def perimeter_mm(self) -> float:
        return 2 * (self.length_mm + self.width_mm)

@dataclass
class ScenarioInput:
    box: Box
    product_weight_kg: float
    boxes_per_layer: int
    n_boxes_total: int
    price_per_box_eur: float
    storage_weeks: int
    storage_cost_internal_per_m3: float
    storage_cost_external_per_m3: float
    use_external: bool
    internal_capacity_m3: Optional[float] = None
    pressures_kpa: Optional[List[float]] = None
    k_factor: float = 5.876
    env_factor: float = 0.95
    stackable: bool = True
    user_max_layers: int = 0
    rack_height_mm: float = 0.0
    customer_height_mm: float = 0.0

@dataclass
class ScenarioResult:
    layers: int
    pallets: int
    carton_cost: float
    storage_cost: float
    total_cost: float
    constraint: Optional[str]

def ask_float(prompt: str, default: float) -> float:
    v = input(f"{prompt} [{default}]: ").strip().replace(',', '.')
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        print("❌ Ongeldige invoer, default gebruikt.")
        return default

def ask_int(prompt: str, default: int) -> int:
    v = input(f"{prompt} [{default}]: ").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        print("❌ Ongeldige invoer, default gebruikt.")
        return default

def ask_bool(prompt: str, default_yes: bool = True) -> bool:
    d = 'j' if default_yes else 'n'
    v = input(f"{prompt} (j/n) [{d}]: ").strip().lower()
    if not v:
        return default_yes
    return v in ('j', 'y', 'yes')

def compute_variability(pressures: List[float]) -> tuple[float, float, float]:
    n = len(pressures)
    avg = sum(pressures) / n
    variance = sum((d - avg) ** 2 for d in pressures) / n
    stddev = math.sqrt(variance)
    cv = (stddev / avg) * 100 if avg != 0 else 0
    return avg, stddev, cv

def get_safety_factor(cv: float) -> float:
    if cv < 5:
        return 1.1
    elif cv < 10:
        return 1.3
    elif cv < 20:
        return 1.6
    else:
        return 2.0

def compute_bct(inp: ScenarioInput) -> float:
    return inp.k_factor * inp.box.ect_knpm * math.sqrt(inp.box.thickness_mm * inp.box.perimeter_mm) * inp.env_factor * 1000

def compute_mech_max_layers(bct_n: float, inp: ScenarioInput) -> int:
    load_layer_n = inp.boxes_per_layer * inp.product_weight_kg * g
    f = 1.0
    if inp.pressures_kpa:
        _, _, cv = compute_variability(inp.pressures_kpa)
        f = get_safety_factor(cv)
        print(f"📏 Gemeten drukvariatie CV = {cv:.1f}% → Veiligheidsfactor f = {f}")
    return max(int(bct_n // (load_layer_n * f)), 1)

def apply_constraints(mech: int, inp: ScenarioInput) -> tuple[int, str]:
    if not inp.stackable:
        return 1, 'niet-stapelbaar'
    max_l = mech
    tag = ''
    if inp.user_max_layers > 0 and inp.user_max_layers < max_l:
        max_l, tag = inp.user_max_layers, 'userlimiet'
    if inp.rack_height_mm > 0:
        fit = int((inp.rack_height_mm - t_pallet_base_mm) // inp.box.height_mm)
        if fit < max_l:
            max_l, tag = fit, 'stellinghoogte'
    if inp.customer_height_mm > 0:
        fit = int((inp.customer_height_mm - t_pallet_base_mm) // inp.box.height_mm)
        if fit < max_l:
            max_l, tag = fit, 'klantlimiet'
    return max(max_l, 1), tag

def pallets_for(layers: int, inp: ScenarioInput) -> int:
    return math.ceil(inp.n_boxes_total / (layers * inp.boxes_per_layer))

def volume_per_pallet(layers: int, inp: ScenarioInput) -> float:
    lx, wx = inp.box.length_mm / 1000, inp.box.width_mm / 1000
    h_total = (t_pallet_base_mm + layers * inp.box.height_mm) / 1000
    return lx * wx * h_total

def compute_storage_cost(pallets: int, vol_per_pallet: float, inp: ScenarioInput) -> float:
    total_vol = pallets * vol_per_pallet
    int_cap = inp.internal_capacity_m3
    if int_cap is None or int_cap <= 0:
        rate = inp.storage_cost_external_per_m3 if inp.use_external else inp.storage_cost_internal_per_m3
        if rate <= 0:
            print("⚠️ Opslagtarief per m³ onbekend, opslagkosten = 0")
            return 0.0
        return total_vol * rate * inp.storage_weeks
    vol_internal = min(total_vol, int_cap)
    vol_external = max(total_vol - int_cap, 0.0)
    cost_int = vol_internal * inp.storage_cost_internal_per_m3 * inp.storage_weeks
    cost_ext = vol_external * inp.storage_cost_external_per_m3 * inp.storage_weeks
    return cost_int + cost_ext

def run(inp: ScenarioInput) -> tuple[ScenarioResult, List[ScenarioResult]]:
    bct_val = compute_bct(inp)
    mech_max = compute_mech_max_layers(bct_val, inp)
    max_layers, constraint = apply_constraints(mech_max, inp)
    results: List[ScenarioResult] = []
    layer_range = [1] if not inp.stackable else range(1, max_layers + 1)
    for layers in layer_range:
        pallets = pallets_for(layers, inp)
        carton = inp.n_boxes_total * inp.price_per_box_eur
        vol = volume_per_pallet(layers, inp)
        storage = compute_storage_cost(pallets, vol, inp)
        total = carton + storage
        results.append(ScenarioResult(layers, pallets, carton, storage, total, constraint if layers == max_layers else ''))
    return results[-1], results

def main():
    print("=== Vrumona Packaging Optimiser v4.3b ===")
    L = ask_float("Lengte doos L (mm)", 300)
    B = ask_float("Breedte doos B (mm)", 200)
    H = ask_float("Hoogte doos H (mm)", 250)
    t = ask_float("Dikte karton t (mm)", 4)
    ect = ask_float("ECT (kN/m)", 6)

    wp = ask_float("Gewicht product (kg)", 8)
    wpkg = ask_float("Gewicht verpakking (kg)", 0.4)
    w = wp + wpkg

    dpl = ask_int("Dozen per laag", 15)
    n = ask_int("Totaal aantal dozen", 900)
    p_box = ask_float("Prijs per doos (€)", 0.42)

    int_rate = ask_float("Opslagkost intern per m³ (€)", 0)
    ext_rate = ask_float("Opslagkost extern per m³ (€)", 0)
    use_ext = ask_bool("Gebruik externe opslagkosten?", False)

    cap_m3 = ask_float("Beschikbare interne opslagcapaciteit (m³, 0=onbeperkt)", 0)

    weeks = ask_int("Opslagduur (weken)", 3)
    stbl = ask_bool("Stapeling toegestaan?", True)
    user_max = ask_int("Maximale stapelhoogte (lagen, 0=geen extra limiet)", 0) if stbl else 0
    r_h = ask_float("Vrije stellinghoogte (mm, 0=onbeperkt)", 0)
    c_h = ask_float("Max pallethoogte klant (mm, 0=onbeperkt)", 0)

    press_vals = None
    druk_n = ask_int("Aantal drukmetingen (0–4)", 0)
    if druk_n in (1, 2, 3, 4):
        press_vals = []
        for i in range(1, druk_n + 1):
            val = ask_float(f"  Drukmeting D{i} (kPa)", 100)
            press_vals.append(val)

    inp = ScenarioInput(
        box=Box(L, B, H, t, ect),
        product_weight_kg=w,
        boxes_per_layer=dpl,
        n_boxes_total=n,
        price_per_box_eur=p_box,
        storage_weeks=weeks,
        storage_cost_internal_per_m3=int_rate,
        storage_cost_external_per_m3=ext_rate,
        use_external=use_ext,
        stackable=stbl,
        user_max_layers=user_max,
        rack_height_mm=r_h,
        customer_height_mm=c_h,
        internal_capacity_m3=cap_m3 if cap_m3 > 0 else None,
        pressures_kpa=press_vals,
    )

    head, all_res = run(inp)

    print("\n📊 Resultaten:")
    for res in all_res:
        flag = f" ← beperkt door {res.constraint}" if res.constraint else ''
        print(f" {res.layers} lagen → pallets={res.pallets}, kosten=€{res.total_cost:,.2f}{flag}")

    print("\n➡️ Geadviseerde stapeling:")
    out = f" {head.layers} lagen, kosten=€{head.total_cost:,.2f}"
    if head.constraint:
        out += f" (beperkt door {head.constraint})"
    print(out)

    if stbl and len(all_res) > 1:
        xs = [r.layers for r in all_res]
        ys_cost = [r.total_cost for r in all_res]
        ys_pallet = [r.pallets for r in all_res]
        plt.figure(); plt.bar(xs, ys_cost); plt.xlabel('Lagen'); plt.ylabel('Kosten €'); plt.title('Kosten vs stapelhoogte'); plt.grid(axis='y'); plt.show()
        plt.figure(); plt.bar(xs, ys_pallet); plt.xlabel('Lagen'); plt.ylabel('Pallets'); plt.title('Palletbehoefte'); plt.grid(axis='y'); plt.show()

if __name__ == '__main__':
    main()
