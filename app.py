
import streamlit as st
import math
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

g = 9.81
t_pallet_base_mm = 120

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

def compute_bct(inp: ScenarioInput) -> float:
    return inp.k_factor * inp.box.ect_knpm * math.sqrt(inp.box.thickness_mm * inp.box.perimeter_mm) * inp.env_factor * 1000

def compute_mech_max_layers(bct_n: float, inp: ScenarioInput) -> int:
    load_layer_n = inp.boxes_per_layer * inp.product_weight_kg * g
    return max(int(bct_n // load_layer_n), 1)

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
    lx, wx = inp.box.length_mm/1000, inp.box.width_mm/1000
    h_total = (t_pallet_base_mm + layers * inp.box.height_mm) / 1000
    return lx * wx * h_total

def compute_storage_cost(pallets: int, vol: float, inp: ScenarioInput) -> float:
    rate = inp.storage_cost_external_per_m3 if inp.use_external else inp.storage_cost_internal_per_m3
    return pallets * vol * rate * inp.storage_weeks if rate > 0 else 0.0

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

# Streamlit UI
st.title("📦 Vrumona Packaging Optimiser v4.1")

L = st.number_input("Lengte doos L (mm)", value=300.0)
B = st.number_input("Breedte doos B (mm)", value=200.0)
H = st.number_input("Hoogte doos H (mm)", value=250.0)
t = st.number_input("Dikte karton t (mm)", value=4.0)
ect = st.number_input("ECT (kN/m)", value=6.0)
w = st.number_input("Gewicht product per doos (kg)", value=8.0)
dpl = st.number_input("Dozen per laag", value=15, step=1)
n = st.number_input("Totaal aantal dozen", value=900, step=1)
p_box = st.number_input("Prijs per doos (€)", value=0.42)
int_rate = st.number_input("Opslagkost intern per m³ (€)", value=0.0)
ext_rate = st.number_input("Opslagkost extern per m³ (€)", value=0.0)
use_ext = st.checkbox("Gebruik externe opslagkosten?", value=False)
weeks = st.number_input("Opslagduur (weken)", value=3, step=1)
stackable = st.checkbox("Stapeling toegestaan?", value=True)
user_max = st.number_input("Maximale stapelhoogte (lagen, 0=geen extra limiet)", value=0, step=1) if stackable else 0
rack_height = st.number_input("Vrije stellinghoogte (mm)", value=0.0)
customer_height = st.number_input("Max pallethoogte klant (mm)", value=0.0)

if st.button("🔍 Optimaliseer stapeling"):
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
        stackable=stackable,
        user_max_layers=user_max,
        rack_height_mm=rack_height,
        customer_height_mm=customer_height,
    )

    head, all_res = run(inp)

    st.subheader("📊 Resultaten")
    for res in all_res:
        label = f" (beperkt door {res.constraint})" if res.constraint else ""
        st.write(f"{res.layers} lagen → pallets = {res.pallets}, kosten = €{res.total_cost:,.2f}{label}")

    st.success(f"Geadviseerde stapeling: {head.layers} lagen, totaal €{head.total_cost:,.2f}")
    if head.constraint:
        st.info(f"Beperkt door: {head.constraint}")

    if stackable and len(all_res) > 1:
        xs = [r.layers for r in all_res]
        ys_cost = [r.total_cost for r in all_res]
        ys_pallet = [r.pallets for r in all_res]

        fig1, ax1 = plt.subplots()
        ax1.bar(xs, ys_cost)
        ax1.set_xlabel("Lagen")
        ax1.set_ylabel("Kosten (€)")
        ax1.set_title("Kosten vs stapelhoogte")
        ax1.grid(axis='y')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.bar(xs, ys_pallet)
        ax2.set_xlabel("Lagen")
        ax2.set_ylabel("Pallets")
        ax2.set_title("Palletbehoefte")
        ax2.grid(axis='y')
        st.pyplot(fig2)
