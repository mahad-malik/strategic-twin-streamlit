import json
import re
import time
import concurrent.futures
from typing import Optional, Dict, Any, List
import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime

st.set_page_config(layout="wide", page_title="STK Produktion GmbH")

try:
    genai.configure(api_key="API_KEY_HERE")
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
except Exception as e:
    st.error(f"FATAL: Failed to configure Google AI. Please check your API key and secrets configuration. Error: {e}")
    st.stop()

def inject_custom_css() -> None:
    """Injects custom CSS for styling the Streamlit application."""
    st.markdown(
        """<style>
        body { color: #E0E0E0; } .stApp { background-color: #0E1117; } .logo-text { font-size: 24px; font-weight: bold; color: #E0E0E0; }
        .kpi-container { background-color: #1A1F2B; border-radius: 12px; padding: 20px 20px 20px 25px; text-align: left; border: 1px solid #333; height: 100%; border-left-width: 5px; }
        .kpi-container.green { border-left-color: #2ECC71; } .kpi-container.red { border-left-color: #E74C3C; } .kpi-container.blue { border-left-color: #3498DB; } .kpi-container.yellow { border-left-color: #F1C40F; }
        .kpi-title { font-size: 14px; color: #A0A0A0; text-transform: uppercase; } .kpi-value { font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 5px; }
        .fidelity-kpi-value { font-size: 28px; font-weight: bold; color: #FFFFFF; margin-top: 5px; }
        .advisor-title { font-size: 20px; font-weight: bold; margin-bottom: 15px; } .briefing-card { padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #3498DB; background-color: rgba(52, 152, 219, 0.1); }
        .briefing-header { font-weight: bold; color: #3498DB; font-size: 16px; margin-bottom: 5px; } .briefing-content { color: #E0E0E0; }
        .cost-flow-title { font-size: 20px; font-weight: bold; margin-bottom: 15px; } .gauge-container { background-color: #1A1F2B; border-radius: 12px; padding: 20px; border: 1px solid #333; }
        .gauge-title { font-size: 16px; color: #A0A0A0; margin-bottom: 15px; } .gauge-bar-background { background-color: #444; border-radius: 5px; height: 20px; width: 100%; overflow: hidden; }
        .gauge-bar-fill { height: 100%; border-radius: 5px; text-align: right; color: white; font-weight: bold; padding-right: 5px;}
        .gauge-bar-fill.green { background-color: #2ECC71; } .gauge-bar-fill.yellow { background-color: #F1C40F; } .gauge-bar-fill.red { background-color: #E74C3C; }
        .gauge-labels { display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px; }
        </style>""",
        unsafe_allow_html=True,
    )

# ==============================================================================
# PART 1: THE CORE STRATEGIC TWIN ENGINE & AI FUNCTIONS
# ==============================================================================

@st.cache_data
def create_enhanced_ontology() -> nx.DiGraph:
    """Create and return the baseline knowledge graph (ontology) used by the twin."""
    G = nx.DiGraph()
    G.add_node("STK Factory", type="Plant", pos=(1.5, 1))
    G.add_node("China Metals Inc.", type="Supplier", country="China", pos=(0, 0))
    G.add_node("Deutsche Aluminium GmbH", type="Supplier", country="Germany", pos=(0, 2))
    G.add_node("PRO-A-250", type="Product", sale_price=25000, required_material="Standard Aluminum", pos=(3, 1.5))
    G.add_node("PRO-B-300", type="Product", sale_price=32000, required_material="Green Aluminum", pos=(3, 0.5))
    G.add_node("Standard Aluminum", type="Material", base_cost=5000, embodied_co2e_kg=800, pos=(1.5, -0.5))
    G.add_node("Green Aluminum", type="Material", base_cost=8500, embodied_co2e_kg=250, pos=(1.5, 2.5))
    G.add_node("Aluminum Smelting", type="ProductionProcess", energy_cost=5400, co2_emissions_kg=1250, pos=(3, 2))
    G.add_node("German Carbon Pricing", type="ExternalFactor", price_2025=55.0, pos=(4.5, 1))
    G.add_node("Factory Carbon Budget", type="Regulation", budget_2025_tons=150, pos=(4.5, 0))
    G.add_edge("China Metals Inc.", "STK Factory", relationship="ROUTE", route_co2_kg=5000, distance_km=10000)
    G.add_edge("Deutsche Aluminium GmbH", "STK Factory", relationship="ROUTE", route_co2_kg=300, distance_km=500)
    G.add_edge("Standard Aluminum", "China Metals Inc.", relationship="SUPPLIED_BY")
    G.add_edge("Standard Aluminum", "Deutsche Aluminium GmbH", relationship="SUPPLIED_BY")
    G.add_edge("Green Aluminum", "Deutsche Aluminium GmbH", relationship="SUPPLIED_BY")
    G.add_edge("STK Factory", "PRO-A-250", relationship="PRODUCES")
    G.add_edge("STK Factory", "PRO-B-300", relationship="PRODUCES")
    G.add_edge("STK Factory", "Aluminum Smelting", relationship="PERFORMS")
    return G


def run_enhanced_simulation(
    graph: nx.DiGraph,
    selected_product: str,
    selected_material: str,
    selected_supplier: str,
    quantity: int,
    tariff_rate_percent: float = 0,
    node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run a deterministic simulation of costs, margins and CO2 footprint.
    Returns a dictionary with KPIs, cost breakdowns, CO2 components and compliance info.
    """
    sim_graph = graph.copy()
    if node_overrides:
        for node_name, attrs in node_overrides.items():
            for attr_name, value in attrs.items():
                sim_graph.nodes[node_name][attr_name] = value

    product_data = sim_graph.nodes[selected_product]
    material_data = sim_graph.nodes[selected_material]
    supplier_data = sim_graph.nodes[selected_supplier]
    material_cost = material_data["base_cost"]
    tariff_cost_unit = material_cost * (tariff_rate_percent / 100.0) if supplier_data.get("country") == "China" else 0
    energy_cost = sim_graph.nodes["Aluminum Smelting"]["energy_cost"]
    production_co2_kg = sim_graph.nodes["Aluminum Smelting"]["co2_emissions_kg"]
    carbon_price = sim_graph.nodes["German Carbon Pricing"]["price_2025"]
    co2_cost_unit = (production_co2_kg / 1000.0) * carbon_price
    total_cost_unit = material_cost + tariff_cost_unit + energy_cost + co2_cost_unit
    gross_margin_unit = product_data["sale_price"] - total_cost_unit
    embodied_co2 = material_data["embodied_co2e_kg"]
    route_co2 = sim_graph.edges[selected_supplier, "STK Factory"]["route_co2_kg"]
    total_co2_footprint_unit_kg = embodied_co2 + production_co2_kg + route_co2
    total_gross_margin = gross_margin_unit * quantity
    total_cost = total_cost_unit * quantity
    total_co2_footprint_kg = total_co2_footprint_unit_kg * quantity
    total_co2_footprint_tons = total_co2_footprint_kg / 1000.0
    climate_regulation_node = sim_graph.nodes["Factory Carbon Budget"]
    carbon_budget_tons = climate_regulation_node["budget_2025_tons"]
    is_compliant = total_co2_footprint_tons <= carbon_budget_tons
    compliance_overshoot_tons = total_co2_footprint_tons - carbon_budget_tons if not is_compliant else 0

    return {
        "kpis": {"Gross Margin": total_gross_margin, "Carbon Footprint": total_co2_footprint_tons, "Total Cost": total_cost},
        "costs": {"Material": material_cost * quantity, "Tariff": tariff_cost_unit * quantity, "Energy": energy_cost * quantity, "Carbon (CO2)": co2_cost_unit * quantity},
        "co2_components": {"material": embodied_co2, "production": production_co2_kg, "route": route_co2, "total_unit": total_co2_footprint_unit_kg},
        "compliance": {"is_compliant": is_compliant, "budget": carbon_budget_tons, "overshoot": compliance_overshoot_tons},
        "selected_material": selected_material,
        "selected_product": selected_product,
        "tariff_rate": tariff_rate_percent,
        "selected_supplier": selected_supplier,
    }


def get_gemini_advice(results: Dict[str, Any]) -> str:
    kpis = results["kpis"]; supplier = results["selected_supplier"]; compliance = results["compliance"]
    action_prompt = ""
    if supplier == "China Metals Inc.": action_prompt = '<action type="switch_supplier" value="Deutsche Aluminium GmbH">Simulate switching to the German supplier to reduce CO₂</action>'
    elif supplier == "Deutsche Aluminium GmbH" and kpis["Total Cost"] > 150000: action_prompt = '<action type="switch_supplier" value="China Metals Inc.">Simulate switching to the Chinese supplier to reduce cost</action>'
    results_summary = json.dumps({"selected_supplier": supplier, "annual_gross_margin_EUR": kpis["Gross Margin"], "annual_carbon_footprint_tons": kpis["Carbon Footprint"], "is_carbon_compliant": compliance["is_compliant"], "carbon_budget_overshoot_tons": compliance["overshoot"]}, indent=2)
    prompt = (
        'You are the Chief Strategy Officer for "STK Produktion GmbH", a German manufacturing firm. '
        "Your task is to provide a concise, three-part strategic briefing based on the current operational data. "
        "The company's primary goals are maximizing profitability while adhering to its carbon budget. "
        f"Analyze the following data snapshot: {results_summary}. Structure your response in this strict XML format. Do not include any other text.\n"
        "<response>\n"
        "    <observation>State the single most important financial or environmental fact from the data. Be specific and quantitative.</observation>\n"
        "    <implication>Explain the direct business consequence of this observation. Why does this number matter to the company's goals?</implication>\n"
        "    <strategic_question>Pose a forward-looking question that this data raises. It should prompt the user to think about a trade-off or a future decision.</strategic_question>\n"
        f"    {action_prompt}\n"
        "</response>"
    )
    try: return model.generate_content(prompt).text
    except Exception: return "<response><observation>AI Offline</observation><implication>API Error</implication><strategic_question>Could not connect.</strategic_question></response>"


def get_gemini_cost_advice(costs_data: Dict[str, Any]) -> str:
    costs_summary = json.dumps(costs_data, indent=2)
    prompt = f"""
    You are a Financial Analyst for STK GmbH. Your task is to analyze the following annual cost breakdown, identify the most significant cost driver, and suggest a specific strategy to optimize it.
    Cost Data:
    {costs_summary}
    Respond in this strict XML format:
    <response>
    <cost_observation>Identify the single largest cost component and state its percentage of the total cost.</cost_observation>
    <optimization_suggestion>Provide a concrete, actionable recommendation to reduce or manage this specific cost.</optimization_suggestion>
    </response>
    """
    try: return model.generate_content(prompt).text
    except Exception: return "<response><cost_observation>AI Offline</cost_observation><optimization_suggestion>Could not connect to AI service.</optimization_suggestion></response>"


def get_gemini_comparison_advice(results: List[Dict[str, Any]]) -> str:
    results_summary = "".join([f"- {res['tariff_rate']}% Tariff: Margin €{res['kpis']['Gross Margin']:,.0f}, CO2 {res['kpis']['Carbon Footprint']:,.1f}t\n" for res in results])
    prompt = (
        "You are a Supply Chain Risk Analyst reporting to the board of STK GmbH. You have just completed a simulation of imposing various import tariffs on the Chinese supplier. "
        "Your goal is to identify the financial breaking point and provide a clear strategic recommendation. Here is the data from the simulation: "
        f"{results_summary}. Present your analysis in the following strict XML format. Do not add any commentary outside the tags.\n"
        "<response>\n"
        "    <key_finding>What is the primary conclusion from this analysis? (e.g., \"The margin erodes rapidly after X% tariff.\")</key_finding>\n"
        "    <tipping_point>Identify the specific tariff level where the financial viability of the Chinese supplier becomes questionable or less profitable than an alternative. Be precise.</tipping_point>\n"
        "    <executive_recommendation>Based on this analysis, what concrete action or policy should the company consider to mitigate this tariff risk? (e.g., \"Initiate dual-sourcing qualification,\" \"Hedge against currency fluctuations,\" etc.)</executive_recommendation>\n"
        "</response>"
    )
    try: return model.generate_content(prompt).text
    except Exception as e: return f"<response><key_finding>AI Offline</key_finding><tipping_point>Error</tipping_point><executive_recommendation>{e}</executive_recommendation></response>"


def get_params_from_natural_language(user_query: str, graph: nx.DiGraph) -> Optional[Dict[str, Any]]:
    products = [n for n, d in graph.nodes(data=True) if d.get("type") == "Product"]
    suppliers = [n for n, d in graph.nodes(data=True) if d.get("type") == "Supplier"]
    prompt = (
        "You are a highly precise Natural Language Understanding (NLU) engine for a logistics application. "
        "Your sole purpose is to parse a user's request and extract specific parameters into a JSON object.\n\n"
        "Rules:\n"
        '1. You must extract values for the following keys: "selected_product", "selected_supplier", "quantity".\n'
        "2. If the user does not specify a value for a key, return null for that key.\n"
        '3. The extracted value MUST be one of the "Available Options" provided below. Do not invent new ones.\n'
        '4. If the user asks for a product or supplier that is not in the options, return null for that key.\n'
        '5. "Quantity" should always be an integer.\n'
        "6. Respond ONLY with the raw, valid JSON object and nothing else.\n\n"
        f'Available Options:\n- Products: {products}\n- Suppliers: {suppliers}\nUser Request: "{user_query}"'
    )
    try:
        response = model.generate_content(prompt)
        cleaned = re.sub(r"```json\n|\n```|```", "", response.text).strip()
        return json.loads(cleaned)
    except Exception as e:
        st.error(f"Error interpreting request: {e}")
        return None


def get_gemini_sensitivity_summary(sensitivity_results: List[Dict[str, Any]]) -> str:
    summary_text = "\n".join([f"- Change in '{res['variable']}': causes a €{res['impact']:,.0f} impact on Gross Margin" for res in sensitivity_results])
    prompt = (
        f"You are a Business Risk Manager for STK GmbH. You are analyzing which business variables have the most significant impact on profitability. "
        f"Your task is to identify the most critical factor and propose a strategy to mitigate its volatility. "
        f"Here is the sensitivity analysis report: {summary_text}. Deliver your findings in the following strict XML format.\n"
        "<response>\n"
        "    <primary_factor>Identify the single variable that poses the greatest risk to the Gross Margin.</primary_factor>\n"
        "    <strategic_implication>Explain in business terms why this factor is so impactful. What part of our operation does it expose?</strategic_implication>\n"
        "    <risk_mitigation_tactic>Suggest a specific, actionable business strategy to reduce the company's exposure to volatility in this primary factor. (e.g., \"Secure long-term energy contracts,\" \"Explore material hedging options,\" etc.)</risk_mitigation_tactic>\n"
        "</response>"
    )
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"<response><primary_factor>AI Offline</primary_factor><strategic_implication>Error</strategic_implication><risk_mitigation_tactic>{e}</risk_mitigation_tactic></response>"

# ==============================================================================
# PART 2: UI RENDERING FUNCTIONS
# ==============================================================================

def render_structured_advice(response_text: str, is_comparison: bool = False, is_sensitivity: bool = False) -> None:
    try:
        if is_sensitivity:
            primary_factor = re.search(r"<primary_factor>(.*?)</primary_factor>", response_text, re.DOTALL).group(1).strip()
            strategic_implication = re.search(r"<strategic_implication>(.*?)</strategic_implication>", response_text, re.DOTALL).group(1).strip()
            recommendation = re.search(r"<risk_mitigation_tactic>(.*?)</risk_mitigation_tactic>", response_text, re.DOTALL).group(1).strip()
            st.markdown(f"**Primary Factor:** {primary_factor}")
            st.markdown(f"**Strategic Implication:** {strategic_implication}")
            st.success(f"**Recommendation:** {recommendation}")
        elif is_comparison:
            key_finding = re.search(r"<key_finding>(.*?)</key_finding>", response_text, re.DOTALL).group(1).strip()
            tipping_point = re.search(r"<tipping_point>(.*?)</tipping_point>", response_text, re.DOTALL).group(1).strip()
            strategic_brief = re.search(r"<executive_recommendation>(.*?)</executive_recommendation>", response_text, re.DOTALL).group(1).strip()
            st.subheader("Analysis Summary")
            st.markdown(f"**Key Finding:** {key_finding}")
            st.markdown(f"**Tipping Point:** {tipping_point}")
            st.subheader("Executive Recommendation")
            st.info(strategic_brief)
        else:
            observation = re.search(r"<observation>(.*?)</observation>", response_text, re.DOTALL).group(1).strip()
            implication = re.search(r"<implication>(.*?)</implication>", response_text, re.DOTALL).group(1).strip()
            suggestion = re.search(r"<strategic_question>(.*?)</strategic_question>", response_text, re.DOTALL).group(1).strip()
            action_match = re.search(r'<action type="(.*?)" value="(.*?)">(.*?)</action>', response_text, re.DOTALL)

            st.markdown(f'<div class="briefing-card"><div class="briefing-header">📊 Observation</div><div class="briefing-content">{observation}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="briefing-card"><div class="briefing-header">💡 Implication</div><div class="briefing-content">{implication}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="briefing-card" style="border-color:#F1C40F; background-color: rgba(241, 196, 15, 0.1);"><div class="briefing-header" style="color:#F1C40F;">🚀 Strategic Question</div><div class="briefing-content">{suggestion}</div></div>', unsafe_allow_html=True)

            if action_match:
                action_type, action_value, action_label = action_match.groups()
                if st.button(action_label, key=f"action_{action_value}", use_container_width=True):
                    if action_type == "switch_supplier":
                        st.session_state.selected_supplier = action_value
                        st.rerun()
    except (AttributeError, IndexError):
        st.warning("AI response was not in the expected format.")
        st.text(response_text)


def render_kpi_dashlets(results: Dict[str, Any]) -> None:
    kpis = results["kpis"]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""<div class="kpi-container green"><div class="kpi-title">Annual Gross Margin</div><div class="kpi-value">€{kpis['Gross Margin']:,.0f}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-container red"><div class="kpi-title">Annual Total Cost</div><div class="kpi-value">€{kpis['Total Cost']:,.0f}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-container blue"><div class="kpi-title">Annual Carbon Footprint</div><div class="kpi-value">{kpis['Carbon Footprint']:,.1f}t CO₂e</div></div>""", unsafe_allow_html=True)


def render_fidelity_kpis(results: Dict[str, Any], run_time: float, actual_margin: float = 0) -> None:
    st.markdown("<h5 style='margin-bottom: 5px;'>Twin Performance & Fidelity</h5>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-container blue"><div class="kpi-title">Sim Time</div><div class="fidelity-kpi-value">{run_time:.2f} s</div></div>""", unsafe_allow_html=True)
    with k2:
        est_cost = 0.00015
        st.markdown(f"""<div class="kpi-container blue"><div class="kpi-title">AI Cost</div><div class="fidelity-kpi-value">~${est_cost:.4f}</div></div>""", unsafe_allow_html=True)
    with k3:
        if actual_margin > 0:
            simulated_margin = results["kpis"]["Gross Margin"]
            deviation = ((simulated_margin - actual_margin) / actual_margin) * 100 if actual_margin != 0 else 0
            color_class = "green" if abs(deviation) <= 5 else "yellow" if abs(deviation) <= 15 else "red"
            value_text = f"{deviation:+.1f}%"
        else:
            color_class = "blue"
            value_text = "N/A"
        st.markdown(f"""<div class="kpi-container {color_class}"><div class="kpi-title">Deviation</div><div class="fidelity-kpi-value">{value_text}</div></div>""", unsafe_allow_html=True)
    with k4:
        if "sensitivity_results" in st.session_state and st.session_state.sensitivity_results:
            margin = results["kpis"]["Gross Margin"]
            if margin > 0:
                max_impact = max(abs(res["impact"]) for res in st.session_state.sensitivity_results)
                confidence_level = "High" if max_impact < (margin * 0.1) else "Medium" if max_impact < (margin * 0.2) else "Low"
            else:
                confidence_level = "N/A"
            color_class = "green" if confidence_level == "High" else "yellow" if confidence_level == "Medium" else "red" if confidence_level == "Low" else "blue"
        else:
            color_class = "blue"
            confidence_level = "N/A"
        st.markdown(f"""<div class="kpi-container {color_class}"><div class="kpi-title">Confidence</div><div class="fidelity-kpi-value">{confidence_level}</div></div>""", unsafe_allow_html=True)


def get_cost_flow_chart_fig(costs: Dict[str, float]) -> go.Figure:
    """Generates and returns the Plotly figure for the cost flow chart."""
    labels = list(costs.keys()) + ["Total Cost"]
    values = list(costs.values()) + [sum(costs.values())]
    text_vals = [f"€{v:,.0f}" for v in values]
    fig = go.Figure(go.Waterfall(
        name="Cost Flow", orientation="v",
        measure=["relative"] * len(costs) + ["total"],
        x=labels, y=values, text=text_vals, textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#3498DB"}},
        totals={"marker": {"color": "#E74C3C"}}
    ))
    fig.update_layout(showlegend=False, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="#E0E0E0", margin=dict(l=0, r=0, t=40, b=0), title="Annual Cost Flow")
    return fig

def render_cost_flow_chart(costs: Dict[str, float]) -> None:
    """Renders the cost flow chart in the Streamlit UI."""
    fig = get_cost_flow_chart_fig(costs)
    st.plotly_chart(fig, use_container_width=True)


def render_cost_breakdown_and_ai(costs: Dict[str, float]) -> None:
    total_cost = sum(costs.values())
    st.markdown("<h6>Cost Breakdown</h6>", unsafe_allow_html=True)
    for key, value in costs.items():
        st.markdown(f"<small>{key}: €{value:,.0f} ({(value / total_cost * 100) if total_cost > 0 else 0:.0f}%)</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h6>AI Cost Advisor</h6>", unsafe_allow_html=True)
    with st.spinner("Analyzing cost structure..."):
        cost_advice_xml = get_gemini_cost_advice(costs)
        try:
            observation = re.search(r"<cost_observation>(.*?)</cost_observation>", cost_advice_xml, re.DOTALL).group(1).strip()
            suggestion = re.search(r"<optimization_suggestion>(.*?)</optimization_suggestion>", cost_advice_xml, re.DOTALL).group(1).strip()
            st.info(f"**Observation:** {observation}")
            st.success(f"**Suggestion:** {suggestion}")
        except (AttributeError, IndexError):
            st.warning("Could not parse AI cost advice.")


def get_stylized_route_map_fig(graph: nx.DiGraph, active_supplier: str, results: Dict[str, Any]) -> go.Figure:
    """Generates and returns the Plotly figure for the supply route map."""
    pos = nx.get_node_attributes(graph, "pos")
    fig = go.Figure()
    supplier_nodes = list(graph.successors(results["selected_material"])) + ["STK Factory"]
    node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
    for node_name in supplier_nodes:
        x, y = pos[node_name]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>{node_name}</b>")
        is_active_node = (node_name == active_supplier or node_name == "STK Factory")
        node_sizes.append(30 if is_active_node else 20)
        node_colors.append("#E74C3C" if is_active_node else "#888")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
        marker=dict(size=node_sizes, color=node_colors, symbol="circle"), hoverinfo="none"
    ))

    shapes = []
    for supplier in graph.successors(results["selected_material"]):
        x0, y0 = pos[supplier]
        x1, y1 = pos["STK Factory"]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ctrl_x, ctrl_y = mx + (y1 - y0) * 0.2, my - (x1 - x0) * 0.2
        path = f"M {x0},{y0} Q {ctrl_x},{ctrl_y} {x1},{y1}"
        is_active = (supplier == active_supplier)
        shapes.append(dict(type="path", path=path, line=dict(color="#E74C3C" if is_active else "#555", width=4 if is_active else 2, dash="solid" if is_active else "dash")))
        if is_active:
            material_data = graph.nodes[results["selected_material"]]
            material_cost = material_data["base_cost"]
            route_co2_kg = graph.edges[supplier, "STK Factory"]["route_co2_kg"]
            fig.add_annotation(x=mx, y=my, text=f"Cost: €{material_cost:,.0f}<br>Route CO₂: {route_co2_kg/1000:,.1f}t", showarrow=False, bgcolor="#0E1117", bordercolor="#fff", borderwidth=1, font=dict(size=12, color="#E0E0E0"))

    fig.update_layout(title="Interactive Supply Route", shapes=shapes, showlegend=False, plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", font_color="#E0E0E0",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5]),
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_stylized_route_map(graph: nx.DiGraph, active_supplier: str, results: Dict[str, Any]) -> None:
    """Renders the supply route map in the Streamlit UI."""
    fig = get_stylized_route_map_fig(graph, active_supplier, results)
    st.plotly_chart(fig, use_container_width=True)


def render_final_advisor_and_compliance(results: Dict[str, Any], is_first_run: bool, show_advice: bool, advice_text: str) -> None:
    compliance_data = results["compliance"]
    footprint_tons = results["kpis"]["Carbon Footprint"]
    budget_tons = compliance_data["budget"]
    percent_of_budget = min((footprint_tons / budget_tons) * 100, 100) if budget_tons > 0 else 0
    color_class = "green" if percent_of_budget <= 85 else "yellow" if percent_of_budget < 100 else "red"
    st.markdown(f"""<div class="gauge-container"><div class="gauge-title">Factory Carbon Budget Compliance</div>
        <div class="gauge-bar-background"><div class="gauge-bar-fill {color_class}" style="width: {percent_of_budget}%;"></div></div>
        <div class="gauge-labels"><span>Current Emissions: <strong>{footprint_tons:,.1f}t</strong></span><span>Annual Budget: <strong>{budget_tons:,.0f}t</strong></span></div></div>""",
        unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="advisor-title">AI Strategic Briefing</div>', unsafe_allow_html=True)
    if is_first_run:
        st.info("Welcome! Use the controls on the left or the command bar above to explore scenarios.")
    elif show_advice:
        render_structured_advice(advice_text)
    else:
        st.info("Tick the 'Show AI Strategic Briefing' box to get narrative insights.")


def run_parallel_scenarios(graph: nx.DiGraph, product: str, material: str, supplier: str, quantity: int, tariffs: List[int]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_tariff = {executor.submit(run_enhanced_simulation, graph, product, material, supplier, quantity, tariff_rate_percent=tariff): tariff for tariff in tariffs}
        for future in concurrent.futures.as_completed(future_to_tariff):
            try:
                results.append(future.result())
            except Exception as exc:
                st.error(f"A scenario generated an exception: {exc}")
    return sorted(results, key=lambda x: x["tariff_rate"])


def get_parallel_comparison_fig(results: List[Dict[str, Any]]) -> go.Figure:
    """Generates and returns the Plotly figure for the parallel scenario comparison."""
    data_for_df = [{"tariff_rate": f"{res['tariff_rate']}% Tariff", "Gross Margin": res["kpis"]["Gross Margin"], "Total Cost": res["kpis"]["Total Cost"]} for res in results]
    df = pd.DataFrame(data_for_df)
    if df.empty:
        return go.Figure().update_layout(title="No scenarios were run.")
    df["tariff_rate_num"] = [res["tariff_rate"] for res in results]
    df = df.sort_values(by="tariff_rate_num").set_index("tariff_rate")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Gross Margin"], name="Gross Margin", marker_color="#2ECC71", text=df["Gross Margin"].apply(lambda x: f"€{x/1000:,.0f}k"), textposition="auto"))
    fig.add_trace(go.Bar(x=df.index, y=df["Total Cost"], name="Total Cost", marker_color="#E74C3C", text=df["Total Cost"].apply(lambda x: f"€{x/1000:,.0f}k"), textposition="auto"))
    fig.update_layout(barmode="group", title_text="Financial Impact Across Tariff Scenarios", plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", font_color="#E0E0E0")
    return fig

def render_parallel_comparison(results: List[Dict[str, Any]]) -> None:
    """Renders the parallel scenario comparison in the Streamlit UI."""
    st.markdown('<div class="cost-flow-title">Visual Comparison Chart</div>', unsafe_allow_html=True)
    if not results:
        st.warning("No scenarios were run.")
        return
    fig = get_parallel_comparison_fig(results)
    st.plotly_chart(fig, use_container_width=True)


def run_sensitivity_analysis(graph: nx.DiGraph, selected_product: str, selected_material: str, selected_supplier: str, quantity: int, variation_percent: int = 10) -> List[Dict[str, Any]]:
    sensitivity_results: List[Dict[str, Any]] = []
    variables_to_test = {
        "Material Cost": {"type": "node", "id": selected_material, "attr": "base_cost"},
        "Energy Cost": {"type": "node", "id": "Aluminum Smelting", "attr": "energy_cost"},
        "Carbon Price": {"type": "node", "id": "German Carbon Pricing", "attr": "price_2025"}
    }
    for var_name, var_details in variables_to_test.items():
        original_value = graph.nodes[var_details["id"]][var_details["attr"]]
        variation = original_value * (variation_percent / 100.0)
        high_run = run_enhanced_simulation(graph, selected_product, selected_material, selected_supplier, quantity, node_overrides={var_details["id"]: {var_details["attr"]: original_value + variation}})
        low_run = run_enhanced_simulation(graph, selected_product, selected_material, selected_supplier, quantity, node_overrides={var_details["id"]: {var_details["attr"]: original_value - variation}})
        impact = (high_run["kpis"]["Gross Margin"] - low_run["kpis"]["Gross Margin"]) / 2
        sensitivity_results.append({"variable": var_name, "impact": impact})
    return sorted(sensitivity_results, key=lambda x: abs(x["impact"]), reverse=True)


def get_tornado_chart_fig(sensitivity_results: List[Dict[str, Any]], variation_percent: int) -> go.Figure:
    """Generates and returns the Plotly figure for the sensitivity tornado chart."""
    if not sensitivity_results:
        return go.Figure().update_layout(title="No sensitivity data to display.")
    df = pd.DataFrame(sensitivity_results).sort_values(by="impact", ascending=True)
    df['base'] = 0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["variable"], x=df["impact"], base=df['base'], orientation="h",
        marker=dict(color=df["impact"].apply(lambda x: "#E74C3C" if x < 0 else "#2ECC71")),
        text=df["impact"].apply(lambda x: f"€{x:,.0f}")
    ))
    fig.update_layout(
        title=f"Impact on Gross Margin for a +/- {variation_percent}% Variation",
        xaxis_title="Change in Annual Gross Margin (€)", yaxis_title="Business Variable",
        plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", font_color="#E0E0E0", bargap=0.4
    )
    return fig

def render_tornado_chart(sensitivity_results: List[Dict[str, Any]], variation_percent: int) -> None:
    """Renders the sensitivity tornado chart in the Streamlit UI."""
    st.markdown('<div class="cost-flow-title">Sensitivity Analysis (Tornado Chart)</div>', unsafe_allow_html=True)
    st.write(f"This chart shows how much the Annual Gross Margin could change (in €) if these key variables fluctuate by **+/- {variation_percent}%**.")
    if not sensitivity_results:
        st.warning("No sensitivity data to display.")
        return
    fig = get_tornado_chart_fig(sensitivity_results, variation_percent)
    st.plotly_chart(fig, use_container_width=True)


def render_ontology_graph(graph: nx.DiGraph) -> None:
    """Renders the complete knowledge graph ontology using Plotly."""
    pos = nx.get_node_attributes(graph, "pos")
    if not pos:
        pos = nx.spring_layout(graph, seed=42, k=0.9)

    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_colors, node_info = [], [], [], [], []
    color_map = {"Plant": "#E74C3C", "Supplier": "#3498DB", "Product": "#2ECC71", "Material": "#F1C40F", "ProductionProcess": "#9B59B6", "ExternalFactor": "#E67E22", "Regulation": "#1ABC9C"}

    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]; node_x.append(x); node_y.append(y)
        node_text.append(node); node_colors.append(color_map.get(attrs.get('type', 'Default'), '#95A5A6'))
        info = f"<b>{node}</b><br>Type: {attrs.get('type', 'N/A')}<br>"
        info += "<br>".join([f"{key.replace('_', ' ').title()}: {val}" for key, val in attrs.items() if key not in ['pos', 'type']])
        node_info.append(info)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(showscale=False, color=node_colors, size=25, line_width=2),
        hovertext=node_info
    )
    node_label_trace = go.Scatter(
        x=node_x, y=node_y, mode='text', text=node_text, textposition="top center",
        hoverinfo='none', textfont=dict(size=10, color="#E0E0E0")
    )

    fig = go.Figure(data=[edge_trace, node_trace, node_label_trace],
             layout=go.Layout(
                showlegend=False, hovermode='closest',
                margin=dict(b=10,l=5,r=5,t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#1A1F2B", paper_bgcolor="#0E1117", font_color="#E0E0E0"
                )
            )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PART 2.5: PDF REPORTING FUNCTIONS (ADVANCED VERSION)
# ==============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "STK Produktion GmbH - Strategic Twin Report", 0, 0, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
        
    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)
        
    def chapter_body(self, content):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 5, content.encode('latin-1', 'replace').decode('latin-1'))
        self.ln()

    def add_kpi_block(self, title, value, color):
        self.set_font("Arial", "B", 10)
        self.set_text_color(*color)
        self.cell(60, 8, title, border=1)
        self.set_font("Arial", "", 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, value, border=1)
        self.ln()

def save_fig_to_bytes(fig: go.Figure) -> bytes:
    """Saves a Plotly figure to a byte stream for embedding."""
    return fig.to_image(format="png", width=800, height=450, scale=2)

@st.cache_data
def create_pdf_report(_results, _advice_text, _cost_advice_text, _G, _st_session_state):
    """Generates the full PDF report from the current session state and results."""
    pdf = PDF()
    pdf.add_page()
    
    # --- Title Page ---
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 15, "Executive Summary", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    pdf.ln(10)
    
    pdf.chapter_title("1. Scenario Parameters")
    pdf.set_font("Arial", "", 11)
    pdf.cell(40, 7, "Product:", 0, 0)
    pdf.cell(0, 7, _st_session_state['selected_product'], 0, 1)
    pdf.cell(40, 7, "Supplier:", 0, 0)
    pdf.cell(0, 7, _st_session_state['selected_supplier'], 0, 1)
    pdf.cell(40, 7, "Annual Quantity:", 0, 0)
    pdf.cell(0, 7, str(_st_session_state['quantity']), 0, 1)
    pdf.ln(5)
    
    pdf.chapter_title("2. Key Performance Indicators")
    kpis = _results['kpis']
    pdf.add_kpi_block("Annual Gross Margin", f"EUR {kpis['Gross Margin']:,.0f}", (34, 167, 240))
    pdf.add_kpi_block("Annual Total Cost", f"EUR {kpis['Total Cost']:,.0f}", (231, 76, 60))
    pdf.add_kpi_block("Annual CO2 Footprint", f"{kpis['Carbon Footprint']:,.1f} tons CO2e", (52, 152, 219))
    pdf.ln(10)
    
    # --- AI Briefing ---
    pdf.chapter_title("3. AI Strategic Briefing")
    try:
        observation = re.search(r"<observation>(.*?)</observation>", _advice_text, re.DOTALL).group(1).strip()
        implication = re.search(r"<implication>(.*?)</implication>", _advice_text, re.DOTALL).group(1).strip()
        question = re.search(r"<strategic_question>(.*?)</strategic_question>", _advice_text, re.DOTALL).group(1).strip()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Observation:", 0, 1)
        pdf.chapter_body(observation)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Implication:", 0, 1)
        pdf.chapter_body(implication)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Strategic Question:", 0, 1)
        pdf.chapter_body(question)
    except:
        pdf.chapter_body("Could not parse AI strategic advice.")
    pdf.ln(5)

    # --- Cost Analysis Page ---
    pdf.add_page()
    pdf.chapter_title("4. Annual Cost Analysis")
    try:
        observation = re.search(r"<cost_observation>(.*?)</cost_observation>", _cost_advice_text, re.DOTALL).group(1).strip()
        suggestion = re.search(r"<optimization_suggestion>(.*?)</optimization_suggestion>", _cost_advice_text, re.DOTALL).group(1).strip()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "AI Cost Observation:", 0, 1)
        pdf.chapter_body(observation)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "AI Optimization Suggestion:", 0, 1)
        pdf.chapter_body(suggestion)
    except:
        pdf.chapter_body("Could not parse AI cost advice.")
    pdf.ln(5)
    
    cost_fig = get_cost_flow_chart_fig(_results["costs"])
    pdf.image(save_fig_to_bytes(cost_fig), w=180)
    pdf.ln(5)

    # --- Supply Route and Compliance ---
    pdf.chapter_title("5. Supply Route & Compliance")
    supply_fig = get_stylized_route_map_fig(_G, _st_session_state['selected_supplier'], _results)
    pdf.image(save_fig_to_bytes(supply_fig), w=180)
    
    compliance = _results["compliance"]
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Compliance Status: {'COMPLIANT' if compliance['is_compliant'] else 'NON-COMPLIANT'}", 0, 1)
    pdf.cell(0, 7, f"Carbon Budget: {compliance['budget']:.1f} tons | Emissions: {_results['kpis']['Carbon Footprint']:.1f} tons", 0, 1)
    pdf.ln(10)
    
    # --- Analysis Section (if run) ---
    active_analysis = _st_session_state.get("active_analysis", "none")
    if active_analysis == "tariff" and _st_session_state.get("parallel_results"):
        pdf.add_page()
        pdf.chapter_title("6. Tariff Scenario Analysis")
        comparison_advice = get_gemini_comparison_advice(_st_session_state['parallel_results'])
        try:
            finding = re.search(r"<key_finding>(.*?)</key_finding>", comparison_advice, re.DOTALL).group(1).strip()
            tipping = re.search(r"<tipping_point>(.*?)</tipping_point>", comparison_advice, re.DOTALL).group(1).strip()
            reco = re.search(r"<executive_recommendation>(.*?)</executive_recommendation>", comparison_advice, re.DOTALL).group(1).strip()
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, "Key Finding:", 0, 1); pdf.chapter_body(finding)
            pdf.cell(0, 7, "Tipping Point:", 0, 1); pdf.chapter_body(tipping)
            pdf.cell(0, 7, "Executive Recommendation:", 0, 1); pdf.chapter_body(reco)
        except:
            pdf.chapter_body("Could not parse AI analysis summary.")
        pdf.ln(5)
        tariff_fig = get_parallel_comparison_fig(_st_session_state['parallel_results'])
        pdf.image(save_fig_to_bytes(tariff_fig), w=180)
        
    elif active_analysis == "sensitivity" and _st_session_state.get("sensitivity_results"):
        pdf.add_page()
        pdf.chapter_title("6. Sensitivity Analysis")
        sensitivity_advice = get_gemini_sensitivity_summary(_st_session_state['sensitivity_results'])
        try:
            factor = re.search(r"<primary_factor>(.*?)</primary_factor>", sensitivity_advice, re.DOTALL).group(1).strip()
            implication = re.search(r"<strategic_implication>(.*?)</strategic_implication>", sensitivity_advice, re.DOTALL).group(1).strip()
            tactic = re.search(r"<risk_mitigation_tactic>(.*?)</risk_mitigation_tactic>", sensitivity_advice, re.DOTALL).group(1).strip()
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, "Primary Risk Factor:", 0, 1); pdf.chapter_body(factor)
            pdf.cell(0, 7, "Strategic Implication:", 0, 1); pdf.chapter_body(implication)
            pdf.cell(0, 7, "Risk Mitigation Tactic:", 0, 1); pdf.chapter_body(tactic)
        except:
             pdf.chapter_body("Could not parse AI analysis summary.")
        pdf.ln(5)
        tornado_fig = get_tornado_chart_fig(_st_session_state['sensitivity_results'], _st_session_state['sensitivity_variation_display'])
        pdf.image(save_fig_to_bytes(tornado_fig), w=180)
    
    return bytes(pdf.output())


# ==============================================================================
# PART 3: MAIN APPLICATION LOGIC
# ==============================================================================
inject_custom_css()
if "G" not in st.session_state: st.session_state.G = create_enhanced_ontology()
if "user_has_interacted" not in st.session_state: st.session_state.user_has_interacted = False
if "quantity" not in st.session_state: st.session_state.quantity = 10
if "selected_product" not in st.session_state: st.session_state.selected_product = "PRO-A-250"
if "selected_supplier" not in st.session_state: st.session_state.selected_supplier = "China Metals Inc."
if "active_analysis" not in st.session_state: st.session_state.active_analysis = "none"
if "pdf_report" not in st.session_state: st.session_state.pdf_report = None

st.markdown('<div class="logo-text">STK Produktion GmbH</div>', unsafe_allow_html=True)
st.write("---")

st.markdown("### Intelligent Command Input")
user_query = st.text_input("Enter your request in natural language (e.g., 'Show me 50 units from the German supplier')", key="user_query_input")
if st.button("Execute Command"):
    if user_query:
        with st.spinner("Interpreting..."):
            params = get_params_from_natural_language(user_query, st.session_state.G)
        if params:
            st.success("Command Interpreted!")
            old_quantity = st.session_state.quantity
            old_supplier = st.session_state.selected_supplier
            old_product = st.session_state.selected_product
            for key, value in params.items():
                if key in st.session_state and value is not None:
                    st.session_state[key] = value
            brief_lines: List[str] = []
            if st.session_state.quantity != old_quantity: brief_lines.append(f"Annual quantity is set to {st.session_state.quantity} units.")
            if st.session_state.selected_supplier != old_supplier: brief_lines.append(f"Supplier is set to {st.session_state.selected_supplier}.")
            if st.session_state.selected_product != old_product: brief_lines.append(f"Product is set to {st.session_state.selected_product}.")
            if brief_lines:
                st.info("✅ Executing Command:\n" + "\n".join(brief_lines))
                time.sleep(2.5)
            st.session_state.user_has_interacted = True
            st.rerun()

# --- App Guide ---
with st.expander("ℹ️ Click here for Application Guide & Data Briefs"):
    st.subheader("How to Use This Dashboard")
    st.write("""
        This dashboard is an interactive Strategic Twin of 'STK Factory'. It allows you to simulate business decisions and immediately see their impact.
        1.  **Control the Scenario**: Use the Intelligent Command Input or the Scenario Controls in the sidebar to set the baseline scenario.
        2.  **Run Deeper Analysis**: Use the Analysis Tools in the sidebar to run what-if scenarios like tariff comparisons or sensitivity analysis.
        3.  **Check Fidelity**: Use the Model Calibration section in the sidebar to validate the twin against your real-world data and see how closely the simulation aligns with reality.
        4.  **Explore the Ontology**: Click the 'Ontology Graph' tab to see the underlying knowledge graph that defines the business ecosystem.
        5.  **Export Your Findings**: Use the Reporting tool in the sidebar to generate and download a comprehensive PDF report of your current scenario and analysis.
        """)
    st.subheader("Product Briefings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="briefing-card"><div class="briefing-header">PRO-A-250: Standard Component</div><p>High-volume product. Challenge: balancing low cost from China against emissions and tariff risks.</p><ul><li><strong>Sale Price:</strong> €25,000</li><li><strong>Material:</strong> Standard Aluminum</li></ul></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="briefing-card" style="border-color:#2ECC71; background-color: rgba(46, 204, 113, 0.1);"><div class="briefing-header" style="color:#2ECC71;">PRO-B-300: Premium Green Component</div><p>Premium, sustainable product using local, higher-cost green materials for compliance and lower CO2.</p><ul><li><strong>Sale Price:</strong> €32,000</li><li><strong>Material:</strong> Green Aluminum</li></ul></div>""", unsafe_allow_html=True)

st.sidebar.title("Scenario Controls")
def mark_interaction() -> None:
    st.session_state.user_has_interacted = True
    st.session_state.pdf_report = None 
st.sidebar.number_input("1. Annual Production Quantity", min_value=1, key="quantity", on_change=mark_interaction)
st.sidebar.selectbox("2. Select a Product", ["PRO-A-250", "PRO-B-300"], key="selected_product", on_change=mark_interaction)

if st.session_state.selected_product == "PRO-A-250":
    selected_material = "Standard Aluminum"
    available_suppliers = ["China Metals Inc.", "Deutsche Aluminium GmbH"]
    st.sidebar.selectbox("3. Select Supplier", available_suppliers, key="selected_supplier", on_change=mark_interaction)
else:
    selected_material = "Green Aluminum"
    st.session_state.selected_supplier = "Deutsche Aluminium GmbH"
    st.sidebar.selectbox("3. Supplier (Locked by Product)", ["Deutsche Aluminium GmbH"], disabled=True, key="supplier_b_disabled_lock")

st.sidebar.title("Model Calibration")
st.sidebar.markdown("Use this to check the twin's accuracy.")
help_text_for_margin = "To check the model's accuracy:\nSet the controls to a historical scenario (e.g., last year's quantity and supplier).\nEnter the actual, real-world Gross Margin you achieved for that period here.\nThe 'Deviation' metric on the dashboard will then show you how close the simulation is to reality."
st.sidebar.number_input("Enter Actual Gross Margin", key="actual_margin_input", min_value=0, help=help_text_for_margin)

G = st.session_state.G
start_time = time.time()
results = run_enhanced_simulation(G, st.session_state.selected_product, selected_material, st.session_state.selected_supplier, st.session_state.quantity)
run_time = time.time() - start_time
advice_text = get_gemini_advice(results)
cost_advice_text = get_gemini_cost_advice(results["costs"])

tab1, tab2 = st.tabs(["Main Dashboard", "Ontology Graph"])

with tab1:
    render_kpi_dashlets(results)
    st.markdown("---")
    main_col1, main_col2 = st.columns([2, 1])
    with main_col1:
        render_fidelity_kpis(results, run_time, st.session_state.get("actual_margin_input", 0))
        st.write("")
        with st.container():
            st.markdown('<div class="cost-flow-title">Interactive Supply Route Analysis</div>', unsafe_allow_html=True)
            render_stylized_route_map(G, active_supplier=st.session_state.selected_supplier, results=results)
            st.write("")
        with st.container():
            st.markdown('<div class="cost-flow-title">Annual Cost Flow Analysis</div>', unsafe_allow_html=True)
            cost_chart_col, cost_ai_col = st.columns([3, 2])
            with cost_chart_col: render_cost_flow_chart(results["costs"])
            with cost_ai_col: render_cost_breakdown_and_ai(results["costs"])
    with main_col2:
        with st.container():
            show_advice = st.checkbox("Show AI Strategic Briefing", value=True, help="Generate narrative advice for the current scenario.", key="get_main_advice_key")
            render_final_advisor_and_compliance(results, not st.session_state.user_has_interacted, show_advice, advice_text)

with tab2:
    st.markdown('<div class="cost-flow-title">Ontology & Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown("""
    This view displays the complete knowledge graph (ontology) that powers the Strategic Twin. An ontology defines the types of objects (nodes) and their relationships (edges) in our business ecosystem.
    - **Nodes**: Represent key entities like suppliers, products, materials, and regulations.
    - **Edges**: Represent the connections and dependencies between these entities, such as supply routes or material requirements.
    - **Colors**: Nodes are colored by their entity type for easier identification. Hover over any node to see its detailed properties.
    """)
    st.markdown("---")
    render_ontology_graph(G)


# --- Sidebar Analysis Tools ---
with st.sidebar.expander("1. Parallel Scenario Analysis", expanded=True):
    st.write("Analyze the financial impact of various tariffs.")
    tariff_scenarios_str = st.text_input("Enter tariffs to compare (e.g., 0, 15, 30)", "0, 10, 20, 30", key="tariff_input_key")
    get_tariff_summary = st.checkbox("Get AI summary for tariffs", value=True, key="get_tariff_summary_key")
    if st.button("Run Tariff Comparison", key="run_tariff_button"):
        st.session_state.active_analysis = "tariff"
        st.session_state.pdf_report = None 
        if st.session_state.selected_supplier != "China Metals Inc.":
            st.session_state.parallel_results = None
        else:
            try:
                tariffs_to_run = [int(t.strip()) for t in tariff_scenarios_str.split(",") if t.strip()]
                with st.spinner("Running tariff scenarios..."):
                    st.session_state.parallel_results = run_parallel_scenarios(G, st.session_state.selected_product, selected_material, st.session_state.selected_supplier, st.session_state.quantity, tariffs_to_run)

            except ValueError:
                st.error("Invalid input. Please enter only comma-separated numbers for the tariffs (e.g., 0, 15, 30).")
                st.session_state.parallel_results = None

with st.sidebar.expander("2. Sensitivity Analysis", expanded=True):
    st.write("Identify which variables have the biggest impact on Gross Margin.")
    sensitivity_variation = st.slider("Variation Percentage (+/-)", 5, 25, 10, 5, key="sensitivity_variation_key")
    get_sensitivity_summary = st.checkbox("Get AI summary for sensitivity", value=True, key="get_sensitivity_summary_key")
    if st.button("Run Sensitivity Analysis", key="run_sensitivity_button"):
        st.session_state.active_analysis = "sensitivity"
        st.session_state.pdf_report = None 
        with st.spinner("Running sensitivity analysis..."):
            st.session_state.sensitivity_results = run_sensitivity_analysis(G, st.session_state.selected_product, selected_material, st.session_state.selected_supplier, st.session_state.quantity, sensitivity_variation)
            st.session_state.sensitivity_variation_display = sensitivity_variation

# --- Sidebar Reporting Tool ---
st.sidebar.title("Reporting")
st.sidebar.markdown("Generate a comprehensive PDF report of the current scenario and analysis.")
if st.sidebar.button("Generate PDF Report", key="generate_pdf_button"):
    with st.spinner("Generating PDF report... This may take a moment."):
        session_state_copy = {
            'selected_product': st.session_state.selected_product,
            'selected_supplier': st.session_state.selected_supplier,
            'quantity': st.session_state.quantity,
            'active_analysis': st.session_state.get('active_analysis'),
            'parallel_results': st.session_state.get('parallel_results'),
            'sensitivity_results': st.session_state.get('sensitivity_results'),
            'sensitivity_variation_display': st.session_state.get('sensitivity_variation_display')
        }
        st.session_state.pdf_report = create_pdf_report(results, advice_text, cost_advice_text, G, session_state_copy)

if st.session_state.pdf_report:
    st.sidebar.download_button(
        label="Download PDF Report",
        data=st.session_state.pdf_report,
        file_name=f"STK_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )


# --- Analysis Results Display (at the bottom of the main tab) ---
active_analysis = st.session_state.get("active_analysis", "none")
if active_analysis != "none":
    with tab1:
        st.markdown("---")
        st.header(f"Analysis Results: {active_analysis.title()}")
        if active_analysis == "tariff":
            with st.container():
                parallel_results = st.session_state.get("parallel_results", None)
                if parallel_results is None:
                    st.warning("Tariff analysis is only applicable to 'China Metals Inc.' Please select it as the supplier and run again.")
                elif len(parallel_results) > 0:
                    render_parallel_comparison(parallel_results)
                    if st.session_state.get("get_tariff_summary_key", False):
                        st.write("---")
                        with st.spinner("Generating AI executive summary..."):
                            comparison_advice = get_gemini_comparison_advice(parallel_results)
                            render_structured_advice(comparison_advice, is_comparison=True)
                else:
                    st.warning("No results found. Please try running the analysis again.")
        elif active_analysis == "sensitivity":
            with st.container():
                sensitivity_results = st.session_state.get("sensitivity_results", [])
                variation = st.session_state.get("sensitivity_variation_display", 10)
                if sensitivity_results:
                    render_tornado_chart(sensitivity_results, variation)
                    if st.session_state.get("get_sensitivity_summary_key", False):
                        st.write("---")
                        with st.spinner("Generating AI executive summary..."):
                            sensitivity_advice = get_gemini_sensitivity_summary(sensitivity_results)
                            render_structured_advice(sensitivity_advice, is_sensitivity=True)
                else:
                    st.warning("No results found. Please try running the analysis again.")