from simulation_engine import Block, Attribute

def create_block_from_text_results(results):
    block = Block("NLBlock")
    selected_values = {}

    for item in results:
        attr = item["Attribute"]
        value = item["Value"]
        unit = item.get("Unit", "")

        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if attr not in selected_values:
            if attr == "ElectricityPrice" and unit.lower() == "eur":
                selected_values[attr] = value
            elif attr == "ConsumptionKWh" and unit.lower() == "kwh":
                selected_values[attr] = value
            elif attr == "MaintenanceCost" and unit.lower() == "eur":
                selected_values[attr] = value

    for attr, value in selected_values.items():
        block.add_attribute(Attribute(attr, "input"))
        block.set_input(attr, value)

    if "ElectricityPrice" in block.attributes and "ConsumptionKWh" in block.attributes:
        def total_energy_formula(ctx, visited, cache):
            return (
                ctx.attributes["ElectricityPrice"].evaluate(ctx, visited, cache) *
                ctx.attributes["ConsumptionKWh"].evaluate(ctx, visited, cache)
            )
        block.add_attribute(Attribute("TotalEnergyCost", "calculated", total_energy_formula))

    return block
