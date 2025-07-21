from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Set
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class Attribute:
    def __init__(
        self,
        name: str,
        attr_type: str = "input",  # 'input' or 'calculated'
        formula: Callable[['Block', Set[str], Dict[str, float]], float] = None
    ):
        self.name = name
        self.attr_type = attr_type
        self.formula = formula
        self.value = None
        self.override_value = None

    def evaluate(self, context: 'Block', visited=None, cache=None):
        if visited is None:
            visited = set()
        if cache is None:
            cache = {}

        if self.name in cache:
            return cache[self.name]

        if self.name in visited:
            raise ValueError(f"Cyclic dependency detected in '{self.name}'")

        visited.add(self.name)

        # 1. Use override if exists
        if self.override_value is not None:
            cache[self.name] = self.override_value
            return self.override_value

        # 2. Use direct input value
        if self.attr_type == "input":
            cache[self.name] = self.value
            return self.value

        # 3. Evaluate formula
        elif self.attr_type == "calculated":
            if self.formula is None:
                raise ValueError(f"No formula for calculated attribute '{self.name}'")
            result = self.formula(context, visited, cache)
            cache[self.name] = result
            return result

        else:
            raise ValueError(f"Unknown attribute type '{self.attr_type}'")


class Block:
    def __init__(self, name: str):
        self.name = name
        self.attributes: Dict[str, Attribute] = {}

    def add_attribute(self, attr: Attribute):
        self.attributes[attr.name] = attr

    def set_input(self, attr_name: str, value: float):
        if attr_name in self.attributes:
            self.attributes[attr_name].value = value
        else:
            raise KeyError(f"Attribute '{attr_name}' not found in Block")

    def override_input(self, attr_name: str, value: float):
        if attr_name in self.attributes:
            self.attributes[attr_name].override_value = value
        else:
            raise KeyError(f"Attribute '{attr_name}' not found in Block")

    def reset_overrides(self):
        for attr in self.attributes.values():
            attr.override_value = None

    def simulate(self):
        results = {}
        cache = {}

        def evaluate_attr(attr):
            return attr.name, attr.evaluate(self, visited=set(), cache=cache)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_attr, attr) for attr in self.attributes.values()]
            for future in futures:
                name, value = future.result()
                results[name] = value

        return results

    def export_pdf_report(self, filename: str, code_str: str):
        """Export a PDF containing the code and the current simulation results."""

        results = self.simulate()
        results_str = "\n".join([f"{k}: {v}" for k, v in results.items()])

        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, f"Simulation Report for Block: {self.name}")

        # Draw code text (monospaced)
        c.setFont("Courier", 8)
        y = height - 100
        for line in code_str.split('\n'):
            if y < 72:  # new page
                c.showPage()
                c.setFont("Courier", 8)
                y = height - 72
            c.drawString(72, y, line)
            y -= 12

        # Space before results
        y -= 24

        # Draw results
        c.setFont("Helvetica-Bold", 12)
        if y < 72:
            c.showPage()
            y = height - 72
        c.drawString(72, y, "Simulation Results:")
        y -= 16

        c.setFont("Courier", 10)
        for line in results_str.split('\n'):
            if y < 72:
                c.showPage()
                c.setFont("Courier", 10)
                y = height - 72
            c.drawString(72, y, line)
            y -= 14

        c.save()
        print(f"PDF report created: {filename}")


# Example usage
if __name__ == "__main__":
    block = Block("EnergyCostBlock")

    # Input attributes
    block.add_attribute(Attribute("ElectricityPrice", "input"))
    block.add_attribute(Attribute("ConsumptionKWh", "input"))

    # Calculated attribute
    def total_energy_cost_formula(ctx, visited, cache):
        ep = ctx.attributes["ElectricityPrice"].evaluate(ctx, visited, cache)
        con = ctx.attributes["ConsumptionKWh"].evaluate(ctx, visited, cache)
        return ep * con

    block.add_attribute(Attribute("TotalEnergyCost", "calculated", total_energy_cost_formula))

    # Set inputs
    block.set_input("ElectricityPrice", 0.30)
    block.set_input("ConsumptionKWh", 1000)

    print("🔍 Simulation Result:", block.simulate())

    # Override for scenario
    block.override_input("ElectricityPrice", 0.50)
    print("📊 Scenario with override:", block.simulate())

    # Reset overrides for PDF generation example
    block.reset_overrides()

    code_string = '''
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Set

class Attribute:
    # (code omitted for brevity)
    pass

class Block:
    # (code omitted for brevity)
    pass

# Simulation and formula definitions
'''

    block.export_pdf_report("simulation_report.pdf", code_string)
