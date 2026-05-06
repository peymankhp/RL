import ast
import unittest
from pathlib import Path


class AlgorithmGuideRenderingTests(unittest.TestCase):
    def test_algorithm_selection_guide_uses_html_renderer_for_cards(self):
        source = Path("rl_portal.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "render_algorithm_selection_guide"
        )

        uses_html_renderer = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "html"
            for node in ast.walk(func)
        )

        self.assertTrue(
            uses_html_renderer,
            "Algorithm guide HTML cards must use st.html so Markdown does not "
            "escape nested markup as visible text.",
        )

    def test_method_comparison_uses_html_renderer_for_rich_cards(self):
        source = Path("rl_portal.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "render_method_comparison"
        )

        uses_html_renderer = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "html"
            for node in ast.walk(func)
        )

        self.assertTrue(
            uses_html_renderer,
            "Method Comparison rich HTML cards must use st.html so Markdown "
            "does not escape nested markup as visible text.",
        )

    def test_method_comparison_axis_2_grid_uses_html_renderer(self):
        source = Path("rl_portal.py").read_text(encoding="utf-8")
        axis_2_start = source.index("# ── Axis 2: Update Timing")
        axis_3_start = source.index("# ── Axis 3: On/Off/Offline")
        axis_2_source = source[axis_2_start:axis_3_start]

        self.assertIn(
            "st.html(\"\"\"",
            axis_2_source,
            "Axis 2 rich HTML grid must use st.html, not st.markdown.",
        )
        self.assertNotIn(
            "st.markdown(\"\"\"\n    <div style=\"display:grid",
            axis_2_source,
            "Axis 2 grid should not be passed through Markdown.",
        )

    def test_home_tabs_start_with_module_catalog_and_no_roadmap(self):
        source = Path("rl_portal.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "show_home"
        )
        tab_calls = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tabs"
            and node.args
            and isinstance(node.args[0], ast.List)
        ]
        labels = [
            [
                item.value
                for item in call.args[0].elts
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
            ]
            for call in tab_calls
        ]

        top_level_tabs = labels[0]
        learning_hub_tabs = labels[1]

        self.assertTrue(
            top_level_tabs[0].endswith("Module Catalog"),
            f"Expected Module Catalog to be first, got {top_level_tabs!r}",
        )
        self.assertTrue(
            top_level_tabs[1].endswith("Learning Hub"),
            f"Expected Learning Hub to be second, got {top_level_tabs!r}",
        )
        self.assertFalse(
            any(label.endswith("Learning Roadmap") for label in top_level_tabs),
            f"Learning Roadmap should be removed from home tabs, got {top_level_tabs!r}",
        )
        self.assertFalse(
            any(label.endswith("Module Catalog") for label in learning_hub_tabs),
            f"Module Catalog should be top-level only, got {learning_hub_tabs!r}",
        )


if __name__ == "__main__":
    unittest.main()
