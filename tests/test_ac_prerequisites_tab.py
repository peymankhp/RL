import ast
import unittest
from pathlib import Path


class ActorCriticPrerequisitesTabTests(unittest.TestCase):
    def setUp(self):
        self.source = Path("_ac_mod.py").read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)
        self.main_ac = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "main_ac"
        )

    def test_prerequisites_tab_is_first_before_overview(self):
        tab_call = next(
            node
            for node in ast.walk(self.main_ac)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tabs"
            and node.args
            and isinstance(node.args[0], ast.List)
        )
        labels = [
            item.value
            for item in tab_call.args[0].elts
            if isinstance(item, ast.Constant)
            and isinstance(item.value, str)
        ]

        self.assertEqual(labels[0], "Prerequisites")
        self.assertTrue(labels[1].endswith("Overview & Why PG?"))

    def test_prerequisites_tab_embeds_reference_html_file(self):
        self.assertIn("rl_methods_reference.html", self.source)
        self.assertIn("components.html", self.source)

    def test_policy_gradient_nutshell_tab_follows_dashboard(self):
        tab_call = next(
            node
            for node in ast.walk(self.main_ac)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tabs"
            and node.args
            and isinstance(node.args[0], ast.List)
        )
        labels = [
            item.value
            for item in tab_call.args[0].elts
            if isinstance(item, ast.Constant)
            and isinstance(item.value, str)
        ]
        dashboard_index = next(
            index for index, label in enumerate(labels) if label.endswith("Dashboard")
        )

        self.assertEqual(
            labels[dashboard_index + 1],
            "Policy Gradient in a nutshell",
        )

    def test_policy_gradient_nutshell_embeds_attached_html_file(self):
        self.assertIn("policy_gradient_deep_dive.html", self.source)

    def test_reference_html_asset_exists(self):
        self.assertTrue(
            Path("portal_data/rl_methods_reference.html").is_file(),
            "Expected copied HTML asset at portal_data/rl_methods_reference.html",
        )

    def test_policy_gradient_nutshell_html_asset_exists(self):
        self.assertTrue(
            Path("portal_data/policy_gradient_deep_dive.html").is_file(),
            "Expected copied HTML asset at portal_data/policy_gradient_deep_dive.html",
        )


if __name__ == "__main__":
    unittest.main()
