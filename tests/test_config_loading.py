import os
import sys
import tempfile
import textwrap
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import load_config


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_defaults_to_no_target_queries(self):
        config = load_config(
            profile="balanced",
            yaml_path="non-existent.yaml",
            dotenv_path="non-existent.env",
        )
        self.assertEqual(config.target_queries, [])

    def test_load_config_supports_camera_id_from_yaml(self):
        yaml_content = textwrap.dedent(
            """
            camera:
              id: 2
              width: 800
              height: 600
            """
        )

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(yaml_content)
            yaml_path = handle.name

        try:
            config = load_config(profile="balanced", yaml_path=yaml_path, dotenv_path="non-existent.env")
            self.assertEqual(config.camera_id, 2)
            self.assertEqual(config.camera_width, 800)
            self.assertEqual(config.camera_height, 600)
        finally:
            os.remove(yaml_path)


if __name__ == "__main__":
    unittest.main()
