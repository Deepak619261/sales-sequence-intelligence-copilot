import os
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads config.yaml from the specified path. Attempts to use PyYAML,
    falling back to a custom lightweight parser if not available.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback manual parser for config.yaml structure
        return _fallback_yaml_parse(config_path)

def _fallback_yaml_parse(path: str) -> Dict[str, Any]:
    config = {}
    current_section = None
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            # Skip comments or empty lines
            if not line_str or line_str.startswith("#"):
                continue
                
            # Check indentation level to see if it's a sub-key
            indent = len(line) - len(line.lstrip())
            
            if ":" in line_str:
                parts = line_str.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                
                # Check for quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                
                # Type conversion
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.isdigit():
                    val = int(val)
                elif _is_float(val):
                    val = float(val)
                elif val == "":
                    val = {}
                
                if indent == 0:
                    current_section = key
                    config[current_section] = val
                else:
                    if isinstance(config.get(current_section), dict):
                        config[current_section][key] = val
                    else:
                        # If current section is empty dict
                        config[current_section] = {key: val}
    return config

def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False
