"""
JSON Serialization Utilities

Handles conversion of numpy objects and other non-JSON-serializable types
to JSON-compatible Python types.
"""

import numpy as np
import json
from typing import Any, Dict, List, Union


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy objects and other non-JSON-serializable types
    to JSON-compatible Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    # Handle numpy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle dictionaries
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle sets
    elif isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    
    # Handle basic Python types (already JSON serializable)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # For other objects, try to convert to string
    else:
        try:
            # Try to get dict representation
            if hasattr(obj, '__dict__'):
                return make_json_serializable(obj.__dict__)
            else:
                return str(obj)
        except:
            return str(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON, handling numpy types.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation
    """
    serializable_obj = make_json_serializable(obj)
    return json.dumps(serializable_obj, **kwargs)


def safe_json_response(obj: Any) -> Dict[str, Any]:
    """
    Prepare an object for JSON response by ensuring all nested objects
    are JSON-serializable.
    
    Args:
        obj: Object to prepare for JSON response
        
    Returns:
        JSON-serializable dictionary
    """
    return make_json_serializable(obj)