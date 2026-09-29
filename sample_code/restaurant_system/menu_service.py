"""Menu Service for Restaurant Management System."""
from typing import Dict, Any

MENU_DB = {
    101: {"id": 101, "name": "Pad Thai", "price": 80.0, "available": True},
    102: {"id": 102, "name": "Tom Yum Goong", "price": 150.0, "available": True},
    103: {"id": 103, "name": "Green Curry", "price": 120.0, "available": False},
}

def get_menu_item(item_id: int) -> Dict[str, Any]:
    """Retrieve menu item details by item ID."""
    return MENU_DB.get(item_id, {})

def is_item_available(item_id: int) -> bool:
    """Check if a menu item is available for ordering."""
    item = get_menu_item(item_id)
    return item.get("available", False)

def calculate_item_price(item_id: int, quantity: int) -> float:
    """Calculate total price for a given menu item and quantity."""
    item = get_menu_item(item_id)
    unit_price = item.get("price", 0.0)
    return unit_price * quantity