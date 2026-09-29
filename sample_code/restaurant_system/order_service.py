"""Order Processing Service for Restaurant Management System."""
from typing import List, Tuple, Dict, Any
from menu_service import get_menu_item, is_item_available, calculate_item_price

def apply_member_discount(total_amount: float, is_member: bool) -> float:
    """Apply a 10% discount if the customer is a member."""
    if is_member:
        return total_amount * 0.90
    return total_amount

def create_order(table_no: int, items_list: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Create a new restaurant order. Calls is_item_available and calculate_item_price."""
    subtotal = 0.0
    ordered_items = []
    
    for item_id, quantity in items_list:
        if is_item_available(item_id):
            item_price = calculate_item_price(item_id, quantity)
            subtotal += item_price
            item_info = get_menu_item(item_id)
            ordered_items.append({
                "item_id": item_id,
                "name": item_info.get("name"),
                "quantity": quantity,
                "price": item_price
            })

    return {
        "table_no": table_no,
        "items": ordered_items,
        "subtotal": subtotal
    }