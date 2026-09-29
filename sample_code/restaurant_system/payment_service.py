"""Payment and Billing Service for Restaurant Management System."""
from typing import Dict, Any, List, Tuple
from order_service import create_order, apply_member_discount
from menu_service import get_menu_item

VAT_RATE = 0.07

def process_payment(table_no: int, items_list: List[Tuple[int, int]], is_member: bool) -> Dict[str, Any]:
    """Process final bill for a table. Calls create_order and apply_member_discount."""
    order = create_order(table_no, items_list)
    subtotal = order.get("subtotal", 0.0)
    
    discounted_amount = apply_member_discount(subtotal, is_member)
    final_total = discounted_amount * (1 + VAT_RATE)
    
    return {
        "table_no": table_no,
        "subtotal": subtotal,
        "discounted_amount": discounted_amount,
        "final_total_vat": round(final_total, 2),
        "status": "PAID"
    }

def generate_receipt(table_no: int, items_list: List[Tuple[int, int]], is_member: bool) -> str:
    """Generate summary receipt text for customer."""
    payment_data = process_payment(table_no, items_list, is_member)
    return f"Receipt Table {table_no}: Total = {payment_data['final_total_vat']} THB (Status: {payment_data['status']})"