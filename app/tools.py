"""Domain tools and action definitions for Inventory & Purchasing Sentinel."""
import json
import logging
import time
from pathlib import Path
from typing import Any
from app.config import ROOT

log = logging.getLogger("tools")

# Sandbox data file path
DATA_PATH = ROOT / "data" / "sample_data.json"

def _load_data() -> dict:
    if not DATA_PATH.exists():
        return {"inventory": [], "purchase_orders": [], "suppliers": []}
    return json.loads(DATA_PATH.read_text())

def _save_data(data: dict):
    DATA_PATH.write_text(json.dumps(data, indent=2))


# --- Action Implementations ---

def check_inventory(sku: str | None = None, low_stock_only: bool = False) -> dict:
    """Tool 1: Read-only check of inventory status."""
    data = _load_data()
    items = data.get("inventory", [])
    
    if sku:
        sku_clean = sku.strip().upper()
        matching = [item for item in items if item.get("sku", "").upper() == sku_clean or sku_clean in item.get("name", "").upper()]
        if not matching:
            return {"status": "success", "result": f"No SKU or item matching '{sku}' found.", "items": []}
        return {"status": "success", "result": f"Found {len(matching)} item(s).", "items": matching}
    
    if low_stock_only:
        low_items = [item for item in items if item.get("stock", 0) <= item.get("reorder_threshold", 0)]
        return {"status": "success", "result": f"Found {len(low_items)} low-stock item(s).", "items": low_items}
    
    return {"status": "success", "result": f"Total {len(items)} items in inventory.", "items": items}


def draft_purchase_order(sku: str, quantity: int, supplier: str | None = None) -> dict:
    """Tool 2: Safe sandbox action to create a DRAFT purchase order."""
    if quantity <= 0:
        return {"status": "error", "error": "Quantity must be greater than 0"}
    
    data = _load_data()
    sku_clean = sku.strip().upper()
    items = data.get("inventory", [])
    item = next((i for i in items if i.get("sku", "").upper() == sku_clean), None)
    
    if not item:
        return {"status": "error", "error": f"SKU '{sku}' not found in inventory system."}
    
    po_count = len(data.get("purchase_orders", [])) + 1
    po_id = f"PO-5{po_count:02d}"
    
    chosen_supplier = supplier or "Default Supplier"
    po_record = {
        "po_id": po_id,
        "sku": item["sku"],
        "item_name": item["name"],
        "quantity": quantity,
        "estimated_cost": round(quantity * item.get("price", 0.0), 2),
        "supplier": chosen_supplier,
        "status": "DRAFT",
        "created_at": time.strftime("%Y-%m-%d")
    }
    
    data.setdefault("purchase_orders", []).append(po_record)
    _save_data(data)
    
    return {
        "status": "success",
        "result": f"Draft Purchase Order {po_id} created successfully for {quantity}x {item['name']}.",
        "po": po_record
    }


def issue_purchase_order(po_id: str) -> dict:
    """Tool 3: Consequential action - BLOCKED unless explicit approval flag passed."""
    # Autonomy boundary protection: Real issuing/sending PO is blocked!
    return {
        "status": "blocked",
        "error": f"Action 'issue_purchase_order' for {po_id} requires human manager approval. Cannot issue real purchase orders automatically.",
        "requires_approval": True
    }


def calculate_reorder_cost(sku: str, target_stock: int) -> dict:
    """Tool 4: Pure calculation helper for reorder costs."""
    data = _load_data()
    sku_clean = sku.strip().upper()
    item = next((i for i in data.get("inventory", []) if i.get("sku", "").upper() == sku_clean), None)
    
    if not item:
        return {"status": "error", "error": f"SKU '{sku}' not found."}
    
    current_stock = item.get("stock", 0)
    needed = max(0, target_stock - current_stock)
    unit_price = item.get("price", 0.0)
    total_cost = round(needed * unit_price, 2)
    
    return {
        "status": "success",
        "result": f"Calculated cost to reach target stock {target_stock} for {item['name']}: {needed} units @ ${unit_price} = ${total_cost}",
        "units_needed": needed,
        "total_cost_usd": total_cost
    }


# --- Tool Argument Schemas (Pydantic Schema Validation) ---
from pydantic import BaseModel, Field, ValidationError

class CheckInventoryArgs(BaseModel):
    sku: str | None = Field(default=None, description="SKU identifier or item name search string")
    low_stock_only: bool = Field(default=False, description="Filter for items at or below reorder threshold")

class DraftPurchaseOrderArgs(BaseModel):
    sku: str = Field(..., description="Target SKU identifier")
    quantity: int = Field(..., gt=0, description="Quantity to order, must be greater than 0")
    supplier: str | None = Field(default=None, description="Optional supplier name")

class IssuePurchaseOrderArgs(BaseModel):
    po_id: str = Field(..., description="Purchase Order ID (e.g. PO-501)")

class CalculateReorderCostArgs(BaseModel):
    sku: str = Field(..., description="Target SKU identifier")
    target_stock: int = Field(..., gt=0, description="Target total stock quantity, must be greater than 0")

TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "check_inventory": CheckInventoryArgs,
    "draft_purchase_order": DraftPurchaseOrderArgs,
    "issue_purchase_order": IssuePurchaseOrderArgs,
    "calculate_reorder_cost": CalculateReorderCostArgs,
}

def validate_tool_arguments(tool_name: str, args: dict) -> tuple[dict | None, str | None]:
    """Validate tool name and arguments against Pydantic schemas. Returns (validated_dict, error_msg)."""
    if tool_name not in TOOL_SCHEMAS:
        return None, f"Unknown tool '{tool_name}'"
    schema = TOOL_SCHEMAS[tool_name]
    try:
        validated_model = schema.model_validate(args)
        return validated_model.model_dump(), None
    except ValidationError as val_err:
        return None, f"Invalid tool arguments for '{tool_name}': {str(val_err)}"


# Tool Registry
TOOLS: dict[str, Any] = {
    "check_inventory": check_inventory,
    "draft_purchase_order": draft_purchase_order,
    "issue_purchase_order": issue_purchase_order,
    "calculate_reorder_cost": calculate_reorder_cost,
}


def execute_tool_with_fault_injection(
    tool_name: str,
    kwargs: dict,
    fault_config: Any,
    fault_triggered_flag: list[bool]
) -> tuple[dict, str]:
    """Gateway tool executor with fault injection capability.
    Returns (result_dict, outcome_str) where outcome_str is 'success', 'timeout', 'malformed_output', etc.
    """
    if tool_name not in TOOLS:
        return {"status": "error", "error": f"Unknown tool '{tool_name}'"}, "rejected"

    # Schema validation for tool arguments
    validated_kwargs, schema_err = validate_tool_arguments(tool_name, kwargs)
    if schema_err:
        return {"status": "error", "error": schema_err}, "contract_error"
    kwargs_to_use = validated_kwargs if validated_kwargs is not None else kwargs

    # Check fault injection for 'first_matching_operation'
    fault_type = fault_config.type if hasattr(fault_config, 'type') else (fault_config.get('type') if isinstance(fault_config, dict) else 'none')

    if fault_type != 'none' and not fault_triggered_flag[0]:
        fault_triggered_flag[0] = True  # Trigger only once
        if fault_type == 'tool_timeout':
            log.warning("Injecting fault: tool_timeout on tool %s", tool_name)
            return {"status": "error", "error": "Tool execution timed out after 5.0 seconds"}, "timeout"
        elif fault_type == 'malformed_tool_output':
            log.warning("Injecting fault: malformed_tool_output on tool %s", tool_name)
            # Return broken non-JSON/raw string or corrupt dict structure
            return {"CORRUPT_RAW_DATA": "===UNPARSABLE BAD TOOL OUTPUT STREAM==="}, "malformed_output"

    # Execute normal tool
    try:
        fn = TOOLS[tool_name]
        res = fn(**kwargs_to_use)
        if res.get("status") == "blocked":
            return res, "rejected"
        return res, "success"
    except Exception as exc:
        log.exception("Exception executing tool %s: %s", tool_name, exc)
        return {"status": "error", "error": f"Tool execution failed with exception: {str(exc)}"}, "exception"
