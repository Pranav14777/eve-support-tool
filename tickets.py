SAMPLE_TICKETS = [
    {
        "id": "TKT-001",
        "title": "Payment via Adyen failing at checkout",
        "description": "Store associates at Hunkemöller Berlin report that payments via Adyen are failing at the POS. Customers are unable to complete purchases. Error message: 'Payment gateway timeout'. Affecting all terminals in the store since 10am.",
        "store": "Hunkemöller Berlin",
        "priority": "high"
    },
    {
        "id": "TKT-002",
        "title": "Click & collect orders not visible in store POS",
        "description": "Online orders placed for click & collect are not appearing in the store's POS system. Store staff cannot locate or fulfill these orders. Customers are arriving to collect their orders but staff cannot find them.",
        "store": "Rituals Amsterdam",
        "priority": "high"
    },
    {
        "id": "TKT-003",
        "title": "Inventory sync showing incorrect stock levels",
        "description": "Stock levels shown in EVA do not match physical inventory. Items showing as in stock are actually out of stock. This started after last night's scheduled sync at 2am.",
        "store": "Dyson - Multiple Stores",
        "priority": "medium"
    },
    {
        "id": "TKT-004",
        "title": "Product import failing with JSON error",
        "description": "Automated product import job is failing. Error log shows: 'Invalid JSON payload - unexpected token at position 142'. New products are not appearing in the system. Import has been failing since yesterday morning.",
        "store": "Kiko Milano",
        "priority": "medium"
    },
    {
        "id": "TKT-005",
        "title": "API returning 401 unauthorized on order endpoint",
        "description": "Integration partner reports receiving 401 Unauthorized response when calling the orders API. Authentication was working fine until two days ago. No changes were made on their end.",
        "store": "Intersport",
        "priority": "high"
    },
    {
        "id": "TKT-006",
        "title": "POS freezing when scanning specific product barcode",
        "description": "Store associates report that scanning barcode 8720181558613 causes the POS to freeze completely. Device requires a full restart. Issue is consistent and reproducible across multiple devices in the store.",
        "store": "AFC Ajax Store",
        "priority": "medium"
    },
    {
        "id": "TKT-007",
        "title": "Prices showing without VAT in German stores",
        "description": "Customer-facing prices in all German store locations are displaying without VAT. This is causing pricing discrepancies at checkout. Issue appears to be affecting all product categories.",
        "store": "Hunkemöller Germany",
        "priority": "high"
    },
    {
        "id": "TKT-008",
        "title": "Customer loyalty points not updating after purchase",
        "description": "Customers report that loyalty points are not being added to their accounts after completing purchases. The issue has been ongoing for 3 days. Customers are complaining at the store level.",
        "store": "Rituals - Multiple Stores",
        "priority": "medium"
    },
    {
        "id": "TKT-009",
        "title": "Receipt language not switching to local language",
        "description": "Receipts are printing in English across all stores in France and Spain. The system should automatically switch receipt language based on store locale settings. Configuration was not changed recently.",
        "store": "Multiple EU Stores",
        "priority": "low"
    },
    {
        "id": "TKT-010",
        "title": "Mobile POS app crashing on iOS update",
        "description": "After updating iOS to version 18.3, the EVA mobile POS app crashes on launch. Store associates cannot use their devices for any operations. Affecting all iOS devices that have been updated.",
        "store": "Dyson UK Stores",
        "priority": "high"
    }
]