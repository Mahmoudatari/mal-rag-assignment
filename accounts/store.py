"""The synthetic account store: three demo customers keyed by account number.

Pure like `pii/`: no network, no database, no secrets, importable standalone.
The records are the "customer's own account context" the brief asks answers to
draw on — per-customer facts only. Product-level rules (rate tables, fee caps,
eligibility) deliberately stay out of the records: they live in the knowledge
base, and duplicating them here is how an account record ends up contradicting
the retrieved passage sitting next to it in the same prompt. The one derived
value carried (`late_charge_lifetime_cap_aed`) is arithmetic over this record's
own numbers, not a copied rule.

Figures reuse the KB documents' own worked examples — the Murabaha 12-month
contract (12,000 + 1,080 over 12 x 1,090), the Ijara month-30 buyout position,
the Wakala AED 40,000 balance — so an answer combining a record with retrieved
rules stays arithmetically consistent with the passages it cites.

**The full account number appears only as the dict key, never inside a record.**
`masked_id` is the only identifier a record carries, so a prompt, trace or
response built from the record cannot leak the full number by construction —
the same shape as `pii_spans` carrying kinds and offsets but never values.

Monetary values are preformatted strings (AED, thousands separators): records
are rendered into prompts verbatim and nothing computes on them.
"""

from copy import deepcopy
from typing import Any

Account = dict[str, Any]

# Masked per the display format the KB documents themselves specify
# (`MAL-****-****-4417` — wakala doc "Data, Balances and Account Numbers",
# late-payment doc "Waivers"): all but the last four digits hidden.
ACCOUNTS: dict[str, Account] = {
    "MAL-1001-2200-4417": {
        "masked_id": "MAL-****-****-4417",
        "holdings": [
            {
                "product": "Murabaha everyday finance",
                "contract_reference": "MUR-2026-0417",
                "status": "active",
                "tier": "core",
                "asset": "laptop bought from a Mal partner merchant",
                "asset_cost_aed": "12,000.00",
                "fixed_profit_aed": "1,080.00",
                "total_sale_price_aed": "13,080.00",
                "tenor_months": 12,
                "monthly_instalment_aed": "1,090.00",
                "instalments_paid": 5,
                "next_instalment_date": "2026-08-25",
                "takaful_cover": "not offered (contract below AED 20,000)",
            },
            {
                "product": "Wakala savings",
                "account_tier": "Wakala Savings Plus",
                "status": "active",
                "average_daily_balance_aed": "40,000.00",
                "actual_profit_rate_last_period_pct": "4.10",
                "accrued_uncredited_profit_aed": "44.20",
            },
        ],
    },
    "MAL-2002-3300-8802": {
        "masked_id": "MAL-****-****-8802",
        "holdings": [
            {
                "product": "Ijara auto lease-to-own",
                "lease_reference": "IJR-2024-00842",
                "status": "active",
                "vehicle": "2023 sedan",
                "acquisition_cost_aed": "96,000.00",
                "term_months": 48,
                "lease_start_date": "2024-01-06",
                "rents_paid": 30,
                "fixed_rent_monthly_aed": "2,000.00",
                "current_monthly_rent_aed": "2,507.50",
                "outstanding_acquisition_cost_aed": "36,000.00",
                "registered_drivers": "1 (aged 25+, licence verified)",
            },
            {
                "product": "Fractional Sukuk investing",
                "portfolio_tier": "balanced",
                "listing": "Ijara sukuk maturing 2031 (asset-backed, secondary-window eligible)",
                "units_held": "9,859",
                "market_value_aed": "10,015.30",
                "cumulative_distributions_received_aed": "222.68",
                "next_expected_distribution": "2027-01-15 (semi-annual; expected, not guaranteed)",
                "latest_order_status": "settled",
            },
        ],
    },
    "MAL-3003-4400-1103": {
        "masked_id": "MAL-****-****-1103",
        "holdings": [
            {
                "product": "Murabaha everyday finance",
                "contract_reference": "MUR-2025-0091",
                "status": "in arrears",
                "outstanding_amount_aed": "18,000.00",
                "monthly_instalment_aed": "1,000.00",
                "missed_instalment": "number 7, due 2026-07-18",
                "arrears_days": 10,
                "late_charges_collected_aed": "100.00",
                "late_charge_lifetime_cap_aed": "180.00",
                "waiver_used_in_last_12_months": "no",
                "hardship_status": "none",
            },
        ],
    },
}


def lookup(account_id: str) -> Account | None:
    """The record for `account_id`, or None for an empty or unknown id.

    Unknown is not an error: the id is an optional request field, and the graph
    treats "no account context" as the ordinary case. A deep copy per call,
    because the caller is a graph node whose return value is checkpointed —
    handing out the module's own dict would let one turn's state mutation edit
    every later lookup in the process.
    """
    record = ACCOUNTS.get(account_id)
    return deepcopy(record) if record is not None else None
