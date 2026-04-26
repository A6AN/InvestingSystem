# Plan for `base_specialist.py`

Based on Section 13 of the `plan.md` master document, my plan for `system/models/base_specialist.py` is to establish the strict contract that all 6 specialists will inherit from.

## Proposed Changes

### `system/models/`
We need to create the `models` directory within `system/`.

#### [NEW] [base_specialist.py](file:///Users/Ayan/Documents/College/Python/InvestingSystem/system/models/base_specialist.py)
I will implement the `SignalContract` and `BaseSpecialist` exactly as specified in the master plan to ensure type safety and pipeline stability.

1. **`SignalContract` Data Class**: 
   - Define the standardized schema for all signal outputs (fields: `specialist`, `timestamp`, `symbol`, `signal`, `confidence`, `strength`, `risk_score`, `regime_fit`, `expected_return`, `uncertainty`, `metadata`).

2. **`BaseSpecialist` Abstract Base Class (ABC)**:
   - Define the `name` property.
   - Enforce implementation of `compute_features(self, data: dict) -> dict` and `generate_signal(self, features: dict) -> SignalContract` by child classes using `@abstractmethod`.
   - Implement the `safe_generate(self, data: dict) -> SignalContract` wrapper. This is critical for pipeline resilience, ensuring that if any specialist crashes, it gracefully returns a fallback contract with a zero signal and logs the error in the `metadata`.
   - Implement the `_validate(self, contract: SignalContract)` helper to enforce strict bounds on signals (`-1, 0, 1`) and scores (`0.0 - 1.0`).
   - Stub the Phase 3+ ML methods (`train`, `save_model`, `load_model`) so rule-based implementations can pass without issue.

## User Review Required

> [!NOTE]
> Are there any additional fields or typing enhancements you'd like me to add to `SignalContract` for Phase 1, or should I stick strictly to the schema provided in `plan.md`?

## Verification Plan

### Automated Tests
- I will create a dummy specialist inherited from `BaseSpecialist` to test `safe_generate`.
- Force an exception in the dummy specialist's `compute_features` method to verify that `safe_generate` correctly catches it and returns a valid fallback `SignalContract` without crashing.
- Test `_validate()` by passing out-of-bounds `confidence` or `signal` values to ensure they are caught.
