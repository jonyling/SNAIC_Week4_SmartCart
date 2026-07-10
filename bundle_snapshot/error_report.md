# SmartCart Day 3 — Error Report

Validation accuracy: 0.940 (crop-aware head)

## Weakest classes

```
               fine      acc  n
   KCT-Chilli-Sauce 0.769231 13
Ayam-Brand-Sardines 0.785714 14
   Meiji-Fresh-Milk 0.785714 14
```

## Hardest confusions

- KCT-Chilli-Sauce -> Meiji-Fresh-Milk (3)
- Meiji-Fresh-Milk -> KCT-Chilli-Sauce (3)
- Chye-Sim -> Kailan (2)
- Ayam-Brand-Sardines -> KCT-Chilli-Sauce (1)
- Ayam-Brand-Sardines -> Marigold-HL-Milk (1)