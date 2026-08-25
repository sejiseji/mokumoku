# Cloud Sprite Atlas Layout

- Image bank 0: variant 0
- Image bank 1: variant 1
- Image bank 2: variant 2
- Cell: 40 x 40 px
- Grid: 6 columns x 6 rows
- Slot: `family_index * 4 + size_index`
- Sprite dimensions: s=16, m=24, l=32, xl=40
- Every sprite is centered inside its cell.

| Row | Col 0 | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---:|---|---|---|---|---|---|
| 0 | INTERNAL s | INTERNAL m | INTERNAL l | INTERNAL xl | EDGE s | EDGE m |
| 1 | EDGE l | EDGE xl | BOTTOM s | BOTTOM m | BOTTOM l | BOTTOM xl |
| 2 | UPDRAFT s | UPDRAFT m | UPDRAFT l | UPDRAFT xl | STRETCH s | STRETCH m |
| 3 | STRETCH l | STRETCH xl | FRAGMENT s | FRAGMENT m | FRAGMENT l | FRAGMENT xl |
| 4 | FADE s | FADE m | FADE l | FADE xl | SERENDIPITY s | SERENDIPITY m |
| 5 | SERENDIPITY l | SERENDIPITY xl | CHARGE s | CHARGE m | CHARGE l | CHARGE xl |

Cell origins are `(column * 40, row * 40)`.
The last cell ends at pixel 239, so the complete atlas fits inside a 256 x 256 Pyxel image bank.
