import power as P
for sigma in (0.0,):
    print(f"sigma={sigma}")
    for n in (14, 34):
        for r in (1, 2, 3, 5):
            row = []
            for pa, pb in ((0.5, 0.6), (0.5, 0.7), (0.5, 0.8)):
                row.append(f"{int(round((pb-pa)*100))}pp:{P.power(n, r, pa, pb, sigma=1e-9, sims=2000):.2f}")
            print(f"  scenes={n:2d} repeats={r} episodes/arm={n*r:3d}  " + "  ".join(row))
